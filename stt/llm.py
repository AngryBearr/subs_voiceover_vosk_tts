"""Minimal chat client (stdlib only) returning parsed JSON answers.

By default requests go to OpenRouter. Transports in front of the model id choose others
(``utils.llm_route``): ``opencode:`` is the OpenCode login of a ChatGPT subscription (no API
cost, no token counts), ``openrouter:`` the paid router;
``opencode+openrouter:openai/gpt-6.1-sol`` falls back to OpenRouter when OpenCode fails.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stt import progress
from stt.runner import REPO_ROOT
from utils.atomic_io import write_text_atomic
from utils.llm_route import ROUTERS, router_model, split_route
from utils.opencode_clean import OpenCodeError, flatten_messages, shared_server

DEFAULT_MODEL = "openai/gpt-6-luna"

log = logging.getLogger(__name__)
_USAGE_LOCK = threading.Lock()  # clients may be shared by worker threads
_KEPT_NOW: set[Path] = set()  # answers kept by this process: a repeated request is a retry and asks again


def load_api_key(env_file: Path = REPO_ROOT / ".env", name: str = "OPENROUTER_API_KEY") -> str:
    """API key ``name`` from the environment or the repo `.env` file (parsed as data)."""
    value = os.environ.get(name)
    if not value and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(rf"^\s*(?:export\s+)?{re.escape(name)}\s*=\s*(.*?)\s*$", line)
            if match:
                value = match.group(1).strip().strip("'\"")
    if not value:
        raise RuntimeError(f"Set {name} in the environment or in .env at the repo root")
    return value


@dataclass
class Usage:
    """Accumulated token usage and cost (USD, as reported by OpenRouter)."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    models: set[str] = field(default_factory=set)

    def add(self, response: dict[str, Any]) -> None:
        with _USAGE_LOCK:
            self._add(response)

    def _add(self, response: dict[str, Any]) -> None:
        usage = response.get("usage") or {}
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.cost += float(usage.get("cost") or 0.0)
        if response.get("model"):
            self.models.add(response["model"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost, 6),
            "models": sorted(self.models),
        }


def _extract_json(text: str) -> Any:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=0)
    return json.loads(text[start:])


class OpenRouterClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        timeout_s: float = 300.0,
        reasoning: str | None = None,
    ) -> None:
        self.model = model
        self.reasoning = reasoning  # off | minimal | low | medium | high; None: model default
        transports, self.bare_model = split_route(model)
        self.transports = transports or ["openrouter"]
        self._active = 0  # index of the transport in use; moves on when one fails for good
        self._api_keys: dict[str, str] = {}
        if api_key:
            self._api_keys[self.transports[0]] = api_key
        elif self.transports[0] != "opencode":
            self._key(self.transports[0])  # fail early without a key
        self.timeout_s = timeout_s
        self.usage = Usage()
        self._lock = threading.Lock()

    @property
    def transport(self) -> str:
        return self.transports[self._active]

    def _key(self, router: str) -> str:
        if router not in self._api_keys:
            self._api_keys[router] = load_api_key(name=ROUTERS[router].api_key_env)
        return self._api_keys[router]

    def _post(self, body: dict[str, Any], router: str = "openrouter") -> dict[str, Any]:
        model = router_model(self.bare_model)
        if model is None:  # permanent: the next transport (if any) takes over, nothing is sent
            raise RuntimeError(f"no OpenRouter id for {self.bare_model}")
        body = {**body, "model": model}
        request = urllib.request.Request(
            ROUTERS[router].base_url + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._key(router)}",
                "Content-Type": "application/json",
                "X-Title": "subs-voiceover-stt",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def _generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """The OpenCode answer in the OpenRouter response shape (no usage: the subscription pays)."""
        text = shared_server(self.bare_model).generate(flatten_messages(messages), self.bare_model, self.reasoning,
                                                       timeout=self.timeout_s)
        return {"model": "opencode:" + self.bare_model,
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}]}

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 16000,
        retries: int = 2,
    ) -> Any:
        """Send one chat request and return the parsed JSON answer.

        Retries on HTTP/JSON errors; when a transport keeps failing, the next one of the model's
        route takes over for this and all later requests of the client.

        Inside a tracked ``stt.run`` stage every answer is also kept under ``<episode>/.partial/llm/<stage>/``
        and reused when the same request comes again, so a stage that was stopped does not pay twice.
        """
        kept = self._partial_path(system, user, temperature, max_tokens)
        if kept is not None and kept not in _KEPT_NOW:
            try:
                answer = json.loads(kept.read_text(encoding="utf-8"))["answer"]
                _KEPT_NOW.add(kept)  # reused once: the same request again is a retry and asks the model
                log.info("LLM answer of an interrupted run reused (%s)", self.bare_model)
                return answer
            except (OSError, ValueError, KeyError, TypeError):
                pass  # no answer yet, or a half-written one: ask again
        answer = self._ask(system, user, temperature, max_tokens, retries)
        if kept is not None:
            try:
                write_text_atomic(kept, json.dumps({"answer": answer}, ensure_ascii=False))
                _KEPT_NOW.add(kept)
            except OSError as exc:  # a full disk must not lose the answer just paid for
                log.warning("could not keep the LLM answer: %s", exc)
        return answer

    def _partial_path(self, system: str, user: str, temperature: float, max_tokens: int) -> Path | None:
        """Where the answer to exactly this request is kept; None outside a tracked stage."""
        folder = progress.partial_dir("llm")
        if folder is None:
            return None
        request = json.dumps([self.bare_model, self.reasoning, system, user, temperature, max_tokens],
                             ensure_ascii=False)
        return folder / f"{hashlib.sha256(request.encode('utf-8')).hexdigest()}.json"

    def _ask(self, system: str, user: str, temperature: float, max_tokens: int, retries: int) -> Any:
        while True:
            active = self._active
            try:
                return self._chat_once(self.transports[active], system, user, temperature, max_tokens, retries)
            except RuntimeError as exc:
                if active + 1 >= len(self.transports):
                    raise
                with self._lock:
                    if self._active == active:
                        self._active = active + 1
                        log.warning("LLM %s via %s failed (%s); using %s from now on", self.bare_model,
                                    self.transports[active], exc, self.transports[active + 1])

    def _chat_once(self, transport: str, system: str, user: str, temperature: float, max_tokens: int,
                   retries: int) -> Any:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        body = {
            "model": self.bare_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        if not self.bare_model.endswith(":online"):  # web search (":online") refuses JSON mode
            body["response_format"] = {"type": "json_object"}
        if self.reasoning == "off":
            body["reasoning"] = {"enabled": False}
        elif self.reasoning:
            body["reasoning"] = {"effort": self.reasoning}
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = self._generate(messages) if transport == "opencode" else self._post(body, transport)
                self.usage.add(response)
                choice = response["choices"][0]
                content = choice["message"].get("content") or ""
                try:
                    return _extract_json(content)
                except json.JSONDecodeError as exc:
                    details = (response.get("usage") or {}).get("completion_tokens_details") or {}
                    raise ValueError(
                        f"no JSON in answer (finish_reason={choice.get('finish_reason')}, "
                        f"content chars={len(content)}, reasoning tokens={details.get('reasoning_tokens')}): "
                        f"{content[:120]!r}"
                    ) from exc
            except OpenCodeError as exc:
                last_error = exc
                if not exc.transient:
                    break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
                if exc.code in (400, 401, 402, 403, 404):  # 404: this router lacks the model
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, KeyError, IndexError) as exc:
                last_error = exc
            log.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, retries + 1, last_error)
            time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"LLM request to {self.bare_model} via {transport} failed: {last_error}")


def subscription_error(model: str, timeout_s: float = 90.0) -> str | None:
    """None when the OpenCode login of the subscription answers for ``model``, else the reason.

    One tiny request; an expired login ("authentication token is expired") would otherwise only
    show up as LLM steps falling back to a paid router in the middle of a run.
    """
    client = OpenRouterClient("opencode:" + split_route(model)[1], timeout_s=timeout_s)
    try:
        client.chat_json("Answer in JSON.", 'Return {"ok": true}', max_tokens=50, retries=1)
    except Exception as exc:  # any failure means the subscription route is not usable now
        return str(exc)[:300]
    return None

