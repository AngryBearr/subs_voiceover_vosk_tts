"""Minimal chat client (stdlib only) returning parsed JSON answers.

By default requests go to OpenRouter. Transports in front of the model id choose others
(``utils.llm_route``): ``opencode:`` is the OpenCode login of a ChatGPT subscription (no API
cost, no token counts), ``openrouter:`` the paid router;
``opencode+openrouter:openai/gpt-6.1-sol`` falls back to OpenRouter for a request when OpenCode fails.
The subscription itself is retried with a growing pause for ``SUBSCRIPTION_RETRY_S`` before that; a weaker model
of the same subscription may answer one request meanwhile (``llm.subscription_fallback``, free).
Without a paid transport in the route (or when the run may not pay) a subscription that does not answer raises
``SubscriptionDown`` and the run stops; ``PaidNotAllowed`` is raised before a paid request is sent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.request  # noqa: F401 - tests patch ``stt.llm.urllib.request.urlopen``
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stt import progress
from stt.runner import REPO_ROOT
from utils.atomic_io import write_text_atomic
from utils.llm_core import (env_value, extract_json, post_chat_completions, record_answer_cost, routed_model,
                            subscription_chat)
from utils.llm_route import (ROUTERS, PaidDown, PaidNotAllowed, SubscriptionDown, is_paid_dead,  # noqa: F401
                             is_subscription_down, paid_allowed, paid_run, require_paid, split_route)
from utils.opencode_clean import (SUBSCRIPTION_BACKOFF_S, SUBSCRIPTION_RETRY_ENV, SUBSCRIPTION_RETRY_S,  # noqa: F401
                                  ModelNotOffered, OpenCodeError, flatten_messages, generate_with_retry,
                                  shared_server, subscription_retry_budget)

DEFAULT_MODEL = "openai/gpt-6-luna"

log = logging.getLogger(__name__)
_USAGE_LOCK = threading.Lock()  # clients may be shared by worker threads
_KEPT_NOW: set[Path] = set()  # answers kept by this process: a repeated request is a retry and asks again


def load_api_key(env_file: Path = REPO_ROOT / ".env", name: str = "OPENROUTER_API_KEY") -> str:
    """API key ``name`` from the environment or the repo `.env` file (parsed as data)."""
    value = env_value(name, env_file)
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
    subscription_fallback_calls: int = 0  # requests answered by the weaker model of the subscription
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
        self.subscription_fallback_calls += 1 if response.get("subscription_fallback") else 0
        if response.get("model"):
            self.models.add(response["model"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost, 6),
            "subscription_fallback_calls": self.subscription_fallback_calls,
            "models": sorted(self.models),
        }


_extract_json = extract_json  # kept for callers of the old private name


class OpenRouterClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        timeout_s: float = 300.0,
        reasoning: str | None = None,
        subscription_retry_s: float | None = None,
    ) -> None:
        self.model = model
        self.reasoning = reasoning  # off | minimal | low | medium | high; None: model default
        transports, self.bare_model = split_route(model)
        self.transports = transports or ["openrouter"]
        self._answered_by = self.transports[0]  # the transport that gave the last answer
        self.subscription_retry_s = (float(subscription_retry_s) if subscription_retry_s is not None
                                     else subscription_retry_budget())
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
        return self._answered_by

    def _key(self, router: str) -> str:
        if router not in self._api_keys:
            self._api_keys[router] = load_api_key(name=ROUTERS[router].api_key_env)
        return self._api_keys[router]

    def _post(self, body: dict[str, Any], router: str = "openrouter") -> dict[str, Any]:
        # no router id is permanent: the next transport (if any) takes over, nothing is sent
        body = {**body, "model": routed_model(self.bare_model)}
        return post_chat_completions(ROUTERS[router].base_url, self._key(router), body, self.timeout_s)

    def _generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """The OpenCode answer in the OpenRouter response shape (no usage: the subscription pays).

        A transient failure (rate limit, 5xx, the model list refreshing) is retried here with a growing pause
        until ``subscription_retry_s`` of waiting is used up; a quota error is not retried. A model with a
        subscription fallback hands this one request to the weaker model when it does not answer in time.
        """
        return subscription_chat(shared_server, messages, self.bare_model, self.reasoning, self.timeout_s,
                                 self.subscription_retry_s)

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
        route takes over for this request only (the next request tries the first transport again).

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
        for index, transport in enumerate(self.transports):
            require_paid(transport, self.bare_model)  # before anything is sent
            try:
                answer = self._chat_once(transport, system, user, temperature, max_tokens, retries)
            except PaidNotAllowed:
                raise
            except RuntimeError as exc:
                later = self.transports[index + 1:]
                if isinstance(exc, SubscriptionDown) and later and not paid_allowed():
                    later = []  # the run may not pay: the subscription's failure stops it
                if not later:
                    if isinstance(exc, PaidDown):
                        exc.subscription_tried = "opencode" in self.transports[:index]
                    if isinstance(exc, SubscriptionDown):
                        log.error("The subscription is not answering, nothing was paid for %s; run again later. / "
                                  "Подписка не отвечает, за %s ничего не оплачено; запустите позже. (%s)",
                                  self.bare_model, self.bare_model, exc)
                    raise
                log.warning("LLM %s via %s failed (%s); this request goes to %s", self.bare_model, transport, exc,
                            later[0])
                continue
            self._answered_by = transport
            return answer
        raise RuntimeError("no transport")  # unreachable: transports is never empty

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
        transient_paid = False  # the last failure of a paid transport was a network / 5xx / rate-limit one
        for attempt in range(retries + 1):
            transient_paid = False
            try:
                response = self._generate(messages) if transport == "opencode" else self._post(body, transport)
                self.usage.add(response)
                if transport != "opencode":
                    record_answer_cost(self.bare_model, transport, response)
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
                transient_paid = exc.code == 429 or exc.code >= 500
                if exc.code in (400, 401, 402, 403, 404):  # 404: this router lacks the model
                    break
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error, transient_paid = exc, True
            except (json.JSONDecodeError, ValueError, KeyError, IndexError) as exc:
                last_error = exc
            log.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, retries + 1, last_error)
            time.sleep(2.0 * (attempt + 1))
        message = f"LLM request to {self.bare_model} via {transport} failed: {last_error}"
        if is_subscription_down(last_error):  # the subscription does not work now (budget used up, limit, login)
            raise SubscriptionDown(message)
        if transport != "opencode" and paid_run() and (transient_paid or is_paid_dead(str(last_error))):
            raise PaidDown(message)  # a run that may pay: the paid transport failed for good too
        raise RuntimeError(message)


def subscription_offered(model: str) -> str | None:
    """None when the OpenCode login lists ``model``, else the reason with the models it does offer.

    The check of a run's start: a model missing from the list (renamed by an update) will not appear by waiting.
    Any other failure here (the server is still starting) is left to the probe's retries.
    """
    try:
        shared_server(split_route(model)[1])
    except ModelNotOffered as exc:
        return str(exc)[:2000]  # keep the list of available models
    except Exception:  # noqa: BLE001 - the probe retries a server that is not ready
        return None
    return None


def subscription_probe(model: str, timeout_s: float = 90.0, retry_s: float | None = None) -> tuple[str | None, bool]:
    """(reason it does not answer or None, True when only the stand-in model of the subscription answered).

    One tiny request through the same path as the real ones: the whole retry budget and the per-request stand-in
    (``llm.subscription_fallback``). An expired login would otherwise only show up in the middle of a run.
    """
    client = OpenRouterClient("opencode:" + split_route(model)[1], timeout_s=timeout_s, subscription_retry_s=retry_s)
    try:
        client.chat_json("Answer in JSON.", 'Return {"ok": true}', max_tokens=50, retries=1)
    except Exception as exc:  # any failure means the subscription route is not usable now
        return str(exc)[:300], False
    return None, client.usage.subscription_fallback_calls > 0


def subscription_error(model: str, timeout_s: float = 90.0, retry_s: float | None = None) -> str | None:
    """None when the OpenCode login of the subscription answers for ``model``, else the reason."""
    return subscription_probe(model, timeout_s, retry_s)[0]
