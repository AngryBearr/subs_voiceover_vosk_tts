"""Minimal OpenRouter chat client (stdlib only) returning parsed JSON answers."""

from __future__ import annotations

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

from stt.runner import REPO_ROOT

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-6-luna"

log = logging.getLogger(__name__)
_USAGE_LOCK = threading.Lock()  # clients may be shared by worker threads


def load_api_key(env_file: Path = REPO_ROOT / ".env") -> str:
    """OPENROUTER_API_KEY from the environment or the repo `.env` file (parsed as data)."""
    value = os.environ.get("OPENROUTER_API_KEY")
    if not value and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"^\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*(.*?)\s*$", line)
            if match:
                value = match.group(1).strip().strip("'\"")
    if not value:
        raise RuntimeError("Set OPENROUTER_API_KEY in the environment or in .env at the repo root")
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
        self.api_key = api_key or load_api_key()
        self.timeout_s = timeout_s
        self.usage = Usage()

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": "subs-voiceover-stt",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 16000,
        retries: int = 2,
    ) -> Any:
        """Send one chat request and return the parsed JSON answer (retries on HTTP/JSON errors)."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "usage": {"include": True},
        }
        if self.reasoning == "off":
            body["reasoning"] = {"enabled": False}
        elif self.reasoning:
            body["reasoning"] = {"effort": self.reasoning}
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = self._post(body)
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
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
                if exc.code in (400, 401, 402, 403):
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, KeyError, IndexError) as exc:
                last_error = exc
            log.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, retries + 1, last_error)
            time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"LLM request to {self.model} failed: {last_error}")
