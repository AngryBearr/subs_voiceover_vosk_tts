"""Minimal OpenAI-compatible chat client with a disk cache.

Every call is a pure function of ``(provider, model, prompt version, request)``;
successful, JSON-parseable answers are cached, so a repeated run makes no
network calls.  The transport is injectable, which keeps tests offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("voiceover")

Transport = Callable[[dict[str, Any]], dict[str, Any]]

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKOFF_SEC = (2.0, 4.0, 8.0)
JSON_REMINDER = "Ответ должен быть только корректным JSON без пояснений."
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```$", re.DOTALL)


class LLMError(RuntimeError):
    """Any failure of the model client; messages never contain the API key."""


class LLMTransientError(LLMError):
    """Rate limit, 5xx, timeout or connection failure: worth retrying."""


class LLMBadResponse(LLMError):
    """The model answered, but not with parseable JSON."""


REASONING_CHOICES = ("default", "off", "low", "medium", "high", "max")


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 4096
    # "default": leave thinking to the server; "off": thinking disabled; low/medium/high/max: thinking on
    # with that effort (DeepSeek fields ``thinking`` and ``reasoning_effort``).
    reasoning: str = "default"
    timeout_sec: float = 120.0
    max_retries: int = 3
    cache_dir: Path = Path("output/llm_cache")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMConfig":
        defaults = cls()
        values: dict[str, Any] = {}
        for name in ("provider", "base_url", "model", "api_key_env"):
            values[name] = str(data.get(name, getattr(defaults, name)))
        values["temperature"] = float(data.get("temperature", defaults.temperature))
        values["max_tokens"] = int(data.get("max_tokens", defaults.max_tokens))
        values["reasoning"] = str(data.get("reasoning", defaults.reasoning))
        if values["reasoning"] not in REASONING_CHOICES:
            raise ValueError(f"llm.reasoning must be one of {REASONING_CHOICES}, got {values['reasoning']!r}")
        values["timeout_sec"] = float(data.get("timeout_sec", defaults.timeout_sec))
        values["max_retries"] = int(data.get("max_retries", defaults.max_retries))
        values["cache_dir"] = Path(str(data.get("cache_dir", defaults.cache_dir)))
        return cls(**values)


@dataclass(frozen=True)
class LLMResult:
    content: dict[str, Any]
    cached: bool
    model: str
    usage: dict[str, int]
    request_sha: str
    tag: str


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse plain ``KEY=VALUE`` lines; comments and blank lines are ignored."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_api_key(env_name: str, env_file: Path | None = None) -> str:
    """Return the key from the environment, falling back to the repo ``.env``."""
    key = os.environ.get(env_name, "").strip()
    if not key:
        key = parse_env_file(env_file if env_file is not None else REPO_ROOT / ".env").get(env_name, "").strip()
    if not key:
        raise LLMError(f"API key is missing: set the {env_name} environment variable or add it to .env")
    return key


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def openai_transport(config: LLMConfig, api_key: str) -> Transport:
    """Build the default transport on top of the installed ``openai`` package."""
    import openai

    client = openai.OpenAI(base_url=config.base_url, api_key=api_key, timeout=config.timeout_sec, max_retries=0)
    transient = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)

    def call(body: dict[str, Any]) -> dict[str, Any]:
        # Provider-specific fields (DeepSeek "thinking") go through extra_body; the SDK rejects unknown keywords.
        request = dict(body)
        extra = {key: request.pop(key) for key in ("thinking", "reasoning_effort") if key in request}
        try:
            return client.chat.completions.create(**request, extra_body=extra or None).model_dump()  # type: ignore[no-any-return]
        except transient as error:
            raise LLMTransientError(f"{type(error).__name__}: {error}") from error
        except openai.APIStatusError as error:
            raise LLMError(f"request rejected with HTTP {error.status_code}: {error.message}") from error
        except openai.OpenAIError as error:
            raise LLMError(f"{type(error).__name__}: {error}") from error

    return call


@dataclass
class LLMClient:
    config: LLMConfig
    transport: Transport | None = None
    env_file: Path | None = None
    sleep: Callable[[float], None] = time.sleep
    calls_made: int = 0
    cache_hits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    models_seen: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return self._stats()

    def _stats(self) -> dict[str, int]:
        return {"calls_made": self.calls_made, "cache_hits": self.cache_hits,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out}

    # ------------------------------------------------------------ public

    def complete_json(self, system: str, user: str, *, prompt_version: str, tag: str) -> LLMResult:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        cfg = self.config
        request_sha = hashlib.sha256(canonical_json({
            "provider": cfg.provider, "model": cfg.model, "prompt_version": prompt_version,
            "temperature": cfg.temperature, "max_tokens": cfg.max_tokens, "reasoning": cfg.reasoning,
            "messages": messages,
        }).encode("utf-8")).hexdigest()
        path = cfg.cache_dir / cfg.provider / f"{request_sha[:24]}.json"
        cached = self._read_cache(path, request_sha)
        if cached is not None:
            with self._lock:
                self.cache_hits += 1
                self.models_seen.add(str(cached["response_model"]))
            return LLMResult(cached["content"], True, str(cached["response_model"]), dict(cached["usage"]),
                             request_sha, tag)
        response = self._call(messages)
        content_text = _message_content(response)
        try:
            content = _parse_json_object(content_text)
        except ValueError:
            if _finish_reason(response) == "length":
                # A reasoning model that ran out of tokens would fail identically a second time.
                raise LLMBadResponse(f"answer for {tag} was cut off at max_tokens={cfg.max_tokens}; "
                                     "raise max_tokens") from None
            log.warning("llm[%s]: answer is not valid JSON, asking once more", tag)
            response = self._call(messages + [{"role": "user", "content": JSON_REMINDER}])
            content_text = _message_content(response)
            try:
                content = _parse_json_object(content_text)
            except ValueError as error:
                raise LLMBadResponse(f"model did not return valid JSON for {tag}: {content_text[:300]!r}") from error
        model = str(response.get("model") or cfg.model)
        usage = _usage(response)
        with self._lock:
            self.models_seen.add(model)
        self._write_cache(path, {
            "request_sha": request_sha, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tag": tag, "request": {"provider": cfg.provider, "model": cfg.model, "prompt_version": prompt_version,
                                     "temperature": cfg.temperature, "max_tokens": cfg.max_tokens,
                                     "reasoning": cfg.reasoning, "messages": messages},
            "response_model": model, "usage": usage, "content_text": content_text, "content": content})
        return LLMResult(content, False, model, usage, request_sha, tag)

    # ----------------------------------------------------------- internals

    def _call(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        cfg = self.config
        body = {"model": cfg.model, "messages": messages, "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens, "response_format": {"type": "json_object"}}
        if cfg.reasoning != "default":
            body["thinking"] = {"type": "disabled" if cfg.reasoning == "off" else "enabled"}
            if cfg.reasoning != "off":
                body["reasoning_effort"] = cfg.reasoning
        with self._lock:
            if self.transport is None:
                self.transport = openai_transport(cfg, load_api_key(cfg.api_key_env, self.env_file))
            transport = self.transport
        attempt = 0
        while True:
            try:
                with self._lock:
                    self.calls_made += 1
                response = transport(body)
                break
            except LLMTransientError as error:
                if attempt >= cfg.max_retries:
                    raise LLMError(f"giving up after {attempt + 1} attempts: {error}") from error
                delay = BACKOFF_SEC[min(attempt, len(BACKOFF_SEC) - 1)]
                log.warning("llm: transient failure (%s), retry in %.0f s", error, delay)
                self.sleep(delay)
                attempt += 1
        usage = _usage(response)
        with self._lock:
            self.tokens_in += usage.get("prompt_tokens", 0)
            self.tokens_out += usage.get("completion_tokens", 0)
        return response

    @staticmethod
    def _read_cache(path: Path, request_sha: str) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("request_sha") != request_sha or not isinstance(
                data.get("content"), dict):
            return None
        return data

    @staticmethod
    def _write_cache(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


def _message_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


def _finish_reason(response: dict[str, Any]) -> str:
    try:
        return str(response["choices"][0].get("finish_reason") or "")
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def _parse_json_object(text: str) -> dict[str, Any]:
    value = json.loads(strip_code_fence(text))
    if not isinstance(value, dict):
        raise ValueError("JSON answer is not an object")
    return value


def _usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {str(k): int(v) for k, v in usage.items() if isinstance(v, int) and not isinstance(v, bool)}
