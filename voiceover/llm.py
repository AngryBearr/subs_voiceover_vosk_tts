"""Minimal OpenAI-compatible chat client with a disk cache.

Every call is a pure function of ``(provider, model, prompt version, request)``;
successful, JSON-parseable answers are cached, so a repeated run makes no
network calls.  The transport is injectable, which keeps tests offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from utils.llm_core import (REPO_ROOT, answered_by_fallback, env_value, parse_env_file,  # noqa: F401
                            parse_json_object, record_answer_cost, routed_model, strip_code_fence,
                            subscription_chat)
from utils.llm_route import (PaidDown, PaidNotAllowed, SubscriptionDown, is_paid_dead, is_subscription_down,
                             paid_allowed, paid_run, require_paid, split_route)

log = logging.getLogger("voiceover")

Transport = Callable[[dict[str, Any]], dict[str, Any]]

BACKOFF_SEC = (2.0, 4.0, 8.0)
JSON_REMINDER = "Ответ должен быть только корректным JSON без пояснений."


class LLMError(RuntimeError):
    """Any failure of the model client; messages never contain the API key."""


class LLMTransientError(LLMError):
    """Rate limit, 5xx, timeout or connection failure: worth retrying."""


class LLMQuotaError(LLMError):
    """The subscription's usage limit is reached: no retries, and the client makes no more calls this run."""


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


def load_api_key(env_name: str, env_file: Path | None = None) -> str:
    """Return the key from the environment, falling back to the repo ``.env``."""
    key = env_value(env_name, env_file)
    if not key:
        raise LLMError(f"API key is missing: set the {env_name} environment variable or add it to .env")
    return key


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def openai_transport(config: LLMConfig, api_key: str) -> Transport:
    """Build the default transport on top of the installed ``openai`` package."""
    import openai

    reports_cost = config.provider == "openrouter"  # OpenRouter prices every answer when asked (usage.cost)
    client = openai.OpenAI(base_url=config.base_url, api_key=api_key, timeout=config.timeout_sec, max_retries=0)
    transient = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)

    def call(body: dict[str, Any]) -> dict[str, Any]:
        # Provider-specific fields (DeepSeek "thinking") go through extra_body; the SDK rejects unknown keywords.
        request = dict(body)
        if reports_cost:
            request["usage"] = {"include": True}
        extra = {key: request.pop(key) for key in ("thinking", "reasoning_effort", "usage") if key in request}
        try:
            return client.chat.completions.create(**request, extra_body=extra or None).model_dump()  # type: ignore[no-any-return]
        except transient as error:
            raise LLMTransientError(f"{type(error).__name__}: {error}") from error
        except openai.APIStatusError as error:
            raise LLMError(f"request rejected with HTTP {error.status_code}: {error.message}") from error
        except openai.OpenAIError as error:
            raise LLMError(f"{type(error).__name__}: {error}") from error

    return call


def opencode_transport(config: LLMConfig) -> Transport:
    """Transport for the OpenCode login of a ChatGPT subscription (``config.model`` is the bare id).

    A model with a subscription fallback (``llm_route.subscription_fallback``) hands one request to the weaker
    model when it does not answer in time; the answer then carries ``subscription_fallback``. A subscription
    that does not answer (budget used up, usage limit) raises ``SubscriptionDown``.
    """
    import utils.opencode_clean as opencode
    from utils.opencode_clean import OpenCodeError

    def call(body: dict[str, Any]) -> dict[str, Any]:
        variant = None if config.reasoning == "default" else config.reasoning
        try:  # a busy subscription is retried with a growing pause before the request fails
            return subscription_chat(opencode.shared_server, body["messages"], config.model,
                                     variant, config.timeout_sec, opencode.subscription_retry_budget())
        except OpenCodeError as error:
            if is_subscription_down(error):
                raise SubscriptionDown(str(error)) from error
            raise (LLMTransientError if error.transient else LLMError)(str(error)) from error

    return call


def route_transport(config: LLMConfig, env_file: Path | None = None) -> Transport:
    """Transport for ``config.model``: its route prefix (``utils.llm_route``) or the configured provider.

    With several transports (``opencode+openrouter:...``, only when the route names a paid one) a transport that
    fails (after the subscription's own retries) hands over to the next one for that request only; the next
    request tries the first transport again.
    """
    from utils.llm_route import ROUTERS, router_model

    names, bare = split_route(config.model)
    if not names:
        paid = openai_transport(config, load_api_key(config.api_key_env, env_file))

        def plain(body: dict[str, Any]) -> dict[str, Any]:
            require_paid(config.provider, config.model)  # a provider without a route is a paid one
            return paid(body)

        return plain
    built: dict[int, Transport] = {}

    def transport(index: int) -> Transport:
        if index not in built:
            name = names[index]
            if name == "opencode":
                built[index] = opencode_transport(replace(config, model=bare))
            else:
                router = ROUTERS[name]
                routed = replace(config, provider=name, base_url=router.base_url, api_key_env=router.api_key_env,
                                 model=router_model(bare) or bare)
                built[index] = openai_transport(routed, load_api_key(router.api_key_env, env_file))
        return built[index]

    def call(body: dict[str, Any]) -> dict[str, Any]:
        for index, name in enumerate(names):
            request = {**body, "model": bare}
            try:
                require_paid(name, bare)  # before anything is sent
                if name != "opencode":
                    request.pop("thinking", None)  # DeepSeek's field; not for OpenAI models on a router
                    request["model"] = routed_model(bare, LLMError)  # never send an OpenCode Go id to a router
                return transport(index)(request)
            except PaidNotAllowed:
                raise
            except Exception as error:  # noqa: BLE001 - any failure of a non-last transport hands over
                if index + 1 >= len(names) or (isinstance(error, SubscriptionDown) and not paid_allowed()):
                    raise
                log.warning("llm: %s via %s failed (%s); this request goes to %s", bare, name, error, names[index + 1])
        raise LLMError("no transport")  # unreachable: names is not empty

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
    subscription_fallback_calls: int = 0  # requests answered by the weaker model of the subscription
    models_seen: set[str] = field(default_factory=set)
    exhausted: str | None = None  # why the model's subscription stopped answering (LLMQuotaError / SubscriptionDown)
    _exhausted_by: type[Exception] = field(default=LLMQuotaError, repr=False)  # raised again by later calls
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return self._stats()

    def _stats(self) -> dict[str, int]:
        stats = {"calls_made": self.calls_made, "cache_hits": self.cache_hits,
                 "tokens_in": self.tokens_in, "tokens_out": self.tokens_out}
        if self.subscription_fallback_calls:  # shown only when the weaker model of the subscription stood in
            stats["subscription_fallback_calls"] = self.subscription_fallback_calls
        return stats

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
        if answered_by_fallback(response):  # the weaker stand-in answered: not cached as the main model's
            return LLMResult(content, False, model, usage, request_sha, tag)
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
                self.transport = route_transport(cfg, self.env_file)
            transport = self.transport
        attempt = 0
        while True:
            if self.exhausted:
                raise self._exhausted_by(self.exhausted)
            try:
                with self._lock:
                    self.calls_made += 1
                response = transport(body)
                break
            except (LLMQuotaError, SubscriptionDown, PaidDown) as error:
                with self._lock:
                    self.exhausted, self._exhausted_by = str(error), type(error)
                raise
            except LLMTransientError as error:
                if attempt >= cfg.max_retries:
                    if self._paid_last():  # a run that may pay: the paid transport still fails after its retries
                        raise self._paid_down(f"giving up after {attempt + 1} attempts: {error}") from error
                    raise LLMError(f"giving up after {attempt + 1} attempts: {error}") from error
                delay = BACKOFF_SEC[min(attempt, len(BACKOFF_SEC) - 1)]
                log.warning("llm: transient failure (%s), retry in %.0f s", error, delay)
                self.sleep(delay)
                attempt += 1
            except LLMError as error:
                if self._paid_last() and is_paid_dead(str(error)):  # refused key, no credits: nothing more to try
                    raise self._paid_down(str(error)) from error
                raise
        usage = _usage(response)
        with self._lock:
            self.tokens_in += usage.get("prompt_tokens", 0)
            self.tokens_out += usage.get("completion_tokens", 0)
            self.subscription_fallback_calls += 1 if answered_by_fallback(response) else 0
        answered = str(response.get("model") or cfg.model)
        if not answered.startswith("opencode:"):  # a paid answer: its price goes to the run's cost log
            names = split_route(cfg.model)[0]
            record_answer_cost(answered, next((n for n in names if n != "opencode"), cfg.provider), response)
        return response

    def _paid_last(self) -> bool:
        """The last transport of this client's route is a paid one and the run may pay (``STT_LLM_PAID=allow``)."""
        names = split_route(self.config.model)[0]
        return paid_run() and (not names or names[-1] != "opencode")

    def _paid_down(self, reason: str) -> PaidDown:
        """The stop for a paid transport that failed for good; the client makes no more calls."""
        names = split_route(self.config.model)[0]
        error = PaidDown(f"{self.config.model}: {reason}"[:300], subscription_tried="opencode" in names)
        with self._lock:
            self.exhausted, self._exhausted_by = str(error), PaidDown
        return error

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


_parse_json_object = parse_json_object  # kept for callers of the old private name


def _usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {str(k): int(v) for k, v in usage.items() if isinstance(v, int) and not isinstance(v, bool)}
