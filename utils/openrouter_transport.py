"""Direct, fail-closed OpenRouter native JSON-schema transport."""

from __future__ import annotations

import copy
import asyncio
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional
from urllib.parse import quote

from utils.llm_types import CallRecord, DetailedResponse
from utils.opencode_transport import OpenCodePromptResult, OpenCodePromptUsage

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_URL = f"{OPENROUTER_API_BASE_URL}/model"
OPENROUTER_ENDPOINTS_URL = f"{OPENROUTER_API_BASE_URL}/models"
OPENROUTER_GENERATION_URL = f"{OPENROUTER_API_BASE_URL}/generation"
OPENROUTER_SNAPSHOT_SCHEMA = "openrouter_production.v1"
OPENROUTER_MAX_OUTPUT_TOKENS = 16_384
_FROZEN_MODELS: dict[str, dict[str, Any]] = {
    "meta/muse-spark-1.3-contributor": {
        "effort": "low", "provider_slug": "meta", "provider_name": "Meta",
        "max_input_rate": Decimal("0.0000001"), "max_output_rate": Decimal("0.0000002"),
    },
    "openai/gpt-6-sol": {
        "effort": "high", "provider_slug": "openai", "provider_name": "OpenAI",
        "max_input_rate": Decimal("0.000005"), "max_output_rate": Decimal("0.000015"),
    },
}
_LEGACY_MODELS: dict[str, dict[str, Any]] = {
    "meta/muse-spark-1.3-contributor": _FROZEN_MODELS["meta/muse-spark-1.3-contributor"],
    "openai/gpt-5.6-sol": {
        "effort": "medium", "provider_slug": "openai", "provider_name": "OpenAI",
        "max_input_rate": Decimal("0.000005"), "max_output_rate": Decimal("0.000015"),
    },
}
_REQUIRED_ENDPOINT_PARAMETERS = frozenset({"max_tokens", "reasoning", "response_format", "structured_outputs"})
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_ERROR_REDACTIONS = (
    (re.compile(r"Bearer\s+\S+", re.IGNORECASE), "Bearer [redacted]"),
    (re.compile(r"sk-or-v1-[A-Za-z0-9_-]+"), "[redacted-key]"),
    (re.compile(r"(?:authorization|api[_ -]?key)\s*[:=]\s*[^\s,;]+", re.IGNORECASE), "[redacted-credential]"),
)
_INTERNAL_VALUE_ERROR_CODES = frozenset({
    "response_json",
    "response_json_empty",
    "response_json_sse",
    "response_json_html",
    "response_json_gzip",
    "response_json_invalid_utf8",
    "response_json_invalid",
    "response_object",
    "provider_error",
    "choices",
    "finish_reason",
    "choice_error",
    "message",
    "content",
    "usage",
    "usage_prompt_details",
    "usage_completion_details",
    "response_too_large",
    "model_required",
    "invalid_model",
    "api_key_required",
    "invalid_timeout",
    "prompts_required",
    "schema_required",
    "invalid_http_referer",
    "invalid_app_title",
    "http_error",
    "http_permanent",
    "model_mismatch",
    "provider_identity_mismatch",
    "receipt_incomplete",
})
_SAFE_VALUE_ERROR_PATTERNS = (
    re.compile(r"usage_[a-z_]+"),
    re.compile(r"invalid_[a-z_]+"),
)


def normalize_openrouter_model_id(model: str) -> str:
    if not isinstance(model, str):
        raise ValueError("model_required")
    value = model.strip()
    if not value or "/" not in value:
        raise ValueError("invalid_model")
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("invalid_model")
    provider, model_id = value.split("/", 1)
    if not provider.strip() or not model_id.strip():
        raise ValueError("invalid_model")
    return value


def _safe_message(value: Any) -> str:
    message = value if isinstance(value, str) else "unspecified provider error"
    message = " ".join(message.split())
    for pattern, replacement in _ERROR_REDACTIONS:
        message = pattern.sub(replacement, message)
    return message[:500] or "unspecified provider error"


def _log_error(status: Any, error_type: Any, message: Any) -> None:
    safe_status = status if isinstance(status, (int, str)) else "unknown"
    safe_type = _safe_message(error_type)[:100]
    # Provider response bodies are not safe operational logs, even after key redaction.
    print(f"OpenRouter error status={safe_status} type={safe_type} message=request failed", file=sys.stderr)


def _decode_json_bytes(raw: bytes) -> Any:
    """Decode a response body as strict JSON without exposing response content."""
    prefix = raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw
    first = prefix.lstrip(b" \t\r\n")
    if not first:
        raise ValueError("response_json_empty")
    if first.startswith(b"data:"):
        raise ValueError("response_json_sse")
    if first.startswith(b"<"):
        raise ValueError("response_json_html")
    if first.startswith(b"\x1f\x8b"):
        raise ValueError("response_json_gzip")
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("response_json_invalid_utf8") from None
    if not decoded.strip():
        raise ValueError("response_json_empty")
    try:
        return json.loads(decoded)
    except json.JSONDecodeError:
        raise ValueError("response_json_invalid") from None


def _safe_value_error_code(exc: ValueError) -> str:
    code = str(exc)
    if code not in _INTERNAL_VALUE_ERROR_CODES and not any(pattern.fullmatch(code) for pattern in _SAFE_VALUE_ERROR_PATTERNS):
        return "protocol_error"
    return code


async def _http_post_json(*, url: str, headers: Dict[str, str], body: Dict[str, Any], timeout: float) -> tuple[int, bytes]:
    import aiohttp

    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=body, timeout=client_timeout) as response:
            data = await _read_bounded_response(response)
            return response.status, data


async def _read_bounded_response(response: Any) -> bytes:
    """Read a response body in bounded chunks without buffering beyond the cap."""
    chunks = bytearray()
    async for chunk in response.content.iter_chunked(64 * 1024):
        chunks.extend(chunk)
        if len(chunks) > _MAX_RESPONSE_BYTES:
            raise ValueError("response_too_large")
    return bytes(chunks)


def _number(value: Any, name: str, integer: bool = True) -> Optional[int | float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"usage_{name}")
    if integer and not isinstance(value, int):
        raise ValueError(f"usage_{name}")
    return value


def _string(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"usage_{name}")
    return value


def _nonempty_optional_string(value: Any, name: str) -> Optional[str]:
    result = _string(value, name)
    if result is not None and not result.strip():
        raise ValueError(f"usage_{name}")
    return result


def _result(data: Any) -> OpenCodePromptResult:
    if not isinstance(data, dict):
        raise ValueError("response_object")
    if "error" in data:
        raise ValueError("provider_error")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("choices")
    choice = choices[0]
    finish = choice.get("finish_reason")
    if finish != "stop":
        raise ValueError("finish_reason")
    if "error" in choice or choice.get("refusal") is not None:
        raise ValueError("choice_error")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("refusal") is not None:
        raise ValueError("message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content")
    response_model = data.get("model")
    if response_model is not None and (not isinstance(response_model, str) or not response_model.strip()):
        raise ValueError("usage_model_id")
    usage_data = data.get("usage")
    if usage_data is None:
        usage = None
    else:
        if not isinstance(usage_data, dict):
            raise ValueError("usage")
        prompt_details = usage_data.get("prompt_tokens_details")
        completion_details = usage_data.get("completion_tokens_details")
        if prompt_details is not None and not isinstance(prompt_details, dict):
            raise ValueError("usage_prompt_details")
        if completion_details is not None and not isinstance(completion_details, dict):
            raise ValueError("usage_completion_details")
        usage = OpenCodePromptUsage(
            input_tokens=_number(usage_data.get("prompt_tokens"), "input_tokens"),
            output_tokens=_number(usage_data.get("completion_tokens"), "output_tokens"),
            reasoning_tokens=_number((completion_details or {}).get("reasoning_tokens"), "reasoning_tokens"),
            total_tokens=_number(usage_data.get("total_tokens"), "total_tokens"),
            cache_read_tokens=_number((prompt_details or {}).get("cached_tokens"), "cache_read_tokens"),
            cache_write_tokens=_number((prompt_details or {}).get("cache_write_tokens"), "cache_write_tokens"),
            cost=_number(usage_data.get("cost"), "cost", integer=False),
            provider_id="openrouter",
            model_id=_nonempty_optional_string(data.get("model"), "model_id"),
            finish=finish,
        )
    return OpenCodePromptResult(text=content.strip(), usage=usage)


async def request_json_schema(
    *, api_key: str, model: str, system_prompt: str, user_prompt: str,
    schema: Dict[str, Any], timeout: float = 180.0,
    temperature: float = 0.0,
    http_referer: Optional[str] = None, app_title: Optional[str] = None,
) -> Optional[OpenCodePromptResult]:
    try:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key_required")
        model_id = normalize_openrouter_model_id(model)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(float(timeout)) or timeout <= 0:
            raise ValueError("invalid_timeout")
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(float(temperature)) or not 0 <= temperature <= 2):
            raise ValueError("invalid_temperature")
        if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
            raise ValueError("prompts_required")
        if not isinstance(schema, dict):
            raise ValueError("schema_required")
        for name, value in (("http_referer", http_referer), ("app_title", app_title)):
            if value is not None and (not isinstance(value, str) or not value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value)):
                raise ValueError(f"invalid_{name}")
        body: Dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "response", "strict": True, "schema": copy.deepcopy(schema)}},
            "provider": {"require_parameters": True},
            "stream": False,
            "temperature": float(temperature),
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if http_referer is not None:
            headers["HTTP-Referer"] = http_referer
        if app_title is not None:
            headers["X-OpenRouter-Title"] = app_title
        status, raw = await _http_post_json(url=OPENROUTER_CHAT_COMPLETIONS_URL, headers=headers, body=body, timeout=float(timeout))
        if not 200 <= status < 300:
            try:
                error_data = _decode_json_bytes(raw)
            except ValueError as exc:
                _log_error(status, "protocol", _safe_value_error_code(exc))
                error_data = {}
            error = error_data.get("error", {}) if isinstance(error_data, dict) else {}
            _log_error(status, error.get("code") or error.get("type"), error.get("message")) if isinstance(error, dict) else _log_error(status, None, None)
            return None
        try:
            data = _decode_json_bytes(raw)
        except ValueError as exc:
            _log_error(status, "protocol", _safe_value_error_code(exc))
            return None
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            error = data["error"]
            _log_error(status, error.get("code") or error.get("type"), error.get("message"))
            return None
        try:
            return _result(data)
        except ValueError as exc:
            _log_error(status, "protocol", _safe_value_error_code(exc))
            return None
    except ValueError as exc:
        code = _safe_value_error_code(exc)
        _log_error("client", "protocol", code)
        return None
    except Exception as exc:
        _log_error("client", type(exc).__name__, "request failed")
        return None


async def _http_get_json(*, url: str, headers: Optional[Dict[str, str]] = None,
                         timeout: float) -> tuple[int, bytes]:
    import aiohttp

    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers or {}, timeout=client_timeout) as response:
            return response.status, await _read_bounded_response(response)


def _decimal_rate(value: Any, name: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f"pricing_{name}")
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError):
        raise ValueError(f"pricing_{name}") from None
    if not number.is_finite() or number < 0:
        raise ValueError(f"pricing_{name}")
    return number


def _rate_strings(pricing: Any) -> tuple[str, str, str]:
    if not isinstance(pricing, dict):
        raise ValueError("pricing_missing")
    input_rates: list[tuple[Decimal, str]] = []
    output_rates: list[tuple[Decimal, str]] = []
    request_rates: list[Decimal] = []

    input_keys = {
        "prompt", "input", "cache_read", "cache_write", "input_cache_read",
        "input_cache_write", "input_cache_write_1h",
    }
    output_keys = {"completion", "output", "long_completion", "long_output"}
    request_keys = {"request", "request_fee"}

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() == "overrides" and (
                    not isinstance(child, list) or any(not isinstance(item, dict) for item in child)
                ):
                    raise ValueError("pricing_override")
                visit(child, path + (str(key).lower(),))
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + (str(index),))
            return
        key = path[-1] if path else ""
        if key in input_keys:
            rate = _decimal_rate(value, key)
            input_rates.append((rate, value))
        elif key in output_keys:
            rate = _decimal_rate(value, key)
            output_rates.append((rate, value))
        elif key in request_keys:
            request_rates.append(_decimal_rate(value, key))

    visit(pricing, ())
    if not input_rates or not output_rates:
        raise ValueError("pricing_prompt_completion")
    input_rate = max(input_rates, key=lambda item: item[0])
    output_rate = max(output_rates, key=lambda item: item[0])
    request_fee = max(request_rates, default=Decimal("0"))
    return input_rate[1], output_rate[1], format(request_fee, "f")


def _payload(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or not isinstance(data.get("data"), dict):
        raise ValueError("catalog_object")
    return data["data"]


def _as_parameters(value: Any) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return set()
    return set(value)


def _reasoning_efforts(value: Any) -> set[str]:
    if not isinstance(value, dict):
        return set()
    result: set[str] = set()
    for key in ("effort", "efforts", "supported_efforts"):
        item = value.get(key)
        if isinstance(item, str):
            result.add(item)
        elif isinstance(item, list) and all(isinstance(entry, str) for entry in item):
            result.update(item)
        elif isinstance(item, dict):
            result.update(key for key, enabled in item.items() if isinstance(key, str) and enabled is True)
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"pricing_{name}")
    return value


def _make_catalog_entry(model_id: str, model_data: dict[str, Any], endpoint: dict[str, Any]) -> dict[str, Any]:
    frozen = {**_LEGACY_MODELS, **_FROZEN_MODELS}[model_id]
    exact_id = model_data.get("id")
    if exact_id != model_id:
        raise ValueError("pricing_model_id")
    canonical = model_data.get("canonical_slug", model_id)
    if not isinstance(canonical, str) or not canonical.strip():
        raise ValueError("pricing_canonical_model")
    endpoint_tag = endpoint.get("tag")
    provider_name = endpoint.get("provider_name")
    if endpoint_tag != frozen["provider_slug"] or provider_name != frozen["provider_name"]:
        raise ValueError("pricing_provider")
    endpoint_identity = endpoint.get("model_id", endpoint.get("id", model_id))
    if endpoint_identity not in {model_id, canonical}:
        raise ValueError("pricing_endpoint_model")
    endpoint_parameters = _as_parameters(endpoint.get("supported_parameters"))
    if not _REQUIRED_ENDPOINT_PARAMETERS <= endpoint_parameters:
        raise ValueError("pricing_parameters")
    if frozen["effort"] not in _reasoning_efforts(model_data.get("reasoning")):
        raise ValueError("pricing_effort")
    context_length = _positive_int(model_data.get("context_length"), "context_length")
    input_limit = endpoint.get("max_prompt_tokens", model_data.get("max_prompt_tokens"))
    if input_limit is None:
        input_limit = context_length - OPENROUTER_MAX_OUTPUT_TOKENS
    input_limit = min(_positive_int(input_limit, "max_input_tokens"), context_length - OPENROUTER_MAX_OUTPUT_TOKENS)
    if input_limit <= 0:
        raise ValueError("pricing_max_input_tokens")
    published_output = endpoint.get("max_completion_tokens", model_data.get("max_completion_tokens"))
    if published_output is not None and (isinstance(published_output, bool) or not isinstance(published_output, int) or published_output < OPENROUTER_MAX_OUTPUT_TOKENS):
        raise ValueError("pricing_max_output_tokens")
    max_output = OPENROUTER_MAX_OUTPUT_TOKENS
    pricing_sources = [model_data.get("pricing")]
    if endpoint.get("pricing") is not None:
        pricing_sources.append(endpoint["pricing"])
    rates: list[tuple[str, str, str]] = [_rate_strings(source) for source in pricing_sources]
    input_rate = max(rates, key=lambda rate: _decimal_rate(rate[0], "input"))[0]
    output_rate = max(rates, key=lambda rate: _decimal_rate(rate[1], "output"))[1]
    request_fee = max((_decimal_rate(rate[2], "request") for rate in rates), default=Decimal("0"))
    if request_fee > 0:
        raise ValueError("pricing_request_fee")
    if _decimal_rate(input_rate, "input") > frozen["max_input_rate"]:
        raise ValueError("pricing_input_rate")
    if _decimal_rate(output_rate, "output") > frozen["max_output_rate"]:
        raise ValueError("pricing_output_rate")
    model_metadata = copy.deepcopy(model_data)
    endpoint_metadata = copy.deepcopy(endpoint)
    model_metadata["source_url"] = f"{OPENROUTER_MODEL_URL}/{quote(model_id, safe='/')}"
    endpoint_metadata["source_url"] = f"{OPENROUTER_ENDPOINTS_URL}/{quote(model_id, safe='/')}/endpoints"
    return {
        "model": model_id,
        "canonical_model": canonical,
        "effort": frozen["effort"],
        "provider_slug": frozen["provider_slug"],
        "provider_name": frozen["provider_name"],
        "context_length": context_length,
        "max_input_tokens": input_limit,
        "max_output_tokens": max_output,
        "max_input_rate": input_rate,
        "max_output_rate": output_rate,
        "request_fee": format(request_fee, "f"),
        "request_fee_source": "listed" if any("request" in source for source in pricing_sources if isinstance(source, dict)) else "not_listed_assumed_zero",
        "model_metadata": model_metadata,
        "endpoint_metadata": endpoint_metadata,
    }


async def _catalog_model(model_id: str, *, timeout: float) -> dict[str, Any]:
    model_url = f"{OPENROUTER_MODEL_URL}/{quote(model_id, safe='/')}"
    endpoint_url = f"{OPENROUTER_ENDPOINTS_URL}/{quote(model_id, safe='/')}/endpoints"
    model_status, model_raw = await _http_get_json(url=model_url, timeout=timeout)
    endpoint_status, endpoint_raw = await _http_get_json(url=endpoint_url, timeout=timeout)
    if not 200 <= model_status < 300 or not 200 <= endpoint_status < 300:
        raise ValueError("pricing_http")
    model_data = _payload(_decode_json_bytes(model_raw))
    endpoints_data = _decode_json_bytes(endpoint_raw)
    if not isinstance(endpoints_data, dict) or not isinstance(endpoints_data.get("data"), dict):
        raise ValueError("pricing_endpoints")
    endpoint_catalog = endpoints_data["data"]
    canonical = model_data.get("canonical_slug", model_id)
    if not isinstance(canonical, str) or not canonical.strip():
        raise ValueError("pricing_canonical_model")
    endpoint_id = endpoint_catalog.get("id")
    if not isinstance(endpoint_id, str) or endpoint_id not in {model_id, canonical}:
        raise ValueError("pricing_endpoint_model")
    endpoints = endpoint_catalog.get("endpoints")
    if not isinstance(endpoints, list):
        raise ValueError("pricing_endpoints")
    frozen = _FROZEN_MODELS[model_id]
    selected = [item for item in endpoints if isinstance(item, dict)
                and item.get("tag") == frozen["provider_slug"]]
    if len(selected) != 1:
        raise ValueError("pricing_endpoint")
    return _make_catalog_entry(model_id, model_data, selected[0])


def _snapshot_id(snapshot: dict[str, Any]) -> str:
    content = copy.deepcopy(snapshot)
    content.pop("id", None)
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


async def fetch_pricing_snapshot(*, timeout: float = 30.0) -> dict[str, Any]:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or timeout <= 0:
        raise ValueError("invalid_timeout")
    models = {model_id: await _catalog_model(model_id, timeout=float(timeout)) for model_id in _FROZEN_MODELS}
    snapshot: dict[str, Any] = {
        "schema": OPENROUTER_SNAPSHOT_SCHEMA,
        "id": "",
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "models": models,
    }
    snapshot["id"] = _snapshot_id(snapshot)
    return copy.deepcopy(snapshot)


def _validate_snapshot_contract(snapshot: dict[str, Any]) -> None:
    if not isinstance(snapshot, dict) or snapshot.get("schema") != OPENROUTER_SNAPSHOT_SCHEMA:
        raise ValueError("pricing_snapshot_schema")
    if not isinstance(snapshot.get("captured_at"), str) or not snapshot["captured_at"].strip():
        raise ValueError("pricing_snapshot_captured_at")
    if not isinstance(snapshot.get("id"), str) or snapshot["id"] != _snapshot_id(snapshot):
        raise ValueError("pricing_snapshot_id")
    models = snapshot.get("models")
    if not isinstance(models, dict) or set(models) not in (set(_FROZEN_MODELS), set(_LEGACY_MODELS)):
        raise ValueError("pricing_snapshot_models")
    roles = _FROZEN_MODELS if set(models) == set(_FROZEN_MODELS) else _LEGACY_MODELS
    for model_id, entry in models.items():
        if not isinstance(entry, dict) or entry.get("model") != model_id:
            raise ValueError("pricing_snapshot_model")
        frozen = roles[model_id]
        if entry.get("effort") != frozen["effort"] or entry.get("provider_slug") != frozen["provider_slug"] or entry.get("provider_name") != frozen["provider_name"]:
            raise ValueError("pricing_snapshot_pin")
        for name in ("context_length", "max_input_tokens", "max_output_tokens"):
            _positive_int(entry.get(name), name)
        if entry["max_output_tokens"] != OPENROUTER_MAX_OUTPUT_TOKENS or entry["max_input_tokens"] > entry["context_length"] - OPENROUTER_MAX_OUTPUT_TOKENS:
            raise ValueError("pricing_snapshot_bounds")
        if _decimal_rate(entry.get("max_input_rate"), "input") > frozen["max_input_rate"] or _decimal_rate(entry.get("max_output_rate"), "output") > frozen["max_output_rate"]:
            raise ValueError("pricing_snapshot_rate")
        request_fee = _decimal_rate(entry.get("request_fee"), "request")
        if request_fee != 0:
            raise ValueError("pricing_snapshot_request_fee")
        if not isinstance(entry.get("request_fee_source"), str) or not isinstance(entry.get("model_metadata"), dict) or not isinstance(entry.get("endpoint_metadata"), dict):
            raise ValueError("pricing_snapshot_metadata")


async def validate_current_pricing(snapshot: dict[str, Any], *, timeout: float = 30.0) -> None:
    _validate_snapshot_contract(snapshot)
    if set(snapshot["models"]) != set(_FROZEN_MODELS):
        raise ValueError("pricing_legacy_snapshot")
    current = await fetch_pricing_snapshot(timeout=timeout)
    for model_id, original in snapshot["models"].items():
        actual = current["models"].get(model_id)
        if not isinstance(actual, dict):
            raise ValueError("pricing_current_model")
        for key in ("model", "canonical_model", "effort", "provider_slug", "provider_name"):
            if actual.get(key) != original.get(key):
                raise ValueError("pricing_current_identity")
        if any(actual[name] > original[name] for name in ("context_length", "max_input_tokens", "max_output_tokens")):
            raise ValueError("pricing_current_bounds")
        if (_decimal_rate(actual["max_input_rate"], "input") > _decimal_rate(original["max_input_rate"], "input")
                or _decimal_rate(actual["max_output_rate"], "output") > _decimal_rate(original["max_output_rate"], "output")):
            raise ValueError("pricing_current_rate")
        if _decimal_rate(actual["request_fee"], "request") > 0:
            raise ValueError("pricing_current_request_fee")


def _receipt_value(data: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in data:
            return data[name]
    usage = data.get("usage")
    if isinstance(usage, dict):
        for name in names:
            if name in usage:
                return usage[name]
    return None


def _optional_token(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    return _number(value, name)


def _usage_receipt(data: dict[str, Any], *, provider_name: Optional[str], model_id: Optional[str], finish: Optional[str],
                   cost: Optional[float]) -> tuple[Optional[OpenCodePromptUsage], Optional[str]]:
    usage_data = data.get("usage")
    usage_data = usage_data if isinstance(usage_data, dict) else {}
    details = usage_data.get("prompt_tokens_details")
    completion_details = usage_data.get("completion_tokens_details")
    if details is not None and not isinstance(details, dict):
        return None, "usage_prompt_details"
    if completion_details is not None and not isinstance(completion_details, dict):
        return None, "usage_completion_details"
    try:
        usage = OpenCodePromptUsage(
            input_tokens=_optional_token(usage_data.get("prompt_tokens"), "input_tokens"),
            output_tokens=_optional_token(usage_data.get("completion_tokens"), "output_tokens"),
            reasoning_tokens=_optional_token((completion_details or {}).get("reasoning_tokens"), "reasoning_tokens"),
            total_tokens=_optional_token(usage_data.get("total_tokens"), "total_tokens"),
            cache_read_tokens=_optional_token((details or {}).get("cached_tokens"), "cache_read_tokens"),
            cache_write_tokens=_optional_token((details or {}).get("cache_write_tokens"), "cache_write_tokens"),
            cost=cost, provider_id=provider_name or "openrouter", model_id=model_id, finish=finish,
        )
    except ValueError as exc:
        return None, _safe_value_error_code(exc)
    if not usage_data and cost is None and provider_name is None and model_id is None:
        return None, None
    return usage, None


def _reported_cost(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise ValueError("usage_cost")
    return float(value)


async def _generation_metadata(*, generation_id: str, api_key: str, timeout: float) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{OPENROUTER_GENERATION_URL}?id={quote(generation_id, safe='')}"
    lookup_timeout = min(float(timeout), 30.0)
    for attempt in range(3):
        try:
            status, raw = await _http_get_json(url=url, headers=headers, timeout=lookup_timeout)
            if 200 <= status < 300:
                value = _payload(_decode_json_bytes(raw))
                return value
        except Exception:
            pass
        if attempt < 2:
            await asyncio.sleep(1.0 if attempt == 0 else 2.0)
    return {}


def _response_identity(data: dict[str, Any]) -> tuple[Optional[str], Optional[str], Optional[bool], Optional[str]]:
    model = data.get("model")
    model = model if isinstance(model, str) and model.strip() else None
    generation_id = _receipt_value(data, "generation_id")
    if generation_id is None:
        value = data.get("id")
        generation_id = value if isinstance(value, str) and value.strip() else None
    provider = _receipt_value(data, "provider_name", "provider")
    if isinstance(provider, dict):
        provider = provider.get("name")
    provider = provider if isinstance(provider, str) and provider.strip() else None
    byok = _receipt_value(data, "is_byok")
    if byok is not None and not isinstance(byok, bool):
        raise ValueError("usage_is_byok")
    return model, generation_id, byok, provider


async def request_json_detailed(
    *, api_key: str, model: str, reasoning_effort: str, system_prompt: str,
    user_prompt: str, schema: dict[str, Any], pricing_snapshot: dict[str, Any],
    timeout: float = 300.0,
) -> DetailedResponse:
    try:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key_required")
        model_id = normalize_openrouter_model_id(model)
        if model_id not in _FROZEN_MODELS:
            raise ValueError("invalid_model")
        frozen = _FROZEN_MODELS[model_id]
        if reasoning_effort != frozen["effort"]:
            raise ValueError("invalid_reasoning_effort")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(float(timeout)) or timeout <= 0:
            raise ValueError("invalid_timeout")
        if not isinstance(system_prompt, str) or not isinstance(user_prompt, str) or not isinstance(schema, dict):
            raise ValueError("request_invalid")
        _validate_snapshot_contract(pricing_snapshot)
        snapshot_entry = pricing_snapshot["models"].get(model_id)
        if not isinstance(snapshot_entry, dict):
            raise ValueError("pricing_snapshot_model")
        body: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "response", "strict": True, "schema": copy.deepcopy(schema)}},
            "provider": {"only": [frozen["provider_slug"]], "require_parameters": True, "allow_fallbacks": False,
            "max_price": {
                "prompt": float(_decimal_rate(snapshot_entry["max_input_rate"], "input") * Decimal("1000000")),
                "completion": float(_decimal_rate(snapshot_entry["max_output_rate"], "output") * Decimal("1000000")),
            }},
            "reasoning": {"effort": reasoning_effort}, "stream": False, "max_tokens": OPENROUTER_MAX_OUTPUT_TOKENS,
        }
    except ValueError as exc:
        _log_error("client", "protocol", _safe_value_error_code(exc))
        return DetailedResponse(None, [], _safe_value_error_code(exc))

    started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started_monotonic = time.monotonic()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    requested_model = model_id
    record = CallRecord("openrouter", model_id, reasoning_effort, started, 0.0, None, "error",
                        requested_model=model_id, requested_variant=reasoning_effort,
                        request_max_tokens=OPENROUTER_MAX_OUTPUT_TOKENS, pricing_snapshot_id=pricing_snapshot["id"])
    data: dict[str, Any] = {}
    status_code: Any = "client"
    try:
        status_code, raw = await _http_post_json(url=OPENROUTER_CHAT_COMPLETIONS_URL, headers=headers, body=body, timeout=float(timeout))
        try:
            decoded = _decode_json_bytes(raw)
            data = decoded if isinstance(decoded, dict) else {}
        except ValueError as exc:
            elapsed = time.monotonic() - started_monotonic
            record.elapsed_sec = elapsed
            record.error = _safe_value_error_code(exc)
            record.status = "malformed"
            _log_error(status_code, "protocol", record.error)
            return DetailedResponse(None, [record], record.error)
    except Exception as exc:
        record.elapsed_sec = time.monotonic() - started_monotonic
        record.error = "timeout" if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) else "transport_error"
        record.status = "timeout" if record.error == "timeout" else "error"
        _log_error(status_code, type(exc).__name__, record.error)
        return DetailedResponse(None, [record], record.error)

    try:
        response_model, generation_id, is_byok, provider_name = _response_identity(data)
        usage_data = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        cost = _reported_cost(_receipt_value(data, "cost"))
        cost_source = "openrouter_usage" if cost is not None else None
        if 200 <= status_code < 300 and generation_id is not None and (cost is None or provider_name is None or is_byok is None):
            metadata = await _generation_metadata(generation_id=generation_id, api_key=api_key, timeout=float(timeout))
            cost = cost if cost is not None else _reported_cost(metadata.get("total_cost"))
            if cost is not None and cost_source is None:
                cost_source = "openrouter_generation"
            provider_name = provider_name or (metadata.get("provider_name") if isinstance(metadata.get("provider_name"), str) else None)
            if is_byok is None and isinstance(metadata.get("is_byok"), bool):
                is_byok = metadata["is_byok"]
            native = metadata.get("native_token")
            if isinstance(native, dict):
                merged_usage = dict(usage_data)
                for key, value in native.items():
                    merged_usage.setdefault(key, value)
                data["usage"] = merged_usage
            native_usage = {
                "prompt_tokens": metadata.get("native_tokens_prompt"),
                "completion_tokens": metadata.get("native_tokens_completion"),
                "reasoning_tokens": metadata.get("native_tokens_reasoning"),
                "cached_tokens": metadata.get("native_tokens_cached"),
            }
            if any(value is not None for value in native_usage.values()):
                merged_usage = dict(data.get("usage") if isinstance(data.get("usage"), dict) else {})
                for key, value in native_usage.items():
                    if value is not None:
                        if key == "cached_tokens":
                            details = dict(merged_usage.get("prompt_tokens_details") or {})
                            details.setdefault("cached_tokens", value)
                            merged_usage["prompt_tokens_details"] = details
                        else:
                            merged_usage.setdefault(key, value)
                data["usage"] = merged_usage
        usage, usage_error = _usage_receipt(data, provider_name=provider_name, model_id=response_model, finish=None, cost=cost)
        record.usage = usage
        record.generation_id = generation_id
        record.provider_name = provider_name
        record.is_byok = is_byok
        record.cost_source = cost_source
        record.model = response_model or model_id
        if usage_error is not None:
            raise ValueError(usage_error)
        if status_code < 200 or status_code >= 300:
            raise ValueError("http_permanent" if 400 <= status_code < 500 and status_code != 429 else "http_error")
        if response_model not in {model_id, snapshot_entry["canonical_model"]}:
            raise ValueError("model_mismatch")
        if provider_name != snapshot_entry["provider_name"] or is_byok is not False:
            raise ValueError("provider_identity_mismatch")
        if generation_id is None:
            raise ValueError("receipt_incomplete")
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("choices")
        choice = choices[0]
        finish = choice.get("finish_reason")
        record.usage = OpenCodePromptUsage(
            input_tokens=usage.input_tokens if usage else None, output_tokens=usage.output_tokens if usage else None,
            reasoning_tokens=usage.reasoning_tokens if usage else None, total_tokens=usage.total_tokens if usage else None,
            cache_read_tokens=usage.cache_read_tokens if usage else None, cache_write_tokens=usage.cache_write_tokens if usage else None,
            cost=cost, provider_id=provider_name, model_id=response_model, finish=finish if isinstance(finish, str) else None,
        )
        if finish != "stop" or "error" in data or "error" in choice or choice.get("refusal") is not None:
            raise ValueError("finish_reason" if finish != "stop" else "choice_error")
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("refusal") is not None:
            raise ValueError("message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content")
        try:
            json.loads(content)
        except (TypeError, json.JSONDecodeError):
            raise ValueError("response_json_invalid") from None
        record.status = "success"
        record.elapsed_sec = time.monotonic() - started_monotonic
        return DetailedResponse(content.strip(), [record], None)
    except (ValueError, TypeError) as exc:
        record.elapsed_sec = time.monotonic() - started_monotonic
        record.status = "error" if str(exc) in {"http_error", "model_mismatch", "provider_identity_mismatch", "receipt_incomplete"} else "malformed"
        record.error = _safe_value_error_code(exc) if str(exc) != "http_error" else "http_error"
        _log_error(status_code, "protocol", record.error)
        return DetailedResponse(None, [record], record.error)
    except Exception:
        record.elapsed_sec = time.monotonic() - started_monotonic
        record.status = "error"
        record.error = "response_error"
        _log_error(status_code, "protocol", record.error)
        return DetailedResponse(None, [record], record.error)
