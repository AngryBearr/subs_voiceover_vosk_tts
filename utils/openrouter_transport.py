"""Direct, fail-closed OpenRouter native JSON-schema transport."""

from __future__ import annotations

import copy
import json
import math
import re
import sys
from typing import Any, Dict, Optional

from utils.opencode_transport import OpenCodePromptResult, OpenCodePromptUsage

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
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
    print(f"OpenRouter error status={safe_status} type={safe_type} message={_safe_message(message)}", file=sys.stderr)


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
            "response_format": {"type": "json_schema", "json_schema": {"name": "subtitle_semantic_verification", "strict": True, "schema": copy.deepcopy(schema)}},
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
