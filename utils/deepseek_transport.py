"""Direct, fail-closed DeepSeek JSON-object transport."""

from __future__ import annotations

import json
import math
import re
import sys
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from utils.opencode_transport import OpenCodePromptResult, OpenCodePromptUsage

DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_REDACTIONS = (
    (re.compile(r"Bearer\s+\S+", re.IGNORECASE), "Bearer [redacted]"),
    (re.compile(r"(?:api[_ -]?key|authorization)\s*[:=]\s*[^\s,;]+", re.IGNORECASE), "[redacted-credential]"),
    (re.compile(r"sk-[A-Za-z0-9_-]+"), "[redacted-key]"),
)
_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9._:/-]+$")


def normalize_deepseek_model_id(model: str) -> str:
    if not isinstance(model, str):
        raise ValueError("model_required")
    value = model.strip()
    if not value or _MODEL_PATTERN.fullmatch(value) is None:
        raise ValueError("invalid_model")
    return value


def normalize_deepseek_base_url(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("base_url_required")
    value = value.strip()
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("invalid_base_url")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username is not None or parsed.password is not None:
        raise ValueError("invalid_base_url")
    if parsed.query or parsed.fragment or "?" in value or "#" in value:
        raise ValueError("invalid_base_url")
    return value


def _safe_message(value: Any) -> str:
    message = value if isinstance(value, str) else "unspecified provider error"
    message = " ".join(message.split())
    for pattern, replacement in _REDACTIONS:
        message = pattern.sub(replacement, message)
    return message[:500] or "unspecified provider error"


def _log_error(status: Any, error_type: Any, message: Any) -> None:
    print(f"DeepSeek error status={status if isinstance(status, (int, str)) else 'unknown'} "
          f"type={_safe_message(error_type)[:100]} message={_safe_message(message)}", file=sys.stderr)


def _decode_json_bytes(raw: bytes) -> Any:
    first = raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw
    first = first.lstrip(b" \t\r\n")
    if not first:
        raise ValueError("response_json_empty")
    if first.startswith(b"data:"):
        raise ValueError("response_json_sse")
    if first.startswith(b"<"):
        raise ValueError("response_json_html")
    if first.startswith(b"\x1f\x8b"):
        raise ValueError("response_json_gzip")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("response_json_invalid_utf8") from None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("response_json_invalid") from None


async def _read_bounded_response(response: Any) -> bytes:
    chunks = bytearray()
    async for chunk in response.content.iter_chunked(64 * 1024):
        chunks.extend(chunk)
        if len(chunks) > _MAX_RESPONSE_BYTES:
            raise ValueError("response_too_large")
    return bytes(chunks)


async def _http_post_json(*, url: str, headers: Dict[str, str], body: Dict[str, Any], timeout: float) -> tuple[int, bytes]:
    """POST JSON through aiohttp and return a bounded raw response body."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            headers=headers,
            json=body,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            return response.status, await _read_bounded_response(response)


def _token(value: Any, name: str, required: bool = False) -> Optional[int]:
    if value is None:
        if required:
            raise ValueError(f"usage_{name}")
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"usage_{name}")
    return value


def _result(data: Any, requested_model: str) -> OpenCodePromptResult:
    if not isinstance(data, dict) or "error" in data:
        raise ValueError("provider_error" if isinstance(data, dict) and "error" in data else "response_object")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("choices")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("finish_reason")
    if "error" in choice or choice.get("refusal") is not None:
        raise ValueError("choice_error")
    message = choice.get("message")
    if not isinstance(message, dict) or "error" in message or message.get("refusal") is not None:
        raise ValueError("message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content")
    response_model = data.get("model")
    if response_model is not None:
        if not isinstance(response_model, str) or not response_model.strip():
            raise ValueError("usage_model_id")
        response_model = normalize_deepseek_model_id(response_model)
    else:
        response_model = requested_model
    usage_data = data.get("usage")
    if not isinstance(usage_data, dict):
        raise ValueError("usage")
    prompt = _token(usage_data.get("prompt_tokens"), "input_tokens")
    hit = _token(usage_data.get("prompt_cache_hit_tokens"), "cache_read_tokens")
    miss = _token(usage_data.get("prompt_cache_miss_tokens"), "cache_miss_tokens")
    if prompt is None:
        if hit is None or miss is None:
            raise ValueError("usage_input_tokens")
        prompt = hit + miss
    elif hit is not None and miss is not None and prompt != hit + miss:
        raise ValueError("usage_input_consistency")
    output = _token(usage_data.get("completion_tokens"), "output_tokens", True)
    total = _token(usage_data.get("total_tokens"), "total_tokens")
    if total is not None and total != prompt + output:
        raise ValueError("usage_total_consistency")
    details = usage_data.get("completion_tokens_details")
    if details is not None and not isinstance(details, dict):
        raise ValueError("usage_completion_details")
    reasoning = _token((details or {}).get("reasoning_tokens"), "reasoning_tokens")
    usage = OpenCodePromptUsage(input_tokens=prompt, output_tokens=output, reasoning_tokens=reasoning,
                                total_tokens=total if total is not None else prompt + output,
                                cache_read_tokens=hit, cache_write_tokens=None, cost=None,
                                provider_id="deepseek", model_id=response_model or requested_model, finish="stop")
    return OpenCodePromptResult(text=content.strip(), usage=usage)


async def request_json_object(*, api_key: str, model: str, system_prompt: str, user_prompt: str,
                              base_url: str = DEEPSEEK_DEFAULT_BASE_URL, timeout: float = 180.0,
                              temperature: float = 0.0) -> Optional[OpenCodePromptResult]:
    try:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key_required")
        model_id = normalize_deepseek_model_id(model)
        url = normalize_deepseek_base_url(base_url).rstrip("/") + "/chat/completions"
        if not isinstance(system_prompt, str) or not system_prompt.strip() or not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("prompts_required")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or timeout <= 0:
            raise ValueError("invalid_timeout")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(float(temperature)) or not 0 <= temperature <= 2:
            raise ValueError("invalid_temperature")
        body: Dict[str, Any] = {"model": model_id, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                                "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"}, "stream": False, "temperature": float(temperature)}
        status, raw = await _http_post_json(
            url=url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            body=body,
            timeout=float(timeout),
        )
        if not 200 <= status < 300:
            try:
                error_data = _decode_json_bytes(raw)
            except ValueError as exc:
                _log_error(status, "protocol", str(exc))
                return None
            error = error_data.get("error") if isinstance(error_data, dict) else None
            if isinstance(error, dict):
                _log_error(status, error.get("code") or error.get("type"), error.get("message"))
            else:
                _log_error(status, "protocol", "provider_error")
            return None
        return _result(_decode_json_bytes(raw), model_id)
    except ValueError as exc:
        _log_error("client", "protocol", _safe_message(str(exc)))
        return None
    except Exception as exc:
        _log_error("client", type(exc).__name__, "request failed")
        return None
