"""Direct, fail-closed Ollama Cloud text JSON transport."""

from __future__ import annotations

import json
import math
import re
import sys
from typing import Any, Dict, Optional

from utils.opencode_transport import OpenCodePromptResult, OpenCodePromptUsage

OLLAMA_CLOUD_CHAT_URL = "https://ollama.com/api/chat"
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_SAFE_CODES = frozenset({
    "response_json", "response_json_empty", "response_json_sse", "response_json_html",
    "response_json_gzip", "response_json_invalid_utf8", "response_json_invalid",
    "response_object", "top_level_error", "model", "done", "done_reason", "message",
    "role", "content", "tool_calls", "usage_prompt_eval_count", "usage_eval_count",
    "response_too_large", "api_key_required", "invalid_model", "model_required",
    "prompts_required", "invalid_timeout", "invalid_temperature", "invalid_seed",
})
_SAFE_PATTERNS = (re.compile(r"usage_[a-z_]+"), re.compile(r"invalid_[a-z_]+"))
_REDACTIONS = (
    (re.compile(r"Bearer\s+\S+", re.IGNORECASE), "[redacted-credential]"),
    (re.compile(r"(?:authorization|api[_ -]?key)\s*[:=]\s*[^\s,;]+", re.IGNORECASE), "[redacted-credential]"),
)


def normalize_ollama_cloud_model_id(model: str) -> str:
    if not isinstance(model, str):
        raise ValueError("model_required")
    value = model.strip()
    if not value:
        raise ValueError("invalid_model")
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("invalid_model")
    if not re.fullmatch(r"[A-Za-z0-9._:/-]+", value):
        raise ValueError("invalid_model")
    return value


def _safe_code(exc: ValueError) -> str:
    code = str(exc)
    return code if code in _SAFE_CODES or any(pattern.fullmatch(code) for pattern in _SAFE_PATTERNS) else "protocol_error"


def _safe_message(value: Any, sensitive: Optional[str] = None) -> str:
    message = value if isinstance(value, str) else "request failed"
    message = " ".join(message.split())
    if sensitive:
        message = message.replace(sensitive, "[redacted-credential]")
    for pattern, replacement in _REDACTIONS:
        message = pattern.sub(replacement, message)
    return message[:200] or "request failed"


def _log(status: Any, error_type: Any, message: Any, sensitive: Optional[str] = None) -> None:
    safe_type = _safe_message(error_type, sensitive)
    print(f"Ollama Cloud error status={status if isinstance(status, (int, str)) else 'client'} "
          f"type={safe_type} message={_safe_message(message, sensitive)}", file=sys.stderr)


def _decode_json_bytes(raw: bytes) -> Any:
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
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("response_json_invalid") from None


async def _read_bounded_response(response: Any) -> bytes:
    chunks = bytearray()
    async for chunk in response.content.iter_chunked(64 * 1024):
        chunks.extend(chunk)
        if len(chunks) > _MAX_RESPONSE_BYTES:
            raise ValueError("response_too_large")
    return bytes(chunks)


async def _http_post_json(*, url: str, headers: Dict[str, str], body: Dict[str, Any], timeout: float) -> tuple[int, bytes]:
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            return response.status, await _read_bounded_response(response)


def _nonnegative_int(data: Dict[str, Any], name: str) -> Optional[int]:
    if name not in data:
        return None
    value = data[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"usage_{name}")
    return value


def _result(data: Any, requested_model: str) -> OpenCodePromptResult:
    if not isinstance(data, dict):
        raise ValueError("response_object")
    if data.get("error") is not None:
        raise ValueError("top_level_error")
    actual_model = data.get("model", requested_model)
    if not isinstance(actual_model, str) or not actual_model.strip():
        raise ValueError("model")
    if data.get("done") is not True:
        raise ValueError("done")
    if data.get("done_reason") != "stop":
        raise ValueError("done_reason")
    message = data.get("message")
    if not isinstance(message, dict):
        raise ValueError("message")
    if message.get("role") != "assistant":
        raise ValueError("role")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content")
    tool_calls = message.get("tool_calls")
    if tool_calls is not None and (not isinstance(tool_calls, list) or tool_calls):
        raise ValueError("tool_calls")
    prompt_count = _nonnegative_int(data, "prompt_eval_count")
    eval_count = _nonnegative_int(data, "eval_count")
    usage = None
    if prompt_count is not None or eval_count is not None:
        usage = OpenCodePromptUsage(
            input_tokens=prompt_count, output_tokens=eval_count,
            total_tokens=prompt_count + eval_count if prompt_count is not None and eval_count is not None else None,
            provider_id="ollama-cloud", model_id=actual_model, finish="stop",
        )
    return OpenCodePromptResult(text=content.strip(), usage=usage)


def _provider_error(data: Any) -> tuple[str, str] | None:
    if not isinstance(data, dict):
        return None
    error = data.get("error")
    if isinstance(error, str):
        return "provider_error", error
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        category = code if isinstance(code, str) and code.strip() else "provider_error"
        detail = message if isinstance(message, str) and message.strip() else "request failed"
        return category, detail
    return None


async def request_strict_json(*, api_key: str, model: str, system_prompt: str, user_prompt: str,
                              timeout: float = 300.0, temperature: float = 0.0,
                              seed: int = 42) -> Optional[OpenCodePromptResult]:
    try:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key_required")
        model_id = normalize_ollama_cloud_model_id(model)
        if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
            raise ValueError("prompts_required")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(float(timeout)) or timeout <= 0):
            raise ValueError("invalid_timeout")
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(float(temperature)) or not 0 <= temperature <= 2):
            raise ValueError("invalid_temperature")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("invalid_seed")
        body = {"model": model_id, "messages": [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}], "stream": False,
                "options": {"temperature": float(temperature), "seed": seed}}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}
        status, raw = await _http_post_json(url=OLLAMA_CLOUD_CHAT_URL, headers=headers, body=body, timeout=float(timeout))
        try:
            data = _decode_json_bytes(raw)
        except ValueError as exc:
            if not 200 <= status < 300:
                _log(status, "protocol", "http_error")
            else:
                _log(status, "protocol", _safe_code(exc))
            return None
        if not 200 <= status < 300:
            provider_error = _provider_error(data)
            if provider_error is None:
                _log(status, "protocol", "http_error")
            else:
                _log(status, provider_error[0], provider_error[1], api_key)
            return None
        try:
            return _result(data, model_id)
        except ValueError as exc:
            _log(status, "protocol", _safe_code(exc))
            return None
    except ValueError as exc:
        _log("client", "protocol", _safe_code(exc))
        return None
    except Exception:
        _log("client", "client_error", "request failed")
        return None
