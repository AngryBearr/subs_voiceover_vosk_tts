"""Direct, fail-closed DeepSeek JSON-object transport."""

from __future__ import annotations

import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from utils.opencode_transport import OpenCodePromptResult, OpenCodePromptUsage
from utils.length_check import check_length
from utils.deepseek_limits import resolve_deepseek_token_limit
from utils.llm_types import CallRecord, DetailedResponse

DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
_LEGACY_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_DETAILED_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_REDACTIONS = (
    (re.compile(r"Bearer\s+\S+", re.IGNORECASE), "Bearer [redacted]"),
    (re.compile(r"(?:api[_ -]?key|authorization)\s*[:=]\s*[^\s,;]+", re.IGNORECASE), "[redacted-credential]"),
    (re.compile(r"sk-[A-Za-z0-9_-]+"), "[redacted-key]"),
)
_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9._:/-]+$")
DEEPSEEK_CURRENT_MODEL = "deepseek-flash"
DEEPSEEK_LEGACY_MODEL = "deepseek-v4-flash"
DEEPSEEK_MODEL_CHOICES = (DEEPSEEK_CURRENT_MODEL, DEEPSEEK_LEGACY_MODEL)
_REASONING_EFFORTS = frozenset({"off", "low", "high", "max"})
_TOOL_NAME = "check_length"
_MAX_TOOL_CALLS_PER_TURN = 32


def _started_at() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_error(value: Any) -> str:
    message = value if isinstance(value, str) else "request failed"
    message = " ".join(message.split())
    for pattern, replacement in _REDACTIONS:
        message = pattern.sub(replacement, message)
    return message[:300] or "request failed"


def _validate_reasoning_effort(value: Optional[str]) -> Optional[str]:
    if value is not None and (not isinstance(value, str) or value not in _REASONING_EFFORTS):
        raise ValueError("invalid_reasoning_effort")
    return value


def _validate_model_choice(value: str) -> str:
    if value not in DEEPSEEK_MODEL_CHOICES:
        raise ValueError("deepseek_model_required")
    return value


def _tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": "Check Unicode Python code-point length against a character budget.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 0},
                },
                "required": ["text", "max_chars"],
                "additionalProperties": False,
            },
        },
    }


def _usage_from_data(data: Any, requested_model: str, finish: Any = None) -> Optional[OpenCodePromptUsage]:
    """Extract whatever valid billing fields are available, without requiring content validity."""
    if not isinstance(data, dict) or not isinstance(data.get("usage"), dict):
        return None
    raw = data["usage"]

    def token(name: str) -> Optional[int]:
        value = raw.get(name)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    prompt = token("prompt_tokens")
    hit = token("prompt_cache_hit_tokens")
    miss = token("prompt_cache_miss_tokens")
    details = raw.get("completion_tokens_details")
    reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if not isinstance(reasoning, int) or isinstance(reasoning, bool) or reasoning < 0:
        reasoning = None
    output = token("completion_tokens")
    total = token("total_tokens")
    effective_prompt = prompt if prompt is not None else (hit + miss if hit is not None and miss is not None else None)
    if effective_prompt is not None and output is not None and total is not None and total != effective_prompt + output:
        total = None
    if reasoning is not None and output is not None and reasoning > output:
        reasoning = None
    if prompt is None and hit is not None and miss is not None and effective_prompt is not None:
        prompt = effective_prompt
    if prompt is None and hit is None and miss is None and output is None and total is None and reasoning is None:
        return None
    model = data.get("model")
    if not isinstance(model, str) or not model.strip():
        model = "unknown"
    return OpenCodePromptUsage(
        input_tokens=prompt, output_tokens=output, reasoning_tokens=reasoning,
        total_tokens=total, cache_read_tokens=hit, cache_write_tokens=None,
        cost=None, provider_id="deepseek", model_id=model, finish=finish if isinstance(finish, str) else None,
    )


def _usage_consistency_error(data: Any) -> Optional[str]:
    """Return an error for contradictory numeric usage fields, if present."""
    if not isinstance(data, dict) or not isinstance(data.get("usage"), dict):
        return None
    raw = data["usage"]

    def token(name: str) -> Optional[int]:
        value = raw.get(name)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    prompt = token("prompt_tokens")
    hit = token("prompt_cache_hit_tokens")
    miss = token("prompt_cache_miss_tokens")
    output = token("completion_tokens")
    total = token("total_tokens")
    details = raw.get("completion_tokens_details")
    reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if not isinstance(reasoning, int) or isinstance(reasoning, bool) or reasoning < 0:
        reasoning = None
    effective_prompt = prompt
    if effective_prompt is None and hit is not None and miss is not None:
        effective_prompt = hit + miss
    if prompt is not None and hit is not None and miss is not None and prompt != hit + miss:
        return "usage_input_consistency"
    if effective_prompt is not None and output is not None and total is not None and total != effective_prompt + output:
        return "usage_total_consistency"
    if reasoning is not None and output is not None and reasoning > output:
        return "usage_reasoning_consistency"
    return None


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
    return await _read_bounded_response_with_limit(response, _DETAILED_MAX_RESPONSE_BYTES)


async def _read_bounded_response_with_limit(response: Any, limit: int) -> bytes:
    chunks = bytearray()
    async for chunk in response.content.iter_chunked(64 * 1024):
        chunks.extend(chunk)
        if len(chunks) > limit:
            raise ValueError("response_too_large")
    return bytes(chunks)


async def _http_post_json(*, url: str, headers: Dict[str, str], body: Dict[str, Any], timeout: float,
                          max_response_bytes: int = _DETAILED_MAX_RESPONSE_BYTES) -> tuple[int, bytes]:
    """POST JSON through aiohttp and return a bounded raw response body."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            headers=headers,
            json=body,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            return response.status, await _read_bounded_response_with_limit(response, max_response_bytes)


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


async def request_json_detailed(
    *, api_key: str, model: str, system_prompt: str, user_prompt: str,
    base_url: str = DEEPSEEK_DEFAULT_BASE_URL, timeout: float = 300.0,
     reasoning_effort: Optional[str] = None, max_tokens: Optional[int] = None,
    allow_length_tool: bool = False, max_tool_rounds: int = 1,
) -> DetailedResponse:
    """Make bounded, auditable DeepSeek calls, including optional length-tool turns."""
    calls: list[CallRecord] = []
    try:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key_required")
        model_id = _validate_model_choice(normalize_deepseek_model_id(model))
        url = normalize_deepseek_base_url(base_url).rstrip("/") + "/chat/completions"
        if not isinstance(system_prompt, str) or not system_prompt.strip() or not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("prompts_required")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or timeout <= 0:
            raise ValueError("invalid_timeout")
        effort = _validate_reasoning_effort(reasoning_effort)
        token_limit = resolve_deepseek_token_limit(effort, max_tokens)
        if isinstance(allow_length_tool, bool) is False or not isinstance(allow_length_tool, bool):
            raise ValueError("invalid_allow_length_tool")
        if isinstance(max_tool_rounds, bool) or not isinstance(max_tool_rounds, int) or max_tool_rounds < 0 or max_tool_rounds > 1:
            raise ValueError("invalid_max_tool_rounds")
        messages: list[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}
        ]
        tools = [_tool_schema()] if allow_length_tool and max_tool_rounds else None
        for turn in range(2 if allow_length_tool and max_tool_rounds else 1):
            body: Dict[str, Any] = {
                "model": model_id, "messages": messages, "response_format": {"type": "json_object"},
                "thinking": {"type": "disabled" if effort in (None, "off") else "enabled"},
                "stream": False, "temperature": 0.0, "max_tokens": token_limit,
            }
            if effort not in (None, "off"):
                body["reasoning_effort"] = effort
            if tools is not None:
                body["tools"] = tools
                body["tool_choice"] = "auto" if turn == 0 else "none"
            started = _started_at()
            started_clock = time.monotonic()
            try:
                status, raw = await _http_post_json(
                    url=url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    body=body, timeout=float(timeout),
                )
                data: Any = None
                decode_error: Optional[str] = None
                try:
                    data = _decode_json_bytes(raw)
                except ValueError as exc:
                    decode_error = str(exc)
                finish = None
                if isinstance(data, dict) and isinstance(data.get("choices"), list) and data["choices"] and isinstance(data["choices"][0], dict):
                    finish = data["choices"][0].get("finish_reason")
                usage = _usage_from_data(data, model_id, finish)
                actual_model = (
                    data.get("model")
                    if isinstance(data, dict) and isinstance(data.get("model"), str) and data["model"].strip()
                    else "unknown"
                )
                elapsed = time.monotonic() - started_clock
                record = CallRecord("deepseek", actual_model, None, started, elapsed, usage, "success",
                                    requested_model=model_id, requested_variant=effort,
                                    request_max_tokens=token_limit, pricing_snapshot_id="deepseek-flash-2026-09-10")
                calls.append(record)
                if not 200 <= status < 300:
                    error = "provider_http_error"
                    if isinstance(data, dict) and isinstance(data.get("error"), dict):
                        error = _safe_error(data["error"].get("message") or data["error"].get("type"))
                    record.status = "error"
                    record.error = error
                    return DetailedResponse(None, calls, error)
                if decode_error is not None:
                    record.status = "malformed"
                    record.error = _safe_error(decode_error)
                    return DetailedResponse(None, calls, "malformed_response")
                try:
                    usage_error = _usage_consistency_error(data)
                    if usage_error is not None:
                        raise ValueError(usage_error)
                    if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or len(data["choices"]) != 1:
                        raise ValueError("choices")
                    choice = data["choices"][0]
                    if not isinstance(choice, dict):
                        raise ValueError("choice")
                    message = choice.get("message")
                    if not isinstance(message, dict):
                        raise ValueError("message")
                    finish = choice.get("finish_reason")
                    tool_calls = message.get("tool_calls")
                    if tool_calls is not None and not isinstance(tool_calls, list):
                        raise ValueError("tool_calls")
                    calls_count = len(tool_calls or [])
                    record.tool_calls = calls_count
                    if calls_count > _MAX_TOOL_CALLS_PER_TURN:
                        raise ValueError("tool_call_limit")
                    # Keep the provider's assistant object, including reasoning_content, only in this local conversation.
                    if tool_calls and turn == 0 and allow_length_tool and max_tool_rounds:
                        messages.append(message)
                        for tool_call in tool_calls:
                            if not isinstance(tool_call, dict):
                                raise ValueError("tool_call")
                            function = tool_call.get("function")
                            call_id = tool_call.get("id")
                            if not isinstance(function, dict) or not isinstance(call_id, str) or not call_id:
                                raise ValueError("tool_call")
                            name = function.get("name")
                            arguments = function.get("arguments")
                            if name != _TOOL_NAME or not isinstance(arguments, str):
                                result = {"error": "unknown_tool"}
                            else:
                                try:
                                    args = json.loads(arguments)
                                    if not isinstance(args, dict) or set(args) != {"text", "max_chars"}:
                                        raise ValueError
                                    result = check_length(args["text"], args["max_chars"])
                                except (TypeError, ValueError, json.JSONDecodeError):
                                    result = {"error": "invalid_tool_arguments"}
                            messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result, ensure_ascii=False, separators=(",", ":"))})
                        continue
                    if finish == "length":
                        raise ValueError("finish_reason_length")
                    if finish != "stop":
                        raise ValueError("finish_reason")
                    content = message.get("content")
                    if not isinstance(content, str) or not content.strip() or message.get("refusal") is not None:
                        raise ValueError("content")
                    return DetailedResponse(content.strip(), calls)
                except (TypeError, ValueError, RuntimeError) as exc:
                    # The usage-bearing call above remains recorded even for invalid content.
                    record.status = "malformed"
                    record.error = _safe_error(str(exc))
                    return DetailedResponse(None, calls, "malformed_response")
            except Exception as exc:
                calls.append(CallRecord(
                    "deepseek", "unknown", None, started, time.monotonic() - started_clock, None,
                    "timeout" if isinstance(exc, TimeoutError) else "error",
                    error=_safe_error(type(exc).__name__), requested_model=model_id,
                    requested_variant=effort, request_max_tokens=token_limit,
                    pricing_snapshot_id="deepseek-flash-2026-09-10",
                ))
                return DetailedResponse(None, calls, "request_failed")
        return DetailedResponse(None, calls, "tool_round_limit")
    except (ValueError, TypeError) as exc:
        return DetailedResponse(None, calls, _safe_error(str(exc)))


async def request_json_object(*, api_key: str, model: str, system_prompt: str, user_prompt: str,
                               base_url: str = DEEPSEEK_DEFAULT_BASE_URL, timeout: float = 180.0,
                               temperature: float = 0.0, reasoning_effort: Optional[str] = None,
                               max_tokens: Optional[int] = None) -> Optional[OpenCodePromptResult]:
    """Legacy optional-result API retained for existing callers."""
    try:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key_required")
        model_id = _validate_model_choice(normalize_deepseek_model_id(model))
        url = normalize_deepseek_base_url(base_url).rstrip("/") + "/chat/completions"
        if not isinstance(system_prompt, str) or not system_prompt.strip() or not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("prompts_required")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or timeout <= 0:
            raise ValueError("invalid_timeout")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(float(temperature)) or not 0 <= temperature <= 2:
            raise ValueError("invalid_temperature")
        effort = _validate_reasoning_effort(reasoning_effort)
        token_limit = resolve_deepseek_token_limit(effort, max_tokens)
        body: Dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled" if effort in (None, "off") else "enabled"},
            "stream": False,
            "temperature": float(temperature),
            "max_tokens": token_limit,
        }
        if effort not in (None, "off"):
            body["reasoning_effort"] = effort
        status, raw = await _http_post_json(
            url=url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            body=body, timeout=float(timeout), max_response_bytes=_LEGACY_MAX_RESPONSE_BYTES,
        )
        if not 200 <= status < 300:
            return None
        return _result(_decode_json_bytes(raw), model_id)
    except Exception as exc:
        _log_error("client", "protocol", _safe_message(str(exc)))
        return None
