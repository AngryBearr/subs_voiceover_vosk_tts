"""Shared typed records for accountable LLM transport calls."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Optional

from utils.deepseek_limits import DEEPSEEK_PROJECT_MAX_TOKENS

if TYPE_CHECKING:
    from utils.opencode_transport import OpenCodePromptUsage


@dataclass
class CallRecord:
    provider: str
    model: str
    variant: Optional[str]
    started_at: str
    elapsed_sec: float
    usage: Optional[OpenCodePromptUsage]
    status: str
    tool_calls: int = 0
    error: Optional[str] = None
    requested_model: Optional[str] = None
    requested_variant: Optional[str] = None
    request_max_tokens: Optional[int] = None
    pricing_snapshot_id: Optional[str] = None


@dataclass
class DetailedResponse:
    text: Optional[str]
    calls: list[CallRecord]
    error: Optional[str] = None


def call_record_from_dict(data: dict[str, Any]) -> CallRecord:
    """Deserialize a persisted call record, including its typed usage object."""
    if not isinstance(data, dict):
        raise TypeError("call record must be a mapping")

    required = ("provider", "model", "started_at", "elapsed_sec", "usage", "status")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"call record missing fields: {', '.join(missing)}")

    def optional_string(name: str) -> Optional[str]:
        value = data.get(name)
        if value is not None and not isinstance(value, str):
            raise TypeError(f"call record {name} must be a string or null")
        return value

    for name in ("provider", "model", "started_at", "status"):
        if not isinstance(data[name], str):
            raise TypeError(f"call record {name} must be a string")
    variant = optional_string("variant")
    elapsed_sec = data["elapsed_sec"]
    if (isinstance(elapsed_sec, bool) or not isinstance(elapsed_sec, (int, float))
            or not math.isfinite(float(elapsed_sec)) or elapsed_sec < 0):
        raise TypeError("call record elapsed_sec must be a finite nonnegative number")

    usage_data = data["usage"]
    usage: Optional["OpenCodePromptUsage"]
    if usage_data is None:
        usage = None
    else:
        if not isinstance(usage_data, dict):
            raise TypeError("call record usage must be a mapping or null")
        from utils.opencode_transport import OpenCodePromptUsage

        usage_fields = {
            "input_tokens", "output_tokens", "reasoning_tokens", "total_tokens",
            "cache_read_tokens", "cache_write_tokens", "cost", "provider_id",
            "model_id", "finish",
        }
        unknown_usage = set(usage_data) - usage_fields
        if unknown_usage:
            raise ValueError(f"usage contains unknown fields: {', '.join(sorted(unknown_usage))}")
        usage = OpenCodePromptUsage(**usage_data)

    tool_calls = data.get("tool_calls", 0)
    if isinstance(tool_calls, bool) or not isinstance(tool_calls, int) or tool_calls < 0:
        raise TypeError("call record tool_calls must be a nonnegative integer")

    error = optional_string("error")
    requested_model = optional_string("requested_model")
    requested_variant = optional_string("requested_variant")
    request_max_tokens = data.get("request_max_tokens")
    if request_max_tokens is not None and (
        isinstance(request_max_tokens, bool) or not isinstance(request_max_tokens, int)
        or not 1 <= request_max_tokens <= DEEPSEEK_PROJECT_MAX_TOKENS
    ):
        raise TypeError("call record request_max_tokens must be a positive integer or null")
    pricing_snapshot_id = optional_string("pricing_snapshot_id")
    return CallRecord(
        provider=data["provider"], model=data["model"], variant=variant,
        started_at=data["started_at"], elapsed_sec=elapsed_sec, usage=usage,
        status=data["status"], tool_calls=tool_calls, error=error,
        requested_model=requested_model, requested_variant=requested_variant,
        request_max_tokens=request_max_tokens, pricing_snapshot_id=pricing_snapshot_id,
    )
