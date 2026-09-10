"""Centralized DeepSeek completion-token limits."""

from __future__ import annotations

from typing import Final


DEEPSEEK_LIMIT_POLICY_VERSION: Final[str] = "deepseek-limits-v2"
DEEPSEEK_PROJECT_MAX_TOKENS: Final[int] = 384_000
DEEPSEEK_PROVIDER_ADVERTISED_CAP_LABEL: Final[str] = "384K"

_DEFAULT_LIMIT: Final[int] = 8_192
_EFFORT_LIMITS: Final[dict[str, int]] = {
    "off": _DEFAULT_LIMIT,
    "low": 65_536,
    "high": 262_144,
    "max": 262_144,
}


def resolve_deepseek_token_limit(reasoning_effort: str | None, override: int | None = None) -> int:
    """Resolve the conservative project ceiling; the exact provider maximum is not verified."""
    if reasoning_effort is not None and (
        not isinstance(reasoning_effort, str) or reasoning_effort not in _EFFORT_LIMITS
    ):
        raise ValueError("invalid_reasoning_effort")
    if override is not None:
        if (
            isinstance(override, bool)
            or not isinstance(override, int)
            or not 1 <= override <= DEEPSEEK_PROJECT_MAX_TOKENS
        ):
            raise ValueError("invalid_max_tokens")
        return override
    return _EFFORT_LIMITS["off" if reasoning_effort is None else reasoning_effort]
