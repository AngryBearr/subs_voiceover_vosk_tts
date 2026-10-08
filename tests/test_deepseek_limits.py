from __future__ import annotations

import pytest

from utils.deepseek_limits import DEEPSEEK_PROJECT_MAX_TOKENS, resolve_deepseek_token_limit


@pytest.mark.parametrize(
    ("effort", "expected"),
    [(None, 8192), ("off", 8192), ("low", 65536), ("high", 262144), ("max", 262144)],
)
def test_reasoning_effort_defaults(effort: str | None, expected: int) -> None:
    assert resolve_deepseek_token_limit(effort) == expected


@pytest.mark.parametrize("effort", ["medium", "disabled", "LOW", "", 1, False])
def test_invalid_reasoning_effort_is_rejected(effort: object) -> None:
    with pytest.raises(ValueError, match="invalid_reasoning_effort"):
        resolve_deepseek_token_limit(effort)  # type: ignore[arg-type]


@pytest.mark.parametrize("override", [True, False, 0, -1, DEEPSEEK_PROJECT_MAX_TOKENS + 1, "8192"])
def test_invalid_override_is_rejected(override: object) -> None:
    with pytest.raises(ValueError, match="invalid_max_tokens"):
        resolve_deepseek_token_limit("low", override)  # type: ignore[arg-type]


def test_explicit_override_is_retained_for_every_mode() -> None:
    for effort in (None, "off", "low", "high", "max"):
        assert resolve_deepseek_token_limit(effort, 123_457) == 123_457
