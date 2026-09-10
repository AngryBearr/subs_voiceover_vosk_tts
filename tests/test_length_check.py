from __future__ import annotations

import pytest

from utils.length_check import check_length


def test_unicode_code_points_and_budget() -> None:
    assert check_length("a 😀 ", 4) == {"length": 4, "max_chars": 4, "over_by": 0, "fits": True}
    assert check_length("абв", 1)["over_by"] == 2


@pytest.mark.parametrize("text", [None, 1, True])
def test_text_type_is_rejected(text: object) -> None:
    with pytest.raises(TypeError):
        check_length(text, 1)  # type: ignore[arg-type]


def test_invalid_budget_is_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        check_length("x", -1)
    with pytest.raises(TypeError):
        check_length("x", True)  # type: ignore[arg-type]
