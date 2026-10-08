from __future__ import annotations

import pytest

from utils.length_check import check_length, prepare_length_text


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


def test_normalized_basis_handles_vosk_punctuation_and_whitespace() -> None:
    text = "  «Да…» — да.  "
    assert prepare_length_text(text) == text
    normalized = prepare_length_text(text, "normalized_tts_text")
    assert normalized == '"Да..." - да.'
    assert check_length(text, len(normalized), "normalized_tts_text")["fits"] is True
    assert check_length(text, len(normalized) - 1, "normalized_tts_text")["fits"] is False
    with pytest.raises(ValueError):
        prepare_length_text(text, "unknown")  # type: ignore[arg-type]
