from __future__ import annotations

from utils.text_metrics import count_words, is_repetition_candidate


def test_count_words_keeps_split_compatibility() -> None:
    assert count_words("... ...") == 2
    assert count_words("  one\ttwo  ") == 2


def test_repetition_uses_whitespace_tokens_and_unicode_edges() -> None:
    assert is_repetition_candidate("Пошли, пошли, пошли.") is True
    assert is_repetition_candidate("ДА! да?") is True
    assert is_repetition_candidate("Да, нет, да.") is False
    assert is_repetition_candidate("... ...") is False
