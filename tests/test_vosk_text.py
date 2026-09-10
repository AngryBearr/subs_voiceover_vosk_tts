"""Tests for the diagnostic Vosk punctuation mapping."""

from __future__ import annotations

from utils.vosk_text import normalize_vosk_text


def test_normalize_vosk_text_translates_only_supported_punctuation() -> None:
    text = "« » “ ” „ ‟ ‚ ‘ ’ ' — – ‒ − …"
    assert normalize_vosk_text(text) == '" " " " " " " " " " - - - - ...'


def test_normalize_vosk_text_preserves_words_numbers_case_and_emoji() -> None:
    text = "  Привет, WORLD 123! 🙂\u00a0ещё\nслово  "
    assert normalize_vosk_text(text) == "Привет, WORLD 123! 🙂 ещё слово"
