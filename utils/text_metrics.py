"""Small dependency-free text metrics shared by timing and shortening code."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


def count_words(text: str) -> int:
    """Count whitespace-separated tokens without changing calibration semantics."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return len(text.split())


def _normalise_edge_punctuation(token: str) -> str:
    start = 0
    end = len(token)
    while start < end and unicodedata.category(token[start]).startswith("P"):
        start += 1
    while end > start and unicodedata.category(token[end - 1]).startswith("P"):
        end -= 1
    return token[start:end].casefold()


def is_repetition_candidate(text: str) -> bool:
    """Return whether all of at least two tokens are the same real word."""
    tokens = text.split()
    if len(tokens) < 2:
        return False
    normalised = [_normalise_edge_punctuation(token) for token in tokens]
    return bool(
        normalised[0]
        and all(normalised_token == normalised[0] for normalised_token in normalised)
        and any(character.isalnum() for character in normalised[0])
    )


@dataclass(frozen=True)
class TextClassification:
    """Shared budget classification used by planner, analyzer and verifier."""

    status: str
    words_count: int
    repetition_candidate: bool
    short_timing_risk: bool
    reason: str


def classify_text(text: str, max_chars: int) -> TextClassification:
    """Classify text against a character cap without modifying the text."""
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 0:
        raise ValueError("max_chars must be a non-negative integer")
    words_count = count_words(text)
    repetition_candidate = is_repetition_candidate(text)
    if len(text) <= max_chars:
        return TextClassification("fits", words_count, repetition_candidate, False, "fits")
    short_timing_risk = 1 <= words_count <= 3
    if short_timing_risk and repetition_candidate:
        return TextClassification(
            "needs_shortening", words_count, repetition_candidate, True, "repetition_candidate"
        )
    if short_timing_risk:
        return TextClassification("short_timing", words_count, repetition_candidate, True, "short_timing")
    return TextClassification("needs_shortening", words_count, repetition_candidate, False, "needs_shortening")
