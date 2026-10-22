"""Deterministic, engine-neutral text measures used by the duration oracle.

Syllables are approximated by Russian vowel letters, which is exact enough
for duration regression: every Russian syllable has exactly one vowel.
"""

from __future__ import annotations

import re
from typing import Any

VOWELS = frozenset("аеёиоуыэюя")
_STRESS_MARKS = ("+", "́")
_PAUSE_RE = re.compile(r"\.{3}|…|[,.;:!?]+|\s[-—–]\s")
_WORD_RE = re.compile(r"[^\W\d_]+|\d+(?:[.,]\d+)*")


def join_lines(text: Any) -> str:
    """Return subtitle text as one line; list-form text is joined by spaces."""
    if isinstance(text, list):
        return " ".join(str(part) for part in text).strip()
    if text is None:
        return ""
    return str(text).strip()


def plain(text: str) -> str:
    """Remove stress marks so that measures do not depend on accentuation."""
    for mark in _STRESS_MARKS:
        text = text.replace(mark, "")
    return text


def syllables(text: str) -> int:
    """Count syllables as Russian vowel letters, case-insensitive."""
    return sum(1 for character in plain(text).lower() if character in VOWELS)


def words(text: str) -> int:
    """Count word-like tokens (letters or numbers)."""
    return len(_WORD_RE.findall(plain(text)))


def pauses(text: str) -> int:
    """Count punctuation groups that a TTS engine typically renders as a pause."""
    return len(_PAUSE_RE.findall(plain(text)))


FEATURE_NAMES: tuple[str, ...] = ("intercept", "syllables", "pauses")


def features(text: str) -> dict[str, float]:
    """Return the oracle's regression features for one text."""
    return {
        "intercept": 1.0,
        "syllables": float(syllables(text)),
        "pauses": float(pauses(text)),
    }
