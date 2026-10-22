"""Rule-based text expansion for TTS with a fail-closed guard.

Numbers, abbreviations, symbols and Latin words are expanded by ``rutextnorm``
(pure rules, deterministic, no model).  The guard makes sure that ordinary
Russian words survive untouched: every Cyrillic word of the input that is not
an abbreviation candidate (followed by a dot) and not a unit after a number
must reappear, in order, in the output.  If the guard fails, the caller keeps
the original text, so no expansion step can ever invent words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rutextnorm import flag_uncertain, normalize_russian

_TOKEN_RE = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*|[^\w\s]")
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]+(?:[-'’][А-Яа-яЁё]+)*")
_NUMBER_RE = re.compile(r"\d")
_GLUED_ELLIPSIS_RE = re.compile(r"(\.{3}|…)(?=[^\W\d_])")


@dataclass(frozen=True)
class Expanded:
    text: str
    changed: bool
    uncertain: tuple[str, ...]  # human-readable notes for spans the rules were unsure about


def _fold(word: str) -> str:
    return word.casefold().replace("ё", "е")


def _must_keep(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text)
    keep: list[str] = []
    for position, token in enumerate(tokens):
        if not _CYRILLIC_RE.fullmatch(token):
            continue
        followed_by_dot = position + 1 < len(tokens) and tokens[position + 1] == "."
        after_number = position > 0 and bool(_NUMBER_RE.search(tokens[position - 1]))
        if not followed_by_dot and not after_number:
            keep.append(_fold(token))
    return keep


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    index = 0
    for item in haystack:
        if index < len(needle) and item == needle[index]:
            index += 1
    return index == len(needle)


def guard(before: str, after: str) -> None:
    """Raise ``ValueError`` when expansion changed a plain Russian word."""
    if not isinstance(after, str):
        raise ValueError("expansion returned non-text")
    expected = _must_keep(before)
    actual = [_fold(token) for token in _TOKEN_RE.findall(after) if _CYRILLIC_RE.fullmatch(token)]
    if not _is_subsequence(expected, actual):
        raise ValueError(f"expansion changed plain words: {expected!r} -> {actual!r}")


def space_after_ellipsis(text: str) -> str:
    """``хотят…хотят`` -> ``хотят… хотят``: the accentizer otherwise glues the words."""
    return _GLUED_ELLIPSIS_RE.sub(r"\1 ", text)


def expand(text: str) -> Expanded:
    """Expand digits/abbreviations/Latin; raises ``ValueError`` if a plain word changed."""
    result = normalize_russian(text)
    guard(text, result)
    notes = tuple(f"{original!r}: {reason}" for _start, _end, original, reason in flag_uncertain(text))
    return Expanded(result, result != text, notes)
