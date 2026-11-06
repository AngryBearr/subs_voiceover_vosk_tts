"""Punctuation cleanup of the text that reaches every TTS engine (Vosk, TeraTTS, ESpeech).

ESpeech (F5-TTS) says a stray "ж" / "з" / "с" sound before a word that follows a straight quote or an opening
bracket ("или ж бегут"), and an audiobook test showed quotes (guillemets included) add nothing the engines need
(GigaAM check: no quotes and « » were both clean, an ASCII ``"`` was not in 4 of 6 takes). So every quote mark
and apostrophe is removed and brackets become commas. Only the synthesis text goes through here; the display text
and the subtitles keep their quotes.
"""

from __future__ import annotations

import re


_TRANSLATION = str.maketrans(
    {
        "«": None,
        "»": None,
        "“": None,
        "”": None,
        "„": None,
        "‟": None,
        "‚": None,
        "‘": None,
        "’": None,
        '"': None,
        "'": None,
        "—": "-",
        "–": "-",
        "‒": "-",
        "−": "-",
        "…": "...",
    }
)

_MARK = "\x00"  # a comma that a bracket asked for; dropped when punctuation is already there
_OPEN_RE = re.compile(r"\s*[(\[{]\s*")
_CLOSE_RE = re.compile(r"\s*[)\]}]\s*")
_MARKS_RE = re.compile(r"(?:\s*" + _MARK + r"\s*)+")
_BEFORE_PUNCT = ",.!?:;-"
_AFTER_PUNCT = ",.!?:;"


def _brackets_to_commas(text: str) -> str:
    """``a (b) c`` -> ``a, b, c``; no comma at the text start / end or next to punctuation."""
    text = _CLOSE_RE.sub(f" {_MARK} ", _OPEN_RE.sub(f" {_MARK} ", text))
    if _MARK not in text:
        return text

    def resolve(match: re.Match[str]) -> str:
        before = match.string[:match.start()].rstrip()
        after = match.string[match.end():].lstrip()
        if not before or not after or before[-1] in _BEFORE_PUNCT or after[0] in _AFTER_PUNCT:
            return " " if before and after and after[0] not in _AFTER_PUNCT else ""
        return ", "

    return _MARKS_RE.sub(resolve, text)


def normalize_vosk_text(text: str) -> str:
    """Remove quote marks, turn brackets into commas, unify dashes and collapse whitespace."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(_brackets_to_commas(text.translate(_TRANSLATION)).split())
