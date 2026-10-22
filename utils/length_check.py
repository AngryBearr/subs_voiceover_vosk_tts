"""Pure subtitle-length validation shared by transports and MCP."""

from __future__ import annotations

from typing import Literal

from utils.vosk_text import normalize_vosk_text


LengthBasis = Literal["display_text", "normalized_tts_text"]


def prepare_length_text(text: str, length_basis: LengthBasis = "display_text") -> str:
    """Return text in the representation used for character-budget checks."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if length_basis == "display_text":
        return text
    if length_basis == "normalized_tts_text":
        return normalize_vosk_text(text)
    raise ValueError("unknown length_basis")


def check_length(
    text: str, max_chars: int, length_basis: LengthBasis = "display_text"
) -> dict[str, int | bool]:
    """Return code-point length information in the selected representation."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError("max_chars must be an integer")
    if max_chars < 0:
        raise ValueError("max_chars must be nonnegative")
    length = len(prepare_length_text(text, length_basis))
    return {
        "length": length,
        "max_chars": max_chars,
        "over_by": max(0, length - max_chars),
        "fits": length <= max_chars,
    }
