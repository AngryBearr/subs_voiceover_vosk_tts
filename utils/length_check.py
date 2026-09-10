"""Pure subtitle-length validation shared by transports and MCP."""

from __future__ import annotations


def check_length(text: str, max_chars: int) -> dict[str, int | bool]:
    """Return code-point length information without normalising the text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError("max_chars must be an integer")
    if max_chars < 0:
        raise ValueError("max_chars must be nonnegative")
    length = len(text)
    return {
        "length": length,
        "max_chars": max_chars,
        "over_by": max(0, length - max_chars),
        "fits": length <= max_chars,
    }
