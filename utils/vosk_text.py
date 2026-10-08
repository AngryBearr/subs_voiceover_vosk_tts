"""Text normalization used only for diagnostic Vosk synthesis."""

from __future__ import annotations


_TRANSLATION = str.maketrans(
    {
        "«": '"',
        "»": '"',
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‚": '"',
        "‘": '"',
        "’": '"',
        "'": '"',
        "—": "-",
        "–": "-",
        "‒": "-",
        "−": "-",
        "…": "...",
    }
)


def normalize_vosk_text(text: str) -> str:
    """Map unsupported punctuation to the model's ASCII punctuation set."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(text.translate(_TRANSLATION).split())
