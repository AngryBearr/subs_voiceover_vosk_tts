"""Shared data structures for STT results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Word:
    """A recognized word; times are seconds on the analyzed audio timeline."""

    start: float
    end: float
    text: str
    score: float | None = None
    speaker: str | None = None
    source: str | None = None  # None: ASR; "ocr": on-screen caption


@dataclass
class Segment:
    """A recognized text span, optionally with word timings and a speaker."""

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    speaker: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Segment:
        words = [Word(**w) for w in data.get("words", [])]
        return cls(
            start=data["start"],
            end=data["end"],
            text=data["text"],
            words=words,
            speaker=data.get("speaker"),
        )


def segments_text(segments: list[Segment]) -> str:
    """Join segment texts into one transcript string."""
    return " ".join(s.text.strip() for s in segments if s.text.strip())
