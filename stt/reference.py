"""Reference transcripts from subtitle files (SRT) for ASR evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.srt_to_json import parse_srt_file

# Non-speech annotations typical for SDH / closed captions.
_TAGS = re.compile(r"<[^>]+>|\{[^}]*\}")
_BRACKETS = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_LYRICS = re.compile(r"[♪♫#][^♪♫#]*(?:[♪♫#]|$)")
_SPEAKER_LABEL = re.compile(r"^\s*(?:>>\s*)?[A-ZА-ЯЁ][A-ZА-ЯЁ0-9 .'&-]{0,30}:\s+")
_DIALOGUE_DASH = re.compile(r"^\s*(?:>>|[-–—])\s*")
_SPACES = re.compile(r"\s+")


def clean_caption_line(line: str) -> str:
    """Strip formatting, sound descriptions, lyrics and speaker labels from a caption line."""
    text = _TAGS.sub(" ", line)
    text = _LYRICS.sub(" ", text)
    text = _BRACKETS.sub(" ", text)
    text = _DIALOGUE_DASH.sub("", text)
    text = _SPEAKER_LABEL.sub("", text)
    return _SPACES.sub(" ", text).strip()


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def load_cues(srt_path: str | Path) -> list[Cue]:
    """Parse an SRT into cleaned cues (seconds), dropping cues with no spoken text."""
    cues: list[Cue] = []
    for item in parse_srt_file(str(srt_path)):
        lines = [clean_caption_line(line) for line in item["text"]]
        text = " ".join(line for line in lines if line)
        if text:
            cues.append(Cue(item["start"] / 1000.0, item["end"] / 1000.0, text))
    cues.sort(key=lambda c: c.start)
    return cues


_ASS_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[.,](\d{1,3})")


def _ass_seconds(value: str) -> float:
    h, m, s, frac = _ASS_TIME.fullmatch(value.strip()).groups()  # type: ignore[union-attr]
    return int(h) * 3600 + int(m) * 60 + int(s) + int(frac.ljust(3, "0")[:3]) / 1000


def load_ass_cues(ass_path: str | Path, *, clean: bool = False) -> list[Cue]:
    """Dialogue lines of an ASS/SSA file as cues (override tags removed, \\N -> space)."""
    cues: list[Cue] = []
    fields: list[str] = []
    in_events = False
    for line in Path(ass_path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if line.startswith("["):
            in_events = line.strip().lower() == "[events]"
            continue
        if not in_events:
            continue
        if line.startswith("Format:"):
            fields = [f.strip().lower() for f in line[len("Format:"):].split(",")]
            continue
        if not line.startswith("Dialogue:") or not fields:
            continue
        values = line[len("Dialogue:"):].split(",", len(fields) - 1)
        row = dict(zip(fields, (v.strip() for v in values)))
        text = re.sub(r"\\[Nnh]", " ", _TAGS.sub("", row.get("text", "")))
        text = clean_caption_line(text) if clean else _SPACES.sub(" ", text).strip()
        if text:
            cues.append(Cue(_ass_seconds(row["start"]), _ass_seconds(row["end"]), text))
    cues.sort(key=lambda c: c.start)
    return cues


def snap_window(cues: list[Cue], start: float, duration: float | None, pad: float = 0.3) -> tuple[float, float]:
    """Move a requested window to subtitle gaps so no cue is cut in half.

    The window starts just before the first cue beginning at/after `start` and ends just
    after the last cue finishing before `start + duration` (padding stays inside gaps).
    """
    if not cues:
        raise ValueError("Subtitle file has no spoken cues")
    end_limit = cues[-1].end + pad if duration is None else start + duration
    inside = [i for i, c in enumerate(cues) if c.start >= start and c.end <= end_limit]
    if not inside:
        raise ValueError(f"No complete subtitle cue in {start:.1f}-{end_limit:.1f}s")
    first, last = inside[0], inside[-1]
    prev_end = cues[first - 1].end if first > 0 else 0.0
    next_start = cues[last + 1].start if last + 1 < len(cues) else cues[last].end + pad
    win_start = max(prev_end, cues[first].start - pad, 0.0)
    win_end = min(next_start, cues[last].end + pad)
    return round(win_start, 3), round(win_end, 3)


def window_reference(cues: list[Cue], start: float, end: float) -> dict[str, Any]:
    """Reference text and cues for cues whose midpoint lies inside [start, end)."""
    selected = [c for c in cues if start <= (c.start + c.end) / 2 < end]
    return {
        "start": start,
        "end": end,
        "text": " ".join(c.text for c in selected),
        "cues": [{"start": c.start, "end": c.end, "text": c.text} for c in selected],
    }
