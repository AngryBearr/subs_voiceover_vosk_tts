"""Speech units: one sentence split over several subtitle cues is spoken once.

A unit is a run of consecutive cues whose texts continue each other (comma,
dash, ellipsis, or a sentence that simply has no ending yet).  The unit is
synthesized as joined text and placed at the first cue's start.  Scene ids
group units separated by long silences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from voiceover.text import syllables

_CLOSING_QUOTES = "»\"'”"
_OPENING_QUOTES = "«\"'“"
_DASHES = "-—–"
_DIALOGUE_RE = re.compile(r"(^|\s)[-—–]\s*[А-ЯЁA-Z]")
_ELLIPSIS_JOIN_RE = re.compile(r"(\.\.\.|…)\s+(\.\.\.|…)")
_LOWER_RE = re.compile(r"[а-яёa-z]")


@dataclass(frozen=True)
class MergePolicy:
    """Limits for merging continuation cues into one unit."""

    enabled: bool = True
    max_gap_ms: int = 300
    max_cues: int = 3
    max_span_ms: int = 8000
    scene_gap_ms: int = 1500

    def __post_init__(self) -> None:
        for name in ("max_gap_ms", "max_cues", "max_span_ms", "scene_gap_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_cues < 1:
            raise ValueError("max_cues must be >= 1")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MergePolicy":
        known = {field: data[field] for field in cls.__dataclass_fields__ if field in data}
        return cls(**known)


@dataclass(frozen=True)
class Unit:
    """One speech unit made of one or more consecutive cues."""

    unit_id: str
    cue_ids: tuple[str, ...]
    position: int
    index: int
    start: int
    end: int
    text: str
    scene_id: str
    merged: bool
    reasons: tuple[str, ...]
    voice: int | str | None = None  # per-cue TTS voice from the input (None: engine default)

    def to_dict(self) -> dict[str, Any]:
        data = {"unit_id": self.unit_id, "cue_ids": list(self.cue_ids), "position": self.position,
                "index": self.index, "start": self.start, "end": self.end, "text": self.text,
                "scene_id": self.scene_id, "merged": self.merged, "reasons": list(self.reasons)}
        if self.voice is not None:
            data["voice"] = self.voice
        return data


def is_dialogue(text: str) -> bool:
    """True when the cue starts with a dash or holds a second speaker's line."""
    stripped = text.strip()
    if stripped and stripped[0] in _DASHES:
        return True
    return _DIALOGUE_RE.search(stripped) is not None


def _ends_with_ellipsis(text: str) -> bool:
    return text.endswith("...") or text.endswith("…")


def continuation_reason(prev_text: str, next_text: str) -> str | None:
    """Why ``next_text`` continues ``prev_text``, or None when it starts a new sentence."""
    prev = prev_text.strip().rstrip(_CLOSING_QUOTES + " \t").rstrip()
    nxt = next_text.strip().lstrip(_OPENING_QUOTES + " \t").lstrip()
    if not prev or not nxt:
        return None
    next_lower = _LOWER_RE.match(nxt[0]) is not None
    next_ellipsis = nxt.startswith("...") or nxt.startswith("…")
    if _ends_with_ellipsis(prev):
        return "ellipsis" if next_lower or next_ellipsis else None
    last = prev[-1]
    if last in ".!?":
        return None
    if last in ",;:":
        return "comma"
    if last in _DASHES:
        return "dash"
    if last.isalnum():
        return "dangling" if next_lower else None
    return None


def join_texts(parts: list[str]) -> str:
    """Join cue texts with single spaces; ``... ...`` at a seam becomes one ellipsis."""
    joined = re.sub(r"\s+", " ", " ".join(parts)).strip()
    return _ELLIPSIS_JOIN_RE.sub("...", joined)


def scene_ids(cues: list[dict[str, Any]], scene_gap_ms: int) -> list[str]:
    """Scene id per cue; a silence of at least ``scene_gap_ms`` starts a new scene."""
    result: list[str] = []
    scene = 0
    for i, cue in enumerate(cues):
        if i > 0 and cue["start"] - cues[i - 1]["end"] >= scene_gap_ms:
            scene += 1
        result.append(f"scene-{scene}")
    return result


def build_units(cues: list[dict[str, Any]], policy: MergePolicy) -> list[Unit]:
    """Group consecutive continuation cues into units."""
    positions = [cue["position"] for cue in cues]
    assert all(a < b for a, b in zip(positions, positions[1:])), "cue positions must be increasing"
    scenes = scene_ids(cues, policy.scene_gap_ms)
    units: list[Unit] = []
    group: list[int] = []
    reasons: list[str] = []

    def close() -> None:
        members = [cues[i] for i in group]
        units.append(Unit(
            unit_id=f"unit-{members[0]['position']}", cue_ids=tuple(m["cue_id"] for m in members),
            position=members[0]["position"], index=members[0]["index"], start=members[0]["start"],
            end=members[-1]["end"], text=join_texts([m["text"] for m in members]), scene_id=scenes[group[0]],
            merged=len(members) > 1, reasons=tuple(reasons), voice=members[0].get("voice")))

    for i, cue in enumerate(cues):
        if group:
            last = cues[group[-1]]
            reason = continuation_reason(last["text"], cue["text"]) if policy.enabled else None
            if (reason is not None
                    and cue["start"] - last["end"] <= policy.max_gap_ms
                    and not is_dialogue(last["text"]) and not is_dialogue(cue["text"])
                    and cue.get("voice") == last.get("voice")
                    and len(group) + 1 <= policy.max_cues
                    and cue["end"] - cues[group[0]]["start"] <= policy.max_span_ms
                    and scenes[i] == scenes[group[0]]):
                group.append(i)
                reasons.append(reason)
                continue
            close()
        group, reasons = [i], []
    if group:
        close()
    return units


def member_timing(unit_cues: list[dict[str, Any]], placement_start_ms: int, spoken_sec: float) -> list[dict[str, Any]]:
    """Estimate when each member's text begins inside the spoken unit (syllable-proportional)."""
    counts = [syllables(cue["text"]) for cue in unit_cues]
    total = sum(counts)
    rows: list[dict[str, Any]] = []
    before = 0
    for i, cue in enumerate(unit_cues):
        fraction = before / total if total > 0 else i / len(unit_cues)
        est = int(round(placement_start_ms + fraction * spoken_sec * 1000.0))
        rows.append({"cue_id": cue["cue_id"], "start": cue["start"], "est_start_ms": est,
                     "drift_ms": est - cue["start"]})
        before += counts[i]
    return rows
