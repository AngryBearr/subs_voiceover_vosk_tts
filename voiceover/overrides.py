"""Manual overrides: decisions the user takes after reading ``export/human_queue.md``.

File format (JSON)::

    {"schema": "voiceover-overrides-v1",
     "units": {"unit-138": {"text": "Но она думает иначе.", "note": "optional"},
               "unit-45":  {"accept": true, "note": "optional"}}}

``text`` is spoken for that unit instead of anything the pipeline chose; ``accept`` keeps the pipeline's
choice but marks the unit as reviewed.  An entry holds one of the two, never both.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

OVERRIDES_SCHEMA = "voiceover-overrides-v1"
_ENTRY_KEYS = {"text", "accept", "note"}


@dataclass(frozen=True)
class Overrides:
    """Validated overrides; ``texts`` maps unit id to the text to speak, ``accepted`` lists reviewed units."""

    texts: dict[str, str] = field(default_factory=dict)
    accepted: frozenset[str] = frozenset()
    notes: dict[str, str] = field(default_factory=dict)
    source: Path | None = None

    @classmethod
    def empty(cls) -> "Overrides":
        return cls()

    @property
    def is_empty(self) -> bool:
        return not self.texts and not self.accepted

    def canonical(self) -> dict[str, Any]:
        """Content that changes the result (notes are free text and left out), for hashing."""
        return {"schema": OVERRIDES_SCHEMA, "text": dict(sorted(self.texts.items())),
                "accept": sorted(self.accepted)}

    def check_units(self, unit_ids: Iterable[str]) -> None:
        """Fail with the list of ids that do not exist in this episode."""
        known = set(unit_ids)
        unknown = sorted((set(self.texts) | self.accepted) - known)
        if unknown:
            raise ValueError(f"overrides name unknown unit ids: {', '.join(unknown)}")


def _parse_entry(unit_id: str, entry: Any) -> tuple[str | None, bool, str | None]:
    if not isinstance(entry, dict):
        raise ValueError(f"overrides: entry for {unit_id} must be an object")
    extra = sorted(set(entry) - _ENTRY_KEYS)
    if extra:
        raise ValueError(f"overrides: entry for {unit_id} has unknown keys: {', '.join(extra)}")
    if "text" in entry and "accept" in entry:
        raise ValueError(f"overrides: entry for {unit_id} has both 'text' and 'accept'; use one of them")
    note = entry.get("note")
    if note is not None and not isinstance(note, str):
        raise ValueError(f"overrides: 'note' of {unit_id} must be a string")
    if "text" in entry:
        text = entry["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"overrides: 'text' of {unit_id} must be a non-empty string")
        return text, False, note
    if entry.get("accept") is not True:
        raise ValueError(f"overrides: entry for {unit_id} needs either 'text' or 'accept': true")
    return None, True, note


def load_overrides(path: Path | None) -> Overrides:
    """Read and validate an overrides file; ``None`` means no overrides."""
    if path is None:
        return Overrides.empty()
    if not path.is_file():
        raise ValueError(f"overrides file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"overrides file {path} is not valid JSON: {error}") from error
    if not isinstance(raw, dict) or raw.get("schema") != OVERRIDES_SCHEMA:
        raise ValueError(f"overrides file {path} must be an object with \"schema\": \"{OVERRIDES_SCHEMA}\"")
    units = raw.get("units")
    if not isinstance(units, dict):
        raise ValueError("overrides: 'units' must be an object keyed by unit id")
    extra = sorted(set(raw) - {"schema", "units"})
    if extra:
        raise ValueError(f"overrides: unknown top-level keys: {', '.join(extra)}")
    texts: dict[str, str] = {}
    accepted: set[str] = set()
    notes: dict[str, str] = {}
    for unit_id, entry in units.items():
        text, accept, note = _parse_entry(unit_id, entry)
        if text is not None:
            texts[unit_id] = text
        if accept:
            accepted.add(unit_id)
        if note:
            notes[unit_id] = note
    return Overrides(texts, frozenset(accepted), notes, path)
