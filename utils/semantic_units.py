"""Deterministic semantic-unit construction for subtitle cue sequences."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from numbers import Real
from typing import Any, Optional

from utils.analyze_text import join_text_lines


@dataclass(frozen=True)
class SemanticUnit:
    """A contiguous source-ordered unit and its changed target cues."""

    unit_id: str
    cue_indices: tuple[int, ...]
    changed_indices: tuple[int, ...]


_CONNECTORS = (
    "и", "а", "но", "или", "либо", "что", "чтобы", "который", "которая",
    "которое", "которые", "если", "когда", "где", "как", "потому что",
    "так как", "так что", "хотя", "ведь", "же", "то",
)
_CONNECTOR_RE = re.compile(r"^(?:" + "|".join(re.escape(c) for c in _CONNECTORS) + r")(?:\b|\s)", re.IGNORECASE)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _validate_records(records: Sequence[Mapping[str, Any]], label: str) -> tuple[list[Mapping[str, Any]], dict[int, Mapping[str, Any]]]:
    if not _is_sequence(records):
        raise ValueError(f"{label} must be a sequence")
    ordered = list(records)
    by_index: dict[int, Mapping[str, Any]] = {}
    for record in ordered:
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} must contain mappings")
        index = record.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"{label} indices must be non-bool integers")
        if index in by_index:
            raise ValueError(f"{label} indices must be unique")
        by_index[index] = record
    return ordered, by_index


def _text(record: Mapping[str, Any]) -> str:
    return join_text_lines(record.get("text", ""))


def _valid_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _normalised_text(value: str) -> str:
    return " ".join(value.split()).strip().lower().replace("ё", "е")


def _first_alpha(value: str) -> str | None:
    for character in value:
        if character.isalpha():
            return character
    return None


def _starts_with_connector(value: str) -> bool:
    trimmed = value.strip().lower().replace("ё", "е")
    return _CONNECTOR_RE.match(trimmed) is not None


def _continues(left: Mapping[str, Any], right: Mapping[str, Any], max_gap_sec: float) -> bool:
    left_end = left.get("end")
    right_start = right.get("start")
    if not _valid_number(left_end) or not _valid_number(right_start):
        return False
    if (float(right_start) - float(left_end)) / 1000.0 > max_gap_sec:
        return False
    left_text = _text(left).strip()
    right_text = _text(right)
    if not left_text or not right_text.strip():
        return False
    trailing = left_text.rstrip()
    if trailing.endswith("...") or trailing.endswith("…"):
        first = _first_alpha(right_text)
        return first is not None and (first.islower() or _starts_with_connector(right_text))
    if trailing[-1] in ".!?":
        return False
    if trailing[-1] in ",:;—–-":
        return True
    first = _first_alpha(right_text)
    return first is not None and (first.islower() or _starts_with_connector(right_text))


def _unit_id(cue_indices: tuple[int, ...], texts: list[str]) -> str:
    payload = {"cue_indices": cue_indices, "texts": [_normalised_text(text) for text in texts]}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return f"su-v1:{cue_indices[0]}:{cue_indices[-1]}:{digest}"


def build_semantic_units(
    original: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    context_items: Optional[Sequence[Mapping[str, Any]]] = None,
    target_indices: Optional[Sequence[int]] = None,
    max_cues: int = 3,
    max_gap_sec: float = 0.3,
) -> tuple[SemanticUnit, ...]:
    """Build non-overlapping source-ordered units around changed target cues."""
    original_order, original_by_index = _validate_records(original, "original")
    candidate_order, candidate_by_index = _validate_records(candidate, "candidate")
    if set(original_by_index) != set(candidate_by_index):
        raise ValueError("original and candidate index sets must match")
    if isinstance(max_cues, bool) or not isinstance(max_cues, int) or not 1 <= max_cues <= 5:
        raise ValueError("max_cues must be an integer from 1 through 5")
    if not _valid_number(max_gap_sec) or float(max_gap_sec) < 0:
        raise ValueError("max_gap_sec must be finite and non-negative")

    if context_items is None:
        source_order = original_order
        source_by_index = dict(original_by_index)
    else:
        source_order, source_by_index = _validate_records(context_items, "context_items")
        if not set(original_by_index).issubset(source_by_index):
            raise ValueError("context_items must contain every original index")
        for index, record in original_by_index.items():
            if _text(source_by_index[index]).strip() != _text(record).strip():
                raise ValueError("context text must match original text")

    source_indices = [int(record["index"]) for record in source_order]
    source_positions = {index: position for position, index in enumerate(source_indices)}
    changed = {index for index in original_by_index if _text(original_by_index[index]) != _text(candidate_by_index[index])}
    if target_indices is None:
        targets = set(changed)
    else:
        if not _is_sequence(target_indices):
            raise ValueError("target_indices must be a sequence")
        targets = set()
        for index in target_indices:
            if isinstance(index, bool) or not isinstance(index, int) or index in targets or index not in changed:
                raise ValueError("target_indices must be unique integer changed indices")
            targets.add(index)
    if not targets:
        return ()
    if not targets.issubset(source_positions):
        raise ValueError("every target must exist in the source order")

    edges = [_continues(source_order[position], source_order[position + 1], float(max_gap_sec)) for position in range(len(source_order) - 1)]
    target_positions = sorted(source_positions[index] for index in targets)
    occupied: set[int] = set()
    units: list[SemanticUnit] = []

    for start_position in target_positions:
        if start_position in occupied:
            continue
        core_end = start_position
        scan = start_position
        while scan + 1 < len(source_order) and edges[scan] and scan + 1 not in occupied:
            scan += 1
            if scan in target_positions:
                if scan - start_position + 1 <= max_cues:
                    core_end = scan
                else:
                    break
        members = list(range(start_position, core_end + 1))
        while len(members) < max_cues:
            added = False
            left = members[0] - 1
            if left >= 0 and left not in occupied and source_indices[left] not in targets and edges[left]:
                members.insert(0, left)
                added = True
            right = members[-1] + 1
            if len(members) < max_cues and right < len(source_order) and right not in occupied and source_indices[right] not in targets and edges[right - 1]:
                members.append(right)
                added = True
            if not added:
                break
        occupied.update(members)
        indices = tuple(source_indices[position] for position in members)
        texts = [_text(source_order[position]) for position in members]
        changed_indices = tuple(index for index in indices if index in targets)
        units.append(SemanticUnit(_unit_id(indices, texts), indices, changed_indices))

    projected = [index for unit in units for index in unit.changed_indices if index in targets]
    if sorted(projected) != sorted(targets) or len(projected) != len(set(projected)):
        raise AssertionError("each target must belong to exactly one semantic unit")
    return tuple(units)
