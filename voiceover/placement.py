"""Shared placement helpers: text targets and the enriched solver output.

Used by the first schedule (original texts) and by the measured re-solve (chosen rungs).
"""

from __future__ import annotations

import math
import time
from typing import Any

from voiceover.oracle import Oracle
from voiceover.text import syllables
from voiceover.timeline import Line, Placement, TimelinePolicy, schedule
from voiceover.units import member_timing

TIMING_LIMITED_MIN_SYLLABLES = 3
# Removals this small are solver granularity (10 ms grid, 0.02 tempo steps), not text work:
# the line is marked "tight" and overruns its slot by at most this much.
TEXT_DEAD_ZONE_SEC = 0.05


def text_band(required_reduction: float, target_syllables: int) -> str:
    """Classify how much text work a line needs; timing_limited lines skip the model."""
    if required_reduction <= 0.0:
        return "none"
    if target_syllables < TIMING_LIMITED_MIN_SYLLABLES or required_reduction > 0.6:
        return "timing_limited"
    if required_reduction <= 0.2:
        return "easy"
    if required_reduction <= 0.4:
        return "medium"
    return "hard"


def syllables_to_remove(current_syllables: int, speech_sec: float, removed_sec: float, oracle: Oracle) -> int:
    """Syllables that must go to save ``removed_sec`` of base-rate speech from this line.

    The line's own measurement gives its personal seconds per syllable (above the
    engine's fixed overhead); the oracle average is the fallback for lines too
    short to measure a rate.  Rounded up: removing too little leaves an overrun.
    """
    overhead = oracle.coefficients["intercept"]
    per_syllable = (speech_sec - overhead) / current_syllables if current_syllables else 0.0
    if per_syllable <= 0:
        per_syllable = oracle.coefficients["syllables"]
    return min(current_syllables, max(1, math.ceil(removed_sec / per_syllable - 1e-9)))


def text_target(row: dict[str, Any], placement: Placement, oracle: Oracle) -> dict[str, Any]:
    current = syllables(row["tts_text"])
    if placement.removed_sec <= TEXT_DEAD_ZONE_SEC or current == 0:
        return {"current_syllables": current, "target_syllables": current, "required_reduction": 0.0,
                "text_band": "none"}
    # removed_sec is measured at the chosen tempo; the line's rate is measured at base rate.
    removed_base_sec = placement.removed_sec * placement.tempo_factor
    target = current - syllables_to_remove(current, float(row["speech_sec"]), removed_base_sec, oracle)
    reduction = round(max(0.0, 1.0 - target / current), 3)
    return {"current_syllables": current, "target_syllables": target, "required_reduction": reduction,
            "text_band": text_band(reduction, target)}


def solve_placements(rows: list[dict[str, Any]], policy: TimelinePolicy, oracle: Oracle,
                     cues_by_id: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], float]:
    """Solve the timeline for duration-like rows and enrich each placement.

    A row needs ``unit_id, start, speech_sec, tts_text, cue_ids, scene_id, merged``.  Rows with a
    synthesis error or no speech are skipped.  Returns (placements, skipped unit ids, solve seconds).
    """
    usable = [row for row in rows if not row.get("error") and row["speech_sec"] > 0]
    lines = [Line(row["unit_id"], int(row["start"]), float(row["speech_sec"])) for row in usable]
    started = time.time()
    placements = schedule(lines, policy)
    solve_sec = time.time() - started
    by_id = {row["unit_id"]: row for row in usable}
    enriched = []
    for placement in placements:
        row = by_id[placement.cue_id]
        record = placement.to_dict()
        record["unit_id"] = record.pop("cue_id")
        record.update({"cue_ids": list(row["cue_ids"]), "scene_id": row["scene_id"], "merged": row["merged"]})
        if row["merged"]:
            record["members"] = member_timing([cues_by_id[c] for c in row["cue_ids"]], placement.start_ms,
                                              placement.spoken_sec)
        record.update(text_target(row, placement, oracle))
        if placement.status == "needs_text" and record["text_band"] == "none":
            record["status"] = "tight"
        enriched.append(record)
    skipped = [row["unit_id"] for row in rows if row.get("error") or row["speech_sec"] <= 0]
    return enriched, skipped, solve_sec
