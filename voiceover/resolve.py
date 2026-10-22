"""Measured re-solve: pick candidate rungs by their real duration and re-run the timeline.

Pure logic, no file I/O.  Manual override texts (``rung == "manual"``) replace the candidate list of their
unit, are never verified and are never stepped in pass 2; ``accept`` overrides only mark the choice reviewed.
  A rung is *usable* when it passed the hard checks, its clip was
synthesised without error and the entailment check accepted it.  Lower rungs change the
line less, so they are preferred.  Pass 1 fits every unit against its schedule0 target;
pass 2 steps down only units that still miss their slot after the neighbours' changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from voiceover.compress import Annotation, RungResult, _is_lossy
from voiceover.measure import MANUAL_RUNG
from voiceover.oracle import Oracle
from voiceover.overrides import Overrides
from voiceover.placement import TEXT_DEAD_ZONE_SEC, solve_placements
from voiceover.timeline import TimelinePolicy


@dataclass
class Resolution:
    passes: list[dict[str, Any]] = field(default_factory=list)
    choices: list[dict[str, Any]] = field(default_factory=list)
    placements: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _Candidate:
    rung: int | str
    speech_sec: float
    row: dict[str, Any]
    data: dict[str, Any]


def _pick(candidates: list[_Candidate], target: float, slack: float) -> tuple[_Candidate, bool]:
    """First candidate that fits the target, else the shortest one."""
    for candidate in candidates:
        if candidate.speech_sec <= target + slack:
            return candidate, True
    return min(candidates, key=lambda c: (c.speech_sec, c.rung)), False


def _status_counts(placements: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for placement in placements:
        counts[placement["status"]] = counts.get(placement["status"], 0) + 1
    return counts


def _reject_reason(measured_ok: set[int], by_checks: list[int]) -> str:
    if not by_checks:
        return "no_candidates"
    if not any(number in measured_ok for number in by_checks):
        return "synthesis"
    return "entailment"


def resolve(units: list[dict[str, Any]], rows0: list[dict[str, Any]], rows1: list[dict[str, Any]],
            verdicts: list[dict[str, Any]], ledger: list[dict[str, Any]], placements0: list[dict[str, Any]],
            policy: TimelinePolicy, oracle: Oracle, cues: list[dict[str, Any]], fit_slack_sec: float,
            overrides: Overrides | None = None) -> Resolution:
    """Choose rungs by measurement and solve the timeline (at most twice)."""
    overrides = overrides or Overrides.empty()
    cues_by_id = {cue["cue_id"]: cue for cue in cues}
    units_by_id = {unit["unit_id"]: unit for unit in units}
    rows0_by_id = {row["unit_id"]: row for row in rows0}
    rows1_by_key = {(row["unit_id"], row["rung"]): row for row in rows1}
    verdict_by_key = {(v["unit_id"], v["rung"]): v for v in verdicts}
    p0_by_id = {p["unit_id"]: p for p in placements0}

    usable: dict[str, list[_Candidate]] = {}
    entries: dict[str, dict[str, Any]] = {}
    choices: dict[str, dict[str, Any]] = {}
    chosen: dict[str, _Candidate] = {}
    for entry in ledger:
        uid = entry["unit_id"]
        entries[uid] = entry
        row0 = rows0_by_id.get(uid)
        p0 = p0_by_id.get(uid)
        target = float(p0["target_base_sec"]) if p0 else float(entry["target_base_sec"])
        by_checks = [r["rung"] for r in entry["rungs"] if not r["checks"]]
        measured_ok = {n for n in by_checks if (uid, n) in rows1_by_key and not rows1_by_key[(uid, n)].get("error")}
        candidates = [_Candidate(r["rung"], float(rows1_by_key[(uid, r["rung"])]["speech_sec"]),
                                 rows1_by_key[(uid, r["rung"])], r)
                      for r in sorted(entry["rungs"], key=lambda r: r["rung"])
                      if r["rung"] in measured_ok and verdict_by_key.get((uid, r["rung"]), {}).get("ok")]
        usable[uid] = candidates
        tried = [{"rung": r["rung"],
                  "speech_sec": rows1_by_key[(uid, r["rung"])]["speech_sec"] if r["rung"] in measured_ok else None,
                  "fits": r["rung"] in measured_ok
                  and float(rows1_by_key[(uid, r["rung"])]["speech_sec"]) <= target + fit_slack_sec,
                  "verify_ok": bool(verdict_by_key.get((uid, r["rung"]), {}).get("ok"))}
                 for r in entry["rungs"] if not r["checks"]]
        original = units_by_id[uid]["text"] if uid in units_by_id else entry["original"]
        choice: dict[str, Any] = {
            "unit_id": uid, "cue_ids": list(entry["cue_ids"]), "status": entry["status"], "chosen_rung": None,
            "chosen_text": original, "original_text": original,
            "original_speech_sec": float(row0["speech_sec"]) if row0 else float(entry["speech_sec"]),
            "chosen_speech_sec": float(row0["speech_sec"]) if row0 else float(entry["speech_sec"]),
            "target_base_sec_pass1": target, "lossy": False, "pass": None, "reject_reason": None,
            "rungs_tried": tried, "reviewed": uid in overrides.accepted}
        if uid not in overrides.texts and entry["status"] != "timing_limited":
            if candidates and row0 is not None:
                pick, fits = _pick(candidates, target, fit_slack_sec)
                chosen[uid] = pick
                choice["status"] = "resolved" if fits else "unresolved"
            else:
                choice.update(status="rejected", reject_reason=_reject_reason(measured_ok, by_checks))
        choices[uid] = choice

    manual = _apply_manual_texts(overrides, units_by_id, rows0_by_id, rows1_by_key, p0_by_id, choices, chosen, usable)

    def apply(uid: str, candidate: _Candidate, pass_number: int) -> None:
        entry = entries[uid]
        annotation = Annotation.from_dict(uid, entry["annotation"]) if entry.get("annotation") else None
        choice = choices[uid]
        choice.update(chosen_rung=candidate.rung, chosen_text=candidate.data["text"],
                      chosen_speech_sec=candidate.speech_sec, lossy=_is_lossy(annotation, RungResult(**candidate.data)),
                      **{"pass": pass_number})

    def solve() -> tuple[list[dict[str, Any]], float]:
        rows: list[dict[str, Any]] = []
        for row in rows0:
            pick = chosen.get(row["unit_id"])
            if pick is not None:
                row = {**row, "speech_sec": pick.speech_sec, "tts_text": pick.row["tts_text"],
                       "display_text": pick.data["text"]}
            rows.append(row)
        placements, _, solve_sec = solve_placements(rows, policy, oracle, cues_by_id)
        return placements, solve_sec

    for uid, candidate in chosen.items():
        if uid not in manual:
            apply(uid, candidate, 1)
    resolution = Resolution()
    placements, solve_sec = solve()
    resolution.passes.append(_pass_record(1, len(chosen), placements, solve_sec))

    stepped = 0
    for placement in placements:
        uid = placement["unit_id"]
        if uid not in chosen or uid in manual or placement["removed_sec"] <= TEXT_DEAD_ZONE_SEC:
            continue
        current = chosen[uid]
        later = [c for c in usable[uid] if c.rung > current.rung and c.speech_sec < current.speech_sec]
        if not later:
            continue
        pick, fits = _pick(later, float(placement["target_base_sec"]), fit_slack_sec)
        chosen[uid] = pick
        apply(uid, pick, 2)
        choices[uid]["status"] = "resolved" if fits else "unresolved"
        stepped += 1
    if stepped:
        placements, solve_sec = solve()
        resolution.passes.append(_pass_record(2, stepped, placements, solve_sec))

    for placement in placements:
        uid = placement["unit_id"]
        placement["text_changed"] = uid in chosen
        placement["chosen_rung"] = chosen[uid].rung if uid in chosen else None
    resolution.placements = placements
    resolution.choices = list(choices.values())
    return resolution


def _apply_manual_texts(overrides: Overrides, units_by_id: dict[str, dict[str, Any]],
                        rows0_by_id: dict[str, dict[str, Any]], rows1_by_key: dict[tuple[str, Any], dict[str, Any]],
                        p0_by_id: dict[str, dict[str, Any]], choices: dict[str, dict[str, Any]],
                        chosen: dict[str, _Candidate], usable: dict[str, list[_Candidate]]) -> set[str]:
    """Make the measured manual text the only candidate of its unit; returns the manual unit ids."""
    for uid, text in sorted(overrides.texts.items()):
        row1 = rows1_by_key.get((uid, MANUAL_RUNG))
        row0 = rows0_by_id.get(uid)
        if row1 is None or row1.get("error") or float(row1["speech_sec"]) <= 0 or row0 is None:
            raise ValueError(f"manual text of {uid} could not be measured: "
                             f"{(row1 or {}).get('error') or 'no clip for the unit or the manual text'}")
        speech = float(row1["speech_sec"])
        if uid not in choices:  # a unit the compressor never looked at
            unit = units_by_id[uid]
            p0 = p0_by_id.get(uid)
            choices[uid] = {
                "unit_id": uid, "cue_ids": list(unit["cue_ids"]), "status": "resolved", "chosen_rung": None,
                "chosen_text": unit["text"], "original_text": unit["text"],
                "original_speech_sec": float(row0["speech_sec"]), "chosen_speech_sec": float(row0["speech_sec"]),
                "target_base_sec_pass1": float(p0["target_base_sec"]) if p0 else float(row0["speech_sec"]),
                "lossy": False, "pass": None, "reject_reason": None, "rungs_tried": [],
                "reviewed": uid in overrides.accepted}
        choices[uid].update(status="manual", chosen_rung=MANUAL_RUNG, chosen_text=text, chosen_speech_sec=speech,
                            lossy=False, reject_reason=None, rungs_tried=[], **{"pass": 1})
        candidate = _Candidate(MANUAL_RUNG, speech, row1, {"text": text})
        chosen[uid] = candidate
        usable[uid] = [candidate]
    return set(overrides.texts)


def _pass_record(number: int, changed: int, placements: list[dict[str, Any]], solve_sec: float) -> dict[str, Any]:
    return {"pass": number, "changed_units": changed, "status_counts": _status_counts(placements),
            "needs_text": sum(1 for p in placements if p["removed_sec"] > TEXT_DEAD_ZONE_SEC),
            "solve_sec": round(solve_sec, 4)}
