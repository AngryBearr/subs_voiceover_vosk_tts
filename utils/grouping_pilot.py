"""Conservative textual continuation scan and bounded local-only timing pilot."""

from __future__ import annotations

import argparse
import json
import math
import re
import wave
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from utils.alignment_pilot import (Cue, GAP_BOUNDARY_MS, MIN_PAUSE_MS, SHIFT_MS,
                                   _text_id, allocate, assemble, boundary_reasons, load_cues)
from utils.shortening_audio import PREPROCESSING, _sha256, _write_json, verify_audio
from utils.vosk_calibration import (max_chars_for_window, model_metadata, predict_character_duration,
                                    profile_identity, validate_character_profile)
from utils.vosk_text import normalize_vosk_text

PROBLEM_STATUSES = {"short_timing", "manual"}
MAX_SPAN_MS = 10000
SELECTION_RULE = (
    "Source punctuation only determines continuation eligibility; prefer complete 2-3-cue "
    "utterances. Rank eligible groups by unresolved-anchor count, manual-anchor count, "
    "cue count, then source position; greedily retain disjoint groups. Split those "
    "chronological groups into six equal-rank bands and take the highest-ranked group "
    "per band (or all if fewer than six). Never rank by timing prediction/measurement."
)


def core(text: str) -> str:
    """Inspect punctuation before closing quotes/brackets without changing display text."""
    return re.sub(r'[\s"\'»”’)}\]]+$', '', text)


def ending(text: str) -> str:
    tail = core(text)
    if tail.endswith(("...", "…")):
        return "ellipsis"
    if tail.endswith((".", "!", "?")):
        return "terminal"
    if tail.endswith((",", ";", ":")):
        return "continuation_punctuation"
    return "no_terminal"


def first_letter(text: str) -> str:
    return next((char for char in text if char.isalpha()), "")


def short_reply(text: str) -> bool:
    words = re.findall(r"[^\W\d_]+", text.casefold())
    return bool(words and len(words) <= 3 and words[0] in {"да", "нет", "ага", "ок", "окей"})


def link_reason(left: str, right: str) -> str:
    if ending(left) == "terminal":
        return "terminal_crossing"
    if ending(left) == "ellipsis":
        return "ellipsis_ambiguous"
    if not first_letter(right).islower():
        return "continuation_without_lowercase"
    return "punctuation_lowercase_continuation" if ending(left) == "continuation_punctuation" else "no_terminal_lowercase_continuation"


def connected(left: Cue, right: Cue) -> bool:
    return (0 <= right.start - left.end <= GAP_BOUNDARY_MS
            and not boundary_reasons([left, right], None, None)
            and not short_reply(left.original_text) and not short_reply(right.original_text)
            and link_reason(left.original_text, right.original_text) in {
                "ellipsis_ambiguous", "punctuation_lowercase_continuation", "no_terminal_lowercase_continuation"})


def candidate(cues: list[Cue], first: int, size: int) -> dict[str, Any]:
    group = cues[first:first + size]
    previous = cues[first - 1] if first else None
    following = cues[first + size] if first + size < len(cues) else None
    hard = boundary_reasons(group, previous.end if previous else None, following.start if following else None)
    links = [link_reason(left.original_text, right.original_text) for left, right in zip(group, group[1:])]
    if "terminal_crossing" in links:
        hard.append("terminal_crossing")
    if any(re.search(r'(?:[!?]|(?<!\.)\.(?!\.))["\'»”’)}\]]*\s+[«“"\']*[^\W\d_]', c.original_text) for c in group):
        hard.append("internal_terminal_crossing")
    if any(short_reply(cue.original_text) for cue in group):
        hard.append("short_reply_boundary")
    span = group[-1].end - group[0].start
    if span > MAX_SPAN_MS:
        hard.append("group_span_cap")
    uncertain = [reason for reason in links if reason in {"ellipsis_ambiguous", "continuation_without_lowercase"}]
    if any("..." in c.original_text or "…" in c.original_text for c in group):
        uncertain.append("ellipsis_ambiguous")
    if any("?" in cue.original_text for cue in group):
        uncertain.append("question_requires_turn_review")
    starts_upper = first_letter(group[0].original_text).isupper()
    before_closed = previous is None or ending(previous.original_text) == "terminal" or group[0].start - previous.end > GAP_BOUNDARY_MS
    if not starts_upper or not before_closed:
        uncertain.append("open_start_boundary")
    ends_closed = ending(group[-1].original_text) == "terminal"
    if not ends_closed:
        uncertain.append("open_end_boundary")
    # ponytail: punctuation chain only, not grammatical parsing or speaker detection.
    chain_first, chain_end = first, first + size
    while chain_first > 0 and connected(cues[chain_first - 1], cues[chain_first]):
        chain_first -= 1
    while chain_end < len(cues) and connected(cues[chain_end - 1], cues[chain_end]):
        chain_end += 1
    if chain_end - chain_first > 3:
        uncertain.append("utterance_exceeds_three_cues")
    source_state = "not_safe" if hard else "ambiguous_or_incomplete" if uncertain else "source_closed_heuristic"
    changed = [cue.cue_id for cue in group if cue.text != cue.original_text]
    current_conflicts = [left.cue_id for left, right in zip(group, group[1:])
                         if link_reason(left.text, right.text) != link_reason(left.original_text, right.original_text)]
    if ending(group[-1].text) != ending(group[-1].original_text):
        current_conflicts.append(group[-1].cue_id)
    if any(short_reply(cue.text) != short_reply(cue.original_text) for cue in group):
        current_conflicts.extend(changed)
    # Changed neighbors have never been reviewed as one utterance. Even compatible
    # punctuation is not a semantic approval; conflicting boundaries use originals.
    measurement_texts = [cue.original_text if current_conflicts else cue.text for cue in group]
    review_reasons = ["heuristic_not_grammar_guarantee"]
    if any(cue.speaker is None for cue in group):
        review_reasons.append("speaker_unknown")
    if changed:
        review_reasons.append("changed_neighbors_joint_context_unreviewed")
    if current_conflicts:
        review_reasons.append("current_text_boundary_conflict_use_originals")
    return {
        "group_id": f"utterance-{first}-{first + size - 1}", "positions": [cue.position for cue in group],
        "cue_ids": [cue.cue_id for cue in group], "source_state": source_state,
        "eligibility": "candidate_requires_review" if source_state == "source_closed_heuristic" else "not_selected_for_audio",
        "reason_codes": list(dict.fromkeys([*hard, *uncertain])), "continuation_evidence": links,
        "review_required": True, "review_reasons": review_reasons,
        "start_boundary_evidence": {"first_uppercase": starts_upper, "previous_closed_or_large_gap": before_closed},
        "end_boundary_closed": ends_closed, "original_joined_text": " ".join(c.original_text for c in group),
        "current_joined_text": " ".join(c.text for c in group), "changed_neighbor_ids": changed,
        "current_text_conflict_ids": list(dict.fromkeys(current_conflicts)),
        "measurement_texts": measurement_texts,
        "measurement_text_policy": "all_originals_due_to_boundary_conflict" if current_conflicts else "saved_current_texts",
        "source_start_ms": group[0].start, "source_end_ms": group[-1].end, "source_span_ms": span,
        "source_gaps_ms": [b.start - a.end for a, b in zip(group, group[1:])],
        "previous_end_ms": previous.end if previous else None, "next_start_ms": following.start if following else None,
        "unresolved_ids": [c.cue_id for c in group if c.status in PROBLEM_STATUSES],
        "manual_anchor_count": sum(c.status == "manual" for c in group),
        "cues": [asdict(cue) | {"text_id": _text_id(cue.text), "original_text_id": _text_id(cue.original_text)} for cue in group],
        "timing_evidence": "predicted_not_measured",
        "timing_classification": "predicted_not_measured" if source_state == "source_closed_heuristic" else "ambiguous_no_auto_audio_or_export",
    }


def rank(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return -len(row["unresolved_ids"]), -row["manual_anchor_count"], -len(row["cue_ids"]), row["positions"][0]


def scan(cues: list[Cue], profile: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    checked = 0
    for first in range(len(cues)):
        for size in (2, 3):
            if first + size > len(cues) or not any(c.status in PROBLEM_STATUSES for c in cues[first:first + size]):
                continue
            checked += 1
            row = candidate(cues, first, size)
            reasons.update(row["reason_codes"])
            available = max(0, row["source_span_ms"] - (size - 1) * MIN_PAUSE_MS)
            row["prediction"] = {
                "evidence": "predicted_not_measured", "profile": "existing_vosk_char_linear",
                "single_utterance_speech_window_ms": available, "reserved_min_pauses_ms": (size - 1) * MIN_PAUSE_MS,
                "max_chars_normalized_single_unit": max_chars_for_window(profile, available, expected_identity=identity),
                "original_joined_prediction_ms": math.ceil(1000 * predict_character_duration(profile, row["original_joined_text"], expected_identity=identity)),
                "current_joined_prediction_ms": math.ceil(1000 * predict_character_duration(profile, row["current_joined_text"], expected_identity=identity)),
                "separate_clips_prediction_ms": math.ceil(sum(1000 * predict_character_duration(profile, text, expected_identity=identity)
                                                            for text in row["measurement_texts"])),
                "single_unit_not_equivalent_to_separate_clip_pauses": True,
            }
            row["prediction"]["predicted_requirement"] = (
                "duration_deficit_under_separate_clip_policy"
                if row["prediction"]["separate_clips_prediction_ms"] + row["prediction"]["reserved_min_pauses_ms"] > row["source_span_ms"]
                else "duration_sum_with_minimum_pauses_within_span_not_a_fit")
            # Keep all bounded competitors, including rejects, not every larger subset.
            rows.append(row)
    used: set[int] = set()
    disjoint: list[dict[str, Any]] = []
    for row in sorted((r for r in rows if r["eligibility"] == "candidate_requires_review"), key=rank):
        conflicts = [r["group_id"] for r in disjoint if set(r["positions"]).intersection(row["positions"])]
        row["competing_disjoint_group_ids"] = conflicts
        row["disjoint_selection"] = "overlap_conflict" if used.intersection(row["positions"]) else "retained_for_review"
        if row["disjoint_selection"] == "retained_for_review":
            used.update(row["positions"])
            disjoint.append(row)
    disjoint.sort(key=lambda r: r["positions"][0])
    selected = disjoint if len(disjoint) <= 6 else [min(disjoint[len(disjoint) * band // 6:len(disjoint) * (band + 1) // 6], key=rank) for band in range(6)]
    unresolved = [c.cue_id for c in cues if c.status in PROBLEM_STATUSES]
    covered = {cue_id for row in disjoint for cue_id in row["unresolved_ids"]}
    return {"candidates": rows, "disjoint_group_ids": [r["group_id"] for r in disjoint],
            "pilot_group_ids": [r["group_id"] for r in selected], "summary": {
                "source_cues": len(cues), "unresolved_anchors": len(unresolved), "bounded_windows_scanned": checked,
                "source_closed_candidates": sum(r["source_state"] == "source_closed_heuristic" for r in rows),
                "ambiguous_or_incomplete_candidates": sum(r["source_state"] == "ambiguous_or_incomplete" for r in rows),
                "not_safe_windows": sum(r["source_state"] == "not_safe" for r in rows),
                "disjoint_candidates_for_review": len(disjoint), "covered_unresolved_anchors": len(covered),
                "unresolved_outside_disjoint_candidates": len(set(unresolved) - covered),
                "reason_counts": dict(sorted(reasons.items()))},
            "unresolved_outside_disjoint_candidate_ids": [cue_id for cue_id in unresolved if cue_id not in covered]}


def classify_timing(row: dict[str, Any], allocation: dict[str, Any], durations: list[int]) -> str:
    if row["eligibility"] != "candidate_requires_review":
        return "ambiguous_no_auto_audio_or_export"
    if allocation["proposal"] is not None:
        return "timing_only_feasible_inspect"
    if sum(durations) + MIN_PAUSE_MS * (len(durations) - 1) > row["source_span_ms"]:
        return "needs_group_shortening"
    return "timing_shift_conflict"


def frozen_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(p): _sha256(p) for root in paths for p in sorted(root.rglob("*")) if p.is_file()}


def run(source: Path, output: Path, model: Path, profile_path: Path, *, inspect: bool = False) -> dict[str, Any]:
    if not inspect and output.exists():
        raise FileExistsError(f"never overwrite a pilot: {output}")
    identity = profile_identity(model_metadata(model), voice=8, rate=1.5)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    validate_character_profile(profile, expected_identity=identity)
    cues = load_cues(source)
    result = scan(cues, profile, identity)
    if inspect:
        return result
    protected = frozen_hashes([source, Path("output/alignment_pilot_20261001")])
    result.update(schema_version=1, status="selected_before_synthesis", selection_rule=SELECTION_RULE,
                  protected_file_hashes=protected, source_dir=str(source), inspect_only=True,
                  settings={"model_path": str(model), "voice": 8, "rate": 1.5, "preprocessing": PREPROCESSING,
                            "identity": identity, "profile_path": str(profile_path), "profile_sha256": _sha256(profile_path)},
                  limits={"cue_count": 3, "span_ms": MAX_SPAN_MS, "source_gap_ms": GAP_BOUNDARY_MS,
                          "start_shift_ms": SHIFT_MS, "min_inter_cue_zero_silence_ms": MIN_PAUSE_MS},
                  llm_calls=0, network_calls=0, provider_cost_usd=0, listened=False,
                  pause_policy="Separate untrimmed clips, retain max(200ms, source gap) first; explicitly reduce to 200ms only if needed. No outside borrowing; no continuous joined TTS was synthesized.")
    source_items = json.loads((source / "review_state.json").read_text(encoding="utf-8"))["source_items"]
    for row in result["candidates"]:
        for provenance in row["cues"]:
            provenance["original_item"] = source_items[provenance["position"]]
    output.mkdir(parents=True)
    checkpoint = output / "candidates.json"
    _write_json(checkpoint, result)
    selected = [r for r in result["candidates"] if r["group_id"] in result["pilot_group_ids"]]
    measurement_cues = [replace(cues[position], text=text) for row in selected
                        for position, text in zip(row["positions"], row["measurement_texts"])]
    if len(measurement_cues) > 18:
        raise ValueError("bounded synthesis limit exceeded")
    measured = verify_audio({"records": [{"cue_id": cue.cue_id, "position": cue.position, "status": "short_timing",
                          "original": cue.text, "budget": {"start": cue.start, "end": cue.end, "available_ms": cue.available_ms}}
                         for cue in measurement_cues]}, output / "individual", model_path=model, voice=8, speech_rate=1.5, limit=18)
    audio_rows = {row["cue_id"]: row for row in measured["records"]}
    tasks: list[dict[str, Any]] = []
    (output / "groups").mkdir()
    for row in selected:
        group = [replace(cues[position], text=text) for position, text in zip(row["positions"], row["measurement_texts"])]
        audio = [audio_rows[cue.cue_id] for cue in group]
        if any(a.get("error") or not a.get("original") for a in audio):
            row.update(timing_evidence="measurement_failed", timing_classification="synthesis_error")
            _write_json(checkpoint, result)
            continue
        durations = [math.ceil(a["original"]["frames"] * 1000 / a["original"]["samplerate"]) for a in audio]
        allocation = allocate(group, durations, previous_end=row["previous_end_ms"], next_start=row["next_start_ms"])
        row.update(timing_evidence="real_local_tts_measured", duration_ms_ceil=durations,
                   audio=[a["original"] for a in audio], allocation=allocation,
                   minimum_required_ms=sum(durations) + MIN_PAUSE_MS * (len(group) - 1),
                   duration_deficit_ms=max(0, sum(durations) + MIN_PAUSE_MS * (len(group) - 1) - row["source_span_ms"]),
                   timing_classification=classify_timing(row, allocation, durations))
        row["baseline_individual_fits"] = [duration <= min(cue.available_ms, row["source_end_ms"] - cue.start)
                                            for cue, duration in zip(group, durations)]
        if allocation["proposal"] is not None:
            paths = [output / "individual" / a["original"]["path"] for a in audio]
            row["assembled_wav"] = assemble(paths, allocation["proposal"]["placements"],
                output / "groups" / f"{row['group_id']}_inspect_only.wav", row["source_start_ms"], row["source_end_ms"])
        if row["timing_classification"] == "needs_group_shortening":
            tasks.append({"group_id": row["group_id"], "status": "offline_input_only_no_request_sent",
                "mode": "future_single_utterance_unit_not_per_cue_rewrite", "original": row["original_joined_text"],
                "saved_current_joined_text": row["current_joined_text"], "cue_provenance": row["cues"],
                "source_start_ms": row["source_start_ms"], "source_end_ms": row["source_end_ms"],
                "reserved_pauses_ms": row["prediction"]["reserved_min_pauses_ms"],
                "speech_budget_ms": row["prediction"]["single_utterance_speech_window_ms"],
                "max_chars": row["prediction"]["max_chars_normalized_single_unit"], "length_basis": "normalized_tts_text",
                "budget_evidence": "calibrated_prediction_not_a_measured_rewrite", "review_required": True,
                "review_reasons": row["review_reasons"], "context": [asdict(c) for c in cues[max(0, row["positions"][0] - 2):row["positions"][-1] + 3]],
                "constraints": "Preserve original meaning as one unit. No automatic assignment to source cues. Joined TTS does not retain internal cue silence automatically; pause meaning, speaker identity, fresh audio timing and alignment need review. Current per-cue shift caps are not a new single-unit product contract.",
                "measured_separate_clip_duration_deficit_ms": row["duration_deficit_ms"],
                "continuous_joined_tts_fit_unknown": True})
        _write_json(checkpoint, result)
    _write_json(output / "group_shortening_inputs.json", {"status": "offline_descriptors_only", "llm_calls": 0, "tasks": tasks})
    if protected != frozen_hashes([source, Path("output/alignment_pilot_20261001")]):
        raise ValueError("protected source/prior pilot bytes changed")
    result.update(status="completed_local_tts_only", protected_hashes_unchanged=True,
                  pilot_summary={"groups": len(selected), "cues": len(measurement_cues),
                      "physical_individual_wavs": measured["summary"]["physical_file_count"],
                      "classifications": dict(Counter(r["timing_classification"] for r in selected)),
                      "group_shortening_descriptors": len(tasks)})
    result["physical_file_hashes"] = frozen_hashes([output / "individual" / "wav", output / "groups"])
    _write_json(checkpoint, result)
    return result


def verify_saved(output: Path) -> dict[str, Any]:
    result = json.loads((output / "candidates.json").read_text(encoding="utf-8"))
    if result["status"] != "completed_local_tts_only" or result["llm_calls"] != 0:
        raise ValueError("not a completed local pilot")
    if any(not Path(path).is_file() or _sha256(Path(path)) != sha for path, sha in result["protected_file_hashes"].items()):
        raise ValueError("protected source/prior pilot bytes changed")
    if result["physical_file_hashes"] != frozen_hashes([output / "individual" / "wav", output / "groups"]):
        raise ValueError("physical WAV bytes changed")
    cues = load_cues(Path(result["source_dir"]))
    settings = result["settings"]
    profile_path = Path(settings["profile_path"])
    if _sha256(profile_path) != settings["profile_sha256"]:
        raise ValueError("calibration profile bytes changed")
    replay_scan = scan(cues, json.loads(profile_path.read_text(encoding="utf-8")), settings["identity"])
    if replay_scan["summary"] != result["summary"] or replay_scan["pilot_group_ids"] != result["pilot_group_ids"]:
        raise ValueError("scan or deterministic selection differs")
    replay_rows = {r["group_id"]: r for r in replay_scan["candidates"]}
    seen: set[int] = set()
    verified_groups = 0
    for row in result["candidates"]:
        if row["group_id"] not in result["pilot_group_ids"]:
            continue
        if seen.intersection(row["positions"]) or row["eligibility"] != "candidate_requires_review":
            raise ValueError("overlapping or ambiguous audio selection")
        for key in ("cue_ids", "original_joined_text", "current_joined_text", "measurement_texts", "source_state", "reason_codes"):
            if row[key] != replay_rows[row["group_id"]][key]:
                raise ValueError("source/current text or boundary evidence differs")
        seen.update(row["positions"])
        group = [replace(cues[position], text=text) for position, text in zip(row["positions"], row["measurement_texts"])]
        durations: list[int] = []
        for cue, audio in zip(group, row["audio"]):
            if audio["tts_text"] != normalize_vosk_text(cue.text):
                raise ValueError("individual normalized text differs")
            with wave.open(str(output / "individual" / audio["path"]), "rb") as wav:
                if wav.getnframes() != audio["frames"] or wav.getframerate() != audio["samplerate"]:
                    raise ValueError("individual WAV metadata differs")
                durations.append(math.ceil(wav.getnframes() * 1000 / wav.getframerate()))
        if durations != row["duration_ms_ceil"]:
            raise ValueError("measured durations differ")
        if allocate(group, row["duration_ms_ceil"], previous_end=row["previous_end_ms"], next_start=row["next_start_ms"]) != row["allocation"]:
            raise ValueError("timing replay differs")
        if classify_timing(row, row["allocation"], durations) != row["timing_classification"]:
            raise ValueError("measured classification differs")
        if row["allocation"]["proposal"] is None:
            if "assembled_wav" in row:
                raise ValueError("failed group exported")
            continue
        evidence = row["assembled_wav"]
        with wave.open(evidence["path"], "rb") as wav:
            frames, rate = wav.getnframes(), wav.getframerate()
            frame_bytes = wav.getnchannels() * wav.getsampwidth()
            pcm = wav.readframes(frames)
        if frames != row["source_span_ms"] * rate // 1000 or frames != evidence["frames"]:
            raise ValueError("physical envelope mismatch")
        for cue, audio, placement, (start, end) in zip(group, row["audio"], row["allocation"]["proposal"]["placements"], evidence["clip_frame_ranges"]):
            if audio["tts_text"] != normalize_vosk_text(cue.text) or start != math.ceil((placement["new_start_ms"] - row["source_start_ms"]) * rate / 1000):
                raise ValueError("physical/text placement mismatch")
            with wave.open(str(output / "individual" / audio["path"]), "rb") as wav:
                clip = wav.readframes(wav.getnframes())
            if pcm[start * frame_bytes:end * frame_bytes] != clip:
                raise ValueError("speech bytes differ")
        if any(any(pcm[a * frame_bytes:b * frame_bytes]) for a, b in evidence["silence_frame_ranges"]):
            raise ValueError("inserted silence is not zero")
        if any(count * 1000 < MIN_PAUSE_MS * rate for count in evidence["inter_cue_zero_frames"]):
            raise ValueError("physical pause below minimum")
        if evidence["speech_frames"] + evidence["inserted_zero_frames"] != frames:
            raise ValueError("frame conservation mismatch")
        verified_groups += 1
    return {"status": "verified_without_inference", "protected_files": len(result["protected_file_hashes"]),
            "wav_files": len(result["physical_file_hashes"]), "assembled_groups_verified": verified_groups,
            "pilot_summary": result["pilot_summary"]}


def joined_timing(frames: int, rate: int, span_ms: int, reserve_ms: int) -> dict[str, Any]:
    """Outer duration checks only; reserving time never proves internal silence."""
    if frames <= 0 or rate <= 0 or not 0 <= reserve_ms < span_ms:
        raise ValueError("invalid measured audio or group window")
    duration = frames * 1000 / rate
    fits_span = frames * 1000 <= span_ms * rate
    return {"joined_duration_ms": duration, "joined_duration_ms_ceil": math.ceil(duration),
            "raw_span_margin_ms": span_ms - duration, "fits_raw_span": fits_span,
            "reserved_budget_margin_ms_diagnostic": span_ms - reserve_ms - duration,
            "fits_reserved_budget_diagnostic": frames * 1000 <= (span_ms - reserve_ms) * rate,
            "timing_status": "fits_span_needs_pause_and_speaker_review" if fits_span else "joined_duration_exceeds_span",
            "internal_pauses_verified": False, "internal_cue_alignment_known": False,
            "per_cue_start_shift_ms": None, "production_approved": False}


def measure_joined(pilot: Path, output: Path) -> dict[str, Any]:
    """Synthesize only the six frozen joins, without recreating individual clips."""
    if output.exists():
        raise FileExistsError(f"never overwrite joined comparison: {output}")
    verify_saved(pilot)
    prior_path = pilot / "candidates.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    settings = prior["settings"]
    if (settings["voice"], settings["rate"], settings["preprocessing"]) != (8, 1.5, PREPROCESSING):
        raise ValueError("joined comparison requires the frozen voice/rate/normalizer")
    model = Path(settings["model_path"])
    identity = profile_identity(model_metadata(model), voice=8, rate=1.5)
    if identity != settings["identity"]:
        raise ValueError("model identity changed since individual synthesis")
    group_ids = prior["pilot_group_ids"]
    by_id = {r["group_id"]: r for r in prior["candidates"]}
    if len(group_ids) != 6 or len(set(group_ids)) != 6:
        raise ValueError("exactly the six frozen groups are required")
    inputs: list[dict[str, Any]] = []
    for group_id in group_ids:
        row = by_id[group_id]
        texts = row["measurement_texts"]
        if not 2 <= len(texts) <= 3 or len(texts) != len(row["cue_ids"]):
            raise ValueError("invalid frozen group text provenance")
        joined = " ".join(texts)
        normalized = normalize_vosk_text(joined)
        reserve = MIN_PAUSE_MS * (len(texts) - 1)
        inputs.append({"group_id": group_id, "cue_ids": row["cue_ids"], "position": row["positions"][0],
            "measurement_texts": texts, "joined_display_text": joined, "joined_text_sha256": _text_id(joined),
            "normalized_tts_text": normalized, "normalized_tts_sha256": _text_id(normalized),
            "original_joined_text": row["original_joined_text"], "saved_current_joined_text": row["current_joined_text"],
            "measurement_text_policy": row["measurement_text_policy"], "cue_provenance": row["cues"],
            "source_start_ms": row["source_start_ms"], "source_end_ms": row["source_end_ms"],
            "source_span_ms": row["source_span_ms"], "reserved_min_intercue_ms_diagnostic": reserve,
            "separate_clip_durations_ms_ceil": row["duration_ms_ceil"],
            "separate_min_required_ms_ceil": sum(row["duration_ms_ceil"]) + reserve,
            "review_reasons": row["review_reasons"], "review_required": True})
    protected = frozen_hashes([pilot])
    result: dict[str, Any] = {"schema_version": 1, "status": "frozen_inputs_before_synthesis",
        "prior_candidate_manifest": {"path": str(prior_path), "sha256": _sha256(prior_path)},
        "prior_pilot_file_hashes": protected, "settings": settings,
        "settings_sha256": _text_id(json.dumps(settings, ensure_ascii=False, sort_keys=True)),
        "groups": inputs, "llm_calls": 0, "network_calls": 0, "provider_cost_usd": 0,
        "individual_clips_regenerated": 0, "listened": False, "text_edits": False,
        "pause_policy": "Continuous unchanged text only. The 200ms per internal boundary is a diagnostic time reserve, not inserted or verified silence. Native internal pauses are unidentified; no word/cue alignment or speaker detection. No trimming, speedup, automatic assignment to old cues or SRT export. The prior per-cue 500ms shift cap cannot be evaluated without internal alignment."}
    output.mkdir(parents=True)
    checkpoint = output / "joined_results.json"
    _write_json(checkpoint, result)
    measured = verify_audio({"records": [{"cue_id": r["group_id"], "position": r["position"],
        "status": "short_timing", "original": r["joined_display_text"],
        "budget": {"start": r["source_start_ms"], "end": r["source_end_ms"], "available_ms": r["source_span_ms"]}}
        for r in inputs]}, output / "continuous", model_path=model, voice=8, speech_rate=1.5,
        text_mode="punctuation", limit=6)
    audio_rows = {r["cue_id"]: r for r in measured["records"]}
    if set(audio_rows) != set(group_ids):
        raise ValueError("joined measurement group IDs differ")
    for row in inputs:
        audio_row = audio_rows[row["group_id"]]
        if audio_row.get("error") or not audio_row.get("original"):
            row.update(timing_status="synthesis_error", error=audio_row.get("error"),
                       fits_raw_span=None, fits_reserved_budget_diagnostic=None)
        else:
            audio = dict(audio_row["original"])
            if audio["tts_text"] != row["normalized_tts_text"] or audio["display_text"] != row["joined_display_text"]:
                raise ValueError("joined text changed during synthesis")
            path = output / "continuous" / audio["path"]
            with wave.open(str(path), "rb") as wav:
                if wav.getnframes() != audio["frames"] or wav.getframerate() != audio["samplerate"]:
                    raise ValueError("joined WAV frame metadata differs")
                if len(wav.readframes(wav.getnframes())) != wav.getnframes() * wav.getnchannels() * wav.getsampwidth():
                    raise ValueError("joined WAV is truncated")
            audio["path"] = str(path.relative_to(output))
            row.update(audio=audio, wav_sha256=_sha256(path), **joined_timing(audio["frames"], audio["samplerate"],
                       row["source_span_ms"], row["reserved_min_intercue_ms_diagnostic"]))
            row["separate_min_required_minus_joined_ms"] = row["separate_min_required_ms_ceil"] - row["joined_duration_ms"]
        _write_json(checkpoint, result)
    if protected != frozen_hashes([pilot]):
        raise ValueError("prior pilot bytes changed")
    verify_saved(pilot)
    result.update(status="completed_local_joined_tts_only", prior_pilot_hashes_unchanged=True,
                  summary={"groups": len(inputs), "physical_joined_wavs": measured["summary"]["physical_file_count"],
                    "fits_raw_span": sum(r["fits_raw_span"] is True for r in inputs),
                    "fits_reserved_budget_diagnostic": sum(r["fits_reserved_budget_diagnostic"] is True for r in inputs),
                    "synthesis_errors": sum(r["timing_status"] == "synthesis_error" for r in inputs)})
    _write_json(checkpoint, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("output/openrouter_episode_20260930_clean"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/grouping_pilot_20261001"))
    parser.add_argument("--model-path", type=Path, default=Path("models/vosk-model-tts-ru-0.10-multi"))
    parser.add_argument("--profile", type=Path, default=Path("output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect-selection", action="store_true")
    mode.add_argument("--verify-saved", action="store_true")
    mode.add_argument("--measure-joined", action="store_true", help="Compare only the frozen six groups as continuous TTS")
    parser.add_argument("--joined-output-dir", type=Path, default=Path("output/grouping_joined_check_20261001"))
    args = parser.parse_args(argv)
    if args.measure_joined:
        result = measure_joined(args.output_dir, args.joined_output_dir)
        print(json.dumps(result["summary"], ensure_ascii=False))
        for row in result["groups"]:
            print(json.dumps({key: row.get(key) for key in ("group_id", "cue_ids", "source_span_ms",
                "separate_min_required_ms_ceil", "joined_duration_ms", "raw_span_margin_ms",
                "fits_raw_span", "fits_reserved_budget_diagnostic", "timing_status")}, ensure_ascii=False))
        return 0
    if args.verify_saved:
        print(json.dumps(verify_saved(args.output_dir), ensure_ascii=False))
        return 0
    result = run(args.source_dir, args.output_dir, args.model_path, args.profile, inspect=args.inspect_selection)
    print(json.dumps({"scan": result["summary"], "pilot": result.get("pilot_summary")}, ensure_ascii=False))
    for row in result["candidates"]:
        if row["group_id"] in result["pilot_group_ids"] or "cue-1519" in row["cue_ids"]:
            print(json.dumps({key: row.get(key) for key in ("group_id", "cue_ids", "original_joined_text", "current_joined_text", "source_state", "reason_codes", "review_reasons", "timing_classification", "duration_ms_ceil", "duration_deficit_ms")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
