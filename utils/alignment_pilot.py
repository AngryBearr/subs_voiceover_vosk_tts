"""Bounded, local-only subtitle timing experiment; never a production export."""

from __future__ import annotations

import argparse
import json
import math
import re
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from utils.json_to_srt import json_to_srt
from utils.shortening_audio import PREPROCESSING, _sha256, _write_json, verify_audio
from utils.vosk_text import normalize_vosk_text

SHIFT_MS = 500
MIN_PAUSE_MS = 200
GAP_BOUNDARY_MS = 500
SELECTION_RULE = (
    "Split chronological short_timing anchors into six equal-rank bands; "
    "within each band choose the smallest source window (then position); "
    "triple bands first prefer anchors with a saved accepted adjacent cue, "
    "excluding already selected cue positions only. Alternate anchor+next pairs "
    "and previous+anchor+next triples. Keep boundary-invalid groups, never rank "
    "by measured or predicted success."
)


@dataclass(frozen=True)
class Cue:
    cue_id: str
    position: int
    index: int
    start: int
    end: int
    text: str
    original_text: str
    text_lines: tuple[str, ...]
    status: str
    available_ms: int
    speaker: str | None = None
    dialogue_marker: bool = False


def load_cues(source_dir: Path) -> list[Cue]:
    """Join immutable source provenance to the saved accepted/current export."""
    report = json.loads((source_dir / "report.json").read_text(encoding="utf-8"))
    state = json.loads((source_dir / "review_state.json").read_text(encoding="utf-8"))
    current = json.loads((source_dir / "subtitles.json").read_text(encoding="utf-8"))
    source = state["source_items"]
    if len(source) != len(current):
        raise ValueError("source/current cue count mismatch")
    records = {row["position"]: row for row in report["records"]}
    cues: list[Cue] = []
    for position, (original, final) in enumerate(zip(source, current)):
        for field in ("start", "end", "index"):
            if original[field] != final[field] or type(final[field]) is not int:
                raise ValueError(f"source provenance mismatch: {position}/{field}")
        lines = final["text"]
        lines = [lines] if isinstance(lines, str) else lines
        original_lines = original["text"]
        original_lines = [original_lines] if isinstance(original_lines, str) else original_lines
        if not all(isinstance(line, str) for line in [*lines, *original_lines]):
            raise ValueError("text lines must be strings")
        # The existing planner joins display lines with spaces for TTS.
        text, original_text = " ".join(lines), " ".join(original_lines)
        row = records.get(position, {})
        status = row.get("status", "unselected")
        expected = row.get("final_text") if status == "accepted" else original_text
        if text != expected:
            raise ValueError(f"saved final/original text mismatch: cue-{position}")
        available = row.get("budget", {}).get("available_ms", final.get("analysis", {}).get("available_ms"))
        if type(available) is not int or available < 0:
            raise ValueError(f"invalid saved available window: cue-{position}")
        speaker = original.get("speaker_id", original.get("speaker"))
        # Gender is not speaker identity. Explicit dialogue prefixes block grouping.
        marker = any(re.match(r"^\s*[-–—]\s|^\s*[^:]{1,30}:\s", line) for line in original_lines)
        cues.append(Cue(f"cue-{position}", position, final["index"], final["start"],
                        final["end"], text, original_text, tuple(lines), status,
                        available, str(speaker) if speaker is not None else None, marker))
    return cues


def select_groups(cues: list[Cue]) -> list[dict[str, Any]]:
    """Select at most six disjoint groups before seeing any audio measurements."""
    anchors = [cue for cue in cues if cue.status == "short_timing"]
    used: set[int] = set()
    groups: list[dict[str, Any]] = []
    for band in range(6):
        candidates = anchors[len(anchors) * band // 6:len(anchors) * (band + 1) // 6]
        def rank(cue: Cue) -> tuple[bool, int, int]:
            adjacent = cues[max(0, cue.position - 1):cue.position + 2]
            prefer_accepted = band % 2 == 1 and any(c.status == "accepted" for c in adjacent)
            return not prefer_accepted, cue.end - cue.start, cue.position

        candidates = sorted(candidates, key=rank)
        for anchor in candidates:
            first = anchor.position if band % 2 == 0 else anchor.position - 1
            size = 2 if band % 2 == 0 else 3
            positions = list(range(first, first + size))
            if first < 0 or positions[-1] >= len(cues) or used.intersection(positions):
                continue
            group = cues[first:first + size]
            groups.append({"group_id": f"group-{band + 1}", "anchor_id": anchor.cue_id,
                           "rank_band": band + 1, "cues": group,
                           "previous_end_ms": cues[first - 1].end if first else None,
                           "next_start_ms": cues[first + size].start if first + size < len(cues) else None})
            used.update(positions)
            break
    return groups


def boundary_reasons(cues: list[Cue], previous_end: int | None, next_start: int | None) -> list[str]:
    reasons: list[str] = []
    if not 2 <= len(cues) <= 3:
        reasons.append("group_size")
    if not cues:
        return reasons
    if any(cue.start < 0 or cue.end <= cue.start for cue in cues):
        reasons.append("invalid_source_timing")
    if previous_end is not None and previous_end > cues[0].start:
        reasons.append("overlap_with_previous_external_cue")
    if next_start is not None and next_start < cues[-1].end:
        reasons.append("overlap_with_next_external_cue")
    if any(cue.dialogue_marker for cue in cues):
        reasons.append("explicit_dialogue_marker")
    for left, right in zip(cues, cues[1:]):
        if right.position != left.position + 1:
            reasons.append("not_adjacent")
        gap = right.start - left.end
        if gap < 0:
            reasons.append("source_overlap")
        if gap > GAP_BOUNDARY_MS:
            reasons.append("large_gap_boundary")
        if left.speaker != right.speaker and (left.speaker is not None or right.speaker is not None):
            reasons.append("speaker_boundary")
    return list(dict.fromkeys(reasons))


def allocate(cues: list[Cue], durations_ms: list[int], *, previous_end: int | None = None,
             next_start: int | None = None) -> dict[str, Any]:
    """Try earliest legal placement, first retaining source gaps, then minimum gaps."""
    if not cues or len(cues) != len(durations_ms) or any(type(ms) is not int or ms <= 0 for ms in durations_ms):
        raise ValueError("one positive, rounded-up measured duration is required per cue")
    reasons = boundary_reasons(cues, previous_end, next_start)
    start, end = cues[0].start, cues[-1].end
    source_gaps = [right.start - left.end for left, right in zip(cues, cues[1:])]
    result: dict[str, Any] = {"status": "unresolved", "reasons": reasons,
                              "envelope_start_ms": start, "envelope_end_ms": end,
                              "envelope_ms": end - start, "source_gaps_ms": source_gaps,
                              "attempts": [], "proposal": None}
    if reasons:
        return result
    preferred = [max(MIN_PAUSE_MS, gap) for gap in source_gaps]
    for label, pauses in (("source_gaps_retained", preferred),
                          ("adjustable_gaps_to_minimum", [MIN_PAUSE_MS] * len(source_gaps))):
        if result["attempts"] and pauses == preferred:
            break
        cursor = start
        placements: list[dict[str, Any]] = []
        violations: list[str] = []
        for i, (cue, duration) in enumerate(zip(cues, durations_ms)):
            new_start = max(cursor, cue.start - SHIFT_MS, start)
            new_end = new_start + duration
            shift = new_start - cue.start
            if abs(shift) > SHIFT_MS:
                violations.append(f"{cue.cue_id}:start_shift_cap")
            if new_end > end:
                violations.append(f"{cue.cue_id}:out_of_envelope")
            placements.append({"cue_id": cue.cue_id, "text_id": _text_id(cue.text),
                               "text": cue.text, "original_start_ms": cue.start,
                               "original_end_ms": cue.end, "new_start_ms": new_start,
                               "new_end_ms": new_end, "start_shift_ms": shift,
                               "duration_ms_ceil": duration,
                               "source_duration_excess_ms": max(0, duration - (cue.end - cue.start))})
            cursor = new_end + (pauses[i] if i < len(pauses) else 0)
        actual_pauses = [right["new_start_ms"] - left["new_end_ms"]
                         for left, right in zip(placements, placements[1:])]
        attempt = {"pause_mode": label, "required_pauses_ms": pauses,
                   "actual_pauses_ms": actual_pauses, "placements": placements,
                   "violations": violations, "end_overrun_ms": max(0, cursor - end),
                   "max_abs_start_shift_ms": max(abs(row["start_shift_ms"]) for row in placements)}
        result["attempts"].append(attempt)
        if not violations:
            unused = end - start - sum(durations_ms) - sum(actual_pauses)
            result.update(status="feasible_inspect_only", reasons=[], proposal=attempt,
                          speech_reserved_ms=sum(durations_ms), pauses_reserved_ms=sum(actual_pauses),
                          unused_edge_silence_ms=unused,
                          conservation_ms=sum(durations_ms) + sum(actual_pauses) + unused)
            return result
    result["reasons"] = result["attempts"][-1]["violations"]
    return result


def _text_id(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assemble(paths: list[Path], placements: list[dict[str, Any]], output: Path,
             envelope_start: int, envelope_end: int) -> dict[str, Any]:
    """Copy complete PCM frames without mixing, trimming, resampling or gain changes."""
    if len(paths) != len(placements) or not paths:
        raise ValueError("WAV/placement mismatch")
    clips: list[bytes] = []
    frames: list[int] = []
    format_key: tuple[int, int, int] | None = None
    for path in paths:
        with wave.open(str(path), "rb") as wav:
            key = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
            if key[1] not in (2, 3, 4) or wav.getcomptype() != "NONE":
                raise ValueError("zero-valued silence requires signed uncompressed PCM")
            if format_key is not None and format_key != key:
                raise ValueError("group WAV formats differ")
            format_key = key
            frames.append(wav.getnframes())
            clips.append(wav.readframes(wav.getnframes()))
    assert format_key is not None
    channels, width, rate = format_key
    total_frames = (envelope_end - envelope_start) * rate // 1000
    frame_bytes = channels * width
    buffer = bytearray(total_frames * frame_bytes)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for placement, clip, count in zip(placements, clips, frames):
        begin = math.ceil((placement["new_start_ms"] - envelope_start) * rate / 1000)
        finish = begin + count
        if begin < cursor or finish > total_frames:
            raise ValueError("physical audio overlaps or exceeds group envelope")
        buffer[begin * frame_bytes:finish * frame_bytes] = clip
        ranges.append((begin, finish))
        cursor = finish
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(buffer)
    with wave.open(str(output), "rb") as wav:
        actual = wav.readframes(wav.getnframes())
        if wav.getnframes() != total_frames or actual != buffer:
            raise ValueError("assembled WAV roundtrip mismatch")
    silent_ranges = [(0, ranges[0][0]), *[(a[1], b[0]) for a, b in zip(ranges, ranges[1:])],
                     (ranges[-1][1], total_frames)]
    if any(any(actual[a * frame_bytes:b * frame_bytes]) for a, b in silent_ranges):
        raise ValueError("inserted silence is not zero-valued")
    if any(actual[a * frame_bytes:b * frame_bytes] != clip for (a, b), clip in zip(ranges, clips)):
        raise ValueError("speech frames were changed")
    inter_frames = [b[0] - a[1] for a, b in zip(ranges, ranges[1:])]
    if any(count * 1000 < MIN_PAUSE_MS * rate for count in inter_frames):
        raise ValueError("physical silence below minimum")
    return {"path": str(output), "frames": total_frames, "samplerate": rate,
            "duration_sec": total_frames / rate, "speech_frames": sum(frames),
            "inserted_zero_frames": total_frames - sum(frames),
            "inter_cue_zero_frames": inter_frames, "silence_frame_ranges": silent_ranges,
            "clip_frame_ranges": ranges, "speech_bytes_unchanged": True,
            "inserted_silence_verified_zero": True,
            "frame_conservation": sum(frames) + sum(b - a for a, b in silent_ranges) == total_frames}


def run_pilot(source_dir: Path, output_dir: Path, model_path: Path, profile_path: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"never overwrite a pilot: {output_dir}")
    # Reuse the existing identity guard; no model download or provider client.
    from utils.vosk_calibration import model_metadata, profile_identity, validate_character_profile

    identity = profile_identity(model_metadata(model_path), voice=8, rate=1.5)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    validate_character_profile(profile, expected_identity=identity)
    cues = load_cues(source_dir)
    groups = select_groups(cues)
    if len(groups) != 6:
        raise ValueError("this bounded episode pilot requires six disjoint groups")
    frozen_hashes = {str(path): _sha256(path) for path in sorted(source_dir.rglob("*")) if path.is_file()}
    output_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema_version": 1, "status": "selected_before_synthesis", "inspect_only": True,
        "source_dir": str(source_dir), "frozen_source_hashes": frozen_hashes,
        "selection_rule": SELECTION_RULE, "short_timing_population": sum(c.status == "short_timing" for c in cues),
        "shift_cap_ms": SHIFT_MS, "min_inter_cue_silence_ms": MIN_PAUSE_MS,
        "large_gap_boundary_ms": GAP_BOUNDARY_MS, "max_group_size": 3,
        "settings": {"model_path": str(model_path), "voice": 8, "rate": 1.5,
                     "preprocessing": PREPROCESSING, "identity": identity,
                     "profile_path": str(profile_path), "profile_sha256": _sha256(profile_path)},
        "llm_calls": 0, "network_calls": 0, "model_provider_cost_usd": 0,
        "speedup_or_trimming": False, "listened": False,
        "speaker_detection": "explicit tags/prefixes only; gender is not identity; no diarization",
        "pause_policy": "Try max(200ms, source positive gap), then explicit 200ms minimum; gaps above 500ms block. Native WAV silence is retained, not counted as spare time. No separate per-cue 200ms overhead is added; no next external gap is borrowed.",
        "groups": [{**{k: v for k, v in group.items() if k != "cues"},
                    "cues": [asdict(cue) | {"text_id": _text_id(cue.text)} for cue in group["cues"]]}
                   for group in groups],
    }
    checkpoint = output_dir / "timing_proposal.json"
    _write_json(checkpoint, manifest)
    selected = [cue for group in groups for cue in group["cues"]]
    # Fresh synthesis only, with exact-text deduplication inside verify_audio.
    audio = verify_audio({"records": [{"cue_id": cue.cue_id, "position": cue.position,
                         "status": "short_timing", "original": cue.text,
                         "budget": {"start": cue.start, "end": cue.end, "available_ms": cue.available_ms}}
                        for cue in selected]}, output_dir / "individual", model_path=model_path,
                         voice=8, speech_rate=1.5, limit=18, text_mode="punctuation")
    measured = {row["cue_id"]: row for row in audio["records"]}
    group_wavs = output_dir / "groups"
    group_wavs.mkdir()
    srt_rows: list[dict[str, Any]] = []
    for group, saved in zip(groups, manifest["groups"]):
        included = group["cues"]
        rows = [measured[cue.cue_id] for cue in included]
        if any(row.get("error") or not row.get("original") for row in rows):
            saved["allocation"] = {"status": "unresolved", "reasons": ["synthesis_error"]}
            _write_json(checkpoint, manifest)
            continue
        durations = [math.ceil(row["original"]["frames"] * 1000 / row["original"]["samplerate"]) for row in rows]
        for i, (cue, row, duration, provenance) in enumerate(zip(included, rows, durations, saved["cues"])):
            if row["original"]["tts_text"] != normalize_vosk_text(cue.text):
                raise ValueError("measured TTS text mismatch")
            next_start = included[i + 1].start if i + 1 < len(included) else group["next_start_ms"]
            enlarged_end = cue.end + max(0, next_start - cue.end - MIN_PAUSE_MS) if next_start is not None else cue.end
            baseline_end = min(enlarged_end, included[-1].end)
            provenance.update(audio=row["original"], duration_ms_ceil=duration,
                              source_window_ms=cue.end - cue.start,
                              source_window_fits=duration <= cue.end - cue.start,
                              saved_enlarged_window_ms=cue.available_ms,
                              saved_enlarged_window_fits=duration <= cue.available_ms,
                              baseline_same_envelope_window_ms=max(0, baseline_end - cue.start),
                              baseline_same_envelope_fits=duration <= baseline_end - cue.start)
        allocation = allocate(included, durations, previous_end=group["previous_end_ms"],
                              next_start=group["next_start_ms"])
        saved["allocation"] = allocation
        saved["baseline_individual_fits"] = sum(c["baseline_same_envelope_fits"] for c in saved["cues"])
        saved["baseline_joint_feasible"] = (all(c["baseline_same_envelope_fits"] for c in saved["cues"])
            and all(right.start - (left.start + duration) >= MIN_PAUSE_MS
                    for left, right, duration in zip(included, included[1:], durations)))
        saved["rescued_cue_ids"] = []
        if allocation["proposal"] is not None:
            proposal = allocation["proposal"]
            paths = [output_dir / "individual" / row["original"]["path"] for row in rows]
            saved["assembled_wav"] = assemble(paths, proposal["placements"],
                group_wavs / f"{group['group_id']}_inspect_only.wav", included[0].start, included[-1].end)
            saved["rescued_cue_ids"] = [cue["cue_id"] for cue in saved["cues"] if not cue["baseline_same_envelope_fits"]]
            for cue, placement in zip(included, proposal["placements"]):
                srt_rows.append({"index": cue.index, "start": placement["new_start_ms"],
                                 "end": placement["new_end_ms"], "text": list(cue.text_lines)})
        _write_json(checkpoint, manifest)
    if srt_rows:
        subset = output_dir / "feasible_inspect_only_subset.json"
        subset.write_text(json.dumps(srt_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        json_to_srt(str(subset), str(output_dir / "feasible_inspect_only_subset.srt"))
    if frozen_hashes != {str(path): _sha256(path) for path in sorted(source_dir.rglob("*")) if path.is_file()}:
        raise ValueError("immutable source output changed during pilot")
    manifest["status"] = "completed_local_tts_only"
    manifest["source_hashes_unchanged"] = True
    manifest["summary"] = {
        "groups": len(groups), "cues": len(selected),
        "physical_individual_wavs": audio["summary"]["physical_file_count"],
        "feasible_groups": sum(g["allocation"]["status"] == "feasible_inspect_only" for g in manifest["groups"]),
        "failed_groups": sum(g["allocation"]["status"] == "unresolved" for g in manifest["groups"]),
        "rescued_cues": sum(len(g.get("rescued_cue_ids", [])) for g in manifest["groups"]),
        "rescued_short_timing_cues": sum(c["status"] == "short_timing" and c["cue_id"] in g.get("rescued_cue_ids", [])
                                         for g in manifest["groups"] for c in g["cues"]),
    }
    _write_json(checkpoint, manifest)
    return manifest


def verify_saved(output_dir: Path) -> dict[str, Any]:
    """Recheck saved selection, timing, source bytes and WAV bytes without synthesis."""
    manifest = json.loads((output_dir / "timing_proposal.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed_local_tts_only" or manifest["llm_calls"] != 0:
        raise ValueError("not a completed local-only pilot")
    source_dir = Path(manifest["source_dir"])
    hashes = {str(path): _sha256(path) for path in sorted(source_dir.rglob("*")) if path.is_file()}
    if hashes != manifest["frozen_source_hashes"]:
        raise ValueError("frozen source bytes changed")
    groups = select_groups(load_cues(source_dir))
    if len(groups) != len(manifest["groups"]):
        raise ValueError("selection count changed")
    seen: set[str] = set()
    feasible = rescued = 0
    for group, saved in zip(groups, manifest["groups"]):
        if group["anchor_id"] != saved["anchor_id"] or group["group_id"] != saved["group_id"]:
            raise ValueError("selection changed")
        if len(group["cues"]) != len(saved["cues"]):
            raise ValueError("group cue count changed")
        durations: list[int] = []
        clips: list[bytes] = []
        for cue, row in zip(group["cues"], saved["cues"]):
            if any(row[key] != value for key, value in asdict(cue).items() if key != "text_lines"):
                raise ValueError("cue provenance changed")
            if row["text_lines"] != list(cue.text_lines) or row["text_id"] != _text_id(cue.text):
                raise ValueError("display text changed")
            audio = row["audio"]
            if audio["tts_text"] != normalize_vosk_text(cue.text):
                raise ValueError("normalized text changed")
            with wave.open(str(output_dir / "individual" / audio["path"]), "rb") as wav:
                if wav.getnframes() != audio["frames"] or wav.getframerate() != audio["samplerate"]:
                    raise ValueError("individual WAV metadata changed")
                durations.append(math.ceil(wav.getnframes() * 1000 / wav.getframerate()))
                clips.append(wav.readframes(wav.getnframes()))
            if cue.cue_id in seen:
                raise ValueError("groups overlap")
            seen.add(cue.cue_id)
        replay = allocate(group["cues"], durations, previous_end=group["previous_end_ms"],
                          next_start=group["next_start_ms"])
        if replay != saved["allocation"]:
            raise ValueError("saved allocation differs from deterministic replay")
        if replay["proposal"] is None:
            if saved.get("assembled_wav") or saved["rescued_cue_ids"]:
                raise ValueError("failed group was exported as fitting")
            continue
        feasible += 1
        rescued += len(saved["rescued_cue_ids"])
        evidence = saved["assembled_wav"]
        with wave.open(evidence["path"], "rb") as wav:
            data = wav.readframes(wav.getnframes())
            frame_bytes = wav.getnchannels() * wav.getsampwidth()
            rate = wav.getframerate()
            if wav.getnframes() != evidence["frames"] or wav.getframerate() != evidence["samplerate"]:
                raise ValueError("group WAV metadata changed")
            if wav.getnframes() != (replay["envelope_end_ms"] - replay["envelope_start_ms"]) * rate // 1000:
                raise ValueError("group WAV exceeds or differs from envelope")
        expected_ranges = []
        for row, clip in zip(replay["proposal"]["placements"], clips):
            begin = math.ceil((row["new_start_ms"] - replay["envelope_start_ms"]) * rate / 1000)
            expected_ranges.append([begin, begin + len(clip) // frame_bytes])
        if evidence["clip_frame_ranges"] != expected_ranges:
            raise ValueError("physical placement differs from timing proposal")
        if any((right[0] - left[1]) * 1000 < MIN_PAUSE_MS * rate
               for left, right in zip(expected_ranges, expected_ranges[1:])):
            raise ValueError("physical inter-cue silence below minimum")
        for (start, end), clip in zip(evidence["clip_frame_ranges"], clips):
            if data[start * frame_bytes:end * frame_bytes] != clip:
                raise ValueError("group speech bytes changed")
        if any(any(data[start * frame_bytes:end * frame_bytes])
               for start, end in evidence["silence_frame_ranges"]):
            raise ValueError("group silence is not zero-valued")
        if evidence["speech_frames"] + evidence["inserted_zero_frames"] != evidence["frames"]:
            raise ValueError("physical frame conservation failed")
    subset = output_dir / "feasible_inspect_only_subset.json"
    expected_subset = [{"index": cue["index"], "start": row["new_start_ms"], "end": row["new_end_ms"],
                        "text": cue["text_lines"]}
                       for group in manifest["groups"] if group["allocation"]["proposal"] is not None
                       for cue, row in zip(group["cues"], group["allocation"]["proposal"]["placements"])]
    if expected_subset:
        if json.loads(subset.read_text(encoding="utf-8")) != expected_subset:
            raise ValueError("subset text/timing changed")
        if (output_dir / "feasible_inspect_only_subset.srt").read_text(encoding="utf-8") != json_to_srt(str(subset)):
            raise ValueError("subset SRT differs from JSON")
    return {"status": "verified_without_inference", "frozen_source_files": len(hashes),
            "groups": len(groups), "cues": len(seen), "feasible_groups": feasible,
            "rescued_cues": rescued, "speech_bytes_unchanged": True, "inserted_silence_verified_zero": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("output/openrouter_episode_20260930_clean"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/alignment_pilot_20261001"))
    parser.add_argument("--model-path", type=Path, default=Path("models/vosk-model-tts-ru-0.10-multi"))
    parser.add_argument("--profile", type=Path, default=Path("output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect-selection", action="store_true", help="Print selection only; no output or synthesis")
    mode.add_argument("--verify-saved", action="store_true", help="Verify existing evidence without synthesis or writing")
    args = parser.parse_args(argv)
    if args.verify_saved:
        print(json.dumps(verify_saved(args.output_dir), sort_keys=True))
        return 0
    if args.inspect_selection:
        cues = load_cues(args.source_dir)
        print(SELECTION_RULE)
        for group in select_groups(cues):
            print(json.dumps({"group": group["group_id"], "anchor": group["anchor_id"],
                "cues": [{"id": c.cue_id, "span_ms": c.end - c.start, "text": c.text, "status": c.status}
                         for c in group["cues"]], "boundary_reasons": boundary_reasons(group["cues"],
                group["previous_end_ms"], group["next_start_ms"])}, ensure_ascii=False))
        return 0
    result = run_pilot(args.source_dir, args.output_dir, args.model_path, args.profile)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
