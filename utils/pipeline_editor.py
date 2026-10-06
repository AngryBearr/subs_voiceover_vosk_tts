"""Full-source manual editor and explicit local audio queue; no model transport."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from pipeline import load_pipeline_config
from utils import pipeline_batch as batch
from utils.analyze_text import join_text_lines
from utils.json_to_srt import _ms_to_srt_timestamp
from utils.pipeline_text import json_sha256, prepare_text, resolve_speed_policy, text_sha256
from utils.srt_to_json import parse_srt_file
from utils.vosk_text import normalize_vosk_text

FIT = batch.FIT
ROOT = batch.ROOT


def _items(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() != ".srt":
        return batch.read_json(path)
    # The general parser repairs malformed indices. Identity imports must not.
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip())
    timestamp = r"(?:\d{1,2}:)?\d{1,2}:\d{1,2}(?:[.,]\d{1,3})?"
    for block in blocks:
        lines = block.splitlines()
        if (len(lines) < 3 or not re.fullmatch(r"\d+", lines[0].strip())
                or not re.fullmatch(rf"\s*{timestamp}\s*-->\s*{timestamp}\s*", lines[1])):
            raise ValueError("Expected complete SRT blocks with explicit integer indices")
    return parse_srt_file(str(path))


def _validate(cues: Any) -> None:
    if not isinstance(cues, list) or not cues or any(not isinstance(cue, dict) for cue in cues):
        raise ValueError("Expected a nonempty full subtitle list")
    indices = [cue.get("index") for cue in cues]
    if any(type(index) is not int or index < 0 for index in indices) or len(set(indices)) != len(indices):
        raise ValueError("Expected unique nonnegative integer indices")
    previous_end = -1
    for cue in cues:
        start, end = cue.get("start"), cue.get("end")
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            raise ValueError("Expected positive duration and nonnegative integer millisecond timestamps")
        if start < previous_end:
            raise ValueError("Subtitle order/overlap is invalid")
        previous_end = end
        text = cue.get("text")
        if not (isinstance(text, str) or isinstance(text, list) and all(isinstance(line, str) for line in text)):
            raise ValueError("Subtitle text must be a string or string list")
        joined = join_text_lines(text)
        if "\x00" in joined or any(not line.strip() for line in re.split(r"\r\n?|\n", joined)):
            raise ValueError("Subtitle text must be nonempty and cannot contain empty SRT lines")
        speaker = cue.get("speaker_id")
        if speaker is not None and (not isinstance(speaker, str) or not speaker.strip()):
            raise ValueError("speaker_id must be nonempty text or null")


def _windows(state: dict[str, Any]) -> list[int]:
    plan = prepare_text(state["cues"], resolve_speed_policy(state["config"]),
                        float(state["config"].get("analysis", {}).get("avg_chars_per_sec", 13)))
    return [row["target"]["available_ms"] for row in plan["entries"]]


def _binding(state: dict[str, Any], cue: dict[str, Any]) -> str:
    return json_sha256({key: cue[key] for key in ("cue_id", "text", "start", "end", "available_ms")} |
                       {"config": state["config_sha256"], "runtime": state["runtime"]})


def _persist(folder: Path, state: dict[str, Any]) -> None:
    state["revision"] = json_sha256({key: value for key, value in state.items() if key != "revision"})
    batch.save_json(folder / "project.json", state)


def _load(folder: Path) -> dict[str, Any]:
    state = batch.read_json(folder / "project.json")
    if state.get("revision") != json_sha256({key: value for key, value in state.items() if key != "revision"}):
        raise ValueError("Project revision checksum mismatch")
    if state.get("schema") != "pipeline-editor-v1":
        raise ValueError("Expected pipeline-editor-v1 project")
    for name, digest in state["snapshots"].items():
        if batch.file_sha256(folder / name) != digest:
            raise ValueError(f"Frozen {name} changed")
    if json_sha256(state["config"]) != state["config_sha256"]:
        raise ValueError("Frozen config binding mismatch")
    source = batch.read_json(folder / "full_source.json")
    if json_sha256(source) != state["baseline_source_sha256"]:
        raise ValueError("Baseline source hash mismatch")
    _validate(state["cues"])
    if len(source) != len(state["cues"]) or any(
            cue["cue_id"] != f"cue-{position}" or cue["index"] != original["index"]
            for position, (cue, original) in enumerate(zip(state["cues"], source))):
        raise ValueError("Full source coverage/identity mismatch")
    if [cue["available_ms"] for cue in state["cues"]] != _windows(state):
        raise ValueError("Canonical window binding mismatch")
    return state


def _runtime(state: dict[str, Any]) -> None:
    if batch.runtime_binding(state["config"]) != state["runtime"]:
        raise ValueError("Runtime/model assets changed; create a new source-only project, no audio rebinding")
    if batch.file_sha256(Path(__file__)) != state["editor_code_sha256"]:
        raise ValueError("Editor runtime changed; create a new project")


def _approved(state: dict[str, Any], cue: dict[str, Any]) -> bool:
    binding = _binding(state, cue)
    return any(cue.get(key, {}).get("binding") == binding for key in ("human_approval", "native_approval"))


def _current(state: dict[str, Any], cue: dict[str, Any]) -> dict[str, Any] | None:
    if not cue["attempts"]:
        return None
    attempt = cue["attempts"][-1]
    return attempt if attempt.get("generation") == cue["generation"] and attempt.get("binding") == _binding(state, cue) else None


def _check_audio(state: dict[str, Any], cue: dict[str, Any]) -> dict[str, Any]:
    attempt = _current(state, cue)
    if not attempt or attempt.get("status") not in FIT:
        raise ValueError("Current measured fit is missing")
    for path, digest in attempt.get("wav_sha256", {}).items():
        if not Path(path).is_file() or batch.file_sha256(Path(path)) != digest:
            raise ValueError("WAV changed/missing")
    final = attempt["final"]
    info = batch.wav_info(Path(final["path"]))
    if info != final or attempt.get("wav_sha256", {}).get(final["path"]) != info["sha256"]:
        raise ValueError("Current measured WAV binding mismatch")
    policy = resolve_speed_policy(state["config"])
    speed, factor = attempt.get("effective_total_speed"), attempt.get("effective_factor")
    if (type(speed) not in (int, float) or type(factor) not in (int, float)
            or not math.isfinite(speed) or not math.isfinite(factor) or factor < 1
            or speed != factor * policy.base_speech_rate or speed > policy.max_total_speed
            or info["frames"] * 1000 > cue["available_ms"] * info["sample_rate"]):
        raise ValueError("Current frame duration/speed exceeds canonical window/cap")
    return info


def _view(state: dict[str, Any]) -> dict[str, Any]:
    view = copy.deepcopy(state)
    runtime_valid = True
    runtime_reason: str | None = None
    try:
        _runtime(state)
    except (ValueError, RuntimeError, OSError) as exc:
        runtime_valid = False
        runtime_reason = str(exc)
    view["runtime_valid"] = runtime_valid
    view["runtime_reason"] = runtime_reason
    for cue, row in zip(state["cues"], view["cues"]):
        attempt = _current(state, cue)
        row["review_status"] = "human_approved" if cue.get("human_approval", {}).get("binding") == _binding(state, cue) else (
            "native_approved" if _approved(state, cue) else "pending")
        row["status"] = attempt.get("status", "pending_audio") if attempt else cue.get("pending_status", "pending_audio")
        if row["status"] == "started":
            row["status"] = "needs_reconciliation"
        row.update(audio_path=None, duration_ms=None)
        row["reason"] = attempt.get("error", row["status"]) if attempt else row["status"]
        if attempt and attempt.get("final") and row["status"] not in FIT:
            try:
                info = batch.wav_info(Path(attempt["final"]["path"]))
                if info == attempt["final"] and attempt.get("wav_sha256", {}).get(info["path"]) == info["sha256"]:
                    row.update(audio_path=info["path"], duration_ms=info["duration_sec"] * 1000)
            except (ValueError, OSError, RuntimeError):
                pass
        if row["status"] in FIT:
            try:
                info = _check_audio(state, cue)
                row.update(audio_path=info["path"], duration_ms=info["duration_sec"] * 1000)
            except (ValueError, OSError, RuntimeError, KeyError) as exc:
                row.update(status="audio_invalid", reason=str(exc))
        if not runtime_valid:
            row.update(status="runtime_changed", reason=runtime_reason, audio_path=None)
        row["needs_human"] = row["review_status"] == "pending" or row["status"] not in FIT
    view["ready_for_mix"] = all(not cue["needs_human"] for cue in view["cues"])
    return view


def create_project(source_path: Path, folder: Path, config_path: Path) -> dict[str, Any]:
    """Freeze a complete source and selected config without inference; never overwrite."""
    source_path, folder = source_path.resolve(), folder.resolve()
    items = _items(source_path)
    _validate(items)
    if not config_path.is_file():
        raise FileNotFoundError(f"Selected config does not exist: {config_path}")
    config = load_pipeline_config(config_path)
    resolve_speed_policy(config)
    from utils.silero_pauses import validate_profile
    validate_profile(config)
    if config.get("tts", {}).get("output_sample_rate") is not None:
        raise ValueError("Use resample.target_sr; TTS fallback resampling is not supported")
    if config.get("audio_analysis", {}).get("prefer_ffmpeg", True) is not True:
        raise ValueError("Bounded audio requires pitch-preserving ffmpeg")
    runtime = batch.runtime_binding(config)
    folder.mkdir(parents=True, exist_ok=False)
    with batch.locked(folder):
        source_name = "source" + source_path.suffix.lower()
        shutil.copyfile(source_path, folder / source_name)
        batch.save_json(folder / "full_source.json", items)
        batch.save_json(folder / "config.json", config)
        cues = [{**copy.deepcopy(item), "cue_id": f"cue-{position}", "text": join_text_lines(item["text"]),
                 "speaker_id": item.get("speaker_id"), "generation": 0, "attempts": [], "human_edited": False,
                 "human_approval": {}, "native_approval": {}}
                for position, item in enumerate(items)]
        state = {"schema": "pipeline-editor-v1", "source_path": str(source_path),
                 "baseline_source_sha256": json_sha256(items), "config": config,
                 "config_sha256": json_sha256(config), "runtime": runtime,
                 "editor_code_sha256": batch.file_sha256(Path(__file__)), "cues": cues, "imports": {},
                 "snapshots": {name: batch.file_sha256(folder / name)
                               for name in (source_name, "full_source.json", "config.json")}}
        for cue, available in zip(cues, _windows(state)):
            cue["available_ms"] = available
        _persist(folder, state)
        return _view(state)


def load_project(folder: Path) -> dict[str, Any]:
    with batch.locked(folder, readonly=True):
        return _view(_load(folder))


def _revision(state: dict[str, Any], expected: str) -> None:
    if state["revision"] != expected:
        raise ValueError("Stale expected_revision; reload before editing")


def _ids(state: dict[str, Any], cue_ids: list[str]) -> list[dict[str, Any]]:
    if not isinstance(cue_ids, list) or any(not isinstance(value, str) for value in cue_ids) or len(set(cue_ids)) != len(cue_ids):
        raise ValueError("Expected unique cue IDs")
    known = {cue["cue_id"]: cue for cue in state["cues"]}
    if any(value not in known for value in cue_ids):
        raise ValueError("Unknown cue ID")
    return [known[value] for value in cue_ids]


def _edit(state: dict[str, Any], edits: list[dict[str, Any]]) -> bool:
    if not isinstance(edits, list) or any(not isinstance(row, dict) or "cue_id" not in row or
            set(row) - {"cue_id", "text", "start", "end", "speaker_id"} for row in edits):
        raise ValueError("Only cue_id, text, start, end, speaker_id edits are allowed")
    rows = _ids(state, [row["cue_id"] for row in edits])
    before = copy.deepcopy(state["cues"])
    for cue, edit in zip(rows, edits):
        if "text" in edit and not isinstance(edit["text"], str):
            raise ValueError("Edited text must be a string")
        cue.update({key: value for key, value in edit.items() if key != "cue_id"})
    _validate(state["cues"])
    for cue, old, available in zip(state["cues"], before, _windows(state)):
        cue["available_ms"] = available
        changed = any(cue[key] != old[key] for key in ("text", "start", "end", "available_ms"))
        if changed:
            cue.update(generation=old["generation"] + 1, human_approval={}, native_approval={}, human_edited=True,
                       pending_status="needs_review" if cue["text"] != old["text"] else "pending_recheck")
        elif cue["speaker_id"] != old["speaker_id"]:
            cue["human_edited"] = True
    return before != state["cues"]


def apply_edits(folder: Path, edits: list[dict[str, Any]], *, expected_revision: str) -> dict[str, Any]:
    with batch.locked(folder):
        state = _load(folder)
        _revision(state, expected_revision)
        if _edit(state, edits):
            _persist(folder, state)
        return _view(state)


def import_srt(folder: Path, srt_path: Path, *, expected_revision: str) -> dict[str, Any]:
    if srt_path.suffix.lower() != ".srt":
        raise ValueError("Expected an SRT file for editor import")
    items = _items(srt_path)
    with batch.locked(folder):
        state = _load(folder)
        _revision(state, expected_revision)
        if [item["index"] for item in items] != [cue["index"] for cue in state["cues"]]:
            raise ValueError("Imported SRT must cover every original index exactly once in original order")
        edits = [{"cue_id": cue["cue_id"], "text": join_text_lines(item["text"]),
                  "start": item["start"], "end": item["end"]} for cue, item in zip(state["cues"], items)]
        if _edit(state, edits):
            _persist(folder, state)
        return _view(state)


def approve_cues(folder: Path, cue_ids: list[str], *, expected_revision: str) -> dict[str, Any]:
    """Human text/timeline approval only; not model review or an audio listening claim."""
    with batch.locked(folder):
        state = _load(folder)
        _revision(state, expected_revision)
        changed = False
        for cue in _ids(state, cue_ids):
            binding = _binding(state, cue)
            if cue.get("human_approval", {}).get("binding") != binding:
                cue["human_approval"] = {"binding": binding, "kind": "human_text_timeline", "unix": time.time()}
                changed = True
        if changed:
            _persist(folder, state)
        return _view(state)


def adopt_batch(folder: Path, batch_folder: Path) -> dict[str, Any]:
    """Copy a strictly validated current batch without modifying its state or audio."""
    with batch.locked(folder):
        state = _load(folder)
        _runtime(state)
        with batch.locked(batch_folder, readonly=True):
            native = batch.load_state(batch_folder)
            source = batch.read_json(batch_folder / "full_source.json")
            if source != batch.read_json(folder / "full_source.json") or native["config"] != state["config"]:
                raise ValueError("Batch full origin/config mismatch")
            if native["binding"]["runtime"] != state["runtime"]:
                raise ValueError("Batch runtime mismatch")
            digest = native["state_sha256"]
            if digest in state["imports"]:
                return _view(state)
            selected = _ids(state, [cue["cue_id"] for cue in native["cues"]])
            if any(cue["human_edited"] or cue.get("provenance") or cue["attempts"] or cue["human_approval"] for cue in selected):
                raise ValueError("Batch would overwrite human edits, existing audio or another import")
            reviews = [value["result"] for _, value in sorted(native["imports"].items(), key=lambda pair: int(pair[0]))
                       if "result" in value]
            accepted = bool(reviews and reviews[-1]["execution"] == "native_live"
                            and reviews[-1]["assembly_review"]["verdict"] == "accept"
                            and reviews[-1]["assembly_review"]["assembly_sha256"] == json_sha256(batch.assembled(native)))
            for cue, imported in zip(selected, native["cues"]):
                if imported["original"] != source[imported["position"]] or cue["available_ms"] != imported["target"]["available_ms"]:
                    raise ValueError("Batch cue origin/window mismatch")
            # An interrupted copy is unadopted evidence, not a permanent retry blocker.
            # Unique attempt directories also preserve any preexisting orphan/user files.
            imports_directory = folder.resolve() / "imports"
            imports_directory.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix=f"{digest}-", dir=imports_directory))
            batch.save_json(directory / "state.json", native)
            for cue, imported in zip(selected, native["cues"]):
                cue["text"] = imported["chosen_text"]
                proof = None
                if accepted:
                    original = join_text_lines(imported["original"]["text"])
                    if cue["text"] == original:
                        proof = {"kind": "native_exact_assembly_original"}
                    else:
                        for value in reviews:
                            for row in value["entries"]:
                                review = row.get("review") or {}
                                if (row["cue_id"] == cue["cue_id"] and row.get("candidate") == cue["text"]
                                        and review.get("verdict") == "accept"
                                        and review.get("text_sha256") == text_sha256(cue["text"])
                                        and value["assembly_review"]["verdict"] == "accept"):
                                    proof = {"kind": "native_exact_candidate_and_assembly", "review": copy.deepcopy(review)}
                if proof:
                    cue["native_approval"] = {**proof, "binding": _binding(state, cue), "batch_sha256": digest,
                                               "assembly_review": copy.deepcopy(reviews[-1]["assembly_review"])}
                cue["provenance"] = {"batch_path": str(batch_folder.resolve()), "state_sha256": digest,
                                     "decisions": copy.deepcopy(imported["decisions"])}
                cue["attempts"] = copy.deepcopy(imported["attempts"])
                for number, attempt in enumerate(cue["attempts"]):
                    paths: dict[str, str] = {}
                    audio_dir = directory / cue["cue_id"] / str(number)
                    audio_dir.mkdir(parents=True)
                    for old, wav_digest in attempt.get("wav_sha256", {}).items():
                        new = audio_dir / Path(old).name
                        shutil.copyfile(old, new)
                        if batch.file_sha256(new) != wav_digest:
                            raise ValueError("Batch WAV changed while copying")
                        paths[old] = str(new)
                    for key in ("raw", "pretempo", "pauses", "edges", "final"):
                        if key in attempt and attempt[key]["path"] in paths:
                            attempt[key]["path"] = paths[attempt[key]["path"]]
                    attempt["wav_sha256"] = {paths[path]: value for path, value in attempt.get("wav_sha256", {}).items()}
                    attempt["generation"] = cue["generation"]
                    if attempt["text_sha256"] == text_sha256(cue["text"]):
                        attempt["binding"] = _binding(state, cue)
                    attempt["runtime"] = state["runtime"]
            state["imports"][digest] = {"path": str(batch_folder.resolve()), "snapshot": str(directory / "state.json")}
            _persist(folder, state)
            return _view(state)


def export_project(folder: Path) -> dict[str, str]:
    """Export all cues, including unresolved ones; statuses belong only in JSON."""
    with batch.locked(folder):
        state = _load(folder)
        view = _view(state)
        directory = folder.resolve() / "exports" / state["revision"]
        directory.mkdir(parents=True, exist_ok=True)
        srt, metadata = directory / "subtitles.srt", directory / "project.json"
        # json_to_srt sorts indices, which would change full-source position identities.
        content = "\n\n".join(f"{cue['index']}\n{_ms_to_srt_timestamp(cue['start'])} --> "
                              f"{_ms_to_srt_timestamp(cue['end'])}\n{cue['text']}" for cue in state["cues"]) + "\n"
        temporary = srt.with_suffix(".srt.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(srt)
        batch.save_json(metadata, view)
        return {"srt_path": str(srt), "json_path": str(metadata)}


def retry_cues(folder: Path, cue_ids: list[str], *, expected_revision: str) -> dict[str, Any]:
    """Explicitly authorize one fresh attempt after a failure/interruption; never automatic."""
    with batch.locked(folder):
        state = _load(folder)
        _revision(state, expected_revision)
        cues = _ids(state, cue_ids)
        for cue in cues:
            attempt = _current(state, cue)
            if not attempt:
                raise ValueError("Retry requires a current failed/interrupted attempt")
            if attempt["status"] in FIT:
                try:
                    _check_audio(state, cue)
                except (ValueError, OSError, RuntimeError, KeyError):
                    raw = attempt.get("raw")
                    try:
                        if (not raw or batch.wav_info(Path(raw["path"])) != raw
                                or attempt.get("wav_sha256", {}).get(raw["path"]) != raw["sha256"]):
                            raise ValueError("Immutable RAW changed/missing; retry cannot resynthesize a completed fit")
                    except (OSError, RuntimeError, KeyError) as exc:
                        raise ValueError("Immutable RAW changed/missing; retry cannot resynthesize a completed fit") from exc
                else:
                    raise ValueError("Retry cannot repeat a valid current fit")
        for cue in cues:
            cue["generation"] += 1
            cue["pending_status"] = "pending_retry"
        if cues:
            _persist(folder, state)
        return _view(state)


def process_pending(folder: Path) -> dict[str, Any]:
    """Explicit audio action: one durable attempt per approved cue generation, no LLM."""
    with batch.locked(folder):
        state = _load(folder)
        _runtime(state)
        engine = batch.AudioEngine(state["config"])
        pause_processor: Any = None
        for cue in state["cues"]:
            if not _approved(state, cue) or _current(state, cue):
                continue
            attempt: dict[str, Any] = {"binding": _binding(state, cue), "generation": cue["generation"],
                "text_sha256": text_sha256(cue["text"]), "runtime": state["runtime"],
                "status": "started", "phase": "normalization", "tts_attempts": 0,
                "available_ms": cue["available_ms"], "started_unix": time.time()}
            reusable = next((old for old in reversed(cue["attempts"])
                             if old.get("text_sha256") == attempt["text_sha256"] and old.get("runtime") == state["runtime"]
                             and old.get("raw") and old.get("status") in FIT | {"unresolved"}), None)
            cue["attempts"].append(attempt)
            _persist(folder, state)  # Intent before normalization, loading assets or synthesis.
            directory = folder.resolve() / "audio" / cue["cue_id"] / str(len(cue["attempts"]))
            inputs: list[str] = []
            started = time.monotonic()
            try:
                directory.mkdir(parents=True, exist_ok=False)
                raw = directory / "raw.wav"
                if reusable:
                    attempt["phase"] = "audio"
                    old_raw = reusable["raw"]
                    if batch.wav_info(Path(old_raw["path"])) != old_raw:
                        raise ValueError("Immutable RAW changed/missing")
                    shutil.copyfile(old_raw["path"], raw)
                    attempt["reused_raw_sha256"] = old_raw["sha256"]
                else:
                    normalized = engine.normalize({**cue, "text": cue["text"]})
                    attempt.update(normalized_text=normalized, engine_input_text=normalize_vosk_text(normalized),
                                   phase="tts", tts_attempts=1)
                    _persist(folder, state)
                    engine.synthesize(normalized, raw, inputs)
                    if inputs != [attempt["engine_input_text"]]:
                        raise ValueError("Engine input differs from bound normalization")
                attempt["phase"] = "audio"
                _persist(folder, state)
                if state["config"].get("silero_pauses", {}).get("enabled", False) and pause_processor is None:
                    from utils.silero_pauses import PauseProcessor
                    pause_processor = PauseProcessor()
                batch.measure_audio(state["config"], raw, directory / "pretempo.wav", directory / "final.wav",
                                    cue["available_ms"] / 1000, attempt, pause_processor, chosen_text=cue["text"],
                                    available_ms=cue["available_ms"])
            except Exception as exc:
                attempt.update(status={"normalization": "normalization_failed", "tts": "tts_failed",
                                       "audio": "audio_failed"}[attempt["phase"]], error=str(exc))
            attempt["captured_engine_inputs"] = inputs
            attempt["elapsed_sec"] = time.monotonic() - started
            attempt["wav_sha256"] = {str(path): batch.file_sha256(path) for path in directory.glob("*.wav")}
            _persist(folder, state)
        return _view(state)


def mix_ready(folder: Path) -> str:
    """Validate current approval, complete coverage and decoded WAVs before full mixing."""
    import numpy as np
    import soundfile as sf
    from utils.audio_mixer import _read_audio, mix_segments
    with batch.locked(folder):
        state = _load(folder)
        _runtime(state)
        for position, cue in enumerate(state["cues"]):
            if not _approved(state, cue):
                raise ValueError(f"Current text/timeline needs approval: {cue['cue_id']}")
            info = _check_audio(state, cue)
            data, rate = sf.read(info["path"], always_2d=True)
            if rate != info["sample_rate"] or len(data) != info["frames"] or not np.isfinite(data).all():
                raise ValueError("Decoded audio frame/finite-sample validation failed")
            if position + 1 < len(state["cues"]) and (cue["start"] * rate + len(data) * 1000
                    > state["cues"][position + 1]["start"] * rate):
                raise ValueError("Final timeline overlap")
        # Fresh indexed copies prevent the mixer's naming convention from touching originals.
        directory = folder.resolve() / "mixes" / f"{state['revision']}-{time.time_ns()}"
        directory.mkdir(parents=True, exist_ok=False)
        settings = state["config"].get("mix", {})
        sample_rate = int(settings.get("sample_rate", 44100))
        channels = int(settings.get("channels", 1))
        if sample_rate <= 0 or channels <= 0:
            raise ValueError("Mix sample rate and channel count must be positive")
        for position, cue in enumerate(state["cues"]):
            info = _check_audio(state, cue)
            staged = directory / f"{cue['index']}.wav"
            shutil.copyfile(info["path"], staged)
            if batch.file_sha256(staged) != info["sha256"]:
                raise ValueError("WAV changed during mix staging")
            converted = _read_audio(staged, sample_rate, channels)
            start_frame = round(cue["start"] * sample_rate / 1000)
            if (not np.isfinite(converted).all() or len(converted) <= 0
                    or len(converted) * 1000 > cue["available_ms"] * sample_rate):
                raise ValueError("Resampled mix frames exceed canonical window")
            if position + 1 < len(state["cues"]) and start_frame + len(converted) > round(
                    state["cues"][position + 1]["start"] * sample_rate / 1000):
                raise ValueError("Resampled mix timeline overlap")
        batch.save_json(directory / "timeline.json", [{key: cue[key] for key in ("index", "start", "end", "text")}
                                                       for cue in state["cues"]])
        return str(mix_segments(directory / "timeline.json", directory, directory / "voiceover",
                                sample_rate=sample_rate, channels=channels,
                                transcode_to=None, missing_policy="error", progress_every=0))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    if Path.cwd().resolve() != ROOT:
        parser.error(f"Run from repository root: {ROOT}")
    commands = parser.add_subparsers(dest="command", required=True)
    help_text = {"init": "Freeze full source/config in a NEW project; no inference",
                 "adopt": "Copy a strictly validated native batch; no inference",
                 "export": "Write FULL draft SRT and status JSON; no inference",
                 "import": "Apply a complete edited SRT; no automatic approval",
                 "approve": "Explicit human current text/timeline approval",
                 "audio": "Explicitly authorize local normalization/TTS for approved pending cues",
                 "mix": "Validate all current audio and assemble full voiceover; no TTS",
                 "status": "Read-only status; no inference",
                 "retry": "Explicitly authorize one fresh failed/interrupted cue attempt"}
    for name in ("init", "adopt", "export", "import", "approve", "audio", "mix", "status", "retry"):
        command = commands.add_parser(name, help=help_text[name])
        command.add_argument("--project", required=True, type=Path)
        if name == "init":
            command.add_argument("source", type=Path)
            command.add_argument("--config", type=Path, default=ROOT / "pipeline_config_vosk.json")
        if name in {"adopt", "import"}:
            command.add_argument("source", type=Path)
        if name in {"import", "approve", "retry"}:
            command.add_argument("--expected-revision", required=True)
        if name in {"approve", "retry"}:
            selection = command.add_mutually_exclusive_group(required=True)
            selection.add_argument("--cue-ids", nargs="+")
            selection.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    folder = args.project
    result: Any
    if args.command == "init":
        result = create_project(args.source, folder, args.config)
    elif args.command == "adopt":
        result = adopt_batch(folder, args.source)
    elif args.command == "import":
        result = import_srt(folder, args.source, expected_revision=args.expected_revision)
    elif args.command in {"approve", "retry"}:
        ids = [cue["cue_id"] for cue in load_project(folder)["cues"]] if args.all else args.cue_ids
        result = (approve_cues if args.command == "approve" else retry_cues)(folder, ids, expected_revision=args.expected_revision)
    else:
        result = {"export": export_project, "audio": process_pending, "mix": mix_ready,
                  "status": load_project}[args.command](folder)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
