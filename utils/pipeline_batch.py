"""Resumable two-round native text -> measured Vosk batch. No model transport."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import html
import importlib.metadata
import json
import math
import os
import platform
import shutil
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from pipeline import load_pipeline_config
from utils.analyze_text import join_text_lines
from utils.length_check import prepare_length_text
from utils.pipeline_text import (apply_reviewed_text, candidate_eligibility, json_sha256, prepare_text,
                                 resolve_speed_policy, text_sha256)
from utils.srt_to_json import parse_srt_file
from utils.vosk_text import normalize_vosk_text

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = ("pipeline.py", "utils/pipeline_batch.py", "utils/pipeline_text.py",
                 "utils/shortening_plan.py", "utils/length_check.py", "utils/text_metrics.py",
                 "utils/vosk_calibration.py", "utils/analyze_text.py", "utils/srt_to_json.py",
                 "utils/text_normalizer.py", "utils/vosk_text.py", "synthesize/synthesize.py",
                  "utils/reduce_silence.py", "utils/audio_resample.py", "utils/audio_analysis.py",
                  "utils/silero_pauses.py")
FIT = {"fits", "accelerated_fits"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, value: Any) -> None:
    """Flush data before atomic replacement; only called under the writer lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def locked(folder: Path, *, readonly: bool = False) -> Iterator[None]:
    # ponytail: one local writer; split locks only if parallel batches need it.
    if not readonly:
        folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("r" if readonly else "a") as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_SH if readonly else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Batch is in use by another process") from exc
        yield


def runtime_binding(config: dict[str, Any]) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("runorm", "ruaccent", "vosk-tts", "onnxruntime", "torch", "transformers",
                    "numpy", "soundfile", "scipy", "soxr", "silero-vad"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    model_path = config.get("tts", {}).get("model_path")
    directories = [ROOT / "runorm_cache"]
    if model_path:
        directories.append(Path(model_path))
    for package, module in (("ruaccent", "ruaccent"), ("runorm", "runorm"), ("vosk-tts", "vosk_tts")):
        try:
            # RUAccent stores downloaded weights in its installed package by default.
            directories.append(Path(importlib.metadata.distribution(package).locate_file(module)))
        except importlib.metadata.PackageNotFoundError:
            pass
    assets = [path for directory in directories if directory.is_dir()
              for path in sorted(directory.rglob("*"))
              if path.is_file() and not {"__pycache__", ".cache", ".locks", ".no_exist", "refs", ".git"}.intersection(path.parts)
              and not path.name.endswith((".lock", ".metadata", ".incomplete"))
              and path.name != ".gitattributes"]
    ffmpeg = shutil.which("ffmpeg")
    from utils.silero_pauses import EDGE_POLICY, runtime_identity, validate_profile
    if config.get("silero_edges", {}).get("enabled", False):
        validate_profile(config)
    pause_contract = (runtime_identity() if config.get("silero_pauses", {}).get("enabled", False)
                      else {"enabled": False})
    return {"python": platform.python_version(), "packages": versions, "silero_pauses": pause_contract,
            "silero_edges": {"enabled": bool(config.get("silero_edges", {}).get("enabled", False)),
                             "policy": dict(EDGE_POLICY)},
            "code": {name: file_sha256(ROOT / name) for name in RUNTIME_FILES},
            "model_assets": {str(path.resolve()): file_sha256(path) for path in assets},
            "ffmpeg": {"path": ffmpeg, "sha256": file_sha256(Path(ffmpeg)) if ffmpeg else None}}


def persist(folder: Path, state: dict[str, Any]) -> None:
    state["state_sha256"] = json_sha256({k: v for k, v in state.items() if k != "state_sha256"})
    save_json(folder / "state.json", state)


def load_state(folder: Path) -> dict[str, Any]:
    state = read_json(folder / "state.json")
    if state.get("state_sha256") != json_sha256({k: v for k, v in state.items() if k != "state_sha256"}):
        raise ValueError("Batch state checksum mismatch")
    binding = state["binding"]
    if json_sha256(binding) != state["binding_sha256"]:
        raise ValueError("Batch binding mismatch")
    if file_sha256(Path(binding["source_path"])) != binding["source_file_sha256"]:
        raise ValueError("Source changed; use a new batch")
    for name in ("full_source.json", "full_plan.json", "config.json"):
        if file_sha256(folder / name) != binding["snapshots"][name]:
            raise ValueError(f"Frozen {name} changed; use a new batch")
    if runtime_binding(state["config"]) != binding["runtime"]:
        raise ValueError("Runtime/model assets changed; explicit reconciliation or a new batch required")
    for number, request in state["requests"].items():
        if request["request_sha256"] != json_sha256({k: v for k, v in request.items() if k != "request_sha256"}):
            raise ValueError(f"Request round {number} checksum mismatch")
        path = folder / f"request_round{number}.json"
        if path.exists() and read_json(path) != request:
            raise ValueError(f"Exported request round {number} changed")
    for cue in state["cues"]:
        if text_sha256(cue["chosen_text"]) != cue["chosen_sha256"]:
            raise ValueError(f"Chosen text binding mismatch: {cue['cue_id']}")
        for attempt in cue["attempts"]:
            if "silero_edges" in attempt and (
                    attempt["silero_edges"].get("chosen_text_sha256") != attempt["text_sha256"]
                    or attempt["silero_edges"].get("available_ms") != cue["target"]["available_ms"]):
                raise ValueError(f"Edge text/window binding mismatch: {cue['cue_id']}")
            for path, digest in attempt.get("wav_sha256", {}).items():
                if not Path(path).is_file() or file_sha256(Path(path)) != digest:
                    raise ValueError(f"WAV changed/missing: {path}")
    return state


def assembled(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"cue_id": cue["cue_id"], "text": cue["chosen_text"]} for cue in state["cues"]]


def proposal_eligibility(cue: dict[str, Any], candidate: str | None, round_number: int) -> str:
    """Reject identical repairs and timing repairs that cannot shorten current text."""
    if round_number == 2 and candidate is not None:
        canonical = prepare_length_text(candidate, "normalized_tts_text")
        current = prepare_length_text(cue["chosen_text"], "normalized_tts_text")
        if canonical == current or any(attempt["text_sha256"] == text_sha256(candidate) for attempt in cue["attempts"]):
            return "repair_unchanged"
        if cue["attempts"][-1]["status"] == "unresolved" and len(canonical) >= len(current):
            return "repair_not_shorter"
    return candidate_eligibility(join_text_lines(cue["original"].get("text", "")), candidate, cue["target"]["status"])


def reviewed_selection(folder: Path, state: dict[str, Any], request: dict[str, Any],
                       rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    artifact = read_json(folder / "full_plan.json")
    artifact["entries"] = rows
    chosen, reports = apply_reviewed_text(read_json(folder / "full_source.json"), artifact,
        resolve_speed_policy(state["config"]), request["policy"]["cps"])
    by_id = {row["cue_id"]: row for row in reports}
    cues = {cue["cue_id"]: cue for cue in state["cues"]}
    for row in rows:
        reason = proposal_eligibility(cues[row["cue_id"]], row.get("candidate"), request["round"])
        if by_id[row["cue_id"]]["reason"] == "applied" and reason != "eligible":
            by_id[row["cue_id"]]["reason"] = reason
    return chosen, by_id


def make_request(folder: Path, state: dict[str, Any], round_number: int,
                 targets: list[dict[str, Any]]) -> dict[str, Any]:
    plan = read_json(folder / "full_plan.json")
    request = {"schema": "pipeline-batch-request-v1", "round": round_number,
               "binding_sha256": state["binding_sha256"], "source_sha256": plan["source_sha256"],
               "plan_sha256": plan["plan_sha256"], "policy": plan["policy"],
               "full_source_path": str((folder / "full_source.json").resolve()),
               "full_plan_path": str((folder / "full_plan.json").resolve()),
               "full_source_cues": len(plan["entries"]),
               "assembled_context": assembled(state), "entries": []}
    for cue in targets:
        position = cue["position"]
        row = copy.deepcopy(plan["entries"][position])
        row["current_text"] = cue["chosen_text"]
        row["current_normalized_chars"] = len(prepare_length_text(cue["chosen_text"], "normalized_tts_text"))
        row["original_context"] = plan["entries"][max(0, position - 3):position + 4]
        if round_number == 2:
            attempt = cue["attempts"][-1]
            row["measured_attempt"] = copy.deepcopy(attempt)
            before = attempt.get("edges", attempt.get("pauses", attempt.get("pretempo", {}))).get("duration_sec")
            if type(before) in (int, float) and math.isfinite(before) and before > 0:
                remaining = plan["policy"]["max_total_speed"] / plan["policy"]["base_speech_rate"]
                chars = row["current_normalized_chars"]
                row["target"]["measured_advisory_max_chars"] = math.floor(
                    chars * row["target"]["available_ms"] / 1000 * remaining / before * 0.9)
                row["target"]["measured_required_reduction_chars"] = max(
                    0, chars - row["target"]["measured_advisory_max_chars"])
            row["target"]["repair_reason"] = attempt["status"]
        request["entries"].append(row)
    request["instructions"] = (
        "One proposal per target, preserve meaning, numbers and natural Russian. Budgets are advisory; "
        "do not remove essential words to force fit. Use full source and original +/-3 context, not just "
        "targets. Independent reviewer checks each exact candidate hash AND all assembled final cue texts. "
        "Candidate and assembly reviews may share the same reviewer invocation, distinct from writer. "
        "Return pipeline-batch-reviewed-v1; Python never calls the writer/reviewer. "
        "Length unit is normalized_tts_text: count canonical punctuation and whitespace, including spaces "
        "and punctuation, before RUNorm/RUAccent and model-added stress. target.max_chars/status remain "
        "historical nominal estimates for the original text. Under budget does not guarantee audio fit; "
        "over budget is allowed to preserve meaning, and refusal is allowed. Semantic acceptance is not "
        "a budget/timing pass. "
        "No accepted text is human/audio approval. After measured repair there are no further loops.")
    if round_number == 2:
        request["instructions"] += (" Aim for measured_advisory_max_chars when meaning permits, not nominal "
                                    "max_chars; measured advice takes precedence. Without valid pretempo "
                                    "measurement, measured advice/reduction is unavailable. "
                                    "Repairs must differ canonically from current text; timing repairs must "
                                    "be shorter than current text. Normalization repairs may be longer if "
                                    "still shorter than original. These are mechanical guards, not meaning checks.")
    request["request_sha256"] = json_sha256(request)
    return request


def candidate_advisory_check(candidate: str | None, target: dict[str, Any], round_number: int) -> dict[str, Any]:
    """Describe length advice only; never decide semantic eligibility or audio fit."""
    chars = len(prepare_length_text(candidate, "normalized_tts_text")) if candidate is not None else None
    maximum = target.get("measured_advisory_max_chars" if round_number == 2 else "max_chars")
    return {"normalized_chars": chars, "advisory_max_chars": maximum,
            "source": "unavailable" if maximum is None else "measured" if round_number == 2 else "nominal",
            "exceeds_advisory": chars > maximum if chars is not None and maximum is not None else None}


def prepare(source: Path, folder: Path, config: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    """Select inclusive zero-based original positions, never replan a sparse subset."""
    source = source.resolve()
    items = parse_srt_file(str(source)) if source.suffix.lower() == ".srt" else read_json(source)
    if not isinstance(items, list) or not 0 <= start <= end < len(items):
        raise ValueError("Expected full subtitle list and valid inclusive start/end positions")
    if any(not isinstance(item, dict) for item in items):
        raise ValueError("Every full-source cue must be an object")
    indices = [item.get("index") for item in items]
    if any(type(index) is not int for index in indices) or len(indices) != len(set(indices)):
        raise ValueError("Full source must have unique integer indices")
    policy = resolve_speed_policy(config)
    from utils.silero_pauses import validate_profile
    validate_profile(config)
    if config.get("tts", {}).get("output_sample_rate") is not None:
        raise ValueError("Use resample.target_sr, not TTS fallback resampling (one engine attempt only)")
    if config.get("audio_analysis", {}).get("prefer_ffmpeg", True) is not True:
        raise ValueError("Bounded batch requires pitch-preserving ffmpeg")
    with locked(folder):
        if (folder / "state.json").exists():
            state = load_state(folder)
            if (state["config"], state["range"], state["binding"]["source_path"]) != (config, [start, end], str(source)):
                raise ValueError("Existing batch has another source/config/range")
            return state
        if any(path.name != ".lock" for path in folder.iterdir()):
            raise ValueError("Preparation directory is not empty; do not overwrite partial preparation")
        plan = prepare_text(items, policy, float(config.get("analysis", {}).get("avg_chars_per_sec", 13)))
        for name, value in (("full_source.json", items), ("full_plan.json", plan), ("config.json", config)):
            save_json(folder / name, value)
        binding = {"source_path": str(source), "source_file_sha256": file_sha256(source),
                   "snapshots": {name: file_sha256(folder / name)
                                 for name in ("full_source.json", "full_plan.json", "config.json")},
                   "runtime": runtime_binding(config), "range": [start, end]}
        cues = [{"cue_id": row["cue_id"], "position": position, "original": row["original"],
                 "target": row["target"], "chosen_text": join_text_lines(row["original"].get("text", "")),
                 "chosen_sha256": text_sha256(join_text_lines(row["original"].get("text", ""))),
                 "attempts": [], "decisions": []}
                for position, row in enumerate(plan["entries"]) if start <= position <= end]
        state = {"schema": "pipeline-batch-state-v1", "binding": binding,
                 "binding_sha256": json_sha256(binding), "config": copy.deepcopy(config),
                 "range": [start, end], "cues": cues, "requests": {}, "imports": {},
                 "native_phase_count": 0, "native_elapsed_sec": [], "audio_elapsed_sec": [], "proposals": {}}
        targets = [cue for cue in cues if cue["target"]["status"] not in {"fits", "invalid"}]
        state["requests"]["1"] = make_request(folder, state, 1, targets)
        if not targets:
            state["imports"]["1"] = {"status": "no_targets"}
        persist(folder, state)
        save_json(folder / "request_round1.json", state["requests"]["1"])
        return state


def identity(value: Any, label: str) -> None:
    if not isinstance(value, dict) or not all(isinstance(value.get(key), str) and value[key].strip()
                                             for key in ("model", "mode", "invocation_id")):
        raise ValueError(f"{label} must identify model, mode and invocation_id")


def validate_reviews(rows: list[dict[str, Any]], request: dict[str, Any]) -> None:
    for row in rows:
        if row.get("candidate") is not None:
            identity(row.get("writer"), "Writer")
        review = row.get("review")
        if review is not None:
            identity(review, "Reviewer")
            if review.get("request_sha256") != request["request_sha256"]:
                raise ValueError("Candidate reviewer request binding mismatch")
            if row.get("writer", {}).get("invocation_id") == review["invocation_id"]:
                raise ValueError("Writer and reviewer must be independent invocations")


def review_input(folder: Path, proposal_path: Path, per_cue_path: Path | None = None) -> dict[str, Any]:
    """Bind writer proposals and export exact hashes/context for an independent reviewer."""
    proposals = read_json(proposal_path)
    with locked(folder):
        state = load_state(folder)
        number = str(proposals.get("round"))
        request = state["requests"].get(number)
        if not request or proposals.get("schema") != "pipeline-batch-proposals-v1":
            raise ValueError("Expected pipeline-batch-proposals-v1 for an exported round")
        for key in ("request_sha256", "binding_sha256", "source_sha256", "plan_sha256", "policy"):
            if proposals.get(key) != request[key]:
                raise ValueError(f"Proposal {key} mismatch")
        if number in state["imports"] or any(attempt["round"] == int(number)
                                            for cue in state["cues"] for attempt in cue["attempts"]):
            raise ValueError("Writer stage is already closed")
        rows = proposals.get("entries")
        if not isinstance(rows, list) or [row.get("cue_id") for row in rows] != [row["cue_id"] for row in request["entries"]]:
            raise ValueError("Writer must return every target exactly once in request order")
        proposed = copy.deepcopy(state)
        for row, source in zip(rows, request["entries"]):
            if row.get("original") != source["original"] or (row.get("candidate") is not None and not isinstance(row["candidate"], str)):
                raise ValueError("Writer original/candidate mismatch")
            identity(row.get("writer"), "Writer")
            row["candidate_sha256"] = text_sha256(row["candidate"]) if row.get("candidate") is not None else None
            row["advisory_check"] = candidate_advisory_check(row.get("candidate"), source["target"], request["round"])
            cue = next(cue for cue in state["cues"] if cue["cue_id"] == row["cue_id"])
            row["eligibility_reason"] = proposal_eligibility(cue, row.get("candidate"), request["round"])
            if row["eligibility_reason"] == "eligible":
                next(cue for cue in proposed["cues"] if cue["cue_id"] == row["cue_id"])["chosen_text"] = row["candidate"]
        if proposals.get("execution") != "native_live":
            raise ValueError("Writer proposals require actual native evidence")
        prior = state["proposals"].get(number)
        if prior is not None and prior != proposals:
            raise ValueError("One writer proposal batch per round; existing proposals differ")
        if per_cue_path is not None:
            per_cue = read_json(per_cue_path)
            if per_cue.get("schema") != "pipeline-batch-reviewed-v1":
                raise ValueError("Expected pipeline-batch-reviewed-v1 per-cue review")
            for key in ("round", "request_sha256", "binding_sha256", "source_sha256", "plan_sha256", "policy", "execution"):
                if per_cue.get(key) != proposals[key]:
                    raise ValueError(f"Per-cue review {key} mismatch")
            reviewed_rows = per_cue.get("entries")
            if not isinstance(reviewed_rows, list) or len(reviewed_rows) != len(rows):
                raise ValueError("Per-cue reviews must cover the frozen proposals")
            for row, proposal in zip(reviewed_rows, rows):
                if any(row.get(key) != proposal.get(key) for key in ("cue_id", "original", "candidate", "writer")):
                    raise ValueError("Per-cue review differs from writer proposal")
                row["advisory_check"] = copy.deepcopy(proposal["advisory_check"])
            validate_reviews(reviewed_rows, request)
            chosen, decisions = reviewed_selection(folder, state, request, reviewed_rows)
            applied = {row["cue_id"] for row in decisions.values() if row["reason"] == "applied"}
            proposed = copy.deepcopy(state)
            for cue in proposed["cues"]:
                if cue["cue_id"] in applied:
                    cue["chosen_text"] = join_text_lines(chosen[cue["position"]]["text"])
        state["proposals"][number] = proposals
        if prior is None:
            state["native_phase_count"] += 1
        persist(folder, state)
        reviewer_input = {**copy.deepcopy(proposals), "schema": "pipeline-batch-review-input-v1",
                          "request": request, "assembled_context": assembled(proposed),
                          "assembly_sha256": json_sha256(assembled(proposed))}
        if per_cue_path is not None:
            reviewer_input["entries"] = reviewed_rows
        save_json(folder / f"reviewer_input_round{number}.json", reviewer_input)
        publish_report(folder, state)
        return state


def ingest_reviewed(folder: Path, result_path: Path) -> dict[str, Any]:
    """Validate the entire import before any chosen text changes."""
    result = read_json(result_path)
    digest = json_sha256(result)
    with locked(folder):
        state = load_state(folder)
        number = str(result.get("round"))
        request = state["requests"].get(number)
        if not request or result.get("schema") != "pipeline-batch-reviewed-v1":
            raise ValueError("Expected pipeline-batch-reviewed-v1 for an exported round")
        for key in ("request_sha256", "binding_sha256", "source_sha256", "plan_sha256", "policy"):
            if result.get(key) != request[key]:
                raise ValueError(f"Reviewed {key} mismatch")
        previous = state["imports"].get(number)
        if previous:
            if previous.get("result_sha256") != digest:
                raise ValueError("Round already imported; no second proposal/review is allowed")
            publish_report(folder, state)
            return state
        if any(attempt["round"] == int(number) for cue in state["cues"] for attempt in cue["attempts"]):
            raise ValueError("Cannot change text after round audio started")
        rows = result.get("entries")
        expected_ids = [row["cue_id"] for row in request["entries"]]
        if not isinstance(rows, list) or [row.get("cue_id") for row in rows] != expected_ids:
            raise ValueError("Return every target exactly once, in request order (null proposal is allowed)")
        if number in state["proposals"]:
            for row, proposal in zip(rows, state["proposals"][number]["entries"]):
                if any(row.get(key) != proposal.get(key) for key in ("cue_id", "original", "candidate", "writer")):
                    raise ValueError("Reviewed result differs from frozen writer proposals")
        chosen, by_id = reviewed_selection(folder, state, request, rows)
        validate_reviews(rows, request)
        proposed = copy.deepcopy(state)
        for cue in proposed["cues"]:
            if cue["cue_id"] in expected_ids and by_id[cue["cue_id"]]["reason"] == "applied":
                cue["chosen_text"] = join_text_lines(chosen[cue["position"]]["text"])
        assembly = result.get("assembly_review")
        identity(assembly, "Assembled reviewer")
        if (assembly.get("request_sha256") != request["request_sha256"]
                or assembly.get("assembly_sha256") != json_sha256(assembled(proposed))
                or assembly.get("verdict") not in {"accept", "revise", "refusal"}):
            raise ValueError("Exact assembled final text review is missing or stale")
        if any(row.get("writer", {}).get("invocation_id") == assembly["invocation_id"] for row in rows):
            raise ValueError("Assembled reviewer must be independent of writer")
        if result.get("execution") != "native_live":
            raise ValueError("Imports require actual native writer/reviewer evidence, not dry-run results")
        for cue in state["cues"]:
            if cue["cue_id"] not in expected_ids:
                continue
            decision = copy.deepcopy(by_id[cue["cue_id"]])
            if assembly["verdict"] != "accept" and decision["reason"] == "applied":
                decision["reason"] = "assembly_rejected"
            if decision["reason"] == "applied":
                cue["chosen_text"] = join_text_lines(chosen[cue["position"]]["text"])
                cue["chosen_sha256"] = text_sha256(cue["chosen_text"])
            cue["decisions"].append({"round": int(number), **decision})
        state["imports"][number] = {"result_sha256": digest, "result": result}
        state["native_phase_count"] += 1 if number in state["proposals"] else 2  # NOT physical requests.
        elapsed = result.get("elapsed_sec")
        if elapsed is not None and (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError("elapsed_sec must be measured finite nonnegative seconds, or absent")
        state["native_elapsed_sec"].append(elapsed)
        persist(folder, state)
        save_json(folder / f"reviewed_round{number}.json", result)
        publish_report(folder, state)
        return state


class AudioEngine:
    """Lazy, process-local model reuse; source-bound normalization for one cue."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.normalizer: Any = None
        self.accentizer: Any = None
        self.synth: Any = None

    def normalize(self, item: dict[str, Any]) -> str:
        from utils.text_normalizer import _load_runorm, _load_ruaccent, runorm_pass, ruaccent_pass
        settings = self.config.get("normalization", {})
        if self.normalizer is None:
            self.normalizer = _load_runorm(str(settings.get("norm_device", "cpu")))
        if self.accentizer is None:
            self.accentizer = _load_ruaccent(str(settings.get("accent_device", "cpu")))
        return join_text_lines(ruaccent_pass(runorm_pass([item], self.normalizer), self.accentizer,
                                            batch_size=1)[0]["text"])

    def synthesize(self, text: str, path: Path, inputs: list[str]) -> None:
        from synthesize.synthesize import create_synth, synthesize_text
        settings = self.config["tts"]
        if self.synth is None:
            self.synth = create_synth(settings["model_path"], str(settings.get("device", "cpu")))
        synthesize_text(text, voice=int(settings.get("voice", 8)),
                        speech_rate=resolve_speed_policy(self.config).base_speech_rate,
                        synth=self.synth, output_path=str(path), output_sample_rate=None,
                        tts_input_texts=inputs)


def wav_info(path: Path) -> dict[str, Any]:
    import soundfile as sf
    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("Empty/invalid WAV")
    return {"path": str(path), "frames": info.frames, "sample_rate": info.samplerate,
            "duration_sec": info.frames / info.samplerate, "sha256": file_sha256(path)}


def measure_audio(config: dict[str, Any], raw: Path, pretempo: Path, final: Path,
                   allowed: float, attempt: dict[str, Any], pause_processor: Any = None,
                   chosen_text: str | None = None, available_ms: int | None = None) -> None:
    from utils.audio_analysis import render_bounded_tempo
    from utils.audio_resample import resample_wav_file
    from utils.reduce_silence import reduce_silence_file
    if config.get("silero_edges", {}).get("enabled", False):
        from utils.silero_pauses import validate_profile
        validate_profile(config)
        if available_ms is None or type(available_ms) is not int or allowed != available_ms / 1000:
            raise ValueError("Edge stage requires the exact canonical available_ms window")
    if pretempo.exists() or final.exists() or attempt.get("raw"):
        raise ValueError("Audio processing already attempted; never repeat stages on saved audio")
    attempt["raw"] = wav_info(raw)
    shutil.copyfile(raw, pretempo)
    settings = config.get("reduce_silence", {})
    if settings.get("enabled", True):
        reduce_silence_file(pretempo, pretempo, silence_dbfs=float(settings.get("silence_dbfs", -35)),
                            threshold_sec=float(settings.get("threshold_sec", 0.5)),
                            target_sec=float(settings.get("target_sec", 0.2)))
    settings = config.get("resample", {})
    if settings.get("enabled", True):
        resample_wav_file(pretempo, pretempo, target_sr=int(settings.get("target_sr", 44100)))
    attempt["pretempo"] = wav_info(pretempo)
    measured_source = pretempo
    if config.get("silero_pauses", {}).get("enabled", False):
        from utils.silero_pauses import PauseProcessor
        processor = pause_processor if pause_processor is not None else PauseProcessor()
        measured_source = pretempo.with_name("pauses.wav")
        attempt["silero_pauses"] = processor.process(pretempo, measured_source)
        attempt["pauses"] = wav_info(measured_source)
        if config.get("silero_edges", {}).get("enabled", False):
            from utils.silero_pauses import process_edges
            edge_source = pretempo.with_name("edges.wav")
            assert available_ms is not None
            attempt["silero_edges"] = process_edges(measured_source, edge_source, attempt["silero_pauses"],
                                                    available_ms=available_ms, chosen_text=chosen_text)
            attempt["edges"] = wav_info(edge_source)
            measured_source = edge_source
    before = wav_info(measured_source)["duration_sec"]
    policy = resolve_speed_policy(config)
    remaining = policy.max_total_speed / policy.base_speech_rate
    factor = 1.0
    attempt["required_factor"] = max(1.0, before / allowed)
    attempt["render_attempts"] = []
    if before > allowed and before / allowed <= remaining and config.get("audio_analysis", {}).get("enabled", True):
        factor, _ = render_bounded_tempo(measured_source, final, allowed_duration_sec=allowed,
                                        max_factor=remaining, attempts=attempt["render_attempts"])
    else:
        shutil.copyfile(measured_source, final)
    attempt["final"] = wav_info(final)
    attempt["effective_factor"] = factor
    attempt["effective_total_speed"] = factor * policy.base_speech_rate
    attempt["status"] = ("accelerated_fits" if factor > 1 else "fits") if attempt["final"]["duration_sec"] <= allowed else "unresolved"


def audio(folder: Path) -> dict[str, Any]:
    """One attempt per cue per round; unknown interrupted attempts never retry."""
    with locked(folder):
        state = load_state(folder)
        round_number = 2 if "2" in state["requests"] else 1
        number = str(round_number)
        if number not in state["imports"]:
            raise ValueError(f"Import reviewed round {number} before audio")
        engine = AudioEngine(state["config"])
        pause_processor: Any = None
        targets = state["cues"] if round_number == 1 else [cue for cue in state["cues"]
            if any(decision["round"] == 2 and decision["reason"] == "applied" for decision in cue["decisions"])]
        for cue in targets:
            if any(attempt["round"] == round_number for attempt in cue["attempts"]):
                continue
            attempt: dict[str, Any] = {"round": round_number, "status": "started", "phase": "normalization",
                                       "text_sha256": cue["chosen_sha256"], "tts_attempts": 0,
                                       "started_unix": time.time(), "available_ms": cue["target"]["available_ms"]}
            cue["attempts"].append(attempt)
            persist(folder, state)  # Intent BEFORE normalization, model load, TTS or tempo.
            started = time.monotonic()
            directory = folder.resolve() / "audio" / cue["cue_id"] / f"round{number}"
            directory.mkdir(parents=True, exist_ok=True)
            inputs: list[str] = []
            try:
                if cue["target"]["status"] == "invalid" or attempt["available_ms"] <= 0:
                    attempt["status"] = "invalid_source"
                else:
                    normalized = engine.normalize({**cue["original"], "text": cue["chosen_text"]})
                    attempt["normalized_text"] = normalized
                    attempt["engine_input_text"] = normalize_vosk_text(normalized)
                    attempt.update(phase="tts", tts_attempts=1)
                    persist(folder, state)
                    engine.synthesize(normalized, directory / "raw.wav", inputs)
                    if inputs != [attempt["engine_input_text"]]:
                        raise ValueError("Engine input capture differs from bound normalization")
                    attempt["phase"] = "audio"
                    persist(folder, state)
                    if state["config"].get("silero_pauses", {}).get("enabled", False) and pause_processor is None:
                        from utils.silero_pauses import PauseProcessor
                        pause_processor = PauseProcessor()
                    measure_audio(state["config"], directory / "raw.wav", directory / "pretempo.wav",
                                  directory / "final.wav", attempt["available_ms"] / 1000, attempt, pause_processor,
                                  chosen_text=cue["chosen_text"], available_ms=attempt["available_ms"])
            except Exception as exc:
                attempt.update(status={"normalization": "normalization_failed", "tts": "tts_failed",
                                       "audio": "audio_failed"}[attempt["phase"]], error=str(exc))
            attempt["captured_engine_inputs"] = inputs
            attempt["elapsed_sec"] = time.monotonic() - started
            attempt["wav_sha256"] = {str(path): file_sha256(path) for path in directory.glob("*.wav")}
            state["audio_elapsed_sec"].append(attempt["elapsed_sec"])
            persist(folder, state)
        if round_number == 1 and "2" not in state["requests"]:
            unresolved = [cue for cue in state["cues"] if cue["attempts"][-1]["status"] in
                          {"unresolved", "normalization_failed"}]
            state["requests"]["2"] = make_request(folder, state, 2, unresolved)
            if not unresolved:
                state["imports"]["2"] = {"status": "no_targets"}
            persist(folder, state)
        if "2" in state["requests"]:
            save_json(folder / "request_round2.json", state["requests"]["2"])
        summary = status_summary(state)
        save_json(folder / "exceptions.json", summary["exceptions"] + summary["mix_blockers"])
        save_json(folder / "fits.json", {"ready_for_separate_mix": summary["ready_for_separate_mix"],
            "assembly_review_status": summary["assembly_review_status"], "mix_blockers": summary["mix_blockers"],
            "source_sha256": state["requests"]["1"]["source_sha256"], "entries": [
                {"cue_id": cue["cue_id"], "index": cue["original"]["index"],
                 "start": cue["original"]["start"], "end": cue["original"]["end"],
                 "available_ms": cue["target"]["available_ms"], "text": cue["chosen_text"],
                 "text_sha256": cue["chosen_sha256"], "wav": cue["attempts"][-1]["final"]}
                for cue in state["cues"] if cue["attempts"] and cue["attempts"][-1]["status"] in FIT]})
        publish_report(folder, state)
        return state


def status_summary(state: dict[str, Any]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    exceptions: list[dict[str, Any]] = []
    for cue in state["cues"]:
        last = cue["attempts"][-1] if cue["attempts"] else {"status": "pending_audio"}
        status = "needs_reconciliation" if last["status"] == "started" else last["status"]
        if (last.get("round") == 1 and any(decision["round"] == 2 and decision["reason"] == "applied"
                                          for decision in cue["decisions"])):
            status = "pending_repaired_audio"
        counts[status] += 1
        if status not in FIT:
            decision = cue["decisions"][-1] if cue["decisions"] else None
            reason = decision["reason"] if decision and decision["round"] == 2 and decision["reason"].startswith("repair_") else status
            exceptions.append({"cue_id": cue["cue_id"], "reason": reason, "audio_status": status,
                               "error": last.get("error"), "text_decision": decision})
    waiting = [number for number in state["requests"] if number not in state["imports"]]
    reviews = [value["result"]["assembly_review"] for _, value in sorted(state["imports"].items(), key=lambda pair: int(pair[0]))
               if "result" in value and "assembly_review" in value["result"]]
    assembly_status = "assembly_rejected_or_unreviewed"
    if reviews and reviews[-1]["verdict"] == "accept" and reviews[-1]["assembly_sha256"] == json_sha256(assembled(state)):
        assembly_status = "accepted_exact_final_assembly"
    elif not reviews and not state["proposals"] and not state["requests"]["1"]["entries"] and all(
            cue["chosen_text"] == join_text_lines(cue["original"].get("text", "")) for cue in state["cues"]):
        # Unchanged trusted source needs no new model grammar approval; never applies after a rejected review.
        assembly_status = "retained_originals_no_model_review_needed"
    blockers = [] if assembly_status != "assembly_rejected_or_unreviewed" else [{"reason": assembly_status}]
    for cue, following in zip(state["cues"], state["cues"][1:]):
        start, next_start = cue["original"].get("start"), following["original"].get("start")
        if type(start) is not int or type(next_start) is not int or next_start < start:
            blockers.append({"reason": "source_time_order_invalid", "cue_id": cue["cue_id"]})
        elif cue["attempts"] and cue["attempts"][-1]["status"] in FIT:
            final = cue["attempts"][-1]["final"]
            if start * final["sample_rate"] + final["frames"] * 1000 > next_start * final["sample_rate"]:
                blockers.append({"reason": "final_timeline_overlap", "cue_id": cue["cue_id"], "next_cue_id": following["cue_id"]})
    start, end = state["cues"][0]["original"].get("start"), state["cues"][-1]["original"].get("end")
    return {"cues": len(state["cues"]), "counts": dict(counts), "exceptions": exceptions,
            "native_targets": {number: [row["cue_id"] for row in request["entries"]]
                               for number, request in state["requests"].items()},
            "full_source_cues": state["requests"]["1"]["full_source_cues"],
            "source_sha256": state["requests"]["1"]["source_sha256"],
            "assembly_review_status": assembly_status, "mix_blockers": blockers,
            "ready_for_separate_mix": not exceptions and not waiting and not blockers,
            "waiting_native_rounds": waiting,
            "automatic_fit_fraction": sum(counts[name] for name in FIT) / len(state["cues"]),
            "tts_attempts": sum(attempt["tts_attempts"] for cue in state["cues"] for attempt in cue["attempts"]),
            "captured_engine_invocations": sum(len(attempt.get("captured_engine_inputs", []))
                                                for cue in state["cues"] for attempt in cue["attempts"]),
            "native_phases": state["native_phase_count"], "physical_model_requests": None,
            "cost_usd": None, "native_elapsed_sec": state["native_elapsed_sec"],
            "audio_elapsed_sec": sum(state["audio_elapsed_sec"]),
            "source_positions": state["range"],
            "continuous_duration_sec": (end - start) / 1000 if type(start) is int and type(end) is int else None}


def publish_report(folder: Path, state: dict[str, Any]) -> None:
    """Standalone escaped report only after actual native results were imported."""
    if not any(value.get("execution") == "native_live" for value in state["proposals"].values()) and not any(
            value.get("result", {}).get("execution") == "native_live" for value in state["imports"].values()):
        return
    escape = lambda value: html.escape(json.dumps(value, ensure_ascii=False, indent=2))
    summary = status_summary(state)
    labels = {"fits": "Уместилась", "accelerated_fits": "Уместилась после ускорения", "unresolved": "Не уместилась",
              "pending_audio": "Ожидает аудио", "needs_reconciliation": "Нужна сверка", "pending_repaired_audio": "Ожидает аудио ремонта",
              "normalization_failed": "Ошибка нормализации", "tts_failed": "Ошибка синтеза", "audio_failed": "Ошибка обработки аудио",
              "invalid_source": "Некорректный исходник", "repair_unchanged": "Ремонт без изменения текста пропущен",
              "repair_not_shorter": "Ремонт не короче текущего текста",
              "assembly_rejected_or_unreviewed": "Сборка отклонена или не проверена",
              "accepted_exact_final_assembly": "Окончательная сборка одобрена моделью",
              "retained_originals_no_model_review_needed": "Сохранён исходник; новая проверка не нужна"}
    fitted = sum(summary["counts"].get(name, 0) for name in FIT)
    repaired_tts = sum(attempt["tts_attempts"] for cue in state["cues"] for attempt in cue["attempts"] if attempt["round"] == 2)
    unchanged = sum(decision["reason"] == "repair_unchanged" for cue in state["cues"] for decision in cue["decisions"])
    sections = ["<h1>Автоматическая обработка реплик</h1><p>Проверка модели — не одобрение человека и не гарантия качества озвучки.</p>",
                f"<p><b>Уместились: {fitted}/{summary['cues']} ({summary['automatic_fit_fraction']:.1%})</b>. "
                f"Попыток синтеза: {summary['tts_attempts']}; повторных после ремонта: {repaired_tts}; "
                f"неизменённых ремонтов пропущено: {unchanged}. Этапов генерации/проверки: {summary['native_phases']}. "
                "Стоимость и число оплачиваемых запросов неизвестны без квитанций.</p>",
                f"<p><b>{'Готово к отдельному сведению' if summary['ready_for_separate_mix'] else 'Сведение и экспорт заблокированы'}</b>. "
                f"Проверка сборки: {html.escape(labels[summary['assembly_review_status']])}.</p>"]
    if summary["exceptions"] or summary["mix_blockers"]:
        sections.append(f"<details><summary>Причины блокировки и исключения</summary><pre>{escape(summary['mix_blockers'] + summary['exceptions'])}</pre></details>")
    sections.append("<table><thead><tr><th>Реплика</th><th>Выбранный текст</th><th>Статус</th><th>Аудио / окно, с</th></tr></thead><tbody>")
    for cue in state["cues"]:
        attempt = cue["attempts"][-1] if cue["attempts"] else {}
        status = "needs_reconciliation" if attempt.get("status") == "started" else attempt.get("status", "pending_audio")
        status = labels.get(status, status)
        duration = attempt.get("final", {}).get("duration_sec")
        duration_text = f"{duration:.3f}" if duration is not None else "—"
        decision = cue["decisions"][-1] if cue["decisions"] else None
        if decision and decision["reason"].startswith("repair_"):
            status += " / " + labels.get(decision["reason"], decision["reason"])
        sections.append(f"<tr><td>{html.escape(cue['cue_id'])}</td><td>{html.escape(cue['chosen_text'])}</td>"
                        f"<td>{html.escape(status)}</td><td>{duration_text} / {cue['target']['available_ms'] / 1000:.3f}</td></tr>")
    sections.append("</tbody></table>")
    sections.append(f"<details><summary>Предложения, модели, режимы и решения проверяющего</summary><pre>{escape({'proposals': state['proposals'], 'reviews': state['imports']})}</pre></details>")
    sections.append(f"<details><summary>Технические данные аудио и статистика</summary><pre>{escape({'summary': summary, 'attempts': {cue['cue_id']: cue['attempts'] for cue in state['cues']}})}</pre></details>")
    document = ('<!doctype html><html lang="ru"><meta charset="utf-8">'
                '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
                'style-src \'unsafe-inline\'; media-src data:"><title>Обработка реплик</title>'
                '<style>body{max-width:1200px;margin:2em auto;font:16px sans-serif}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
                'table{width:100%;border-collapse:collapse}td,th{border:1px solid #bbb;padding:.5em;text-align:left}details{margin:1em 0}</style>'
                + "".join(sections) + "</html>")
    temporary = folder / "report.html.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(folder / "report.html")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    if Path.cwd().resolve() != ROOT:
        parser.error(f"Run the batch CLI from repository root: {ROOT}")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Freeze full source and export one native batch; no inference")
    prepare_parser.add_argument("source", type=Path)
    prepare_parser.add_argument("--start", required=True, type=int, help="Inclusive zero-based original position")
    prepare_parser.add_argument("--end", required=True, type=int, help="Inclusive zero-based original position")
    prepare_parser.add_argument("--config", type=Path, default=ROOT / "pipeline_config_vosk.json")
    for command in (prepare_parser, commands.add_parser("review-input", help="Bind native writer proposals; export reviewer hashes"),
                    commands.add_parser("ingest-reviewed"), commands.add_parser("audio"),
                    commands.add_parser("status", help="Read-only; no inference or output rewriting")):
        command.add_argument("--output-dir", required=True, type=Path)
        if command.prog.endswith(("ingest-reviewed", "review-input")):
            command.add_argument("result", type=Path)
        if command.prog.endswith("review-input"):
            command.add_argument("--per-cue-reviewed", type=Path, help="Reassemble exact accepted subset before final assembled review")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if not args.config.is_file():
            parser.error("Config does not exist")
        state = prepare(args.source, args.output_dir, load_pipeline_config(args.config), args.start, args.end)
    elif args.command == "review-input":
        state = review_input(args.output_dir, args.result, args.per_cue_reviewed)
    elif args.command == "ingest-reviewed":
        state = ingest_reviewed(args.output_dir, args.result)
    elif args.command == "audio":
        state = audio(args.output_dir)
    else:
        with locked(args.output_dir, readonly=True):
            state = load_state(args.output_dir)
    print(json.dumps(status_summary(state), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
