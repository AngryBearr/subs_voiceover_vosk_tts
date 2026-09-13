"""Durable IO and offline export helpers for subtitle review snapshots."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from utils.shortening_plan import CuePlan, PlannerConfig
from utils.shortening_resume import (
    RUNNER_ALGORITHM_VERSION,
    ResumeError,
    atomic_json,
    canonical_hash,
    load_baseline,
)
from utils.shortening_review import (
    ReviewError,
    build_review_state,
    make_review_queue,
    render_outcomes_markdown,
    render_review_markdown,
    validate_edit_queue,
    validate_review_state,
)


def _fail(message: str) -> None:
    raise ReviewError(message)


def _read(path: Path, label: str) -> Any:
    try:
        raw = path.read_bytes().decode("utf-8-sig")
        def reject_constant(value: str) -> Any:
            raise ValueError(f"non-finite JSON value {value} is not allowed")
        def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON object key {key!r}")
                result[key] = value
            return result
        return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=reject_duplicate)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReviewError(str(exc)) from exc


def _json_hash(value: Any) -> str:
    try:
        return canonical_hash(value)
    except ResumeError as exc:
        raise ReviewError(str(exc)) from exc


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        try:
            directory = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def _review_metadata(report: dict[str, Any]) -> dict[str, Any]:
    original = report.get("metadata")
    metadata = copy.deepcopy(original) if isinstance(original, dict) else {}
    metadata["review_runtime"] = {
        "models": copy.deepcopy(report.get("model")),
        "prompt_version": report.get("prompt_version"),
        "run_config": copy.deepcopy(report.get("config", {}).get("run")),
    }
    return metadata


def _manifest(output_dir: Path) -> dict[str, Any]:
    value = _read(output_dir / "run_manifest.json", "run_manifest.json")
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        _fail("invalid run manifest; preserve the directory")
    if value.get("runner_algorithm_version") != RUNNER_ALGORITHM_VERSION:
        _fail("run manifest algorithm version is incompatible")
    contract = value.get("contract")
    if not isinstance(contract, dict) or value.get("contract_hash") != _json_hash(contract):
        _fail("run manifest checksum is corrupt")
    return value


def _planner(report: dict[str, Any]) -> PlannerConfig:
    config = report.get("config")
    payload = config.get("planner") if isinstance(config, dict) else None
    if not isinstance(payload, dict):
        _fail("report planner configuration is missing")
    try:
        return PlannerConfig(**copy.deepcopy(payload))
    except (TypeError, ValueError) as exc:
        _fail(f"report planner configuration is invalid: {exc}")


def _plans(payload: Any) -> list[CuePlan]:
    if not isinstance(payload, list):
        _fail("resume baseline plans are missing")
    result: list[CuePlan] = []
    try:
        result = [CuePlan(**copy.deepcopy(item)) for item in payload if isinstance(item, dict)]
    except (TypeError, ValueError) as exc:
        _fail(f"resume baseline plans are invalid: {exc}")
    if len(result) != len(payload):
        _fail("resume baseline plans are malformed")
    return result


def state_from_run(output_dir: Path, report: dict[str, Any], current_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct the exact review state represented by a runner checkpoint."""
    output_dir = Path(output_dir)
    if not isinstance(report, dict) or not isinstance(current_items, list):
        _fail("checkpoint report/items are malformed")
    manifest = _manifest(output_dir)
    contract = manifest["contract"]
    source = contract.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("items"), list) or not isinstance(source.get("sha256"), str):
        _fail("run manifest source contract is malformed")
    if _json_hash(source["items"]) != source["sha256"]:
        _fail("run manifest source checksum is corrupt")
    try:
        baseline = load_baseline(output_dir)
    except ResumeError as exc:
        _fail(f"resume baseline is invalid: {exc}")
    if baseline is None:
        _fail("resume baseline is missing; preserve the directory")
    if baseline.get("source_hash") != source["sha256"]:
        _fail("resume baseline source does not match run manifest")
    config = report.get("config")
    if not isinstance(config, dict) or config.get("planner") != contract.get("planner") or config.get("run") != contract.get("run"):
        _fail("checkpoint planner does not match frozen run contract")
    source_manifest = report.get("source_manifest")
    if not isinstance(source_manifest, dict) or source_manifest.get("sha256") != source["sha256"]:
        _fail("checkpoint source manifest does not match frozen run contract")
    if report.get("prompt_version") != contract.get("prompt_version"):
        _fail("checkpoint prompt version does not match frozen run contract")
    expected_models = contract.get("models")
    if not isinstance(expected_models, dict):
        _fail("frozen model contract is malformed")
    if report.get("model") != {"generator": expected_models.get("generator"), "arbiter": expected_models.get("arbiter")}:
        _fail("checkpoint model does not match frozen run contract")
    expected_metadata = contract.get("report_metadata")
    if report.get("metadata") != expected_metadata:
        _fail("checkpoint report metadata does not match frozen run contract")
    plans = _plans(baseline.get("plans"))
    records = report.get("records")
    selected = report.get("selected_ids")
    if not isinstance(records, list) or not isinstance(selected, list) or any(not isinstance(item, str) for item in selected):
        _fail("checkpoint records or selected IDs are malformed")
    if baseline.get("selected_ids") != selected:
        _fail("checkpoint selected IDs do not match resume baseline")
    # Hashing is also the canonical-JSON validation step.  Comparing a value
    # with its deepcopy is tautological and cannot detect non-serializable data.
    _json_hash(current_items)
    try:
        planner = _planner(report)
        state = build_review_state(
            source_items=copy.deepcopy(source["items"]),
            current_items=copy.deepcopy(current_items),
            plans=plans,
            records=copy.deepcopy(records),
            planner_config=planner,
            metadata=_review_metadata(report),
            selected_ids=list(selected),
        )
    except ReviewError:
        raise
    except Exception as exc:
        _fail(f"checkpoint review state is invalid: {exc}")
    supplied_snapshot = report.get("review_snapshot_id")
    if supplied_snapshot is not None and supplied_snapshot != state["snapshot_id"]:
        _fail("checkpoint review snapshot does not match report")
    return state


def check_review_overwrite(output_dir: Path, replacement_state: dict[str, Any] | None = None) -> None:
    """Reject an edited queue before a resume can replace primary exports."""
    output_dir = Path(output_dir)
    review_path = output_dir / "review.json"
    state_path = output_dir / "review_state.json"
    if not review_path.exists():
        return
    current = _read(review_path, "review.json")
    if replacement_state is not None:
        try:
            validate_review_state(replacement_state)
            generated = make_review_queue(replacement_state)
        except ReviewError as exc:
            _fail(f"trusted replacement review state is invalid: {exc}")
        if _json_hash(current) == _json_hash(generated):
            # A crash may have published review.json immediately before the
            # binding state write.  The queue itself is trusted only because it
            # is an exact canonical queue for the replayed checkpoint state.
            atomic_json(state_path, replacement_state)
            return
    if not state_path.exists():
        _fail("review.json exists without review_state.json; preserve/copy the edited queue before resuming")
    saved = _read(state_path, "review_state.json")
    if not isinstance(saved, dict):
        _fail("review_state.json is corrupt; preserve/copy the edited queue before resuming")
    try:
        validate_review_state(saved)
        generated = make_review_queue(saved)
        validate_edit_queue(saved, current)
    except ReviewError as exc:
        _fail(f"edited review queue cannot be trusted; preserve/copy it before resuming: {exc}")
    if _json_hash(current) != _json_hash(generated):
        _fail("review.json contains edits; preserve/copy the edited queue before resuming")


def write_review_artifacts(output_dir: Path, state: dict[str, Any]) -> None:
    validate_review_state(state)
    output_dir = Path(output_dir)
    queue = make_review_queue(state)
    atomic_json(output_dir / "outcomes.json", {
        "schema_version": 1,
        "snapshot_id": state["snapshot_id"],
        "summary": copy.deepcopy(state["summary"]),
        "cues": copy.deepcopy(state["outcomes"]),
    })
    _atomic_text(output_dir / "outcomes.md", render_outcomes_markdown(state))
    atomic_json(output_dir / "review.json", queue)
    _atomic_text(output_dir / "review.md", render_review_markdown(state, queue))
    # This is the binding write: a complete state is published last.
    atomic_json(output_dir / "review_state.json", state)


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        _fail(f"cannot read claimed metadata file {path}: {exc}")


def _valid_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        _fail(f"{label} is malformed")
    try:
        int(value, 16)
    except ValueError:
        _fail(f"{label} is malformed")
    return value


def _claimed_path(metadata: dict[str, Any], section: dict[str, Any], key: str) -> Path | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        _fail(f"metadata {key} is malformed")
    return Path(value)


def _validate_external_metadata(state: dict[str, Any], source_path: Path | None) -> None:
    metadata = state.get("metadata")
    if metadata is None:
        return
    if not isinstance(metadata, dict):
        _fail("review metadata is malformed")
    input_meta = metadata.get("input")
    if input_meta is not None:
        if not isinstance(input_meta, dict) or not isinstance(input_meta.get("sha256"), str):
            _fail("claimed input metadata is malformed")
        claimed = source_path or _claimed_path(metadata, input_meta, "absolute_path") or _claimed_path(metadata, input_meta, "path")
        if claimed is None:
            _fail("claimed input metadata has no path")
        if _sha256_file(claimed) != input_meta["sha256"]:
            _fail("raw input hash does not match review metadata")
    calibration = metadata.get("calibration")
    if calibration is not None:
        if not isinstance(calibration, dict):
            _fail("claimed calibration metadata is malformed")
        profile = calibration.get("profile")
        identity = calibration.get("identity")
        config_profile = state.get("planner_config", {}).get("calibration_profile")
        config_identity = state.get("planner_config", {}).get("calibration_identity")
        claimed_path = _claimed_path(metadata, calibration, "absolute_path") or _claimed_path(metadata, calibration, "path")
        claims_profile = profile is not None
        claims_path = claimed_path is not None
        claims_hash = "file_sha256" in calibration or "profile_sha256" in calibration
        if calibration and (claims_profile or claims_path or claims_hash):
            file_hash = _valid_sha256(calibration.get("file_sha256"), "calibration file hash")
            profile_hash = _valid_sha256(calibration.get("profile_sha256"), "calibration profile hash")
            if claimed_path is None:
                _fail("claimed calibration metadata has no byte path")
            if _sha256_file(claimed_path) != file_hash:
                _fail("calibration profile file does not match review metadata")
            if profile is None or not isinstance(profile, dict):
                _fail("claimed calibration profile is malformed")
            if _json_hash(profile) != profile_hash:
                _fail("calibration profile hash does not match stored profile")
            if identity is None or not isinstance(identity, dict):
                _fail("ordinary calibration profile identity is missing")
        if profile is not None and _json_hash(profile) != _json_hash(config_profile):
            _fail("frozen calibration profile does not match planner metadata")
        if identity is not None and _json_hash(identity) != _json_hash(config_identity):
            _fail("frozen calibration identity does not match planner metadata")
    model_path = metadata.get("model_path")
    if model_path is not None:
        if not isinstance(model_path, str) or not model_path:
            _fail("model_path metadata is malformed")
        model_meta = metadata.get("model_metadata")
        if not isinstance(model_meta, dict):
            _fail("claimed model metadata is missing")
        try:
            from utils.vosk_calibration import model_metadata, profile_identity, validate_character_profile
            actual = model_metadata(Path(model_path))
            if _json_hash(actual) != _json_hash(model_meta):
                _fail("model metadata does not match frozen review metadata")
            voice = metadata.get("voice")
            rate = metadata.get("rate")
            calibration = metadata.get("calibration")
            if isinstance(voice, bool) or not isinstance(voice, int) or isinstance(rate, bool) or not isinstance(rate, (int, float)):
                _fail("model voice/rate metadata is malformed")
            actual_identity = profile_identity(actual, voice, float(rate))
            if isinstance(calibration, dict) and calibration.get("identity") is not None and _json_hash(actual_identity) != _json_hash(calibration["identity"]):
                _fail("model profile identity does not match frozen review metadata")
            if isinstance(calibration, dict) and calibration.get("profile") is not None:
                validate_character_profile(calibration["profile"], expected_identity=actual_identity)
        except ReviewError:
            raise
        except Exception as exc:
            _fail(f"claimed model metadata cannot be verified: {exc}")


def _validate_subtitles(state: dict[str, Any], directory: Path) -> None:
    path = directory / "subtitles.json"
    current = _read(path, "subtitles.json")
    if not isinstance(current, list) or any(not isinstance(item, dict) for item in current):
        _fail("subtitles.json must be an array of objects")
    if _json_hash(current) != state["current_sha256"]:
        _fail("subtitles.json does not match review state current snapshot")


def load_review_bundle(directory: Path, *, source_path: Path | None = None) -> dict[str, Any]:
    """Load and validate a normal or derived offline review bundle."""
    directory = Path(directory)
    state_value = _read(directory / "review_state.json", "review_state.json")
    if not isinstance(state_value, dict):
        _fail("review_state.json is malformed")
    try:
        validate_review_state(state_value)
    except ReviewError as exc:
        _fail(f"review_state.json is invalid: {exc}")
    _validate_external_metadata(state_value, source_path)
    _validate_subtitles(state_value, directory)
    queue = _read(directory / "review.json", "review.json")
    if not isinstance(queue, dict):
        _fail("review.json is malformed")
    try:
        validate_edit_queue(state_value, queue)
    except ReviewError as exc:
        _fail(f"review.json is invalid: {exc}")
    outcomes = _read(directory / "outcomes.json", "outcomes.json")
    expected_outcomes = {
        "schema_version": 1,
        "snapshot_id": state_value["snapshot_id"],
        "summary": state_value["summary"],
        "cues": state_value["outcomes"],
    }
    if outcomes != expected_outcomes:
        _fail("outcomes.json does not match review state")
    report: dict[str, Any] | None = None
    manifest_path = directory / "run_manifest.json"
    if manifest_path.exists():
        manifest = _manifest(directory)
        contract = manifest["contract"]
        if state_value["source_items"] != contract["source"]["items"]:
            _fail("review source differs from frozen run manifest")
        if state_value["planner_config"] != contract.get("planner"):
            _fail("review planner differs from frozen run manifest")
        report_path = directory / "report.json"
        report = _read(report_path, "report.json")
        if not isinstance(report, dict):
            _fail("report.json is malformed")
        if report.get("review_snapshot_id") != state_value["snapshot_id"]:
            _fail("report review snapshot does not match review state")
        if state_value.get("metadata") != _review_metadata(report):
            _fail("review metadata does not match report runtime metadata")
        try:
            reconstructed = state_from_run(directory, report, state_value["current_items"])
        except ReviewError as exc:
            _fail(f"review state cannot be reconstructed from run report: {exc}")
        if reconstructed["snapshot_id"] != state_value["snapshot_id"] or reconstructed != state_value:
            _fail("review state does not match immutable run report and checkpoint selectors")
    else:
        derived = []
        for name in ("application_report.json", "reanalysis_report.json"):
            if (directory / name).exists():
                derived.append(name)
        if len(derived) != 1:
            _fail("bundle is neither a resumable run nor a derived export")
        report_value = _read(directory / derived[0], derived[0])
        if not isinstance(report_value, dict) or report_value.get("snapshot_id") != state_value["snapshot_id"]:
            _fail("derived report does not bind to review state")
        if report_value.get("source_sha256") != state_value["source_sha256"] or report_value.get("current_sha256") != state_value["current_sha256"]:
            _fail("derived report checksums do not match review state")
        report = report_value
    return {"state": state_value, "queue": queue, "outcomes": outcomes, "report": report}


def write_derived_bundle(output_dir: Path, state: dict[str, Any], provenance: dict[str, Any], *, kind: str = "application") -> dict[str, Any]:
    if kind not in {"application", "reanalysis"}:
        _fail("derived bundle kind is invalid")
    validate_review_state(state)
    output_dir = Path(output_dir)
    if output_dir.exists():
        _fail(f"output directory already exists: {output_dir}")
    if not isinstance(provenance, dict):
        _fail("derived provenance is malformed")
    output_dir.mkdir(parents=True)
    try:
        atomic_json(output_dir / "subtitles.json", state["current_items"])
        from utils.shortening_runner import _export_srt
        export_error = _export_srt(copy.deepcopy(state["current_items"]), output_dir)
        write_review_artifacts(output_dir, state)
        report = {
            "schema_version": 1,
            "parent_snapshot_id": provenance.get("parent_snapshot_id"),
            "snapshot_id": state["snapshot_id"],
            "source_sha256": state["source_sha256"],
            "current_sha256": state["current_sha256"],
            "explicit_approved": copy.deepcopy(provenance.get("explicit_approved", [])),
            "changed_cue_ids": copy.deepcopy(provenance.get("changed_cue_ids", [])),
            "summary": copy.deepcopy(state["summary"]),
            "export_error": export_error,
            "evidence_warning": "Human/model evidence is not audio measurement; no LLM or TTS was run by this export.",
            "provenance": copy.deepcopy(provenance),
        }
        atomic_json(output_dir / f"{kind}_report.json", report)
        return report
    except Exception:
        # The destination is intentionally not treated as a resumable run.  A
        # partially written directory is retained for forensic inspection.
        raise


__all__ = [
    "check_review_overwrite", "load_review_bundle", "state_from_run",
    "write_derived_bundle",
]
