from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from utils.shortening_plan import PlannerConfig, plan_subtitles
from utils.shortening_review import build_review_state, make_review_queue
from utils.shortening_review_io import load_review_bundle, write_derived_bundle
from utils.apply_review import apply_review


def _state() -> dict[str, Any]:
    source = [{"index": 3, "start": 0, "end": 2000, "text": ["a long source cue", "fragment"], "unknown": {"x": 1}}]
    config = PlannerConfig(cps=5, max_speed=1.0)
    return build_review_state(
        source_items=source, current_items=copy.deepcopy(source), plans=plan_subtitles(source, config),
        records=[], planner_config=config, metadata={"test": True}, selected_ids=["cue-0"],
    )


def test_derived_bundle_preserves_source_shape_and_has_no_runner_manifest(tmp_path: Path) -> None:
    state = _state()
    output = tmp_path / "derived"
    report = write_derived_bundle(output, state, {"parent_snapshot_id": state["snapshot_id"], "changed_cue_ids": []})
    assert report["snapshot_id"] == state["snapshot_id"]
    assert not (output / "run_manifest.json").exists()
    bundle = load_review_bundle(output)
    assert bundle["state"]["current_items"] == state["current_items"]
    assert make_review_queue(bundle["state"])["snapshot_id"] == report["snapshot_id"]


def test_apply_review_changes_only_approved_text(tmp_path: Path) -> None:
    state = _state()
    source_bundle = tmp_path / "source"
    write_derived_bundle(source_bundle, state, {"parent_snapshot_id": state["snapshot_id"], "changed_cue_ids": []})
    queue = json.loads((source_bundle / "review.json").read_text(encoding="utf-8"))
    entry = queue["entries"][0]
    entry.update({"approved": True, "decision": "replace", "replacement_text": "short"})
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(queue, ensure_ascii=False), encoding="utf-8")
    destination = tmp_path / "applied"
    result = apply_review(source_bundle, edited, destination)
    assert result["text_changed_count"] == 1
    exported = json.loads((destination / "subtitles.json").read_text(encoding="utf-8"))
    assert exported[0]["text"] == ["short"]
    assert exported[0]["unknown"] == {"x": 1}

import asyncio
import hashlib
import pytest
from filelock import FileLock
from utils.llm_types import DetailedResponse
from utils.shortening_review import ReviewError
from utils.shortening_runner import ModelSpec, RunConfig, run_trial
from utils.vosk_calibration import model_metadata, profile_identity


def _ordinary_bundle(root: Path) -> tuple[Path, Path, Path, Path]:
    model = root / "model"
    model.mkdir()
    (model / "model.onnx").write_bytes(b"fake-model")
    (model / "dictionary").write_text("fake-dictionary", encoding="utf-8")
    (model / "config.json").write_text(json.dumps({"num_speakers": 9}), encoding="utf-8")
    items = [
        {"index": 70, "start": 0, "end": 2000, "duration": 2000,
         "text": ["alpha beta gamma", "delta epsilon"], "unknown": {"keep": [1, 2]}},
        {"index": 9, "start": 2200, "end": 4200, "duration": 2000,
         "text": "one two three four five", "unknown": "untouched"},
        {"index": 900, "start": 4400, "end": 6400, "duration": 2000, "text": "fit"},
    ]
    source = root / "source.json"
    source.write_bytes(b"\xef\xbb\xbf" + json.dumps(items).encode())
    model_meta = model_metadata(model)
    identity = profile_identity(model_meta, 8, 1.5)
    profile = {"status": "ok", "rate": 1.5, "normalizer": identity["normalizer"],
               "input_character_basis": "normalized_tts_text", "identity": identity,
               "models": {"char_linear": {"name": "char_linear", "rate": 1.5,
                                          "coefficients": [0.0, 0.1]}},
               "char_linear_guard_sec": 0.0}
    profile_path = root / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    config = PlannerConfig(length_basis="normalized_tts_text", calibration_profile=profile,
                           calibration_identity=identity, max_speed=1.5)
    plans = plan_subtitles(items, config)
    selected = [p.cue_id for p in plans if p.status in {"needs_shortening", "short_timing"}]
    metadata = {
        "input": {"absolute_path": str(source), "path": str(source), "format": "json",
                  "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "calibration": {"absolute_path": str(profile_path), "path": str(profile_path),
                        "file_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                        "profile_sha256": __import__("utils.shortening_resume", fromlist=["canonical_hash"]).canonical_hash(profile),
                        "profile": profile, "identity": identity, "rate": 1.5},
        "model_path": str(model), "model_metadata": model_meta, "voice": 8, "rate": 1.5,
        "text_mode": "punctuation", "length_basis": "normalized_tts_text",
    }
    async def backend(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        targets = json.loads(user.split("Input JSON:\n", 1)[1])["targets"]
        if "verdict" in json.dumps(schema):
            rows = [{"cue_id": row["cue_id"], "verdict": "impossible", "issues": [], "advice": "Review manually"} for row in targets]
        else:
            rows = [{"cue_id": row["cue_id"], "status": "cannot_shorten", "text": row["original"]} for row in targets]
        return DetailedResponse(json.dumps({"results": rows}), [])
    output = root / "run"
    asyncio.run(run_trial(items=items, plans=plans, target_ids=selected, planner_config=config,
                         run_config=RunConfig(max_repairs=0), generator=ModelSpec("fixture", "g", "off"),
                         arbiter=ModelSpec("fixture", "a", "off"), output_dir=output,
                         request_fn=backend, report_metadata=metadata))
    return output, source, profile_path, model


def _edited_copy(run: Path, destination: Path) -> tuple[dict[str, Any], Path]:
    queue = json.loads((run / "review.json").read_text(encoding="utf-8"))
    queue["entries"][0].update(approved=True, decision="replace", replacement_text="short")
    destination.write_text(json.dumps(queue), encoding="utf-8")
    return queue, destination


def _files(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file() and p.name != ".run.lock"}


def test_apply_acceptance_real_run_preserves_other_cues_and_journals(tmp_path: Path) -> None:
    run, _, _, _ = _ordinary_bundle(tmp_path)
    queue, edit = _edited_copy(run, tmp_path / "edited.json")
    assert len(queue["entries"]) == 2
    queue["entries"][1]["replacement_text"] = "NOT APPROVED"
    edit.write_text(json.dumps(queue), encoding="utf-8")
    before_files = _files(run)
    before = json.loads((run / "subtitles.json").read_text())
    output = tmp_path / "applied"
    result = apply_review(run, edit, output)
    after = json.loads((output / "subtitles.json").read_text())
    expected = copy.deepcopy(before)
    expected[0]["text"] = ["short"]
    assert after == expected
    assert result["approved_count"] == 1 and result["text_changed_count"] == 1
    assert _files(run) == before_files
    bundle = load_review_bundle(output)
    marker = bundle["state"]["outcomes"][0]
    assert marker["validation_source"] == "human"
    assert marker["outcome"] == "accepted_correction"
    assert marker["measured_duration_sec"] is None and marker["measured_fit"] is None
    assert marker["reusable_checks"]["measured"] is False
    assert not (output / "run_manifest.json").exists()
    next_output = tmp_path / "second"
    apply_review(output, output / "review.json", next_output)
    again = load_review_bundle(next_output)["state"]
    assert again["current_items"] == after
    assert again["outcomes"][0]["validation_source"] == "human"
    assert again["outcomes"][0]["outcome"] == "accepted_correction"
    assert again["outcomes"][1]["category"] == bundle["state"]["outcomes"][1]["category"]


@pytest.mark.parametrize("damage", ["unknown", "duplicate", "stale", "budget", "context", "overlong", "newline", "defer", "resolved", "keep_mismatch"])
def test_apply_acceptance_rejects_whole_queue_before_output(tmp_path: Path, damage: str) -> None:
    run, _, _, _ = _ordinary_bundle(tmp_path)
    queue, edit = _edited_copy(run, tmp_path / "edited.json")
    row = queue["entries"][1]
    row.update(approved=True, decision="replace", replacement_text="short")
    if damage == "unknown":
        row["cue_id"] = "cue-999"
    elif damage == "duplicate":
        queue["entries"].append(copy.deepcopy(queue["entries"][0]))
    elif damage == "stale":
        queue["snapshot_id"] = "stale"
    elif damage == "budget":
        row["max_chars"] += 1
    elif damage == "context":
        row["context_before"] = []
    elif damage == "overlong":
        row["replacement_text"] = "x" * 10000
    elif damage == "newline":
        row["replacement_text"] = "one\ntwo"
    elif damage == "defer":
        row["decision"] = "defer"
    elif damage == "resolved":
        row["cue_id"] = "cue-2"
    else:
        row["decision"] = "keep"
        row["replacement_text"] = "different"
    edit.write_text(json.dumps(queue), encoding="utf-8")
    before = _files(run)
    output = tmp_path / "rejected"
    with pytest.raises(ReviewError):
        apply_review(run, edit, output)
    assert not output.exists()
    assert _files(run) == before


@pytest.mark.parametrize("damage", ["source", "profile", "model", "subtitles"])
def test_apply_acceptance_external_hash_mismatch_rejected(tmp_path: Path, damage: str) -> None:
    run, source, profile, model = _ordinary_bundle(tmp_path)
    _, edit = _edited_copy(run, tmp_path / "edited.json")
    if damage == "source":
        source.write_bytes(source.read_bytes() + b"\n")
    elif damage == "profile":
        profile.write_bytes(profile.read_bytes() + b"\n")
    elif damage == "model":
        (model / "dictionary").write_text("changed")
    else:
        current = json.loads((run / "subtitles.json").read_text())
        current[0]["text"] = "changed"
        (run / "subtitles.json").write_text(json.dumps(current))
    before = _files(run)
    output = tmp_path / "rejected"
    with pytest.raises(ReviewError):
        apply_review(run, edit, output)
    assert not output.exists()
    assert _files(run) == before


def test_apply_acceptance_lock_existing_and_nested_outputs(tmp_path: Path) -> None:
    run, _, _, _ = _ordinary_bundle(tmp_path)
    _, edit = _edited_copy(run, tmp_path / "edited.json")
    output = tmp_path / "applied"
    with FileLock(str(run / ".run.lock"), timeout=0):
        with pytest.raises(ReviewError, match="locked"):
            apply_review(run, edit, output)
    assert not output.exists()
    with pytest.raises(ReviewError):
        apply_review(run, edit, run / "nested")
    assert not (run / "nested").exists()
    apply_review(run, edit, output)
    before = _files(output)
    with pytest.raises(ReviewError):
        apply_review(run, edit, output)
    assert _files(output) == before


def test_apply_acceptance_keep_multifragment_list_no_collapse(tmp_path: Path) -> None:
    previous = _state()
    current = copy.deepcopy(previous["current_items"])
    current[0]["text"] = ["a", "b"]
    state = build_review_state(source_items=previous["source_items"], current_items=current,
                               plans=plan_subtitles(previous["source_items"], PlannerConfig(cps=5, max_speed=1.0)),
                               records=[], planner_config=PlannerConfig(cps=5, max_speed=1.0),
                               metadata=previous["metadata"], selected_ids=["cue-0"], previous_state=previous)
    source = tmp_path / "bundle"
    write_derived_bundle(source, state, {"parent_snapshot_id": previous["snapshot_id"]})
    queue = make_review_queue(state)
    queue["entries"][0].update(approved=True, decision="keep", replacement_text="a b")
    edited = tmp_path / "keep.json"
    edited.write_text(json.dumps(queue))
    output = tmp_path / "kept"
    result = apply_review(source, edited, output)
    assert result["approved_count"] == 1 and result["text_changed_count"] == 0
    assert json.loads((output / "subtitles.json").read_text())[0]["text"] == ["a", "b"]
