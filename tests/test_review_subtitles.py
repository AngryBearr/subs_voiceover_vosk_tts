from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import pytest
from filelock import FileLock

from utils.apply_review import apply_review
from utils.llm_types import DetailedResponse
from utils.review_subtitles import main, reanalyze
from utils.shortening_plan import PlannerConfig, plan_subtitles
from utils.shortening_review import ReviewError
from utils.shortening_review_io import load_review_bundle
from utils.shortening_runner import ModelSpec, RunConfig, run_trial


def _items() -> list[dict[str, Any]]:
    return [
        {"index": 21, "start": 0, "end": 1000, "text": ["okay"], "unknown": {"keep": "first"}},
        {"index": 22, "start": 2000, "end": 3000, "text": ["a deliberately long subtitle cue"], "unknown": {"keep": "primary"}},
        {"index": 23, "start": 4000, "end": 5000, "text": ["fine"], "unknown": {"keep": "last"}},
    ]


def _run(tmp_path: Path, *, verdict: str = "revise") -> Path:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(items, planner)

    async def fake(model: ModelSpec, system: str, user: str, schema: dict[str, Any], allow_tool: bool) -> DetailedResponse:
        del model, system, user, allow_tool
        if "verdict" in json.dumps(schema):
            payload: dict[str, Any] = {"results": [{"cue_id": "cue-1", "verdict": verdict, "issues": [], "advice": ""}]}
        else:
            payload = {"results": [{"cue_id": "cue-1", "text": "tiny", "status": "ok"}]}
        return DetailedResponse(json.dumps(payload), [])

    output = tmp_path / "run"
    asyncio.run(run_trial(
        items=items,
        plans=plans,
        target_ids=["cue-1"],
        planner_config=planner,
        run_config=RunConfig(max_repairs=0),
        generator=ModelSpec("fake", "generator", "v"),
        arbiter=ModelSpec("fake", "arbiter", "v"),
        output_dir=output,
        request_fn=fake,
    ))
    return output


def test_cli_current_json_and_force_recheck_write_loadable_real_bundle(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, verdict="revise")
    original = load_review_bundle(run_dir)
    current = copy.deepcopy(original["state"]["current_items"])
    current[1]["text"] = ["tiny"]
    current_path = tmp_path / "current.json"
    current_path.write_text(json.dumps(current), encoding="utf-8")

    derived = tmp_path / "reanalysis"
    assert main([str(run_dir), "--output-dir", str(derived), "--current-json", str(current_path), "--force-recheck"]) == 0
    bundle = load_review_bundle(derived)
    assert bundle["state"]["current_items"] == current
    assert bundle["report"]["current_sha256"] == bundle["state"]["current_sha256"]
    assert bundle["report"]["provenance"]["force_recheck"] is True


def test_neighbor_timing_invalidates_cached_semantic_but_primary_json_is_exact(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, verdict="accept")
    state = load_review_bundle(run_dir)["state"]
    current = copy.deepcopy(state["current_items"])
    current[0]["end"] = 1200
    current_path = tmp_path / "neighbor.json"
    current_path.write_text(json.dumps(current), encoding="utf-8")

    result = reanalyze(run_dir, tmp_path / "neighbor-reanalysis", current_path)
    bundle = load_review_bundle(tmp_path / "neighbor-reanalysis")
    assert bundle["state"]["current_items"][1] == state["current_items"][1]
    primary = next(row for row in bundle["state"]["outcomes"] if row["cue_id"] == "cue-1")
    assert primary["reusable_checks"]["semantic"] is False
    assert result["unresolved_count"] >= 1


def test_external_fit_change_is_unresolved_and_grows_intervention_denominator(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, verdict="accept")
    state = load_review_bundle(run_dir)["state"]
    current = copy.deepcopy(state["current_items"])
    current[0]["text"] = ["fine"]
    output = tmp_path / "fit-change"
    reanalyze(run_dir, output, _write_json(tmp_path / "fit-change.json", current))
    changed = load_review_bundle(output)["state"]
    row = next(item for item in changed["outcomes"] if item["cue_id"] == "cue-0")
    assert row["outcome"] not in {"unchanged_and_fits", "accepted_correction"}
    assert changed["summary"]["intervention_count"] > state["summary"]["intervention_count"]


def test_force_recheck_invalidates_model_evidence_but_preserves_cheap_prediction(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, verdict="accept")
    state = load_review_bundle(run_dir)["state"]
    output = tmp_path / "forced"
    reanalyze(run_dir, output, force_recheck=True)
    forced = load_review_bundle(output)["state"]
    model_row = next(item for item in forced["outcomes"] if item["cue_id"] == "cue-1")
    fit_row = next(item for item in forced["outcomes"] if item["cue_id"] == "cue-0")
    assert model_row["reusable_checks"]["semantic"] is False
    assert fit_row["validation_source"] == "prediction"
    assert state["outcomes"][0]["validation_source"] == "prediction"


def _write_json(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_reanalysis_approval_applies_to_updated_queue_and_old_queue_is_stale(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, verdict="revise")
    state = load_review_bundle(run_dir)["state"]
    current = copy.deepcopy(state["current_items"])
    current[1]["text"] = ["tiny"]
    reanalysis = tmp_path / "updated"
    reanalyze(run_dir, reanalysis, _write_json(tmp_path / "updated.json", current))
    queue = load_review_bundle(reanalysis)["queue"]
    assert queue["entries"]
    edited = copy.deepcopy(queue)
    edited["entries"][0].update({"approved": True, "decision": "keep", "replacement_text": "tiny"})
    edited_path = _write_json(tmp_path / "approved.json", edited)
    applied = tmp_path / "applied"
    result = apply_review(reanalysis, edited_path, applied)
    assert result["text_changed_count"] == 0
    assert load_review_bundle(applied)["state"]["summary"]["unresolved_count"] == 0

    stale_path = _write_json(tmp_path / "stale.json", load_review_bundle(run_dir)["queue"])
    with pytest.raises(ReviewError):
        apply_review(reanalysis, stale_path, tmp_path / "stale-output")
    assert not (tmp_path / "stale-output").exists()


def test_public_reanalysis_api_rejects_held_source_lock(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, verdict="revise")
    lock = FileLock(str(run_dir / ".run.lock"), timeout=0)
    lock.acquire()
    try:
        with pytest.raises(ReviewError, match="locked"):
            reanalyze(run_dir, tmp_path / "blocked")
    finally:
        lock.release()
