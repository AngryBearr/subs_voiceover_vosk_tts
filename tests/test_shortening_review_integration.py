from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from utils.llm_types import DetailedResponse
from utils.shortening_plan import PlannerConfig, plan_subtitles
from utils.shortening_review import ReviewError
from utils.shortening_review_io import load_review_bundle
from utils.shortening_runner import ModelSpec, RunConfig, run_trial


def _items() -> list[dict[str, Any]]:
    return [
        {"index": 10, "start": 0, "end": 1000, "text": ["short"], "analysis": {"legacy": True}},
        {"index": 11, "start": 2000, "end": 3000, "text": ["a long subtitle cue that needs shortening"], "analysis": {"legacy": True}},
    ]


def test_runner_writes_full_outcomes_and_review_queue(tmp_path: Path) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(items, planner)
    selected = [plan.cue_id for plan in plans if plan.status == "needs_shortening"]

    async def fake(*_: Any) -> DetailedResponse:
        return DetailedResponse(json.dumps({"results": []}), [])

    output = tmp_path / "run"
    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=selected, planner_config=planner,
        run_config=RunConfig(max_repairs=0), generator=ModelSpec("fake", "model", "v"),
        arbiter=ModelSpec("fake", "model", "v"), output_dir=output, request_fn=fake,
    ))
    assert report["review_snapshot_id"]
    assert len(report["outcomes"]) == len(items)
    bundle = load_review_bundle(output)
    assert bundle["state"]["snapshot_id"] == report["review_snapshot_id"]
    assert (output / "outcomes.md").is_file()
    assert (output / "review.md").is_file()


def test_resume_refuses_edited_queue_before_replacing_exports(tmp_path: Path) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(items, planner)

    async def fake(*_: Any) -> DetailedResponse:
        return DetailedResponse(json.dumps({"results": []}), [])

    output = tmp_path / "run"
    asyncio.run(run_trial(
        items=items, plans=plans, target_ids=[], planner_config=planner,
        run_config=RunConfig(max_repairs=0), generator=ModelSpec("fake", "model", "v"),
        arbiter=ModelSpec("fake", "model", "v"), output_dir=output, request_fn=fake,
    ))
    queue_path = output / "review.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["snapshot_id"] = "edited"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    report_bytes = (output / "report.json").read_bytes()
    try:
        asyncio.run(run_trial(
            items=items, plans=plans, target_ids=[], planner_config=planner,
            run_config=RunConfig(max_repairs=0), generator=ModelSpec("fake", "model", "v"),
            arbiter=ModelSpec("fake", "model", "v"), output_dir=output, request_fn=fake, resume=True,
        ))
    except ReviewError:
        pass
    else:
        raise AssertionError("edited review queue should block resume")
    assert (output / "report.json").read_bytes() == report_bytes


def test_resume_repairs_interrupted_review_state_without_repeating_receipt(tmp_path: Path, monkeypatch: Any) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(items, planner)
    calls = 0

    async def fake(*_: Any) -> DetailedResponse:
        nonlocal calls
        calls += 1
        return DetailedResponse(json.dumps({"results": []}), [])

    import utils.shortening_review_io as review_io

    original_atomic_json = review_io.atomic_json
    review_writes = 0

    def crash_after_review(path: Path, value: Any) -> None:
        nonlocal review_writes
        original_atomic_json(path, value)
        if path.name == "review.json":
            review_writes += 1
            if review_writes == 2:
                raise RuntimeError("injected crash after review queue publication")

    monkeypatch.setattr(review_io, "atomic_json", crash_after_review)
    output = tmp_path / "run"
    try:
        asyncio.run(run_trial(
            items=items, plans=plans, target_ids=["cue-1"], planner_config=planner,
            run_config=RunConfig(max_repairs=0), generator=ModelSpec("fake", "model", "v"),
            arbiter=ModelSpec("fake", "model", "v"), output_dir=output, request_fn=fake,
        ))
    except RuntimeError:
        pass
    else:
        raise AssertionError("the injected checkpoint crash was not reached")
    assert calls == 1

    monkeypatch.setattr(review_io, "atomic_json", original_atomic_json)
    resumed = asyncio.run(run_trial(
        items=copy.deepcopy(items), plans=plans, target_ids=["cue-1"], planner_config=planner,
        run_config=RunConfig(max_repairs=0), generator=ModelSpec("fake", "model", "v"),
        arbiter=ModelSpec("fake", "model", "v"), output_dir=output, request_fn=fake, resume=True,
    ))
    assert resumed["run_state"] == "completed"
    assert calls == 1
    assert load_review_bundle(output)["state"]["snapshot_id"] == resumed["review_snapshot_id"]


def test_interrupted_checkpoint_never_repairs_a_changed_queue(tmp_path: Path, monkeypatch: Any) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(items, planner)

    async def fake(*_: Any) -> DetailedResponse:
        return DetailedResponse(json.dumps({"results": []}), [])

    import utils.shortening_review_io as review_io

    original_atomic_json = review_io.atomic_json

    def crash_after_first_review(path: Path, value: Any) -> None:
        original_atomic_json(path, value)
        if path.name == "review.json":
            raise RuntimeError("injected crash after review queue publication")

    monkeypatch.setattr(review_io, "atomic_json", crash_after_first_review)
    output = tmp_path / "run"
    try:
        asyncio.run(run_trial(
            items=items, plans=plans, target_ids=[], planner_config=planner,
            run_config=RunConfig(max_repairs=0), generator=ModelSpec("fake", "model", "v"),
            arbiter=ModelSpec("fake", "model", "v"), output_dir=output, request_fn=fake,
        ))
    except RuntimeError:
        pass
    else:
        raise AssertionError("the injected checkpoint crash was not reached")
    queue_path = output / "review.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["snapshot_id"] = "user-edited"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    monkeypatch.setattr(review_io, "atomic_json", original_atomic_json)
    with pytest.raises(ReviewError):
        asyncio.run(run_trial(
            items=items, plans=plans, target_ids=[], planner_config=planner,
            run_config=RunConfig(max_repairs=0), generator=ModelSpec("fake", "model", "v"),
            arbiter=ModelSpec("fake", "model", "v"), output_dir=output, request_fn=fake, resume=True,
        ))
