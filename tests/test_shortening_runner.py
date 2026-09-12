from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from utils.llm_types import CallRecord, DetailedResponse
from utils.opencode_transport import OpenCodePromptUsage
from utils.shortening_plan import PlannerConfig, plan_subtitles
from utils.shortening_costs import summarize_costs
from utils.shortening_runner import ModelSpec, RunConfig, _record_call, run_trial


def _items(count: int = 7) -> list[dict[str, Any]]:
    return [{"index": 99 if i < 2 else i, "start": i * 3000, "end": i * 3000 + 1000,
             "text": ["alpha beta gamma delta"], "analysis": {"legacy": i}} for i in range(count)]


def test_batched_acceptance_and_original_metadata(tmp_path: Path) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    ids = [plan.cue_id for plan in plans if plan.status == "needs_shortening"]
    calls: list[tuple[ModelSpec, bool, list[str]]] = []

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        target_ids = [row["cue_id"] for row in payload["targets"]]
        calls.append((spec, tools, target_ids))
        is_review = "verdict" in json.dumps(schema)
        if is_review:
            text = json.dumps({"results": [{"cue_id": cue_id, "verdict": "accept", "issues": [], "advice": ""} for cue_id in target_ids]})
        else:
            text = json.dumps({"results": [{"cue_id": cue_id, "text": "short", "status": "ok"} for cue_id in target_ids]})
        usage = OpenCodePromptUsage(input_tokens=10, output_tokens=3)
        return DetailedResponse(text, [CallRecord(spec.provider, spec.model, spec.variant, "now", 0.1, usage, "success")])

    report = asyncio.run(run_trial(items=items, plans=plans, target_ids=ids, planner_config=planner,
                                   run_config=RunConfig(gen_batch_size=6, review_batch_size=12),
                                   generator=ModelSpec("deepseek", "deepseek-v4-flash", "off"),
                                   arbiter=ModelSpec("opencode", "openai/gpt-5.6-sol", "medium"),
                                   output_dir=tmp_path / "trial", request_fn=fake))
    assert report["summary"]["accepted_count"] == len(ids)
    assert sum(1 for spec, _, _ in calls if spec.provider == "deepseek") == 2
    assert sum(1 for spec, _, _ in calls if spec.provider == "opencode") == 1
    assert all(not tools for spec, tools, _ in calls if spec.provider == "opencode")
    exported = json.loads((tmp_path / "trial" / "subtitles.json").read_text(encoding="utf-8"))
    assert exported[0]["index"] == items[0]["index"]
    assert exported[0]["start"] == items[0]["start"]


def test_output_directory_must_be_new(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    try:
        asyncio.run(run_trial(items=[], plans=[], target_ids=[], planner_config=PlannerConfig(), run_config=RunConfig(),
                              generator=ModelSpec("deepseek", "deepseek-v4-flash", "off"),
                              arbiter=ModelSpec("opencode", "sol", "medium"), output_dir=output,
                              request_fn=lambda *_: None))
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output should fail")


def _run_config(tmp_path: Path, *, max_repairs: int = 1, review_batch_size: int = 12) -> tuple[list[dict[str, Any]], list[Any], PlannerConfig, list[str]]:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    ids = [plan.cue_id for plan in plans if plan.status == "needs_shortening"]
    return items, plans, planner, ids


def test_short_timing_is_denominator_only(tmp_path: Path) -> None:
    items = _items()
    items[0]["text"] = ["verylongword unusualword"]
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    short_id = next(plan.cue_id for plan in plans if plan.status == "short_timing")
    long_id = next(plan.cue_id for plan in plans if plan.status == "needs_shortening")
    calls: list[list[str]] = []

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        ids = [row["cue_id"] for row in payload["targets"]]
        calls.append(ids)
        text = json.dumps({"results": [{"cue_id": cue_id, "text": "short", "status": "ok"} for cue_id in ids]})
        if "verdict" in json.dumps(schema):
            text = json.dumps({"results": [{"cue_id": cue_id, "verdict": "accept", "issues": [], "advice": ""} for cue_id in ids]})
        return DetailedResponse(text, [])

    report = asyncio.run(run_trial(items=items, plans=plans, target_ids=[short_id, long_id], planner_config=planner,
                                   run_config=RunConfig(), generator=ModelSpec("deepseek", "m", "off"),
                                   arbiter=ModelSpec("opencode", "a", "medium"), output_dir=tmp_path / "short", request_fn=fake))
    assert short_id not in [cue_id for batch in calls for cue_id in batch]
    assert report["summary"]["selected_count"] == 2
    assert report["summary"]["short_timing_count"] == 1


def test_cannot_shorten_revision_gets_repair(tmp_path: Path) -> None:
    items, plans, planner, ids = _run_config(tmp_path)
    target = ids[0]
    stages: list[str] = []
    generation_count = 0
    review_count = 0

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        nonlocal generation_count, review_count
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        target_ids = [row["cue_id"] for row in payload["targets"]]
        is_review = "verdict" in json.dumps(schema)
        if is_review:
            review_count += 1
            stages.append("recheck" if review_count > 1 else "review")
            verdict = "accept" if review_count > 1 else "revise"
            text = json.dumps({"results": [{"cue_id": cue_id, "verdict": verdict, "issues": [] if verdict == "accept" else ["shorten"], "advice": "fix"} for cue_id in target_ids]})
        else:
            generation_count += 1
            stages.append("repair" if "prior_candidate" in user else "generation")
            status = "ok" if generation_count > 1 else "cannot_shorten"
            text = json.dumps({"results": [{"cue_id": cue_id, "text": "short" if status == "ok" else next(row["original"] for row in payload["targets"] if row["cue_id"] == cue_id), "status": status} for cue_id in target_ids]})
        return DetailedResponse(text, [])

    report = asyncio.run(run_trial(items=items, plans=plans, target_ids=[target], planner_config=planner,
                                   run_config=RunConfig(max_repairs=1), generator=ModelSpec("deepseek", "m", "off"),
                                   arbiter=ModelSpec("opencode", "a", "medium"), output_dir=tmp_path / "cannot", request_fn=fake))
    assert stages == ["generation", "review", "repair", "recheck"]
    assert report["records"][0]["status"] == "accepted"


def test_repair_batches_only_pending_latest_judgment(tmp_path: Path) -> None:
    items, plans, planner, ids = _run_config(tmp_path)
    calls: list[tuple[str, list[str]]] = []
    review_count = 0

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        target_ids = [row["cue_id"] for row in payload["targets"]]
        nonlocal review_count
        is_review = "verdict" in json.dumps(schema)
        if not is_review:
            stage = "repair" if "prior_candidate" in user else "generation"
            calls.append((stage, target_ids))
            text = json.dumps({"results": [{"cue_id": cue_id, "text": "short", "status": "ok"} for cue_id in target_ids]})
        else:
            review_count += 1
            stage = "recheck" if review_count > 1 else "review"
            calls.append((stage, target_ids))
            text = json.dumps({"results": [{"cue_id": cue_id, "verdict": "revise" if cue_id == ids[0] else "accept",
                                               "issues": ["meaning"] if cue_id == ids[0] else [], "advice": "latest"} for cue_id in target_ids]})
        return DetailedResponse(text, [])

    report = asyncio.run(run_trial(items=items, plans=plans, target_ids=ids, planner_config=planner,
                                   run_config=RunConfig(gen_batch_size=6, review_batch_size=12, max_repairs=1),
                                   generator=ModelSpec("deepseek", "m", "off"), arbiter=ModelSpec("opencode", "a", "medium"),
                                   output_dir=tmp_path / "pending", request_fn=fake))
    assert sum(stage == "generation" for stage, _ in calls) == 2
    assert sum(stage == "review" for stage, _ in calls) == 1
    assert sum(stage == "repair" for stage, _ in calls) == 1
    assert sum(stage == "recheck" for stage, _ in calls) == 1
    downstream = [cue_id for stage, batch in calls if stage in {"repair", "recheck"} for cue_id in batch]
    assert set(downstream) == {ids[0]}
    assert report["records"][0]["status"] == "manual"
    assert all(report["records"][index]["status"] == "accepted" for index in range(1, len(ids)))


def test_invalid_generation_and_whitespace_fail_closed(tmp_path: Path) -> None:
    items, plans, planner, ids = _run_config(tmp_path)

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        target_ids = [row["cue_id"] for row in payload["targets"]]
        if "verdict" in json.dumps(schema):
            text = json.dumps({"results": [{"cue_id": cue_id, "verdict": "accept", "issues": [], "advice": ""} for cue_id in target_ids]})
        else:
            text = json.dumps({"results": [{"cue_id": cue_id, "text": " bad\ntext ", "status": "ok"} for cue_id in target_ids]})
        return DetailedResponse(text, [])

    report = asyncio.run(run_trial(items=items, plans=plans, target_ids=[ids[0]], planner_config=planner,
                                   run_config=RunConfig(), generator=ModelSpec("deepseek", "m", "off"),
                                   arbiter=ModelSpec("opencode", "a", "medium"), output_dir=tmp_path / "invalid", request_fn=fake))
    assert report["records"][0]["status"] == "error"
    assert report["records"][0]["final_text"] == report["records"][0]["original"]


def test_request_exception_marks_accounting_unknown_and_preserves_source(tmp_path: Path) -> None:
    items, plans, planner, ids = _run_config(tmp_path)
    source = json.loads(json.dumps(items))

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        raise RuntimeError("unknown transport outcome")

    report = asyncio.run(run_trial(items=items, plans=plans, target_ids=[ids[0]], planner_config=planner,
                                   run_config=RunConfig(), generator=ModelSpec("deepseek", "m", "off"),
                                   arbiter=ModelSpec("opencode", "a", "medium"), output_dir=tmp_path / "unknown", request_fn=fake))
    assert items == source
    assert report["run_state"] == "incomplete"
    assert report["summary"]["accounting_complete"] is False
    assert report["summary"]["request_failures"][0]["stage"] == "generation"
    assert report["cost"]["direct"]["call_count"] == 0
    assert report["cost"]["direct"]["known_subtotal_usd"] == 0.0
    assert report["cost"]["direct"]["known_subtotal_is_partial"] is True
    assert report["cost"]["projection_500"]["unknown_cost"] is True
    assert report["cost"]["projection_500"]["applicable"] is False
    assert "incomplete_run" in report["cost"]["projection_500"]["inapplicable_reasons"]


def test_known_deepseek_cost_survives_unknown_openai_accounting(tmp_path: Path) -> None:
    items, plans, planner, ids = _run_config(tmp_path)
    usage = OpenCodePromptUsage(input_tokens=45_532, output_tokens=0, cache_read_tokens=80)

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        target_ids = [row["cue_id"] for row in payload["targets"]]
        if spec.provider == "deepseek":
            text = json.dumps({"results": [{"cue_id": cue_id, "text": "short", "status": "ok"} for cue_id in target_ids]})
            call = CallRecord("deepseek", "m", "off", "2026-09-07T04:00:00Z", 0.1, usage, "success")
        else:
            text = json.dumps({"results": [{"cue_id": cue_id, "verdict": "accept", "issues": [], "advice": ""} for cue_id in target_ids]})
            call = CallRecord("openai", "gpt", "medium", "2026-09-07T04:00:00Z", 0.1, None, "success")
        return DetailedResponse(text, [call])

    report = asyncio.run(run_trial(items=items, plans=plans, target_ids=[ids[0]], planner_config=planner,
                                   run_config=RunConfig(), generator=ModelSpec("deepseek", "m", "off"),
                                   arbiter=ModelSpec("openai", "gpt", "medium"), output_dir=tmp_path / "partial",
                                   request_fn=fake))
    assert report["cost"]["direct"]["known_subtotal_usd"] == 0.01
    assert report["cost"]["direct"]["accounting_complete"] is False
    assert report["cost"]["direct"]["direct_paid_accounting_complete"] is True
    assert report["cost"]["direct"]["known_subtotal_is_partial"] is True
    assert report["cost"]["projection_500"]["unknown_cost"] is True


def test_projection_is_inapplicable_for_model_error_even_with_known_usage(tmp_path: Path) -> None:
    items, plans, planner, ids = _run_config(tmp_path)
    usage = OpenCodePromptUsage(input_tokens=10, output_tokens=3)

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        target_ids = [row["cue_id"] for row in payload["targets"]]
        call = CallRecord("deepseek", "deepseek-v4-flash", "off", "2026-09-07T04:00:00Z", 0.1, usage, "error")
        if "verdict" in json.dumps(schema):
            return DetailedResponse(None, [call], "review failed")
        text = json.dumps({"results": [{"cue_id": cue_id, "text": "short", "status": "ok"} for cue_id in target_ids]})
        return DetailedResponse(text, [call])

    report = asyncio.run(run_trial(items=items, plans=plans, target_ids=[ids[0]], planner_config=planner,
                                   run_config=RunConfig(max_repairs=0), generator=ModelSpec("deepseek", "m", "off"),
                                   arbiter=ModelSpec("deepseek", "m", "off"), output_dir=tmp_path / "model-error",
                                   request_fn=fake))
    projection = report["cost"]["projection_500"]
    assert report["summary"]["error_count"] == 1
    assert projection["applicable"] is False
    assert "model_or_protocol_errors" in projection["inapplicable_reasons"]
    assert "missing_usage" not in projection["inapplicable_reasons"]


def test_projection_is_applicable_for_reviewed_manual_semantic_attempt(tmp_path: Path) -> None:
    items, plans, planner, ids = _run_config(tmp_path)
    usage = OpenCodePromptUsage(input_tokens=10, output_tokens=3)

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        target_ids = [row["cue_id"] for row in payload["targets"]]
        call = CallRecord("deepseek", "deepseek-v4-flash", "off", "2026-09-07T04:00:00Z", 0.1, usage, "success")
        if "verdict" in json.dumps(schema):
            text = json.dumps({"results": [{"cue_id": cue_id, "verdict": "impossible", "issues": ["meaning"], "advice": ""}
                                             for cue_id in target_ids]})
        else:
            text = json.dumps({"results": [{"cue_id": cue_id, "text": "short", "status": "ok"} for cue_id in target_ids]})
        return DetailedResponse(text, [call])

    report = asyncio.run(run_trial(items=items, plans=plans, target_ids=[ids[0]], planner_config=planner,
                                   run_config=RunConfig(max_repairs=0), generator=ModelSpec("deepseek", "m", "off"),
                                   arbiter=ModelSpec("deepseek", "m", "off"), output_dir=tmp_path / "manual-semantic",
                                   request_fn=fake))
    projection = report["cost"]["projection_500"]
    assert report["records"][0]["status"] == "manual"
    assert report["records"][0]["failure"] == "semantic"
    assert projection["applicable"] is True
    assert projection["inapplicable_reasons"] == []


def test_projection_is_inapplicable_without_a_reviewed_candidate(tmp_path: Path) -> None:
    items = _items()
    items[0]["text"] = ["verylongword unusualword"]
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    short_id = next(plan.cue_id for plan in plans if plan.status == "short_timing")

    report = asyncio.run(run_trial(items=items, plans=plans, target_ids=[short_id], planner_config=planner,
                                   run_config=RunConfig(), generator=ModelSpec("deepseek", "m", "off"),
                                   arbiter=ModelSpec("deepseek", "m", "off"), output_dir=tmp_path / "no-review",
                                   request_fn=lambda *_: None))
    projection = report["cost"]["projection_500"]
    assert report["run_state"] == "completed"
    assert projection["applicable"] is False
    assert "no_reviewed_candidates" in projection["inapplicable_reasons"]


def test_persisted_cost_estimates_match_primary_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "ledger"
    output_dir.mkdir()
    timestamp = "2026-09-07T04:00:00Z"
    deepseek = CallRecord(
        "deepseek", "deepseek-v4-flash", "off", timestamp, 0.1,
        OpenCodePromptUsage(input_tokens=1_000_000, output_tokens=0), "success",
    )
    openai = CallRecord(
        "openai", "gpt-5.6-luna", "medium", timestamp, 0.1,
        OpenCodePromptUsage(input_tokens=10, output_tokens=4), "success",
    )
    missing = CallRecord("openai", "gpt-5.6-luna", "medium", timestamp, 0.1, None, "error")
    records = [deepseek, openai, missing]
    for index, call in enumerate(records):
        _record_call(call, stage="test", target_ids=[str(index)], model=ModelSpec(call.provider, call.model, call.variant or "off"),
                     output_dir=output_dir, request_id=str(index))

    rows = [json.loads(line) for line in (output_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    estimates = [row["cost_estimate"] for row in rows]
    primary = summarize_costs(records)
    assert sum(row["cost_usd"] or 0.0 for row in estimates) == primary["known_subtotal_usd"]
    assert estimates[1]["unknown"] is False
    assert estimates[1]["cost_usd"] == 0.0
    assert estimates[1]["direct_cost_usd"] == 0.0
    assert estimates[2]["unknown"] is True
    assert estimates[2]["cost_usd"] is None
