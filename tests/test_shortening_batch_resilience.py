"""Focused SH-02 tests for row salvage and bounded recovery rounds."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from utils.llm_types import CallRecord, DetailedResponse
from utils.opencode_transport import OpenCodePromptUsage
from utils.shortening_plan import PlannerConfig, plan_subtitles
from utils.shortening_prompts import (
    parse_generation,
    parse_generation_partial,
    parse_review,
    parse_review_partial,
)
from utils.shortening_runner import ModelSpec, RunConfig, run_trial


def _items(count: int = 6) -> list[dict[str, Any]]:
    return [{"index": 100 + i, "start": i * 3000, "end": i * 3000 + 1000,
             "text": ["alpha beta gamma delta"], "analysis": {"keep": i}}
            for i in range(count)]


def _row_generation(cue_id: str, text: str = "short") -> dict[str, Any]:
    return {"cue_id": cue_id, "text": text, "status": "ok"}


def _row_review(cue_id: str, verdict: str = "accept") -> dict[str, Any]:
    return {"cue_id": cue_id, "verdict": verdict,
             "issues": [] if verdict == "accept" else ["meaning"], "advice": "fix"}


def _call(spec: ModelSpec, *, status: str = "success", usage: OpenCodePromptUsage | None = None) -> CallRecord:
    """Build a deterministic usage-bearing fake call for runner accounting tests."""
    return CallRecord(spec.provider, spec.model, spec.variant, "2026-09-10T00:00:00Z", 0.1,
                      usage or OpenCodePromptUsage(input_tokens=10, output_tokens=3), status)


def _request_rows(output_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (output_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
            if json.loads(line)["state"] == "start"]


def _request_stages(output_dir: Path) -> list[tuple[str, list[str]]]:
    return [(row["stage"], row["target_ids"]) for row in _request_rows(output_dir)]


def test_partial_parsers_salvage_rows_and_reject_ambiguous_or_foreign_data() -> None:
    expected = ["a", "b", "c"]
    payload = {"results": [
        _row_generation("a"),
        {"cue_id": "b", "text": "bad", "status": []},
        {"cue_id": "b", "text": "conflict", "status": "ok"},
        {"cue_id": "foreign", "text": "wrong", "status": "ok"},
        {"text": "no id", "status": "ok"},
        12,
    ]}
    parsed = parse_generation_partial(json.dumps(payload), expected)
    assert list(parsed.results) == ["a"]
    assert set(parsed.errors) == {"b", "c"}
    assert len(parsed.response_issues) >= 3

    review = parse_review_partial(json.dumps({"results": [
        _row_review("a"), {"cue_id": "b", "verdict": {"bad": []}, "issues": [], "advice": ""},
    ]}), expected)
    assert list(review.results) == ["a"]
    assert set(review.errors) == {"b", "c"}

    duplicate = '{"results":[{"cue_id":"a","text":"x","status":"ok"}],"results":[]}'
    for parser in (parse_generation, parse_review):
        try:
            parser(duplicate, ["a"])
        except ValueError as exc:
            assert "duplicate JSON object key" in str(exc)
        else:
            raise AssertionError("duplicate object keys must be rejected")


def test_malformed_envelopes_and_expected_id_validation() -> None:
    parsed = parse_review_partial("not json", ["a", "b"])
    assert parsed.results == {}
    assert list(parsed.errors) == ["a", "b"]
    assert parsed.response_issues
    for expected in (["a", "a"], ["a", 1]):
        try:
            parse_generation_partial('{"results":[]}', expected)  # type: ignore[list-item]
        except ValueError:
            pass
        else:
            raise AssertionError("invalid expected IDs must fail upfront")


def test_six_row_generation_error_preserves_five_neighbors_without_retry(tmp_path: Path) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    ids = [plan.cue_id for plan in plans]
    requested: list[tuple[str, list[str]]] = []

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        target_ids = [row["cue_id"] for row in payload["targets"]]
        is_review = "verdict" in json.dumps(schema)
        requested.append(("review" if is_review else "generation", target_ids))
        if is_review:
            result = [_row_review(cue_id) for cue_id in target_ids]
        else:
            result = [_row_generation(cue_id) for cue_id in target_ids if cue_id != "cue-2"]
        return DetailedResponse(json.dumps({"results": result}), [])

    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=RunConfig(gen_batch_size=6, max_repairs=0),
        generator=ModelSpec("deepseek", "m", "off"), arbiter=ModelSpec("openai", "m", "off"),
        output_dir=tmp_path / "max0", request_fn=fake,
    ))
    records = {record["cue_id"]: record for record in report["records"]}
    assert sum(record["status"] == "accepted" for record in records.values()) == 5
    assert records["cue-2"]["status"] == "error"
    assert records["cue-2"]["failure"] == "schema"
    assert records["cue-2"]["final_text"] == records["cue-2"]["original"]
    assert requested == [("generation", ids), ("review", [cue_id for cue_id in ids if cue_id != "cue-2"])]


def test_generation_and_review_recovery_target_only_unresolved_rows(tmp_path: Path) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    ids = [plan.cue_id for plan in plans]
    calls: list[tuple[str, list[str]]] = []
    invocation = 0

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        nonlocal invocation
        invocation += 1
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        target_ids = [row["cue_id"] for row in payload["targets"]]
        is_review = "verdict" in json.dumps(schema)
        if not is_review:
            stage = "repair" if "prior_candidate" in user else "generation"
            calls.append((stage, target_ids))
            result = [_row_generation(cue_id) for cue_id in target_ids if invocation != 1 or cue_id != "cue-2"]
        else:
            stage = "review" if invocation == 2 else "recheck"
            calls.append((stage, target_ids))
            result = [_row_review(cue_id) for cue_id in target_ids if invocation != 2 or cue_id != "cue-3"]
        return DetailedResponse(json.dumps({"results": result}), [])

    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=RunConfig(gen_batch_size=6, max_repairs=1),
        generator=ModelSpec("deepseek", "m", "off"), arbiter=ModelSpec("openai", "m", "off"),
        output_dir=tmp_path / "recover", request_fn=fake,
    ))
    assert calls == [
        ("generation", ids),
        ("review", [cue_id for cue_id in ids if cue_id != "cue-2"]),
        ("repair", ["cue-2"]),
        ("recheck", ["cue-2", "cue-3"]),
    ]
    assert all(record["status"] == "accepted" for record in report["records"])
    assert all(record["initial_accepted"] is False for record in report["records"] if record["cue_id"] in {"cue-2", "cue-3"})
    assert report["summary"]["request_counts"] == {"generation": 1, "review": 1, "repair": 1, "recheck": 1}


@pytest.mark.parametrize("malformed_kind", ["invalid", "missing", "duplicate"])
def test_six_row_generation_partial_salvage_exports_five_and_preserves_source(
    tmp_path: Path, malformed_kind: str,
) -> None:
    items = _items()
    source = copy.deepcopy(items)
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    ids = [plan.cue_id for plan in plans]
    bad_id = "cue-2"

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        target_ids = [row["cue_id"] for row in payload["targets"]]
        if "verdict" in json.dumps(schema):
            result = [_row_review(cue_id) for cue_id in target_ids]
        else:
            result = [_row_generation(cue_id) for cue_id in target_ids if cue_id != bad_id]
            if malformed_kind == "invalid":
                result.append({"cue_id": bad_id, "text": "short", "status": []})
            elif malformed_kind == "duplicate":
                result.extend([_row_generation(bad_id, "first"), _row_generation(bad_id, "second")])
        return DetailedResponse(json.dumps({"results": result}), [_call(spec)])

    output = tmp_path / malformed_kind
    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=RunConfig(gen_batch_size=6, max_repairs=0),
        generator=ModelSpec("deepseek", "m", "off"), arbiter=ModelSpec("openai", "m", "off"),
        output_dir=output, request_fn=fake,
    ))
    records = {record["cue_id"]: record for record in report["records"]}
    assert sum(record["status"] == "accepted" for record in records.values()) == 5
    assert records[bad_id]["status"] == "error"
    assert records[bad_id]["failure"] == "schema"
    assert records[bad_id]["final_text"] == records[bad_id]["original"]
    assert items == source

    saved_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    saved_subtitles = json.loads((output / "subtitles.json").read_text(encoding="utf-8"))
    assert saved_report["records"] == report["records"]
    assert len(saved_subtitles) == len(source)
    for before, after in zip(source, saved_subtitles):
        for key in ("index", "start", "end"):
            assert after[key] == before[key]
        assert set(after) == set(before)
        assert after["analysis"]["keep"] == before["analysis"]["keep"]
    assert saved_subtitles[0]["text"] == ["short"]
    assert saved_subtitles[2]["text"] == source[2]["text"]


def test_runner_salvages_safe_rows_while_isolating_foreign_missing_and_nonobject_extras(
    tmp_path: Path,
) -> None:
    items = _items()
    source = copy.deepcopy(items)
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    ids = [plan.cue_id for plan in plans]

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        target_ids = [row["cue_id"] for row in payload["targets"]]
        if "verdict" in json.dumps(schema):
            safe = [_row_review(cue_id) for cue_id in target_ids]
        else:
            safe = [_row_generation(cue_id) for cue_id in target_ids]
        safe.extend([{"cue_id": "foreign", "text": "must not apply", "status": "ok"},
                     {"text": "missing id", "status": "ok"}, 42])
        return DetailedResponse(json.dumps({"results": safe}), [_call(spec)])

    output = tmp_path / "safe-extras"
    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=RunConfig(max_repairs=0),
        generator=ModelSpec("deepseek", "m", "off"), arbiter=ModelSpec("openai", "m", "off"),
        output_dir=output, request_fn=fake,
    ))
    assert all(record["status"] == "accepted" for record in report["records"])
    assert items == source
    for record in report["records"]:
        assert all(event["response_issues"] for event in record["history"])
    saved = json.loads((output / "subtitles.json").read_text(encoding="utf-8"))
    assert all(item["text"] != ["must not apply"] for item in saved)
    assert [item["text"] for item in saved] == [["short"]] * len(saved)


@pytest.mark.parametrize("review_malformed_kind", ["invalid", "missing"])
@pytest.mark.parametrize("max_repairs", [0, 1])
def test_six_row_review_partial_salvage_rechecks_only_failed_id_and_preserves_candidate(
    tmp_path: Path, max_repairs: int, review_malformed_kind: str,
) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    ids = [plan.cue_id for plan in plans]
    bad_id = "cue-2"
    checkpoint_before_retry: dict[str, Any] | None = None

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        nonlocal checkpoint_before_retry
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        target_ids = [row["cue_id"] for row in payload["targets"]]
        is_review = "verdict" in json.dumps(schema)
        if not is_review:
            result = [_row_generation(cue_id) for cue_id in target_ids]
        elif target_ids == [bad_id]:
            checkpoint_before_retry = json.loads((tmp_path / f"{review_malformed_kind}-{max_repairs}" / "report.json").read_text(encoding="utf-8"))
            result = [_row_review(bad_id)]
        else:
            result = [_row_review(cue_id) for cue_id in target_ids if cue_id != bad_id]
            if review_malformed_kind == "invalid":
                result.append({"cue_id": bad_id, "verdict": [], "issues": [], "advice": ""})
        return DetailedResponse(json.dumps({"results": result}), [_call(spec)])

    output = tmp_path / f"{review_malformed_kind}-{max_repairs}"
    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=RunConfig(gen_batch_size=6, review_batch_size=6, max_repairs=max_repairs),
        generator=ModelSpec("deepseek", "m", "off"), arbiter=ModelSpec("openai", "m", "off"),
        output_dir=output, request_fn=fake,
    ))
    assert _request_stages(output) == [
        ("generation", ids),
        ("review", ids),
    ] + ([("recheck", [bad_id])] if max_repairs else [])
    records = {record["cue_id"]: record for record in report["records"]}
    assert records[bad_id]["candidate"] == "short"
    if max_repairs:
        assert checkpoint_before_retry is not None
        checkpoint_records = {record["cue_id"]: record for record in checkpoint_before_retry["records"]}
        assert sum(record["status"] == "accepted" for record in checkpoint_records.values()) == 5
        assert all(record["candidate"] == "short" for record in checkpoint_records.values())
        assert records[bad_id]["status"] == "accepted"
    else:
        assert records[bad_id]["status"] == "error"
        assert records[bad_id]["failure"] == "schema"
    assert report["summary"]["request_counts"] == {
        "generation": 1, "review": 1, "repair": 0, "recheck": 1 if max_repairs else 0,
    }


def test_generation_only_recovery_repairs_and_rechecks_bad_id_only(tmp_path: Path) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    ids = [plan.cue_id for plan in plans]
    bad_id = "cue-2"
    generation_attempts = 0

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        nonlocal generation_attempts
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        target_ids = [row["cue_id"] for row in payload["targets"]]
        if "verdict" in json.dumps(schema):
            result = [_row_review(cue_id) for cue_id in target_ids]
        else:
            generation_attempts += 1
            result = [_row_generation(cue_id) for cue_id in target_ids
                      if generation_attempts > 1 or cue_id != bad_id]
        return DetailedResponse(json.dumps({"results": result}), [_call(spec)])

    output = tmp_path / "generation-only"
    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=RunConfig(gen_batch_size=6, max_repairs=1),
        generator=ModelSpec("deepseek", "m", "off"), arbiter=ModelSpec("openai", "m", "off"),
        output_dir=output, request_fn=fake,
    ))
    assert _request_stages(output) == [
        ("generation", ids), ("review", [cue_id for cue_id in ids if cue_id != bad_id]),
        ("repair", [bad_id]), ("recheck", [bad_id]),
    ]
    assert all(record["status"] == "accepted" for record in report["records"])
    assert all(bad_id not in target_ids for stage, target_ids in _request_stages(output)
               if stage in {"review", "recheck"} and target_ids != [bad_id])


def test_mixed_repair_rows_retry_only_malformed_generation_and_keep_neighbor(tmp_path: Path) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    ids = [plan.cue_id for plan in plans]
    revised = {"cue-2", "cue-5"}
    malformed = "cue-5"
    repair_calls = 0

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        nonlocal repair_calls
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        target_ids = [row["cue_id"] for row in payload["targets"]]
        if "verdict" in json.dumps(schema):
            if target_ids == ids:
                result = [_row_review(cue_id, "revise" if cue_id in revised else "accept") for cue_id in target_ids]
            else:
                result = [_row_review(cue_id) for cue_id in target_ids]
        else:
            repair_calls += "prior_candidate" in user
            if repair_calls == 1:
                result = [_row_generation(cue_id) for cue_id in target_ids if cue_id != malformed]
                result.append({"cue_id": malformed, "text": "short", "status": []})
            else:
                result = [_row_generation(cue_id) for cue_id in target_ids]
        return DetailedResponse(json.dumps({"results": result}), [_call(spec)])

    output = tmp_path / "mixed-repair"
    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=RunConfig(gen_batch_size=6, review_batch_size=6, max_repairs=2),
        generator=ModelSpec("deepseek", "m", "off"), arbiter=ModelSpec("openai", "m", "off"),
        output_dir=output, request_fn=fake,
    ))
    assert _request_stages(output) == [
        ("generation", ids), ("review", ids),
        ("repair", sorted(revised)), ("recheck", ["cue-2"]),
        ("repair", [malformed]), ("recheck", [malformed]),
    ]
    records = {record["cue_id"]: record for record in report["records"]}
    assert records["cue-2"]["status"] == "accepted"
    assert records[malformed]["status"] == "accepted"
    assert all(records[cue_id]["status"] == "accepted" for cue_id in ids)


def test_recheck_malformed_verdict_retries_same_candidate_without_generation(tmp_path: Path) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    ids = [plan.cue_id for plan in plans]
    revised = {"cue-1", "cue-4"}
    malformed = "cue-1"
    recheck_candidates: list[tuple[list[str], list[str]]] = []

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        target_ids = [row["cue_id"] for row in payload["targets"]]
        if "verdict" not in json.dumps(schema):
            result = [_row_generation(cue_id) for cue_id in target_ids]
        elif target_ids == ids:
            result = [_row_review(cue_id, "revise" if cue_id in revised else "accept") for cue_id in target_ids]
        else:
            recheck_candidates.append((target_ids, [row["candidate"] for row in payload["targets"]]))
            result = [{"cue_id": cue_id, "verdict": [], "issues": [], "advice": ""}
                      if cue_id == malformed else _row_review(cue_id) for cue_id in target_ids]
        return DetailedResponse(json.dumps({"results": result}), [_call(spec)])

    output = tmp_path / "recheck-mixed"
    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=RunConfig(gen_batch_size=6, review_batch_size=6, max_repairs=2),
        generator=ModelSpec("deepseek", "m", "off"), arbiter=ModelSpec("openai", "m", "off"),
        output_dir=output, request_fn=fake,
    ))
    assert _request_stages(output) == [
        ("generation", ids), ("review", ids),
        ("repair", sorted(revised)), ("recheck", sorted(revised)),
        ("recheck", [malformed]),
    ]
    assert recheck_candidates == [(["cue-1", "cue-4"], ["short", "short"]), ([malformed], ["short"])]
    records = {record["cue_id"]: record for record in report["records"]}
    assert records["cue-4"]["status"] == "accepted"
    assert records[malformed]["status"] == "error"
    assert records[malformed]["failure"] == "schema"
    assert not any(stage == "repair" and malformed in target_ids
                   for stage, target_ids in _request_stages(output)[4:])


@pytest.mark.parametrize("max_repairs", [0, 1, 2])
def test_persistent_malformed_generation_has_bounded_calls_and_no_reviews(
    tmp_path: Path, max_repairs: int,
) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    target = ["cue-2"]

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        assert "verdict" not in json.dumps(schema)
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        return DetailedResponse(json.dumps({"results": [
            {"cue_id": payload["targets"][0]["cue_id"], "text": "short", "status": []},
        ]}), [_call(spec)])

    output = tmp_path / f"persistent-generation-{max_repairs}"
    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=target, planner_config=planner,
        run_config=RunConfig(max_repairs=max_repairs),
        generator=ModelSpec("deepseek", "m", "off"), arbiter=ModelSpec("openai", "m", "off"),
        output_dir=output, request_fn=fake,
    ))
    assert _request_stages(output) == [("generation", target)] + [("repair", target)] * max_repairs
    record = report["records"][0]
    assert record["status"] == "error"
    assert record["failure"] == "schema"
    assert len(record["history"]) == 1 + max_repairs
    assert report["summary"]["request_counts"] == {
        "generation": 1, "review": 0, "repair": max_repairs, "recheck": 0,
    }


@pytest.mark.parametrize("max_repairs", [0, 1, 2])
def test_persistent_malformed_review_has_bounded_judge_calls_and_schema_failure(
    tmp_path: Path, max_repairs: int,
) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    target = ["cue-2"]

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        if "verdict" not in json.dumps(schema):
            result: dict[str, Any] = _row_generation(target[0])
        else:
            result = {"cue_id": target[0], "verdict": [], "issues": [], "advice": ""}
        return DetailedResponse(json.dumps({"results": [result]}), [_call(spec)])

    output = tmp_path / f"persistent-review-{max_repairs}"
    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=target, planner_config=planner,
        run_config=RunConfig(max_repairs=max_repairs),
        generator=ModelSpec("deepseek", "m", "off"), arbiter=ModelSpec("openai", "m", "off"),
        output_dir=output, request_fn=fake,
    ))
    assert _request_stages(output) == [("generation", target), ("review", target)] + [("recheck", target)] * max_repairs
    record = report["records"][0]
    assert record["status"] == "error"
    assert record["failure"] == "schema"
    assert len(record["history"]) == 2 + max_repairs
    assert report["summary"]["request_counts"] == {
        "generation": 1, "review": 1, "repair": 0, "recheck": max_repairs,
    }


def test_malformed_initial_review_then_revise_is_manual_without_generation_repair(tmp_path: Path) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    target = ["cue-2"]
    judge_calls = 0

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        nonlocal judge_calls
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        if "verdict" not in json.dumps(schema):
            result: dict[str, Any] = _row_generation(target[0])
        else:
            judge_calls += 1
            result = ({"cue_id": target[0], "verdict": [], "issues": [], "advice": ""}
                      if judge_calls == 1 else _row_review(target[0], "revise"))
        return DetailedResponse(json.dumps({"results": [result]}), [_call(spec)])

    output = tmp_path / "malformed-then-revise"
    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=target, planner_config=planner,
        run_config=RunConfig(max_repairs=1),
        generator=ModelSpec("deepseek", "m", "off"), arbiter=ModelSpec("openai", "m", "off"),
        output_dir=output, request_fn=fake,
    ))
    assert _request_stages(output) == [("generation", target), ("review", target), ("recheck", target)]
    record = report["records"][0]
    assert record["status"] == "manual"
    assert record["failure"] in {"semantic", "timing"}
    assert report["summary"]["request_counts"]["repair"] == 0


def test_initial_impossible_is_terminal_even_with_two_repair_rounds(tmp_path: Path) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    target = ["cue-2"]

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        if "verdict" not in json.dumps(schema):
            result: dict[str, Any] = _row_generation(target[0])
        else:
            result = _row_review(target[0], "impossible")
        return DetailedResponse(json.dumps({"results": [result]}), [_call(spec)])

    output = tmp_path / "impossible"
    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=target, planner_config=planner,
        run_config=RunConfig(max_repairs=2),
        generator=ModelSpec("deepseek", "m", "off"), arbiter=ModelSpec("openai", "m", "off"),
        output_dir=output, request_fn=fake,
    ))
    assert _request_stages(output) == [("generation", target), ("review", target)]
    assert report["records"][0]["status"] == "manual"
    assert report["records"][0]["failure"] == "semantic"


@pytest.mark.parametrize(
    ("max_repairs", "persistent", "expected_stages"),
    [
        (0, False, [("generation", ["cue-2"])]),
        (1, False, [("generation", ["cue-2"]), ("repair", ["cue-2"]), ("recheck", ["cue-2"])]),
        (2, True, [("generation", ["cue-2"]), ("repair", ["cue-2"]), ("repair", ["cue-2"])]),
    ],
)
def test_protocol_misuse_cannot_shorten_changed_text_is_never_semantic_and_never_applied(
    tmp_path: Path, max_repairs: int, persistent: bool, expected_stages: list[tuple[str, list[str]]],
) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    target = ["cue-2"]
    repair_feedback: list[dict[str, Any]] = []
    repair_prior_candidates: list[str] = []
    generation_calls = 0

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        nonlocal generation_calls
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        if "verdict" in json.dumps(schema):
            result: dict[str, Any] = _row_review(target[0])
        else:
            generation_calls += 1
            if "prior_candidate" in user:
                row = payload["targets"][0]
                repair_feedback.append(json.loads(row["feedback"]))
                repair_prior_candidates.append(row["prior_candidate"])
            if not persistent and generation_calls > 1:
                result = _row_generation(target[0], "short")
            else:
                result = {"cue_id": target[0], "text": "invented replacement", "status": "cannot_shorten"}
        return DetailedResponse(json.dumps({"results": [result]}), [_call(spec)])

    output = tmp_path / f"protocol-{max_repairs}-{persistent}"
    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=target, planner_config=planner,
        run_config=RunConfig(max_repairs=max_repairs),
        generator=ModelSpec("deepseek", "m", "off"), arbiter=ModelSpec("openai", "m", "off"),
        output_dir=output, request_fn=fake,
    ))
    assert _request_stages(output) == expected_stages
    record = report["records"][0]
    assert all(event.get("candidate") != "invented replacement" for event in record["history"])
    if max_repairs == 1:
        assert record["status"] == "accepted"
        assert record["final_text"] == "short"
        assert repair_feedback[0]["failure"]["class"] == "protocol_misuse"
        assert repair_feedback[0]["failure"]["error"] == "cannot_shorten returned changed text"
        assert repair_prior_candidates == [""]
    else:
        assert record["final_text"] == record["original"]
        assert record["status"] == "manual"
        assert record["failure"] == "protocol_misuse"
        assert record["final_review"] is None


def test_honest_initial_cannot_shorten_is_reviewed_with_original_text(tmp_path: Path) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    target = ["cue-2"]
    review_candidates: list[str] = []

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        if "verdict" in json.dumps(schema):
            review_candidates.extend(row["candidate"] for row in payload["targets"])
            result: dict[str, Any] = _row_review(target[0], "impossible")
        else:
            result = {"cue_id": target[0], "text": plans[2].text, "status": "cannot_shorten"}
        return DetailedResponse(json.dumps({"results": [result]}), [_call(spec)])

    output = tmp_path / "honest-cannot"
    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=target, planner_config=planner,
        run_config=RunConfig(max_repairs=0),
        generator=ModelSpec("deepseek", "m", "off"), arbiter=ModelSpec("openai", "m", "off"),
        output_dir=output, request_fn=fake,
    ))
    assert _request_stages(output) == [("generation", target), ("review", target)]
    assert review_candidates == [plans[2].text]
    assert report["records"][0]["failure"] == "semantic"


def test_honest_repair_cannot_shorten_clears_old_review_and_skips_recheck(tmp_path: Path) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    target = ["cue-2"]
    generation_calls = 0

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        nonlocal generation_calls
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        if "verdict" in json.dumps(schema):
            result: dict[str, Any] = _row_review(target[0], "revise")
        else:
            generation_calls += 1
            status = "ok" if generation_calls == 1 else "cannot_shorten"
            result = {"cue_id": target[0], "text": "short" if status == "ok" else plans[2].text, "status": status}
        return DetailedResponse(json.dumps({"results": [result]}), [_call(spec)])

    output = tmp_path / "honest-repair-cannot"
    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=target, planner_config=planner,
        run_config=RunConfig(max_repairs=2),
        generator=ModelSpec("deepseek", "m", "off"), arbiter=ModelSpec("openai", "m", "off"),
        output_dir=output, request_fn=fake,
    ))
    assert _request_stages(output) == [
        ("generation", target), ("review", target), ("repair", target),
    ]
    record = report["records"][0]
    assert record["status"] == "manual"
    assert record["failure"] == "cannot_shorten"
    assert record["final_review"] is None
    assert [event["stage"] for event in record["history"]] == ["generation", "review", "repair"]


def test_repair_local_invalid_candidate_clears_stale_review_but_keeps_history(tmp_path: Path) -> None:
    items = _items()
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    target = ["cue-2"]
    generation_calls = 0

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        nonlocal generation_calls
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        if "verdict" in json.dumps(schema):
            result: dict[str, Any] = _row_review(target[0], "revise")
        else:
            generation_calls += 1
            text = "short" if generation_calls == 1 else " bad\ntext "
            result = _row_generation(target[0], text)
        return DetailedResponse(json.dumps({"results": [result]}), [_call(spec)])

    output = tmp_path / "repair-invalid-local"
    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=target, planner_config=planner,
        run_config=RunConfig(max_repairs=1),
        generator=ModelSpec("deepseek", "m", "off"), arbiter=ModelSpec("openai", "m", "off"),
        output_dir=output, request_fn=fake,
    ))
    record = report["records"][0]
    assert _request_stages(output) == [("generation", target), ("review", target), ("repair", target)]
    assert record["status"] == "error"
    assert record["failure"] == "schema"
    assert record["candidate"] == " bad\ntext "
    assert record["final_review"] is None
    assert any(event.get("stage") == "review" and "judgment" in event for event in record["history"])
