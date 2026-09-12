from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from utils.llm_types import CallRecord, DetailedResponse
from utils.opencode_transport import OpenCodePromptUsage
from utils.shortening_experiment import (
    EXPECTED_PLANNER,
    _audit_status,
    _build_effects,
    _ledger_summary,
    _planner_manifest,
    _run_audits,
    _run_candidates,
    _validate_and_load,
    run_experiment,
)
from utils.shortening_plan import PlannerConfig, plan_subtitles
from utils.shortening_prompts import PROMPT_VERSION
from utils.shortening_runner import ModelSpec


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    items = [{"index": index, "start": index * 1400, "end": index * 1400 + 1000, "text": "This is a deliberately long cue for testing." + str(index), "unknown": {"x": index}} for index in range(2)]
    source_dir = tmp_path / "corpus" / "sources"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "source.json"
    source_path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    plans = plan_subtitles(items, PlannerConfig())
    cases = []
    for plan in plans:
        cases.append({"case_id": "s:" + plan.cue_id, "source_id": "s", "cue_id": plan.cue_id,
                      "position": plan.position, "source_index": plan.position, "original": plan.text,
                      "max_chars": plan.max_chars, "available_ms": plan.available_ms, "severity": plan.severity,
                      "context": [{"cue_id": "cue-" + str(i), "text": row["text"], "start": row["start"], "end": row["end"]} for i, row in enumerate(items)]})
    manifest = {"schema_version": 1, "planner_config": {"cps": 13.0, "max_speed": 1.5, "gap_ms": 200, "context_before": 3, "context_after": 3},
                "sources": [{"source_id": "s", "json_path": "sources/source.json", "split": "test", "parsed_sha256": _hash(items), "sha256": "unused"}],
                "cases": cases, "audit_case_ids": [], "crosscheck_case_ids": [], "order_check_case_ids": []}
    return manifest, tmp_path / "corpus"


def test_experiment_shares_standard_generation_with_a_and_calls_c_once_per_stage(tmp_path: Path) -> None:
    manifest, manifest_dir = _fixture(tmp_path)
    calls: list[tuple[str, ModelSpec]] = []

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        ids = [row.get("cue_id") for row in payload["targets"]]
        is_candidates = "candidates" in schema["properties"]["results"]["items"]["properties"] and "selected_candidate_id" not in schema["properties"]["results"]["items"]["properties"]
        is_selection = "selected_candidate_id" in schema["properties"]["results"]["items"]["properties"]
        stage = "candidates" if is_candidates else "selection" if is_selection else "review" if "verdict" in schema["properties"]["results"]["items"]["properties"] else "generation"
        calls.append((stage, spec))
        if is_candidates:
            result = {"results": [{"cue_id": cue_id, "status": "ok", "candidates": [{"candidate_id": "one", "text": "short"}, {"candidate_id": "two", "text": "short two"}]} for cue_id in ids]}
        elif is_selection:
            options = payload["targets"]
            result = {"results": [{"cue_id": row["cue_id"], "selected_candidate_id": row["candidates"][0]["candidate_id"], "issues": [], "advice": ""} for row in options]}
        elif stage == "review":
            result = {"results": [{"cue_id": cue_id, "verdict": "accept", "issues": [], "advice": ""} for cue_id in ids]}
        else:
            result = {"results": [{"cue_id": cue_id, "text": "short", "status": "ok"} for cue_id in ids]}
        usage = OpenCodePromptUsage(input_tokens=10, output_tokens=2)
        return DetailedResponse(json.dumps(result), [CallRecord(spec.provider, spec.model, spec.variant, "2026-09-10T00:00:00Z", 0.1, usage, "success")])

    output = tmp_path / "experiment"
    result = asyncio.run(run_experiment(manifest, manifest_dir, output, generator=ModelSpec("deepseek", "deepseek-flash", "low"), repetitions=2, best_of=5, request_fn=fake))
    assert result["status"] == "completed"
    assert len(result["repetitions"]) == 2
    assert "arm_counts_whole_48" not in result
    assert result["arm_counts_whole_sample"]["A_unreviewed"]["mean"] == result["repetitions"][0]["arm_counts"]["A_unreviewed"]
    assert sum(stage == "generation" for stage, _ in calls) == 2
    assert sum(stage == "review" for stage, _ in calls) == 2
    assert sum(stage == "candidates" for stage, _ in calls) == 2
    assert sum(stage == "selection" for stage, _ in calls) == 2
    assert len(list((output / "r1" / "best5").rglob("calls.jsonl"))) == 1
    experiment_manifest = json.loads((output / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert experiment_manifest["prompt_versions"]["standard"] == PROMPT_VERSION
    case_results = json.loads((output / "case_results.json").read_text(encoding="utf-8"))
    assert case_results and all(row["prompt_versions"]["standard"] == PROMPT_VERSION for row in case_results)
    projection = result["cost"]["projection_500"]["B"]
    assert len(projection["per_repetition"]) == 2
    assert projection["known_cost_usd"]["mean"] == projection["per_repetition"][0]["known_cost_usd"]
    exported = json.loads((output / "r1" / "best5" / "s" / "subtitles.json").read_text(encoding="utf-8"))
    assert exported[0]["unknown"] == {"x": 0}
    assert exported[0]["start"] == 0


def test_experiment_skips_unreferenced_fit_and_calibration_sources(tmp_path: Path) -> None:
    manifest, manifest_dir = _fixture(tmp_path)
    source_path = manifest_dir / "sources" / "source.json"
    items = json.loads(source_path.read_text(encoding="utf-8"))
    for source_id, split in (("fit", "fit"), ("calibration", "calibration")):
        extra_path = manifest_dir / "sources" / f"{source_id}.json"
        extra_path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        manifest["sources"].append({"source_id": source_id, "json_path": f"sources/{source_id}.json",
                                    "split": split, "parsed_sha256": _hash(items), "sha256": "unused"})
    manifest["cases"] = [manifest["cases"][0]]
    calls: list[tuple[str, list[str]]] = []

    async def fake(_: ModelSpec, __: str, user: str, schema: dict[str, Any], ___: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        ids = [row.get("cue_id") for row in payload["targets"]]
        props = schema["properties"]["results"]["items"]["properties"]
        if "candidates" in props:
            calls.append(("candidates", ids))
            answer = {"results": [{"cue_id": cue_id, "status": "ok", "candidates": [{"candidate_id": "one", "text": "short"}]} for cue_id in ids]}
        elif "selected_candidate_id" in props:
            calls.append(("selection", ids))
            answer = {"results": [{"cue_id": row["cue_id"], "selected_candidate_id": row["candidates"][0]["candidate_id"], "issues": [], "advice": ""} for row in payload["targets"]]}
        elif "verdict" in props:
            calls.append(("review", ids))
            answer = {"results": [{"cue_id": cue_id, "verdict": "accept", "issues": [], "advice": ""} for cue_id in ids]}
        else:
            calls.append(("generation", ids))
            answer = {"results": [{"cue_id": cue_id, "text": "short", "status": "ok"} for cue_id in ids]}
        return DetailedResponse(json.dumps(answer), [])

    output = tmp_path / "experiment"
    result = asyncio.run(run_experiment(manifest, manifest_dir, output, generator=ModelSpec("deepseek", "deepseek-flash", "off"),
                                        repetitions=1, best_of=5, request_fn=fake))
    assert result["planned_case_count_per_repetition"] == 1
    assert {stage for stage, _ in calls} == {"generation", "review", "candidates", "selection"}
    assert not (output / "r1" / "standard" / "fit").exists()
    assert not (output / "r1" / "standard" / "calibration").exists()
    assert not (output / "r1" / "best5" / "fit").exists()
    assert not (output / "r1" / "best5" / "calibration").exists()


def test_empty_candidate_cases_make_no_request_or_export(tmp_path: Path) -> None:
    calls = 0

    async def fake(*_: object) -> DetailedResponse:
        nonlocal calls
        calls += 1
        return DetailedResponse("{}", [])

    records, summary = asyncio.run(_run_candidates(
        items=[], plans=[], cases=[], repetition=0, seed=20260910,
        generator=ModelSpec("deepseek", "deepseek-flash", "off"), request_fn=fake,
        output_dir=tmp_path / "empty",
    ))
    assert records == []
    assert summary["request_counts"] == {"generation_candidates": 0, "selection": 0}
    assert calls == 0
    assert not (tmp_path / "empty").exists()


def test_audit_masks_arms_and_skips_empty_case_networks(tmp_path: Path) -> None:
    manifest, manifest_dir = _fixture(tmp_path)
    manifest["audit_case_ids"] = [case["case_id"] for case in manifest["cases"]]
    manifest["crosscheck_case_ids"] = [manifest["cases"][0]["case_id"]]
    manifest["order_check_case_ids"] = [manifest["cases"][0]["case_id"]]
    audit_calls: list[str] = []

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        props = schema["properties"]["results"]["items"]["properties"]
        if "case_id" in props:
            audit_calls.append(spec.provider)
            rows = []
            for target in payload["targets"]:
                options = target["options"]
                rows.append({"case_id": target["case_id"], "judgments": [{"option_id": option["option_id"], "verdict": "accept", "issues": []} for option in options], "preferred_option_id": options[0]["option_id"]})
            answer = {"results": rows}
        elif "selected_candidate_id" in props:
            answer = {"results": [{"cue_id": target["cue_id"], "selected_candidate_id": target["candidates"][0]["candidate_id"], "issues": [], "advice": ""} for target in payload["targets"]]}
        elif "candidates" in props:
            answer = {"results": [{"cue_id": target["cue_id"], "status": "cannot_shorten", "candidates": []} for target in payload["targets"]]}
        elif "verdict" in props:
            answer = {"results": [{"cue_id": target["cue_id"], "verdict": "accept", "issues": [], "advice": ""} for target in payload["targets"]]}
        else:
            answer = {"results": [{"cue_id": target["cue_id"], "text": "short", "status": "ok"} for target in payload["targets"]]}
        return DetailedResponse(json.dumps(answer), [])

    result = asyncio.run(run_experiment(manifest, manifest_dir, tmp_path / "audit-experiment", generator=ModelSpec("deepseek", "deepseek-flash", "low"), repetitions=2, best_of=5, audit=True, request_fn=fake))
    assert result["audits"]["main"]
    assert result["effects"]["cross_judge_agreement"]["agree"] >= 0
    assert audit_calls.count("deepseek") == 4
    assert audit_calls.count("opencode") == 2
    assert len(result["audits_by_repetition"]) == 2
    assert len(result["effects_by_repetition"]) == 2
    assert all(row["repetition"] in {1, 2} for row in json.loads((tmp_path / "audit-experiment" / "case_results.json").read_text(encoding="utf-8")))


def test_missing_audit_arm_is_unavailable_and_judge_failure_is_audit_error() -> None:
    row = {"status": "complete", "judgment": {"judgments": []}, "mapping": {
        "one": {"arms": ["B"], "text_sha": "x", "text": "b"},
    }}
    assert _audit_status(row, "A") == "unavailable"
    assert _audit_status({"status": "error", "judgment": None, "mapping": {}}, "A") == "audit_error"


def test_effects_do_not_treat_missing_sol_arm_as_audit_agreement() -> None:
    cases = [{"case_id": "s:cue-0"}]
    audits = {"main": {"s:cue-0": {"status": "complete", "judgment": {"judgments": [
        {"option_id": "b", "verdict": "accept"},
    ]}, "mapping": {"b": {"arms": ["B"]}}}}}
    effects = _build_effects(audits, cases, {"s:cue-0": {"B": "selected"}})
    assert effects["paired_transitions"].get("harmed", 0) == 0
    assert effects["solaccepted_vs_flash_audit"]["B"].get("agree") == 1
    assert effects["solaccepted_vs_flash_audit"]["B"].get("disagree", 0) == 0


def test_effects_use_only_audited_cases_and_missing_arm_is_unavailable() -> None:
    cases = [{"case_id": f"s:cue-{index}"} for index in range(4)]
    audits = {"main": {
        "s:cue-0": {"status": "complete", "judgment": {"judgments": [
            {"option_id": "a0", "verdict": "reject"}, {"option_id": "b0", "verdict": "accept"},
        ]}, "mapping": {"a0": {"arms": ["A"]}, "b0": {"arms": ["B"]}}},
        "s:cue-1": {"status": "complete", "judgment": {"judgments": [
            {"option_id": "a1", "verdict": "accept"}, {"option_id": "b1", "verdict": "reject"},
        ]}, "mapping": {"a1": {"arms": ["A"]}, "b1": {"arms": ["B"]}}},
    }}
    effects = _build_effects(audits, cases, {case["case_id"]: {"A": "a", "B": "b"} for case in cases})
    assert effects["planned_audit_case_count"] == 2
    assert effects["paired_transitions"] == {"rescued": 1, "harmed": 1}
    assert effects["auditable_paired_case_count"] == 2

    missing_b = _build_effects(
        {"main": {"s:cue-0": {"status": "complete", "judgment": {"judgments": [
            {"option_id": "a", "verdict": "accept"},
        ]}, "mapping": {"a": {"arms": ["A"]}}}}},
        [{"case_id": "s:cue-0"}], {"s:cue-0": {"A": "a"}},
    )
    assert missing_b["paired_transitions"].get("harmed", 0) == 0
    assert missing_b["paired_transitions"]["unavailable"] == 1


def test_ledger_counts_error_request_without_fabricating_a_paid_call(tmp_path: Path) -> None:
    calls = tmp_path / "calls.jsonl"
    requests = tmp_path / "requests.jsonl"
    requests.write_text("\n".join([
        json.dumps({"request_id": "r1", "stage": "generation", "state": "start"}),
        json.dumps({"request_id": "r1", "stage": "generation", "state": "complete", "error": "transport"}),
    ]) + "\n", encoding="utf-8")
    summary = _ledger_summary([calls], stage="generation")
    assert summary["ledger_call_count"] == 0
    assert summary["known_subtotal_usd"] == 0
    assert summary["error_request_count"] == 1
    assert summary["unknown_request_count"] == 1
    assert summary["logical_error_ids"] == ["r1"]
    assert summary["projection_incomplete"] is True


def test_context_corruption_is_rejected_before_request(tmp_path: Path) -> None:
    manifest, manifest_dir = _fixture(tmp_path)
    manifest["cases"][0]["context"][0]["text"] = "corrupt"
    calls = 0

    async def fake(*_: object) -> DetailedResponse:
        nonlocal calls
        calls += 1
        return DetailedResponse("{}", [])

    try:
        asyncio.run(run_experiment(manifest, manifest_dir, tmp_path / "corrupt-output",
                                   generator=ModelSpec("opencode", "openai/gpt-5.6-luna", "medium"), request_fn=fake))
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt context should fail validation")
    assert calls == 0


def test_declared_raw_source_is_required_and_hash_checked(tmp_path: Path) -> None:
    manifest, manifest_dir = _fixture(tmp_path)
    raw_root = tmp_path / "Downloads"
    raw_root.mkdir()
    raw = raw_root / "trial.srt"
    raw.write_text("raw source", encoding="utf-8")
    manifest["input_root"] = str(raw_root)
    manifest["sources"][0].update({"path": "trial.srt", "sha256": "stale"})
    try:
        _validate_and_load(manifest, manifest_dir)
    except ValueError as exc:
        assert "raw hash mismatch" in str(exc)
    else:
        raise AssertionError("stale declared raw source must fail")
    raw.unlink()
    manifest["sources"][0]["sha256"] = "unused"
    try:
        _validate_and_load(manifest, manifest_dir)
    except ValueError as exc:
        assert "raw file missing" in str(exc)
    else:
        raise AssertionError("missing declared raw source must fail")


def test_length_tools_are_enabled_only_for_generators(tmp_path: Path) -> None:
    manifest, manifest_dir = _fixture(tmp_path)
    seen: list[tuple[str, bool]] = []

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        props = schema["properties"]["results"]["items"]["properties"]
        if "candidates" in props:
            stage = "candidate"
            answer = {"results": [{"cue_id": row["cue_id"], "status": "cannot_shorten", "candidates": []} for row in payload["targets"]]}
        elif "selected_candidate_id" in props:
            stage = "selection"
            answer = {"results": [{"cue_id": row["cue_id"], "selected_candidate_id": None, "issues": [], "advice": ""} for row in payload["targets"]]}
        elif "verdict" in props:
            stage = "review"
            answer = {"results": [{"cue_id": row["cue_id"], "verdict": "accept", "issues": [], "advice": ""} for row in payload["targets"]]}
        else:
            stage = "generation"
            answer = {"results": [{"cue_id": row["cue_id"], "text": "short", "status": "ok"} for row in payload["targets"]]}
        seen.append((stage, tools))
        return DetailedResponse(json.dumps(answer), [])

    asyncio.run(run_experiment(manifest, manifest_dir, tmp_path / "tools", generator=ModelSpec("opencode", "openai/gpt-5.6-luna", "medium"),
                               repetitions=1, best_of=5, request_fn=fake, length_tools=True))
    assert any(stage == "generation" and enabled for stage, enabled in seen)
    assert any(stage == "candidate" and enabled for stage, enabled in seen)
    assert all(not enabled for stage, enabled in seen if stage in {"review", "selection"})


def test_order_check_reuses_main_order_in_literal_reverse(tmp_path: Path) -> None:
    cases = [{"case_id": "s:cue-0", "original": "original", "max_chars": 20}]
    manifest = {"audit_case_ids": ["s:cue-0"], "crosscheck_case_ids": [], "order_check_case_ids": ["s:cue-0"]}
    arm_values = {"s:cue-0": {"A": "alpha", "B": "bravo", "C": "charlie"}}
    packets: list[list[str]] = []

    async def fake(_: ModelSpec, __: str, user: str, ___: dict[str, Any], ____: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        packets.append([option["text"] for option in payload["targets"][0]["options"]])
        options = payload["targets"][0]["options"]
        return DetailedResponse(json.dumps({"results": [{"case_id": "s:cue-0", "judgments": [
            {"option_id": option["option_id"], "verdict": "accept", "issues": []} for option in options
        ], "preferred_option_id": options[0]["option_id"]}]}), [])

    audits = asyncio.run(_run_audits(cases=cases, arm_values=arm_values, manifest=manifest,
                                     output_dir=tmp_path / "order", request_fn=fake, seed=4))
    assert packets[1] == list(reversed(packets[0]))
    assert audits["order"]["s:cue-0"]["status"] == "complete"


def test_legacy_and_default_planner_manifests_are_accepted() -> None:
    legacy = {"planner_config": {
        "cps": EXPECTED_PLANNER.cps,
        "max_speed": EXPECTED_PLANNER.max_speed,
        "gap_ms": EXPECTED_PLANNER.gap_ms,
        "context_before": EXPECTED_PLANNER.context_before,
        "context_after": EXPECTED_PLANNER.context_after,
    }}
    _planner_manifest(legacy)

    default = {"planner_config": asdict(EXPECTED_PLANNER)}
    _planner_manifest(default)


@pytest.mark.parametrize(
    "change",
    [
        {"length_basis": "normalized_tts_text"},
        {"calibration_profile": {}},
        {"calibration_identity": {}},
    ],
)
def test_frozen_planner_rejects_nondefault_additive_fields(change: dict[str, Any]) -> None:
    configured = asdict(EXPECTED_PLANNER)
    configured.update(change)
    with pytest.raises(ValueError, match="fixed experiment budget"):
        _planner_manifest({"planner_config": configured})


def test_frozen_planner_requires_every_original_field() -> None:
    configured = asdict(EXPECTED_PLANNER)
    del configured["cps"]
    with pytest.raises(ValueError, match="fixed experiment budget"):
        _planner_manifest({"planner_config": configured})
