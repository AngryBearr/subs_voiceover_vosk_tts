from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import pytest

import utils.verify_subtitles_cascade as cascade
from utils.evaluate_semantic_benchmark import evaluate_benchmark
from utils.verify_subtitles_cascade import build_arbiter_prior_report, build_tiebreak_prior_report, merge_cascade_results


def _stage(verdict: str = "pass") -> Dict[str, Any]:
    reason = None if verdict == "pass" else f"verdict_{verdict}"
    return {"verdict": verdict, "issues": [], "severity": None, "explanation": "ok", "error": None, "reason": reason, "fallback": reason}


def _report(verdict: str = "pass") -> Dict[str, Any]:
    row = {"index": 2, "unit_id": "primary", "unit_indices": [1, 2], **_stage(verdict)}
    unit = {"unit_id": "primary", "cue_indices": [1, 2], "changed_indices": [2], **_stage(verdict)}
    return {"schema_version": 1, "semantic_units_enabled": True, "context_mode": "full", "selected_indices": [2], "targets": [row], "units": [unit], "requests": 1}


def _usage(requests: int = 1, available: bool = False) -> Dict[str, Any]:
    value: Dict[str, Any] = {"requests": requests, "token_usage_available": available, "successful_responses": 1, "transport_retries": 0, "transport_failures": 0, "schema_failures": 0, "provider_ids": ["provider"], "model_ids": ["model"], "finishes": ["stop"]}
    if available:
        value.update({field: 10 for field in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens")})
    return value


def _arbiter(verdict: str = "pass", index: int = 2, severity: Any = None, explanation: Any = "arbiter") -> Dict[str, Any]:
    return {"index": index, "unit_id": "arbiter-unit", "unit_indices": [index], "verdict": verdict, "issues": [], "severity": severity, "explanation": explanation, "error": None if verdict != "technical" else "transport_failure", "reason": None if verdict == "pass" else "verdict_" + verdict if verdict in {"fail", "uncertain"} else "transport_failure", "fallback": None if verdict == "pass" else "verdict_" + verdict if verdict in {"fail", "uncertain"} else "transport_failure"}


def test_prior_contains_exact_candidate_evidence() -> None:
    candidate = [{"index": 1, "text": ["context"]}, {"index": 2, "text": ["changed", "line"]}]
    prior = build_arbiter_prior_report(candidate, _report())
    assert prior == {"schema_version": 1, "outcomes": [{"index": 2, "candidate_text": "changed line", "outcome": "verified"}]}


def test_prior_rejects_incomplete_primary_projection() -> None:
    report = _report()
    report["targets"][0]["reason"] = "broken"
    with pytest.raises(ValueError):
        build_arbiter_prior_report([{"index": 2, "text": "new"}], report)


def test_merge_retains_primary_membership_and_requires_arbiter_coverage() -> None:
    original = [{"index": 1, "text": "context"}, {"index": 2, "text": "old"}]
    candidate = [{"index": 1, "text": "context"}, {"index": 2, "text": "new"}]
    arbiter = {"schema_version": 1, "semantic_units_enabled": True, "context_mode": "full", "targets": [{"index": 2, "unit_id": "different", "unit_indices": [2], **_stage("pass")}], "requests": 1}
    output, report, usage = merge_cascade_results(original, candidate, _report(), {"requests": 1, "token_usage_available": False}, arbiter, {"requests": 1, "token_usage_available": False}, primary_model="deepseek-v4-flash", arbiter_model="openai/gpt-5.6-sol")
    assert output[1]["text"] == "new"
    assert report["units"][0]["unit_id"] == "primary"
    assert report["targets"][0]["unit_id"] == "primary"
    assert usage["backend"] == "cascade"


def test_merge_is_pristine_and_arbiter_fail_restores_original() -> None:
    original = [{"index": 1, "text": "old"}, {"index": 2, "text": ["before", "shape"]}]
    candidate = [{"index": 1, "text": "old"}, {"index": 2, "text": ["changed"]}]
    primary = _report()
    arbiter = {"schema_version": 1, "semantic_units_enabled": True, "context_mode": "full", "targets": [_arbiter("fail", 2)], "requests": 1}
    primary_before, arbiter_before = copy.deepcopy(primary), copy.deepcopy(arbiter)
    output, report, _ = merge_cascade_results(original, candidate, primary, _usage(), arbiter, _usage(), primary_model="p", arbiter_model="a")
    assert output[1]["text"] == ["before", "shape"]
    assert report["targets"][0]["verdict"] == "fail"
    assert primary == primary_before and arbiter == arbiter_before
    assert report["stages"]["primary"] == primary_before and report["stages"]["arbiter"] == arbiter_before


def test_actual_changed_coverage_rejects_missing_and_extra_targets() -> None:
    original = [{"index": 1, "text": "a"}, {"index": 2, "text": "b"}]
    candidate = [{"index": 1, "text": "changed"}, {"index": 2, "text": "also changed"}]
    primary = _report(); primary["targets"][0]["index"] = 1; primary["selected_indices"] = [1]; primary["units"][0]["changed_indices"] = [1]; primary["units"][0]["cue_indices"] = [1]; primary["targets"][0]["unit_indices"] = [1]
    arbiter = {"schema_version": 1, "semantic_units_enabled": True, "context_mode": "full", "targets": [], "requests": 0}
    with pytest.raises(ValueError):
        merge_cascade_results(original, candidate, primary, _usage(), arbiter, _usage(0), primary_model="p", arbiter_model="a")


def test_two_changed_indices_use_fail_precedence_and_routing_counts() -> None:
    original = [{"index": 1, "text": "a"}, {"index": 2, "text": "b"}, {"index": 3, "text": "c"}]
    candidate = [{"index": 1, "text": "a1"}, {"index": 2, "text": "b1"}, {"index": 3, "text": "c"}]
    primary = _report(); primary["targets"][0]["index"] = 1; primary["targets"][0]["unit_indices"] = [1, 2]; primary["units"][0]["cue_indices"] = [1, 2]; primary["units"][0]["changed_indices"] = [1, 2]; primary["targets"].append({**primary["targets"][0], "index": 2}); primary["selected_indices"] = [1, 2]
    arbiter = {"schema_version": 1, "semantic_units_enabled": True, "context_mode": "full", "targets": [_arbiter("pass", 1), _arbiter("fail", 2)], "requests": 1}
    _, report, _ = merge_cascade_results(original, candidate, primary, _usage(), arbiter, _usage(), primary_model="p", arbiter_model="a")
    assert report["verified_indices"] == [] and report["routing"]["escalated_changed_indices"] == [1, 2]
    assert {row["verdict"] for row in report["targets"]} == {"fail"}


@pytest.mark.parametrize("row", [_arbiter("uncertain"), _arbiter("technical"), _arbiter("pass", severity="critical"), _arbiter("pass", explanation="")])
def test_invalid_or_nonpass_arbiter_rows_fail_closed_with_valid_projection(row: Dict[str, Any]) -> None:
    original = [{"index": 1, "text": "old"}, {"index": 2, "text": "old"}]
    candidate = [{"index": 1, "text": "old"}, {"index": 2, "text": "new"}]
    arbiter = {"schema_version": 1, "semantic_units_enabled": True, "context_mode": "full", "targets": [row], "requests": 1}
    _, report, _ = merge_cascade_results(original, candidate, _report(), _usage(), arbiter, _usage(), primary_model="p", arbiter_model="a")
    result = report["targets"][0]
    assert result["explanation"] and (result["verdict"] in {"uncertain", None})
    if result["verdict"] is None:
        assert result["error"] == result["reason"] == result["fallback"] == "cascade_arbiter_projection_failure"


def test_primary_nonpass_is_not_overridden_and_membership_is_primary() -> None:
    original = [{"index": 1, "text": "old"}, {"index": 2, "text": "old"}]
    candidate = [{"index": 1, "text": "new"}, {"index": 2, "text": "new"}]
    primary = _report("uncertain"); primary["targets"][0]["index"] = 1; primary["targets"][0]["unit_indices"] = [1, 2]; primary["units"][0]["cue_indices"] = [1, 2]; primary["units"][0]["changed_indices"] = [1, 2]; primary["selected_indices"] = [1, 2]
    primary["targets"].append({"index": 2, "unit_id": "primary", "unit_indices": [1, 2], **_stage("uncertain")})
    arbiter = {"schema_version": 1, "semantic_units_enabled": True, "context_mode": "full", "targets": [_arbiter("pass", 1), _arbiter("pass", 2)], "requests": 1}
    _, report, _ = merge_cascade_results(original, candidate, primary, _usage(), arbiter, _usage(), primary_model="p", arbiter_model="a")
    assert all(row["verdict"] == "uncertain" for row in report["targets"])
    assert all(row["unit_id"] == "primary" and row["unit_indices"] == [1, 2] for row in report["targets"])


def test_usage_complete_partial_and_malformed_metadata() -> None:
    original = [{"index": 1, "text": "old"}, {"index": 2, "text": "old"}]
    candidate = [{"index": 1, "text": "old"}, {"index": 2, "text": "new"}]
    primary, arbiter = _report(), {"schema_version": 1, "semantic_units_enabled": True, "context_mode": "full", "targets": [_arbiter()], "requests": 2}
    p_usage, a_usage = _usage(1, True), _usage(2, True); p_usage.update({"input_tokens": 100, "output_tokens": 20, "reasoning_tokens": 5, "total_tokens": 120, "cache_read_tokens": 10, "cache_write_tokens": 2, "provider_ids": ["p"]}); a_usage.update({"provider_ids": ["a"], "model_ids": ["am"], "finishes": ["length"]})
    _, _, usage = merge_cascade_results(original, candidate, primary, p_usage, arbiter, a_usage, primary_model="p", arbiter_model="a")
    assert usage["input_tokens"] == 110 and usage["provider_ids"] == ["a", "p"] and "deepseek_tariff_estimate_usd" in usage
    p_usage["token_usage_available"] = False
    for field in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens"):
        p_usage.pop(field)
    _, _, partial = merge_cascade_results(original, candidate, primary, p_usage, arbiter, a_usage, primary_model="p", arbiter_model="a")
    assert partial["token_usage_available"] is False and "input_tokens" not in partial
    p_usage["requests"] = True
    with pytest.raises(ValueError):
        merge_cascade_results(original, candidate, primary, p_usage, arbiter, a_usage, primary_model="p", arbiter_model="a")


def test_evaluator_accepts_merged_cascade_artifact() -> None:
    original = [{"index": 1, "text": "old"}, {"index": 2, "text": "old"}]
    candidate = [{"index": 1, "text": "old"}, {"index": 2, "text": "new"}]
    primary, arbiter = _report(), {"schema_version": 1, "semantic_units_enabled": True, "context_mode": "full", "targets": [_arbiter()], "requests": 1}
    _, report, usage = merge_cascade_results(original, candidate, primary, _usage(), arbiter, _usage(), primary_model="p", arbiter_model="a")
    result = evaluate_benchmark({"cases": [{"episode": "E", "index": 2, "expected": "pass", "clear": True}]}, {"E": report}, {"E": usage})
    assert result["status"] == "qualified"


def test_parser_defaults_and_required_full_context() -> None:
    parser = cascade.build_parser()
    args = parser.parse_args(["original.json", "candidate.json", "--context-source", "context.json"])
    assert args.primary_model == "deepseek-v4-flash"
    assert args.arbiter_model == "openai/gpt-5.6-sol"
    assert args.batch_size == 4 and args.context_window == 3 and args.max_gap == 0.3
    with pytest.raises(SystemExit):
        parser.parse_args(["original.json", "candidate.json"])


@pytest.mark.parametrize("field,value", [("primary_timeout", True), ("primary_timeout", float("inf")), ("primary_temperature", True), ("primary_temperature", 3.0)])
def test_primary_numeric_config_is_fail_closed(field: str, value: Any) -> None:
    args = cascade.build_parser().parse_args(["original.json", "candidate.json", "--context-source", "context.json"])
    setattr(args, field, value)
    with pytest.raises(ValueError):
        cascade._validate_config(args)


def test_valid_technical_arbiter_row_is_projected_safely() -> None:
    original = [{"index": 1, "text": "context"}, {"index": 2, "text": "old"}]
    candidate = [{"index": 1, "text": "context"}, {"index": 2, "text": "new"}]
    row = _arbiter("pass")
    row.update({
        "verdict": None,
        "issues": ["transport"],
        "severity": "major",
        "explanation": "transport unavailable",
        "error": "transport_failure",
        "reason": "transport_failure",
        "fallback": "transport_failure",
    })
    arbiter = {
        "schema_version": 1,
        "semantic_units_enabled": True,
        "context_mode": "full",
        "targets": [row],
        "requests": 1,
    }
    output, report, _ = merge_cascade_results(
        original,
        candidate,
        _report(),
        _usage(),
        arbiter,
        _usage(),
        primary_model="p",
        arbiter_model="a",
    )
    target = report["targets"][0]
    assert output[1]["text"] == "old"
    assert target["verdict"] is None
    assert target["error"] == target["reason"] == target["fallback"] == "cascade_arbiter_projection_failure"
    assert target["explanation"]


def test_stage_usage_true_requires_every_token_field() -> None:
    original = [{"index": 1, "text": "context"}, {"index": 2, "text": "old"}]
    candidate = [{"index": 1, "text": "context"}, {"index": 2, "text": "new"}]
    arbiter = {
        "schema_version": 1,
        "semantic_units_enabled": True,
        "context_mode": "full",
        "targets": [_arbiter()],
        "requests": 1,
    }
    malformed = _usage(1, True)
    malformed.pop("cache_write_tokens")
    with pytest.raises(ValueError, match="primary_missing_cache_write_tokens"):
        merge_cascade_results(
            original,
            candidate,
            _report(),
            malformed,
            arbiter,
            _usage(1, False),
            primary_model="p",
            arbiter_model="a",
        )


def _write_cli_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    original = tmp_path / "original.json"
    candidate = tmp_path / "candidate.json"
    context = tmp_path / "context.json"
    original_items = [
        {"index": 1, "text": ["context"]},
        {"index": 2, "text": ["old"]},
    ]
    candidate_items = [
        {"index": 1, "text": ["context"]},
        {"index": 2, "text": ["new"]},
    ]
    original.write_text(json.dumps(original_items), encoding="utf-8")
    candidate.write_text(json.dumps(candidate_items), encoding="utf-8")
    context.write_text(json.dumps(original_items), encoding="utf-8")
    return original, candidate, context


def test_invalid_input_precedes_key_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original, candidate, context = _write_cli_inputs(tmp_path)
    looked_up = False

    def key(_: str) -> str:
        nonlocal looked_up
        looked_up = True
        return "secret"

    monkeypatch.setattr(cascade, "load_api_key", key)
    assert cascade.main([
        str(original),
        str(candidate),
        "--context-source", str(context),
        "--deepseek-base-url", "ftp://invalid",
    ]) == 2
    assert looked_up is False
    assert cascade.main([
        str(tmp_path / "missing.json"),
        str(candidate),
        "--context-source", str(context),
    ]) == 2
    assert looked_up is False


def test_missing_key_returns_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original, candidate, context = _write_cli_inputs(tmp_path)
    monkeypatch.setattr(cascade, "load_api_key", lambda _: "")
    assert cascade.main([
        str(original),
        str(candidate),
        "--context-source", str(context),
    ]) == 1


def test_main_forwards_stages_and_writes_safe_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, candidate, context = _write_cli_inputs(tmp_path)
    captured: Dict[str, Any] = {}

    async def request_json(**kwargs: Any) -> None:
        captured["request"] = kwargs
        return None

    async def primary_stage(left: Any, right: Any, **kwargs: Any) -> tuple[Any, Dict[str, Any], Dict[str, Any]]:
        captured["primary"] = kwargs
        await kwargs["request_callable"]("primary prompt", kwargs["model"])
        return right, _report(), _usage()

    def arbiter_stage(left: Any, right: Any, **kwargs: Any) -> tuple[Any, Dict[str, Any], Dict[str, Any]]:
        captured["arbiter"] = kwargs
        report = {
            "schema_version": 1,
            "semantic_units_enabled": True,
            "context_mode": "full",
            "targets": [_arbiter()],
            "requests": 1,
        }
        return right, report, _usage()

    monkeypatch.setattr(cascade, "request_json_object", request_json)
    monkeypatch.setattr(cascade, "verify_items", primary_stage)
    monkeypatch.setattr(cascade, "verify_items_live", arbiter_stage)
    monkeypatch.setattr(cascade, "load_api_key", lambda _: pytest.fail("explicit key must win"))
    output_dir = tmp_path / "out"
    secret = "DIRECT_SECRET"
    assert cascade.main([
        str(original),
        str(candidate),
        "--context-source", str(context),
        "--deepseek-api-key", secret,
        "--deepseek-base-url", "https://example.test/api",
        "--primary-timeout", "9",
        "--primary-temperature", "0.4",
        "--primary-concurrency", "2",
        "--primary-transport-retries", "2",
        "--arbiter-concurrency", "4",
        "--arbiter-transport-retries", "3",
        "--batch-size", "2",
        "--context-window", "5",
        "--schema-retries", "0",
        "--semantic-unit-max-cues", "2",
        "--max-gap", "0.2",
        "--output-dir", str(output_dir),
    ]) == 0
    assert captured["request"] == {
        "api_key": secret,
        "model": "deepseek-v4-flash",
        "system_prompt": cascade.UNIT_VERIFIER_SYSTEM_PROMPT,
        "user_prompt": "primary prompt",
        "base_url": "https://example.test/api",
        "timeout": 9.0,
        "temperature": 0.4,
    }
    assert captured["primary"]["semantic_units"] is True
    assert captured["primary"]["backend"] == "deepseek"
    assert captured["primary"]["accounting"] == "metered_api"
    assert captured["primary"]["concurrency"] == 2
    assert captured["primary"]["transport_retries"] == 2
    assert captured["arbiter"]["structured_output"] is True
    assert captured["arbiter"]["semantic_units"] is True
    assert captured["arbiter"]["concurrency"] == 4
    assert captured["arbiter"]["transport_retries"] == 3
    assert captured["arbiter"]["review_report"]["outcomes"][0]["outcome"] == "verified"
    for suffix in (".json", ".report.json", ".usage.json"):
        path = output_dir / f"candidate_cascade_verified{suffix}"
        assert path.exists()
        data = path.read_bytes()
        assert data.endswith(b"\n")
        assert secret.encode() not in data


def test_main_returns_two_on_server_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original, candidate, context = _write_cli_inputs(tmp_path)

    async def primary_stage(left: Any, right: Any, **_: Any) -> tuple[Any, Dict[str, Any], Dict[str, Any]]:
        return right, _report(), _usage()

    def timeout(*_: Any, **__: Any) -> Any:
        raise TimeoutError("server timeout")

    monkeypatch.setattr(cascade, "verify_items", primary_stage)
    monkeypatch.setattr(cascade, "verify_items_live", timeout)
    assert cascade.main([
        str(original),
        str(candidate),
        "--context-source", str(context),
        "--deepseek-api-key", "secret",
    ]) == 2


def test_cli_disagreement_uses_singleton_tiebreak_and_agreement_skips_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original, candidate, context = _write_cli_inputs(tmp_path)
    calls: list[Dict[str, Any]] = []

    async def primary_stage(left: Any, right: Any, **_: Any) -> tuple[Any, Dict[str, Any], Dict[str, Any]]:
        return right, _report(), _usage()

    def live_stage(left: Any, right: Any, **kwargs: Any) -> tuple[Any, Dict[str, Any], Dict[str, Any]]:
        calls.append(kwargs)
        verdict = "fail" if len(calls) == 1 else "pass"
        return right, _tiebreak_report(verdict), _usage()

    monkeypatch.setattr(cascade, "verify_items", primary_stage)
    monkeypatch.setattr(cascade, "verify_items_live", live_stage)
    assert cascade.main([str(original), str(candidate), "--context-source", str(context), "--deepseek-api-key", "secret", "--batch-size", "3", "--output-dir", str(tmp_path / "disagree")]) == 0
    assert len(calls) == 2
    assert calls[0]["batch_size"] == 3 and calls[1]["batch_size"] == 1
    assert calls[0]["structured_output"] is True and calls[1]["structured_output"] is True
    assert calls[0]["model"] == calls[1]["model"] == "openai/gpt-5.6-sol"
    assert calls[0]["context_items"] == calls[1]["context_items"]
    assert calls[1]["review_report"]["outcomes"][0]["outcome"] == "verified"
    calls.clear()

    def agreeing_live(left: Any, right: Any, **kwargs: Any) -> tuple[Any, Dict[str, Any], Dict[str, Any]]:
        calls.append(kwargs)
        return right, _tiebreak_report("pass"), _usage()

    monkeypatch.setattr(cascade, "verify_items_live", agreeing_live)
    assert cascade.main([str(original), str(candidate), "--context-source", str(context), "--deepseek-api-key", "secret", "--batch-size", "3", "--output-dir", str(tmp_path / "agree")]) == 0
    assert len(calls) == 1
    agree_report = json.loads((tmp_path / "agree" / "candidate_cascade_verified.report.json").read_text(encoding="utf-8"))
    assert "tiebreak_batch_size" not in agree_report["routing"]


def test_stage_usage_rejects_negative_counter() -> None:
    original = [{"index": 1, "text": "context"}, {"index": 2, "text": "old"}]
    candidate = [{"index": 1, "text": "context"}, {"index": 2, "text": "new"}]
    arbiter = {
        "schema_version": 1,
        "semantic_units_enabled": True,
        "context_mode": "full",
        "targets": [_arbiter()],
        "requests": 1,
    }
    malformed = _usage()
    malformed["transport_failures"] = -1
    with pytest.raises(ValueError, match="primary_transport_failures"):
        merge_cascade_results(
            original,
            candidate,
            _report(),
            malformed,
            arbiter,
            _usage(),
            primary_model="p",
            arbiter_model="a",
        )


def _polarity_report() -> Dict[str, Any]:
    report = _report("fail")
    report["targets"][0]["issues"] = ["polarity"]
    report["units"][0]["issues"] = ["polarity"]
    return report


def test_polarity_failure_is_routed_and_arbiter_pass_wins() -> None:
    original = [{"index": 1, "text": "context"}, {"index": 2, "text": "old"}]
    candidate = [{"index": 1, "text": "context"}, {"index": 2, "text": "new"}]
    primary = _polarity_report()
    prior = build_arbiter_prior_report(candidate, primary)
    assert prior["outcomes"] == [{"index": 2, "candidate_text": "new", "outcome": "verified"}]
    arbiter = {"schema_version": 1, "semantic_units_enabled": True, "context_mode": "full", "targets": [_arbiter()], "requests": 1}
    output, report, _ = merge_cascade_results(original, candidate, primary, _usage(), arbiter, _usage(), primary_model="p", arbiter_model="a")
    assert output[1]["text"] == "new"
    assert report["verified_indices"] == [2]
    assert report["routing"]["policy"] == "confirm_primary_pass_or_polarity_with_singleton_tiebreak"
    assert report["routing"]["escalated_unit_count"] == 1


def test_non_polarity_primary_failure_is_not_routed_or_overridden() -> None:
    original = [{"index": 1, "text": "context"}, {"index": 2, "text": "old"}]
    candidate = [{"index": 1, "text": "context"}, {"index": 2, "text": "new"}]
    primary = _report("fail")
    primary["targets"][0]["issues"] = ["degree"]
    primary["units"][0]["issues"] = ["degree"]
    prior = build_arbiter_prior_report(candidate, primary)
    assert prior["outcomes"][0]["outcome"] == "unresolved"
    arbiter = {"schema_version": 1, "semantic_units_enabled": True, "context_mode": "full", "targets": [_arbiter()], "requests": 1}
    output, report, _ = merge_cascade_results(original, candidate, primary, _usage(), arbiter, _usage(), primary_model="p", arbiter_model="a")
    assert output[1]["text"] == "old"
    assert report["targets"][0]["verdict"] == "fail"
    assert report["routing"]["escalated_unit_count"] == 0


def test_multi_index_polarity_unit_routes_atomically() -> None:
    original = [{"index": 1, "text": "a"}, {"index": 2, "text": "b"}, {"index": 3, "text": "c"}]
    candidate = [{"index": 1, "text": "a1"}, {"index": 2, "text": "b1"}, {"index": 3, "text": "c"}]
    primary = _polarity_report()
    primary["targets"][0]["index"] = 1
    primary["targets"][0]["unit_indices"] = [1, 2]
    primary["targets"][0]["issues"] = ["polarity"]
    primary["units"][0]["cue_indices"] = [1, 2]
    primary["units"][0]["changed_indices"] = [1, 2]
    primary["units"][0]["issues"] = ["polarity"]
    primary["targets"].append({**primary["targets"][0], "index": 2})
    primary["selected_indices"] = [1, 2]
    arbiter = {"schema_version": 1, "semantic_units_enabled": True, "context_mode": "full", "targets": [_arbiter("pass", 1), _arbiter("pass", 2)], "requests": 1}
    output, report, _ = merge_cascade_results(original, candidate, primary, _usage(), arbiter, _usage(), primary_model="p", arbiter_model="a")
    assert [item["text"] for item in output] == ["a1", "b1", "c"]
    assert report["verified_indices"] == [1, 2]
    assert report["routing"]["escalated_changed_indices"] == [1, 2]
    assert report["routing"]["escalated_changed_count"] == 2


def test_malformed_primary_issues_reject_routing() -> None:
    report = _polarity_report()
    report["units"][0]["issues"] = ["polarity", 1]
    report["targets"][0]["issues"] = ["polarity", 1]
    with pytest.raises(ValueError, match="primary_issues"):
        build_arbiter_prior_report([{"index": 2, "text": "new"}], report)


def _tiebreak_report(verdict: str = "pass", index: int = 2) -> Dict[str, Any]:
    return {"schema_version": 1, "semantic_units_enabled": True, "context_mode": "full", "targets": [_arbiter(verdict, index)], "requests": 1}


def test_tiebreak_prior_selects_only_disagreements() -> None:
    candidate = [{"index": 1, "text": "context"}, {"index": 2, "text": "new"}]
    primary = _report()
    agree = _tiebreak_report("pass")
    assert build_tiebreak_prior_report(candidate, primary, agree)["outcomes"][0]["outcome"] == "unresolved"
    disagree = _tiebreak_report("fail")
    assert build_tiebreak_prior_report(candidate, primary, disagree)["outcomes"][0]["outcome"] == "verified"
    polarity = _polarity_report()
    assert build_tiebreak_prior_report(candidate, polarity, agree)["outcomes"][0]["outcome"] == "verified"


@pytest.mark.parametrize("primary,arbiter,tiebreak,expected", [("pass", "fail", "pass", "pass"), ("pass", "fail", "fail", "fail"), ("pass", "uncertain", "uncertain", "uncertain"), ("pass", "technical", "pass", "pass")])
def test_tiebreak_voting_uses_semantic_majority(primary: str, arbiter: str, tiebreak: str, expected: str) -> None:
    original = [{"index": 1, "text": "context"}, {"index": 2, "text": "old"}]
    candidate = [{"index": 1, "text": "context"}, {"index": 2, "text": "new"}]
    primary_report = _report(primary)
    arbiter_report = _tiebreak_report(arbiter)
    if arbiter == "technical":
        arbiter_report["targets"][0] = _arbiter("technical")
    tiebreak_report = _tiebreak_report(tiebreak)
    _, report, _ = merge_cascade_results(original, candidate, primary_report, _usage(), arbiter_report, _usage(), primary_model="p", arbiter_model="a", tiebreak_report=tiebreak_report, tiebreak_usage=_usage())
    assert report["targets"][0]["verdict"] == expected


def test_tiebreak_three_distinct_votes_are_uncertain_and_missing_is_technical() -> None:
    original = [{"index": 1, "text": "context"}, {"index": 2, "text": "old"}]
    candidate = [{"index": 1, "text": "context"}, {"index": 2, "text": "new"}]
    arbiter = _tiebreak_report("fail")
    tiebreak = _tiebreak_report("uncertain")
    primary = _report("pass")
    _, distinct, _ = merge_cascade_results(original, candidate, primary, _usage(), arbiter, _usage(), primary_model="p", arbiter_model="a", tiebreak_report=tiebreak, tiebreak_usage=_usage())
    assert distinct["targets"][0]["verdict"] == "uncertain"
    _, missing, usage = merge_cascade_results(original, candidate, primary, _usage(), arbiter, _usage(), primary_model="p", arbiter_model="a", tiebreak_report=_tiebreak_report("technical"), tiebreak_usage=_usage())
    assert missing["targets"][0]["verdict"] is None
    assert usage["stages"]["tiebreaker"]["requests"] == 1


def test_tiebreak_stage_is_pristine_and_routing_usage_are_three_stage() -> None:
    original = [{"index": 1, "text": "context"}, {"index": 2, "text": "old"}]
    candidate = [{"index": 1, "text": "context"}, {"index": 2, "text": "new"}]
    primary, arbiter, tiebreak = _report("pass"), _tiebreak_report("fail"), _tiebreak_report("pass")
    import copy as _copy
    snapshot = _copy.deepcopy(tiebreak)
    _, report, usage = merge_cascade_results(original, candidate, primary, _usage(), arbiter, _usage(), primary_model="p", arbiter_model="a", tiebreak_report=tiebreak, tiebreak_usage=_usage())
    assert tiebreak == snapshot
    assert report["stages"]["tiebreaker"] == snapshot
    assert report["routing"]["tiebreak_unit_ids"] == ["primary"]
    assert report["routing"]["tiebreak_changed_indices"] == [2]
    assert report["routing"]["tiebreak_batch_size"] == 1
    assert usage["requests"] == 3


def test_tiebreak_membership_maps_by_index_and_evaluator_accepts() -> None:
    original = [{"index": 1, "text": "context"}, {"index": 2, "text": "old"}]
    candidate = [{"index": 1, "text": "context"}, {"index": 2, "text": "new"}]
    primary, arbiter, tiebreak = _report("pass"), _tiebreak_report("fail"), _tiebreak_report("pass")
    arbiter["targets"][0]["unit_id"] = "other"
    tiebreak["targets"][0]["unit_id"] = "another"
    _, report, usage = merge_cascade_results(original, candidate, primary, _usage(), arbiter, _usage(), primary_model="p", arbiter_model="a", tiebreak_report=tiebreak, tiebreak_usage=_usage())
    assert report["targets"][0]["unit_id"] == "primary"
    result = evaluate_benchmark({"cases": [{"episode": "E", "index": 2, "expected": "pass", "clear": True}]}, {"E": report}, {"E": usage})
    assert result["status"] == "qualified"
