"""Synthetic coverage for the offline benchmark evaluator."""
from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Dict

import pytest

from utils.evaluate_semantic_benchmark import EvaluationError, evaluate_benchmark


def manifest(*cases: Dict[str, Any]) -> Dict[str, Any]:
    return {"cases": list(cases)}


def case(episode: str, index: int, expected: str = "pass", clear: bool = True) -> Dict[str, Any]:
    return {"episode": episode, "index": index, "expected": expected, "clear": clear}


def artifacts(indices: list[int], verdicts: Dict[int, Any], expected_units: bool = True) -> tuple[Dict[str, Any], Dict[str, Any]]:
    targets = []
    units = []
    verified, unresolved, fallback = [], [], []
    for index in indices:
        verdict = verdicts[index]
        if verdict == "pass":
            error = reason = fallback_value = None
            verified.append(index)
        elif verdict in {"fail", "uncertain"}:
            error = None
            reason = fallback_value = f"verdict_{verdict}"
            unresolved.append(index)
            fallback.append(index)
        else:
            error, reason, fallback_value = "transport failure", "transport failure", "transport failure"
            unresolved.append(index)
            fallback.append(index)
        target = {"index": index, "verdict": verdict, "issues": [], "severity": "minor" if verdict == "uncertain" else None,
                  "explanation": "synthetic", "error": error, "reason": reason, "fallback": fallback_value,
                  "unit_id": f"u{index}", "unit_indices": [index]}
        targets.append(target)
        units.append({**target, "cue_indices": [index], "changed_indices": [index]})
    report = {"schema_version": 1, "context_mode": "full", "semantic_units_enabled": expected_units,
              "targets": targets, "selected_indices": indices, "verified_indices": verified,
              "unresolved_indices": unresolved, "fallback_indices": fallback, "units": units,
              "model": "m", "backend": "opencode", "accounting": "subscription", "requests": len(indices)}
    usage = {"model": "m", "backend": "opencode", "accounting": "subscription", "requests": len(indices),
             "successful_responses": len(verified), "transport_retries": 0, "transport_failures": 0,
             "schema_failures": 0, "token_usage_available": True, "input_tokens": 2,
             "output_tokens": 1, "total_tokens": 3,
             "provider_ids": ["p"], "model_ids": ["provider-m"], "finishes": ["stop"],
             "provider_reported_cost_usd": 0, "provider_cost_is_billing_authoritative": True,
             "usage_response_count": len(verified)}
    return report, usage


def test_ideal_clear_cases_qualify_with_wrong_diagnostics() -> None:
    man = manifest(case("E", 1, "pass"), case("E", 2, "pass", False))
    report, usage = artifacts([1, 2], {1: "pass", 2: "fail"})
    result = evaluate_benchmark(man, {"E": report}, {"E": usage})
    assert result["status"] == "qualified"
    assert result["clear"]["correct"] == 1
    assert result["diagnostic"]["false_rejects"] == [["E", 2]]


@pytest.mark.parametrize("verdict,outcome", [("pass", "false_pass"), ("fail", "false_reject"), ("uncertain", "semantic_uncertain"), (None, "technical_unresolved")])
def test_clear_failure_categories(verdict: Any, outcome: str) -> None:
    man = manifest(case("E", 1, "fail" if verdict == "pass" else "pass"))
    report, usage = artifacts([1], {1: verdict})
    result = evaluate_benchmark(man, {"E": report}, {"E": usage})
    assert result["cases"][0]["outcome"] == outcome


def test_episode_and_index_validation_is_not_positional() -> None:
    man = manifest(case("A", 7), case("B", 7))
    report_a, usage_a = artifacts([7], {7: "pass"})
    report_b, usage_b = artifacts([7], {7: "pass"})
    result = evaluate_benchmark(man, {"A": report_a, "B": report_b}, {"A": usage_a, "B": usage_b})
    assert len(result["cases"]) == 2


def test_duplicate_episode_is_rejected() -> None:
    man = manifest(case("E", 1))
    report, usage = artifacts([1], {1: "pass"})
    with pytest.raises(EvaluationError):
        evaluate_benchmark(man, {"E": report, "X": report}, {"E": usage})


def test_projection_and_sparse_context_are_rejected() -> None:
    man = manifest(case("E", 1))
    report, usage = artifacts([1], {1: "pass"})
    report["targets"][0]["fallback"] = "verdict_pass"
    with pytest.raises(EvaluationError):
        evaluate_benchmark(man, {"E": report}, {"E": usage})


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens"])
def test_flat_token_fields_are_aggregated_with_explicit_zero(field: str) -> None:
    man = manifest(case("A", 1), case("B", 1))
    report_a, usage_a = artifacts([1], {1: "pass"})
    report_b, usage_b = artifacts([1], {1: "pass"})
    usage_a[field] = 0
    usage_b.pop(field, None)
    result = evaluate_benchmark(man, {"A": report_a, "B": report_b}, {"A": usage_a, "B": usage_b})
    assert result["usage"]["tokens"][field] == {"value": 0, "availability": "partial"}


@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
def test_flat_token_values_reject_bool_negative_and_non_int(value: Any) -> None:
    man = manifest(case("E", 1))
    report, usage = artifacts([1], {1: "pass"})
    usage["input_tokens"] = value
    with pytest.raises(EvaluationError):
        evaluate_benchmark(man, {"E": report}, {"E": usage})


def test_token_availability_requires_core_metadata_and_rejects_false_with_tokens() -> None:
    man = manifest(case("E", 1))
    report, usage = artifacts([1], {1: "pass"})
    usage["token_usage_available"] = True
    usage.pop("input_tokens")
    usage.pop("output_tokens")
    usage.pop("total_tokens")
    with pytest.raises(EvaluationError):
        evaluate_benchmark(man, {"E": report}, {"E": usage})
    usage["token_usage_available"] = False
    usage["input_tokens"] = 1
    with pytest.raises(EvaluationError):
        evaluate_benchmark(man, {"E": report}, {"E": usage})


@pytest.mark.parametrize("authority,status", [(True, "billing_authoritative"), (False, "informational")])
def test_cost_zero_is_present_and_authority_is_reported(authority: bool, status: str) -> None:
    man = manifest(case("E", 1))
    report, usage = artifacts([1], {1: "pass"})
    usage["provider_cost_is_billing_authoritative"] = authority
    usage["provider_reported_cost_usd"] = 0
    result = evaluate_benchmark(man, {"E": report}, {"E": usage})
    assert result["usage"]["cost"] == 0
    assert result["usage"]["cost_status"] == status


def test_cost_missing_is_none_and_partial_cost_is_not_zero() -> None:
    man = manifest(case("A", 1), case("B", 1))
    report_a, usage_a = artifacts([1], {1: "pass"})
    report_b, usage_b = artifacts([1], {1: "pass"})
    usage_b.pop("provider_reported_cost_usd")
    result = evaluate_benchmark(man, {"A": report_a, "B": report_b}, {"A": usage_a, "B": usage_b})
    assert result["usage"]["cost_status"] == "partial"
    assert result["usage"]["cost"] == 0
    usage_a["provider_reported_cost_usd"] = float("nan")
    with pytest.raises(EvaluationError):
        evaluate_benchmark(manifest(case("A", 1)), {"A": report_a}, {"A": usage_a})


def test_cost_authority_mismatch_is_invalid() -> None:
    man = manifest(case("A", 1), case("B", 1))
    report_a, usage_a = artifacts([1], {1: "pass"})
    report_b, usage_b = artifacts([1], {1: "pass"})
    usage_b["provider_cost_is_billing_authoritative"] = False
    with pytest.raises(EvaluationError):
        evaluate_benchmark(man, {"A": report_a, "B": report_b}, {"A": usage_a, "B": usage_b})


def test_provider_model_finish_fields_are_unioned() -> None:
    man = manifest(case("A", 1), case("B", 1))
    report_a, usage_a = artifacts([1], {1: "pass"})
    report_b, usage_b = artifacts([1], {1: "pass"})
    usage_b["provider_ids"] = ["other"]
    usage_b["model_ids"] = ["other-model"]
    usage_b["finishes"] = ["length"]
    result = evaluate_benchmark(man, {"A": report_a, "B": report_b}, {"A": usage_a, "B": usage_b})
    assert result["usage"]["provider_ids"] == ["other", "p"]
    assert result["usage"]["model_ids"] == ["other-model", "provider-m"]
    assert result["usage"]["finishes"] == ["length", "stop"]


def test_shared_unit_and_context_only_members_are_valid() -> None:
    man = manifest(case("E", 2, "fail"), case("E", 3, "fail"))
    report, usage = artifacts([2, 3], {2: "fail", 3: "fail"})
    shared = {"unit_id": "shared", "cue_indices": [1, 2, 3], "changed_indices": [2, 3], "verdict": "fail",
              "issues": [], "severity": "major", "explanation": "shared", "error": None,
              "reason": "verdict_fail", "fallback": "verdict_fail"}
    for target in report["targets"]:
        target.update({"unit_id": "shared", "unit_indices": [1, 2, 3], "issues": [], "severity": "major", "explanation": "shared"})
    report["units"] = [shared]
    result = evaluate_benchmark(man, {"E": report}, {"E": usage})
    assert result["clear"]["correct"] == 2


def test_unit_membership_order_overlap_orphan_and_projection_mismatch_are_invalid() -> None:
    man = manifest(case("E", 1), case("E", 2))
    report, usage = artifacts([1, 2], {1: "pass", 2: "pass"})
    report["units"][0]["cue_indices"] = [99, 1]
    with pytest.raises(EvaluationError):
        evaluate_benchmark(man, {"E": report}, {"E": usage})
    report, usage = artifacts([1, 2], {1: "pass", 2: "pass"})
    report["units"][1]["cue_indices"] = [1, 2]
    with pytest.raises(EvaluationError):
        evaluate_benchmark(man, {"E": report}, {"E": usage})
    report, usage = artifacts([1, 2], {1: "pass", 2: "pass"})
    report["units"][0]["changed_indices"] = [1, 2]
    with pytest.raises(EvaluationError):
        evaluate_benchmark(man, {"E": report}, {"E": usage})
    report, usage = artifacts([1, 2], {1: "pass", 2: "pass"})
    report["units"][0]["issues"] = ["different"]
    with pytest.raises(EvaluationError):
        evaluate_benchmark(man, {"E": report}, {"E": usage})


def test_metadata_backend_model_accounting_and_requests_must_match() -> None:
    man = manifest(case("E", 1))
    report, usage = artifacts([1], {1: "pass"})
    for key, value in (("backend", "openrouter"), ("accounting", "metered_api"), ("model", "other")):
        report[key] = value
        with pytest.raises(EvaluationError):
            evaluate_benchmark(man, {"E": report}, {"E": usage})
        report[key] = usage[key]
    usage["requests"] = 2
    with pytest.raises(EvaluationError):
        evaluate_benchmark(man, {"E": report}, {"E": usage})


def test_cascade_hybrid_pair_is_accepted() -> None:
    report, usage = artifacts([1], {1: "pass"})
    report["backend"] = usage["backend"] = "cascade"
    report["accounting"] = usage["accounting"] = "hybrid"
    result = evaluate_benchmark(manifest(case("E", 1)), {"E": report}, {"E": usage})
    assert result["clear"]["correct"] == 1


def test_cli_qualified_rejected_invalid_duplicate_and_newline(tmp_path: Any) -> None:
    man = manifest(case("E", 1))
    report, usage = artifacts([1], {1: "pass"})
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    usage_path = tmp_path / "usage.json"
    output_path = tmp_path / "result.json"
    for path, value in ((manifest_path, man), (report_path, report), (usage_path, usage)):
        path.write_text(json.dumps(value), encoding="utf-8")
    command = [sys.executable, "-m", "utils.evaluate_semantic_benchmark", "--manifest", str(manifest_path),
               "--episode-report", f"E={report_path}", "--episode-usage", f"E={usage_path}", "--output", str(output_path)]
    qualified = subprocess.run(command, capture_output=True, text=True)
    assert qualified.returncode == 0
    assert output_path.read_text(encoding="utf-8").endswith("\n")
    report["targets"][0]["verdict"] = "fail"
    report["targets"][0]["reason"] = report["targets"][0]["fallback"] = "verdict_fail"
    report["verified_indices"] = []
    report["unresolved_indices"] = [1]
    report["fallback_indices"] = [1]
    report["units"][0]["verdict"] = "fail"
    report["units"][0]["reason"] = report["units"][0]["fallback"] = "verdict_fail"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode == 1
    duplicate = subprocess.run(command + ["--episode-report", f"E={report_path}"], capture_output=True, text=True)
    assert duplicate.returncode == 2
    report_path.write_text("not json", encoding="utf-8")
    invalid = subprocess.run(command, capture_output=True, text=True)
    assert invalid.returncode == 2
    assert invalid.stderr.strip() == "INVALID_ARTIFACTS"


def test_evaluator_has_no_transport_imports_or_calls() -> None:
    source = open("utils/evaluate_semantic_benchmark.py", encoding="utf-8").read()
    assert "import transport" not in source
    assert "subprocess" not in source
    assert "requests." not in source


def test_deepseek_metered_tokens_without_provider_cost_have_none_status() -> None:
    report, usage = artifacts([1], {1: "pass"})
    report["backend"] = usage["backend"] = "deepseek"
    report["accounting"] = usage["accounting"] = "metered_api"
    usage.pop("provider_reported_cost_usd")
    usage["provider_cost_is_billing_authoritative"] = False
    result = evaluate_benchmark(manifest(case("E", 1)), {"E": report}, {"E": usage})
    assert result["usage"]["cost_status"] == "none"
    assert result["usage"]["cost"] is None
    assert result["usage"]["tokens"]["input_tokens"]["value"] == 2
    assert result["usage"]["provider_ids"] == ["p"] and result["usage"]["model_ids"] == ["provider-m"] and result["usage"]["finishes"] == ["stop"]
