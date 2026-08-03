"""Conservative DeepSeek then OpenCode semantic verification cascade."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.analyze_text import join_text_lines
from utils.deepseek_transport import DEEPSEEK_DEFAULT_BASE_URL, normalize_deepseek_base_url, normalize_deepseek_model_id, request_json_object
from utils.opencode_transport import OpenCodePromptResult
from utils.shorten_helpers import load_api_key, load_json, validate_context_source
from utils.verify_subtitles_opencode import UNIT_VERIFIER_SYSTEM_PROMPT, _validate_items, _validate_runtime_args, _validate_server_config, verify_items, verify_items_live

CASCADE_BACKEND = "cascade"
CASCADE_ACCOUNTING = "hybrid"
CASCADE_POLICY = "confirm_primary_pass_or_polarity_with_singleton_tiebreak"
_PROJECTION_FIELDS = ("verdict", "issues", "severity", "explanation", "error", "reason", "fallback")
_SEVERITIES = {"none": 0, "minor": 1, "major": 2}
_TOKEN_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens")


def _fail(message: str) -> None:
    raise ValueError(message)


def _indices(value: Any, name: str) -> List[int]:
    if not isinstance(value, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in value):
        _fail(name)
    if len(set(value)) != len(value):
        _fail(name)
    return list(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(name)
    return value


def _string_list(value: Any, name: str) -> List[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        _fail(name)
    return list(value)


def _stage_usage(usage: Dict[str, Any], name: str) -> None:
    if not isinstance(usage, dict):
        _fail(f"{name}_usage")
    for field in ("successful_responses", "transport_retries", "transport_failures", "schema_failures"):
        if field in usage:
            _nonnegative_int(usage[field], f"{name}_{field}")
    if "usage_response_count" in usage:
        _nonnegative_int(usage["usage_response_count"], f"{name}_usage_response_count")
    if "schema_split_retry" in usage and not isinstance(usage["schema_split_retry"], bool):
        _fail(f"{name}_schema_split_retry")
    available = usage.get("token_usage_available")
    if not isinstance(available, bool):
        _fail(f"{name}_token_availability")
    if available:
        for field in _TOKEN_FIELDS:
            if field not in usage:
                _fail(f"{name}_missing_{field}")
            _nonnegative_int(usage[field], f"{name}_{field}")
    elif any(field in usage for field in _TOKEN_FIELDS):
        _fail(f"{name}_token_availability_mismatch")
    for field in ("provider_ids", "model_ids", "finishes"):
        if field in usage:
            _string_list(usage[field], f"{name}_{field}")


def _primary_units(primary_report: Dict[str, Any], expected_changed: Optional[List[int]] = None) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]], List[int]]:
    if not isinstance(primary_report, dict) or primary_report.get("schema_version") != 1 or primary_report.get("semantic_units_enabled") is not True or primary_report.get("context_mode") != "full":
        _fail("primary_metadata")
    targets = primary_report.get("targets")
    units = primary_report.get("units")
    if not isinstance(targets, list) or not isinstance(units, list) or not units:
        _fail("primary_schema")
    by_target: Dict[int, Dict[str, Any]] = {}
    by_unit: Dict[str, Dict[str, Any]] = {}
    for raw in units:
        if not isinstance(raw, dict) or not isinstance(raw.get("unit_id"), str) or not raw["unit_id"].strip() or raw["unit_id"] in by_unit:
            _fail("primary_unit")
        cue = _indices(raw.get("cue_indices"), "primary_cue_indices")
        changed = _indices(raw.get("changed_indices"), "primary_changed_indices")
        if not set(changed).issubset(cue):
            _fail("primary_unit_membership")
        if not isinstance(raw.get("issues"), list) or any(not isinstance(issue, str) for issue in raw["issues"]):
            _fail("primary_issues")
        by_unit[raw["unit_id"]] = raw
    for raw in targets:
        if not isinstance(raw, dict) or isinstance(raw.get("index"), bool) or not isinstance(raw.get("index"), int):
            _fail("primary_target")
        index = raw["index"]
        if index in by_target:
            _fail("primary_target_duplicate")
        unit_id = raw.get("unit_id")
        unit_indices = _indices(raw.get("unit_indices"), "primary_target_membership")
        if unit_id not in by_unit or by_unit[unit_id].get("cue_indices") != unit_indices or index not in by_unit[unit_id].get("changed_indices", []):
            _fail("primary_target_link")
        for field in _PROJECTION_FIELDS:
            if raw.get(field) != by_unit[unit_id].get(field):
                _fail("primary_projection_mismatch")
        if raw.get("verdict") not in {"pass", "fail", "uncertain", None}:
            _fail("primary_verdict")
        by_target[index] = raw
    memberships: Dict[int, str] = {}
    for unit_id, unit in by_unit.items():
        for index in unit["changed_indices"]:
            if index in memberships:
                _fail("primary_changed_overlap")
            memberships[index] = unit_id
    if set(memberships) != set(by_target):
        _fail("primary_linkage")
    selected = _indices(primary_report.get("selected_indices"), "primary_selected")
    if set(selected) != set(by_target):
        _fail("primary_selected")
    if expected_changed is not None:
        if set(by_target) != set(expected_changed) or set(selected) != set(expected_changed):
            _fail("primary_changed_coverage")
    for unit in by_unit.values():
        if not unit["changed_indices"]:
            _fail("primary_empty_unit")
    ordered = expected_changed if expected_changed is not None else list(memberships)
    return by_target, by_unit, list(ordered)


def _should_escalate(unit: Dict[str, Any], projection: Dict[str, Any]) -> bool:
    """Select a complete unit for arbiter confirmation using structured risk only."""
    verdict = projection.get("verdict")
    if verdict == "pass" and projection.get("error") is None and projection.get("reason") is None and projection.get("fallback") is None:
        return True
    issues = unit.get("issues")
    if not isinstance(issues, list) or any(not isinstance(issue, str) for issue in issues):
        _fail("primary_issues")
    return verdict != "pass" and "polarity" in issues


def build_arbiter_prior_report(candidate: List[Dict[str, Any]], primary_report: Dict[str, Any]) -> Dict[str, Any]:
    """Build the evaluator-compatible prior consumed by the second verifier."""
    _validate_items(candidate, candidate)
    by_target, by_unit, _ = _primary_units(primary_report)
    candidate_by_index = {int(item["index"]): item for item in candidate}
    if len(candidate_by_index) != len(candidate):
        _fail("candidate_indices")
    if not set(by_target).issubset(candidate_by_index):
        _fail("candidate_target_indices")
    outcomes: List[Dict[str, Any]] = []
    for index, target in by_target.items():
        verified = _should_escalate(by_unit[target["unit_id"]], target)
        outcomes.append({"index": index, "candidate_text": join_text_lines(candidate_by_index[index].get("text", "")), "outcome": "verified" if verified else "unresolved"})
    return {"schema_version": 1, "outcomes": outcomes}


def build_tiebreak_prior_report(candidate: List[Dict[str, Any]], primary_report: Dict[str, Any], arbiter_report: Dict[str, Any]) -> Dict[str, Any]:
    """Build a unit-complete prior for disagreements in the first arbiter stage."""
    _validate_items(candidate, candidate)
    by_target, by_unit, changed = _primary_units(primary_report)
    candidate_by_index = {int(item["index"]): item for item in candidate}
    if len(candidate_by_index) != len(candidate) or not set(by_target).issubset(candidate_by_index):
        _fail("candidate_target_indices")
    arbiter_targets = _stage_targets(arbiter_report, changed, "arbiter")
    selected: Set[int] = set()
    for unit in by_unit.values():
        representative = by_target[unit["changed_indices"][0]]
        if not _should_escalate(unit, representative):
            continue
        projection, valid = _aggregate_stage_rows([arbiter_targets[index] for index in unit["changed_indices"]])
        if not valid or projection.get("verdict") != representative.get("verdict"):
            selected.update(unit["changed_indices"])
    outcomes = [{"index": index, "candidate_text": join_text_lines(candidate_by_index[index].get("text", "")), "outcome": "verified" if index in selected else "unresolved"} for index in changed]
    return {"schema_version": 1, "outcomes": outcomes}


def _projection_ok(row: Dict[str, Any]) -> bool:
    verdict = row.get("verdict")
    if verdict not in {"pass", "fail", "uncertain", None} or not isinstance(row.get("issues"), list) or any(not isinstance(x, str) for x in row["issues"]):
        return False
    severity = row.get("severity")
    if severity is None:
        severity = "none"
    if severity not in _SEVERITIES:
        return False
    if not isinstance(row.get("explanation"), (str, type(None))) or isinstance(row.get("explanation"), str) and not row["explanation"].strip():
        return False
    if verdict == "pass":
        return row.get("error") is None and row.get("reason") is None and row.get("fallback") is None
    if verdict in {"fail", "uncertain"}:
        return row.get("error") is None and row.get("reason") == f"verdict_{verdict}" and row.get("fallback") == f"verdict_{verdict}"
    return all(isinstance(row.get(x), str) and row.get(x).strip() and not row.get(x).startswith("verdict_") for x in ("error", "reason", "fallback"))


def _safe_projection(verdict: Optional[str], issues: List[str], severity: str, explanation: Optional[str], error: Optional[str] = None) -> Dict[str, Any]:
    if verdict == "pass":
        return {"verdict": "pass", "issues": issues, "severity": severity, "explanation": explanation, "error": None, "reason": None, "fallback": None}
    if verdict in {"fail", "uncertain"}:
        return {"verdict": verdict, "issues": issues, "severity": severity, "explanation": explanation, "error": None, "reason": f"verdict_{verdict}", "fallback": f"verdict_{verdict}"}
    code = error or "cascade_arbiter_projection_failure"
    safe_explanation = explanation if isinstance(explanation, str) and explanation.strip() else code
    return {
        "verdict": None,
        "issues": issues,
        "severity": severity,
        "explanation": safe_explanation,
        "error": code,
        "reason": code,
        "fallback": code,
    }


def _aggregate_stage_rows(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], bool]:
    """Aggregate rows by primary membership and report whether all rows are valid."""
    if not rows or not all(_projection_ok(row) for row in rows):
        return _safe_projection(None, [], "none", "cascade_arbiter_projection_failure", "cascade_arbiter_projection_failure"), False
    issues: List[str] = []
    explanations: List[str] = []
    verdicts = [row["verdict"] for row in rows]
    for row in rows:
        for issue in row["issues"]:
            if issue not in issues:
                issues.append(issue)
        if row.get("explanation") and row["explanation"] not in explanations:
            explanations.append(row["explanation"])
    verdict: Optional[str] = None if any(value is None for value in verdicts) else "fail" if any(value == "fail" for value in verdicts) else "uncertain" if any(value == "uncertain" for value in verdicts) else "pass"
    severity = max(((row.get("severity") or "none") for row in rows), key=lambda value: _SEVERITIES[value])
    explanation = " | ".join(explanations)[:2000] or ("cascade_arbiter_projection_failure" if verdict is None else "cascade_arbiter_projection")
    return _safe_projection(verdict, issues, severity, explanation), True


def _stage_targets(report: Dict[str, Any], changed: List[int], name: str) -> Dict[int, Dict[str, Any]]:
    if report.get("schema_version") != 1 or report.get("semantic_units_enabled") is not True or report.get("context_mode") != "full":
        _fail(f"{name}_metadata")
    raw_targets = report.get("targets")
    if not isinstance(raw_targets, list):
        _fail(f"{name}_targets")
    targets: Dict[int, Dict[str, Any]] = {}
    for row in raw_targets:
        if not isinstance(row, dict) or isinstance(row.get("index"), bool) or not isinstance(row.get("index"), int) or row["index"] in targets:
            _fail(f"{name}_target")
        targets[row["index"]] = row
    if set(targets) != set(changed):
        _fail(f"{name}_coverage")
    return targets


def _optional_stage_targets(report: Any, changed: List[int]) -> Dict[int, Dict[str, Any]]:
    """Read an optional tiebreak stage; malformed rows remain fail-closed sentinels."""
    if not isinstance(report, dict) or report.get("schema_version") != 1 or report.get("semantic_units_enabled") is not True or report.get("context_mode") != "full":
        return {}
    raw_targets = report.get("targets")
    if not isinstance(raw_targets, list):
        return {}
    targets: Dict[int, Dict[str, Any]] = {}
    for row in raw_targets:
        if isinstance(row, dict) and isinstance(row.get("index"), int) and not isinstance(row.get("index"), bool) and row["index"] not in targets and row["index"] in changed:
            targets[row["index"]] = row
    return targets


def merge_cascade_results(original: List[Dict[str, Any]], candidate: List[Dict[str, Any]], primary_report: Dict[str, Any], primary_usage: Dict[str, Any], arbiter_report: Dict[str, Any], arbiter_usage: Dict[str, Any], *, primary_model: str, arbiter_model: str, tiebreak_report: Optional[Dict[str, Any]] = None, tiebreak_usage: Optional[Dict[str, Any]] = None) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Merge stages without allowing an incomplete or unsafe arbiter result to pass."""
    _validate_items(original, candidate)
    primary_snapshot = copy.deepcopy(primary_report)
    arbiter_snapshot = copy.deepcopy(arbiter_report)
    tiebreak_snapshot = copy.deepcopy(tiebreak_report) if tiebreak_report is not None else None
    primary_usage_snapshot = copy.deepcopy(primary_usage)
    arbiter_usage_snapshot = copy.deepcopy(arbiter_usage)
    tiebreak_usage_snapshot = copy.deepcopy(tiebreak_usage) if tiebreak_usage is not None else None
    if (tiebreak_snapshot is None) != (tiebreak_usage_snapshot is None):
        _fail("tiebreak_stage_pair")
    original_by_index = {int(x["index"]): x for x in original}
    candidate_by_index = {int(x["index"]): x for x in candidate}
    changed = [int(item["index"]) for item in original if join_text_lines(item.get("text", "")) != join_text_lines(candidate_by_index[int(item["index"])].get("text", ""))]
    _stage_usage(primary_usage_snapshot, "primary")
    _stage_usage(arbiter_usage_snapshot, "arbiter")
    if tiebreak_usage_snapshot is not None:
        _stage_usage(tiebreak_usage_snapshot, "tiebreaker")
    for stage_name, stage_report in (("primary", primary_snapshot), ("arbiter", arbiter_snapshot), ("tiebreaker", tiebreak_snapshot)):
        if stage_report is None:
            continue
        if not isinstance(stage_report, dict):
            _fail(f"{stage_name}_report")
        for field in ("requests", "transport_retries", "failures"):
            if field in stage_report:
                _nonnegative_int(stage_report[field], f"{stage_name}_{field}")
        if "schema_split_retry" in stage_report and not isinstance(stage_report["schema_split_retry"], bool):
            _fail(f"{stage_name}_schema_split_retry")
    primary_report_requests = _nonnegative_int(primary_snapshot.get("requests"), "primary_requests")
    arbiter_report_requests = _nonnegative_int(arbiter_snapshot.get("requests"), "arbiter_requests")
    primary_usage_requests = _nonnegative_int(primary_usage_snapshot.get("requests"), "primary_usage_requests")
    arbiter_usage_requests = _nonnegative_int(arbiter_usage_snapshot.get("requests"), "arbiter_usage_requests")
    if primary_usage_requests != primary_report_requests or arbiter_usage_requests != arbiter_report_requests:
        _fail("stage_requests")
    tiebreak_report_requests = tiebreak_usage_requests = 0
    if tiebreak_snapshot is not None and tiebreak_usage_snapshot is not None:
        tiebreak_report_requests = _nonnegative_int(tiebreak_snapshot.get("requests"), "tiebreaker_requests")
        tiebreak_usage_requests = _nonnegative_int(tiebreak_usage_snapshot.get("requests"), "tiebreaker_usage_requests")
        if tiebreak_report_requests != tiebreak_usage_requests:
            _fail("stage_requests")
    primary_targets, primary_units, changed = _primary_units(primary_snapshot, changed)
    final_units = copy.deepcopy(primary_units)
    arbiter_targets = _stage_targets(arbiter_snapshot, changed, "arbiter")
    tiebreak_targets = _optional_stage_targets(tiebreak_snapshot, changed) if tiebreak_snapshot is not None else None
    unit_for: Dict[int, str] = {i: uid for uid, unit in primary_units.items() for i in unit["changed_indices"]}
    final = copy.deepcopy(candidate)
    final_targets: List[Dict[str, Any]] = []
    verified: List[int] = []
    escalated: List[str] = []
    tiebreak_units: List[str] = []
    tiebreak_indices: List[int] = []
    for uid, unit in final_units.items():
        p = primary_targets[unit["changed_indices"][0]]
        if _should_escalate(unit, p):
            escalated.append(uid)
            rows = [arbiter_targets[i] for i in unit["changed_indices"]]
            arbiter_projection, arbiter_valid = _aggregate_stage_rows(rows)
            primary_verdict = p.get("verdict")
            tiebreak_needed = not arbiter_valid or arbiter_projection.get("verdict") != primary_verdict
            if tiebreak_snapshot is None:
                projection = arbiter_projection
            elif tiebreak_needed:
                tiebreak_units.append(uid)
                tiebreak_indices.extend(unit["changed_indices"])
                if tiebreak_targets is None:
                    projection = _safe_projection(None, [], "none", "tiebreak_missing", "tiebreak_missing")
                else:
                    tiebreak_rows = [tiebreak_targets[i] for i in unit["changed_indices"] if i in tiebreak_targets]
                    tiebreak_projection, tiebreak_valid = _aggregate_stage_rows(tiebreak_rows)
                    semantic = [value for value in (primary_verdict, arbiter_projection.get("verdict"), tiebreak_projection.get("verdict")) if value in {"pass", "fail", "uncertain"}]
                    counts = {value: semantic.count(value) for value in {"pass", "fail", "uncertain"}}
                    majority = next((value for value in ("pass", "fail", "uncertain") if counts[value] >= 2), None)
                    if majority is not None:
                        source = tiebreak_projection if tiebreak_projection.get("verdict") == majority else arbiter_projection if arbiter_projection.get("verdict") == majority else p
                        projection = copy.deepcopy(source)
                    elif not tiebreak_valid or not arbiter_valid or any(value not in {"pass", "fail", "uncertain"} for value in (primary_verdict, arbiter_projection.get("verdict"), tiebreak_projection.get("verdict"))):
                        projection = _safe_projection(None, [], "none", "tiebreak_projection_failure", "tiebreak_projection_failure")
                    else:
                        projection = _safe_projection("uncertain", [], "none", "tiebreak_three_way_uncertain")
            else:
                projection = arbiter_projection
        else:
            projection = {field: copy.deepcopy(p.get(field)) for field in _PROJECTION_FIELDS}
        for index in unit["changed_indices"]:
            target = copy.deepcopy(primary_targets[index])
            target.update(copy.deepcopy(projection))
            target["unit_id"] = uid
            target["unit_indices"] = copy.deepcopy(unit["cue_indices"])
            final_targets.append(target)
            if target["verdict"] == "pass" and target["error"] is None and target["reason"] is None and target["fallback"] is None:
                verified.append(index)
            else:
                for item in final:
                    if int(item["index"]) == index:
                        item["text"] = copy.deepcopy(original_by_index[index].get("text"))
        unit.update(copy.deepcopy(projection))
    final_targets.sort(key=lambda x: changed.index(x["index"]))
    unresolved = [i for i in changed if i not in verified]
    present_reports = [("primary", primary_snapshot), ("arbiter", arbiter_snapshot)]
    present_usages = [("primary", primary_usage_snapshot), ("arbiter", arbiter_usage_snapshot)]
    if tiebreak_snapshot is not None and tiebreak_usage_snapshot is not None:
        present_reports.append(("tiebreaker", tiebreak_snapshot))
        present_usages.append(("tiebreaker", tiebreak_usage_snapshot))
    requests = sum(_nonnegative_int(stage.get("requests"), f"{name}_requests") for name, stage in present_reports)
    transport_retries = sum(_nonnegative_int(stage.get("transport_retries", 0), f"{name}_transport_retries") for name, stage in present_reports)
    transport_failures = sum(_nonnegative_int(stage.get("transport_failures", 0), f"{name}_transport_failures") for name, stage in present_usages)
    schema_failures = sum(_nonnegative_int(stage.get("schema_failures", 0), f"{name}_schema_failures") for name, stage in present_usages)
    failures = sum(_nonnegative_int(stage.get("failures", 0), f"{name}_failures") for name, stage in present_reports)
    schema_split_retry = any(bool(stage.get("schema_split_retry")) for _, stage in present_reports)
    routing = {"policy": CASCADE_POLICY, "primary_unit_count": len(primary_units), "escalated_unit_ids": escalated, "escalated_changed_indices": [i for i in changed if unit_for[i] in escalated], "escalated_unit_count": len(escalated), "escalated_changed_count": sum(len(final_units[u]["changed_indices"]) for u in escalated), "tiebreak_unit_ids": tiebreak_units, "tiebreak_changed_indices": tiebreak_indices, "tiebreak_unit_count": len(tiebreak_units), "tiebreak_changed_count": len(tiebreak_indices)}
    if tiebreak_snapshot is not None:
        routing["tiebreak_batch_size"] = 1
    report = {"schema_version": 1, "backend": CASCADE_BACKEND, "accounting": CASCADE_ACCOUNTING, "model": f"{primary_model}+{arbiter_model}", "semantic_units_enabled": True, "context_mode": "full", "total_count": len(original), "changed_count": len(changed), "selected_count": len(changed), "verified_count": len(verified), "unresolved_count": len(unresolved), "fallback_count": len(unresolved), "selected_indices": list(changed), "verified_indices": verified, "unresolved_indices": unresolved, "fallback_indices": unresolved, "status": "completed" if not unresolved else "completed_with_unresolved", "targets": final_targets, "units": [copy.deepcopy(unit) for unit in final_units.values()], "requests": requests, "transport_retries": transport_retries, "transport_failures": transport_failures, "schema_failures": schema_failures, "failures": failures, "schema_split_retry": schema_split_retry, "routing": routing}
    report["stages"] = {name: copy.deepcopy(stage) for name, stage in present_reports}
    usage = _merge_usage(present_usages, report["model"], requests)
    for item in final:
        index = int(item["index"])
        if index in changed:
            row = next(x for x in final_targets if x["index"] == index)
            row["output"] = join_text_lines(item.get("text", ""))
    return final, report, usage


def _merge_usage(stages: List[Tuple[str, Dict[str, Any]]], model: str, requests: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {"backend": CASCADE_BACKEND, "accounting": CASCADE_ACCOUNTING, "model": model, "requests": requests, "stages": {name: copy.deepcopy(stage) for name, stage in stages}, "successful_responses": sum(_nonnegative_int(stage.get("successful_responses", 0), f"{name}_successful_responses") for name, stage in stages), "transport_retries": sum(_nonnegative_int(stage.get("transport_retries", 0), f"{name}_transport_retries") for name, stage in stages), "transport_failures": sum(_nonnegative_int(stage.get("transport_failures", 0), f"{name}_transport_failures") for name, stage in stages), "schema_failures": sum(_nonnegative_int(stage.get("schema_failures", 0), f"{name}_schema_failures") for name, stage in stages), "schema_split_retry": any(bool(stage.get("schema_split_retry")) for _, stage in stages), "provider_ids": sorted({item for name, stage in stages for item in stage.get("provider_ids", [])}), "model_ids": sorted({item for name, stage in stages for item in stage.get("model_ids", [])}), "finishes": sorted({item for name, stage in stages for item in stage.get("finishes", [])}), "provider_cost_is_billing_authoritative": False}
    if any("usage_response_count" in stage for _, stage in stages):
        result["usage_response_count"] = sum(_nonnegative_int(stage.get("usage_response_count", 0), f"{name}_usage_response_count") for name, stage in stages)
    if all(stage.get("token_usage_available") is True for _, stage in stages):
        result["token_usage_available"] = True
        for field in _TOKEN_FIELDS:
            result[field] = sum(_nonnegative_int(stage[field], f"{name}_{field}") for name, stage in stages)
    else:
        result["token_usage_available"] = False
    primary = stages[0][1]
    if all(x in primary for x in ("input_tokens", "output_tokens", "cache_read_tokens")) and all(not isinstance(primary.get(x), bool) and isinstance(primary.get(x), int) and primary.get(x) >= 0 for x in ("input_tokens", "output_tokens", "cache_read_tokens")) and primary["cache_read_tokens"] <= primary["input_tokens"]:
        result["deepseek_tariff_estimate_usd"] = (primary["input_tokens"] - primary["cache_read_tokens"]) * 0.14 / 1_000_000 + primary["cache_read_tokens"] * 0.0028 / 1_000_000 + primary["output_tokens"] * 0.28 / 1_000_000
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed DeepSeek/OpenCode semantic cascade.")
    parser.add_argument("original"); parser.add_argument("candidate"); parser.add_argument("--context-source", required=True)
    parser.add_argument("--deepseek-api-key"); parser.add_argument("--deepseek-base-url", default=DEEPSEEK_DEFAULT_BASE_URL); parser.add_argument("--primary-model", default="deepseek-v4-flash"); parser.add_argument("--primary-timeout", type=float, default=180); parser.add_argument("--primary-temperature", type=float, default=0); parser.add_argument("--primary-concurrency", type=int, default=1); parser.add_argument("--primary-transport-retries", type=int, default=1)
    parser.add_argument("--arbiter-model", default="openai/gpt-5.6-sol"); parser.add_argument("--server-url"); parser.add_argument("--hostname", default="127.0.0.1"); parser.add_argument("--port", type=int); parser.add_argument("--arbiter-concurrency", type=int, default=3); parser.add_argument("--arbiter-transport-retries", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--context-window", type=int, default=3); parser.add_argument("--schema-retries", type=int, choices=[0, 1], default=1); parser.add_argument("--semantic-unit-max-cues", type=int, default=3); parser.add_argument("--max-gap", type=float, default=0.3); parser.add_argument("--output-dir", default="output/cascade_semantic_verified")
    return parser


def _validate_config(args: argparse.Namespace) -> None:
    _validate_runtime_args(args.primary_model, args.context_window, args.batch_size, args.primary_concurrency, args.primary_transport_retries, args.schema_retries, args.semantic_unit_max_cues, args.max_gap)
    _validate_runtime_args(args.arbiter_model, args.context_window, args.batch_size, args.arbiter_concurrency, args.arbiter_transport_retries, args.schema_retries, args.semantic_unit_max_cues, args.max_gap)
    _validate_server_config(args.server_url, args.hostname, args.port)
    normalize_deepseek_model_id(args.primary_model); normalize_deepseek_base_url(args.deepseek_base_url)
    if isinstance(args.primary_timeout, bool) or not isinstance(args.primary_timeout, (int, float)) or not math.isfinite(float(args.primary_timeout)) or args.primary_timeout <= 0 or isinstance(args.primary_temperature, bool) or not isinstance(args.primary_temperature, (int, float)) or not math.isfinite(float(args.primary_temperature)) or not 0 <= args.primary_temperature <= 2:
        raise ValueError("invalid_primary_settings")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_config(args)
        original = load_json(Path(args.original)); candidate = load_json(Path(args.candidate)); context = load_json(Path(args.context_source))
        _validate_items(original, candidate); validate_context_source(original, context)
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr); return 2
    api_key = args.deepseek_api_key if args.deepseek_api_key is not None else load_api_key("DEEPSEEK_API_KEY")
    if not api_key or not api_key.strip():
        print("DEEPSEEK_API_KEY is required", file=sys.stderr); return 1
    async def request(prompt: str, model: str) -> Optional[OpenCodePromptResult]:
        return await request_json_object(api_key=api_key, model=model, system_prompt=UNIT_VERIFIER_SYSTEM_PROMPT, user_prompt=prompt, base_url=args.deepseek_base_url, timeout=args.primary_timeout, temperature=args.primary_temperature)
    try:
        _primary_output, primary_report, primary_usage = asyncio.run(verify_items(original, candidate, model=args.primary_model, context_items=context, context_window=args.context_window, batch_size=args.batch_size, concurrency=args.primary_concurrency, transport_retries=args.primary_transport_retries, schema_retries=args.schema_retries, request_callable=request, semantic_units=True, semantic_unit_max_cues=args.semantic_unit_max_cues, semantic_unit_max_gap_sec=args.max_gap, backend="deepseek", accounting="metered_api", cost_is_billing_authoritative=False))
        prior = build_arbiter_prior_report(candidate, primary_report)
        _arbiter_output, arbiter_report, arbiter_usage = verify_items_live(original, candidate, model=args.arbiter_model, context_items=context, context_window=args.context_window, batch_size=args.batch_size, concurrency=args.arbiter_concurrency, transport_retries=args.arbiter_transport_retries, schema_retries=args.schema_retries, review_report=prior, server_url=args.server_url, hostname=args.hostname, port=args.port, semantic_units=True, semantic_unit_max_cues=args.semantic_unit_max_cues, semantic_unit_max_gap_sec=args.max_gap, structured_output=True)
        tiebreak_prior = build_tiebreak_prior_report(candidate, primary_report, arbiter_report)
        tiebreak_report: Optional[Dict[str, Any]] = None
        tiebreak_usage: Optional[Dict[str, Any]] = None
        if any(row.get("outcome") == "verified" for row in tiebreak_prior["outcomes"]):
            _tiebreak_output, tiebreak_report, tiebreak_usage = verify_items_live(original, candidate, model=args.arbiter_model, context_items=context, context_window=args.context_window, batch_size=1, concurrency=args.arbiter_concurrency, transport_retries=args.arbiter_transport_retries, schema_retries=args.schema_retries, review_report=tiebreak_prior, server_url=args.server_url, hostname=args.hostname, port=args.port, semantic_units=True, semantic_unit_max_cues=args.semantic_unit_max_cues, semantic_unit_max_gap_sec=args.max_gap, structured_output=True)
        output, report, usage = merge_cascade_results(original, candidate, primary_report, primary_usage, arbiter_report, arbiter_usage, primary_model=args.primary_model, arbiter_model=args.arbiter_model, tiebreak_report=tiebreak_report, tiebreak_usage=tiebreak_usage)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr); return 2
    destination = Path(args.output_dir); destination.mkdir(parents=True, exist_ok=True); stem = Path(args.candidate).stem + "_cascade_verified"
    for suffix, value in ((".json", output), (".report.json", report), (".usage.json", usage)):
        (destination / f"{stem}{suffix}").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
