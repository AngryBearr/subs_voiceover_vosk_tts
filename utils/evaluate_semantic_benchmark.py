"""Offline, tracked semantic benchmark evaluator.

This module deliberately contains no provider or transport imports.  Reports and
usage files are untrusted data; validation is fail closed before scoring.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

Pair = Tuple[str, int]
KNOWN_PAIRS = {("opencode", "subscription"), ("openrouter", "metered_api"), ("ollama_cloud", "subscription")}
VERDICTS = {"pass", "fail", "uncertain", None}


class EvaluationError(ValueError):
    """A structural, configuration, or artifact validation error."""

    def __init__(self, kind: str, message: str = "invalid") -> None:
        super().__init__(message)
        self.kind = kind


def _bad(kind: str, message: str = "invalid") -> None:
    raise EvaluationError(kind, message)


def _obj(value: Any, name: str, kind: str = "artifacts") -> Dict[str, Any]:
    if not isinstance(value, dict):
        _bad(kind, name)
    return value


def _list(value: Any, name: str, kind: str = "artifacts") -> List[Any]:
    if not isinstance(value, list):
        _bad(kind, name)
    return value


def _int(value: Any, name: str, kind: str = "artifacts") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _bad(kind, name)
    return value


def _nonblank(value: Any, name: str, kind: str = "artifacts") -> str:
    if not isinstance(value, str) or not value.strip():
        _bad(kind, name)
    return value


def _indices(value: Any, name: str, kind: str = "artifacts") -> List[int]:
    values = _list(value, name, kind)
    result = [_int(item, name, kind) for item in values]
    if len(result) != len(set(result)):
        _bad(kind, name)
    return result


def _same_set(left: Iterable[int], right: Iterable[int]) -> bool:
    return set(left) == set(right)


def _metadata(source: Mapping[str, Any], key: str) -> Any:
    if key in source:
        return source[key]
    metadata = source.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def _manifest_cases(manifest: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Set[str]]:
    _obj(manifest, "manifest", "manifest")
    cases = _list(manifest.get("cases"), "cases", "manifest")
    if not cases:
        _bad("manifest", "empty cases")
    result: Dict[str, Dict[str, Any]] = {}
    pairs: Set[Pair] = set()
    episodes: Set[str] = set()
    for case in cases:
        item = _obj(case, "case", "manifest")
        episode = _nonblank(item.get("episode"), "episode", "manifest")
        index = _int(item.get("index"), "index", "manifest")
        expected = item.get("expected")
        if expected not in {"pass", "fail"}:
            _bad("manifest", "expected")
        clear = item.get("clear")
        if not isinstance(clear, bool):
            _bad("manifest", "clear")
        pair = (episode, index)
        if pair in pairs:
            _bad("manifest", "duplicate case")
        pairs.add(pair)
        episodes.add(episode)
        result[f"{episode}\x00{index}"] = {"episode": episode, "index": index, "expected": expected, "clear": clear}
    return result, episodes


def _projection(target: Dict[str, Any], index: int) -> None:
    verdict = target.get("verdict")
    if verdict not in VERDICTS:
        _bad("artifacts", "verdict")
    error, reason, fallback = target.get("error"), target.get("reason"), target.get("fallback")
    verified = target.get("verified")
    unresolved = target.get("unresolved")
    fallback_set = target.get("fallback_indices")
    # Target-level booleans are accepted; aggregate sets are checked separately.
    issues = target.get("issues")
    if issues is not None:
        if not isinstance(issues, list) or any(not isinstance(item, str) for item in issues):
            _bad("artifacts", "issues")
    severity = target.get("severity")
    if severity is not None and (not isinstance(severity, str) or not severity.strip()):
        _bad("artifacts", "severity")
    explanation = target.get("explanation")
    if explanation is not None and (not isinstance(explanation, str) or not explanation.strip()):
        _bad("artifacts", "explanation")
    if verdict == "pass":
        if error is not None or reason is not None or fallback is not None:
            _bad("artifacts", "pass projection")
    elif verdict in {"fail", "uncertain"}:
        if error is not None or reason != f"verdict_{verdict}" or fallback != f"verdict_{verdict}":
            _bad("artifacts", "semantic projection")
    else:
        if not all(isinstance(value, str) and value.strip() for value in (error, reason, fallback)):
            _bad("artifacts", "technical projection")
        if any(isinstance(value, str) and value.startswith("verdict_") for value in (error, reason, fallback)):
            _bad("artifacts", "technical reason")


def _set_field(report: Dict[str, Any], *names: str) -> Set[int]:
    for name in names:
        if name in report:
            return set(_indices(report[name], name))
    return set()


def _validate_episode(episode: str, report: Dict[str, Any], wanted: Set[int]) -> Dict[int, Dict[str, Any]]:
    if report.get("schema_version") != 1 or report.get("context_mode") != "full" or report.get("semantic_units_enabled") is not True:
        _bad("artifacts", "report metadata")
    targets = _list(report.get("targets"), "targets")
    got: Dict[int, Dict[str, Any]] = {}
    for raw in targets:
        target = _obj(raw, "target")
        index = _int(target.get("index"), "target index")
        if index in got or index not in wanted:
            _bad("artifacts", "target indices")
        _nonblank(target.get("unit_id"), "unit_id")
        units = _indices(target.get("unit_indices"), "unit_indices")
        if index not in units:
            _bad("artifacts", "unit membership")
        _projection(target, index)
        got[index] = target
    if set(got) != wanted:
        _bad("artifacts", "missing target")
    selected = _indices(report.get("selected_indices"), "selected_indices")
    if set(selected) != wanted or len(selected) != len(wanted):
        _bad("artifacts", "selected indices")
    verified = _set_field(report, "verified_indices", "verified")
    unresolved = _set_field(report, "unresolved_indices", "unresolved")
    fallback = _set_field(report, "fallback_indices", "fallback")
    # ``fallback`` is an annotation on unresolved targets, so semantic and
    # technical fallback indices necessarily overlap unresolved_indices.
    if (verified | unresolved | fallback) != wanted or (verified & unresolved) or (verified & fallback):
        _bad("artifacts", "report sets")
    for index, target in got.items():
        verdict = target.get("verdict")
        if verdict == "pass" and (index not in verified or index in unresolved or index in fallback):
            _bad("artifacts", "pass set")
        if verdict in {"fail", "uncertain", None} and (index not in unresolved or index not in fallback or index in verified):
            _bad("artifacts", "unresolved set")

    units = _list(report.get("units"), "units")
    seen: Dict[str, Dict[str, Any]] = {}
    for raw in units:
        unit = _obj(raw, "unit")
        unit_id = _nonblank(unit.get("unit_id"), "unit_id")
        if unit_id in seen:
            _bad("artifacts", "duplicate unit")
        cue_indices = _indices(unit.get("cue_indices"), "cue_indices")
        changed = _indices(unit.get("changed_indices"), "changed_indices")
        if not set(changed).issubset(wanted) or not set(cue_indices).issuperset(set(changed)):
            _bad("artifacts", "unit indices")
        for other in seen.values():
            if set(cue_indices) & set(other["cue_indices"]):
                _bad("artifacts", "overlapping unit cues")
        seen[unit_id] = unit
    if not seen:
        _bad("artifacts", "units")
    for index, target in got.items():
        unit_id = target["unit_id"]
        if unit_id not in seen:
            _bad("artifacts", "missing linked unit")
        unit = seen[unit_id]
        if unit["cue_indices"] != target["unit_indices"]:
            _bad("artifacts", "unit membership mismatch")
        for field in ("verdict", "issues", "severity", "explanation", "error", "reason", "fallback"):
            if unit.get(field) != target.get(field):
                _bad("artifacts", "unit projection mismatch")
        if index not in unit["changed_indices"]:
            _bad("artifacts", "changed membership")
    memberships: Dict[int, str] = {}
    for unit_id, unit in seen.items():
        for index in unit["changed_indices"]:
            if index in memberships and memberships[index] != unit_id:
                _bad("artifacts", "overlapping units")
            memberships[index] = unit_id
    if set(memberships) != wanted:
        _bad("artifacts", "orphan changed indices")
    # Context-only cue members are permitted, but no changed member may be one.
    return got


def _consistency(reports: Mapping[str, Dict[str, Any]], usages: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    values: Dict[str, str] = {}
    for episode, artifact in list(reports.items()) + list(usages.items()):
        for key in ("model", "backend", "accounting"):
            value = _metadata(artifact, key)
            if not isinstance(value, str) or not value.strip():
                _bad("artifacts", f"missing {key}")
            if key in values and values[key] != value:
                _bad("artifacts", f"inconsistent {key}")
            values[key] = value
        if (values["backend"], values["accounting"]) not in KNOWN_PAIRS:
            _bad("artifacts", "unknown backend/accounting")
    authoritative: Optional[bool] = None
    for episode, usage in usages.items():
        report = reports[episode]
        if _metadata(report, "model") != _metadata(usage, "model") or _metadata(report, "backend") != _metadata(usage, "backend") or _metadata(report, "accounting") != _metadata(usage, "accounting"):
            _bad("artifacts", "report usage metadata")
        rr, ur = report.get("requests"), usage.get("requests")
        if rr != ur or not isinstance(rr, int) or isinstance(rr, bool):
            _bad("artifacts", "requests")
        emitted = _metadata(usage, "usage_metadata_emitted")
        if emitted is not None and not isinstance(emitted, bool):
            _bad("artifacts", "usage metadata flag")
        if emitted is True:
            authority = _metadata(usage, "authoritative")
            if not isinstance(authority, bool):
                _bad("artifacts", "authority")
            if authoritative is not None and authoritative != authority:
                _bad("artifacts", "authority mismatch")
            authoritative = authority
    return {"model": values["model"], "backend": values["backend"], "accounting": values["accounting"], "authoritative": authoritative}


def _usage(artifacts: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    totals = {"requests": 0, "successful_responses": 0, "transport_retries": 0, "transport_failures": 0, "schema_failures": 0}
    token_fields = ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens")
    tokens: Dict[str, Any] = {name: {"value": None, "availability": "none"} for name in token_fields}
    token_present: Dict[str, int] = {name: 0 for name in token_fields}
    costs: List[float] = []
    cost_present = 0
    cost_authorities: Set[bool] = set()
    provider_ids: Set[str] = set()
    model_ids: Set[str] = set()
    finishes: Set[str] = set()
    for usage in artifacts.values():
        for key in totals:
            value = usage.get(key, 0)
            _int(value, key)
            totals[key] += value
        for key in ("provider_ids", "model_ids", "finishes"):
            value = usage.get(key, [])
            for item in _list(value, key):
                _nonblank(item, key)
                (provider_ids if key == "provider_ids" else model_ids if key == "model_ids" else finishes).add(item)
        if "usage_response_count" in usage:
            response_count = usage["usage_response_count"]
            if isinstance(response_count, bool) or not isinstance(response_count, int) or response_count < 0:
                _bad("artifacts", "usage_response_count")
        available = usage.get("token_usage_available")
        if not isinstance(available, bool):
            _bad("artifacts", "token_usage_available")
        present_core = False
        for name in token_fields:
            value = usage.get(name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    _bad("artifacts", "token value")
                tokens[name]["value"] = (tokens[name]["value"] or 0) + value
                token_present[name] += 1
                if name in ("input_tokens", "output_tokens", "total_tokens"):
                    present_core = True
            elif name in usage and usage[name] is not None:
                _bad("artifacts", "token value")
        if available and not present_core:
            _bad("artifacts", "missing core token metadata")
        if not available and any(name in usage and usage[name] is not None for name in token_fields):
            _bad("artifacts", "token availability mismatch")
        if "provider_reported_cost_usd" in usage and usage["provider_reported_cost_usd"] is not None:
            cost = usage["provider_reported_cost_usd"]
            if not isinstance(cost, (int, float)) or isinstance(cost, bool) or not math.isfinite(float(cost)) or cost < 0:
                _bad("artifacts", "cost")
            costs.append(float(cost))
            cost_present += 1
            authority = usage.get("provider_cost_is_billing_authoritative")
            if not isinstance(authority, bool):
                _bad("artifacts", "cost authority")
            cost_authorities.add(authority)
        elif "provider_cost_is_billing_authoritative" in usage and usage["provider_cost_is_billing_authoritative"] is not None:
            if not isinstance(usage["provider_cost_is_billing_authoritative"], bool):
                _bad("artifacts", "cost authority")
    if len(cost_authorities) > 1:
        _bad("artifacts", "mixed cost authority")
    if cost_present == 0:
        cost_status = "none"
    elif cost_present < len(artifacts):
        cost_status = "partial"
    elif cost_authorities == {True}:
        cost_status = "billing_authoritative"
    else:
        cost_status = "informational"
    for name in token_fields:
        if token_present[name] == len(artifacts):
            tokens[name]["availability"] = "all"
        elif token_present[name]:
            tokens[name]["availability"] = "partial"
    return {**totals, "tokens": tokens, "cost": sum(costs) if costs else None, "cost_status": cost_status,
            "provider_ids": sorted(provider_ids), "model_ids": sorted(model_ids), "finishes": sorted(finishes)}


def evaluate_benchmark(manifest: Dict[str, Any], reports: Mapping[str, Dict[str, Any]], usages: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Validate and score tracked report/usage artifacts without I/O or network calls."""
    cases, episodes = _manifest_cases(manifest)
    if set(reports) != episodes or set(usages) != episodes:
        _bad("artifacts", "episode set")
    validated: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for episode in episodes:
        _obj(reports[episode], "report")
        _obj(usages[episode], "usage")
        wanted = {case["index"] for case in cases.values() if case["episode"] == episode}
        validated[episode] = _validate_episode(episode, reports[episode], wanted)
    consistency = _consistency(reports, usages)
    usage = _usage(usages)
    rows: List[Dict[str, Any]] = []
    buckets: Dict[str, Dict[str, Any]] = {"clear": {"total": 0, "correct": 0, "false_passes": [], "false_rejects": [], "uncertain": [], "technical": []},
                                         "diagnostic": {"total": 0, "correct": 0, "false_passes": [], "false_rejects": [], "uncertain": [], "technical": []}}
    for case in cases.values():
        target = validated[case["episode"]][case["index"]]
        verdict = target.get("verdict")
        outcome = "correct" if verdict == case["expected"] else "semantic_uncertain" if verdict == "uncertain" else "technical_unresolved" if verdict is None else "false_pass" if case["expected"] == "fail" and verdict == "pass" else "false_reject" if case["expected"] == "pass" and verdict == "fail" else "incorrect"
        bucket = buckets["clear" if case["clear"] else "diagnostic"]
        bucket["total"] += 1
        if outcome == "correct":
            bucket["correct"] += 1
        elif outcome == "false_pass":
            bucket["false_passes"].append([case["episode"], case["index"]])
        elif outcome == "false_reject":
            bucket["false_rejects"].append([case["episode"], case["index"]])
        elif outcome == "semantic_uncertain":
            bucket["uncertain"].append([case["episode"], case["index"]])
        elif outcome == "technical_unresolved":
            bucket["technical"].append([case["episode"], case["index"]])
        rows.append({"episode": case["episode"], "index": case["index"], "clear": case["clear"], "expected": case["expected"], "verdict": verdict, "outcome": outcome, "unit_id": target["unit_id"], "unit_indices": target["unit_indices"], "error": target.get("error")})
    for bucket in buckets.values():
        bucket["score"] = bucket["correct"] / bucket["total"] if bucket["total"] else 0.0
        bucket["counts"] = {"false_pass": len(bucket["false_passes"]),
                             "false_reject": len(bucket["false_rejects"]),
                             "semantic_uncertain": len(bucket["uncertain"]),
                             "technical_unresolved": len(bucket["technical"])}
    clear = buckets["clear"]
    technical = bool(clear["technical"])
    semantic = bool(clear["false_passes"] or clear["false_rejects"] or clear["uncertain"])
    status = "qualified" if not technical and not semantic and clear["correct"] == clear["total"] else "not_qualified_semantic_and_technical" if technical and semantic else "not_qualified_technical" if technical else "not_qualified_semantic"
    return {"status": status, "complete": True, "integrity": True, "consistency": consistency,
            "complete_ok": True, "integrity_ok": True, "consistency_ok": True, "cases": rows,
            "clear_score": clear["score"], "diagnostic_score": buckets["diagnostic"]["score"], "overall_score": sum(1 for row in rows if row["outcome"] == "correct") / len(rows),
            "clear": clear, "diagnostic": buckets["diagnostic"], "usage": usage}


def _load(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        _bad("artifacts", "invalid json")
    return _obj(value, "json")


def _assignment(value: str) -> Tuple[str, str]:
    if "=" not in value:
        _bad("config", "malformed assignment")
    episode, path = value.split("=", 1)
    _nonblank(episode, "episode", "config")
    _nonblank(path, "path", "config")
    return episode, path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="skill_test/semantic_benchmark60/manifest.json")
    parser.add_argument("--episode-report", action="append", required=True)
    parser.add_argument("--episode-usage", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = _load(args.manifest)
        reports: Dict[str, Dict[str, Any]] = {}
        usages: Dict[str, Dict[str, Any]] = {}
        for assignment in args.episode_report:
            episode, path = _assignment(assignment)
            if episode in reports:
                _bad("config", "duplicate report episode")
            reports[episode] = _load(path)
        for assignment in args.episode_usage:
            episode, path = _assignment(assignment)
            if episode in usages:
                _bad("config", "duplicate usage episode")
            usages[episode] = _load(path)
        result = evaluate_benchmark(manifest, reports, usages)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        return 0 if result["status"] == "qualified" else 1
    except EvaluationError as error:
        print(f"INVALID_{error.kind.upper()}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
