"""Read-only duplicate analysis for the saved SH08 response receipts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from utils.shortening_prompts import parse_generation_partial, parse_review_partial
from utils.shortening_resume import canonical_hash


EXPECTED_CASES = 60
PROMPT_MARKER = "Input JSON:\n"
DESCRIPTOR_FIELDS = (
    "rep", "stage", "generator", "arbiter", "batch", "spec", "system_prompt",
    "user_prompt", "schema",
)


class AnalysisError(ValueError):
    """The saved analysis inputs conflict or cannot be trusted."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_copy_without(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{label} must be an object")
    return value


def _prompt_payload(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    prompt = descriptor.get("user_prompt")
    if not isinstance(prompt, str) or PROMPT_MARKER not in prompt:
        raise AnalysisError("descriptor user_prompt has no Input JSON payload")
    try:
        payload = json.loads(prompt.split(PROMPT_MARKER, 1)[1])
    except (TypeError, json.JSONDecodeError) as exc:
        raise AnalysisError("descriptor Input JSON payload is malformed") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("targets"), list) or not isinstance(payload.get("context"), list):
        raise AnalysisError("descriptor Input JSON payload is incomplete")
    targets = payload["targets"]
    if not targets or any(not isinstance(item, Mapping) or not isinstance(item.get("cue_id"), str) for item in targets):
        raise AnalysisError("descriptor targets are invalid")
    ids = [str(item["cue_id"]) for item in targets]
    if len(ids) != len(set(ids)):
        raise AnalysisError("descriptor targets contain duplicate case IDs")
    return payload


def _saved_response(receipt: Mapping[str, Any]) -> str | None:
    response = receipt.get("response")
    if isinstance(response, Mapping) and isinstance(response.get("text"), str):
        return str(response["text"])
    if isinstance(receipt.get("text"), str):
        return str(receipt["text"])
    raw = response.get("raw") if isinstance(response, Mapping) else None
    snapshot = raw.get("detailed_response_snapshot") if isinstance(raw, Mapping) else None
    return str(snapshot["text"]) if isinstance(snapshot, Mapping) and isinstance(snapshot.get("text"), str) else None


def _normal(text: str) -> str:
    return " ".join(text.split())


def _check_envelope(value: Mapping[str, Any], label: str) -> None:
    checksum = value.get("checksum")
    if isinstance(checksum, str) and canonical_hash(_json_copy_without(value, "checksum")) != checksum:
        raise AnalysisError(f"{label} checksum mismatch")


def _validate_inputs(
    case_results: Mapping[str, Any], ledger: Mapping[str, Any], selection: Mapping[str, Any],
) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    _check_envelope(case_results, "case_results")
    _check_envelope(ledger, "ledger")
    rows = case_results.get("rows")
    cases = selection.get("cases")
    requests = ledger.get("requests")
    if not isinstance(rows, list) or len(rows) != EXPECTED_CASES:
        raise AnalysisError("case_results must contain exactly 60 rows")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise AnalysisError("selection must contain exactly 60 cases")
    if not isinstance(requests, dict):
        raise AnalysisError("ledger requests must be an object")
    case_ids = [row.get("case_id") for row in cases if isinstance(row, Mapping)]
    row_ids = [row.get("case_id") for row in rows if isinstance(row, Mapping)]
    if len(case_ids) != EXPECTED_CASES or len(row_ids) != EXPECTED_CASES or set(case_ids) != set(row_ids) or len(set(case_ids)) != EXPECTED_CASES:
        raise AnalysisError("selection and case_results case IDs conflict")
    ledger_checksum = ledger.get("checksum")
    if isinstance(case_results.get("ledger_checksum"), str) and case_results.get("ledger_checksum") != ledger_checksum:
        raise AnalysisError("case_results ledger checksum does not match ledger")
    descriptors: dict[str, Mapping[str, Any]] = {}
    for key, request in requests.items():
        if not isinstance(key, str) or not isinstance(request, Mapping):
            raise AnalysisError("ledger request is invalid")
        descriptor = request.get("descriptor")
        if not isinstance(descriptor, Mapping):
            raise AnalysisError(f"request {key} has no descriptor")
        if canonical_hash(descriptor) != key:
            raise AnalysisError(f"request {key} descriptor key mismatch")
        if any(field not in descriptor for field in DESCRIPTOR_FIELDS):
            raise AnalysisError(f"request {key} descriptor is incomplete")
        payload = _prompt_payload(descriptor)
        if any(item["cue_id"] not in set(case_ids) for item in payload["targets"]):
            raise AnalysisError(f"request {key} targets a case outside selection")
        descriptors[key] = descriptor
    return [str(case_id) for case_id in case_ids], descriptors


def _parse_request(
    key: str, request: Mapping[str, Any], descriptor: Mapping[str, Any], payload: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], bool]:
    target_ids = [str(item["cue_id"]) for item in payload["targets"] if isinstance(item, Mapping)]
    text = _saved_response(request)
    if request.get("status") != "complete" or text is None or not text.strip():
        return {}, {case_id: "unavailable saved response" for case_id in target_ids}, True
    parser = parse_generation_partial if descriptor.get("stage") in {"generation", "repair"} else parse_review_partial
    parsed = parser(text, target_ids)
    return parsed.results, parsed.errors, False


def _base_stat(cases: int = EXPECTED_CASES) -> dict[str, Any]:
    return {
        "cases": cases, "attempted_targets": 0, "attempted_requests": 0,
        "available_response_targets": 0, "unavailable_response_targets": 0,
        "parse_error_targets": 0, "cannot_shorten_targets": 0,
    }


def _generation_analysis(
    case_ids: list[str], requests: Mapping[str, Mapping[str, Any]], descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    values: dict[tuple[int, str, str], tuple[str, str]] = {}
    stats: dict[str, dict[str, Any]] = {}
    parse_errors: list[dict[str, str]] = []
    generators = sorted({str(d.get("generator")) for d in descriptors.values() if d.get("stage") == "generation"})
    for generator in generators:
        stats[generator] = _base_stat()
        stats[generator]["rep1_ok"] = 0
        stats[generator]["rep2_ok"] = 0
        stats[generator]["cannot_shorten"] = 0
        stats[generator]["missing_or_invalid"] = 0
        stats[generator]["unavailable_cases_by_rep"] = {"rep1": EXPECTED_CASES, "rep2": EXPECTED_CASES}
        stats[generator]["no_ok_result_cases_by_rep"] = {"rep1": EXPECTED_CASES, "rep2": EXPECTED_CASES}
    seen_cases: dict[tuple[str, int], set[str]] = defaultdict(set)
    attempted_cases: dict[tuple[str, int], set[str]] = defaultdict(set)
    for key, descriptor in descriptors.items():
        if descriptor.get("stage") != "generation":
            continue
        generator = str(descriptor.get("generator"))
        rep = descriptor.get("rep")
        if not isinstance(rep, int) or rep not in (1, 2):
            raise AnalysisError(f"generation request {key} has invalid repetition")
        payload = _prompt_payload(descriptor)
        result, errors, unavailable = _parse_request(key, requests[key], descriptor, payload)
        stat = stats[generator]
        stat["attempted_requests"] += 1
        stat["attempted_targets"] += len(payload["targets"])
        stat["unavailable_response_targets"] += len(payload["targets"]) if unavailable else 0
        stat["available_response_targets"] += 0 if unavailable else len(payload["targets"])
        attempted_cases[(generator, rep)].update(
            str(item["cue_id"]) for item in payload["targets"] if isinstance(item, Mapping)
        )
        for case_id in [str(item["cue_id"]) for item in payload["targets"] if isinstance(item, Mapping)]:
            if case_id in errors:
                stat["parse_error_targets"] += 0 if unavailable else 1
                stat["missing_or_invalid"] += 1
                if unavailable:
                    continue
                parse_errors.append({"receipt_key": key, "case_id": case_id, "reason": errors[case_id]})
            row = result.get(case_id)
            if row is None:
                continue
            if row.get("status") == "cannot_shorten":
                stat["cannot_shorten"] += 1
                stat["cannot_shorten_targets"] += 1
                continue
            text = row.get("text")
            if row.get("status") != "ok" or not isinstance(text, str) or not text.strip():
                stat["missing_or_invalid"] += 1
                continue
            coordinate = (rep, generator, case_id)
            if coordinate in values:
                raise AnalysisError(f"duplicate parsed generation target {rep}/{generator}/{case_id}")
            values[coordinate] = (text, key)
            stat[f"rep{rep}_ok"] += 1
            seen_cases[(generator, rep)].add(case_id)
    evidence_exact: list[dict[str, str]] = []
    evidence_whitespace: list[dict[str, str]] = []
    intersections: dict[str, int] = defaultdict(int)
    total_intersection = 0
    for generator in generators:
        for rep in (1, 2):
            stat = stats[generator]
            stat["unavailable_cases_by_rep"][f"rep{rep}"] = len(set(case_ids) - attempted_cases[(generator, rep)])
            stat["no_ok_result_cases_by_rep"][f"rep{rep}"] = len(set(case_ids) - seen_cases[(generator, rep)])
        for case_id in case_ids:
            left = values.get((1, generator, case_id))
            right = values.get((2, generator, case_id))
            if left is None or right is None:
                continue
            intersections[generator] += 1
            total_intersection += 1
            evidence = {"generator": generator, "case_id": case_id, "rep1_receipt_key": left[1], "rep2_receipt_key": right[1]}
            if left[0] == right[0]:
                evidence_exact.append(evidence)
            if _normal(left[0]) == _normal(right[0]):
                evidence_whitespace.append(evidence)
    for generator in generators:
        stats[generator]["intersection_ok_denominator"] = intersections[generator]
        stats[generator]["exact_duplicate_count"] = sum(item["generator"] == generator for item in evidence_exact)
        stats[generator]["whitespace_normalized_duplicate_count"] = sum(item["generator"] == generator for item in evidence_whitespace)
        stats[generator]["benchmark_evidence_only"] = True
    return {
        "cases_total": len(case_ids), "per_generator": stats,
        "total": {
            "intersection_ok_denominator": total_intersection,
            "exact_duplicate_count": len(evidence_exact),
            "whitespace_normalized_duplicate_count": len(evidence_whitespace),
            "benchmark_evidence_only": True,
        },
        "exact_duplicate_evidence": evidence_exact,
        "whitespace_normalized_duplicate_evidence": evidence_whitespace,
        "parse_error_evidence": parse_errors,
    }


def _without_candidate(value: Any) -> Any:
    if isinstance(value, Mapping):
        # actual_chars is derived from candidate and is not an independent scope input.
        return {str(key): _without_candidate(item) for key, item in value.items() if key not in {"candidate", "actual_chars"}}
    if isinstance(value, list):
        return [_without_candidate(item) for item in value]
    return deepcopy(value)


def _review_scope_signature(descriptor: Mapping[str, Any], payload: Mapping[str, Any]) -> tuple[Any, ...]:
    target_scope = _without_candidate(payload["targets"])
    return (
        descriptor.get("rep"), descriptor.get("generator"), descriptor.get("arbiter"), descriptor.get("spec"),
        descriptor.get("system_prompt"), descriptor.get("schema"), payload.get("context"), target_scope,
    )


def _repair_analysis(
    requests: Mapping[str, Mapping[str, Any]], descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = {}
    duplicate_rows: list[dict[str, str]] = []
    recheck_index: dict[tuple[Any, Any, Any, str], list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    review_index: dict[tuple[Any, Any, Any, str], list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    parsed_requests: dict[str, tuple[dict[str, dict[str, Any]], dict[str, str], bool, Mapping[str, Any]]] = {}
    for key, descriptor in descriptors.items():
        stage = descriptor.get("stage")
        if stage not in {"review", "recheck", "repair"}:
            continue
        payload = _prompt_payload(descriptor)
        parsed_requests[key] = (*_parse_request(key, requests[key], descriptor, payload), payload)
        for item in payload["targets"]:
            if not isinstance(item, Mapping):
                continue
            coordinate = (descriptor.get("rep"), descriptor.get("generator"), descriptor.get("arbiter"), str(item["cue_id"]))
            record = (key, descriptor, item)
            if stage == "review":
                review_index[coordinate].append(record)
            elif stage == "recheck":
                recheck_index[coordinate].append(record)
    for key, descriptor in descriptors.items():
        if descriptor.get("stage") != "repair":
            continue
        generator = str(descriptor.get("generator"))
        arbiter = str(descriptor.get("arbiter"))
        bucket_key = f"{generator}/{arbiter}"
        stat = stats.setdefault(bucket_key, {
            "generator": generator, "arbiter": descriptor.get("arbiter"), "attempted_requests": 0,
            "attempted_targets": 0, "unavailable_response_targets": 0, "parse_error_targets": 0,
            "cannot_shorten_targets": 0, "comparable_ok_outputs": 0, "exact_same_as_prior_count": 0,
            "whitespace_normalized_same_as_prior_count": 0,
        })
        payload = _prompt_payload(descriptor)
        result, errors, unavailable = parsed_requests[key][:3]
        stat["attempted_requests"] += 1
        stat["attempted_targets"] += len(payload["targets"])
        if unavailable:
            stat["unavailable_response_targets"] += len(payload["targets"])
        for item in payload["targets"]:
            case_id = str(item["cue_id"])
            if case_id in errors:
                if not unavailable:
                    stat["parse_error_targets"] += 1
                continue
            row = result.get(case_id)
            if row is None:
                continue
            if row.get("status") == "cannot_shorten":
                stat["cannot_shorten_targets"] += 1
                continue
            text = row.get("text")
            prior = item.get("prior_candidate")
            if row.get("status") != "ok" or not isinstance(text, str) or not isinstance(prior, str):
                continue
            stat["comparable_ok_outputs"] += 1
            exact = text == prior
            whitespace = _normal(text) == _normal(prior)
            if exact:
                stat["exact_same_as_prior_count"] += 1
            if whitespace:
                stat["whitespace_normalized_same_as_prior_count"] += 1
            if not exact and not whitespace:
                continue
            coordinate = (descriptor.get("rep"), descriptor.get("generator"), descriptor.get("arbiter"), case_id)
            rechecks = recheck_index.get(coordinate, [])
            initial = review_index.get(coordinate, [])
            scope_status = "not_sent"
            proof = "unknown"
            recheck_key: str | None = None
            if rechecks:
                recheck_key = rechecks[0][0]
                scope_status = "different_scope"
                for _, initial_descriptor, initial_item in initial:
                    for candidate_key, recheck_descriptor, recheck_item in rechecks:
                        initial_payload = _prompt_payload(initial_descriptor)
                        recheck_payload = _prompt_payload(recheck_descriptor)
                        if _review_scope_signature(initial_descriptor, initial_payload) == _review_scope_signature(recheck_descriptor, recheck_payload):
                            scope_status = "same_scope"
                            proof = "proven_avoidable"
                            recheck_key = candidate_key
                            break
                    if proof == "proven_avoidable":
                        break
            evidence = {
                "receipt_key": key, "case_id": case_id, "generator": generator,
                "arbiter": arbiter, "comparison": "exact" if exact else "whitespace_only",
                "recheck_sent": bool(rechecks), "recheck_receipt_key": recheck_key,
                "recheck_scope": scope_status, "avoidability": proof,
            }
            duplicate_rows.append(evidence)
    whole: dict[str, Any] = {
        "recheck_targets_sent_for_duplicate_repairs": sum(item["recheck_sent"] for item in duplicate_rows),
        "duplicate_repair_targets": len(duplicate_rows),
        "proven_avoidable_recheck_targets": sum(item["avoidability"] == "proven_avoidable" for item in duplicate_rows),
        "unknown_or_different_scope_targets": sum(item["recheck_scope"] == "different_scope" for item in duplicate_rows),
        "not_sent_targets": sum(item["recheck_scope"] == "not_sent" for item in duplicate_rows),
        "proven_avoidable_whole_request_keys": [],
        "unknown_or_different_scope_whole_request_keys": [],
    }
    by_recheck: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in duplicate_rows:
        if item["recheck_receipt_key"] is not None:
            by_recheck[item["recheck_receipt_key"]].append(item)
    for key, descriptor in descriptors.items():
        if descriptor.get("stage") != "recheck" or key not in by_recheck:
            continue
        payload = _prompt_payload(descriptor)
        evidence = by_recheck[key]
        if len(evidence) == len(payload["targets"]) and all(item["avoidability"] == "proven_avoidable" for item in evidence):
            whole["proven_avoidable_whole_request_keys"].append(key)
        else:
            whole["unknown_or_different_scope_whole_request_keys"].append(key)
    return {"per_generator_arbiter": stats, "duplicate_repair_evidence": duplicate_rows, "whole_request_assessment": whole}


def analyze(case_results: dict[str, Any], ledger: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    """Analyze saved receipts without contacting a model or changing inputs."""
    case_ids, descriptors = _validate_inputs(case_results, ledger, selection)
    requests = ledger["requests"]
    generation = _generation_analysis(case_ids, requests, descriptors)
    repairs = _repair_analysis(requests, descriptors)
    return {
        "schema": "sh08_duplicate_analysis.v1", "analysis": "read_only_saved_responses",
        "source_case_count": len(case_ids), "source_request_count": len(requests),
        "generation": generation, "repairs": repairs,
        "caution": "Historic v2 prompts and the new duplicate-analysis rule were not tested by inference here.",
    }


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise AnalysisError(f"{path} must contain an object")
    return value


def run_cli(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run the offline analysis and create a new report directory."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    if not run_dir.is_dir():
        raise AnalysisError(f"run directory is missing: {run_dir}")
    if not output_dir.parent.is_dir():
        raise AnalysisError(f"output parent directory is missing: {output_dir.parent}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    source_paths = {name: run_dir / f"{name}.json" for name in ("case_results", "ledger", "selection")}
    if any(not path.is_file() for path in source_paths.values()):
        raise AnalysisError("one or more authoritative source files are missing")
    case_results = _load(source_paths["case_results"])
    ledger = _load(source_paths["ledger"])
    selection = _load(source_paths["selection"])
    report = analyze(case_results, ledger, selection)
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report["sources"] = {
        name: {"path": str(path), "sha256": _sha256(path)} for name, path in source_paths.items()
    }
    report["source_integrity"] = {
        "case_results_checksum_verified": isinstance(case_results.get("checksum"), str),
        "ledger_checksum_verified": isinstance(ledger.get("checksum"), str),
        "case_results_ledger_checksum_matches": case_results.get("ledger_checksum") == ledger.get("checksum"),
        "manifest_checksum_declared_by_case_results": case_results.get("manifest_checksum"),
        "manifest_checksum_verified": False,
        "selection_hash": canonical_hash(selection),
    }
    output_dir.mkdir()
    with (output_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_cli(args.run_dir, args.output_dir)
    print(json.dumps({
        "source_case_count": report["source_case_count"],
        "source_request_count": report["source_request_count"],
        "generation": report["generation"],
        "repairs": report["repairs"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
