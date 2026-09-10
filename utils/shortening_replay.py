"""Replay frozen shortening outputs with a new contextual audit.

The replay deliberately has no candidate-generation path.  Candidate responses are
read from the old experiment's request ledger, parsed independently per cue, and
only the new selector and audit stages can create physical calls.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.llm_types import CallRecord, DetailedResponse, call_record_from_dict
from utils.opencode_clean import CleanOpenCodeServer
from utils.shortening_backends import LiveBackend, transport_runtime_metadata
from utils.shortening_candidate_prompts import (
    AUDIT_PROMPT_VERSION,
    AUDIT_SYSTEM_PROMPT,
    CANDIDATE_PROMPT_VERSION,
    SELECTION_PROMPT_VERSION,
    SELECTION_SYSTEM_PROMPT,
    build_selection_prompt,
    parse_candidates_partial,
    parse_selection,
    selection_schema,
)
from utils.shortening_costs import estimate_record_cost
from utils.shortening_experiment import (
    AUDIT_PROMPT_VERSION as EXPERIMENT_AUDIT_PROMPT_VERSION,
    CROSS_AUDITOR,
    EXPECTED_PLANNER,
    MAIN_AUDITOR,
    SOL_SELECTOR,
    _build_effects,
    _aggregate_effects,
    _canonical,
    _export_srt,
    _first_generation,
    _ledger_summary,
    _local_check,
    _norm,
    _opaque_id,
    _run_audits,
    _safe_relative,
    _sha,
    _source_cases,
    _source_record_map,
    _validate_and_load,
)
from utils.shortening_plan import plan_subtitles
from utils.shortening_runner import (
    RequestFunction,
    _atomic_json,
    _prepare_checkpoint,
    _request,
)


REPLAY_VERSION = "shortening-replay-v1"
ROOT_STATUS = "new_replayed_analysis"


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} cannot be read") from exc


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _relative_ref(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("reference escapes experiment root") from exc


def _validate_text_hash(row: dict[str, Any], name: str) -> None:
    value = row.get("text")
    expected = row.get("candidate_hash")
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} text is invalid")
    if expected is not None:
        if not isinstance(expected, str) or value is None or _sha_text(value) != expected:
            raise ValueError(f"{name} candidate hash mismatch")


def _validate_frozen_row(row: dict[str, Any], root: Path) -> None:
    for arm in ("A", "B", "C"):
        value = row.get(arm)
        if not isinstance(value, dict):
            continue
        _validate_text_hash(value, f"{arm} frozen row")
        for candidate in value.get("candidates", []):
            if not isinstance(candidate, dict):
                raise ValueError("candidate row is invalid")
            text = candidate.get("text")
            normalized_hash = candidate.get("normalized_hash")
            if normalized_hash is not None:
                if not isinstance(text, str) or not isinstance(normalized_hash, str):
                    raise ValueError("candidate normalized hash is invalid")
                if _sha_text(_norm(text)) != normalized_hash:
                    raise ValueError("candidate normalized hash mismatch")
            candidate_hash = candidate.get("candidate_hash")
            if candidate_hash is not None:
                if not isinstance(text, str) or not isinstance(candidate_hash, str) or _sha_text(text) != candidate_hash:
                    raise ValueError("candidate hash mismatch")
        provenance = value.get("provenance")
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError("frozen provenance is invalid")


def _validate_generation_provenance(old_row: dict[str, Any], generation_ref: dict[str, Any]) -> None:
    candidate = old_row.get("C")
    if not isinstance(candidate, dict) or not isinstance(candidate.get("provenance"), dict):
        return
    reference = candidate["provenance"].get("source_generation_request")
    if not isinstance(reference, dict):
        return
    for field in ("request_id", "file_sha256", "text_sha256"):
        expected = reference.get(field)
        actual = generation_ref.get(field)
        if expected is not None and expected != actual:
            raise ValueError(f"old generation provenance mismatch: {field}")


def _load_old_rows(experiment_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = experiment_root / "experiment_manifest.json"
    results_path = experiment_root / "case_results.json"
    old_manifest = _read_json(manifest_path, "existing experiment manifest")
    old_rows = _read_json(results_path, "existing case results")
    if not isinstance(old_manifest, dict) or not isinstance(old_rows, list) or any(not isinstance(row, dict) for row in old_rows):
        raise ValueError("existing experiment outputs have an invalid shape")
    for row in old_rows:
        _validate_frozen_row(row, experiment_root)
    return old_manifest, old_rows


def _check_frozen_identity(old_manifest: dict[str, Any], rows: list[dict[str, Any]], manifest: dict[str, Any], cases: list[dict[str, Any]]) -> int:
    corpus_hash = old_manifest.get("corpus_sha256")
    expected_hash = _canonical(manifest)
    if corpus_hash != expected_hash:
        raise ValueError("corpus canonical hash mismatch")
    expected_ids = [case["case_id"] for case in cases]
    if old_manifest.get("sample_case_ids") != expected_ids:
        raise ValueError("corpus sample case IDs mismatch")
    params = old_manifest.get("params")
    repetitions = params.get("repetitions") if isinstance(params, dict) else None
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        repetitions = max((int(row.get("repetition", 0)) for row in rows), default=0)
    if repetitions <= 0:
        raise ValueError("existing repetition count is invalid")
    by_rep: dict[int, set[str]] = {}
    expected_set = set(expected_ids)
    for row in rows:
        repetition = row.get("repetition")
        case_id = row.get("global_case_id")
        if isinstance(repetition, bool) or not isinstance(repetition, int) or not 1 <= repetition <= repetitions:
            raise ValueError("existing case result repetition is invalid")
        if not isinstance(case_id, str) or case_id not in expected_set:
            raise ValueError("existing case result case ID mismatch")
        by_rep.setdefault(repetition, set()).add(case_id)
    if any(by_rep.get(index, set()) - expected_set for index in range(1, repetitions + 1)):
        raise ValueError("existing case result identity mismatch")
    return repetitions


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError("existing request ledger cannot be read") from exc
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _old_generation_response(path: Path, root: Path) -> dict[str, Any]:
    rows = _jsonl(path)
    starts: dict[str, dict[str, Any]] = {}
    completions: dict[str, dict[str, Any]] = {}
    for row in rows:
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or row.get("stage") != "generation_candidates":
            continue
        if row.get("state") == "start":
            starts[request_id] = row
        elif row.get("state") == "complete":
            completions[request_id] = row
    candidates = [(request_id, starts[request_id], completions.get(request_id)) for request_id in starts if request_id in completions]
    if not candidates:
        return {"status": "unavailable", "reason": "missing_generation_request", "request_id": None, "text": None,
                "request_path": _relative_ref(root, path),
                "text_sha256": None, "file_sha256": _sha(path.read_bytes()) if path.exists() else None}
    request_id, start, completion = candidates[-1]
    text = completion.get("text")
    error = completion.get("error")
    if error or not isinstance(text, str):
        reason = "transport" if error else "missing_final_text"
        text = None
    else:
        reason = None
    try:
        file_hash = _sha(path.read_bytes())
    except OSError:
        file_hash = None
    return {"status": "available" if text is not None else "unavailable", "reason": reason,
            "request_id": request_id, "text": text, "text_sha256": _sha_text(text) if text is not None else None,
            "file_sha256": file_hash, "request_path": _relative_ref(root, path),
            "prompt_version": start.get("prompt_version")}


def _old_calls(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("calls.jsonl") if path.is_file())


def _old_cost_summary(paths: list[Path], root: Path | None = None, stage: str | None = None) -> dict[str, Any]:
    """Account old physical calls without copying them into the replay ledger."""
    records: list[CallRecord] = []
    unknown = 0
    errors = 0
    serialization_errors = 0
    no_record = 0
    raw_record_count = 0
    by_stage: Counter[str] = Counter()
    refs: list[str] = []
    for path in paths:
        refs.append(_relative_ref(root, path) if root is not None else path.name)
        try:
            lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        except OSError:
            lines = []
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                if stage is None:
                    raw_record_count += 1
                    unknown += 1
                    serialization_errors += 1
                continue
            if not isinstance(row, dict):
                if stage is None:
                    raw_record_count += 1
                    unknown += 1
                    serialization_errors += 1
                continue
            if stage is not None and row.get("stage") != stage:
                continue
            raw_record_count += 1
            row_stage = str(row.get("stage", "unknown"))
            by_stage[row_stage] += 1
            record_data = row.get("record")
            if not isinstance(record_data, dict):
                no_record += 1
                unknown += 1
                continue
            try:
                record = call_record_from_dict(record_data)
                estimate = estimate_record_cost(record)
            except (TypeError, ValueError):
                serialization_errors += 1
                unknown += 1
                record = None
            if record is not None:
                records.append(record)
                if estimate.get("known_cost_usd") is None:
                    unknown += 1
                if record.status == "error" or record.error:
                    errors += 1
    total = sum(float(estimate_record_cost(record).get("known_cost_usd") or 0.0) for record in records)
    tokens: Counter[str] = Counter()
    for record in records:
        if record.usage is None:
            continue
        for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
            value = getattr(record.usage, field, None)
            if isinstance(value, int):
                tokens[field] += value
    return {"physical_calls": raw_record_count, "recorded_calls": len(records),
            "known_valid_count": len(records) - sum(
                estimate_record_cost(record).get("known_cost_usd") is None for record in records
            ), "raw_record_count": raw_record_count, "known_subtotal_usd": total,
            "rawrecordcount": raw_record_count, "unknown_call_count": unknown,
            "unknown_count": unknown, "serialization_error_count": serialization_errors,
            "no_record_count": no_record, "error_count": errors + serialization_errors + no_record,
            "actual_call_error_count": errors, "stage_counts": dict(by_stage), "tokens": dict(tokens),
            "source_refs": refs, "source_paths": refs, "stage_filter": stage,
            "unknown_charged_usage": unknown > 0,
            "accounting_complete": unknown == 0,
            "accounting_note": "Old calls are referenced for attribution and never copied as replay receipts."}


def _arm_from_old(row: dict[str, Any], arm: str, case: dict[str, Any], experiment_root: Path) -> dict[str, Any]:
    old = row.get(arm)
    if not isinstance(old, dict):
        old = {"text": None, "status": "unavailable", "reason": "missing_old_arm"}
    text = old.get("text")
    if text is not None and not isinstance(text, str):
        raise ValueError(f"old {arm} text is invalid")
    result = copy.deepcopy(old)
    result["text"] = text
    result["status"] = old.get("status", "unavailable")
    result["frozen"] = True
    result["source_ref"] = {"path": _relative_ref(experiment_root, experiment_root / "case_results.json"),
                             "row_sha256": _sha(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))}
    result["candidate_hash"] = _sha_text(text) if text is not None else old.get("candidate_hash")
    if arm == "A":
        result["status"] = "unreviewed" if text is not None else "unavailable"
    return result


def _candidate_records(
    response: dict[str, Any], source_id: str, cases: list[dict[str, Any]], plans: dict[str, Any], seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    expected_ids = [case["cue_id"] for case in cases]
    errors: dict[str, str] = {}
    if response.get("status") != "available" or not isinstance(response.get("text"), str):
        reason = str(response.get("reason") or "unavailable")
        return {cue_id: [] for cue_id in expected_ids}, {cue_id: reason for cue_id in expected_ids}
    parsed, parse_errors = parse_candidates_partial(response["text"], expected_ids)
    errors.update(parse_errors)
    rows_by_id: dict[str, list[dict[str, Any]]] = {cue_id: [] for cue_id in expected_ids}
    for cue_id in expected_ids:
        parsed_row = parsed.get(cue_id)
        if parsed_row is None:
            continue
        seen: set[str] = set()
        for ordinal, candidate in enumerate(parsed_row["candidates"]):
            text = candidate["text"]
            normalized = _norm(text)
            local = _local_check(text, plans[cue_id])
            rejection: str | None = None
            if normalized in seen:
                rejection = "duplicate_normalized_text"
            elif not local["valid"]:
                rejection = "local_schema"
            elif not local["within_length"]:
                rejection = "overbudget"
            candidate_id = _opaque_id(seed, source_id, cue_id, str(ordinal), _sha_text(text))
            rows_by_id[cue_id].append({
                "candidate_id": candidate_id,
                "model_candidate_id": candidate["candidate_id"],
                "text": text,
                "local": local,
                "normalized_hash": _sha_text(normalized),
                "rejection_reason": rejection,
                "valid": rejection is None,
                "ordinal": ordinal,
            })
            seen.add(normalized)
    return rows_by_id, errors


async def _select_source(
    *, source_id: str, source_items: list[dict[str, Any]], cases: list[dict[str, Any]], plans: dict[str, Any], candidates: dict[str, list[dict[str, Any]]],
    generation_ref: dict[str, Any], old_generation_prompt_version: str,
    old_generation_prompt_snapshot_hash: str | None, generation_error: dict[str, str], repetition: int,
    seed: int, output_dir: Path, request_fn: RequestFunction,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_by_id = {cue_id: [row for row in candidates[cue_id] if row["valid"]] for cue_id in candidates}
    selection_ids = [cue_id for cue_id in (case["cue_id"] for case in cases) if valid_by_id[cue_id]]
    selection_errors: dict[str, str] = {}
    selection_results: dict[str, dict[str, Any]] = {}
    request_count = 0
    if selection_ids:
        target_set = set(selection_ids)
        # Selection receives the complete immutable source, not a reconstructed or
        # model-written context window. Target labels are transport metadata only.
        context: list[dict[str, Any]] = []
        for position, item in enumerate(source_items):
            text_value = item.get("text")
            text = text_value if isinstance(text_value, str) else " ".join(text_value) if isinstance(text_value, list) and all(isinstance(part, str) for part in text_value) else ""
            row: dict[str, Any] = {"cue_id": f"cue-{position}", "text": text, "is_target": f"cue-{position}" in target_set}
            if isinstance(item.get("start"), int) and not isinstance(item.get("start"), bool):
                row["start"] = item["start"]
            if isinstance(item.get("end"), int) and not isinstance(item.get("end"), bool):
                row["end"] = item["end"]
            context.append(row)
        targets: list[dict[str, Any]] = []
        for case in cases:
            cue_id = case["cue_id"]
            if cue_id not in target_set:
                continue
            targets.append({"cue_id": cue_id, "original": case["original"], "max_chars": case["max_chars"],
                            "candidates": [{"candidate_id": row["candidate_id"], "text": row["text"],
                                            "actual_chars": len(row["text"]), "max_chars": case["max_chars"]}
                                           for row in valid_by_id[cue_id]]})
        stage_calls: dict[str, list[CallRecord]] = {}
        counts = {"selection": 0}
        failures: list[dict[str, Any]] = []
        response = await _request(request_fn, SOL_SELECTOR, SELECTION_SYSTEM_PROMPT,
                                  build_selection_prompt(context, targets), selection_schema(), False,
                                  stage="selection", ids=selection_ids, output_dir=output_dir,
                                  stage_calls=stage_calls, request_counts=counts, request_failures=failures)
        request_count = counts.get("selection", 0)
        if response.error is not None or response.text is None:
            selection_errors.update({cue_id: "transport" for cue_id in selection_ids})
        else:
            try:
                selection_results = parse_selection(response.text, selection_ids)
            except (TypeError, ValueError, json.JSONDecodeError):
                selection_errors.update({cue_id: "schema" for cue_id in selection_ids})
            else:
                known = {cue_id: {row["candidate_id"] for row in valid_by_id[cue_id]} for cue_id in selection_ids}
                for cue_id, selected in selection_results.items():
                    selected_id = selected.get("selected_candidate_id")
                    if selected_id is not None and selected_id not in known[cue_id]:
                        selection_errors[cue_id] = "unknown_selected_candidate"
    records: list[dict[str, Any]] = []
    for case in cases:
        cue_id = case["cue_id"]
        selected_id = selection_results.get(cue_id, {}).get("selected_candidate_id")
        selected = next((row for row in valid_by_id[cue_id] if row["candidate_id"] == selected_id), None)
        generation_failure = generation_error.get(cue_id)
        selection_failure = selection_errors.get(cue_id)
        error = generation_failure or selection_failure
        if generation_failure:
            # A malformed candidate row is a protocol error, not a semantic
            # cannot-shorten response. Transport/missing-response failures
            # remain unavailable so the frozen reason is not rewritten.
            status = "error" if generation_failure not in {
                "missing_generation_request", "transport", "missing_final_text", "unavailable"
            } else "unavailable"
        elif selection_failure:
            status = "error" if selection_failure in {"transport", "schema", "unknown_selected_candidate"} else "unavailable"
        elif selected is not None:
            status = "accepted"
        else:
            status = "unavailable"
        records.append({
            "case_id": case["case_id"], "global_case_id": case["case_id"], "local_cue_id": cue_id,
            "source_id": source_id, "position": case["position"], "source_index": case.get("source_index"),
            "original": case["original"], "budget": {"max_chars": case["max_chars"], "available_ms": case["available_ms"]},
            "candidates": copy.deepcopy(candidates[cue_id]),
            "candidate_counts": {"requested_n": 5, "returned_n": len(candidates[cue_id]),
                                  "distinct_n": len({row["normalized_hash"] for row in candidates[cue_id]}),
                                  "valid_n": len(valid_by_id[cue_id])},
            "candidate_errors": {"generation": generation_error.get(cue_id), "selection": selection_errors.get(cue_id)},
            "candidate_error": {"reason": generation_failure} if generation_failure and status == "error" else None,
            "selection": selection_results.get(cue_id),
            "selected_candidate_id": selected_id if status == "accepted" else None,
            "selected_text": selected["text"] if status == "accepted" and selected is not None else None,
            "final_text": selected["text"] if status == "accepted" and selected is not None else case["original"],
            "status": status,
            "provenance": {"candidate_prompt_version": old_generation_prompt_version,
                           "candidate_prompt_snapshot_hash": old_generation_prompt_snapshot_hash,
                           "selection_prompt_version": SELECTION_PROMPT_VERSION,
                           "selector": asdict(SOL_SELECTOR), "replay_repetition": repetition,
                           "source_generation_request": copy.deepcopy(generation_ref)},
        })
    report = {"schema_version": 1, "prompt_version": SELECTION_PROMPT_VERSION,
              "candidate_generation_reused": True, "request_counts": {"selection": request_count},
              "request_failures": failures if selection_ids else [], "records": records,
              "generation_errors": generation_error,
              "selected_count": sum(row["status"] == "accepted" for row in records),
              "dropped_bad_row_reasons": dict(Counter(row["rejection_reason"] for record in records for row in record["candidates"] if row.get("rejection_reason"))),
              "source_id": source_id,
              "source_generation_request_id": generation_ref.get("request_id"),
              "source_generation_request_path": generation_ref.get("request_path"),
              "source_generation_request_hash": generation_ref.get("file_sha256"),
              "source_generation_response_hash": generation_ref.get("text_sha256")}
    subtitles = _prepare_checkpoint(copy.deepcopy(source_items), records, list(plans.values()), EXPECTED_PLANNER)
    _atomic_json(output_dir / "subtitles.json", subtitles)
    export_error = _export_srt(subtitles, output_dir)
    report["export_error"] = export_error
    _atomic_json(output_dir / "report.json", report)
    return records, report


def _old_prompt_version(old_manifest: dict[str, Any]) -> str:
    versions = old_manifest.get("prompt_versions")
    if isinstance(versions, dict):
        for key in ("candidates", "generation", "standard"):
            value = versions.get(key)
            if isinstance(value, str):
                return value
    return "original_experiment_manifest"


def _old_candidate_prompt_snapshot_hash(old_manifest: dict[str, Any]) -> str | None:
    prompt_strings = old_manifest.get("prompt_strings")
    candidate_system = prompt_strings.get("candidate_system") if isinstance(prompt_strings, dict) else None
    return _sha_text(candidate_system) if isinstance(candidate_system, str) else None


def _source_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    return {str(source["source_id"]): str(source.get("parsed_sha256", source.get("parsedsha")))
            for source in manifest.get("sources", []) if isinstance(source, dict)}


def _repetition_paths(paths: list[Path], repetition: int) -> list[Path]:
    marker = f"r{repetition}"
    return [path for path in paths if marker in path.parts]


def _workflow_projection(
    *,
    repetitions: int,
    factor: float,
    old_paths: list[Path],
    old_root: Path,
    old_stage: str | None,
    new_paths: list[Path],
    new_stage: str | None,
    label: str,
    disabled_reason: str | None = None,
) -> dict[str, Any]:
    """Project one production workflow, retaining lower bounds when incomplete."""
    if disabled_reason is not None:
        return {"workflow": label, "applicable": False, "inapplicable": True,
                "reason": disabled_reason, "unknown_cost": False, "per_repetition": []}
    per_repetition: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        old_summary = _old_cost_summary(_repetition_paths(old_paths, repetition), old_root, old_stage)
        new_summary = _ledger_summary(_repetition_paths(new_paths, repetition), stage=new_stage)
        known_lower_bound = (old_summary["known_subtotal_usd"] + new_summary["known_subtotal_usd"]) * factor
        unknown_call_count = old_summary["unknown_call_count"] + new_summary["unknown_call_count"]
        unknown_request_count = new_summary.get("unknown_request_count", 0)
        accounting_complete = unknown_call_count == 0 and unknown_request_count == 0
        per_repetition.append({
            "repetition": repetition,
            "known_lower_bound_usd": known_lower_bound,
            "projected_cost_usd": known_lower_bound if accounting_complete else None,
            "known_cost_usd": known_lower_bound,
            "accounting_complete": accounting_complete,
            "unknown_cost": not accounting_complete,
            "inapplicable": not accounting_complete,
            "applicable": accounting_complete,
            "unknown_call_count": unknown_call_count,
            "unknown_request_count": unknown_request_count,
            "source_refs": old_summary["source_refs"] + new_summary.get("source_refs", []),
        })
    lower_bounds = [row["known_lower_bound_usd"] for row in per_repetition]
    incomplete = [row["repetition"] for row in per_repetition if not row["accounting_complete"]]
    mean = sum(lower_bounds) / len(lower_bounds) if lower_bounds else 0.0
    return {
        "workflow": label,
        "known_cost_usd": {"mean": mean, "min": min(lower_bounds) if lower_bounds else 0.0,
                            "max": max(lower_bounds) if lower_bounds else 0.0},
        "known_cost_usd_mean": mean,
        "known_cost_usd_min": min(lower_bounds) if lower_bounds else 0.0,
        "known_cost_usd_max": max(lower_bounds) if lower_bounds else 0.0,
        "per_repetition": per_repetition,
        "accounting_complete": not incomplete,
        "unknown_cost": bool(incomplete),
        "inapplicable": bool(incomplete),
        "applicable": not incomplete,
        "inapplicable_repetitions": incomplete,
        "basis": "known lower bound projected from each repetition; audits excluded",
        "final_audit_included": False,
    }


def _new_replay_projection(paths: list[Path], repetitions: int, factor: float, denominator: int) -> dict[str, Any]:
    per_repetition: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        summary = _ledger_summary(_repetition_paths(paths, repetition))
        accounting_complete = summary["unknown_call_count"] == 0 and summary.get("unknown_request_count", 0) == 0
        known_lower_bound = summary["known_subtotal_usd"] * factor
        per_repetition.append({
            "repetition": repetition,
            "known_lower_bound_usd": known_lower_bound,
            "projected_cost_usd": known_lower_bound if accounting_complete else None,
            "known_cost_usd": known_lower_bound,
            "accounting_complete": accounting_complete,
            "unknown_cost": not accounting_complete,
            "inapplicable": not accounting_complete,
            "applicable": accounting_complete,
            "unknown_call_count": summary["unknown_call_count"],
            "unknown_request_count": summary.get("unknown_request_count", 0),
        })
    return {"diagnostic_only": True, "denominator": denominator, "factor": factor,
            "per_repetition": per_repetition,
            "per_repetition_new_known_cost_usd": [row["known_cost_usd"] for row in per_repetition]}


async def replay_experiment(
    experiment_root: Path,
    manifest: dict[str, Any],
    manifest_dir: Path,
    output_dir: Path,
    *,
    request_fn: RequestFunction,
    seed: int = 20260910,
) -> dict[str, Any]:
    """Replay selection and contextual audits over immutable old outputs."""
    if not isinstance(experiment_root, Path) or not isinstance(manifest_dir, Path) or not isinstance(output_dir, Path):
        raise TypeError("paths must be Path objects")
    if experiment_root.resolve() in output_dir.resolve().parents:
        raise ValueError("replay output must not be inside the original experiment root")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    sources, source_meta, cases = _validate_and_load(manifest, manifest_dir)
    old_manifest, old_rows = _load_old_rows(experiment_root)
    repetitions = _check_frozen_identity(old_manifest, old_rows, manifest, cases)
    old_best_of = old_manifest.get("params", {}).get("best_of") if isinstance(old_manifest.get("params"), dict) else None
    if old_best_of not in {0, 5}:
        raise ValueError("existing best_of must be 0 or 5")
    old_generation_prompt_version = _old_prompt_version(old_manifest)
    old_generation_prompt_snapshot_hash = _old_candidate_prompt_snapshot_hash(old_manifest)
    old_by_rep_case = {(row.get("repetition"), row.get("global_case_id")): row for row in old_rows}
    for row in old_rows:
        context = row.get("context")
        case = next((item for item in cases if item["case_id"] == row.get("global_case_id")), None)
        if case is not None and context is not None and context != case["context"]:
            raise ValueError(f"immutable context mismatch: {row.get('global_case_id')}")
    output_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    new_manifest: dict[str, Any] = {
        "schema_version": 1, "replay_version": REPLAY_VERSION, "status": "running",
        "root_status": ROOT_STATUS, "original_experiment_root": _relative_ref(experiment_root.parent, experiment_root),
        "original_experiment_manifest_sha256": _sha((experiment_root / "experiment_manifest.json").read_bytes()),
        "corpus_sha256": _canonical(manifest), "sample_case_ids": [case["case_id"] for case in cases],
        "params": {"repetitions": repetitions, "seed": seed, "old_best_of": old_best_of},
        "prompt_versions": {"selection": SELECTION_PROMPT_VERSION, "audit": AUDIT_PROMPT_VERSION,
                              "old_candidate_generation": _old_prompt_version(old_manifest),
                              "old_audit": "auditv1_superseded"},
        "original_prompt_versions": copy.deepcopy(old_manifest.get("prompt_versions", {})),
        "original_prompt_strings": copy.deepcopy(old_manifest.get("prompt_strings", {})),
         "prompt_snapshot_ids": {"selection": _sha_text(SELECTION_SYSTEM_PROMPT), "audit": _sha_text(AUDIT_SYSTEM_PROMPT),
                                 "old_candidate_generation": old_generation_prompt_snapshot_hash},
        "prompt_strings": {"selection_system": SELECTION_SYSTEM_PROMPT, "audit_system": AUDIT_SYSTEM_PROMPT},
        "models": {"selector": asdict(SOL_SELECTOR), "main_auditor": asdict(MAIN_AUDITOR), "cross_auditor": asdict(CROSS_AUDITOR)},
        "source_hashes": _source_hashes(manifest),
        "audit_v1_superseded": {"reason": "missing_context", "semantic_claim": False, "pooled_with_v2": False},
        "replay_generation_requests": 0, "clock": {"started_at": started},
    }
    runtime_metadata = getattr(request_fn, "runtime_metadata", None)
    if isinstance(runtime_metadata, dict):
        new_manifest["transport_runtime_metadata"] = {str(key): str(value) for key, value in runtime_metadata.items()}
    _atomic_json(output_dir / "experiment_manifest.json", new_manifest)

    all_case_results: list[dict[str, Any]] = []
    all_new_ledgers: list[Path] = []
    audits_by_rep: list[dict[str, Any]] = []
    effects_by_rep: list[dict[str, Any]] = []
    rep_summaries: list[dict[str, Any]] = []
    old_calls = _old_calls(experiment_root)
    old_cost = _old_cost_summary(old_calls, experiment_root)
    old_candidate_paths = [path for path in old_calls if "/best5/" in path.as_posix()]
    old_standard_paths = [path for path in old_calls if "/standard/" in path.as_posix()]
    old_audit_paths = [path for path in old_calls if "/audits/" in path.as_posix()]
    source_generation_refs: dict[str, dict[str, Any]] = {}

    for repetition in range(1, repetitions + 1):
        rep_root = output_dir / f"r{repetition}"
        rep_root.mkdir()
        arm_values: dict[str, dict[str, str]] = {}
        rep_counts: Counter[str] = Counter()
        source_reports: dict[str, Any] = {}
        c_records_by_id: dict[str, dict[str, Any]] = {}
        referenced_source_ids = sorted({case["source_id"] for case in cases})
        for source_id in referenced_source_ids:
            local_cases = _source_cases(cases, source_id)
            if len(local_cases) > 6:
                raise ValueError(f"selection batch exceeds six cases: {source_id}")
            plans = {plan.cue_id: plan for plan in plan_subtitles(sources[source_id], EXPECTED_PLANNER)}
            generation_error: dict[str, str] = {}
            candidate_rows: dict[str, list[dict[str, Any]]] = {case["cue_id"]: [] for case in local_cases}
            generation_ref: dict[str, Any] = {"status": "not_used", "source_id": source_id}
            if old_best_of == 5:
                request_path = experiment_root / f"r{repetition}" / "best5" / source_id / "requests.jsonl"
                if experiment_root.resolve() not in request_path.resolve().parents:
                    raise ValueError("old generation request path escapes experiment root")
                generation_ref = _old_generation_response(request_path, experiment_root)
                for local_case in local_cases:
                    _validate_generation_provenance(old_by_rep_case.get((repetition, local_case["case_id"]), {}), generation_ref)
                source_generation_refs[f"r{repetition}:{source_id}"] = generation_ref
                candidate_rows, generation_error = _candidate_records(generation_ref, source_id, local_cases, plans, seed)
                c_records, c_report = await _select_source(
                    source_id=source_id, source_items=sources[source_id], cases=local_cases, plans=plans, candidates=candidate_rows,
                     generation_ref=generation_ref, old_generation_prompt_version=old_generation_prompt_version,
                     old_generation_prompt_snapshot_hash=old_generation_prompt_snapshot_hash,
                    generation_error=generation_error, repetition=repetition, seed=seed,
                    output_dir=rep_root / "best5" / source_id, request_fn=request_fn,
                )
                all_new_ledgers.append(rep_root / "best5" / source_id / "calls.jsonl")
                for record in c_records:
                    c_records_by_id[record["case_id"]] = record
                source_reports[source_id] = {"generation_request": generation_ref, "selection": c_report,
                                             "source_hash": _source_hashes(manifest).get(source_id),
                                             "generated_variant_bytes": sum(len(row["text"].encode("utf-8")) for rows in candidate_rows.values() for row in rows)}
            else:
                source_reports[source_id] = {"generation_request": generation_ref, "selection": {"skipped": True},
                                             "source_hash": _source_hashes(manifest).get(source_id), "generated_variant_bytes": 0}
        for case in cases:
            case_id = case["case_id"]
            old_row = old_by_rep_case.get((repetition, case_id), {})
            a = _arm_from_old(old_row, "A", case, experiment_root)
            b = _arm_from_old(old_row, "B", case, experiment_root)
            c = copy.deepcopy(c_records_by_id.get(case_id, {"status": "unavailable", "reason": "best_of_disabled" if old_best_of == 0 else "generation_unavailable", "selected_text": None}))
            a_text, b_text, c_text = a.get("text"), b.get("text"), c.get("selected_text")
            values = {arm: text for arm, text in (("A", a_text), ("B", b_text), ("C", c_text)) if isinstance(text, str)}
            arm_values[case_id] = values
            rep_counts["A"] += a_text is not None
            rep_counts["B"] += b_text is not None
            rep_counts["C"] += c_text is not None
            rep_counts["outcome_errors"] += any(arm.get("status") == "error" for arm in (a, b, c) if isinstance(arm, dict))
            all_case_results.append({
                "global_case_id": case_id, "local_cue_id": case["cue_id"], "source_id": case["source_id"],
                "repetition": repetition, "original": case["original"],
                "budget": {"max_chars": case["max_chars"], "available_ms": case["available_ms"]},
                "context": copy.deepcopy(case["context"]), "source_index": case.get("source_index"),
                "prompt_versions": {"standard": old_manifest.get("prompt_versions", {}).get("standard") if isinstance(old_manifest.get("prompt_versions"), dict) else None,
                                    "old_candidate_generation": _old_prompt_version(old_manifest),
                                    "selection": SELECTION_PROMPT_VERSION, "audit": AUDIT_PROMPT_VERSION},
                "A": a, "B": b, "C": c,
            })
        _atomic_json(output_dir / "case_results.json", all_case_results)
        rep_audits = await _run_audits(cases=cases, arm_values=arm_values, manifest=manifest,
                                       output_dir=rep_root / "audits", request_fn=request_fn, seed=seed)
        audit_paths = [rep_root / "audits" / label / "calls.jsonl" for label in ("main", "crosscheck", "order")]
        all_new_ledgers.extend(audit_paths)
        rep_effects = _build_effects(rep_audits, cases, arm_values)
        audits_by_rep.append(rep_audits)
        effects_by_rep.append(rep_effects)
        rep_summary = {"repetition": repetition, "arm_counts": dict(rep_counts), "audits": rep_audits,
                       "effects": rep_effects, "source_reports": source_reports,
                       "paired_contextual_effects_denominator": len(manifest.get("audit_case_ids", [])),
                       "judge_versions": {"main": AUDIT_PROMPT_VERSION, "crosscheck": AUDIT_PROMPT_VERSION, "order": AUDIT_PROMPT_VERSION},
                       "full_source_hashes": _source_hashes(manifest),
                       "bytesCgeneratedvariants": sum(int(item.get("generated_variant_bytes", 0)) for item in source_reports.values()),
                       "replay_generation_requests": 0}
        rep_summaries.append(rep_summary)
        _atomic_json(output_dir / "experiment_summary.json", {"status": "running", "repetitions": rep_summaries})

    new_ledger = _ledger_summary(all_new_ledgers)
    new_selection = _ledger_summary([path for path in all_new_ledgers if "/best5/" in path.as_posix()], stage="selection")
    new_audits = _ledger_summary([path for path in all_new_ledgers if "/audits/" in path.as_posix()])
    new_ledger["source_refs"] = [_relative_ref(output_dir, path) for path in all_new_ledgers]
    new_ledger["stage_filter"] = None
    new_selection["source_refs"] = [_relative_ref(output_dir, path) for path in all_new_ledgers if "/best5/" in path.as_posix()]
    new_selection["stage_filter"] = "selection"
    new_audits["source_refs"] = [_relative_ref(output_dir, path) for path in all_new_ledgers if "/audits/" in path.as_posix()]
    new_audits["stage_filter"] = None
    old_standard_generation = _old_cost_summary(old_standard_paths, experiment_root, stage="generation")
    old_generation = _old_cost_summary(old_candidate_paths, experiment_root, stage="generation_candidates")
    old_standard = _old_cost_summary(old_standard_paths, experiment_root)
    old_audits = _old_cost_summary(old_audit_paths, experiment_root)
    old_selector_legacy = _old_cost_summary(old_candidate_paths, experiment_root, stage="selection")
    old_generation_paths_by_rep = old_candidate_paths
    old_standard_paths_by_rep = old_standard_paths
    effects = {"by_repetition": effects_by_rep, "aggregate": _aggregate_effects(effects_by_rep) if effects_by_rep else {}}
    projection_factor = 500 / max(1, len(cases))
    final: dict[str, Any] = {
        "status": "completed", "root_status": ROOT_STATUS, "schema_version": 1, "corpus_sha256": _canonical(manifest),
        "sample_case_ids": [case["case_id"] for case in cases], "planned_case_count_per_repetition": len(cases),
        "repetitions": rep_summaries, "audits_by_repetition": audits_by_rep, "effects_by_repetition": effects_by_rep,
        "effects": effects, "raw_audits_retained": True, "preference_and_order_informativeness": "stored in audits_by_repetition",
         "cost": {
             "new_physical": new_ledger, "new_selection_v2": new_selection, "new_audits_v2": new_audits,
             "old_referenced": old_cost, "old_physical": old_cost,
             "old_standard_generation_A": old_standard_generation, "old_standard_B": old_standard,
             "old_candidate_generation_C": old_generation, "old_selector_legacy_spend": old_selector_legacy,
             "old_audit_v1_superseded": old_audits,
            "development_spent": {"old_known_subtotal_usd": old_cost["known_subtotal_usd"],
                                  "new_known_subtotal_usd": new_ledger["known_subtotal_usd"],
                                  "cumulative_known_subtotal_usd": old_cost["known_subtotal_usd"] + new_ledger["known_subtotal_usd"],
                                  "old_source_refs": [_relative_ref(experiment_root, path) for path in old_calls],
                                  "new_source_refs": [_relative_ref(output_dir, path) for path in all_new_ledgers],
                                  "unknown_charged_usage": bool(old_cost.get("unknown_charged_usage") or new_ledger.get("unknown_call_count"))},
             "by_workflow": {"A": {"attribution": "old standard generation shared with B; not added twice",
                                    "old_standard_generation": old_standard_generation, "old": old_standard_generation},
                             "B": {"attribution": "old standard unchanged", "old_standard_full": old_standard, "old": old_standard},
                             "C": {"old_candidates_generation": old_generation, "old_actual_generation_candidates": old_generation,
                                   "new_selection_v2": new_selection,
                                   "old_selector_v1_excluded": True}, "audits": {"old": old_audits, "new": new_audits}},
             "workflow_projections_500": {
                 "A": _workflow_projection(repetitions=repetitions, factor=projection_factor,
                                             old_paths=old_standard_paths_by_rep, old_root=experiment_root,
                                             old_stage="generation", new_paths=[], new_stage=None,
                                             label="old_standard_generation"),
                 "B": _workflow_projection(repetitions=repetitions, factor=projection_factor,
                                             old_paths=old_standard_paths_by_rep, old_root=experiment_root,
                                             old_stage=None, new_paths=[], new_stage=None,
                                             label="old_standard_full"),
                 "C": _workflow_projection(repetitions=repetitions, factor=projection_factor,
                                             old_paths=old_generation_paths_by_rep, old_root=experiment_root,
                                             old_stage="generation_candidates",
                                             new_paths=[path for path in all_new_ledgers if "/best5/" in path.as_posix()],
                                             new_stage="selection",
                                             label="old_candidates_generation_plus_new_selection",
                                             disabled_reason="best_of_disabled" if old_best_of == 0 else None),
             },
             "projection_500": _new_replay_projection(all_new_ledgers, repetitions, projection_factor, len(cases)),
             "new_replay_cost_projection_500": _new_replay_projection(
                 all_new_ledgers, repetitions, projection_factor, len(cases)),
        },
        "source_generation_refs": source_generation_refs, "replay_generation_requests": 0,
        "clock": {"started_at": started, "finished_at": datetime.now(timezone.utc).isoformat()},
    }
    _atomic_json(output_dir / "experiment_summary.json", final)
    new_manifest["status"] = "completed"
    new_manifest["clock"]["finished_at"] = final["clock"]["finished_at"]
    _atomic_json(output_dir / "experiment_manifest.json", new_manifest)
    _atomic_json(output_dir / "case_results.json", all_case_results)
    return final


def _build_live_backend(timeout: float) -> tuple[CleanOpenCodeServer, LiveBackend]:
    metadata = transport_runtime_metadata()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DeepSeek API key required for auditor")
    required = {"openai/gpt-5.6-sol": {"medium"}, "openai/gpt-5.6-luna": {"medium"}}
    server = CleanOpenCodeServer(required_models=required, timeout=min(timeout, 45.0))
    server.start()
    backend = LiveBackend(server, deepseek_api_key=os.environ["DEEPSEEK_API_KEY"], timeout=timeout, max_tokens=None)
    backend.runtime_metadata = metadata
    return server, backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay frozen shortening outputs with contextual audits")
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if args.experiment_root.resolve() in args.output_dir.resolve().parents:
        raise SystemExit("replay output must not be inside the original experiment root")
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    try:
        corpus = _read_json(args.corpus_manifest, "corpus manifest")
        if not isinstance(corpus, dict):
            raise ValueError("corpus manifest must be an object")
        _validate_and_load(corpus, args.corpus_manifest.parent)
        old_manifest, old_rows = _load_old_rows(args.experiment_root)
        _, _, cases = _validate_and_load(corpus, args.corpus_manifest.parent)
        repetitions = _check_frozen_identity(old_manifest, old_rows, corpus, cases)
        if not args.execute:
            args.output_dir.mkdir(parents=True)
            _atomic_json(args.output_dir / "replay_plan.json", {
                "status": "dry_run", "replay_version": REPLAY_VERSION, "experiment_root": str(args.experiment_root),
                "corpus_sha256": _canonical(corpus), "sample_case_ids": [case["case_id"] for case in cases],
                "repetitions": repetitions, "old_best_of": old_manifest.get("params", {}).get("best_of") if isinstance(old_manifest.get("params"), dict) else None,
                "execute_required_for_auth": True, "replay_generation_requests": 0, "timeout": args.timeout,
            })
            print(json.dumps({"status": "dry_run", "output": str(args.output_dir)}))
            return
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        server, backend = _build_live_backend(args.timeout)
        try:
            result = asyncio.run(replay_experiment(args.experiment_root, corpus, args.corpus_manifest.parent,
                                                   args.output_dir, request_fn=backend))
        finally:
            server.stop()
        print(json.dumps({"status": result["status"], "output": str(args.output_dir)}))
    except (OSError, TypeError, ValueError, RuntimeError, TimeoutError) as exc:
        raise SystemExit(type(exc).__name__ + ": validation or preflight failed") from exc


if __name__ == "__main__":
    main()
