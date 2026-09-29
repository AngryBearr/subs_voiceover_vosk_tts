"""Blinded, source-isolated A/B/C shortening experiment orchestration."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

from utils.llm_types import CallRecord
from utils.length_check import check_length
from utils.opencode_clean import CleanOpenCodeServer
from utils.shortening_backends import LiveBackend
from utils.shortening_candidate_prompts import (
    AUDIT_PROMPT_VERSION, AUDIT_SYSTEM_PROMPT, CANDIDATE_PROMPT_VERSION, CANDIDATE_SYSTEM_PROMPT,
    SELECTION_PROMPT_VERSION, SELECTION_SYSTEM_PROMPT, audit_schema,
    build_audit_prompt, build_candidate_prompt, build_selection_prompt,
    candidate_schema, parse_audit, parse_candidates_partial, parse_selection, selection_schema,
)
from utils.shortening_costs import CURRENT_PRICE_SNAPSHOT_ID, PRICE_SNAPSHOTS, estimate_record_cost, summarize_costs
from utils.deepseek_limits import resolve_deepseek_token_limit
from utils.shortening_plan import CuePlan, PlannerConfig, context_items, plan_subtitles
from utils.shortening_prompts import ARBITER_SYSTEM_PROMPT, GENERATOR_SYSTEM_PROMPT, PROMPT_VERSION
from utils.shortening_runner import (
    ModelSpec, RequestFunction, RunConfig, _append_jsonl, _atomic_json, _checkpoint,
    _export_srt, _local_check, _prepare_checkpoint, _request, run_trial,
)


EXPERIMENT_VERSION = "shortening-experiment-v1"
EXPECTED_PLANNER = PlannerConfig(cps=13.0, max_speed=1.5, gap_ms=200, context_before=3, context_after=3)
SOL_SELECTOR = ModelSpec("opencode", "openai/gpt-5.6-sol", "medium")
MAIN_AUDITOR = ModelSpec("deepseek", "deepseek-flash", "low")
CROSS_AUDITOR = ModelSpec("opencode", "openai/gpt-5.6-luna", "medium")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> str:
    return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def _text(item: dict[str, Any]) -> str:
    value = item.get("text")
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return " ".join(value)
    return ""


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _safe_relative(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("source path must be a relative string")
    candidate = (root / value).resolve()
    if root.resolve() not in candidate.parents:
        raise ValueError("source path escapes corpus directory")
    return candidate


def _planner_manifest(manifest: dict[str, Any]) -> None:
    configured = manifest.get("planner_config")
    if configured is None:
        return
    if not isinstance(configured, dict):
        raise ValueError("corpus planner configuration does not match the fixed experiment budget")
    expected = asdict(EXPECTED_PLANNER)
    original_fields = ("cps", "max_speed", "gap_ms", "context_before", "context_after")
    additive_defaults = {
        "length_basis": expected["length_basis"],
        "calibration_profile": expected["calibration_profile"],
        "calibration_identity": expected["calibration_identity"],
    }
    if any(field not in configured or configured[field] != expected[field] for field in original_fields):
        raise ValueError("corpus planner configuration does not match the fixed experiment budget")
    if any(field in configured and configured[field] != default for field, default in additive_defaults.items()):
        raise ValueError("corpus planner configuration does not match the fixed experiment budget")


def _expected_case_context(items: list[dict[str, Any]], position: int) -> list[dict[str, Any]]:
    """Rebuild the immutable positional context exported with the corpus."""
    start = max(0, position - EXPECTED_PLANNER.context_before)
    end = min(len(items), position + EXPECTED_PLANNER.context_after + 1)
    context: list[dict[str, Any]] = []
    for index in range(start, end):
        item = items[index]
        row: dict[str, Any] = {"cue_id": f"cue-{index}", "text": _text(item)}
        if isinstance(item.get("start"), int) and not isinstance(item.get("start"), bool):
            row["start"] = item["start"]
        if isinstance(item.get("end"), int) and not isinstance(item.get("end"), bool):
            row["end"] = item["end"]
        context.append(row)
    return context


def _validate_and_load(manifest: dict[str, Any], manifest_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("sources"), list) or not isinstance(manifest.get("cases"), list):
        raise ValueError("invalid corpus manifest")
    _planner_manifest(manifest)
    sources: dict[str, list[dict[str, Any]]] = {}
    source_meta: dict[str, dict[str, Any]] = {}
    for source in manifest["sources"]:
        if not isinstance(source, dict) or not isinstance(source.get("source_id"), str):
            raise ValueError("invalid source manifest row")
        source_id = source["source_id"]
        if source_id in sources:
            raise ValueError("duplicate source_id")
        path = _safe_relative(manifest_dir, source.get("json_path"))
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("source JSON cannot be read") from exc
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ValueError("source JSON must be a list of objects")
        parsed_expected = source.get("parsed_sha256", source.get("parsedsha"))
        if not isinstance(parsed_expected, str) or _canonical(items) != parsed_expected:
            raise ValueError(f"source parsed hash mismatch: {source_id}")
        raw_declared = "raw_source_path" in source or "path" in source
        raw_path_value = source.get("raw_source_path", source.get("path"))
        if raw_declared:
            if not isinstance(raw_path_value, str) or not raw_path_value:
                raise ValueError(f"source raw path missing: {source_id}")
            input_root_value = manifest.get("input_root")
            if not isinstance(input_root_value, str) or not input_root_value:
                raise ValueError(f"source input root missing: {source_id}")
            raw_path = _safe_relative(Path(input_root_value), raw_path_value)
            if not raw_path.exists() or not raw_path.is_file():
                raise ValueError(f"source raw file missing: {source_id}")
            raw_expected = source.get("sha256", source.get("sha", source.get("raw_sha256")))
            if not isinstance(raw_expected, str) or _sha(raw_path.read_bytes()) != raw_expected:
                raise ValueError(f"source raw hash mismatch: {source_id}")
        sources[source_id] = copy.deepcopy(items)
        source_meta[source_id] = source
    cases = manifest["cases"]
    if any(not isinstance(case, dict) for case in cases):
        raise ValueError("invalid case manifest row")
    case_ids: set[str] = set()
    for case in cases:
        case_id = case.get("case_id")
        source_id, cue_id = case.get("source_id"), case.get("cue_id")
        if not isinstance(case_id, str) or case_id in case_ids or not isinstance(source_id, str) or source_id not in sources or not isinstance(cue_id, str):
            raise ValueError("invalid or duplicate case")
        if source_meta[source_id].get("split") != "test":
            raise ValueError("experiment cases must come only from test sources")
        case_ids.add(case_id)
        plans = plan_subtitles(sources[source_id], EXPECTED_PLANNER)
        by_id = {plan.cue_id: plan for plan in plans}
        plan = by_id.get(cue_id)
        source_item = sources[source_id][plan.position] if plan is not None and 0 <= plan.position < len(sources[source_id]) else {}
        if (plan is None or case_id != f"{source_id}:{cue_id}" or case.get("position") != plan.position
                or case.get("source_index", source_item.get("index", plan.position)) != source_item.get("index", plan.position)
                or case.get("severity") != plan.severity
                or any(case.get(key) != getattr(plan, attr) for key, attr in (("original", "text"), ("max_chars", "max_chars"), ("available_ms", "available_ms")))):
            raise ValueError(f"case does not match immutable source: {case_id}")
        context = case.get("context")
        if not isinstance(context, list) or context != _expected_case_context(sources[source_id], plan.position):
            raise ValueError(f"case context missing: {case_id}")
    known_case_ids = {case["case_id"] for case in cases}
    for field in ("audit_case_ids", "crosscheck_case_ids", "order_check_case_ids"):
        values = manifest.get(field, [])
        if not isinstance(values, list) or any(not isinstance(value, str) or value not in known_case_ids for value in values) or len(set(values)) != len(values):
            raise ValueError(f"invalid frozen {field}")
    if not set(manifest.get("crosscheck_case_ids", [])).issubset(set(manifest.get("audit_case_ids", []))) or not set(manifest.get("order_check_case_ids", [])).issubset(set(manifest.get("crosscheck_case_ids", []))):
        raise ValueError("audit subsets are not nested")
    return sources, source_meta, cases


def _source_cases(cases: list[dict[str, Any]], source_id: str) -> list[dict[str, Any]]:
    return sorted((case for case in cases if case["source_id"] == source_id), key=lambda case: (case.get("position", 0), case["cue_id"]))


def _json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _ledger_summary(paths: list[Path], *, stage: str | None = None) -> dict[str, Any]:
    rows = [row for path in paths for row in _json_lines(path) if stage is None or row.get("stage") == stage]
    request_paths = [path.with_name("requests.jsonl") for path in paths]
    request_rows = [row for path in request_paths for row in _json_lines(path) if stage is None or row.get("stage") == stage]
    known = 0.0
    unknown = 0
    tokens = Counter()
    providers: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    for row in rows:
        stage = str(row.get("stage", "unknown"))
        stages[stage] += 1
        provider = str(row.get("provider", "unknown"))
        providers[provider] += 1
        estimate = row.get("cost_estimate") if isinstance(row.get("cost_estimate"), dict) else {}
        if isinstance(estimate.get("known_cost_usd"), (int, float)):
            known += float(estimate["known_cost_usd"])
        else:
            unknown += 1
        record = row.get("record") if isinstance(row.get("record"), dict) else {}
        usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
        for key in ("input_tokens", "output_tokens", "reasoning_tokens"):
            if isinstance(usage.get(key), int):
                tokens[key] += usage[key]
    starts: dict[str, dict[str, Any]] = {}
    completions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(request_rows):
        request_id = row.get("request_id")
        logical_id = request_id if isinstance(request_id, str) and request_id else f"missing-request-id-{index}"
        if row.get("state") == "start":
            starts.setdefault(logical_id, row)
        elif row.get("state") == "complete":
            completions[logical_id].append(row)
    call_request_ids = {
        row["request_id"] for row in rows
        if isinstance(row.get("request_id"), str) and row["request_id"]
    }
    logical_error_ids = sorted(
        logical_id
        for logical_id, completion_rows in completions.items()
        if any(bool(row.get("error")) for row in completion_rows)
    )
    unknown_request_ids = [logical_id for logical_id in logical_error_ids if logical_id not in call_request_ids]
    actual_call_error_ids = {
        row["request_id"] for row in rows
        if isinstance(row.get("request_id"), str)
        and isinstance(row.get("record"), dict)
        and (row["record"].get("status") == "error" or row["record"].get("error"))
    }
    return {"actual_request_count": len(starts), "ledger_call_count": len(rows), "known_subtotal_usd": known, "unknown_call_count": unknown,
            "error_request_count": len(logical_error_ids), "logical_error_ids": logical_error_ids,
            "unknown_request_count": len(unknown_request_ids), "unknown_request_ids": unknown_request_ids,
            "actual_call_error_count": len(actual_call_error_ids),
            "projection_incomplete": bool(unknown_request_ids),
            "provider_counts": dict(providers), "stage_counts": dict(stages), "tokens": dict(tokens),
            "physical_calls": len(rows)}


def _source_record_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["cue_id"]: record for record in report.get("records", []) if isinstance(record, dict) and isinstance(record.get("cue_id"), str)}


def _first_generation(record: dict[str, Any]) -> dict[str, Any] | None:
    for event in record.get("history", []):
        if isinstance(event, dict) and event.get("stage") == "generation":
            return event
    return None


def _opaque_id(seed: int, *parts: str) -> str:
    return "o-" + hashlib.sha256((str(seed) + "|" + "|".join(parts)).encode()).hexdigest()[:16]


async def _run_candidates(
    *, items: list[dict[str, Any]], plans: list[CuePlan], cases: list[dict[str, Any]], repetition: int,
    seed: int, generator: ModelSpec, request_fn: RequestFunction, output_dir: Path,
    length_tools: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not cases:
        return [], {
            "request_counts": {"generation_candidates": 0, "selection": 0},
            "request_failures": [],
            "records": [],
            "candidate_count_missing": 0,
            "candidate_count_duplicates": 0,
            "cost": summarize_costs([]),
        }
    output_dir.mkdir(parents=True)
    ids = [case["cue_id"] for case in cases]
    context = context_items(items, plans, ids, EXPECTED_PLANNER)
    plan_by_id = {plan.cue_id: plan for plan in plans}
    stage_calls: dict[str, list[CallRecord]] = {}
    request_counts: dict[str, int] = {"generation_candidates": 0, "selection": 0}
    failures: list[dict[str, Any]] = []
    candidates_by_id: dict[str, list[dict[str, Any]]] = {cue_id: [] for cue_id in ids}
    generation_errors: dict[str, str] = {}
    target_payload = [{"cue_id": cue_id, "original": plan_by_id[cue_id].text, "max_chars": plan_by_id[cue_id].max_chars} for cue_id in ids]
    response = await _request(request_fn, generator, CANDIDATE_SYSTEM_PROMPT, build_candidate_prompt(context, target_payload, 5),
                              candidate_schema(), length_tools, stage="generation_candidates", ids=ids, output_dir=output_dir,
                              stage_calls=stage_calls, request_counts=request_counts, request_failures=failures)
    parsed: dict[str, dict[str, Any]] | None = None
    if response.error is None and response.text is not None:
        parsed, partial_errors = parse_candidates_partial(response.text, ids)
        generation_errors.update(partial_errors)
    if parsed is None:
        for cue_id in ids:
            generation_errors[cue_id] = "transport" if response.error is not None else "schema"
    else:
        for cue_id in ids:
            if cue_id not in parsed:
                continue
            row = parsed[cue_id]
            seen_normalized: dict[str, str] = {}
            for ordinal, candidate in enumerate(row["candidates"]):
                text = candidate["text"]
                plan = plan_by_id[cue_id]
                local = _local_check(text, plan)
                normalized = _norm(text)
                reason: str | None = None
                if normalized in seen_normalized:
                    reason = "duplicate_normalized_text"
                elif not local["valid"]:
                    reason = "local_schema"
                elif not local["within_length"]:
                    reason = "overbudget"
                opaque = _opaque_id(seed, str(repetition), cue_id, str(ordinal), text)
                candidate_row = {"candidate_id": opaque, "model_candidate_id": candidate["candidate_id"], "text": text,
                                 "local": local, "normalized_hash": _sha(normalized.encode()), "rejection_reason": reason,
                                 "valid": reason is None}
                candidates_by_id[cue_id].append(candidate_row)
                if normalized not in seen_normalized:
                    seen_normalized[normalized] = opaque
    valid_options: dict[str, list[dict[str, Any]]] = {cue_id: [row for row in rows if row["valid"]] for cue_id, rows in candidates_by_id.items()}
    selection_ids = [cue_id for cue_id in ids if valid_options[cue_id]]
    selection_results: dict[str, dict[str, Any]] = {}
    selection_errors: dict[str, str] = {}
    if selection_ids:
        rng_targets: list[dict[str, Any]] = []
        for cue_id in selection_ids:
            plan = plan_by_id[cue_id]
            options = [{"candidate_id": row["candidate_id"], "text": row["text"],
                        "actual_chars": len(row["text"]), "max_chars": plan.max_chars}
                       for row in valid_options[cue_id]]
            random.Random(f"{seed}:selection:{repetition}:{cue_id}").shuffle(options)
            rng_targets.append({"cue_id": cue_id, "original": plan.text, "max_chars": plan.max_chars, "candidates": options})
        selector_context = [{"cue_id": row["cue_id"], "text": row["text"], "is_target": row["cue_id"] in selection_ids} for row in context]
        selector = await _request(request_fn, SOL_SELECTOR, SELECTION_SYSTEM_PROMPT, build_selection_prompt(selector_context, rng_targets),
                                   selection_schema(), False, stage="selection", ids=selection_ids, output_dir=output_dir,
                                   stage_calls=stage_calls, request_counts=request_counts, request_failures=failures)
        if selector.error is None and selector.text is not None:
            try:
                selection_results = parse_selection(selector.text, selection_ids)
            except Exception:
                selection_errors.update({cue_id: "schema" for cue_id in selection_ids})
        else:
            selection_errors.update({cue_id: "transport" for cue_id in selection_ids})
        known = {cue_id: {row["candidate_id"] for row in valid_options[cue_id]} for cue_id in selection_ids}
        for cue_id, result in list(selection_results.items()):
            selected = result["selected_candidate_id"]
            if selected is not None and selected not in known[cue_id]:
                selection_errors[cue_id] = "unknown_selected_candidate"
    records: list[dict[str, Any]] = []
    for case in cases:
        cue_id = case["cue_id"]
        selected_id = selection_results.get(cue_id, {}).get("selected_candidate_id")
        selected = next((row for row in valid_options[cue_id] if row["candidate_id"] == selected_id), None)
        error = generation_errors.get(cue_id) or selection_errors.get(cue_id)
        status = "accepted" if selected is not None and error is None else ("error" if error else "unavailable")
        candidate_rows = candidates_by_id[cue_id]
        records.append({"case_id": case["case_id"], "global_case_id": case["case_id"], "local_cue_id": cue_id,
                        "source_id": case["source_id"], "original": case["original"], "budget": {"max_chars": case["max_chars"], "available_ms": case["available_ms"]},
                        "candidates": candidate_rows, "candidate_counts": {"requested_n": 5, "returned_n": len(candidate_rows),
                                                                                "distinct_n": len({row["normalized_hash"] for row in candidate_rows}),
                                                                                "valid_n": sum(row["valid"] for row in candidate_rows)},
                        "candidate_errors": {"generation": generation_errors.get(cue_id), "selection": selection_errors.get(cue_id)},
                        "selection": selection_results.get(cue_id), "selected_candidate_id": selected_id if error is None else None,
                        "selected_text": selected["text"] if selected is not None and error is None else None, "status": status,
                        "provenance": {"candidate_prompt_version": CANDIDATE_PROMPT_VERSION, "selection_prompt_version": SELECTION_PROMPT_VERSION,
                                        "generator": asdict(generator), "selector": asdict(SOL_SELECTOR)}})
    source_copy = copy.deepcopy(items)
    c_checkpoint_records = [{"cue_id": row["local_cue_id"], "position": plan_by_id[row["local_cue_id"]].position,
                             "status": "accepted" if row["status"] == "accepted" else "manual", "final_text": row["selected_text"] or row["original"]}
                            for row in records]
    subtitles = _prepare_checkpoint(source_copy, c_checkpoint_records, plans, EXPECTED_PLANNER)
    _atomic_json(output_dir / "subtitles.json", subtitles)
    _export_srt(subtitles, output_dir)
    summary = {"request_counts": request_counts, "request_failures": failures,
               "records": records, "generation_errors": copy.deepcopy(generation_errors),
               "candidate_count_missing": sum(not row["candidates"] for row in records),
               "candidate_count_duplicates": sum(sum(candidate["rejection_reason"] == "duplicate_normalized_text" for candidate in row["candidates"]) for row in records),
               "cost": summarize_costs([call for calls in stage_calls.values() for call in calls])}
    _atomic_json(output_dir / "report.json", {"schema_version": 1, "prompt_versions": {"generation": CANDIDATE_PROMPT_VERSION, "selection": SELECTION_PROMPT_VERSION}, "summary": summary, "records": records})
    return records, summary


def _options_for_case(
    case: dict[str, Any], arms: dict[str, str], seed: int, salt: str,
    ordered_texts: list[str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    by_text: dict[str, dict[str, Any]] = {}
    for arm, text in arms.items():
        if not isinstance(text, str) or not text:
            continue
        key = _norm(text)
        if key not in by_text:
            by_text[key] = {"text": text, "arms": [arm]}
        else:
            by_text[key]["arms"].append(arm)
    options: list[dict[str, str]] = []
    reverse: dict[str, dict[str, Any]] = {}
    ordered_values = (
        [by_text[_norm(text)] for text in ordered_texts if _norm(text) in by_text]
        if ordered_texts is not None
        else sorted(by_text.values(), key=lambda row: _sha(row["text"].encode()))
    )
    if len(ordered_values) != len(by_text) or len({_norm(row["text"]) for row in ordered_values}) != len(by_text):
        raise ValueError("audit option order does not match available alternatives")
    for index, value in enumerate(ordered_values):
        option_id = _opaque_id(seed, salt, case["case_id"], str(index), value["text"])
        options.append({"option_id": option_id, "text": value["text"]})
        reverse[option_id] = {"text_sha": _sha(value["text"].encode()), "arms": value["arms"], "text": value["text"]}
    if ordered_texts is None:
        random.Random(f"{seed}:{salt}:{case['case_id']}").shuffle(options)
    return options, reverse


async def _run_audits(*, cases: list[dict[str, Any]], arm_values: dict[str, dict[str, str]], manifest: dict[str, Any], output_dir: Path,
                      request_fn: RequestFunction, seed: int) -> dict[str, Any]:
    audit_ids = [case_id for case_id in manifest.get("audit_case_ids", []) if isinstance(case_id, str)]
    cross_ids = [case_id for case_id in manifest.get("crosscheck_case_ids", []) if isinstance(case_id, str)]
    order_ids = [case_id for case_id in manifest.get("order_check_case_ids", []) if isinstance(case_id, str)]
    case_by_id = {case["case_id"]: case for case in cases}
    manifest_has_case_context = isinstance(manifest.get("cases"), list) and bool(manifest.get("cases"))
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {
        "prompt_version": AUDIT_PROMPT_VERSION, "context_included": True,
        "main": {}, "crosscheck": {}, "order": {}, "mappings": {},
        "judge_notes": {
            "main": "Different provider family from Sol selector; generator-family overlap is possible, so self-preference remains a caveat.",
            "crosscheck": "Same provider family as Sol selector; second opinion, not ground truth.",
            "order": "Same main auditor with reversed option order; measured stability only.",
        },
    }
    main_option_order: dict[str, list[str]] = {}

    async def judge(label: str, ids: list[str], model: ModelSpec, *, reverse_order: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {}
        judge_dir = output_dir / label
        judge_dir.mkdir(parents=True, exist_ok=True)
        for batch_number in range(0, len(ids), 6):
            batch_ids = ids[batch_number:batch_number + 6]
            targets: list[dict[str, Any]] = []
            request_ids: list[str] = []
            option_ids: dict[str, set[str]] = {}
            mapping: dict[str, Any] = {}
            case_context_hashes: dict[str, str] = {}
            for case_id in batch_ids:
                case = case_by_id.get(case_id)
                if case is None:
                    continue
                if "context" in case:
                    context = case["context"]
                    if not isinstance(context, list) or not context:
                        raise ValueError(f"audit context missing: {case_id}")
                elif manifest_has_case_context:
                    raise ValueError(f"audit context missing: {case_id}")
                else:
                    context = []
                case_context_hashes[case_id] = _canonical(context)
                ordered_texts = None
                if reverse_order:
                    ordered_texts = list(reversed(main_option_order.get(case_id, [])))
                options, reverse = _options_for_case(
                    case, arm_values.get(case_id, {}), seed, f"audit:{label}:{batch_number}", ordered_texts
                )
                if label == "main":
                    main_option_order[case_id] = [option["text"] for option in options]
                option_ids[case_id] = {option["option_id"] for option in options}
                mapping[case_id] = reverse
                if options:
                    request_ids.append(case_id)
                    audit_options = [
                        {**option, "actual_chars": len(option["text"]), "max_chars": case["max_chars"]}
                        for option in options
                    ]
                    target: dict[str, Any] = {
                        "case_id": case_id, "original": case["original"],
                        "max_chars": case["max_chars"], "context": copy.deepcopy(context),
                        "options": audit_options,
                    }
                    if isinstance(case.get("source_id"), str) and case["source_id"]:
                        target["source_id"] = _opaque_id(seed, "audit-source", case["source_id"])
                    targets.append(target)
                else:
                    result[case_id] = {"status": "unavailable", "judgment": None, "error": None,
                                       "mapping": {}, "prompt_version": AUDIT_PROMPT_VERSION,
                                       "context_included": True, "context_hash": case_context_hashes[case_id]}
            context_hashes = case_context_hashes
            _atomic_json(judge_dir / f"mapping-{batch_number // 6}.json", {
                "prompt_version": AUDIT_PROMPT_VERSION, "context_included": True,
                "context_hashes": context_hashes, "mappings": mapping,
            })
            if not targets:
                continue
            stage_calls: dict[str, list[CallRecord]] = {}
            counts = {"audit": 0}
            failures: list[dict[str, Any]] = []
            response = await _request(request_fn, model, AUDIT_SYSTEM_PROMPT,
                                      build_audit_prompt(targets), audit_schema(), False, stage="audit", ids=request_ids,
                                      output_dir=judge_dir, stage_calls=stage_calls, request_counts=counts, request_failures=failures)
            parsed: dict[str, dict[str, Any]] | None = None
            error: str | None = None
            if response.error is None and response.text is not None:
                try:
                    parsed = parse_audit(response.text, request_ids, {key: option_ids[key] for key in request_ids})
                except Exception:
                    error = "schema"
            else:
                error = "transport"
            for case_id in request_ids:
                result[case_id] = {"status": "error" if error else "complete",
                                   "judgment": parsed.get(case_id) if parsed else None, "error": error,
                                   "mapping": mapping.get(case_id, {}), "prompt_version": AUDIT_PROMPT_VERSION,
                                   "context_included": True,
                                   "context_hash": context_hashes.get(case_id)}
            _atomic_json(judge_dir / f"batch-{batch_number // 6}.json", {
                "prompt_version": AUDIT_PROMPT_VERSION, "context_included": True,
                "context_hashes": context_hashes, "results": result,
                "request_counts": counts, "request_failures": failures,
            })
        return result

    all_results["main"] = await judge("main", audit_ids, MAIN_AUDITOR)
    all_results["crosscheck"] = await judge("crosscheck", cross_ids, CROSS_AUDITOR)
    all_results["order"] = await judge("order", order_ids, MAIN_AUDITOR, reverse_order=True)
    planned_by_label = {"main": audit_ids, "crosscheck": cross_ids, "order": order_ids}
    for label in ("main", "crosscheck", "order"):
        rows = all_results[label].values()
        statuses = Counter(row.get("status") for row in rows)
        verdicts = Counter(
            judgment.get("verdict")
            for row in all_results[label].values()
            if isinstance(row.get("judgment"), dict)
            for judgment in row["judgment"].get("judgments", [])
            if isinstance(judgment, dict)
        )
        all_results[label + "_counts"] = {"planned_case_count": len(planned_by_label[label]),
                                           "case_status_counts": dict(statuses), "option_verdict_counts": dict(verdicts)}
    return all_results


def _audit_status(row: dict[str, Any] | None, arm: str) -> str:
    if row is None or row.get("status") == "unavailable":
        return "unavailable"
    if row.get("status") in {"error", "audit_error"} or not isinstance(row.get("judgment"), dict):
        return "audit_error"
    mapping = row.get("mapping", {})
    if not isinstance(mapping, dict) or not any(
        isinstance(value, dict) and arm in value.get("arms", []) for value in mapping.values()
    ):
        return "unavailable"
    accepted = [judgment for judgment in row["judgment"].get("judgments", [])
                if isinstance(judgment, dict) and judgment.get("verdict") == "accept" and judgment.get("option_id") in mapping]
    return "accept" if any(arm in mapping[item["option_id"]].get("arms", []) for item in accepted) else "reject"


def _build_effects(audits: dict[str, Any], cases: list[dict[str, Any]], arm_values: dict[str, dict[str, str]]) -> dict[str, Any]:
    transitions: Counter[str] = Counter()
    sol_flash: dict[str, Counter[str]] = {"B": Counter(), "C": Counter()}
    bc_transitions: Counter[str] = Counter()
    known_pair_count = 0
    unknown_pair_count = 0
    cases = [case for case in cases if case["case_id"] in audits.get("main", {})]
    c_enabled = any("C" in arm_values.get(case["case_id"], {}) for case in cases)
    for case in cases:
        case_id = case["case_id"]
        a, b, c = (_audit_status(audits.get("main", {}).get(case_id), arm) for arm in ("A", "B", "C"))
        if a == "reject" and b == "accept":
            transitions["rescued"] += 1
        elif a == "accept" and b == "reject":
            transitions["harmed"] += 1
        elif a == "accept" and b == "accept":
            transitions["bothpass"] += 1
        elif a == "reject" and b == "reject":
            transitions["bothfail"] += 1
        elif "audit_error" in {a, b}:
            transitions["pairedunknown"] += 1
        elif "unavailable" in {a, b}:
            transitions["unavailable"] += 1
        if a in {"accept", "reject"} and b in {"accept", "reject"}:
            known_pair_count += 1
        else:
            unknown_pair_count += 1
        if c_enabled:
            if b == "accept" and c == "accept": bc_transitions["bothpass"] += 1
            elif b == "reject" and c == "reject": bc_transitions["bothfail"] += 1
            elif "audit_error" in {b, c}: bc_transitions["pairedunknown"] += 1
            elif "unavailable" in {b, c}: bc_transitions["unavailable"] += 1
            else: bc_transitions["different"] += 1
        for arm, flash in (("B", b), ("C", c)):
            sol_selected = arm in arm_values.get(case_id, {})
            if sol_selected and flash in {"accept", "reject"}:
                if sol_selected == (flash == "accept"):
                    sol_flash[arm]["agree"] += 1
                else:
                    sol_flash[arm]["disagree"] += 1
    same = 0
    different = 0
    for case_id, row in audits.get("crosscheck", {}).items():
        main = audits.get("main", {}).get(case_id)
        if main is None or row.get("status") != "complete" or main.get("status") != "complete":
            continue
        main_map = main.get("mapping", {})
        cross_map = row.get("mapping", {})
        main_verdict = {main_map[j["option_id"]]["text_sha"]: j["verdict"] for j in main["judgment"]["judgments"] if j["option_id"] in main_map}
        cross_verdict = {cross_map[j["option_id"]]["text_sha"]: j["verdict"] for j in row["judgment"]["judgments"] if j["option_id"] in cross_map}
        if main_verdict == cross_verdict:
            same += 1
        else:
            different += 1
    stable = 0
    reversed_count = 0
    for case_id, row in audits.get("order", {}).items():
        main = audits.get("main", {}).get(case_id)
        if not main or row.get("status") != "complete" or main.get("status") != "complete":
            continue
        left = {main["mapping"][j["option_id"]]["text_sha"]: j["verdict"] for j in main["judgment"]["judgments"] if j["option_id"] in main["mapping"]}
        right = {row["mapping"][j["option_id"]]["text_sha"]: j["verdict"] for j in row["judgment"]["judgments"] if j["option_id"] in row["mapping"]}
        if left == right:
            stable += 1
        else:
            reversed_count += 1
    return {"planned_audit_case_count": len(cases), "auditable_paired_case_count": known_pair_count,
            "auditable_paired_count": known_pair_count, "auditable_pair_count": known_pair_count,
            "known_pair_count": known_pair_count,
            "unknown_pair_count": unknown_pair_count,
            "known_unknown_counts": {"known": known_pair_count, "unknown": unknown_pair_count},
            "paired_transitions": dict(transitions), "B_vs_C": dict(bc_transitions) if c_enabled else {"enabled": False},
            "solaccepted_vs_flash_audit": {arm: dict(values) for arm, values in sol_flash.items()},
            "cross_judge_agreement": {"agree": same, "disagree": different},
            "order_reversal": {"stable": stable, "disagree": reversed_count}}


def _aggregate_effects(effect_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not effect_rows:
        return {}

    def numeric_stats(values: list[int]) -> dict[str, float | int]:
        return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}

    def sum_counter(name: str) -> dict[str, int]:
        keys = {key for row in effect_rows for key in row.get(name, {}) if isinstance(row.get(name), dict)}
        return {key: sum(int(row.get(name, {}).get(key, 0)) for row in effect_rows) for key in sorted(keys)}

    aggregate = copy.deepcopy(effect_rows[0])
    aggregate["planned_audit_case_count"] = max(int(row.get("planned_audit_case_count", 0)) for row in effect_rows)
    for name in ("paired_transitions", "B_vs_C"):
        if name == "B_vs_C" and not any(row.get(name, {}).get("enabled") is not False for row in effect_rows):
            aggregate[name] = {"enabled": False}
        else:
            aggregate[name] = sum_counter(name)
    aggregate["auditable_paired_case_count"] = sum(int(row.get("auditable_paired_case_count", 0)) for row in effect_rows)
    aggregate["auditable_paired_count"] = aggregate["auditable_paired_case_count"]
    aggregate["auditable_pair_count"] = sum(int(row.get("auditable_pair_count", 0)) for row in effect_rows)
    aggregate["known_pair_count"] = sum(int(row.get("known_pair_count", 0)) for row in effect_rows)
    aggregate["unknown_pair_count"] = sum(int(row.get("unknown_pair_count", 0)) for row in effect_rows)
    aggregate["known_unknown_counts"] = {
        "known": aggregate["known_pair_count"], "unknown": aggregate["unknown_pair_count"],
    }
    sol_flash: dict[str, dict[str, int]] = {}
    for arm in ("B", "C"):
        sol_flash[arm] = {
            key: sum(int(row.get("solaccepted_vs_flash_audit", {}).get(arm, {}).get(key, 0)) for row in effect_rows)
            for key in ("agree", "disagree")
            if any(key in row.get("solaccepted_vs_flash_audit", {}).get(arm, {}) for row in effect_rows)
        }
    aggregate["solaccepted_vs_flash_audit"] = sol_flash
    aggregate["cross_judge_agreement"] = sum_counter("cross_judge_agreement")
    aggregate["order_reversal"] = sum_counter("order_reversal")
    aggregate["descriptive_statistics"] = {
        "paired_transitions": {
            key: numeric_stats([int(row.get("paired_transitions", {}).get(key, 0)) for row in effect_rows])
            for key in sorted({key for row in effect_rows for key in row.get("paired_transitions", {})})
        },
        "B_vs_C": {
            key: numeric_stats([int(row.get("B_vs_C", {}).get(key, 0)) for row in effect_rows])
            for key in sorted({key for row in effect_rows for key in row.get("B_vs_C", {}) if key != "enabled"})
        },
        "cross_judge_agreement": {
            key: numeric_stats([int(row.get("cross_judge_agreement", {}).get(key, 0)) for row in effect_rows])
            for key in sorted({key for row in effect_rows for key in row.get("cross_judge_agreement", {})})
        },
        "order_reversal": {
            key: numeric_stats([int(row.get("order_reversal", {}).get(key, 0)) for row in effect_rows])
            for key in sorted({key for row in effect_rows for key in row.get("order_reversal", {})})
        },
    }
    aggregate["aggregation_note"] = (
        "Summed observations across repetitions of the same corpus cases; the planned denominator remains "
        f"{aggregate['planned_audit_case_count']} unique cases, not {aggregate['planned_audit_case_count'] * len(effect_rows)} independent cases."
    )
    return aggregate


async def run_experiment(manifest: dict[str, Any], manifest_dir: Path, output_dir: Path, *, generator: ModelSpec,
                         repetitions: int = 2, best_of: int = 5, audit: bool = False, request_fn: RequestFunction,
                         seed: int = 20260910, length_tools: bool = False) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if best_of not in {0, 5}:
        raise ValueError("best_of must be 0 or 5")
    if not isinstance(length_tools, bool):
        raise TypeError("length_tools must be bool")
    if not isinstance(generator, ModelSpec):
        raise TypeError("generator must be ModelSpec")
    sources, source_meta, cases = _validate_and_load(manifest, Path(manifest_dir))
    output_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    plan_manifest = {"schema_version": 1, "experiment_version": EXPERIMENT_VERSION, "status": "running",
                     "corpus_sha256": _canonical(manifest), "sample_case_ids": [case["case_id"] for case in cases],
                     "params": {"repetitions": repetitions, "best_of": best_of, "audit": audit, "seed": seed,
                                 "length_tools": length_tools, "planner": asdict(EXPECTED_PLANNER),
                                 "run": asdict(RunConfig(gen_batch_size=6, review_batch_size=12, max_repairs=1, length_tools=length_tools))},
                     "case_metadata": [{"case_id": case["case_id"], "source_id": case["source_id"], "cue_id": case["cue_id"],
                                        "original": case["original"], "max_chars": case["max_chars"], "available_ms": case["available_ms"],
                                        "context": copy.deepcopy(case["context"])} for case in cases],
                     "source_partition_claims": [{"source_id": source_id, "split": source_meta[source_id].get("split"),
                                                  "case_ids": [case["case_id"] for case in _source_cases(cases, source_id)]}
                                                 for source_id in sorted(sources)],
                     "token_limits": {"generator": resolve_deepseek_token_limit(generator.variant) if generator.provider == "deepseek" else None,
                                      "main_auditor": resolve_deepseek_token_limit(MAIN_AUDITOR.variant), "cross_auditor": None},
                     "price_snapshot": {"id": CURRENT_PRICE_SNAPSHOT_ID, "value": copy.deepcopy(PRICE_SNAPSHOTS[CURRENT_PRICE_SNAPSHOT_ID])},
                     "prompt_strings": {"standard_generator": GENERATOR_SYSTEM_PROMPT, "standard_arbiter": ARBITER_SYSTEM_PROMPT,
                                        "candidate_system": CANDIDATE_SYSTEM_PROMPT, "selection_system": SELECTION_SYSTEM_PROMPT,
                                        "audit_system": AUDIT_SYSTEM_PROMPT},
                     "prompt_versions": {"standard": PROMPT_VERSION, "candidates": CANDIDATE_PROMPT_VERSION, "selection": SELECTION_PROMPT_VERSION, "audit": AUDIT_PROMPT_VERSION},
                     "models": {"generator": asdict(generator), "selector": asdict(SOL_SELECTOR), "main_auditor": asdict(MAIN_AUDITOR), "cross_auditor": asdict(CROSS_AUDITOR)},
                     "routing_note": "A shares B generation; B uses the standard repair loop; C uses one candidate batch and one Sol selection batch.",
                     "clock": {"started_at": started}}
    _atomic_json(output_dir / "experiment_manifest.json", plan_manifest)
    all_case_results: list[dict[str, Any]] = []
    ledger_paths: list[Path] = []
    repetition_ledgers: list[dict[str, list[Path]]] = []
    repetition_summaries: list[dict[str, Any]] = []
    audits_by_repetition: list[dict[str, Any]] = []
    effects_by_repetition: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        rep_root = output_dir / f"r{repetition + 1}"
        rep_root.mkdir()
        arm_values: dict[str, dict[str, str]] = {}
        c_records_by_id: dict[str, dict[str, Any]] = {}
        rep_audits: dict[str, Any] = {}
        rep_effects: dict[str, Any] = {}
        counts: Counter[str] = Counter()
        source_request_counts: dict[str, dict[str, int]] = {}
        source_candidate_summaries: dict[str, dict[str, Any]] = {}
        repetition_standard_paths: list[Path] = []
        repetition_candidate_paths: list[Path] = []
        referenced_source_ids = sorted({case["source_id"] for case in cases})
        for source_id in referenced_source_ids:
            items = copy.deepcopy(sources[source_id])
            plans = plan_subtitles(items, EXPECTED_PLANNER)
            local_cases = _source_cases(cases, source_id)
            target_ids = [case["cue_id"] for case in local_cases]
            standard_dir = rep_root / "standard" / source_id
            b_report = await run_trial(items=items, plans=plans, target_ids=target_ids, planner_config=EXPECTED_PLANNER,
                                       run_config=RunConfig(gen_batch_size=6, review_batch_size=12, max_repairs=1, length_tools=length_tools), generator=generator,
                                       arbiter=SOL_SELECTOR, output_dir=standard_dir, request_fn=request_fn)
            ledger_paths.append(standard_dir / "calls.jsonl")
            repetition_standard_paths.append(standard_dir / "calls.jsonl")
            b_records = _source_record_map(b_report)
            c_records: list[dict[str, Any]] = []
            c_summary: dict[str, Any] = {}
            if best_of == 5:
                c_records, c_summary = await _run_candidates(items=items, plans=plans, cases=local_cases, repetition=repetition,
                                                              seed=seed, generator=generator, request_fn=request_fn,
                                                              output_dir=rep_root / "best5" / source_id, length_tools=length_tools)
                ledger_paths.append(rep_root / "best5" / source_id / "calls.jsonl")
                repetition_candidate_paths.append(rep_root / "best5" / source_id / "calls.jsonl")
                c_records_by_id.update({record["case_id"]: record for record in c_records})
            for case in local_cases:
                cue_id, case_id = case["cue_id"], case["case_id"]
                b = b_records.get(cue_id, {})
                first = _first_generation(b)
                a_local_fit = bool(
                    first
                    and first.get("generator_status") == "ok"
                    and first.get("local", {}).get("valid")
                    and first.get("local", {}).get("within_length")
                )
                a_text = first.get("candidate") if a_local_fit else None
                b_text = b.get("final_text") if b.get("status") == "accepted" else None
                c = c_records_by_id.get(case_id, {})
                c_text = c.get("selected_text") if c.get("status") == "accepted" else None
                counts["A_unreviewed"] += a_text is not None
                counts["A_available_localfit"] += a_text is not None
                counts["B_available_localfit"] += a_local_fit
                counts["B_selected_by_Sol"] += b_text is not None
                counts["C_selected_by_Sol"] += c_text is not None
                counts["solvererrors"] += b.get("status") == "error" or c.get("status") == "error"
                counts["candidate_missing"] += int(best_of == 5 and not c.get("candidates"))
                counts["candidate_duplicates"] += sum(candidate.get("rejection_reason") == "duplicate_normalized_text" for candidate in c.get("candidates", []))
                result = {"global_case_id": case_id, "local_cue_id": cue_id, "source_id": source_id, "repetition": repetition + 1,
                          "original": case["original"],
                          "budget": {"max_chars": case["max_chars"], "available_ms": case["available_ms"]},
                          "context": copy.deepcopy(case["context"]), "prompt_versions": {"standard": PROMPT_VERSION, "candidates": CANDIDATE_PROMPT_VERSION if best_of else None},
                          "A": {"text": a_text, "status": "unreviewed" if a_text is not None else "unavailable", "candidate_hash": _sha(a_text.encode()) if a_text else None},
                          "B": {"text": b_text, "status": "accepted" if b_text is not None else "unavailable", "candidate_hash": _sha(b_text.encode()) if b_text else None},
                          "C": c if best_of == 5 else {"status": "unavailable", "reason": "best_of_disabled"},
                          "source_index": case.get("source_index")}
                all_case_results.append(result)
                arm_values[case_id] = {arm: text for arm, text in (("A", a_text), ("B", b_text), ("C", c_text)) if text is not None}
            source_request_counts[source_id] = b_report.get("summary", {}).get("request_counts", {})
            source_candidate_summaries[source_id] = c_summary
            _atomic_json(output_dir / "case_results.json", all_case_results)
            _atomic_json(output_dir / "experiment_summary.json", {"status": "running", "repetitions": repetition_summaries, "ledger": _ledger_summary(ledger_paths)})
        if audit:
            rep_audits = await _run_audits(cases=cases, arm_values=arm_values, manifest=manifest,
                                           output_dir=rep_root / "audits", request_fn=request_fn, seed=seed)
            audit_paths = [rep_root / "audits" / label / "calls.jsonl" for label in ("main", "crosscheck", "order")]
            ledger_paths.extend(audit_paths)
            rep_effects = _build_effects(rep_audits, cases, arm_values)
            audits_by_repetition.append(rep_audits)
            effects_by_repetition.append(rep_effects)
        rep_summary = {"repetition": repetition + 1, "arm_counts": dict(counts),
                        "standard_request_counts": source_request_counts,
                        "candidate_summary": source_candidate_summaries,
                        "audits": rep_audits if audit else {}, "effects": rep_effects if audit else {}}
        repetition_ledgers.append({"standard": repetition_standard_paths, "best5": repetition_candidate_paths})
        repetition_summaries.append(rep_summary)
        _atomic_json(output_dir / "case_results.json", all_case_results)
        _atomic_json(output_dir / "experiment_summary.json", {"status": "running", "repetitions": repetition_summaries, "ledger": _ledger_summary(ledger_paths)})
    audits = audits_by_repetition[0] if audits_by_repetition else {}
    effects = _aggregate_effects(effects_by_repetition) if effects_by_repetition else {}
    aggregate_counts = {key: {"mean": sum(summary["arm_counts"].get(key, 0) for summary in repetition_summaries) / repetitions,
                              "min": min(summary["arm_counts"].get(key, 0) for summary in repetition_summaries),
                              "max": max(summary["arm_counts"].get(key, 0) for summary in repetition_summaries)}
                        for key in sorted({key for summary in repetition_summaries for key in summary["arm_counts"]})}
    ledger = _ledger_summary(ledger_paths)
    b_ledger = _ledger_summary([path for path in ledger_paths if "/standard/" in str(path)])
    c_ledger = _ledger_summary([path for path in ledger_paths if "/best5/" in str(path)])
    a_ledger = _ledger_summary([path for path in ledger_paths if "/standard/" in str(path)], stage="generation")
    b_cost = b_ledger["known_subtotal_usd"]
    a_cost = a_ledger["known_subtotal_usd"]
    c_cost = c_ledger["known_subtotal_usd"]
    audit_cost = _ledger_summary([path for path in ledger_paths if "/audits/" in str(path)])["known_subtotal_usd"]
    factor = 500 / max(1, len(cases))

    def projection(stage: str | None, key: str) -> dict[str, Any]:
        per_repetition: list[dict[str, Any]] = []
        for index, paths_by_stage in enumerate(repetition_ledgers, 1):
            summary = _ledger_summary(paths_by_stage[key], stage=stage)
            unknown = bool(summary["unknown_call_count"] or summary["projection_incomplete"])
            per_repetition.append({
                "repetition": index,
                "known_cost_usd": summary["known_subtotal_usd"] * factor,
                "projected_tokens": {name: value * factor for name, value in summary["tokens"].items()},
                "unknown_cost": unknown,
                "applicable": not unknown,
                "unknown_ledger": unknown,
                "charged_error_count": summary["error_request_count"],
                "unknown_request_count": summary["unknown_request_count"],
                "projection_incomplete": summary["projection_incomplete"],
            })
        costs = [row["known_cost_usd"] for row in per_repetition]
        mean = sum(costs) / len(costs) if costs else 0.0
        inapplicable = [row["repetition"] for row in per_repetition if not row["applicable"]]
        return {
            "known_cost_usd": {"mean": mean, "min": min(costs) if costs else 0.0, "max": max(costs) if costs else 0.0},
            "known_cost_usd_mean": mean,
            "known_cost_usd_min": min(costs) if costs else 0.0,
            "known_cost_usd_max": max(costs) if costs else 0.0,
            "per_repetition": per_repetition,
            "unknown_cost": bool(inapplicable),
            "applicable": not inapplicable,
            "inapplicable_repetitions": inapplicable,
            "basis": "mean/min/max of per-repetition projections; no final audit included",
            "same_corpus_mix_approximation": True,
            "final_audit_included": False,
        }
    final = {"status": "completed", "schema_version": 1, "corpus_sha256": _canonical(manifest),
              "sample_case_ids": [case["case_id"] for case in cases], "planned_case_count_per_repetition": len(cases), "repetitions": repetition_summaries,
              "arm_counts_whole_sample": aggregate_counts,
             "arm_counts_per_repetition": aggregate_counts,
             "arm_counts_note": f"Repeated runs reuse the same {len(cases)} cases; means/min/max are descriptive, not independent winner claims.",
             "audits": audits, "audits_alias_note": "Backwards-compatible alias for the first repetition; use audits_by_repetition for every repetition.",
             "audits_by_repetition": audits_by_repetition, "effects": effects,
             "effects_by_repetition": effects_by_repetition, "cost": {
                  "experimental_physical": {**ledger, "total_known_subtotal_usd": ledger["known_subtotal_usd"],
                                               "total_note": "All repetitions are included once; audits are itemized separately below."},
                  "by_arm": {"A": {"known_subtotal_usd": a_cost, "attribution": "generation shared with B; not added to physical total"},
                             "B": {"known_subtotal_usd": b_cost, "workflow": "standard"},
                             "C": {"known_subtotal_usd": c_cost, "workflow": "best5"},
                             "audits": {"known_subtotal_usd": audit_cost, "workflow": "separate; excluded from production-arm projections"}},
                  "projection_500": {"A": projection("generation", "standard"), "B": projection(None, "standard"),
                                     "C": projection(None, "best5") if best_of == 5 else {"applicable": False, "reason": "best_of_disabled"}},
              },
             "routing_note": plan_manifest["routing_note"], "clock": {"started_at": started, "finished_at": datetime.now(timezone.utc).isoformat()}}
    _atomic_json(output_dir / "case_results.json", all_case_results)
    _atomic_json(output_dir / "experiment_summary.json", final)
    plan_manifest["status"] = "completed"
    plan_manifest["clock"]["finished_at"] = final["clock"]["finished_at"]
    _atomic_json(output_dir / "experiment_manifest.json", plan_manifest)
    return final


def _generator(value: str) -> ModelSpec:
    provider, variant = value.split(":", 1)
    if provider == "deepseek" and variant in {"off", "low", "high", "max"}:
        return ModelSpec("deepseek", "deepseek-flash", variant)
    if provider == "opencode" and variant in {"none", "low", "medium", "high"}:
        return ModelSpec("opencode", "openai/gpt-5.6-luna", variant)
    raise ValueError("invalid generator")


def _live_backend(spec: ModelSpec, audit: bool, timeout: float, *, length_tools: bool = False) -> tuple[CleanOpenCodeServer, LiveBackend]:
    required = {"openai/gpt-5.6-sol": {"medium"}, "openai/gpt-5.6-luna": {"medium"}}
    if spec.provider == "opencode":
        required[spec.model].add(spec.variant)
    server = CleanOpenCodeServer(enable_length_tool=length_tools, required_models=required, timeout=min(timeout, 45.0))
    server.start()
    backend = LiveBackend(server, deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY"), timeout=timeout, max_tokens=None)
    return server, backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the source-isolated A/B/C shortening experiment")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--best-of", type=int, choices=[0, 5], default=5)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--length-tools", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        generator = _generator(args.generator)
        _validate_and_load(manifest, args.manifest.parent)
        if not args.execute:
            args.output_dir.mkdir(parents=True)
            _atomic_json(args.output_dir / "experiment_plan.json", {"status": "dry_run", "manifest": _canonical(manifest), "generator": asdict(generator),
                                                                      "repetitions": args.repetitions, "best_of": args.best_of, "audit": args.audit,
                                                                      "length_tools": args.length_tools, "timeout": args.timeout})
            print(json.dumps({"status": "dry_run", "output": str(args.output_dir)}))
            return
        if generator.provider == "deepseek" or args.audit:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
        server, backend = _live_backend(generator, args.audit, args.timeout, length_tools=args.length_tools)
        try:
            from utils.shortening_html import live_html_report
            with live_html_report(args.output_dir):
                result = asyncio.run(run_experiment(manifest, args.manifest.parent, args.output_dir, generator=generator,
                                                    repetitions=args.repetitions, best_of=args.best_of, audit=args.audit,
                                                    request_fn=backend, length_tools=args.length_tools))
        finally:
            server.stop()
        print(json.dumps({"status": result["status"], "output": str(args.output_dir)}))
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(type(exc).__name__ + ": validation or preflight failed") from exc


if __name__ == "__main__":
    main()
