"""Standalone, fail-closed semantic verification through local OpenCode OAuth."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, Union

from utils.analyze_text import join_text_lines
from utils.opencode_transport import OpenCodePromptResult, OpenCodeServer, _get_auth_header, create_session, delete_session, send_prompt_result
from utils.shorten_helpers import load_json, validate_context_source
from utils.semantic_requirements import PLANNER_ISSUE_CODES, extract_semantic_requirements
from utils.semantic_units import SemanticUnit, build_semantic_units

ISSUE_CODES = PLANNER_ISSUE_CODES
VERDICTS = frozenset({"pass", "fail", "uncertain"})
SEVERITIES = frozenset({"none", "minor", "major"})
RequestResponse = Union[str, OpenCodePromptResult]
RequestFunc = Callable[[str, str], Awaitable[Optional[RequestResponse]]]


def unit_verification_schema() -> Dict[str, Any]:
    """Return a fresh OpenCode structured-output schema for semantic units."""
    return {
        "type": "object",
        "required": ["results"],
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["cue_indices", "verdict", "issues", "severity", "explanation"],
                    "additionalProperties": False,
                    "properties": {
                        "cue_indices": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
                        "verdict": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
                        "issues": {"type": "array", "items": {"type": "string", "enum": sorted(ISSUE_CODES)}},
                        "severity": {"type": "string", "enum": ["none", "minor", "major"]},
                        "explanation": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
    }

INDEPENDENT_VERIFIER_SYSTEM_PROMPT = (
    "You are an independent semantic verifier. Return exactly JSON with keys results only; each result has exactly "
    "index, verdict, issues, severity, explanation. verdict is pass, fail, or uncertain; severity is none, minor, or major; "
    "issues must contain only these accepted codes: " + ", ".join(sorted(ISSUE_CODES)) + ". explanation is a short nonempty string. "
    "Do not use markdown, tools, rewrites, or reasoning. Judge propositions and context independently. Preserve propositions, "
    "entities/addressees, agent/predicate/object, polarity/negation, modality, tense, time, causality, comparison, alternatives, "
    "references, spatial relations, speech act, and cross-cue boundaries. Filler deletion and natural compression are allowed; "
    "literal substring preservation is not required. Pass requires empty issues and severity none; fail or uncertain requires "
    "nonempty issues and minor or major severity."
)

UNIT_VERIFIER_SYSTEM_PROMPT = (
    "You are an independent semantic verifier. Return exactly a JSON object with exactly one root key, results. "
    "Each result object must contain exactly these keys and no others: cue_indices, verdict, issues, severity, explanation. "
    "Never return unit_id, changed_indices, original_cues, candidate_cues, context, risk_hints, markdown, tools, rewrites, or reasoning. "
    "Copy cue_indices exactly from the target: never add, remove, reorder, or invent indices. "
    "Compare the complete meaning of original_cues with candidate_cues using context. Judge changed_indices; unchanged members and context are evidence only. "
    "Preserve every proposition, entity and addressee, agent/predicate/object, polarity and negation, modality, tense, time, causality, comparison, alternatives, references, spatial relation, speech act, and cross-cue boundary. "
    "Filler deletion, natural paraphrase, and natural compression are allowed; literal substring preservation is not required. "
    "Discourse markers or hesitations such as э-э, ну, итак, and да may be removed only when they are contextual filler and not an answer, polarity change, or speech-act change. "
    "Do not fail solely for style or wording, or for an implicit agent or reference that remains unambiguous from the complete unit and context. "
    "Fail or mark uncertain only for material semantic loss, material semantic change, or material ambiguity. "
    "verdict is pass, fail, or uncertain; severity is none, minor, or major; issues must contain only these accepted codes: "
    + ", ".join(sorted(ISSUE_CODES))
    + ". explanation is a short nonempty string. Pass requires empty issues and severity none. Fail or uncertain requires nonempty issues and minor or major severity. "
    "These verdict, issues, and severity consistency rules are exact."
)


def parse_verification_response(raw: str, expected_indices: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    """Strictly parse the verifier schema; no coercion or repair is allowed."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("schema_empty")
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("schema_json") from exc
    if not isinstance(value, dict) or set(value) != {"results"} or not isinstance(value["results"], list):
        raise ValueError("schema_keys")
    expected = set(expected_indices)
    result: Dict[int, Dict[str, Any]] = {}
    for row in value["results"]:
        if not isinstance(row, dict) or set(row) != {"index", "verdict", "issues", "severity", "explanation"}:
            raise ValueError("result_keys")
        index = row["index"]
        verdict = row["verdict"]
        issues = row["issues"]
        severity = row["severity"]
        explanation = row["explanation"]
        if isinstance(index, bool) or not isinstance(index, int) or index not in expected or index in result:
            raise ValueError("result_index")
        if verdict not in VERDICTS or severity not in SEVERITIES or not isinstance(issues, list) or not all(isinstance(x, str) and x in ISSUE_CODES for x in issues):
            raise ValueError("result_values")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("result_explanation")
        if verdict == "pass" and (issues or severity != "none"):
            raise ValueError("pass_consistency")
        if verdict in {"fail", "uncertain"} and (not issues or severity not in {"minor", "major"}):
            raise ValueError("nonpass_consistency")
        result[index] = row
    if set(result) != expected:
        raise ValueError("result_indices")
    return result


def parse_unit_verification_response(raw: str, expected_units: Sequence[SemanticUnit]) -> Dict[str, Dict[str, Any]]:
    """Strictly parse one result for each expected cue tuple.

    The returned mapping uses internal unit IDs only for downstream convenience;
    those IDs are never accepted from model output.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("schema_empty")
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("schema_json") from exc
    if not isinstance(value, dict) or set(value) != {"results"} or not isinstance(value["results"], list):
        raise ValueError("schema_keys")
    expected_by_cues = {tuple(unit.cue_indices): unit for unit in expected_units}
    if len(expected_by_cues) != len(expected_units):
        raise ValueError("expected_duplicate_cue_indices")
    if len({unit.unit_id for unit in expected_units}) != len(expected_units):
        raise ValueError("expected_duplicate_unit_id")
    result: Dict[str, Dict[str, Any]] = {}
    for row in value["results"]:
        if not isinstance(row, dict) or set(row) != {"cue_indices", "verdict", "issues", "severity", "explanation"}:
            raise ValueError("result_keys")
        cue_indices = row["cue_indices"]
        cue_tuple = tuple(cue_indices) if isinstance(cue_indices, list) else None
        if (not isinstance(cue_indices, list) or any(isinstance(index, bool) or not isinstance(index, int) for index in cue_indices)
                or cue_tuple not in expected_by_cues or expected_by_cues[cue_tuple].unit_id in result):
            raise ValueError("result_cue_indices")
        unit = expected_by_cues[cue_tuple]
        verdict, issues, severity, explanation = row["verdict"], row["issues"], row["severity"], row["explanation"]
        if verdict not in VERDICTS or severity not in SEVERITIES or not isinstance(issues, list) or not all(isinstance(x, str) and x in ISSUE_CODES for x in issues):
            raise ValueError("result_values")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("result_explanation")
        if verdict == "pass" and (issues or severity != "none"):
            raise ValueError("pass_consistency")
        if verdict in {"fail", "uncertain"} and (not issues or severity not in {"minor", "major"}):
            raise ValueError("nonpass_consistency")
        result[unit.unit_id] = row
    if set(result) != {unit.unit_id for unit in expected_units}:
        raise ValueError("result_units")
    return result


def _validate_items(original: Any, candidate: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not isinstance(original, list) or not isinstance(candidate, list):
        raise ValueError("input_not_arrays")
    def indexed(items: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict) or isinstance(item.get("index"), bool) or not isinstance(item.get("index"), int) or item["index"] in out:
                raise ValueError("invalid_or_duplicate_index")
            out[item["index"]] = item
        return out
    left, right = indexed(original), indexed(candidate)
    if set(left) != set(right):
        raise ValueError("index_set_mismatch")
    return [left[index] for index in left], [right[index] for index in left]


def _target_payload(index: int, original_by_index: Dict[int, Dict[str, Any]], candidate_by_index: Dict[int, Dict[str, Any]], context_slice: List[Dict[str, Any]]) -> Dict[str, Any]:
    original_text = join_text_lines(original_by_index[index].get("text", ""))
    candidate_text = join_text_lines(candidate_by_index[index].get("text", ""))
    requirements = extract_semantic_requirements(original_text, candidate_text)
    context = [{"index": int(item["index"]), "original_text": join_text_lines(item.get("text", "")), "current_text": join_text_lines(candidate_by_index[int(item["index"])].get("text", "")) if int(item["index"]) in candidate_by_index else join_text_lines(item.get("text", ""))} for item in context_slice]
    return {"index": index, "original_text": original_text, "candidate_text": candidate_text,
            "context": context,
            "risk_hints": [requirement.as_dict() for requirement in requirements]}


def _parse_prior_outcomes(report: Any, known_indices: set[int], candidate_by_index: Dict[int, Dict[str, Any]]) -> Dict[int, str]:
    if not isinstance(report, dict) or not isinstance(report.get("outcomes"), list):
        raise ValueError("prior_schema_outcomes")
    result: Dict[int, str] = {}
    for row in report["outcomes"]:
        if not isinstance(row, dict):
            raise ValueError("prior_schema_row")
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("prior_schema_index")
        if index not in known_indices:
            raise ValueError("prior_schema_unknown_index")
        if index in result:
            raise ValueError("prior_schema_duplicate_index")
        outcome = row.get("outcome")
        if not isinstance(outcome, str):
            raise ValueError("prior_schema_outcome")
        result[index] = outcome
        evidence: Any = None
        for key in ("candidate_text", "final_text", "candidate"):
            if key in row and row[key] is not None:
                evidence = row[key]
                break
        if evidence is None and isinstance(row.get("lineage"), dict) and "final" in row["lineage"]:
            evidence = row["lineage"]["final"]
        if evidence is None:
            raise ValueError("prior_schema_missing_evidence")
        if join_text_lines(evidence) != join_text_lines(candidate_by_index[index].get("text", "")):
            raise ValueError("prior_schema_text_mismatch")
    return result


def _restore_original_text(output_item: Dict[str, Any], original_item: Dict[str, Any]) -> None:
    original = join_text_lines(original_item.get("text", ""))
    output_item["text"] = [original] if isinstance(output_item.get("text"), list) else original


def _validate_runtime_args(model: str, context_window: int, batch_size: int, concurrency: int, transport_retries: int, schema_retries: int,
                           semantic_unit_max_cues: int = 3, semantic_unit_max_gap_sec: float = 0.3) -> None:
    if (not model or context_window < 0 or not 1 <= batch_size <= 100 or not 1 <= concurrency <= 100
            or not 0 <= transport_retries <= 5 or schema_retries not in {0, 1}
            or isinstance(semantic_unit_max_cues, bool) or not isinstance(semantic_unit_max_cues, int) or not 1 <= semantic_unit_max_cues <= 5
            or isinstance(semantic_unit_max_gap_sec, bool) or not isinstance(semantic_unit_max_gap_sec, (int, float))
            or not math.isfinite(float(semantic_unit_max_gap_sec)) or semantic_unit_max_gap_sec < 0):
        raise ValueError("invalid_arguments")


def _validate_server_config(server_url: Optional[str], hostname: str, port: Optional[int]) -> None:
    if server_url is not None and not server_url.strip():
        raise ValueError("server_url_required")
    if server_url is None and hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("auto-start hostname must be localhost or 127.0.0.1")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")


async def verify_items(
    original: List[Dict[str, Any]], candidate: List[Dict[str, Any]], *, model: str = "openai/gpt-5.6-sol", context_items: Optional[List[Dict[str, Any]]] = None,
    context_window: int = 3, batch_size: int = 4, concurrency: int = 3, transport_retries: int = 1, schema_retries: int = 1,
    review_report: Optional[Dict[str, Any]] = None, request_callable: Optional[RequestFunc] = None,
    semantic_units: bool = False, semantic_unit_max_cues: int = 3, semantic_unit_max_gap_sec: float = 0.3,
    backend: str = "opencode", accounting: str = "subscription", cost_is_billing_authoritative: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    if not isinstance(backend, str) or not backend.strip() or not isinstance(accounting, str) or not accounting.strip() or not isinstance(cost_is_billing_authoritative, bool):
        raise ValueError("invalid_backend_accounting")
    _validate_runtime_args(model, context_window, batch_size, concurrency, transport_retries, schema_retries, semantic_unit_max_cues, semantic_unit_max_gap_sec)
    originals, candidates = _validate_items(original, candidate)
    if context_items is not None:
        validate_context_source(originals, context_items)
    by_index = {int(item["index"]): item for item in originals}
    candidate_by_index = {int(item["index"]): item for item in candidates}
    changed = [i for i in by_index if join_text_lines(by_index[i].get("text", "")) != join_text_lines(candidate_by_index[i].get("text", ""))]
    prior: Dict[int, str] = {}
    schema_error: Optional[str] = None
    if review_report is not None:
        try:
            prior = _parse_prior_outcomes(review_report, set(by_index), candidate_by_index)
        except ValueError as exc:
            schema_error = str(exc)
    selected = changed if review_report is None else [i for i in changed if prior.get(i) == "verified"]
    if selected and request_callable is None:
        raise ValueError("request_callable_required")
    output = copy.deepcopy(candidate)
    details: Dict[int, Dict[str, Any]] = {i: {"index": i, "original": join_text_lines(by_index[i].get("text", "")), "candidate": join_text_lines(candidate_by_index[i].get("text", "")), "output": join_text_lines(candidate_by_index[i].get("text", "")), "prior_outcome": prior.get(i), "verdict": None, "issues": [], "severity": None, "explanation": None, "error": None, "reason": None, "fallback": None} for i in changed}
    if schema_error:
        selected = []
        for detail in details.values(): detail.update(error=schema_error, reason="prior_schema_failure", fallback="prior_schema_failure")
    else:
        for i in changed:
            if i not in selected:
                details[i]["reason"] = "prior_not_verified" if review_report is not None else "unchanged_selection"
                details[i]["fallback"] = "prior_not_verified" if review_report is not None else "not_selected"
    effective_candidates = copy.deepcopy(candidates)
    effective_candidate_by_index = {int(item["index"]): item for item in effective_candidates}
    if not schema_error:
        for index in changed:
            if index not in selected:
                effective_candidate_by_index[index]["text"] = copy.deepcopy(by_index[index].get("text", ""))
    requests = transport_retries_used = transport_failures = schema_failures = successful = 0
    usage_totals = {name: 0 for name in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens")}
    reported_cost = 0.0
    usage_response_count = 0
    token_usage_response_count = 0
    provider_ids: set[str] = set()
    model_ids: set[str] = set()
    finishes: set[str] = set()
    schema_split_retry = False
    sem = asyncio.Semaphore(concurrency)
    context_by_index = {int(x["index"]): x for x in (context_items or originals)}
    ordered_context = list(context_by_index.values())
    positions = {int(item["index"]): pos for pos, item in enumerate(ordered_context)}

    units: Tuple[SemanticUnit, ...] = ()
    if semantic_units and not schema_error:
        units = build_semantic_units(originals, effective_candidates, context_items=context_items, target_indices=selected,
                                     max_cues=semantic_unit_max_cues, max_gap_sec=semantic_unit_max_gap_sec)

    async def request_once(prompt: str) -> Optional[str]:
        nonlocal requests, transport_failures, reported_cost, usage_response_count, token_usage_response_count
        async with sem:
            requests += 1
            try:
                response = await request_callable(prompt, model)
                if isinstance(response, OpenCodePromptResult):
                    raw_response = response.text
                    if response.usage is not None:
                        usage_response_count += 1
                        has_token_usage = response.usage.cost is not None or any(getattr(response.usage, name) is not None for name in usage_totals)
                        if has_token_usage:
                            token_usage_response_count += 1
                            for name in usage_totals:
                                value = getattr(response.usage, name)
                                if value is not None:
                                    usage_totals[name] += value
                            if response.usage.cost is not None:
                                reported_cost += response.usage.cost
                            if response.usage.provider_id is not None:
                                provider_ids.add(response.usage.provider_id)
                            if response.usage.model_id is not None:
                                model_ids.add(response.usage.model_id)
                            if response.usage.finish is not None:
                                finishes.add(response.usage.finish)
                elif isinstance(response, str):
                    raw_response = response
                else:
                    raw_response = None
                if raw_response is None or not raw_response.strip():
                    transport_failures += 1
                    return None
                return raw_response
            except Exception:
                transport_failures += 1
                return None

    def unit_payload(unit: SemanticUnit) -> Dict[str, Any]:
        unit_positions = [positions[index] for index in unit.cue_indices]
        original_cues = [{"index": index, "text": join_text_lines(context_by_index[index].get("text", ""))} for index in unit.cue_indices]
        candidate_cues = [{"index": index, "text": join_text_lines(effective_candidate_by_index[index].get("text", "")) if index in effective_candidate_by_index else join_text_lines(context_by_index[index].get("text", ""))} for index in unit.cue_indices]
        first, last = min(unit_positions), max(unit_positions)
        context = [{"index": int(item["index"]), "original_text": join_text_lines(item.get("text", "")),
                    "current_text": join_text_lines(item.get("text", ""))}
                   for pos, item in enumerate(ordered_context) if pos < first and first - pos <= context_window or pos > last and pos - last <= context_window]
        risk_hints = []
        for index in unit.changed_indices:
            original_text = join_text_lines(by_index[index].get("text", ""))
            candidate_text = join_text_lines(effective_candidate_by_index[index].get("text", ""))
            risk_hints.append({"index": index, "requirements": [requirement.as_dict() for requirement in extract_semantic_requirements(original_text, candidate_text)]})
        return {"cue_indices": list(unit.cue_indices), "changed_indices": list(unit.changed_indices),
                "original_cues": original_cues, "candidate_cues": candidate_cues, "context": context, "risk_hints": risk_hints}

    async def verify_batch(batch: Union[List[int], List[SemanticUnit]]) -> None:
        nonlocal transport_retries_used, schema_failures, successful, schema_split_retry
        payload = {"targets": []}
        if semantic_units:
            unit_batch = batch
            payload["targets"] = [unit_payload(unit) for unit in unit_batch]  # type: ignore[arg-type]
            changed_indices = [index for unit in unit_batch for index in unit.changed_indices]  # type: ignore[union-attr]
        else:
            indices = batch
            for i in indices:  # type: ignore[union-attr]
                pos = positions[i]
                payload["targets"].append(_target_payload(i, by_index, candidate_by_index, ordered_context[max(0, pos-context_window):pos+context_window+1]))
            changed_indices = indices  # type: ignore[assignment]
        raw: Optional[str] = None
        for attempt in range(transport_retries + 1):
            raw = await request_once(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            if raw is not None: break
            if attempt < transport_retries:
                transport_retries_used += 1
                await asyncio.sleep(min(2 ** attempt, 4))
        if raw is None:
            for i in changed_indices: details[i].update(error="transport_failure", reason="transport_failure", fallback="transport_failure")
            return
        try:
            parsed = parse_unit_verification_response(raw, batch) if semantic_units else parse_verification_response(raw, batch)  # type: ignore[arg-type]
        except ValueError as exc:
            schema_failures += 1
            if len(batch) > 1 and schema_retries > 0:
                schema_split_retry = True
                await asyncio.gather(*(verify_batch([unit]) for unit in batch))
                return
            for i in changed_indices: details[i].update(error=str(exc), reason="schema_failure", fallback="schema_failure")
            return
        successful += 1
        if semantic_units:
            for unit in batch:  # type: ignore[union-attr]
                row = parsed[unit.unit_id]  # type: ignore[index]
                for i in unit.changed_indices:
                    details[i].update(verdict=row["verdict"], issues=row["issues"], severity=row["severity"], explanation=row["explanation"])
                    if row["verdict"] != "pass": details[i].update(reason="verdict_" + row["verdict"], fallback="verdict_" + row["verdict"])
        else:
            for i, row in parsed.items():
                details[i].update(verdict=row["verdict"], issues=row["issues"], severity=row["severity"], explanation=row["explanation"])
                if row["verdict"] != "pass": details[i].update(reason="verdict_" + row["verdict"], fallback="verdict_" + row["verdict"])
    if semantic_units:
        await asyncio.gather(*(verify_batch(list(units[start:start + batch_size])) for start in range(0, len(units), batch_size)))
    else:
        await asyncio.gather(*(verify_batch(selected[start:start + batch_size]) for start in range(0, len(selected), batch_size)))
    for item in output:
        index = int(item["index"])
        detail = details.get(index)
        if detail and detail["fallback"] and detail["fallback"] != "not_selected":
            _restore_original_text(item, by_index[index])
            detail["output"] = join_text_lines(item.get("text", ""))
    verified = [i for i in selected if details[i]["fallback"] is None]
    unresolved = [i for i in changed if details[i]["fallback"] is not None and details[i]["fallback"] != "not_selected"]
    if semantic_units:
        unit_rows: List[Dict[str, Any]] = []
        for unit in units:
            if unit.changed_indices:
                first = details[unit.changed_indices[0]]
                unit_verdict = first["verdict"]
                unit_error = first["error"]
                unit_reason = first["reason"]
                unit_fallback = first["fallback"]
                unit_issues = first["issues"]
                unit_severity = first["severity"]
                unit_explanation = first["explanation"]
            else:
                unit_verdict = unit_error = unit_reason = unit_fallback = unit_issues = unit_severity = unit_explanation = None
            for index in unit.changed_indices:
                details[index]["unit_id"] = unit.unit_id
                details[index]["unit_indices"] = list(unit.cue_indices)
            unit_rows.append({"unit_id": unit.unit_id, "cue_indices": list(unit.cue_indices), "changed_indices": list(unit.changed_indices),
                              "verdict": unit_verdict, "issues": unit_issues, "severity": unit_severity, "explanation": unit_explanation,
                              "error": unit_error, "reason": unit_reason, "fallback": unit_fallback})
        report_units = unit_rows
    report = {"schema_version": 1, "backend": backend, "accounting": accounting, "model": model, "total_count": len(originals), "changed_count": len(changed), "selected_count": len(selected), "selected_indices": selected, "verified_indices": verified, "unresolved_indices": unresolved, "fallback_indices": unresolved, "status": "completed" if not unresolved else "completed_with_unresolved", "context_mode": "full" if context_items is not None else "sparse", "targets": [details[i] for i in changed], "requests": requests, "transport_retries": transport_retries_used, "schema_split_retry": schema_split_retry, "failures": transport_failures + schema_failures}
    if semantic_units:
        report["semantic_units_enabled"] = True
        report["units"] = report_units
    usage = {"backend": backend, "accounting": accounting, "model": model, "requests": requests, "transport_retries": transport_retries_used, "schema_split_retry": schema_split_retry, "successful_responses": successful, "transport_failures": transport_failures, "schema_failures": schema_failures, "token_usage_available": token_usage_response_count > 0}
    if token_usage_response_count:
        usage.update(usage_totals, provider_reported_cost_usd=reported_cost, provider_cost_is_billing_authoritative=cost_is_billing_authoritative, usage_response_count=usage_response_count, provider_ids=sorted(provider_ids), model_ids=sorted(model_ids), finishes=sorted(finishes))
    return output, report, usage


async def _request_live(prompt: str, model: str, base_url: str, auth_header: Optional[Dict[str, str]] = None,
                        structured_output: bool = True) -> Optional[OpenCodePromptResult]:
    if not base_url.strip():
        raise ValueError("base_url_required")
    session = await create_session(base_url, title="independent-verification", auth_header=auth_header)
    try:
        system_prompt = INDEPENDENT_VERIFIER_SYSTEM_PROMPT
        output_schema: Optional[Dict[str, Any]] = None
        is_unit_payload = False
        try:
            payload = json.loads(prompt)
            targets = payload.get("targets") if isinstance(payload, dict) else None
            first_target = targets[0] if isinstance(targets, list) and targets else None
            if isinstance(first_target, dict) and "cue_indices" in first_target and "original_cues" in first_target:
                system_prompt = UNIT_VERIFIER_SYSTEM_PROMPT
                is_unit_payload = True
                if structured_output:
                    output_schema = unit_verification_schema()
        except (TypeError, json.JSONDecodeError):
            pass
        if not is_unit_payload:
            return await send_prompt_result(base_url, session, system_prompt, prompt, model, auth_header=auth_header)
        if output_schema is None:
            return await send_prompt_result(base_url, session, system_prompt, prompt, model, auth_header=auth_header,
                                            output_schema=None)
        try:
            return await send_prompt_result(base_url, session, system_prompt, prompt, model, auth_header=auth_header, output_schema=output_schema)
        except TypeError as exc:
            # Keep compatibility with injected legacy request doubles that predate output_schema.
            if "unexpected keyword argument 'output_schema'" not in str(exc):
                raise
            return await send_prompt_result(base_url, session, system_prompt, prompt, model, auth_header=auth_header)
    finally:
        await delete_session(base_url, session, auth_header=auth_header)


def verify_items_live(
    original: List[Dict[str, Any]], candidate: List[Dict[str, Any]], *,
    model: str = "openai/gpt-5.6-sol", context_items: Optional[List[Dict[str, Any]]] = None,
    context_window: int = 3, batch_size: int = 4, concurrency: int = 3,
    transport_retries: int = 1, schema_retries: int = 1,
    review_report: Optional[Dict[str, Any]] = None, server_url: Optional[str] = None,
    hostname: str = "127.0.0.1", port: Optional[int] = None,
    semantic_units: bool = False, semantic_unit_max_cues: int = 3, semantic_unit_max_gap_sec: float = 0.3,
    structured_output: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Run the live verifier while owning the optional local server lifecycle."""
    _validate_runtime_args(model, context_window, batch_size, concurrency, transport_retries, schema_retries, semantic_unit_max_cues, semantic_unit_max_gap_sec)
    _validate_server_config(server_url, hostname, port)
    server: Optional[OpenCodeServer] = None
    try:
        if server_url is not None:
            base_url = server_url.rstrip("/")
            auth_header = _get_auth_header()
        else:
            server = OpenCodeServer(port, hostname)
            server.start()
            base_url = server.base_url
            auth_header = None

        async def live(prompt: str, requested_model: str) -> Optional[OpenCodePromptResult]:
            return await _request_live(prompt, requested_model, base_url, auth_header, structured_output)

        return asyncio.run(verify_items(
            original, candidate, model=model, context_items=context_items,
            context_window=context_window, batch_size=batch_size, concurrency=concurrency,
            transport_retries=transport_retries, schema_retries=schema_retries,
            review_report=review_report, request_callable=live,
            semantic_units=semantic_units, semantic_unit_max_cues=semantic_unit_max_cues, semantic_unit_max_gap_sec=semantic_unit_max_gap_sec,
        ))
    finally:
        if server is not None:
            server.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed independent subtitle verifier.")
    parser.add_argument("original"); parser.add_argument("candidate")
    parser.add_argument("--review-report"); parser.add_argument("--context-source"); parser.add_argument("--model", default="openai/gpt-5.6-sol"); parser.add_argument("--server-url"); parser.add_argument("--hostname", default="127.0.0.1"); parser.add_argument("--port", type=int); parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--concurrency", type=int, default=3); parser.add_argument("--context-window", type=int, default=3); parser.add_argument("--transport-retries", type=int, default=1, help="Number of extra transport attempts."); parser.add_argument("--schema-retries", type=int, choices=[0, 1], default=1, help="Split one malformed multi-item batch into singleton requests once (0 or 1). "); parser.add_argument("--semantic-units", action="store_true"); parser.add_argument("--semantic-unit-max-cues", type=int, default=3); parser.add_argument("--semantic-unit-max-gap-sec", type=float, default=0.3); parser.add_argument("--structured-output", action=argparse.BooleanOptionalAction, default=False); parser.add_argument("--output-dir", default="output/semantic_verified")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_runtime_args(args.model, args.context_window, args.batch_size, args.concurrency, args.transport_retries, args.schema_retries, args.semantic_unit_max_cues, args.semantic_unit_max_gap_sec)
        _validate_server_config(args.server_url, args.hostname, args.port)
    except ValueError as exc:
        print(str(exc), file=sys.stderr); return 2
    try:
        original = load_json(Path(args.original)); candidate = load_json(Path(args.candidate)); context = load_json(Path(args.context_source)) if args.context_source else None
        report = load_json(Path(args.review_report)) if args.review_report else None
        _validate_items(original, candidate)
        if context is not None: validate_context_source(original, context)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(str(exc), file=sys.stderr); return 2
    output, verification, usage = verify_items_live(
            original, candidate, model=args.model, context_items=context,
            context_window=args.context_window, batch_size=args.batch_size,
            concurrency=args.concurrency, transport_retries=args.transport_retries,
            schema_retries=args.schema_retries, review_report=report,
             server_url=args.server_url, hostname=args.hostname, port=args.port,
             semantic_units=args.semantic_units, semantic_unit_max_cues=args.semantic_unit_max_cues, semantic_unit_max_gap_sec=args.semantic_unit_max_gap_sec,
             structured_output=args.structured_output,
     )
    destination = Path(args.output_dir); destination.mkdir(parents=True, exist_ok=True); output_path = destination / (Path(args.candidate).stem + "_independent_verified.json")
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / (Path(args.candidate).stem + "_independent_verified.report.json")).write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / (Path(args.candidate).stem + "_independent_verified.usage.json")).write_text(json.dumps(usage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
