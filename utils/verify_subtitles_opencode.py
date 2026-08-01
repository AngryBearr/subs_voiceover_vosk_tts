"""Standalone, fail-closed semantic verification through local OpenCode OAuth."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from utils.analyze_text import join_text_lines
from utils.opencode_transport import OpenCodeServer, _get_auth_header, create_session, delete_session, send_prompt
from utils.shorten_helpers import load_json, validate_context_source
from utils.semantic_requirements import PLANNER_ISSUE_CODES, extract_semantic_requirements

ISSUE_CODES = PLANNER_ISSUE_CODES
VERDICTS = frozenset({"pass", "fail", "uncertain"})
SEVERITIES = frozenset({"none", "minor", "major"})
RequestFunc = Callable[[str, str], Awaitable[Optional[str]]]

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


def _validate_runtime_args(model: str, context_window: int, batch_size: int, concurrency: int, transport_retries: int, schema_retries: int) -> None:
    if not model or context_window < 0 or not 1 <= batch_size <= 100 or not 1 <= concurrency <= 100 or not 0 <= transport_retries <= 5 or schema_retries not in {0, 1}:
        raise ValueError("invalid_arguments")


async def verify_items(
    original: List[Dict[str, Any]], candidate: List[Dict[str, Any]], *, model: str = "openai/gpt-5.6-luna", context_items: Optional[List[Dict[str, Any]]] = None,
    context_window: int = 3, batch_size: int = 4, concurrency: int = 3, transport_retries: int = 1, schema_retries: int = 1,
    review_report: Optional[Dict[str, Any]] = None, request_callable: Optional[RequestFunc] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    _validate_runtime_args(model, context_window, batch_size, concurrency, transport_retries, schema_retries)
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
    requests = transport_retries_used = transport_failures = schema_failures = successful = 0
    schema_split_retry = False
    sem = asyncio.Semaphore(concurrency)
    context_by_index = {int(x["index"]): x for x in (context_items or originals)}
    ordered_context = list(context_by_index.values())

    async def request_once(prompt: str) -> Optional[str]:
        nonlocal requests, transport_failures
        async with sem:
            requests += 1
            try:
                response = await request_callable(prompt, model)
                if response is None or not response.strip():
                    transport_failures += 1
                    return None
                return response
            except Exception:
                transport_failures += 1
                return None

    async def verify_batch(indices: List[int]) -> None:
        nonlocal transport_retries_used, schema_failures, successful, schema_split_retry
        payload = {"targets": []}
        positions = {int(item["index"]): pos for pos, item in enumerate(ordered_context)}
        for i in indices:
            pos = positions[i]
            payload["targets"].append(_target_payload(i, by_index, candidate_by_index, ordered_context[max(0, pos-context_window):pos+context_window+1]))
        raw: Optional[str] = None
        for attempt in range(transport_retries + 1):
            raw = await request_once(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            if raw is not None: break
            if attempt < transport_retries:
                transport_retries_used += 1
                await asyncio.sleep(min(2 ** attempt, 4))
        if raw is None:
            for i in indices: details[i].update(error="transport_failure", reason="transport_failure", fallback="transport_failure")
            return
        try:
            parsed = parse_verification_response(raw, indices)
        except ValueError as exc:
            schema_failures += 1
            if len(indices) > 1 and schema_retries > 0:
                schema_split_retry = True
                await asyncio.gather(*(verify_batch([i]) for i in indices))
                return
            for i in indices: details[i].update(error=str(exc), reason="schema_failure", fallback="schema_failure")
            return
        successful += 1
        for i, row in parsed.items():
            details[i].update(verdict=row["verdict"], issues=row["issues"], severity=row["severity"], explanation=row["explanation"])
            if row["verdict"] != "pass": details[i].update(reason="verdict_" + row["verdict"], fallback="verdict_" + row["verdict"])
    await asyncio.gather(*(verify_batch(selected[start:start + batch_size]) for start in range(0, len(selected), batch_size)))
    for item in output:
        index = int(item["index"])
        detail = details.get(index)
        if detail and detail["fallback"] and detail["fallback"] != "not_selected":
            _restore_original_text(item, by_index[index])
            detail["output"] = join_text_lines(item.get("text", ""))
    verified = [i for i in selected if details[i]["fallback"] is None]
    unresolved = [i for i in changed if details[i]["fallback"] is not None and details[i]["fallback"] != "not_selected"]
    report = {"schema_version": 1, "backend": "opencode", "accounting": "subscription", "model": model, "total_count": len(originals), "changed_count": len(changed), "selected_count": len(selected), "selected_indices": selected, "verified_indices": verified, "unresolved_indices": unresolved, "fallback_indices": unresolved, "status": "completed" if not unresolved else "completed_with_unresolved", "context_mode": "full" if context_items is not None else "sparse", "targets": [details[i] for i in changed], "requests": requests, "transport_retries": transport_retries_used, "schema_split_retry": schema_split_retry, "failures": transport_failures + schema_failures}
    usage = {"backend": "opencode", "accounting": "subscription", "model": model, "requests": requests, "transport_retries": transport_retries_used, "schema_split_retry": schema_split_retry, "successful_responses": successful, "transport_failures": transport_failures, "schema_failures": schema_failures, "token_usage_available": False}
    return output, report, usage


async def _request_live(prompt: str, model: str, base_url: str, auth_header: Optional[Dict[str, str]] = None) -> Optional[str]:
    if not base_url.strip():
        raise ValueError("base_url_required")
    session = await create_session(base_url, title="independent-verification", auth_header=auth_header)
    try:
        return await send_prompt(base_url, session, INDEPENDENT_VERIFIER_SYSTEM_PROMPT, prompt, model, auth_header=auth_header)
    finally:
        await delete_session(base_url, session, auth_header=auth_header)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed independent subtitle verifier.")
    parser.add_argument("original"); parser.add_argument("candidate")
    parser.add_argument("--review-report"); parser.add_argument("--context-source"); parser.add_argument("--model", default="openai/gpt-5.6-luna"); parser.add_argument("--server-url"); parser.add_argument("--hostname", default="127.0.0.1"); parser.add_argument("--port", type=int); parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--concurrency", type=int, default=3); parser.add_argument("--context-window", type=int, default=3); parser.add_argument("--transport-retries", type=int, default=1, help="Number of extra transport attempts."); parser.add_argument("--schema-retries", type=int, choices=[0, 1], default=1, help="Split one malformed multi-item batch into singleton requests once (0 or 1). "); parser.add_argument("--output-dir", default="output/semantic_verified")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_runtime_args(args.model, args.context_window, args.batch_size, args.concurrency, args.transport_retries, args.schema_retries)
    except ValueError as exc:
        print(str(exc), file=sys.stderr); return 2
    if not args.server_url and args.hostname not in {"127.0.0.1", "localhost"}:
        print("auto-start hostname must be localhost or 127.0.0.1", file=sys.stderr); return 2
    if args.port is not None and (args.port <= 0 or args.port > 65535):
        print("port must be between 1 and 65535", file=sys.stderr); return 2
    try:
        original = load_json(Path(args.original)); candidate = load_json(Path(args.candidate)); context = load_json(Path(args.context_source)) if args.context_source else None
        report = load_json(Path(args.review_report)) if args.review_report else None
        _validate_items(original, candidate)
        if context is not None: validate_context_source(original, context)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(str(exc), file=sys.stderr); return 2
    server: Optional[OpenCodeServer] = None
    try:
        if args.server_url:
            base_url = args.server_url.rstrip("/")
        else:
            server = OpenCodeServer(args.port, args.hostname); server.start(); base_url = server.base_url
        auth_header = _get_auth_header() if args.server_url else None
        async def live(prompt: str, model: str) -> Optional[str]:
            return await _request_live(prompt, model, base_url, auth_header)
        output, verification, usage = asyncio.run(verify_items(original, candidate, model=args.model, context_items=context, context_window=args.context_window, batch_size=args.batch_size, concurrency=args.concurrency, transport_retries=args.transport_retries, schema_retries=args.schema_retries, review_report=report, request_callable=live))
        destination = Path(args.output_dir); destination.mkdir(parents=True, exist_ok=True); output_path = destination / (Path(args.candidate).stem + "_independent_verified.json")
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (destination / (Path(args.candidate).stem + "_independent_verified.report.json")).write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (destination / (Path(args.candidate).stem + "_independent_verified.usage.json")).write_text(json.dumps(usage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        if server is not None:
            server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
