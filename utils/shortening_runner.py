"""Pure, accountable batching and arbitration loop for subtitle shortening."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from utils.json_to_srt import json_to_srt
from utils.llm_types import CallRecord, DetailedResponse
from utils.length_check import check_length
from utils.shortening_costs import estimate_record_cost, project_500, summarize_costs
from utils.shortening_plan import (
    CuePlan,
    PlannerConfig,
    _severity,
    analysis_for_text,
    context_items,
    make_batches,
)
from utils.shortening_prompts import (
    ARBITER_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    PartialParseResult,
    PROMPT_VERSION,
    build_generation_prompt,
    build_review_prompt,
    generation_schema,
    parse_generation_partial,
    parse_review_partial,
    review_schema,
)
from utils.text_metrics import classify_text


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    variant: str


@dataclass(frozen=True)
class RunConfig:
    gen_batch_size: int = 6
    review_batch_size: int = 12
    max_repairs: int = 1
    max_context_chars: int = 20000
    length_tools: bool = False

    def __post_init__(self) -> None:
        for name in ("gen_batch_size", "review_batch_size", "max_context_chars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.max_repairs, bool) or not isinstance(self.max_repairs, int) or not 0 <= self.max_repairs <= 2:
            raise ValueError("max_repairs must be between 0 and 2")
        if not isinstance(self.length_tools, bool):
            raise TypeError("length_tools must be bool")


RequestFunction = Callable[[ModelSpec, str, str, dict[str, Any], bool], Awaitable[DetailedResponse]]
OriginalTimingVerifier = Callable[[list[CuePlan], Path], dict[str, dict[str, Any]]]


def _json_default(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return str(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(handle)
    temporary = Path(raw)
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    handle, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(handle)
    temporary = Path(raw)
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=_json_default, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _safe_error(response: DetailedResponse | None, fallback: str) -> str:
    if response is None or response.error is None:
        return fallback
    return type(response.error).__name__ if not isinstance(response.error, str) else response.error[:120]


def _record_call(call: CallRecord, *, stage: str, target_ids: list[str], model: ModelSpec,
                 output_dir: Path, request_id: str) -> None:
    payload = {
        "request_id": request_id,
        "stage": stage,
        "target_ids": list(target_ids),
        "model": asdict(model),
        "provider": call.provider,
        "record": asdict(call),
        "cost_estimate": estimate_record_cost(call),
    }
    _append_jsonl(output_dir / "calls.jsonl", payload)


def _source_hash(items: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _target_context(items: list[dict[str, Any]], plans: list[CuePlan], ids: list[str], planner: PlannerConfig) -> list[dict[str, Any]]:
    return context_items(items, plans, ids, planner)


def _plan_payload(plan: CuePlan) -> dict[str, Any]:
    return asdict(plan)


def _local_check(candidate: str, plan: CuePlan) -> dict[str, Any]:
    length = check_length(candidate, plan.max_chars)
    valid = bool(candidate.strip()) and "\n" not in candidate and "\r" not in candidate and candidate == candidate.strip()
    return {
        "nonempty": bool(candidate.strip()),
        "single_line": "\n" not in candidate and "\r" not in candidate,
        "trimmed": candidate == candidate.strip(),
        "actual_chars": length["length"],
        "max_chars": length["max_chars"],
        "within_length": length["fits"],
        "valid": valid,
    }


def _new_record(item: dict[str, Any], plan: CuePlan) -> dict[str, Any]:
    return {
        "cue_id": plan.cue_id,
        "position": plan.position,
        "source_index": item.get("index"),
        "original": plan.text,
        "candidate": None,
        "final_text": plan.text,
        "status": "short_timing" if plan.status == "short_timing" else "pending",
        "budget": _plan_payload(plan),
        "forecast": _plan_payload(plan),
        "source_analysis": copy.deepcopy(item.get("analysis")) if isinstance(item.get("analysis"), dict) else copy.deepcopy(item.get("analysis")),
        "history": [],
        "final_review": None,
        "initial_accepted": False,
    }


def _reason_for(plan: CuePlan, candidate: str, local: dict[str, Any], verdict: str | None, cannot: bool = False) -> str:
    if cannot:
        return "cannot_shorten"
    if not local["valid"]:
        return "schema"
    if not local["within_length"]:
        return "timing"
    if verdict in {"revise", "impossible"}:
        return "semantic"
    return "semantic"


def _request_log(output_dir: Path, request_id: str, stage: str, model: ModelSpec,
                 target_ids: list[str], system: str, user: str, schema: dict[str, Any],
                 state: str, text: str | None = None, error: str | None = None,
                 started_at: str | None = None, elapsed_sec: float | None = None) -> None:
    # Prompts are saved for reproduction, but credentials and model reasoning are
    # never put into this record; the response field contains only final text.
    payload: dict[str, Any] = {
        "request_id": request_id, "stage": stage, "model": asdict(model),
        "target_ids": list(target_ids), "state": state,
    }
    if state == "start":
        payload.update({"system_prompt": system, "user_prompt": user, "output_schema": schema})
    else:
        payload.update({"text": text, "error": error})
    if started_at is not None:
        payload["started_at"] = started_at
    if elapsed_sec is not None:
        payload["elapsed_sec"] = elapsed_sec
    _append_jsonl(output_dir / "requests.jsonl", payload)


async def _request(request_fn: RequestFunction, model: ModelSpec, system: str, user: str,
                   schema: dict[str, Any], allow_tool: bool, *, stage: str,
                   ids: list[str], output_dir: Path, stage_calls: dict[str, list[CallRecord]],
                   request_counts: dict[str, int], request_failures: list[dict[str, Any]]) -> DetailedResponse:
    request_id = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    request_counts[stage] = request_counts.get(stage, 0) + 1
    _request_log(output_dir, request_id, stage, model, ids, system, user, schema, "start",
                 started_at=started_at, elapsed_sec=0.0)
    caught_failure = False
    try:
        response = await request_fn(model, system, user, schema, allow_tool)
    except (Exception, asyncio.CancelledError) as exc:
        caught_failure = True
        error = type(exc).__name__
        request_failures.append({"request_id": request_id, "stage": stage, "target_ids": list(ids),
                                 "model": asdict(model), "error": error, "started_at": started_at,
                                 "accounting_incomplete": True})
        response = DetailedResponse(None, [], error)
    if not caught_failure and (response.error is not None or response.text is None):
        response_accounting_complete = bool(response.calls) and bool(
            summarize_costs(response.calls).get("accounting_complete", False)
        )
        request_failures.append({"request_id": request_id, "stage": stage, "target_ids": list(ids),
                                 "model": asdict(model), "error": _safe_error(response, "transport"),
                                 "started_at": started_at,
                                 "accounting_incomplete": not response_accounting_complete})
    for call in response.calls:
        _record_call(call, stage=stage, target_ids=ids, model=model, output_dir=output_dir, request_id=request_id)
        stage_calls.setdefault(stage, []).append(call)
    elapsed = time.monotonic() - started
    if request_failures and request_failures[-1].get("request_id") == request_id:
        request_failures[-1]["elapsed_sec"] = elapsed
    _request_log(output_dir, request_id, stage, model, ids, system, user, schema, "complete",
                 response.text if response.error is None else None, response.error,
                 started_at=started_at, elapsed_sec=elapsed)
    return response


def _status_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def _apply_text(item: dict[str, Any], text: str) -> None:
    if isinstance(item.get("text"), list):
        item["text"] = [text]
    else:
        item["text"] = text


def _export_srt(items: list[dict[str, Any]], output_dir: Path) -> str | None:
    chronological = sorted(enumerate(items), key=lambda pair: (pair[1].get("start", 0) if isinstance(pair[1].get("start"), int) else 0, pair[0]))
    canonical: list[dict[str, Any]] = []
    for number, (_, item) in enumerate(chronological, 1):
        entry = copy.deepcopy(item)
        entry["index"] = number
        canonical.append(entry)
    handle, raw = tempfile.mkstemp(suffix=".json", dir=str(output_dir))
    os.close(handle)
    source = Path(raw)
    try:
        source.write_text(json.dumps(canonical, ensure_ascii=False), encoding="utf-8")
        result = json_to_srt(str(source))
        assert isinstance(result, str)
        _atomic_text(output_dir / "subtitles.srt", result)
        return None
    except Exception as exc:
        return type(exc).__name__
    finally:
        source.unlink(missing_ok=True)


def _build_report(items: list[dict[str, Any]], plans: list[CuePlan], target_ids: list[str],
                  planner: PlannerConfig, run_config: RunConfig, generator: ModelSpec,
                  arbiter: ModelSpec, output_dir: Path, records: list[dict[str, Any]],
                  stage_calls: dict[str, list[CallRecord]], source_path: str | None,
                   source_hash: str, export_error: str | None = None, selected_initial: dict[str, Any] | None = None,
                   *, run_state: str = "running", request_counts: dict[str, int] | None = None,
                   request_failures: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    accepted = sum(record["status"] == "accepted" for record in records)
    actual_fits = sum(record["status"] == "actual_fits" for record in records)
    manual = len(records) - accepted - actual_fits
    costs = {stage: summarize_costs(stage_calls.get(stage, []))
             for stage in ("generation", "review", "repair", "recheck")}
    all_calls = [call for calls in stage_calls.values() for call in calls]
    direct = summarize_costs(all_calls)
    repairs = sum(1 for record in records for event in record["history"] if event.get("stage") == "repair")
    # A failed response is recorded even when the provider supplied a usage-bearing
    # call record.  In that case the known ledger remains usable.  Failure entries
    # without an explicit accounting flag are incomplete for safety, and each
    # request is evaluated independently rather than against the run-wide ledger.
    accounting_complete = bool(direct.get("accounting_complete", True)) and not any(
        failure.get("accounting_incomplete", True) for failure in (request_failures or [])
    )
    if not accounting_complete:
        direct["accounting_complete"] = False
        direct["known_subtotal_is_partial"] = True
    observed_counts = dict(request_counts or {})
    try:
        projection = project_500(
            selected_count=len(records),
            tokens={key: direct.get(key) for key in ("input_tokens", "output_tokens", "reasoning_tokens")},
            known_cost_usd=direct["known_subtotal_usd"],
            request_counts=observed_counts,
            observed_repair_fraction=repairs / len(records) if records else 0.0,
            accounting_complete=accounting_complete,
            gen_batch_size=run_config.gen_batch_size,
            review_batch_size=run_config.review_batch_size,
        ) if records else {"unknown_cost": True, "basis": "no selected cues"}
    except TypeError as exc:
        # Keep compatibility while the companion accounting module is upgraded.
        if "unexpected keyword" not in str(exc):
            raise
        projection = project_500(
            selected_count=len(records),
            tokens={key: direct.get(key) for key in ("input_tokens", "output_tokens", "reasoning_tokens")},
            known_cost_usd=direct["known_subtotal_usd"], request_counts=observed_counts,
            observed_repair_fraction=repairs / len(records) if records else 0.0)
    selected_nonshort_count = sum(
        record.get("budget", {}).get("status") != "short_timing"
        for record in records
        if isinstance(record.get("budget"), dict)
    )
    has_valid_final_review = any(
        isinstance(record.get("final_review"), dict)
        and record["final_review"].get("verdict") in {"accept", "revise", "impossible"}
        for record in records
    )
    has_model_or_protocol_errors = any(
        record.get("status") == "error" or record.get("failure") == "protocol_misuse"
        for record in records
    )
    inapplicable_reasons: list[str] = []
    if run_state != "completed":
        inapplicable_reasons.append("incomplete_run")
    if has_model_or_protocol_errors:
        inapplicable_reasons.append("model_or_protocol_errors")
    if not accounting_complete:
        inapplicable_reasons.append("missing_usage")
    if selected_nonshort_count <= 0 or not has_valid_final_review:
        inapplicable_reasons.append("no_reviewed_candidates")
    projection["applicable"] = not inapplicable_reasons
    projection["inapplicable_reasons"] = inapplicable_reasons
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_state": run_state,
        "config": {"planner": asdict(planner), "run": asdict(run_config)},
        "model": {"generator": asdict(generator), "arbiter": asdict(arbiter)},
        "prompt_version": PROMPT_VERSION,
        "selected_ids": list(target_ids),
        "source_manifest": {"path": source_path, "absolute_path": os.path.abspath(source_path) if source_path else None, "sha256": source_hash},
        "records": records,
        "summary": {
            "selected_count": len(records), "accepted_count": accepted,
            "actual_fits_count": actual_fits, "resolved_count": accepted + actual_fits,
            "manual_count": manual,
            # Keep every originally selected cue in the denominator, including measured fits.
            "manual_fraction": manual / len(records) if records else 0.0,
            "within_manual_threshold": (manual / len(records) if records else 0.0) <= 0.15,
            "initial_accepted_count": sum(record["initial_accepted"] for record in records),
            "short_timing_count": sum(record["budget"]["status"] == "short_timing" for record in records),
            "error_count": sum(record["status"] == "error" for record in records),
            "invalid_input_count": sum(plan.status == "invalid" for plan in plans),
            "invalid_input_positions": [plan.position for plan in plans if plan.status == "invalid"],
            "status_counts": _status_counts(records),
            "repairs": repairs,
            "export_error": export_error,
            "accounting_complete": accounting_complete,
            "request_counts": observed_counts,
            "request_failures": copy.deepcopy(request_failures or []),
        },
        "srt_renumbering": "canonical chronological numbering is used only for SRT; JSON source indices are preserved",
        "cost": {"stages": costs, "direct": direct, "projection_500": projection,
                 "coverage_note": "OpenCode hidden title/compaction steps may be outside this logical-session ledger; API-equivalent usage is informational, not an invoice."},
    }
    if selected_initial is not None:
        report["selection"] = selected_initial
    return report


def _checkpoint(output_dir: Path, report: dict[str, Any], subtitles: list[dict[str, Any]]) -> None:
    _atomic_json(output_dir / "report.json", report)
    _atomic_json(output_dir / "subtitles.json", subtitles)
    report["summary"]["export_error"] = _export_srt(subtitles, output_dir)
    _atomic_json(output_dir / "report.json", report)


def _prepare_checkpoint(source: list[dict[str, Any]], records: list[dict[str, Any]],
                        plans: list[CuePlan], planner: PlannerConfig) -> list[dict[str, Any]]:
    """Rebuild a checkpoint from immutable source rows and accepted records."""
    subtitles = copy.deepcopy(source)
    record_by_position = {record["position"]: record for record in records}
    for record in records:
        if record.get("status") == "accepted":
            _apply_text(subtitles[record["position"]], record["final_text"])
    for plan in plans:
        if plan.position >= len(subtitles) or not isinstance(subtitles[plan.position], dict):
            continue
        if plan.status == "invalid":
            continue
        item = subtitles[plan.position]
        text_value = item.get("text")
        text = text_value if isinstance(text_value, str) else " ".join(text_value) if isinstance(text_value, list) and all(isinstance(part, str) for part in text_value) else None
        if text is None:
            continue
        analysis = copy.deepcopy(item.get("analysis")) if isinstance(item.get("analysis"), dict) else {}
        analysis.update(analysis_for_text(plan, text, planner))
        record = record_by_position.get(plan.position)
        if record is not None and record.get("timing_evidence") is not None:
            analysis["timing_evidence"] = copy.deepcopy(record["timing_evidence"])
        item["analysis"] = analysis
    return subtitles


def _forecast_plan(plan: CuePlan) -> CuePlan:
    """Derive risk metadata for plans constructed by older callers."""
    classification = classify_text(plan.text, plan.max_chars) if plan.status != "invalid" else None
    if classification is None:
        return plan
    return replace(
        plan,
        status=classification.status,
        severity=_severity(plan.reduction_fraction, classification.status),
        words_count=classification.words_count,
        repetition_candidate=classification.repetition_candidate,
        short_timing_risk=classification.short_timing_risk,
        reason=classification.reason,
    )


def _preflight_error_record(record: dict[str, Any], error_type: str) -> None:
    record["status"] = "timing_unverified"
    record["final_text"] = record["original"]
    existing = record.get("timing_evidence")
    if not isinstance(existing, dict) or not isinstance(existing.get("error"), (dict, str)):
        record["timing_evidence"] = {"error": {"type": error_type}}


def _validate_timing_evidence(
    plan: CuePlan, evidence: Any
) -> tuple[bool, bool, dict[str, Any]]:
    """Return (valid, fits, evidence) while never trusting a supplied fits flag."""
    if not isinstance(evidence, dict):
        return False, False, {"error": {"type": "MalformedEvidence"}}
    if evidence.get("error") is not None:
        return False, False, copy.deepcopy(evidence)
    if evidence.get("error_type") is not None or evidence.get("status") == "error":
        return False, False, copy.deepcopy(evidence)
    if "cue_id" in evidence and evidence.get("cue_id") != plan.cue_id:
        return False, False, {"error": {"type": "CueIdMismatch"}}
    if evidence.get("original") != plan.text:
        return False, False, {"error": {"type": "OriginalMismatch"}}
    if "text" in evidence and evidence.get("text") != plan.text:
        return False, False, {"error": {"type": "TextMismatch"}}
    available = evidence.get("available_ms")
    if isinstance(available, bool) or not isinstance(available, (int, float)):
        return False, False, {"error": {"type": "WindowMismatch"}}
    try:
        available_float = float(available)
    except (OverflowError, TypeError, ValueError):
        return False, False, {"error": {"type": "WindowMismatch"}}
    if not math.isfinite(available_float) or available_float != plan.available_ms:
        return False, False, {"error": {"type": "WindowMismatch"}}
    duration = evidence.get("duration_sec")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        return False, False, {"error": {"type": "InvalidDuration"}}
    try:
        duration_float = float(duration)
    except (OverflowError, TypeError, ValueError):
        return False, False, {"error": {"type": "InvalidDuration"}}
    if not math.isfinite(duration_float) or duration_float <= 0:
        return False, False, {"error": {"type": "InvalidDuration"}}
    return True, duration_float <= available / 1000.0, copy.deepcopy(evidence)


def _apply_preflight(
    selected: list[CuePlan],
    record_by_id: dict[str, dict[str, Any]],
    verifier: OriginalTimingVerifier,
    output_dir: Path,
) -> tuple[dict[str, CuePlan], set[str]]:
    risk_plans = [plan for plan in selected if plan.short_timing_risk]
    effective = {plan.cue_id: plan for plan in selected}
    risk_ids = {plan.cue_id for plan in risk_plans}
    if not risk_plans:
        return effective, set()
    try:
        evidence_map = verifier(risk_plans, output_dir)
    except Exception as exc:
        for plan in risk_plans:
            _preflight_error_record(record_by_id[plan.cue_id], type(exc).__name__)
        return effective, risk_ids
    if not isinstance(evidence_map, dict):
        for plan in risk_plans:
            _preflight_error_record(record_by_id[plan.cue_id], "MalformedEvidenceMap")
        return effective, risk_ids
    unknown = set(evidence_map) - risk_ids
    missing = risk_ids - set(evidence_map)
    for plan in risk_plans:
        record = record_by_id[plan.cue_id]
        if unknown or plan.cue_id in missing:
            _preflight_error_record(record, "EvidenceKeyMismatch")
            continue
        valid, fits, evidence = _validate_timing_evidence(plan, evidence_map[plan.cue_id])
        record["timing_evidence"] = evidence
        if not valid:
            error = evidence.get("error")
            error_type = (
                error.get("type")
                if isinstance(error, dict)
                else error
                if isinstance(error, str)
                else evidence.get("error_type")
            )
            _preflight_error_record(record, str(error_type or "InvalidEvidence"))
            continue
        record["timing_evidence"] = evidence
        if fits:
            record["status"] = "actual_fits"
            record["candidate"] = plan.text
            record["final_text"] = plan.text
            actual_plan = replace(
                plan,
                status="fits",
                severity="fits",
                reduction_fraction=0.0,
                short_timing_risk=False,
                reason="actual_fit",
                estimate_source="vosk_measurement",
            )
            effective[plan.cue_id] = actual_plan
            record["budget"] = _plan_payload(actual_plan)
        elif plan.repetition_candidate:
            # Repeated short phrases remain eligible for contextual shortening.
            effective_plan = replace(plan, status="needs_shortening")
            effective[plan.cue_id] = effective_plan
            record["budget"] = _plan_payload(effective_plan)
        else:
            record["status"] = "short_timing"
    return effective, risk_ids


async def run_trial(*, items: list[dict[str, Any]], plans: list[CuePlan], target_ids: list[str],
                    planner_config: PlannerConfig, run_config: RunConfig, generator: ModelSpec,
                    arbiter: ModelSpec, output_dir: Path, request_fn: RequestFunction,
                    original_timing_verifier: OriginalTimingVerifier | None = None) -> dict[str, Any]:
    """Run one independent trial and write fail-closed checkpoints after each batch."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise TypeError("items must be a list of objects")
    if not isinstance(plans, list) or not all(isinstance(plan, CuePlan) for plan in plans):
        raise TypeError("plans must be CuePlan objects")
    if not isinstance(run_config, RunConfig):
        raise TypeError("run_config must be RunConfig")
    plans = [_forecast_plan(plan) for plan in plans]
    plan_by_id = {plan.cue_id: plan for plan in plans}
    if len(plan_by_id) != len(plans):
        raise ValueError("duplicate plan cue_id")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("target_ids must not contain duplicates")
    if any(target_id not in plan_by_id for target_id in target_ids):
        raise ValueError("unknown target cue_id")
    selected = [plan_by_id[target_id] for target_id in target_ids]
    if any(plan.status not in {"needs_shortening", "short_timing"} for plan in selected):
        raise ValueError("all selected targets must need shortening or have short timing")
    output_dir.mkdir(parents=True)
    records = [_new_record(items[plan.position], plan) for plan in selected]
    record_by_id = {record["cue_id"]: record for record in records}
    source_copy = copy.deepcopy(items)
    stage_calls: dict[str, list[CallRecord]] = {}
    request_counts: dict[str, int] = {stage: 0 for stage in ("generation", "review", "repair", "recheck")}
    request_failures: list[dict[str, Any]] = []
    source_path: str | None = None

    effective_selected = selected
    if original_timing_verifier is not None:
        effective_map, _ = _apply_preflight(selected, record_by_id, original_timing_verifier, output_dir)
        effective_selected = [effective_map[plan.cue_id] for plan in selected]
        effective_by_id = {plan.cue_id: plan for plan in effective_selected}
        plans = [effective_by_id.get(plan.cue_id, plan) for plan in plans]
        plan_by_id = {plan.cue_id: plan for plan in plans}

    def save(run_state: str) -> dict[str, Any]:
        report = _build_report(items, plans, target_ids, planner_config, run_config, generator, arbiter,
                               output_dir, records, stage_calls, source_path, _source_hash(items),
                               run_state=run_state, request_counts=request_counts,
                               request_failures=request_failures)
        _checkpoint(output_dir, report, _prepare_checkpoint(source_copy, records, plans, planner_config))
        return report

    save("running")
    eligible_ids = [
        plan.cue_id
        for plan in effective_selected
        if plan.status == "needs_shortening"
        and record_by_id[plan.cue_id].get("status") not in {"actual_fits", "timing_unverified"}
    ]

    generation_queue: list[str] = []
    review_queue: list[str] = []

    def can_retry(round_index: int | None) -> bool:
        return run_config.max_repairs > 0 if round_index is None else round_index + 1 < run_config.max_repairs

    def queue_unique(queue: list[str], cue_id: str) -> None:
        if cue_id not in queue and record_by_id[cue_id].get("status") not in {"accepted", "actual_fits", "short_timing"}:
            queue.append(cue_id)

    def checkpoint() -> None:
        report = _build_report(items, plans, target_ids, planner_config, run_config, generator, arbiter,
                               output_dir, records, stage_calls, source_path, _source_hash(items),
                               run_state="incomplete" if request_failures else "running",
                               request_counts=request_counts, request_failures=request_failures)
        _checkpoint(output_dir, report, _prepare_checkpoint(source_copy, records, plans, planner_config))

    async def run_generation(
        ids: list[str], *, stage: str, round_index: int | None,
    ) -> list[str]:
        review_ids: list[str] = []
        feedback: dict[str, dict[str, Any]] = {}
        if stage == "repair":
            for cue_id in ids:
                record = record_by_id[cue_id]
                candidate = record.get("candidate")
                feedback[cue_id] = {
                    "candidate": candidate if isinstance(candidate, str) else "",
                    "feedback": json.dumps({
                        "issues": record.get("final_review", {}).get("issues", []) if isinstance(record.get("final_review"), dict) else [],
                        "advice": record.get("final_review", {}).get("advice", "") if isinstance(record.get("final_review"), dict) else "",
                        "overage": max(0, len(candidate) - plan_by_id[cue_id].max_chars) if isinstance(candidate, str) else 0,
                        "failure": {
                            "class": record.get("failure"),
                            **({"error": next((event["error"] for event in reversed(record["history"])
                                               if isinstance(event.get("error"), str)), None)}
                               if any(isinstance(event.get("error"), str) for event in record["history"]) else {}),
                        },
                    }, ensure_ascii=False),
                }
        for batch in make_batches(source_copy, plans, ids, planner_config, run_config.gen_batch_size, run_config.max_context_chars):
            context = _target_context(source_copy, plans, batch, planner_config)
            targets: list[dict[str, Any]] = []
            for cue_id in batch:
                plan = plan_by_id[cue_id]
                row: dict[str, Any] = {"cue_id": cue_id, "original": plan.text, "max_chars": plan.max_chars}
                if stage == "repair":
                    row.update({"prior_candidate": feedback[cue_id]["candidate"], "feedback": feedback[cue_id]["feedback"]})
                targets.append(row)
            response = await _request(
                request_fn, generator, GENERATOR_SYSTEM_PROMPT, build_generation_prompt(context, targets),
                generation_schema(), run_config.length_tools, stage=stage, ids=batch, output_dir=output_dir,
                stage_calls=stage_calls, request_counts=request_counts, request_failures=request_failures,
            )
            parsed: PartialParseResult | None = None
            if response.error is None and response.text is not None:
                parsed = parse_generation_partial(response.text, batch)
            else:
                parsed = None
            for cue_id in batch:
                record = record_by_id[cue_id]
                attempt = sum(1 for event in record["history"] if event.get("stage") == stage) + 1
                prior_candidate = record.get("candidate")
                response_issues = parsed.response_issues if parsed is not None else []
                if response.error is not None or response.text is None:
                    record["status"] = "error"
                    record["failure"] = "transport"
                    record["history"].append({"stage": stage, "attempt": attempt, "status": "error",
                                              "failure": "transport", "response_issues": response_issues})
                    continue
                assert parsed is not None
                if cue_id in parsed.errors:
                    reason = parsed.errors[cue_id]
                    record["status"] = "error"
                    record["failure"] = "schema"
                    record["history"].append({"stage": stage, "attempt": attempt, "status": "error",
                                              "failure": "schema", "error": reason,
                                              "response_issues": response_issues})
                    if can_retry(round_index):
                        queue_unique(generation_queue, cue_id)
                    continue
                row = parsed.results[cue_id]
                candidate = row["text"] if row["status"] == "ok" else record["original"]
                local = _local_check(candidate, plan_by_id[cue_id])
                record["candidate"] = candidate
                # Any successfully parsed generation row supersedes the prior
                # candidate, including protocol and local-validation failures.
                # The old judgment remains available in history only.
                record["final_review"] = None
                record["generator_status"] = row["status"]
                record["history"].append({"stage": stage, "attempt": attempt,
                                          "generator_status": row["status"], "candidate": candidate,
                                          "local": local, "response_issues": response_issues})
                if row["status"] == "cannot_shorten":
                    if row["text"] not in ("", record["original"]):
                        # The claimed replacement is protocol data, never a candidate.
                        record["candidate"] = prior_candidate if isinstance(prior_candidate, str) else None
                        record["history"][-1]["candidate"] = record["original"]
                        record["history"][-1]["local"] = _local_check(record["original"], plan_by_id[cue_id])
                        record["failure"] = "protocol_misuse"
                        record["history"][-1]["error"] = "cannot_shorten returned changed text"
                        record["status"] = "manual"
                        record["final_text"] = record["original"]
                        if can_retry(round_index):
                            queue_unique(generation_queue, cue_id)
                    elif stage == "repair":
                        record["failure"] = "cannot_shorten"
                        record["status"] = "manual"
                        record["final_text"] = record["original"]
                    else:
                        record["failure"] = "cannot_shorten"
                        record["status"] = "pending"
                        queue_unique(review_ids, cue_id)
                elif not local["valid"]:
                    record["status"] = "error"
                    record["failure"] = "schema"
                    record["history"][-1]["error"] = "generated candidate failed local format validation"
                    if can_retry(round_index):
                        queue_unique(generation_queue, cue_id)
                else:
                    # A valid new candidate clears only stale schema state; its complete history remains.
                    record.pop("failure", None)
                    record["status"] = "pending"
                    record["final_review"] = None
                    queue_unique(review_ids, cue_id)
            checkpoint()
        return review_ids

    def apply_review_result(
        cue_id: str, verdict: dict[str, Any], *, initial: bool, round_index: int | None,
        next_generation: list[str],
    ) -> None:
        record = record_by_id[cue_id]
        if record.get("failure") == "protocol_misuse":
            record["status"] = "manual"
            record["final_text"] = record["original"]
            return
        candidate = record["original"] if record.get("failure") == "cannot_shorten" else record.get("candidate", record["original"])
        local = _local_check(candidate, plan_by_id[cue_id])
        record["final_review"] = verdict
        if verdict["verdict"] == "accept" and local["valid"] and local["within_length"]:
            record.pop("failure", None)
            record["status"] = "accepted"
            record["final_text"] = candidate
            if initial:
                record["initial_accepted"] = True
        elif verdict["verdict"] == "impossible":
            record["status"] = "manual"
            record["failure"] = "semantic"
            record["final_text"] = record["original"]
        elif can_retry(round_index):
            if record.get("failure") == "schema":
                record.pop("failure", None)
            record["status"] = "pending"
            queue_unique(next_generation, cue_id)
        else:
            record["status"] = "manual"
            record["failure"] = _reason_for(plan_by_id[cue_id], candidate, local, verdict["verdict"])
            record["final_text"] = record["original"]

    async def run_review(
        ids: list[str], *, stage: str, initial: bool, round_index: int | None,
    ) -> tuple[list[str], list[str]]:
        next_generation: list[str] = []
        next_review: list[str] = []
        for batch in make_batches(source_copy, plans, ids, planner_config, run_config.review_batch_size, run_config.max_context_chars):
            context = _target_context(source_copy, plans, batch, planner_config)
            targets: list[dict[str, Any]] = []
            for cue_id in batch:
                record = record_by_id[cue_id]
                candidate = record["original"] if record.get("failure") in {"cannot_shorten", "protocol_misuse"} else record.get("candidate", record["original"])
                targets.append({"cue_id": cue_id, "original": record["original"], "candidate": candidate,
                                "max_chars": record["budget"]["max_chars"], "actual_chars": len(candidate)})
            response = await _request(
                request_fn, arbiter, ARBITER_SYSTEM_PROMPT, build_review_prompt(context, targets), review_schema(), False,
                stage=stage, ids=batch, output_dir=output_dir, stage_calls=stage_calls,
                request_counts=request_counts, request_failures=request_failures,
            )
            parsed: PartialParseResult | None = (
                parse_review_partial(response.text, batch)
                if response.error is None and response.text is not None
                else None
            )
            for cue_id in batch:
                record = record_by_id[cue_id]
                attempt = sum(1 for event in record["history"] if event.get("stage") == stage) + 1
                response_issues = parsed.response_issues if parsed is not None else []
                if response.error is not None or response.text is None:
                    record["status"] = "error"
                    record["failure"] = "transport"
                    record["history"].append({"stage": stage, "attempt": attempt, "status": "error",
                                              "failure": "transport", "response_issues": response_issues})
                elif parsed is not None and cue_id in parsed.errors:
                    record["status"] = "error"
                    record["failure"] = "schema"
                    record["history"].append({"stage": stage, "attempt": attempt, "status": "error",
                                              "failure": "schema", "error": parsed.errors[cue_id],
                                              "response_issues": response_issues})
                    if can_retry(round_index):
                        queue_unique(next_review, cue_id)
                else:
                    assert parsed is not None
                    verdict = parsed.results[cue_id]
                    record["history"].append({"stage": stage, "attempt": attempt,
                                              "judgment": verdict, "response_issues": response_issues})
                    apply_review_result(cue_id, verdict, initial=initial, round_index=round_index,
                                        next_generation=next_generation)
            checkpoint()
        return next_generation, next_review

    initial_review = await run_generation(eligible_ids, stage="generation", round_index=None)
    repair_from_review, review_recovery = await run_review(
        initial_review, stage="review", initial=True, round_index=None,
    )
    generation_queue = list(dict.fromkeys(generation_queue + repair_from_review))
    review_queue = list(dict.fromkeys(review_queue + review_recovery))

    for round_index in range(run_config.max_repairs):
        # Snapshot both queues: work discovered during this round belongs to the next round,
        # except that a newly generated candidate receives its review in this same round.
        generation_ids = list(generation_queue)
        review_ids = list(review_queue)
        generation_queue = []
        review_queue = []
        if not generation_ids and not review_ids:
            break
        generated_review = await run_generation(generation_ids, stage="repair", round_index=round_index)
        review_ids = list(dict.fromkeys(generated_review + review_ids))
        next_generation, next_review = await run_review(
            review_ids, stage="recheck", initial=False, round_index=round_index,
        )
        generation_queue = list(dict.fromkeys(generation_queue + next_generation))
        review_queue = list(dict.fromkeys(review_queue + next_review))

    for record in records:
        if record["status"] == "pending":
            candidate = record.get("candidate") if isinstance(record.get("candidate"), str) else record["original"]
            local = _local_check(candidate, plan_by_id[record["cue_id"]])
            verdict = record.get("final_review", {}).get("verdict") if isinstance(record.get("final_review"), dict) else None
            record["status"] = "manual"
            record["failure"] = _reason_for(plan_by_id[record["cue_id"]], candidate, local, verdict)
        if record["status"] != "accepted":
            record["final_text"] = record["original"]
    final_report = _build_report(items, plans, target_ids, planner_config, run_config, generator, arbiter, output_dir,
                                 records, stage_calls, source_path, _source_hash(items),
                                 run_state="incomplete" if request_failures else "completed",
                                 request_counts=request_counts, request_failures=request_failures)
    _checkpoint(output_dir, final_report, _prepare_checkpoint(source_copy, records, plans, planner_config))
    return final_report
