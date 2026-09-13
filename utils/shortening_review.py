"""Pure, durable outcome and human-review state for subtitle shortening.

The module intentionally has no filesystem, transport, audio, or model side
effects.  A state is a complete snapshot, and its checksum covers every field
except ``snapshot_id``.
"""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum
import copy
import math
from typing import Any

from utils.length_check import check_length
from utils.shortening_plan import CuePlan, PlannerConfig, plan_subtitles
from utils.shortening_resume import canonical_hash


class ReviewError(ValueError):
    """Raised when a review snapshot or edit queue is not trustworthy."""


class Outcome(str, Enum):
    UNCHANGED_AND_FITS = "unchanged_and_fits"
    ACCEPTED_CORRECTION = "accepted_correction"
    NEEDS_ATTENTION = "needs_attention"
    INVALID = "invalid"
    NOT_CHECKED = "not_checked"


OUTCOME_UNCHANGED_AND_FITS = Outcome.UNCHANGED_AND_FITS.value
OUTCOME_ACCEPTED_CORRECTION = Outcome.ACCEPTED_CORRECTION.value
OUTCOME_NEEDS_ATTENTION = Outcome.NEEDS_ATTENTION.value
OUTCOME_INVALID = Outcome.INVALID.value
OUTCOME_NOT_CHECKED = Outcome.NOT_CHECKED.value
OUTCOMES = tuple(item.value for item in Outcome)
TERMINAL_OUTCOMES = frozenset({OUTCOME_UNCHANGED_AND_FITS, OUTCOME_ACCEPTED_CORRECTION})


def _fail(message: str) -> None:
    raise ReviewError(message)


def _hash(value: Any) -> str:
    try:
        return canonical_hash(value)
    except Exception as exc:
        raise ReviewError(f"value is not canonical JSON: {exc}") from exc


def _copy(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise ReviewError(f"value cannot be copied: {exc}") from exc


def _text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return " ".join(value)
    return None


def _valid_time(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _timing(item: dict[str, Any]) -> dict[str, Any]:
    start = _copy(item.get("start"))
    end = _copy(item.get("end"))
    duration = (
        _copy(item["duration"])
        if "duration" in item
        else end - start
        if _valid_time(start) and _valid_time(end) and end > start
        else None
    )
    return {"start": start, "end": end, "duration": duration}


def _timing_valid(timing: dict[str, Any]) -> bool:
    return (
        _valid_time(timing.get("start"))
        and _valid_time(timing.get("end"))
        and timing["end"] > timing["start"]
    )


def _strict_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _local(text: str | None, max_chars: Any, basis: Any) -> tuple[bool, int | None, bool]:
    if text is None or isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 0:
        return False, None, False
    try:
        length = check_length(text, max_chars, basis)
    except (TypeError, ValueError):
        return False, None, False
    valid = bool(text.strip()) and text == text.strip() and "\n" not in text and "\r" not in text
    return valid, int(length["length"]), bool(length["fits"])


def _config(config: PlannerConfig) -> dict[str, Any]:
    if not isinstance(config, PlannerConfig):
        _fail("planner_config must be a PlannerConfig")
    return _copy(asdict(config))


def _profile_fingerprint(config: PlannerConfig) -> str:
    return _hash({"calibration_profile": config.calibration_profile, "calibration_identity": config.calibration_identity})


def _context(items: list[dict[str, Any]], position: int, config: PlannerConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build before/after neighbors using the planner's chronological rule."""
    rows: list[dict[str, Any]] = []
    for pos, item in enumerate(items):
        timing = _timing(item)
        if not _timing_valid(timing):
            continue
        rows.append({
            "cue_id": f"cue-{pos}",
            "position": pos,
            "source_index": _copy(item.get("index")),
            # The planner's display text is the normalized string, including
            # for callers that store subtitle text as a list of fragments.
            "text": _text(item.get("text")),
            "start": timing["start"],
            "end": timing["end"],
            "duration": timing["duration"],
        })
    rows.sort(key=lambda row: (row["start"], row["position"]))
    target = next((row for row in rows if row["position"] == position), None)
    if target is None:
        return [], []
    index = rows.index(target)
    return (
        _copy(rows[max(0, index - config.context_before):index]),
        _copy(rows[index + 1:index + 1 + config.context_after]),
    )


def _context_hash(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> str:
    return _hash({"before": before, "after": after})


def _neighbor_timing_hash(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> str:
    return _hash({"before": [{k: row.get(k) for k in ("cue_id", "position", "start", "end", "duration")} for row in before],
                 "after": [{k: row.get(k) for k in ("cue_id", "position", "start", "end", "duration")} for row in after]})


def _candidate_texts(record: dict[str, Any]) -> list[str]:
    found: list[str] = []
    def add(value: Any) -> None:
        if isinstance(value, str) and value not in found:
            found.append(value)
    add(record.get("candidate"))
    add(record.get("final_text"))
    history = record.get("history")
    if isinstance(history, list):
        for event in history:
            if isinstance(event, dict):
                add(event.get("text"))
                add(event.get("candidate"))
                add(event.get("final_text"))
    return found


def _record_reason(record: dict[str, Any], default: str) -> str:
    for key in ("reason", "failure", "error"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            for subkey in ("reason", "message", "type", "error_type"):
                if isinstance(value.get(subkey), str) and value[subkey]:
                    return value[subkey]
    return default


def _error_category(record: dict[str, Any]) -> str:
    detail = record.get("failure", record.get("error", record.get("reason", record.get("error_type"))))
    text = str(detail).lower()
    if any(word in text for word in ("schema", "protocol", "malformed", "validation", "json", "cue_id")):
        return "protocol_error"
    return "transport_error"


def _current_plans(items: list[dict[str, Any]], config: PlannerConfig) -> dict[int, CuePlan]:
    try:
        return {plan.position: plan for plan in plan_subtitles(items, config)}
    except Exception as exc:
        raise ReviewError(f"cannot recompute current plans: {exc}") from exc


def _binding(source_fp: str, current_fp: str, timing_fp: str, profile_fp: str, config_fp: str,
             original_context_fp: str, current_context_fp: str, neighbor_timing_fp: str) -> dict[str, str]:
    return {
        "source_text_fingerprint": source_fp,
        "current_text_fingerprint": current_fp,
        "timing_fingerprint": timing_fp,
        "profile_fingerprint": profile_fp,
        "config_fingerprint": config_fp,
        "original_context_fingerprint": original_context_fp,
        "current_context_fingerprint": current_context_fp,
        "neighbor_timing_fingerprint": neighbor_timing_fp,
    }


def _previous_outcome(previous: dict[str, Any] | None, cue_id: str) -> dict[str, Any] | None:
    if previous is None:
        return None
    values = previous.get("outcomes")
    if not isinstance(values, list):
        return None
    return next((row for row in values if isinstance(row, dict) and row.get("cue_id") == cue_id), None)


def _same_canonical(left: Any, right: Any) -> bool:
    try:
        return _hash(left) == _hash(right)
    except ReviewError:
        return False


def _metadata_setting(metadata: Any, names: set[str]) -> Any:
    """Find an expected audio setting in report metadata, if one is present."""
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if key in names:
                return value
        for value in metadata.values():
            found = _metadata_setting(value, names)
            if found is not None:
                return found
    elif isinstance(metadata, list):
        for value in metadata:
            found = _metadata_setting(value, names)
            if found is not None:
                return found
    return None


def _valid_measurement(
    evidence: Any, plan: CuePlan, source_text: str | None,
    metadata: dict[str, Any] | None = None,
) -> tuple[float, bool, dict[str, Any]] | None:
    if not isinstance(evidence, dict):
        return None
    if evidence.get("error") is not None or evidence.get("error_type") is not None or evidence.get("status") == "error":
        return None
    if "cue_id" in evidence and evidence.get("cue_id") != plan.cue_id:
        return None
    if "original" not in evidence or evidence.get("original") != source_text:
        return None
    if "text" in evidence and evidence.get("text") != source_text:
        return None
    expected_voice = _metadata_setting(metadata, {"voice", "expected_voice"})
    expected_rate = _metadata_setting(metadata, {"rate", "speech_rate", "expected_rate", "expected_speech_rate"})
    if expected_voice is not None:
        supplied_voice = evidence.get("voice")
        if supplied_voice is not None and not _same_canonical(supplied_voice, expected_voice):
            return None
    if expected_rate is not None:
        supplied_rate = evidence.get("rate", evidence.get("speech_rate"))
        if supplied_rate is not None and not _same_canonical(supplied_rate, expected_rate):
            return None
    available = _strict_float(evidence.get("available_ms"))
    duration = _strict_float(evidence.get("duration_sec"))
    if available is None or duration is None or duration <= 0 or available != float(plan.available_ms):
        return None
    return duration, duration <= available / 1000.0, _copy(evidence)


def _state_checksum(payload: dict[str, Any]) -> str:
    without = {key: _copy(value) for key, value in payload.items() if key != "snapshot_id"}
    return _hash(without)


def build_review_state(
    *, source_items: list[dict[str, Any]], current_items: list[dict[str, Any]],
    plans: list[CuePlan], records: list[dict[str, Any]], planner_config: PlannerConfig,
    metadata: dict[str, Any] | None, selected_ids: list[str],
    previous_state: dict[str, Any] | None = None, force_recheck: bool = False,
    human_approvals: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a serializable snapshot without mutating any caller-owned value."""
    if not isinstance(source_items, list) or not isinstance(current_items, list) or not isinstance(plans, list) or not isinstance(records, list):
        _fail("items, plans, and records must be lists")
    if len(source_items) != len(current_items) or len(plans) != len(source_items):
        _fail("source/current/plans counts must match")
    if isinstance(force_recheck, bool) is False:
        _fail("force_recheck must be boolean")
    config_payload = _config(planner_config)
    if any(not isinstance(item, dict) for item in source_items + current_items + records):
        _fail("items and records must contain dictionaries")
    if any(not isinstance(item, dict) for item in source_items):
        _fail("source items must contain dictionaries")
    for left, right in zip(source_items, current_items):
        if not _same_canonical(left.get("index"), right.get("index")):
            _fail("source/current source-index identity mismatch")
    plan_by_id: dict[str, CuePlan] = {}
    positions: set[int] = set()
    for plan in plans:
        if not isinstance(plan, CuePlan) or not isinstance(plan.cue_id, str) or plan.cue_id in plan_by_id:
            _fail("plans must have unique CuePlan cue ids")
        if isinstance(plan.position, bool) or not isinstance(plan.position, int) or plan.position in positions:
            _fail("plans must have unique positions")
        if plan.cue_id != f"cue-{plan.position}":
            _fail("plans must use canonical position cue IDs")
        plan_by_id[plan.cue_id] = plan
        positions.add(plan.position)
    if positions != set(range(len(source_items))):
        _fail("plans must cover every source position")
    if not isinstance(selected_ids, list):
        _fail("selected_ids must be a list")
    selected = list(selected_ids)
    if any(not isinstance(value, str) for value in selected) or len(set(selected)) != len(selected):
        _fail("selected_ids must be unique strings")
    if any(value not in plan_by_id for value in selected):
        _fail("selected_ids contains an unknown cue")
    if human_approvals is not None and (not isinstance(human_approvals, dict) or any(key not in plan_by_id for key in human_approvals)):
        _fail("human_approvals contains an unknown cue")
    previous = _copy(previous_state) if previous_state is not None else None
    if previous is not None:
        validate_review_state(previous)
    source_copy = _copy(source_items)
    current_copy = _copy(current_items)
    records_copy = _copy(records)
    source_sha = _hash(source_copy)
    current_sha = _hash(current_copy)
    if previous is not None and (
        previous.get("source_sha256") != source_sha
        or not _same_canonical(previous.get("source_items"), source_copy)
    ):
        _fail("previous state source does not match current frozen source")
    # Semantic evidence is only reusable for the exact planner *and* runtime
    # contract which produced it.  The report metadata carries model,
    # calibration, prompt, and other runtime identity, so it belongs in this
    # binding rather than being treated as descriptive-only data.
    config_fp = _hash({"planner": config_payload, "metadata": metadata})
    profile_fp = _profile_fingerprint(planner_config)
    record_by_id: dict[str, dict[str, Any]] = {}
    for record in records_copy:
        cue_id = record.get("cue_id")
        if not isinstance(cue_id, str) or cue_id not in plan_by_id or cue_id in record_by_id:
            _fail("records must have known unique cue IDs")
        if "position" in record:
            position = record["position"]
            if isinstance(position, bool) or not isinstance(position, int) or position != plan_by_id[cue_id].position:
                _fail("record position does not match cue ID")
        record_by_id[cue_id] = record
    current_plan_by_position = _current_plans(current_copy, planner_config)
    budget_inputs_changed = previous is not None and (
        not _same_canonical(
            [_timing(item) for item in previous["current_items"]],
            [_timing(item) for item in current_copy],
        )
        or any(
            row.get("config_fingerprint") != config_fp
            or row.get("profile_fingerprint") != profile_fp
            for row in previous["outcomes"]
        )
    )
    outcome_rows: list[dict[str, Any]] = []
    unresolved: set[str] = set()

    previous_initial = previous.get("initial_intervention_ids") if previous else None
    if previous_initial is not None:
        if not isinstance(previous_initial, list) or any(not isinstance(x, str) for x in previous_initial):
            _fail("previous initial_intervention_ids malformed")
        initial_ids = list(dict.fromkeys(previous_initial))
        if any(value not in plan_by_id for value in initial_ids):
            _fail("previous initial_intervention_ids contains an unknown cue")
    else:
        initial_ids = [plan.cue_id for plan in plans if plan.status != "fits"]
        initial_ids.extend(value for value in selected if value not in initial_ids)
    previous_interventions = previous.get("intervention_ids", []) if previous else []
    if not isinstance(previous_interventions, list) or any(not isinstance(x, str) for x in previous_interventions):
        _fail("previous intervention_ids malformed")
    if any(value not in plan_by_id for value in previous_interventions):
        _fail("previous intervention_ids contains an unknown cue")
    intervention_ids = list(dict.fromkeys([*previous_interventions, *initial_ids]))

    for position in range(len(source_copy)):
        plan = next(item for item in plans if item.position == position)
        record = record_by_id.get(plan.cue_id, {})
        source_item = source_copy[position]
        current_item = current_copy[position]
        source_raw = source_item.get("text")
        current_raw = current_item.get("text")
        source_text = _text(source_raw)
        current_text = _text(current_raw)
        source_timing = _timing(source_item)
        current_timing = _timing(current_item)
        source_valid_timing = _timing_valid(source_timing) and _same_canonical(plan.start, source_timing["start"]) and _same_canonical(plan.end, source_timing["end"])
        current_plan = current_plan_by_position.get(position)
        old = _previous_outcome(previous, plan.cue_id)
        recalculate_current_budget = budget_inputs_changed
        frozen_max_chars = (
            old.get("max_chars")
            if isinstance(old, dict) and isinstance(old.get("max_chars"), int) and not isinstance(old.get("max_chars"), bool)
            else plan.max_chars
        )
        max_chars = (
            current_plan.max_chars
            if recalculate_current_budget and current_plan is not None and _timing_valid(current_timing)
            else frozen_max_chars
        )
        if not isinstance(max_chars, int) or max_chars < 0:
            max_chars = None
        plan_malformed = (
            not isinstance(plan.length_basis, str)
            or plan.length_basis not in {"display_text", "normalized_tts_text"}
            or isinstance(plan.max_chars, bool)
            or not isinstance(plan.max_chars, int)
            or plan.max_chars < 0
            or isinstance(plan.available_ms, bool)
            or not isinstance(plan.available_ms, int)
            or plan.available_ms < 0
            or (plan.start is not None and not _valid_time(plan.start))
            or (plan.end is not None and not _valid_time(plan.end))
        )
        source_shape_valid, _, _ = _local(source_text, plan.max_chars, plan.length_basis)
        current_shape_valid, actual_chars, local_fits = _local(current_text, max_chars, plan.length_basis)
        local_valid = current_shape_valid
        changed = source_raw != current_raw
        before, after = _context(source_copy, position, planner_config)
        current_before, current_after = _context(current_copy, position, planner_config)
        original_context_fp = _context_hash(before, after)
        current_context_fp = _context_hash(current_before, current_after)
        neighbor_timing_fp = _neighbor_timing_hash(current_before, current_after)
        source_fp = _hash({"text": source_raw, "timing": source_timing})
        current_fp = _hash(current_text)
        timing_fp = _hash({"source": source_timing, "current": current_timing, "source_available_ms": plan.available_ms,
                           "current_available_ms": current_plan.available_ms if current_plan is not None else plan.available_ms})
        binding = _binding(source_fp, current_fp, timing_fp, profile_fp, config_fp, original_context_fp, current_context_fp, neighbor_timing_fp)
        record_status = record.get("status")
        reason = str(plan.reason or "unknown")
        category = "semantic_unverified"
        outcome = OUTCOME_NEEDS_ATTENTION
        validation_source: str | None = None
        measured_duration: float | None = None
        measured_fit: bool | None = None
        evidence: dict[str, Any] = {}
        reusable = {"semantic": False, "measured": False}
        candidate_values = _candidate_texts(record)
        accepted_text = record.get("final_text") if isinstance(record.get("final_text"), str) else None
        final_review = record.get("final_review")
        model_accepted = (
            record_status == "accepted"
            and isinstance(final_review, dict)
            and final_review.get("verdict") == "accept"
            and isinstance(accepted_text, str)
            and accepted_text == current_text
        )
        candidate_valid = _local(accepted_text, max_chars, plan.length_basis) if accepted_text is not None else (False, None, False)
        approval = human_approvals.get(plan.cue_id) if human_approvals else None
        approval_stale = False
        invalid_data = (
            plan_malformed
            or source_text is None
            or current_text is None
            or not source_shape_valid
            or not current_shape_valid
            or not source_valid_timing
            or not _timing_valid(current_timing)
            or plan.status == "invalid"
        )
        if invalid_data:
            outcome = OUTCOME_INVALID
            if source_text is None or current_text is None or not source_shape_valid or not current_shape_valid or plan.reason == "invalid_text" or (plan.status == "invalid" and not plan.invalid_timing):
                category, reason = "invalid_text", "invalid_text"
            else:
                category, reason = "invalid_timing", "invalid_timing"
        if approval is not None:
            if not isinstance(approval, dict) or not isinstance(approval.get("text"), str) or not isinstance(approval.get("reviewed_context_fingerprint"), str):
                _fail("malformed human approval")
            if invalid_data:
                _fail(f"invalid cue {plan.cue_id} cannot be approved")
            if not invalid_data:
                approval_text = approval["text"]
                approval_prediction = _local(approval_text, max_chars, plan.length_basis)
                approval_valid = approval_prediction[0] and approval_prediction[2] and approval_text == current_text
                if not approval_valid:
                    _fail(f"human approval for {plan.cue_id} does not match current text or budget")
                if approval["reviewed_context_fingerprint"] == current_context_fp:
                    validation_source = "human"
                    reusable["semantic"] = True
                    evidence["semantic"] = {"source": "human", "validated_context_kind": "current",
                        "validated_context_fingerprint": current_context_fp, "final_context_checked": True,
                        "reuse_guard_context_fingerprint": current_context_fp, "approved_text": approval_text,
                        "valid": True, "binding": _copy(binding)}
                    outcome = OUTCOME_ACCEPTED_CORRECTION if changed else OUTCOME_UNCHANGED_AND_FITS
                    reason, category = "human_approved", "semantic"
                else:
                    approval_stale = True
                    reason, category = "context_changed", "semantic_unverified"

        # Invalid source/current data and invalid plans always win over predictions.
        if outcome == OUTCOME_NEEDS_ATTENTION and not invalid_data:
            error_status = record_status in {"error", "transport_error", "protocol_error"}
            if error_status and previous is None:
                category = _error_category(record)
                reason = _record_reason(record, category)
                outcome = OUTCOME_INVALID
            elif previous is None and record.get("failure") == "protocol_misuse":
                outcome = OUTCOME_NEEDS_ATTENTION
                category, reason = "protocol_error", "protocol_misuse"
            elif not previous:
                measurement = _valid_measurement(record.get("timing_evidence"), plan, source_text, metadata)
                if measurement is not None and not changed:
                    measured_duration, measured_fit, raw_evidence = measurement
                    validation_source = "measured"
                    reusable["measured"] = True
                    evidence["measurement"] = {"source": "measured", "timing_evidence": raw_evidence,
                        "valid": True, "binding": _copy(binding)}
                    outcome = OUTCOME_UNCHANGED_AND_FITS if measured_fit else OUTCOME_NEEDS_ATTENTION
                    category, reason = ("measured", "actual_fit") if measured_fit else ("timing_unverified", "actual_timing_over_window")
                elif model_accepted and candidate_valid[0] and candidate_valid[2]:
                    validation_source = "model"
                    reusable["semantic"] = True
                    evidence["semantic"] = {"source": "model", "validated_context_kind": "original",
                        "validated_context_fingerprint": original_context_fp, "final_context_checked": False,
                        "reuse_guard_context_fingerprint": current_context_fp, "approved_text": accepted_text,
                        "valid": True, "binding": _copy(binding)}
                    outcome = OUTCOME_ACCEPTED_CORRECTION if changed else OUTCOME_UNCHANGED_AND_FITS
                    category, reason = "semantic", "accepted_by_model"
                elif record_status in {None, "pending", "short_timing", "timing_unverified", "manual", "protocol_misuse"}:
                    if approval_stale:
                        outcome = OUTCOME_NEEDS_ATTENTION
                        category, reason = "semantic_unverified", "context_changed"
                    elif record_status == "short_timing" or plan.status == "short_timing":
                        outcome = OUTCOME_NEEDS_ATTENTION
                        category, reason = "short_timing", "short_timing"
                    elif record_status == "timing_unverified":
                        outcome = OUTCOME_NEEDS_ATTENTION
                        category, reason = "timing_unverified", "timing_unverified"
                    elif record_status == "manual":
                        outcome = OUTCOME_NEEDS_ATTENTION
                        category, reason = "semantic_unverified", "manual"
                    elif record_status == "protocol_misuse":
                        outcome = OUTCOME_NEEDS_ATTENTION
                        category, reason = "protocol_error", "protocol_misuse"
                    elif changed:
                        outcome = OUTCOME_NEEDS_ATTENTION
                        category, reason = "semantic_unverified", "external_change_requires_approval"
                    elif plan.status == "fits" and local_valid and local_fits:
                        validation_source = "prediction"
                        outcome = OUTCOME_UNCHANGED_AND_FITS
                        category, reason = "prediction", "fits"
                    elif previous and old and old.get("outcome") in TERMINAL_OUTCOMES:
                        category, reason = "semantic_unverified", "evidence_stale"
                    else:
                        outcome = OUTCOME_NOT_CHECKED
                        category, reason = "semantic_unverified", "not_checked"
                else:
                    category, reason = "semantic_unverified", _record_reason(record, "semantic_unverified")
            else:
                old_evidence = old.get("evidence") if isinstance(old, dict) else None
                old_semantic = old_evidence.get("semantic") if isinstance(old_evidence, dict) else None
                old_measurement = old_evidence.get("measurement") if isinstance(old_evidence, dict) else None
                old_binding = old_semantic.get("binding") if isinstance(old_semantic, dict) else None
                force = force_recheck
                semantic_ok = isinstance(old_semantic, dict) and old_semantic.get("valid") is True and isinstance(old_binding, dict) and not force and all(old_binding.get(k) == binding.get(k) for k in ("source_text_fingerprint", "current_text_fingerprint", "timing_fingerprint", "profile_fingerprint", "config_fingerprint", "original_context_fingerprint", "current_context_fingerprint"))
                measurement_ok = isinstance(old_measurement, dict) and old_measurement.get("valid") is True and not force
                old_measure_binding = old_measurement.get("binding") if isinstance(old_measurement, dict) else None
                measurement_ok = measurement_ok and isinstance(old_measure_binding, dict) and all(old_measure_binding.get(k) == binding.get(k) for k in ("source_text_fingerprint", "timing_fingerprint", "profile_fingerprint", "config_fingerprint", "neighbor_timing_fingerprint"))
                current_reuse_valid, _, current_reuse_fits = _local(current_text, max_chars, plan.length_basis)
                if (
                    semantic_ok
                    and isinstance(old_semantic.get("approved_text"), str)
                    and old_semantic["approved_text"] == current_text
                    and current_reuse_valid
                    and current_reuse_fits
                ):
                    evidence["semantic"] = _copy(old_semantic)
                    validation_source = str(old_semantic.get("source"))
                    reusable["semantic"] = True
                    outcome = OUTCOME_ACCEPTED_CORRECTION if changed else OUTCOME_UNCHANGED_AND_FITS
                    category, reason = "semantic", "reused_evidence"
                elif measurement_ok:
                    old_measurement_value = old_measurement.get("timing_evidence")
                    measured = _valid_measurement(old_measurement_value, plan, source_text, metadata)
                    if measured is not None and not changed:
                        measured_duration, measured_fit, _ = measured
                        evidence["measurement"] = _copy(old_measurement)
                        validation_source = "measured"
                        reusable["measured"] = True
                        outcome = OUTCOME_UNCHANGED_AND_FITS if measured_fit else OUTCOME_NEEDS_ATTENTION
                        category, reason = ("measured", "reused_measurement") if measured_fit else ("timing_unverified", "actual_timing_over_window")
                if not reusable["semantic"] and not reusable["measured"]:
                    reason = "force_recheck" if force else "evidence_stale"
                    category = "semantic_unverified"
                    was_intervened = bool(old and (old.get("outcome") not in TERMINAL_OUTCOMES or plan.cue_id in initial_ids or changed))
                    if not changed and local_valid and local_fits and plan.status == "fits":
                        validation_source = "prediction"
                        outcome = OUTCOME_UNCHANGED_AND_FITS
                        category, reason = "prediction", "fits"
                    elif (
                        not force
                        and isinstance(old, dict)
                        and old.get("outcome") not in TERMINAL_OUTCOMES
                        and old.get("current_text_fingerprint") == current_fp
                        and old.get("timing_fingerprint") == timing_fp
                        and old.get("config_fingerprint") == config_fp
                        and old.get("profile_fingerprint") == profile_fp
                        and old.get("current_context_fingerprint") == current_context_fp
                    ):
                        outcome = old["outcome"]
                        category = old["category"]
                        reason = old["reason"]
                    else:
                        outcome = OUTCOME_NEEDS_ATTENTION if was_intervened or changed else OUTCOME_NOT_CHECKED

        if approval_stale and outcome != OUTCOME_INVALID:
            outcome = OUTCOME_NEEDS_ATTENTION
            category, reason = "semantic_unverified", "context_changed"
        if plan_malformed or source_text is None or current_text is None or not source_valid_timing or not _timing_valid(current_timing):
            actual_chars = None
        if outcome in {OUTCOME_NEEDS_ATTENTION, OUTCOME_NOT_CHECKED, OUTCOME_INVALID}:
            unresolved.add(plan.cue_id)
        row: dict[str, Any] = {
            "cue_id": plan.cue_id, "position": plan.position, "source_index": _copy(source_item.get("index")),
            "text_changed": changed, "outcome": outcome, "reason": reason, "category": category,
            "validation_source": validation_source, "source_text": source_text, "current_text": current_text,
            "source_timing": source_timing, "current_timing": current_timing,
            "max_chars": max_chars, "actual_chars": actual_chars, "length_basis": plan.length_basis,
            "profile_fingerprint": profile_fp, "config_fingerprint": config_fp,
            "source_fingerprint": source_fp, "current_text_fingerprint": current_fp,
            "timing_fingerprint": timing_fp, "original_context_fingerprint": original_context_fp,
            "current_context_fingerprint": current_context_fp, "measured_duration_sec": measured_duration,
            "measured_fit": measured_fit, "evidence": evidence, "reusable_checks": reusable,
            "context_before": current_before, "context_after": current_after,
            "original_context_before": before, "original_context_after": after,
            "prediction": {"valid": local_valid, "actual_chars": actual_chars, "max_chars": max_chars,
                            "fits": local_fits if actual_chars is not None else None, "length_basis": plan.length_basis},
        }
        if isinstance(final_review, dict):
            row["last_model_review"] = _copy(final_review)
        row["candidates"] = candidate_values
        outcome_rows.append(row)

    if previous is not None and force_recheck:
        # Retain provenance in the rows while ensuring no old measurement leaks
        # into a forced snapshot.
        for row in outcome_rows:
            row["measured_duration_sec"] = None
            row["measured_fit"] = None
            row["reusable_checks"]["measured"] = False
    initial_count = len(initial_ids)
    initial_unresolved = sum(1 for row in outcome_rows if row["cue_id"] in initial_ids and row["outcome"] not in TERMINAL_OUTCOMES)
    intervention_ids.extend(cue_id for cue_id in sorted(unresolved, key=lambda cid: plan_by_id[cid].position) if cue_id not in intervention_ids)
    summary_counts = {outcome: sum(1 for row in outcome_rows if row["outcome"] == outcome) for outcome in OUTCOMES}
    summary: dict[str, Any] = {"counts": summary_counts, **{f"{key}_count": value for key, value in summary_counts.items()}}
    summary.update({
        "initial_intervention_count": initial_count,
        "initial_unresolved_count": initial_unresolved,
        "initial_unresolved_fraction": initial_unresolved / initial_count if initial_count else 0.0,
        "intervention_count": len(intervention_ids),
        "unresolved_count": len(unresolved),
        "unresolved_fraction": len(unresolved) / len(intervention_ids) if intervention_ids else 0.0,
        "unresolved_outside_initial_count": sum(1 for cue_id in unresolved if cue_id not in initial_ids),
        "evidence_unavailable": sum(1 for row in outcome_rows if row["category"] in {"timing_unverified", "semantic_unverified"}),
    })
    state: dict[str, Any] = {
        "schema_version": 1, "snapshot_id": "", "source_items": source_copy, "current_items": current_copy,
        "source_sha256": source_sha, "current_sha256": current_sha, "plans": [_copy(asdict(plan)) for plan in plans],
        "records": records_copy, "planner_config": config_payload, "metadata": _copy(metadata), "selected_ids": selected,
        "initial_intervention_ids": initial_ids, "intervention_ids": intervention_ids, "outcomes": outcome_rows,
        "summary": summary,
    }
    state["snapshot_id"] = _state_checksum(state)
    return state


def _expected_queue_entry(row: dict[str, Any]) -> dict[str, Any]:
    candidates = list(row.get("candidates", []))
    return {
        "cue_id": row["cue_id"], "position": row["position"], "source_index": row["source_index"],
        "original": row["source_text"], "current_text": row["current_text"],
        "source_timing": _copy(row["source_timing"]), "current_timing": _copy(row["current_timing"]),
        "context_before": _copy(row["context_before"]), "context_after": _copy(row["context_after"]),
        "original_context_before": _copy(row["original_context_before"]),
        "original_context_after": _copy(row["original_context_after"]),
        "current_context_fingerprint": row["current_context_fingerprint"], "candidates": candidates,
        "max_chars": row["max_chars"], "actual_chars": row["actual_chars"], "length_basis": row["length_basis"],
        "reason": row["reason"], "category": row["category"],
        "last_model_review": _copy(row.get("last_model_review")),
        "proposed_action": _proposed_action(row),
        "approved": False,
        "decision": "defer",
        "replacement_text": row["current_text"] if isinstance(row.get("current_text"), str) else "",
    }


def _proposed_action(row: dict[str, Any]) -> str:
    category = row.get("category")
    if category == "short_timing":
        return "check_short_timing"
    if category == "invalid_timing":
        return "fix_timing_externally"
    if category in {"protocol_error", "transport_error"}:
        return "inspect_protocol" if category == "protocol_error" else "recheck"
    if category == "invalid_text":
        return "fix_source_externally_then_start_new_run"
    return "edit_text"


def make_review_queue(state: dict[str, Any]) -> dict[str, Any]:
    validate_review_state(state)
    unresolved = [row for row in state["outcomes"] if row["outcome"] not in TERMINAL_OUTCOMES]
    return {
        "schema_version": 1, "snapshot_id": state["snapshot_id"], "source_sha256": state["source_sha256"],
        "current_sha256": state["current_sha256"], "entries": [_expected_queue_entry(row) for row in unresolved],
    }


def validate_review_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        _fail("invalid review state schema")
    required = {"schema_version", "snapshot_id", "source_items", "current_items", "source_sha256", "current_sha256", "plans", "records", "planner_config", "metadata", "selected_ids", "initial_intervention_ids", "intervention_ids", "outcomes", "summary"}
    if set(state) != required:
        _fail("review state shape mismatch")
    if not isinstance(state.get("source_items"), list) or not isinstance(state.get("current_items"), list) or len(state["source_items"]) != len(state["current_items"]):
        _fail("review state item shape mismatch")
    if any(not isinstance(item, dict) for item in state["source_items"] + state["current_items"]):
        _fail("review state items malformed")
    if any(not _same_canonical(left.get("index"), right.get("index")) for left, right in zip(state["source_items"], state["current_items"])):
        _fail("review state source-index identity mismatch")
    if _hash(state["source_items"]) != state["source_sha256"] or _hash(state["current_items"]) != state["current_sha256"]:
        _fail("review state item checksum mismatch")
    if not isinstance(state["plans"], list) or not isinstance(state["records"], list) or len(state["plans"]) != len(state["source_items"]):
        _fail("review state plans/records shape mismatch")
    if any(not isinstance(plan, dict) for plan in state["plans"]) or any(not isinstance(record, dict) for record in state["records"]):
        _fail("review state plans/records malformed")
    plan_ids = [plan.get("cue_id") for plan in state["plans"]]
    plan_positions = [plan.get("position") for plan in state["plans"]]
    if (
        any(not isinstance(cue_id, str) for cue_id in plan_ids)
        or len(set(plan_ids)) != len(plan_ids)
        or any(isinstance(position, bool) or not isinstance(position, int) for position in plan_positions)
        or set(plan_positions) != set(range(len(plan_positions)))
        or any(cue_id != f"cue-{position}" for cue_id, position in zip(plan_ids, plan_positions))
    ):
        _fail("review state plan identity malformed")
    plan_by_id = {plan["cue_id"]: plan for plan in state["plans"]}
    record_ids: list[Any] = []
    for record in state["records"]:
        cue_id = record.get("cue_id")
        if not isinstance(cue_id, str) or cue_id not in plan_by_id or cue_id in record_ids:
            _fail("review state record identity malformed")
        record_ids.append(cue_id)
        if "position" in record:
            position = record["position"]
            if isinstance(position, bool) or not isinstance(position, int) or position != plan_by_id[cue_id]["position"]:
                _fail("review state record position malformed")
    if not isinstance(state["outcomes"], list) or len(state["outcomes"]) != len(state["source_items"]):
        _fail("review state outcomes shape mismatch")
    outcome_required = {"cue_id", "position", "source_index", "text_changed", "outcome", "reason", "category", "validation_source", "source_text", "current_text", "source_timing", "current_timing", "max_chars", "actual_chars", "length_basis", "profile_fingerprint", "config_fingerprint", "source_fingerprint", "current_text_fingerprint", "timing_fingerprint", "original_context_fingerprint", "current_context_fingerprint", "measured_duration_sec", "measured_fit", "evidence", "reusable_checks", "context_before", "context_after", "original_context_before", "original_context_after", "prediction"}
    if any(not isinstance(row, dict) or row.get("outcome") not in OUTCOMES or not outcome_required.issubset(row) for row in state["outcomes"]):
        _fail("review state outcome malformed")
    outcome_ids = [row.get("cue_id") for row in state["outcomes"]]
    if (
        any(not isinstance(cue_id, str) for cue_id in outcome_ids)
        or len(set(outcome_ids)) != len(outcome_ids)
        or any(isinstance(row.get("position"), bool) or not isinstance(row.get("position"), int) or row["position"] != index for index, row in enumerate(state["outcomes"]))
        or set(outcome_ids) != set(plan_ids)
    ):
        _fail("review state outcome identity malformed")
    for row in state["outcomes"]:
        if not isinstance(row["text_changed"], bool) or not isinstance(row["reason"], str) or not isinstance(row["category"], str):
            _fail("review state outcome scalar malformed")
        if not isinstance(row["validation_source"], (str, type(None))) or row["validation_source"] not in {None, "prediction", "model", "human", "measured"}:
            _fail("review state validation source malformed")
        if row["source_text"] is not None and not isinstance(row["source_text"], str):
            _fail("review state source text malformed")
        if row["current_text"] is not None and not isinstance(row["current_text"], str):
            _fail("review state current text malformed")
        if row["max_chars"] is not None and (isinstance(row["max_chars"], bool) or not isinstance(row["max_chars"], int) or row["max_chars"] < 0):
            _fail("review state budget malformed")
        if row["actual_chars"] is not None and (isinstance(row["actual_chars"], bool) or not isinstance(row["actual_chars"], int) or row["actual_chars"] < 0):
            _fail("review state actual length malformed")
        if not isinstance(row["length_basis"], str) or row["length_basis"] not in {"display_text", "normalized_tts_text"}:
            _fail("review state length basis malformed")
        duration = row["measured_duration_sec"]
        if duration is not None and (_strict_float(duration) is None or float(duration) <= 0):
            _fail("review state measured duration malformed")
        if row["measured_fit"] is not None and not isinstance(row["measured_fit"], bool):
            _fail("review state measured flag malformed")
        if not isinstance(row["source_timing"], dict) or not isinstance(row["current_timing"], dict) or not isinstance(row["evidence"], dict) or not isinstance(row["reusable_checks"], dict) or row["reusable_checks"].keys() != {"semantic", "measured"} or any(not isinstance(value, bool) for value in row["reusable_checks"].values()):
            _fail("review state evidence shape malformed")
        if not isinstance(row["context_before"], list) or not isinstance(row["context_after"], list) or not isinstance(row["original_context_before"], list) or not isinstance(row["original_context_after"], list):
            _fail("review state context malformed")
    for key in ("selected_ids", "initial_intervention_ids", "intervention_ids"):
        if not isinstance(state[key], list) or any(not isinstance(value, str) for value in state[key]) or len(set(state[key])) != len(state[key]):
            _fail(f"review state {key} malformed")
        if any(value not in plan_ids for value in state[key]):
            _fail(f"review state {key} contains an unknown cue")
    if not isinstance(state["snapshot_id"], str) or _state_checksum(state) != state["snapshot_id"]:
        _fail("review state snapshot checksum mismatch")


def validate_edit_queue(state: dict[str, Any], edited: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate an edited queue and return only explicitly approved edits."""
    validate_review_state(state)
    expected = make_review_queue(state)
    if not isinstance(edited, dict) or set(edited) != set(expected):
        _fail("edit queue shape mismatch")
    if type(edited.get("schema_version")) is not type(expected["schema_version"]):
        _fail("edit queue schema version malformed")
    for key in ("schema_version", "snapshot_id", "source_sha256", "current_sha256"):
        if not _same_canonical(edited.get(key), expected[key]):
            _fail("edit queue snapshot mismatch")
    entries = edited.get("entries")
    if not isinstance(entries, list):
        _fail("edit queue entries must be a list")
    expected_by_id = {entry["cue_id"]: entry for entry in expected["entries"]}
    seen: set[str] = set()
    allowed = set(expected_by_id[next(iter(expected_by_id))]) | {"approved", "decision", "replacement_text"} if expected_by_id else {"approved", "decision", "replacement_text"}
    approved: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("cue_id"), str) or entry["cue_id"] in seen or entry["cue_id"] not in expected_by_id:
            _fail("unknown or duplicate edit queue entry")
        seen.add(entry["cue_id"])
        if set(entry) - allowed:
            _fail("edit queue contains an unknown field")
        expected_entry = expected_by_id[entry["cue_id"]]
        immutable_entry = {key: value for key, value in expected_entry.items() if key not in {"approved", "decision", "replacement_text"}}
        if set(entry) != set(expected_entry) | {"approved", "decision", "replacement_text"}:
            _fail("edit queue entry shape mismatch")
        for key, value in immutable_entry.items():
            if not _same_canonical(entry.get(key), value):
                _fail("edit queue immutable field mismatch")
        if type(entry.get("approved")) is not bool or type(entry.get("decision")) is not str or entry.get("decision") not in {"defer", "replace", "keep"} or type(entry.get("replacement_text")) is not str:
            _fail("edit queue editable fields malformed")
        if not entry["approved"]:
            continue
        if entry["decision"] not in {"keep", "replace"}:
            _fail("approved queue entry must keep or replace")
        current = expected_entry["current_text"]
        text = entry["replacement_text"]
        if not isinstance(current, str) or not isinstance(expected_entry["max_chars"], int) or expected_entry["category"] in {"invalid_timing", "invalid_text"}:
            _fail("invalid cue cannot be approved through text queue")
        if entry["decision"] == "keep" and text != current:
            _fail("keep must preserve current text")
        if not text or text != text.strip() or "\n" in text or "\r" in text:
            _fail("replacement is not a valid single-line text")
        valid, _, fits = _local(text, expected_entry["max_chars"], expected_entry["length_basis"])
        if not valid or not fits:
            _fail("replacement does not fit its budget")
        approved[entry["cue_id"]] = {"text": text, "reviewed_context_fingerprint": expected_entry["current_context_fingerprint"]}
    return approved


def _md(value: Any) -> str:
    if value is None:
        return "(none)"
    text = str(value).replace("`", "\\`").replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    return text


def _context_md(rows: list[dict[str, Any]]) -> str:
    return "; ".join(f"{_md(row.get('cue_id'))} [{_md(row.get('start'))}-{_md(row.get('end'))}]: {_md(row.get('text'))}" for row in rows) or "(none)"


def render_review_markdown(state: dict[str, Any], queue: dict[str, Any]) -> str:
    validate_review_state(state)
    if not isinstance(queue, dict) or queue.get("snapshot_id") != state["snapshot_id"]:
        _fail("queue does not belong to state")
    groups = {
        "short_timing": "Short timing", "timing_unverified": "Timing unverified",
        "invalid_timing": "Invalid timing", "invalid_text": "Invalid text",
        "protocol_error": "Protocol errors", "transport_error": "Transport errors",
        "semantic": "Semantic", "semantic_unverified": "Semantic / unverified",
    }
    by_category: dict[str, list[dict[str, Any]]] = {key: [] for key in groups}
    entries = queue.get("entries", [])
    if not isinstance(entries, list):
        _fail("queue entries must be a list")
    for entry in entries:
        if isinstance(entry, dict):
            category = str(entry.get("category"))
            by_category.setdefault(category, []).append(entry)
    unknown_categories = sorted(key for key in by_category if key not in groups)
    ordered_groups = [*groups.items(), *[(key, f"Unknown category: {key}") for key in unknown_categories]]
    lines = ["# Shortening review queue", ""]
    for category, title in ordered_groups:
        rows = by_category.get(category, [])
        if not rows:
            continue
        lines += [f"## {title}", ""]
        for entry in rows:
            review = entry.get("last_model_review")
            if isinstance(review, dict):
                issues = review.get("issues")
                advice = review.get("advice")
                critique = f"Issues: {_md(issues)}; Advice: {_md(advice)}"
            else:
                critique = _md(None)
            candidates = entry.get("candidates", [])
            candidate_text = ", ".join(value for value in candidates if isinstance(value, str)) if isinstance(candidates, list) else candidates
            lines += [f"### {_md(entry.get('cue_id'))} (position {_md(entry.get('position'))})",
                      f"- Original: {_md(entry.get('original'))}", f"- Current: {_md(entry.get('current_text'))}",
                      f"- Candidates: {_md(candidate_text)}",
                      f"- Context before: {_context_md(entry.get('context_before', []))}", f"- Context after: {_context_md(entry.get('context_after', []))}",
                      f"- Budget: {_md(entry.get('max_chars'))} ({_md(entry.get('length_basis'))}); actual: {_md(entry.get('actual_chars'))}",
                      f"- Reason: {_md(entry.get('reason'))}", f"- Latest critique: {critique}",
                      f"- Action: {_md(entry.get('proposed_action'))}", ""]
    return "\n".join(lines)


def render_outcomes_markdown(state: dict[str, Any]) -> str:
    validate_review_state(state)
    lines = ["# Shortening outcomes", "", "| Cue | Position | Status | Changed | Validation source | Source | Current | Measured duration | Measured fit | Reason |", "|---|---:|---|---|---|---|---|---:|---|---|"]
    for row in state["outcomes"]:
        lines.append("| " + " | ".join((_md(row.get("cue_id")), _md(row.get("position")), _md(row.get("outcome")), _md(row.get("text_changed")), _md(row.get("validation_source")), _md(row.get("source_text")), _md(row.get("current_text")), _md(row.get("measured_duration_sec")), _md(row.get("measured_fit")), _md(row.get("reason")))) + " |")
    return "\n".join(lines) + "\n"


__all__ = ["ReviewError", "Outcome", "OUTCOME_UNCHANGED_AND_FITS", "OUTCOME_ACCEPTED_CORRECTION", "OUTCOME_NEEDS_ATTENTION", "OUTCOME_INVALID", "OUTCOME_NOT_CHECKED", "OUTCOMES", "TERMINAL_OUTCOMES", "build_review_state", "make_review_queue", "validate_review_state", "validate_edit_queue", "render_review_markdown", "render_outcomes_markdown"]
