"""Deterministic planning helpers for subtitle shortening.

This module deliberately contains no transport or model code.  It turns subtitle
records into stable, position-based plans and validates model-shaped results.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from typing import Any

from utils.text_metrics import classify_text, count_words, is_repetition_candidate


@dataclass(frozen=True)
class PlannerConfig:
    """Configuration for the character budget and context windows."""

    cps: float = 13.0
    max_speed: float = 1.5
    gap_ms: int = 200
    context_before: int = 3
    context_after: int = 3

    def __post_init__(self) -> None:
        try:
            cps = Decimal(str(self.cps))
            speed = Decimal(str(self.max_speed))
        except (InvalidOperation, ValueError):
            raise ValueError("cps and max_speed must be finite numbers") from None
        if not cps.is_finite() or cps <= 0:
            raise ValueError("cps must be finite and positive")
        if not speed.is_finite() or speed < Decimal("1") or speed > Decimal("1.5"):
            raise ValueError("max_speed must be between 1.0 and 1.5")
        for name, value in (
            ("gap_ms", self.gap_ms),
            ("context_before", self.context_before),
            ("context_after", self.context_after),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class CuePlan:
    """The stable plan for one source-position subtitle cue."""

    cue_id: str
    position: int
    text: str
    start: int | None
    end: int | None
    available_ms: int
    max_chars: int
    required_ratio: float | None
    reduction_fraction: float
    severity: str
    status: str
    words_count: int = 0
    repetition_candidate: bool = False
    short_timing_risk: bool = False
    reason: str = "unknown"
    estimate_source: str = "nominal_cps_estimate"
    invalid_timing: bool = False


@dataclass(frozen=True)
class _Record:
    position: int
    text: str
    text_valid: bool
    start: int | None
    end: int | None

    @property
    def cue_id(self) -> str:
        return "cue-" + str(self.position)

    @property
    def timing_valid(self) -> bool:
        return (
            self.start is not None
            and self.end is not None
            and self.start >= 0
            and self.end >= 0
            and self.end > self.start
        )


def _valid_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _normalise_text(value: Any) -> tuple[str, bool]:
    if isinstance(value, str):
        return value, True
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return " ".join(value), True
    return "", False


def _records(items: list[dict[str, Any]]) -> list[_Record]:
    result: list[_Record] = []
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            item = {}
        text, text_valid = (
            _normalise_text(item["text"]) if "text" in item else ("", False)
        )
        result.append(
            _Record(
                position=position,
                text=text,
                text_valid=text_valid,
                start=_valid_int(item.get("start")),
                end=_valid_int(item.get("end")),
            )
        )
    return result


def _decimal_budget(config: PlannerConfig, available_ms: int) -> int:
    budget = (
        Decimal(str(config.cps))
        * Decimal(str(config.max_speed))
        * Decimal(available_ms)
        / Decimal(1000)
    )
    return int(budget.to_integral_value(rounding=ROUND_FLOOR))


def _severity(reduction_fraction: float, status: str) -> str:
    if status == "invalid":
        return "invalid"
    if status == "fits":
        return "fits"
    if reduction_fraction <= 0.20:
        return "easy"
    if reduction_fraction <= 0.40:
        return "medium"
    return "hard"


def plan_subtitles(items: list[dict[str, Any]], config: PlannerConfig) -> list[CuePlan]:
    """Build source-position plans without changing ``items``."""

    if not isinstance(config, PlannerConfig):
        raise TypeError("config must be a PlannerConfig")
    records = _records(items)
    valid = sorted(
        (record for record in records if record.timing_valid),
        key=lambda record: (record.start, record.position),
    )
    available_by_position: dict[int, int] = {}
    for index, record in enumerate(valid):
        assert record.start is not None and record.end is not None
        available = record.end - record.start
        if index + 1 < len(valid):
            next_start = valid[index + 1].start
            assert next_start is not None
            available += max(0, next_start - record.end - config.gap_ms)
        available_by_position[record.position] = available

    plans: list[CuePlan] = []
    for record in records:
        cue_id = record.cue_id
        if not record.timing_valid or not record.text_valid:
            words_count = count_words(record.text) if record.text_valid else 0
            repetition_candidate = is_repetition_candidate(record.text) if record.text_valid else False
            invalid_timing = not record.timing_valid
            plans.append(
                CuePlan(
                    cue_id=cue_id,
                    position=record.position,
                    text=record.text,
                    start=record.start,
                    end=record.end,
                    available_ms=0,
                    max_chars=0,
                    required_ratio=None,
                    reduction_fraction=0.0,
                    severity="invalid",
                    status="invalid",
                    words_count=words_count,
                    repetition_candidate=repetition_candidate,
                    reason="invalid_timing" if invalid_timing else "invalid_text",
                    invalid_timing=invalid_timing,
                )
            )
            continue

        assert record.start is not None and record.end is not None
        available_ms = available_by_position[record.position]
        max_chars = _decimal_budget(config, available_ms)
        original_length = len(record.text)
        required_ratio = (
            (Decimal(original_length) / Decimal(str(config.cps)))
            / (Decimal(available_ms) / Decimal(1000))
            if available_ms > 0
            else None
        )
        required_ratio_float = float(required_ratio) if required_ratio is not None else None
        reduction = (
            max(0.0, 1.0 - (max_chars / original_length))
            if original_length
            else 0.0
        )
        classification = classify_text(record.text, max_chars)
        plans.append(
            CuePlan(
                cue_id=cue_id,
                position=record.position,
                text=record.text,
                start=record.start,
                end=record.end,
                available_ms=available_ms,
                max_chars=max_chars,
                required_ratio=required_ratio_float,
                reduction_fraction=reduction,
                severity=_severity(reduction, classification.status),
                status=classification.status,
                words_count=classification.words_count,
                repetition_candidate=classification.repetition_candidate,
                short_timing_risk=classification.short_timing_risk,
                reason=classification.reason,
            )
        )
    return plans


def _validate_target_ids(target_ids: list[str], known_ids: set[str]) -> None:
    if any(not isinstance(target_id, str) for target_id in target_ids):
        raise ValueError("target_ids must contain strings")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("target_ids must not contain duplicates")
    unknown = [target_id for target_id in target_ids if target_id not in known_ids]
    if unknown:
        raise ValueError(f"unknown cue_id(s): {', '.join(unknown)}")


def select_sample(
    plans: list[CuePlan],
    per_severity: int = 4,
    seed: int = 0,
    short_count: int = 0,
) -> list[str]:
    """Select a deterministic, source-ordered stratified sample."""

    for name, value in (("per_severity", per_severity), ("short_count", short_count)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    by_severity: dict[str, list[CuePlan]] = {"easy": [], "medium": [], "hard": []}
    short: list[CuePlan] = []
    for plan in plans:
        if plan.status == "needs_shortening" and plan.severity in by_severity:
            by_severity[plan.severity].append(plan)
        elif plan.status == "short_timing":
            short.append(plan)

    rng = random.Random(seed)

    def choose(pool: list[CuePlan], count: int) -> list[CuePlan]:
        ordered = sorted(pool, key=lambda plan: (plan.position, plan.cue_id))
        return rng.sample(ordered, min(count, len(ordered)))

    selected = [
        plan
        for severity in sorted(by_severity)
        for plan in choose(by_severity[severity], per_severity)
    ]
    selected.extend(choose(short, short_count))
    selected_ids = {plan.cue_id for plan in selected}
    return [plan.cue_id for plan in sorted(selected, key=lambda plan: plan.position) if plan.cue_id in selected_ids]


def _plan_map(plans: list[CuePlan]) -> dict[str, CuePlan]:
    result: dict[str, CuePlan] = {}
    for plan in plans:
        if plan.cue_id in result:
            raise ValueError(f"duplicate plan cue_id: {plan.cue_id}")
        result[plan.cue_id] = plan
    return result


def context_items(
    items: list[dict[str, Any]],
    plans: list[CuePlan],
    target_ids: list[str],
    config: PlannerConfig,
) -> list[dict[str, Any]]:
    """Return the deduplicated chronological context for target cues."""

    if not isinstance(config, PlannerConfig):
        raise TypeError("config must be a PlannerConfig")
    records = _records(items)
    record_by_id = {record.cue_id: record for record in records}
    plan_by_id = _plan_map(plans)
    known = set(record_by_id) & set(plan_by_id)
    _validate_target_ids(target_ids, known)
    valid = sorted(
        (record for record in records if record.timing_valid),
        key=lambda record: (record.start, record.position),
    )
    valid_index = {record.cue_id: index for index, record in enumerate(valid)}
    included: set[str] = set()
    for target_id in target_ids:
        if target_id in valid_index:
            index = valid_index[target_id]
            included.update(
                record.cue_id
                for record in valid[
                    max(0, index - config.context_before) : index + config.context_after + 1
                ]
            )
        else:
            # An invalid target has no chronological neighbours but remains visible.
            included.add(target_id)

    ordered = [record for record in valid if record.cue_id in included]
    ordered.extend(
        record
        for record in sorted(records, key=lambda record: record.position)
        if record.cue_id in included and not record.timing_valid
    )
    target_set = set(target_ids)
    result: list[dict[str, Any]] = []
    for record in ordered:
        compact: dict[str, Any] = {
            "cue_id": record.cue_id,
            "text": record.text,
            "is_target": record.cue_id in target_set,
        }
        if record.start is not None:
            compact["start"] = record.start
        if record.end is not None:
            compact["end"] = record.end
        result.append(compact)
    return result


def make_batches(
    items: list[dict[str, Any]],
    plans: list[CuePlan],
    target_ids: list[str],
    config: PlannerConfig,
    batch_size: int = 6,
    max_context_chars: int = 20000,
) -> list[list[str]]:
    """Greedily split targets by count and compact serialized context size."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if (
        isinstance(max_context_chars, bool)
        or not isinstance(max_context_chars, int)
        or max_context_chars <= 0
    ):
        raise ValueError("max_context_chars must be a positive integer")
    plan_by_id = _plan_map(plans)
    _validate_target_ids(target_ids, set(plan_by_id))
    if not target_ids:
        return []

    def context_size(ids: list[str]) -> int:
        payload = context_items(items, plans, ids, config)
        return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    batches: list[list[str]] = []
    current: list[str] = []
    for target_id in target_ids:
        candidate = current + [target_id]
        too_many = len(candidate) > batch_size
        too_large = not too_many and context_size(candidate) > max_context_chars
        if too_many or too_large:
            if not current:
                raise ValueError(f"context for target {target_id} exceeds max_context_chars")
            batches.append(current)
            current = [target_id]
            if context_size(current) > max_context_chars:
                raise ValueError(f"context for target {target_id} exceeds max_context_chars")
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def analysis_for_text(
    plan: CuePlan,
    text: str,
    config: PlannerConfig,
) -> dict[str, Any]:
    """Recompute the legacy analysis shape using the plan's full budget."""

    if not isinstance(config, PlannerConfig):
        raise TypeError("config must be a PlannerConfig")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    same_text = text == plan.text
    estimate_source = plan.estimate_source if same_text else "nominal_cps_estimate"
    budget_basis = (
        "measured_budget_estimate"
        if estimate_source == "measured_budget_estimate"
        else "stricter_character_budget"
    )
    base: dict[str, Any] = {
        "duration_sec": None,
        "estimated_sec": None,
        "estimate_label": "nominal_cps_estimate",
        "estimate_basis": "nominal_cps_estimate",
        "budget_label": (
            "measured_budget_estimate"
            if estimate_source == "measured_budget_estimate"
            else "stricter_character_budget"
        ),
        "budget_basis": (
            "measured_budget_estimate"
            if estimate_source == "measured_budget_estimate"
            else "stricter_character_budget"
        ),
        "max_chars": None,
        "budget_remaining_chars": None,
        "mismatch_ratio": None,
        "diff_sec": None,
        "extended_duration_sec": None,
        "extended_mismatch_ratio": None,
        "used_gap_sec": None,
        "words_count": count_words(text),
        "is_short_segment": False,
        "is_checked": False,
        "is_critical": False,
        "available_ms": plan.available_ms,
        "short_timing_risk": False,
        "repetition_candidate": is_repetition_candidate(text),
        "reason": "invalid_timing" if plan.invalid_timing else "invalid_text",
        "estimate_source": estimate_source,
        "invalid_timing": plan.invalid_timing,
    }
    if plan.status == "invalid" or plan.start is None or plan.end is None or plan.end <= plan.start:
        return base

    assert plan.start is not None and plan.end is not None
    duration_ms = plan.end - plan.start
    duration_sec = duration_ms / 1000.0
    estimated_sec = len(text) / float(config.cps)
    mismatch_ratio = estimated_sec / duration_sec
    diff_sec = estimated_sec - duration_sec
    extended_duration_sec: float | None = None
    extended_mismatch_ratio: float | None = None
    used_gap_sec: float | None = None
    if plan.available_ms > duration_ms:
        extended_duration_sec = plan.available_ms / 1000.0
        extended_mismatch_ratio = estimated_sec / extended_duration_sec
        used_gap_sec = (plan.available_ms - duration_ms) / 1000.0
    classification = classify_text(text, plan.max_chars)
    words_count = classification.words_count
    short_timing_risk = classification.short_timing_risk
    reason = classification.reason
    is_short_segment = duration_sec < 1.0
    is_checked = True
    # The acoustic character cap is intentionally stricter than the nominal CPS
    # estimate.  It is a diagnostic flag, not a semantic judgment.
    is_critical = classification.status != "fits"
    if same_text and plan.status == "fits" and plan.estimate_source == "vosk_measurement":
        short_timing_risk = False
        is_critical = False
        reason = "actual_fit"
    base.update(
        {
            "duration_sec": duration_sec,
            "estimated_sec": estimated_sec,
            "max_chars": plan.max_chars,
            "budget_remaining_chars": plan.max_chars - len(text),
            "mismatch_ratio": mismatch_ratio,
            "diff_sec": diff_sec,
            "extended_duration_sec": extended_duration_sec,
            "extended_mismatch_ratio": extended_mismatch_ratio,
            "used_gap_sec": used_gap_sec,
            "words_count": words_count,
            "is_short_segment": is_short_segment,
            "is_checked": is_checked,
            "is_critical": is_critical,
            "available_ms": plan.available_ms,
            "short_timing_risk": short_timing_risk,
            "repetition_candidate": classification.repetition_candidate,
            "reason": reason,
            "estimate_source": estimate_source,
            "budget_label": budget_basis,
            "budget_basis": budget_basis,
            "invalid_timing": plan.invalid_timing,
        }
    )
    return base
