"""Conservative, usage-based accounting for subtitle shortening experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from utils.llm_types import CallRecord


PRICE_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing"
OPENROUTER_USAGE_SOURCE = "https://openrouter.ai/docs/api/api-reference/generations/get-request-&-usage-metadata-for-a-generation"
LEGACY_PRICE_SNAPSHOT_ID = "deepseek-v4-2026-09-09"
CURRENT_PRICE_SNAPSHOT_ID = "deepseek-flash-2026-09-10"
PRICE_SNAPSHOT_DATE = "2026-09-10"
DEEPSEEK_FLASH_MODEL = "deepseek-flash"
PRICE_SNAPSHOTS: dict[str, dict[str, Any]] = {
    LEGACY_PRICE_SNAPSHOT_ID: {
        "date": "2026-09-09",
        "source": PRICE_SOURCE,
        "model": "deepseek-v4-flash",
        "provider_declared_version": "not_attested_actual",
        "note": "Historical tariff retained for records without pricing_snapshot_id.",
        "rates": {
            "peak": {"cache_hit": 0.014, "cache_miss": 0.44, "output": 1.32},
            "offpeak": {"cache_hit": 0.007, "cache_miss": 0.22, "output": 0.66},
        },
    },
    CURRENT_PRICE_SNAPSHOT_ID: {
        "date": "2026-09-10",
        "source": PRICE_SOURCE,
        "model": "deepseek-flash",
        "provider_declared_version": "DeepSeek V4.1",
        "note": "Provider announced the V4.1 alias update; this snapshot does not attest a returned model version.",
        "rates": {
            "peak": {"cache_hit": 0.006, "cache_miss": 0.30, "output": 1.20},
            "offpeak": {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.60},
        },
    },
}
DEEPSEEK_RATES_USD_PER_MILLION: dict[str, dict[str, float]] = PRICE_SNAPSHOTS[CURRENT_PRICE_SNAPSHOT_ID]["rates"]  # type: ignore[assignment]


@dataclass(frozen=True)
class CostEstimate:
    cost_usd: float | None
    peak_upper_cost_usd: float | None
    unknown: bool
    reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    cache_hit_tokens: int | None
    cache_miss_tokens: int | None
    tariff_started_at: str | None = None
    tariff_period: str | None = None
    tariff_date: str = PRICE_SNAPSHOT_DATE
    tariff_source: str = PRICE_SOURCE
    pricing_snapshot_id: str | None = CURRENT_PRICE_SNAPSHOT_ID
    direct_cost_usd: float | None = None
    informational_api_equivalent_cost_usd: float | None = None
    accounting_complete: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_peak_utc(value: datetime | None = None) -> bool:
    """Return whether a UTC instant is in DeepSeek's weekday peak window."""
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    return current.weekday() < 5 and (1 <= current.hour < 4 or 6 <= current.hour < 10)


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _usage_values(usage: Any) -> tuple[int | None, int | None, int | None]:
    return (
        _integer(getattr(usage, "input_tokens", None)),
        _integer(getattr(usage, "output_tokens", None)),
        _integer(getattr(usage, "cache_read_tokens", None)),
    )


def _snapshot(snapshot_id: str) -> dict[str, Any] | None:
    value = PRICE_SNAPSHOTS.get(snapshot_id)
    return value if isinstance(value, dict) else None


def estimate_call_cost(usage: Any, *, at: datetime | None = None,
                       snapshot_id: str = CURRENT_PRICE_SNAPSHOT_ID,
                       provider: str | None = None) -> CostEstimate:
    """Estimate usage for a known tariff; OpenRouter uses its reported USD cost."""
    if provider == "openrouter" or getattr(usage, "provider_id", None) == "openrouter":
        input_tokens, output_tokens, hit = _usage_values(usage)
        reported = getattr(usage, "cost", None)
        valid_reported = (
            isinstance(reported, (int, float)) and not isinstance(reported, bool)
            and math.isfinite(float(reported)) and reported >= 0
        )
        if not valid_reported:
            return CostEstimate(
                None, None, True, "missing_or_invalid_reported_cost", input_tokens, output_tokens, hit, None,
                tariff_date=None, tariff_source=OPENROUTER_USAGE_SOURCE,
                pricing_snapshot_id=None, accounting_complete=False,
            )
        return CostEstimate(
            float(reported), None, False, None, input_tokens, output_tokens, hit, None,
            tariff_date=None, tariff_source=OPENROUTER_USAGE_SOURCE,
            pricing_snapshot_id=None, direct_cost_usd=float(reported), accounting_complete=True,
        )
    snapshot = _snapshot(snapshot_id)
    if snapshot is None:
        input_tokens, output_tokens, hit = _usage_values(usage)
        return CostEstimate(
            None, None, True, "unknown_pricing_snapshot", input_tokens, output_tokens, hit, None,
            pricing_snapshot_id=snapshot_id,
        )
    rates_table = snapshot["rates"]
    tariff_date = str(snapshot["date"])
    tariff_source = str(snapshot["source"])
    input_tokens, output_tokens, hit = _usage_values(usage)
    if input_tokens is None or output_tokens is None:
        return CostEstimate(
            None, None, True, "missing_input_or_output_usage", input_tokens, output_tokens, hit, None,
            tariff_date=tariff_date, tariff_source=tariff_source,
            pricing_snapshot_id=snapshot_id, accounting_complete=False,
        )
    raw_hit = getattr(usage, "cache_read_tokens", None)
    if raw_hit is not None and hit is None:
        return CostEstimate(
            None, None, True, "invalid_cache_hit_usage", input_tokens, output_tokens, None, None,
            tariff_date=tariff_date, tariff_source=tariff_source,
            pricing_snapshot_id=snapshot_id, accounting_complete=False,
        )
    if hit is not None and hit > input_tokens:
        return CostEstimate(
            None, None, True, "cache_hit_exceeds_input", input_tokens, output_tokens, hit, None,
            tariff_date=tariff_date, tariff_source=tariff_source,
            pricing_snapshot_id=snapshot_id,
            accounting_complete=False,
        )
    # Cache-hit accounting is absent in some provider responses.  Charging all
    # input as a miss is deliberately conservative and is marked as estimated.
    hit_tokens = hit or 0
    miss_tokens = input_tokens - hit_tokens
    rate_key = "peak" if is_peak_utc(at) else "offpeak"
    rates = rates_table[rate_key]
    cost = (hit_tokens * rates["cache_hit"] + miss_tokens * rates["cache_miss"] + output_tokens * rates["output"]) / 1_000_000
    upper = (
        hit_tokens * rates_table["peak"]["cache_hit"]
        + miss_tokens * rates_table["peak"]["cache_miss"]
        + output_tokens * rates_table["peak"]["output"]
    ) / 1_000_000
    started = at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if at is not None else None
    return CostEstimate(
        cost, upper, False, None if hit is not None else "cache_hit_absent_all_miss_estimate",
        input_tokens, output_tokens, hit, miss_tokens, started, rate_key, tariff_date, tariff_source,
        snapshot_id,
    )


def _parse_utc(value: Any) -> datetime | None:
    """Parse a record timestamp without silently substituting the current time."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_openai_subscription(record: Any) -> bool:
    provider = str(getattr(record, "provider", "") or "").lower()
    if provider == "openrouter":
        return False
    requested = str(getattr(record, "requested_model", "") or "").lower()
    model = str(getattr(record, "model", "") or "").lower()
    return provider == "openai" or requested.startswith("openai/") or model.startswith("openai/")


def _record_estimate(record: Any) -> CostEstimate:
    usage = getattr(record, "usage", None)
    is_openai = _is_openai_subscription(record)
    informational = getattr(usage, "cost", None) if usage is not None and is_openai else None
    if informational is not None and (
        isinstance(informational, bool) or not isinstance(informational, (int, float))
        or not math.isfinite(float(informational)) or informational < 0
    ):
        informational = None

    started = _parse_utc(getattr(record, "started_at", None))
    provider = str(getattr(record, "provider", "unknown") or "unknown")
    record_snapshot_id = getattr(record, "pricing_snapshot_id", None)
    if provider == "deepseek":
        record_snapshot_id = record_snapshot_id or LEGACY_PRICE_SNAPSHOT_ID
    record_snapshot = _snapshot(record_snapshot_id)
    record_date = str(record_snapshot["date"]) if record_snapshot is not None else None
    record_source = str(record_snapshot["source"]) if record_snapshot is not None else None
    accounting_snapshot_id = record_snapshot_id if provider == "deepseek" else None
    accounting_date = record_date if accounting_snapshot_id is not None else None
    accounting_source = record_source if accounting_snapshot_id is not None else None
    if provider == "openrouter":
        accounting_snapshot_id = record_snapshot_id
        accounting_source = OPENROUTER_USAGE_SOURCE
    fallback_date = None if provider == "openrouter" else PRICE_SNAPSHOT_DATE
    fallback_source = OPENROUTER_USAGE_SOURCE if provider == "openrouter" else PRICE_SOURCE
    if started is None:
        return CostEstimate(
            None, None, True, "invalid_started_at", None, None, None, None,
            tariff_date=accounting_date or fallback_date, tariff_source=accounting_source or fallback_source,
            pricing_snapshot_id=accounting_snapshot_id,
            informational_api_equivalent_cost_usd=float(informational) if informational is not None else None,
            accounting_complete=False,
        )
    if usage is None:
        return CostEstimate(
            None, None, True, "missing_usage", None, None, None, None,
            tariff_started_at=started.isoformat().replace("+00:00", "Z"),
            tariff_period="peak" if is_peak_utc(started) else "offpeak",
            tariff_date=accounting_date or fallback_date, tariff_source=accounting_source or fallback_source,
            pricing_snapshot_id=accounting_snapshot_id,
            informational_api_equivalent_cost_usd=float(informational) if informational is not None else None,
            accounting_complete=False,
        )
    if provider == "openrouter":
        estimate = estimate_call_cost(usage, provider="openrouter")
        return CostEstimate(
            estimate.cost_usd, estimate.peak_upper_cost_usd, estimate.unknown, estimate.reason,
            estimate.input_tokens, estimate.output_tokens, estimate.cache_hit_tokens, estimate.cache_miss_tokens,
            tariff_started_at=started.isoformat().replace("+00:00", "Z"),
            tariff_period="peak" if is_peak_utc(started) else "offpeak",
            tariff_date=None, tariff_source=OPENROUTER_USAGE_SOURCE,
            pricing_snapshot_id=getattr(record, "pricing_snapshot_id", None),
            direct_cost_usd=estimate.cost_usd,
            accounting_complete=estimate.accounting_complete,
        )
    if is_openai:
        input_tokens, output_tokens, hit = _usage_values(usage)
        complete = input_tokens is not None and output_tokens is not None
        return CostEstimate(
            0.0 if complete else None, 0.0 if complete else None, not complete,
            None if complete else "missing_input_or_output_usage", input_tokens, output_tokens, hit,
            (input_tokens - hit) if input_tokens is not None and hit is not None and hit <= input_tokens else None,
            started.isoformat().replace("+00:00", "Z"),
            "peak" if is_peak_utc(started) else "offpeak",
            accounting_date or PRICE_SNAPSHOT_DATE, accounting_source or PRICE_SOURCE, accounting_snapshot_id,
            informational_api_equivalent_cost_usd=float(informational) if informational is not None else None,
            accounting_complete=complete,
        )
    if provider != "deepseek":
        return CostEstimate(
            None, None, True, "unknown_direct_provider", *_usage_values(usage), None,
            tariff_started_at=started.isoformat().replace("+00:00", "Z"),
            tariff_period="peak" if is_peak_utc(started) else "offpeak",
            tariff_date=accounting_date or fallback_date, tariff_source=accounting_source or fallback_source,
            pricing_snapshot_id=accounting_snapshot_id,
            informational_api_equivalent_cost_usd=float(informational) if informational is not None else None,
            accounting_complete=False,
        )
    estimate = estimate_call_cost(usage, at=started, snapshot_id=record_snapshot_id)
    return CostEstimate(
        estimate.cost_usd, estimate.peak_upper_cost_usd, estimate.unknown, estimate.reason,
        estimate.input_tokens, estimate.output_tokens, estimate.cache_hit_tokens, estimate.cache_miss_tokens,
        estimate.tariff_started_at, estimate.tariff_period, estimate.tariff_date, estimate.tariff_source,
        estimate.pricing_snapshot_id,
        direct_cost_usd=estimate.cost_usd,
        informational_api_equivalent_cost_usd=float(informational) if informational is not None else None,
        accounting_complete=estimate.accounting_complete,
    )


def estimate_record_cost(record: CallRecord) -> dict[str, Any]:
    """Return a JSON-ready per-call ledger row for ``calls.jsonl`` consumers."""
    estimate = _record_estimate(record)
    result = estimate.to_dict()
    result.update({
        "provider": getattr(record, "provider", "unknown"),
        "model": getattr(record, "model", "unknown"),
        "requested_model": getattr(record, "requested_model", None),
        "status": getattr(record, "status", None),
        "started_at": getattr(record, "started_at", None),
        "direct_cost_known": estimate.direct_cost_usd is not None or estimate.cost_usd is not None,
        "direct_cost_usd": estimate.cost_usd,
        "api_equivalent_cost_usd": estimate.informational_api_equivalent_cost_usd,
        "tariff_at_start": estimate.tariff_started_at,
        "rate_period": estimate.tariff_period,
        "price_snapshot_date": estimate.tariff_date,
        "price_source": estimate.tariff_source,
        "known_cost_usd": estimate.cost_usd,
    })
    cost_source = getattr(record, "cost_source", None)
    if cost_source is None:
        cost_source = getattr(getattr(record, "usage", None), "cost_source", None)
    if cost_source is not None:
        result["provider_reported_cost_source"] = cost_source
        result["cost_source"] = cost_source
    return result


def summarize_costs(records: Iterable[Any], *, provider: str | None = None) -> dict[str, Any]:
    """Summarize direct usage and retain OpenAI's informational API cost."""
    selected = [record for record in records if provider is None or getattr(record, "provider", None) == provider]
    estimates = [_record_estimate(record) for record in selected]
    known = [item.cost_usd for item in estimates if item.cost_usd is not None]
    upper = [item.peak_upper_cost_usd for item in estimates if item.peak_upper_cost_usd is not None]
    usage = [getattr(record, "usage", None) for record in selected]

    def total(name: str) -> int:
        return sum(getattr(item, name) or 0 for item in usage if item is not None)

    provider_counts: dict[str, int] = {}
    provider_tokens: dict[str, dict[str, int]] = {}
    missing_by_provider: dict[str, int] = {}
    for record, estimate in zip(selected, estimates):
        provider_name = str(getattr(record, "provider", "unknown") or "unknown")
        provider_counts[provider_name] = provider_counts.get(provider_name, 0) + 1
        bucket = provider_tokens.setdefault(provider_name, {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cache_read_tokens": 0})
        for name in bucket:
            value = getattr(getattr(record, "usage", None), name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                bucket[name] += value
        if estimate.unknown:
            missing_by_provider[provider_name] = missing_by_provider.get(provider_name, 0) + 1

    step_elapsed = sum(float(getattr(record, "elapsed_sec", 0.0) or 0.0) for record in selected)
    logical_elapsed_by_request: dict[str, float] = {}
    for record in selected:
        started = str(getattr(record, "started_at", ""))
        elapsed = float(getattr(record, "elapsed_sec", 0.0) or 0.0)
        logical_elapsed_by_request[started] = max(logical_elapsed_by_request.get(started, 0.0), elapsed)
    api_cost = sum(
        item.informational_api_equivalent_cost_usd or 0.0
        for item in estimates
    )
    openrouter_reported_cost = sum(
        item.cost_usd or 0.0
        for record, item in zip(selected, estimates)
        if str(getattr(record, "provider", "") or "").lower() == "openrouter"
    )
    reported_cost_sources = sorted({
        str(source)
        for record in selected
        for source in (
            getattr(record, "cost_source", None),
            getattr(getattr(record, "usage", None), "cost_source", None),
        )
        if source is not None
    })
    direct_candidates = [
        (record, estimate)
        for record, estimate in zip(selected, estimates)
        if not _is_openai_subscription(record)
    ]
    direct_paid_accounting_complete = not any(
        estimate.unknown for _, estimate in direct_candidates
    )
    snapshot_ids = sorted({
        estimate.pricing_snapshot_id for estimate in estimates
        if estimate.pricing_snapshot_id is not None
    })
    single_snapshot = PRICE_SNAPSHOTS.get(snapshot_ids[0]) if len(snapshot_ids) == 1 else None
    return {
        "snapshot_date": single_snapshot["date"] if single_snapshot is not None else None,
        "price_source": single_snapshot["source"] if single_snapshot is not None else None,
        "used_pricing_snapshot_ids": snapshot_ids,
        "pricing_snapshot_ids": snapshot_ids,
        "used_snapshot_ids": snapshot_ids,
        "known_subtotal_usd": sum(known),
        "peak_upper_estimate_usd": sum(upper),
        "unknown_call_count": sum(item.unknown for item in estimates),
        "unknown_usage_count": sum(item.reason in {"missing_usage", "missing_input_or_output_usage"} for item in estimates),
        "accounting_complete": not any(item.unknown for item in estimates),
        "direct_paid_accounting_complete": direct_paid_accounting_complete,
        "call_count": len(selected),
        "provider_counts": provider_counts,
        "provider_token_counts": provider_tokens,
        "missing_usage_by_provider": missing_by_provider,
        "logical_elapsed_sec": sum(logical_elapsed_by_request.values()),
        "assistant_step_elapsed_sec": step_elapsed,
        "elapsed_note": "assistant-step elapsed is not request wall time; request metadata is supplied separately",
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "reasoning_tokens": total("reasoning_tokens"),
        "cache_read_tokens": total("cache_read_tokens"),
        "opencode_api_equivalent_cost_usd": api_cost,
        "openai_api_equivalent_cost_usd": api_cost,
        "openrouter_reported_cost_usd": openrouter_reported_cost,
        "openai_subscription_usage": "reported separately as API-equivalent informational usage; no invoice or free/unlimited claim",
        "cache_policy": "missing cache-hit usage is charged as all cache-miss and marked estimated",
        "provider_reported_cost_sources": reported_cost_sources,
    }


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def project_500(*, selected_count: int, tokens: dict[str, Any] | None = None,
                known_cost_usd: float | None = None, request_counts: dict[str, Any] | None = None,
                observed_repair_fraction: float = 0.0, accounting_complete: bool = True,
                gen_batch_size: int = 6, review_batch_size: int = 12) -> dict[str, Any]:
    """Project one trial to 500 selected cues, never aggregating competing models."""
    _positive_int("selected_count", selected_count)
    _positive_int("gen_batch_size", gen_batch_size)
    _positive_int("review_batch_size", review_batch_size)
    if not isinstance(accounting_complete, bool):
        raise TypeError("accounting_complete must be bool")
    if isinstance(observed_repair_fraction, bool) or not isinstance(observed_repair_fraction, (int, float)):
        raise ValueError("observed_repair_fraction must be finite")
    if not math.isfinite(float(observed_repair_fraction)) or observed_repair_fraction < 0 or observed_repair_fraction > 2:
        raise ValueError("observed_repair_fraction must be between 0 and 2")
    if known_cost_usd is not None and (
        isinstance(known_cost_usd, bool) or not isinstance(known_cost_usd, (int, float))
        or not math.isfinite(float(known_cost_usd)) or known_cost_usd < 0
    ):
        raise ValueError("known_cost_usd must be finite and non-negative")
    factor = 500 / selected_count
    source_tokens = tokens or {}
    projected_tokens = {
        key: (value * factor if isinstance(value, (int, float)) and not isinstance(value, bool) else None)
        for key, value in source_tokens.items()
    }
    estimated_repair_cues = 500 * float(observed_repair_fraction)
    counts = {
        "generation": math.ceil(500 / gen_batch_size),
        "review": math.ceil(500 / review_batch_size),
        "repair": math.ceil(estimated_repair_cues / gen_batch_size) if estimated_repair_cues else 0,
        "recheck": math.ceil(estimated_repair_cues / review_batch_size) if estimated_repair_cues else 0,
    }
    phase_total = sum(counts.values())
    counts.update(dict(request_counts or {}))
    counts["total"] = phase_total
    return {
        "basis": "single trial, same difficulty/context approximation; competing model experiments are not summed",
        "selected_count": selected_count,
        "scale_factor": factor,
        "projected_tokens": projected_tokens,
        "projected_known_cost_usd": known_cost_usd * factor if known_cost_usd is not None else None,
        "unknown_cost": not accounting_complete or known_cost_usd is None,
        "accounting_complete": accounting_complete and known_cost_usd is not None,
        "projected_request_counts": counts,
        "observed_request_counts": dict(request_counts or {}),
        "estimated_repair_cues": estimated_repair_cues,
        "observed_repair_fraction": observed_repair_fraction,
        "batch_sizes": {"generation": gen_batch_size, "review": review_batch_size},
        "approximation": True,
    }
