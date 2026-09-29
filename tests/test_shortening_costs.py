from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

import pytest

from utils.llm_types import CallRecord, call_record_from_dict
from utils.opencode_transport import OpenCodePromptUsage
from utils.shortening_costs import (
    CURRENT_PRICE_SNAPSHOT_ID,
    LEGACY_PRICE_SNAPSHOT_ID,
    OPENROUTER_USAGE_SOURCE,
    PRICE_SOURCE,
    estimate_call_cost,
    estimate_record_cost,
    is_peak_utc,
    project_500,
    summarize_costs,
)


def _usage(**kwargs: int) -> OpenCodePromptUsage:
    return OpenCodePromptUsage(**kwargs)


def test_peak_boundaries_and_reasoning_is_not_added_twice() -> None:
    monday = datetime(2026, 9, 7, tzinfo=timezone.utc)
    assert is_peak_utc(monday.replace(hour=1))
    assert not is_peak_utc(monday.replace(hour=4))
    assert is_peak_utc(monday.replace(hour=6))
    assert not is_peak_utc(monday.replace(hour=10))
    estimate = estimate_call_cost(_usage(input_tokens=1_000_000, output_tokens=1_000, reasoning_tokens=900), at=monday.replace(hour=2))
    assert estimate.cost_usd == pytest.approx(0.30 + 1_200 / 1_000_000)


def test_missing_usage_is_unknown_and_failed_call_is_counted() -> None:
    failed = CallRecord("deepseek", "deepseek-v4-flash", "off", "now", 0.1, None, "error")
    summary = summarize_costs([failed])
    assert summary["unknown_call_count"] == 1
    assert not summary["accounting_complete"]


def test_projection_scales_one_trial() -> None:
    value = project_500(selected_count=5, tokens={"input_tokens": 10}, known_cost_usd=2.0)
    assert value["scale_factor"] == 100
    assert value["projected_tokens"]["input_tokens"] == 1000
    assert value["projected_known_cost_usd"] == 200


def test_record_timestamp_controls_tariff_and_invalid_time_is_unknown() -> None:
    usage = _usage(input_tokens=1_000_000, output_tokens=0)
    peak = CallRecord("deepseek", "deepseek-v4-flash", "off", "2026-09-07T02:00:00Z", 0.1, usage, "success")
    offpeak = CallRecord("deepseek", "deepseek-v4-flash", "off", "2026-09-07T04:00:00Z", 0.1, usage, "success")
    assert estimate_record_cost(peak)["tariff_period"] == "peak"
    assert estimate_record_cost(offpeak)["tariff_period"] == "offpeak"
    invalid = CallRecord("deepseek", "deepseek-v4-flash", "off", "not-a-time", 0.1, usage, "error")
    assert estimate_record_cost(invalid)["unknown"] is True
    assert estimate_record_cost(invalid)["cost_usd"] is None


def test_record_without_snapshot_keeps_legacy_tariff_and_mixed_summary_names_sources() -> None:
    usage = _usage(input_tokens=1_000_000, output_tokens=0)
    timestamp = "2026-09-07T02:00:00Z"
    legacy = CallRecord("deepseek", "deepseek-v4-flash", "off", timestamp, 0.1, usage, "success")
    current = CallRecord(
        "deepseek", "deepseek-flash", "off", timestamp, 0.1, usage, "success",
        pricing_snapshot_id=CURRENT_PRICE_SNAPSHOT_ID,
    )
    assert estimate_record_cost(legacy)["pricing_snapshot_id"] == LEGACY_PRICE_SNAPSHOT_ID
    assert estimate_record_cost(legacy)["price_snapshot_date"] == "2026-09-09"
    assert estimate_record_cost(legacy)["price_source"] == PRICE_SOURCE
    assert estimate_record_cost(legacy)["cost_usd"] == pytest.approx(0.44)
    assert estimate_record_cost(current)["cost_usd"] == pytest.approx(0.30)
    summary = summarize_costs([legacy, current])
    assert summary["used_pricing_snapshot_ids"] == [CURRENT_PRICE_SNAPSHOT_ID, LEGACY_PRICE_SNAPSHOT_ID]
    assert summary["snapshot_date"] is None
    assert summary["price_source"] is None


def test_openai_usage_cost_is_informational_and_missing_usage_is_incomplete() -> None:
    usage = _usage(input_tokens=10, output_tokens=4, cost=1.25)
    record = CallRecord("openai", "gpt-5.6-luna", "medium", "2026-09-07T02:00:00Z", 0.1, usage, "success",
                        requested_model="openai/gpt-5.6-luna")
    summary = summarize_costs([record])
    assert summary["known_subtotal_usd"] == 0.0
    assert summary["openai_api_equivalent_cost_usd"] == 1.25
    missing = CallRecord("openai", "gpt-5.6-luna", "medium", "2026-09-07T02:00:00Z", 0.1, None, "error",
                         requested_model="openai/gpt-5.6-luna")
    incomplete = summarize_costs([missing])
    assert incomplete["unknown_call_count"] == 1
    assert not incomplete["accounting_complete"]


def test_openrouter_reported_cost_is_paid_even_for_openai_model() -> None:
    usage = _usage(input_tokens=10, output_tokens=4, reasoning_tokens=3, cost=1.25)
    record = CallRecord("openrouter", "openai/gpt-5.6-sol", "medium", "2026-09-07T02:00:00Z", 0.1, usage, "success",
                        requested_model="openai/gpt-5.6-sol")
    estimate = estimate_record_cost(record)
    assert estimate["cost_usd"] == 1.25
    assert summarize_costs([record])["known_subtotal_usd"] == 1.25


def test_openrouter_receipt_does_not_inherit_deepseek_tariff_metadata() -> None:
    record = CallRecord(
        "openrouter", "openai/gpt-5.6-sol", "medium", "2026-09-07T02:00:00Z", 0.1,
        _usage(input_tokens=10, output_tokens=4, cost=1.25), "success",
        requested_model="openai/gpt-5.6-sol", requested_variant="medium",
        pricing_snapshot_id="openrouter-snapshot", cost_source="openrouter_usage",
    )
    estimate = estimate_record_cost(record)
    assert estimate["price_snapshot_date"] is None
    assert estimate["price_source"] == OPENROUTER_USAGE_SOURCE
    assert estimate["pricing_snapshot_id"] == "openrouter-snapshot"
    assert estimate["cost_source"] == "openrouter_usage"


def test_openrouter_zero_cost_is_known_but_null_cost_is_unknown() -> None:
    timestamp = "2026-09-07T02:00:00Z"
    zero = CallRecord("openrouter", "meta/muse-spark-1.3-contributor", "low", timestamp, 0.1,
                      _usage(input_tokens=10, output_tokens=4, reasoning_tokens=9, cost=0), "success")
    unknown = CallRecord("openrouter", "meta/muse-spark-1.3-contributor", "low", timestamp, 0.1,
                         _usage(input_tokens=10, output_tokens=4, reasoning_tokens=9, cost=None), "error")
    zero_summary = summarize_costs([zero])
    assert zero_summary["known_subtotal_usd"] == 0.0
    assert zero_summary["unknown_call_count"] == 0
    mixed = summarize_costs([zero, unknown])
    assert mixed["known_subtotal_usd"] == 0.0
    assert mixed["unknown_call_count"] == 1


def test_mixed_openrouter_models_report_direct_totals_without_double_counting_reasoning() -> None:
    timestamp = "2026-09-07T02:00:00Z"
    muse = CallRecord("openrouter", "meta/muse-spark-1.3-contributor", "low", timestamp, 0.1,
                      _usage(input_tokens=10, output_tokens=4, reasoning_tokens=9, cost=0.3), "success")
    sol = CallRecord("openrouter", "openai/gpt-5.6-sol", "medium", timestamp, 0.1,
                     _usage(input_tokens=20, output_tokens=5, reasoning_tokens=11, cost=0.7), "success")
    summary = summarize_costs([muse, sol])
    assert summary["known_subtotal_usd"] == 1.0
    assert summary["openrouter_reported_cost_usd"] == 1.0
    assert summary["input_tokens"] == 30
    assert summary["output_tokens"] == 9
    assert summary["reasoning_tokens"] == 20


def test_openai_subscription_detection_still_applies_to_non_openrouter_provider() -> None:
    record = CallRecord("openai", "gpt-5.6-sol", "medium", "2026-09-07T02:00:00Z", 0.1,
                        _usage(input_tokens=10, output_tokens=4, cost=1.25), "success",
                        requested_model="openai/gpt-5.6-sol")
    assert estimate_record_cost(record)["cost_usd"] == 0.0
    assert summarize_costs([record])["known_subtotal_usd"] == 0.0


def test_missing_openai_usage_does_not_make_known_deepseek_direct_accounting_unknown() -> None:
    timestamp = "2026-09-07T04:00:00Z"
    deepseek = CallRecord(
        "deepseek", "deepseek-v4-flash", "off", timestamp, 0.1,
        _usage(input_tokens=10, output_tokens=4), "success",
    )
    openai = CallRecord("openai", "gpt", "medium", timestamp, 0.1, None, "error")
    summary = summarize_costs([deepseek, openai])
    assert summary["direct_paid_accounting_complete"] is True
    assert summary["accounting_complete"] is False


def test_unknown_provider_is_not_marked_direct_paid_complete() -> None:
    timestamp = "2026-09-07T04:00:00Z"
    records = [
        CallRecord("unknown", "model", "off", timestamp, 0.1,
                   _usage(input_tokens=10, output_tokens=4), "success"),
    ]
    summary = summarize_costs(records)
    assert summary["direct_paid_accounting_complete"] is False


def test_unknown_direct_provider_with_usage_is_unknown_without_cost_error() -> None:
    record = CallRecord("anthropic", "model", "off", "2026-09-07T04:00:00Z", 0.1, _usage(input_tokens=10, output_tokens=4), "success")
    estimate = estimate_record_cost(record)
    assert estimate["unknown"] is True
    assert estimate["cost_usd"] is None
    assert summarize_costs([record])["accounting_complete"] is False


def test_projection_uses_actual_batches_and_both_repair_requests() -> None:
    value = project_500(selected_count=5, known_cost_usd=0.0, accounting_complete=False,
                        gen_batch_size=7, review_batch_size=13, observed_repair_fraction=1.5)
    assert value["projected_request_counts"] == {
        "generation": 72, "review": 39, "repair": 108, "recheck": 58, "total": 277,
    }
    assert value["unknown_cost"] is True


def test_call_record_from_dict_roundtrips_nested_usage_and_optional_fields() -> None:
    original = CallRecord(
        "openai", "gpt-5.6-luna", "medium", "2026-09-07T04:00:00Z", 0.1,
        _usage(input_tokens=10, output_tokens=4, cost=1.25), "success", tool_calls=2,
        error="warning", requested_model="openai/gpt-5.6-luna", requested_variant="medium",
    )
    restored = call_record_from_dict(asdict(original))
    assert restored == original
    assert isinstance(restored.usage, OpenCodePromptUsage)


def test_call_record_from_dict_rejects_invalid_mapping_and_usage() -> None:
    with pytest.raises(TypeError):
        call_record_from_dict([])  # type: ignore[arg-type]
    data = asdict(CallRecord("deepseek", "m", "off", "2026-09-07T04:00:00Z", 0.1, None, "success"))
    data["usage"] = {"input_tokens": "unknown"}
    with pytest.raises(ValueError):
        call_record_from_dict(data)
