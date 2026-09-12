"""Tests for deterministic subtitle shortening planning."""

from __future__ import annotations

import copy
import json
import random

import pytest

from utils.shortening_plan import (
    CuePlan,
    PlannerConfig,
    analysis_for_text,
    context_items,
    make_batches,
    plan_subtitles,
    select_sample,
)


@pytest.mark.parametrize(
    "text",
    ["Да.", "Я не знаю.", "Пошли, пошли, пошли.", "Да, нет, да."],
)
def test_russian_short_examples_share_fit_and_overflow_classification(text: str) -> None:
    fit = plan_subtitles([{"start": 0, "end": 2000, "text": text}], PlannerConfig())[0]
    overflow = plan_subtitles([{"start": 0, "end": 100, "text": text}], PlannerConfig())[0]
    assert fit.status == "fits"
    assert fit.short_timing_risk is False
    assert overflow.status in {"short_timing", "needs_shortening"}
    assert overflow.short_timing_risk is True
    assert overflow.words_count == len(text.split())


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, 100), (0, 0), (100, 0), (False, 100), (0, True), ("0", 100)],
)
def test_non_integer_and_invalid_timing_values_are_invalid(start: object, end: object) -> None:
    plan = plan_subtitles([{"start": start, "end": end, "text": "Да."}], PlannerConfig())[0]
    assert plan.status == "invalid"
    assert plan.invalid_timing is True


def test_budget_boundaries_and_last_cue() -> None:
    items = [
        {"start": 0, "end": 1000, "text": "x x x " + "x" * 18},
        {"start": 1200, "end": 2200, "text": "last"},
    ]
    plans = plan_subtitles(items, PlannerConfig())
    assert plans[0].available_ms == 1000
    assert plans[0].max_chars == 19
    assert plans[0].status == "needs_shortening"
    assert plans[1].available_ms == 1000
    assert plans[1].max_chars == 19
    assert plans[1].status == "fits"


def test_overlap_and_valid_chronological_neighbour() -> None:
    items = [
        {"start": 2000, "end": 3000, "text": "later"},
        {"start": 0, "end": 1000, "text": "first"},
        {"start": 900, "end": 1100, "text": "overlap"},
    ]
    plans = plan_subtitles(items, PlannerConfig(gap_ms=200))
    assert [plan.cue_id for plan in plans] == ["cue-0", "cue-1", "cue-2"]
    assert plans[1].available_ms == 1000
    assert plans[2].available_ms == 900


def test_invalid_timing_text_and_position_ids() -> None:
    items = [
        {"index": 9001, "start": 0, "end": 0, "text": "bad"},
        {"index": 2, "start": None, "end": 3, "text": "bad"},
        {"index": 2, "start": 0, "end": 1000, "text": ["a", "b"]},
        {"index": 3, "start": 1000, "end": 2000, "text": 12},
        {"index": 4, "start": -100, "end": 50, "text": "negative"},
    ]
    plans = plan_subtitles(items, PlannerConfig())
    assert [plan.cue_id for plan in plans] == ["cue-0", "cue-1", "cue-2", "cue-3", "cue-4"]
    assert plans[0].status == plans[1].status == plans[3].status == plans[4].status == "invalid"
    assert plans[2].text == "a b"
    assert items[0]["index"] == 9001


def test_no_mutation_unicode_join_and_short_timing() -> None:
    items = [{"start": 0, "end": 100, "text": ["é", "界"]}]
    before = copy.deepcopy(items)
    plan = plan_subtitles(items, PlannerConfig())[0]
    assert items == before
    assert plan.text == "é 界"
    assert plan.max_chars == 1
    assert plan.status == "short_timing"


def test_decimal_threshold_and_exact_fit() -> None:
    config = PlannerConfig(cps=13.325, max_speed=1.5)
    plan = plan_subtitles([{"start": 0, "end": 2000, "text": "x" * 39}], config)[0]
    assert plan.max_chars == 39
    assert plan.status == "fits"


def test_stratified_sampling_is_seeded_and_source_ordered() -> None:
    items = [
        {"start": index * 1000, "end": index * 1000 + 100, "text": "one two three four five six"}
        for index in range(12)
    ]
    plans = plan_subtitles(items, PlannerConfig())
    first = select_sample(plans, per_severity=2, seed=7)
    random_state = random.getstate()
    assert first == select_sample(plans, per_severity=2, seed=7)
    assert random.getstate() == random_state
    assert first == ["cue-2", "cue-5"]
    assert first == sorted(first, key=lambda value: int(value.split("-")[1]))
    assert len(first) == 2


def test_context_union_and_batch_limits() -> None:
    items = [
        {"start": index * 1000, "end": index * 1000 + 100, "text": f"cue {index}"}
        for index in range(8)
    ]
    config = PlannerConfig(context_before=1, context_after=1)
    plans = plan_subtitles(items, config)
    context = context_items(items, plans, ["cue-2", "cue-3"], config)
    assert [row["cue_id"] for row in context] == ["cue-1", "cue-2", "cue-3", "cue-4"]
    assert sum(row["cue_id"] == "cue-2" for row in context) == 1
    assert [row["is_target"] for row in context] == [False, True, True, False]
    assert make_batches(items, plans, ["cue-2", "cue-3", "cue-7"], config, batch_size=2) == [
        ["cue-2", "cue-3"],
        ["cue-7"],
    ]


def test_batch_context_cap_and_single_target_error() -> None:
    items = [{"start": 0, "end": 1000, "text": "x" * 100}]
    config = PlannerConfig()
    plans = plan_subtitles(items, config)
    with pytest.raises(ValueError, match="exceeds"):
        make_batches(items, plans, ["cue-0"], config, max_context_chars=10)


def test_analysis_keeps_gap_after_candidate_shortening() -> None:
    items = [
        {"start": 0, "end": 1000, "text": "long original phrase"},
        {"start": 1500, "end": 2500, "text": "next"},
    ]
    config = PlannerConfig()
    plan = plan_subtitles(items, config)[0]
    result = analysis_for_text(plan, "ok", config)
    assert result["extended_duration_sec"] == 1.3
    assert result["used_gap_sec"] == 0.3
    assert result["extended_mismatch_ratio"] < 1.3
    assert result["is_critical"] is False
    assert all(value != float("inf") for value in result.values() if isinstance(value, float))


def test_invalid_analysis_is_json_safe() -> None:
    plan = CuePlan("cue-0", 0, "", None, None, 0, 0, None, 0.0, "invalid", "invalid")
    assert json.dumps(analysis_for_text(plan, "anything", PlannerConfig()))
