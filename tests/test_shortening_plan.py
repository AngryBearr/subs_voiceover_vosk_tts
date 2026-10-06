"""Tests for deterministic subtitle shortening planning."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, replace

import pytest

from utils.vosk_text import normalize_vosk_text
from utils.shortening_plan import (
    CuePlan,
    PlannerConfig,
    analysis_for_text,
    context_items,
    make_batches,
    plan_subtitles,
    select_sample,
)


def _calibration_config(*, coefficients: list[float], guard: float = 0.0) -> PlannerConfig:
    identity = {
        "engine": "vosk", "voice": 8, "rate": 1.5,
        "model_config_sha256": None, "model_fingerprint": "fingerprint",
        "normalizer": "vosk_punctuation_v1_no_ruaccent", "normalizer_sha256": "hash",
    }
    profile = {
        "status": "ok", "identity": identity,
        "normalizer": "vosk_punctuation_v1_no_ruaccent",
        "input_character_basis": "normalized_tts_text", "rate": 1.5,
        "models": {"char_linear": {"name": "char_linear", "rate": 1.5, "coefficients": coefficients}},
        "char_linear_guard_sec": guard,
    }
    return PlannerConfig(
        cps=13.0, max_speed=1.5, length_basis="normalized_tts_text",
        calibration_profile=profile, calibration_identity=identity,
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


def test_normalized_planning_and_analysis_share_character_basis() -> None:
    config = PlannerConfig(length_basis="normalized_tts_text", cps=13, max_speed=1.0)
    original = "«Да…»"
    plan = plan_subtitles([{"start": 0, "end": 1000, "text": original}], config)[0]
    assert plan.text == original
    assert plan.length_basis == "normalized_tts_text"
    analysis = analysis_for_text(plan, original, config)
    assert analysis["length_basis"] == "normalized_tts_text"
    assert analysis["budget_remaining_chars"] == plan.max_chars - len('"Да..."')


def test_calibrated_budget_can_be_below_nominal_without_double_speed() -> None:
    config = _calibration_config(coefficients=[0.2, 0.1], guard=0.1)
    original = "«абвгде»"
    prepared = normalize_vosk_text(original)
    plan = plan_subtitles([{"start": 0, "end": 1000, "text": original}], config)[0]

    assert plan.max_chars == int((1.0 - 0.2 - 0.1) // 0.1)
    assert plan.max_chars < int(13.0 * 1.5)
    assert plan.reduction_fraction == pytest.approx(1.0 - plan.max_chars / len(prepared))
    assert plan.required_ratio == pytest.approx((0.2 + 0.1 * len(prepared)) / 1.0)
    analysis = analysis_for_text(plan, original, config)
    assert analysis["budget_remaining_chars"] == plan.max_chars - len(prepared)
    assert analysis["estimate_label"] == "vosk_calibration_char_linear"
    assert analysis["estimate_basis"] == "vosk_calibration_char_linear"


def test_calibrated_budget_can_exceed_nominal_and_uses_normalized_fields() -> None:
    config = _calibration_config(coefficients=[0.1, 0.02])
    original = "а  б, в"
    prepared = normalize_vosk_text(original)
    plan = plan_subtitles([{"start": 0, "end": 1000, "text": original}], config)[0]

    assert plan.max_chars == 45
    assert plan.max_chars > int(13.0 * 1.5)
    assert plan.status == "fits"
    assert plan.reduction_fraction == 0.0
    assert plan.required_ratio == pytest.approx((0.1 + 0.02 * len(prepared)) / 1.0)
    analysis = analysis_for_text(plan, original, config)
    assert analysis["budget_remaining_chars"] == plan.max_chars - len(prepared)


def test_measured_fit_keeps_evidence_source_separate_from_calibrated_forecast() -> None:
    config = _calibration_config(coefficients=[0.1, 0.02])
    plan = plan_subtitles([{"start": 0, "end": 1000, "text": "а б"}], config)[0]
    measured = replace(plan, estimate_source="vosk_measurement")
    analysis = analysis_for_text(measured, measured.text, config)
    assert analysis["estimate_source"] == "vosk_measurement"
    assert analysis["estimate_label"] == "vosk_calibration_char_linear"
    assert analysis["estimate_basis"] == "vosk_calibration_char_linear"
    assert analysis["reason"] == "actual_fit"
    assert analysis["is_critical"] is False


def _frozen_calibration_config() -> PlannerConfig:
    # Frozen voice-8/rate-1.5 coefficients; tests do not need ignored artifacts.
    return _calibration_config(
        coefficients=[0.5066993855016315, 0.058044399760824324],
        guard=0.6197317842554448,
    )


@pytest.mark.parametrize(
    ("available_ms", "target", "raw"),
    [(158, 3, 0), (456, 8, 0), (1150, 22, 0), (1182, 23, 0), (1184, 23, 0), (1185, 1, 1)],
)
def test_frozen_zero_forecast_uses_nominal_target_without_relaxing_positive_cap(
    available_ms: int, target: int, raw: int,
) -> None:
    config = _frozen_calibration_config()
    items = [{"index": 7, "start": 0, "end": available_ms, "text": "Да."}]
    source = copy.deepcopy(items)
    plan = plan_subtitles(items, config)[0]
    assert items == source
    assert plan.max_chars == target
    assert plan.calibrated_max_chars == raw
    assert plan.estimate_source == "vosk_calibration_char_linear"
    assert CuePlan(**asdict(plan)) == plan
    analysis = analysis_for_text(plan, plan.text, config)
    assert analysis["calibrated_max_chars"] == raw
    assert analysis["max_chars"] == target
    assert analysis["estimate_label"] == "vosk_calibration_char_linear"
    assert analysis["reason"] != "actual_fit"
    if raw == 0:
        assert plan.status == "fits"  # Text target fit is not measured audio fit.
        assert plan.short_timing_risk and analysis["short_timing_risk"]
        assert analysis["budget_basis"] == "nominal_cps_fallback_zero_calibration"
        assert analysis["reason"] == "calibrated_zero_budget"
        assert "Audio timing is unverified" in analysis["timing_warning"]
        assert analysis["is_critical"] is False
    else:
        assert plan.status == "short_timing"
        assert analysis["budget_basis"] == "vosk_calibration_char_linear"


@pytest.mark.parametrize(
    ("text", "status"),
    [("Непредставимо.", "short_timing"), ("Это длинная реплика из нескольких слов.", "needs_shortening")],
)
def test_zero_calibration_overflow_keeps_text_classification(text: str, status: str) -> None:
    plan = plan_subtitles([{"start": 0, "end": 158, "text": text}], _frozen_calibration_config())[0]
    assert plan.status == status
    assert plan.max_chars == 3 and plan.calibrated_max_chars == 0
    assert plan.short_timing_risk


def test_ultrashort_window_does_not_invent_one_character_or_valid_timing() -> None:
    config = _frozen_calibration_config()
    for start, end in ((0, 0), (1, 0), (-1, 1), ("0", 158), (0, "158")):
        invalid = plan_subtitles([{"start": start, "end": end, "text": "Да."}], config)[0]
        assert invalid.status == "invalid" and invalid.invalid_timing
        assert invalid.max_chars == 0 and invalid.calibrated_max_chars is None
    plan = plan_subtitles([{"start": 0, "end": 1, "text": "Да."}], config)[0]
    assert plan.max_chars == plan.calibrated_max_chars == 0
    assert plan.status == "short_timing" and not plan.invalid_timing
    analysis = analysis_for_text(plan, plan.text, config)
    assert analysis["budget_basis"] == "vosk_calibration_char_linear"
    assert "no positive text target is representable" in analysis["timing_warning"]


def test_fallback_measurement_warning_clears_only_for_exact_measured_original() -> None:
    config = _frozen_calibration_config()
    plan = plan_subtitles([{"start": 0, "end": 158, "text": "Да."}], config)[0]
    measured = replace(plan, reason="actual_fit", short_timing_risk=False, estimate_source="vosk_measurement")
    original = analysis_for_text(measured, measured.text, config)
    assert original["reason"] == "actual_fit" and not original["short_timing_risk"]
    assert "timing_warning" not in original
    candidate = analysis_for_text(measured, "Да!", config)
    assert candidate["reason"] == "calibrated_zero_budget" and candidate["short_timing_risk"]
    assert candidate["estimate_source"] == "vosk_calibration_char_linear"
    assert "Audio timing is unverified" in candidate["timing_warning"]


def test_zero_fallback_does_not_bypass_profile_identity_check() -> None:
    config = _frozen_calibration_config()
    assert config.calibration_identity is not None
    identity = dict(config.calibration_identity, voice=7)
    with pytest.raises(ValueError, match="identity mismatch"):
        replace(config, calibration_identity=identity)
