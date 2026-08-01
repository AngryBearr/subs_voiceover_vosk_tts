"""Characterization tests for the subtitle shortening domain core."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from utils.calibrate_tts_budget import POST_SILENCE_MEASUREMENT_LABEL as CALIBRATION_LABEL
from utils.calibrate_tts_budget import count_ascii_punctuation as calibration_punctuation
from utils.shortening_domain import (
    POST_SILENCE_MEASUREMENT_LABEL,
    CalibratedDurationModel,
    CharacterRateBudgetEstimator,
    DurationSelectionPolicy,
    TimingWindow,
    count_ascii_punctuation,
    load_calibrated_duration_profile,
)


def test_explicit_available_and_effective() -> None:
    window = TimingWindow.from_item({
        "start": 100,
        "end": 300,
        "analysis": {
            "available_duration_sec": 4.0,
            "effective_duration_sec": 5.0,
            "is_checked": True,
        },
    })
    assert window == TimingWindow(4.0, 5.0, True, True)
    assert window.as_analysis_fields() == {
        "available_duration_sec": 4.0,
        "effective_duration_sec": 5.0,
    }


def test_legacy_duration_and_extended_duration() -> None:
    window = TimingWindow.from_item({
        "analysis": {"duration_sec": 2.0, "extended_duration_sec": 3.5}
    })
    assert (window.base_duration_sec, window.effective_duration_sec) == (2.0, 3.5)


def test_estimated_mismatch_and_extended_mismatch() -> None:
    window = TimingWindow.from_item({
        "analysis": {
            "estimated_sec": 6.0,
            "mismatch_ratio": 2.0,
            "extended_mismatch_ratio": 1.5,
        }
    })
    assert (window.base_duration_sec, window.effective_duration_sec) == (3.0, 4.0)


def test_timestamp_fallback_without_analysis() -> None:
    window = TimingWindow.from_item({"start": 1200, "end": 3700})
    assert window.base_duration_sec == 2.5
    assert window.effective_duration_sec == 2.5
    assert window.has_source_timing is True


def test_flags_are_preserved_and_source_timing_is_false_for_zero_window() -> None:
    window = TimingWindow.from_item({
        "start": 10,
        "end": 10,
        "analysis": {"is_checked": 1},
    })
    assert window.is_checked is True
    assert window.has_source_timing is False


def test_from_item_does_not_mutate_input() -> None:
    item: dict[str, Any] = {
        "start": 0,
        "end": 2000,
        "analysis": {"duration_sec": 2.0, "is_checked": False},
    }
    original = deepcopy(item)
    TimingWindow.from_item(item)
    assert item == original


def test_fractional_budget_formula() -> None:
    window = TimingWindow(1.0, 2.55, True, True)
    assert CharacterRateBudgetEstimator(13.0).max_chars(window, 1.0) == 33


def test_budget_lower_bound_is_one() -> None:
    window = TimingWindow(0.0, 0.0, False, False)
    assert CharacterRateBudgetEstimator(13.0).max_chars(window, 0.0) == 1


def _model(**overrides: Any) -> CalibratedDurationModel:
    values: dict[str, Any] = {
        "intercept_sec": 0.5, "seconds_per_budget_char": 0.1,
        "seconds_per_punctuation": 0.2, "measurement": POST_SILENCE_MEASUREMENT_LABEL,
        "voice": "voice", "rate": "+0%", "sample_count": 30, "episode_count": 2,
        "validation_method": "leave_one_episode_out", "validation_mae_sec": 0.1,
        "validation_rmse_sec": 0.2, "validation_r_squared": 0.8,
    }
    values.update(overrides)
    return CalibratedDurationModel(**values)


def test_punctuation_helper_and_measurement_label_are_shared() -> None:
    assert count_ascii_punctuation(".,!?:;...") == 9
    assert calibration_punctuation is count_ascii_punctuation
    assert CALIBRATION_LABEL == POST_SILENCE_MEASUREMENT_LABEL


def test_duration_prediction_string_and_list_text() -> None:
    model = _model()
    assert model.predict_seconds("Hi!") == pytest.approx(1.0)
    assert model.predict_seconds(["Hi", "there!"]) == pytest.approx(1.6)


def _profile() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile_type": "calibrated_post_silence_duration",
        "measurement": POST_SILENCE_MEASUREMENT_LABEL,
        "voice": "ru-RU-DmitryNeural", "rate": "+0%",
        "coefficients": {"intercept_sec": 0.1, "seconds_per_budget_char": 0.05,
                         "seconds_per_punctuation": 0.02},
        "training": {"sample_count": 30, "episode_count": 2},
        "validation": {"method": "leave_one_episode_out", "mae_sec": 0.1,
                        "rmse_sec": 0.2, "r_squared": 0.8},
    }


def test_strict_profile_load_returns_all_fields(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_profile()), encoding="utf-8")
    model = load_calibrated_duration_profile(path)
    assert (model.measurement, model.voice, model.rate) == (
        POST_SILENCE_MEASUREMENT_LABEL, "ru-RU-DmitryNeural", "+0%"
    )
    assert (model.intercept_sec, model.seconds_per_budget_char, model.seconds_per_punctuation) == (0.1, 0.05, 0.02)
    assert (model.sample_count, model.episode_count) == (30, 2)
    assert (model.validation_method, model.validation_mae_sec, model.validation_rmse_sec, model.validation_r_squared) == (
        "leave_one_episode_out", 0.1, 0.2, 0.8
    )


@pytest.mark.parametrize("change", [
    lambda value: ([], True),
    lambda value: value.pop("voice"),
    lambda value: value.update(extra=True),
    lambda value: value.update(schema_version=2),
    lambda value: value.update(schema_version=1.0),
    lambda value: value.update(profile_type="other"),
    lambda value: value.update(measurement="raw_edge_mp3_before_ffmpeg_and_silence_reduction"),
    lambda value: value.update(voice=""),
    lambda value: value.update(rate=3),
    lambda value: value["coefficients"].update(intercept_sec=True),
    lambda value: value["coefficients"].update(seconds_per_budget_char=float("nan")),
    lambda value: value["coefficients"].update(seconds_per_punctuation=-1),
    lambda value: value["training"].update(sample_count=True),
    lambda value: value["training"].update(episode_count=1.5),
    lambda value: value["training"].update(sample_count=29),
    lambda value: value["training"].update(episode_count=1),
    lambda value: value["validation"].update(method="in_sample"),
    lambda value: value["validation"].update(mae_sec=-1),
    lambda value: value["validation"].update(rmse_sec=float("inf")),
    lambda value: value["validation"].update(r_squared=1.1),
])
def test_strict_profile_rejects_invalid_shapes_and_values(tmp_path: Path, change: Any) -> None:
    value = _profile()
    result = change(value)
    if isinstance(result, tuple) and result[1] is True:
        value = result[0]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(value, allow_nan=True), encoding="utf-8")
    with pytest.raises(ValueError):
        load_calibrated_duration_profile(path)


@pytest.mark.parametrize("field,value", [
    ("sample_count", 30.0), ("sample_count", True), ("episode_count", 2.0),
    ("episode_count", True), ("voice", 1), ("rate", None),
    ("validation_mae_sec", True), ("validation_rmse_sec", float("nan")),
    ("validation_r_squared", 2.0),
])
def test_direct_model_rejects_invalid_provenance(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        _model(**{field: value})


@pytest.mark.parametrize("field,value", [
    ("intercept_sec", True), ("intercept_sec", "0.1"), ("intercept_sec", float("nan")),
    ("seconds_per_budget_char", False), ("seconds_per_budget_char", "0.1"),
    ("seconds_per_budget_char", float("inf")), ("seconds_per_punctuation", True),
    ("seconds_per_punctuation", "0.1"), ("seconds_per_punctuation", float("nan")),
])
def test_direct_model_rejects_invalid_coefficients_with_value_error(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        _model(**{field: value})


@pytest.mark.parametrize("ratio", [True, 0, -1, float("nan"), float("inf")])
def test_duration_selection_policy_rejects_invalid_fit_ratio(ratio: Any) -> None:
    with pytest.raises(ValueError):
        DurationSelectionPolicy(_model(), ratio)


def test_duration_selection_policy_accepts_default_ratio() -> None:
    assert DurationSelectionPolicy(_model()).fit_ratio == 1.0
