"""Characterization tests for the subtitle shortening domain core."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from utils.shortening_domain import CharacterRateBudgetEstimator, TimingWindow


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
