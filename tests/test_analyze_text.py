from __future__ import annotations

import json

import pytest

from utils.analyze_text import analyze_subtitles, main
from utils.shortening_plan import CuePlan, PlannerConfig, analysis_for_text


def test_analyzer_checks_short_cues_and_preserves_source_fields() -> None:
    items = [
        {"index": 41, "start": 0, "end": 100, "text": "Да."},
        {"index": 42, "start": 300, "end": 400, "text": "Пошли, пошли, пошли."},
        {"index": 43, "start": "bad", "end": 500, "text": "ok", "unknown": True},
    ]
    result = analyze_subtitles(items, 13.0, 1.5)
    assert result["checked_count"] == 2
    assert items[0]["analysis"]["is_short_segment"] is True
    assert items[0]["analysis"]["is_checked"] is True
    assert items[0]["analysis"]["short_timing_risk"] is True
    assert items[1]["analysis"]["repetition_candidate"] is True
    assert items[1]["analysis"]["reason"] == "repetition_candidate"
    assert items[2]["analysis"]["invalid_timing"] is True
    assert items[2]["analysis"]["is_checked"] is False
    assert items[2]["unknown"] is True
    assert items[0]["index"] == 41


def test_duration_shortness_is_independent_from_word_count() -> None:
    result = analyze_subtitles(
        [{"start": 0, "end": 900, "text": "Я не знаю."}], 13.0, 1.5
    )
    analysis = result["items"][0]["analysis"]
    assert analysis["is_short_segment"] is True
    assert analysis["words_count"] == 3
    assert analysis["is_checked"] is True


def test_analyzer_fit_clears_short_risk() -> None:
    result = analyze_subtitles(
        [{"start": 0, "end": 2000, "text": "Да, да."}], 13.0, 1.5
    )
    analysis = result["items"][0]["analysis"]
    assert analysis["short_timing_risk"] is False
    assert analysis["repetition_candidate"] is True
    assert analysis["is_critical"] is False


def test_invalid_text_metadata_matches_planner_without_stringifying_none() -> None:
    items = [{"start": 0, "end": 1000, "text": None, "unknown": "kept"}]
    result = analyze_subtitles(items, 13.0, 1.5)
    analysis = result["items"][0]["analysis"]
    assert analysis["words_count"] == 0
    assert analysis["repetition_candidate"] is False
    assert analysis["reason"] == "invalid_text"
    assert analysis["is_checked"] is False
    assert items[0]["unknown"] == "kept"


def test_analysis_labels_nominal_and_profile_budgets_and_requires_same_text_for_audio_fit() -> None:
    config = PlannerConfig()
    nominal = CuePlan("cue-0", 0, "Да.", 0, 1000, 1000, 13, None, 0.0, "fits", "fits")
    profile = CuePlan(
        "cue-1", 1, "Да.", 0, 1000, 1000, 13, None, 0.0, "fits", "fits",
        estimate_source="measured_budget_estimate",
    )
    actual = CuePlan(
        "cue-2", 2, "Да.", 0, 1000, 1000, 13, None, 0.0, "fits", "fits",
        estimate_source="vosk_measurement",
    )
    assert analysis_for_text(nominal, nominal.text, config)["budget_basis"] == "stricter_character_budget"
    assert analysis_for_text(profile, profile.text, config)["budget_basis"] == "measured_budget_estimate"
    changed = analysis_for_text(actual, "Пошли, пошли, пошли.", config)
    assert changed["estimate_source"] == "nominal_cps_estimate"
    assert changed["reason"] != "actual_fit"
    assert changed["is_critical"] is True


def test_analyzer_cli_rejects_bad_speed_as_argparse_error_and_preserves_input(tmp_path) -> None:
    source = tmp_path / "source.json"
    original = [{"start": 0, "end": 1000, "text": "Да."}]
    source.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        main([str(source), "--threshold", "1.6"])
    assert raised.value.code == 2
    assert json.loads(source.read_text(encoding="utf-8")) == original
