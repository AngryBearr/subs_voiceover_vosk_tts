"""Tests for measured, diagnostic timing profiles."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from utils.shorten_subtitles import main as shortening_main
from utils.shortening_plan import PlannerConfig, plan_subtitles
from utils.shortening_timing import apply_timing_profile, build_timing_profile, main


def _reports(*, final_duration: float = 2.0, available_ms: int = 1000) -> tuple[dict[str, object], dict[str, object]]:
    trial = {
        "source_manifest": {"sha256": "input-sha"},
        "records": [
            {
                "cue_id": "cue-0",
                "original": "abcdefghij",
                "final_text": "abcdef",
                "budget": {"available_ms": available_ms, "max_chars": 13},
            }
        ],
    }
    audio = {
        "source_report": {"absolute_path": "/tmp/trial-report.json", "sha256": "trial-report-sha"},
        "source_input_sha256": "input-sha",
        "model_path": "model",
        "voice": 8,
        "speech_rate": 1.5,
        "preprocessing": "raw_no_ruaccent",
        "records": [
            {
                "cue_id": "cue-0",
                "available_ms": available_ms,
                "original": {"duration_sec": 1.5},
                "final": {"duration_sec": final_duration},
            }
        ],
    }
    return audio, trial


def test_decimal_floor_after_margin_without_second_speed_factor() -> None:
    audio, trial = _reports(final_duration=2.0)
    profile = build_timing_profile(audio, trial, margin=0.95)
    # min(10 * 1 / 1.5, 6 * 1 / 2, 13) * .95 = 2.85 -> 2.
    assert profile["overrides"]["cue-0"]["max_chars"] == 2  # type: ignore[index]
    assert profile["expected_speech_rate"] == 1.5


def test_voice_zero_is_valid_but_bool_and_non_integer_are_rejected() -> None:
    audio, trial = _reports(final_duration=1.0)
    audio["voice"] = 0
    assert build_timing_profile(audio, trial)["voice"] == 0
    for voice in (False, 8.0):
        invalid_audio = copy.deepcopy(audio)
        invalid_audio["voice"] = voice
        with pytest.raises(ValueError, match="voice"):
            build_timing_profile(invalid_audio, trial)


def test_new_profile_requires_input_hash_and_candidate_integrity() -> None:
    audio, trial = _reports()
    missing_hash = copy.deepcopy(audio)
    del missing_hash["source_input_sha256"]
    with pytest.raises(ValueError, match="source_input_sha256"):
        build_timing_profile(missing_hash, trial)
    wrong_hash = copy.deepcopy(audio)
    wrong_hash["source_input_sha256"] = "wrong-input-sha"
    with pytest.raises(ValueError, match="source_input_sha256"):
        build_timing_profile(wrong_hash, trial)

    wrong_duration_candidate = copy.deepcopy(audio)
    wrong_duration_candidate["records"][0]["final_duration_sec"] = 1.0  # type: ignore[index]
    with pytest.raises(ValueError, match="does not match"):
        build_timing_profile(wrong_duration_candidate, trial)

    wrong_window = copy.deepcopy(audio)
    wrong_window["records"][0]["available_ms"] = 900  # type: ignore[index]
    with pytest.raises(ValueError, match="trial budget"):
        build_timing_profile(wrong_window, trial)

    wrong_original = copy.deepcopy(audio)
    wrong_original["records"][0]["original_text"] = "other phrase"  # type: ignore[index]
    with pytest.raises(ValueError, match="original"):
        build_timing_profile(wrong_original, trial)
    wrong_final = copy.deepcopy(audio)
    wrong_final["records"][0]["final_text"] = "other phrase"  # type: ignore[index]
    with pytest.raises(ValueError, match="final_text"):
        build_timing_profile(wrong_final, trial)


def test_fit_is_kept_and_missing_duration_is_attention() -> None:
    audio, trial = _reports(final_duration=1.0)
    audio["records"] = [
        audio["records"][0],  # type: ignore[index]
        {"cue_id": "cue-1", "available_ms": 1000, "final": None},
    ]
    trial["records"] = [
        trial["records"][0],  # type: ignore[index]
        {"cue_id": "cue-1", "original": "x", "budget": {"available_ms": 1000, "max_chars": 13}},
    ]
    profile = build_timing_profile(audio, trial)
    assert profile["overrides"] == {}
    assert profile["kept_fit_ids"] == ["cue-0"]
    assert profile["unresolved_synthesis_ids"] == ["cue-1"]


def test_apply_rejects_hash_rate_window_and_bad_types() -> None:
    plans = plan_subtitles([{"start": 0, "end": 1000, "text": "long phrase here"}], PlannerConfig())
    profile = {
        "schema_version": 1,
        "profile_version": 1,
        "source_input_sha256": "input-sha",
        "expected_speech_rate": 1.5,
        "overrides": {"cue-0": {"max_chars": 2, "previous_max_chars": plans[0].max_chars, "available_ms": plans[0].available_ms}},
    }
    with pytest.raises(ValueError, match="hash"):
        apply_timing_profile(plans, profile, source_sha256="wrong", config=PlannerConfig())
    with pytest.raises(ValueError, match="rate"):
        apply_timing_profile(plans, {**profile, "expected_speech_rate": 1.0}, source_sha256="input-sha", config=PlannerConfig())
    with pytest.raises(ValueError, match="window"):
        apply_timing_profile(plans, {**profile, "overrides": {"cue-0": {**profile["overrides"]["cue-0"], "available_ms": 0}}}, source_sha256="input-sha", config=PlannerConfig())  # type: ignore[index]
    with pytest.raises(ValueError, match="max_chars"):
        apply_timing_profile(plans, {**profile, "overrides": {"cue-0": {**profile["overrides"]["cue-0"], "max_chars": True}}}, source_sha256="input-sha", config=PlannerConfig())  # type: ignore[index]


def test_apply_does_not_mutate_plans_and_invalid_stays_invalid() -> None:
    plans = plan_subtitles([
        {"start": 0, "end": 1000, "text": "long phrase here"},
        {"start": 0, "end": 0, "text": "bad"},
    ], PlannerConfig())
    original = copy.deepcopy(plans)
    profile = {
        "schema_version": 1,
        "profile_version": 1,
        "source_input_sha256": "input-sha",
        "expected_speech_rate": 1.5,
        "overrides": {"cue-0": {"max_chars": 2, "previous_max_chars": plans[0].max_chars, "available_ms": plans[0].available_ms}},
    }
    changed = apply_timing_profile(plans, profile, source_sha256="input-sha", config=PlannerConfig())
    assert plans == original
    assert changed[0].max_chars == 2
    assert changed[1].status == "invalid"


def test_profile_cli_does_not_overwrite(tmp_path: Path) -> None:
    audio, trial = _reports(final_duration=1.0)
    audio_path = tmp_path / "audio.json"
    trial_path = tmp_path / "trial.json"
    output = tmp_path / "profile.json"
    audio_path.write_text(json.dumps(audio), encoding="utf-8")
    trial_path.write_text(json.dumps(trial), encoding="utf-8")
    audio["source_report"] = {
        "absolute_path": str(trial_path.resolve()),
        "sha256": hashlib.sha256(trial_path.read_bytes()).hexdigest(),
    }
    audio_path.write_text(json.dumps(audio), encoding="utf-8")
    assert main([str(audio_path), "--trial-report", str(trial_path), "--output", str(output)]) == 0
    with pytest.raises(FileExistsError):
        main([str(audio_path), "--trial-report", str(trial_path), "--output", str(output)])


def test_profile_cli_rejects_different_trial_path_and_hash(tmp_path: Path) -> None:
    audio, trial = _reports(final_duration=1.0)
    audio_path = tmp_path / "audio.json"
    trial_path = tmp_path / "trial.json"
    other_trial_path = tmp_path / "other-trial.json"
    output = tmp_path / "profile.json"
    trial_path.write_text(json.dumps(trial), encoding="utf-8")
    other_trial_path.write_text(json.dumps(trial), encoding="utf-8")
    audio["source_report"] = {
        "absolute_path": str(other_trial_path.resolve()),
        "sha256": hashlib.sha256(trial_path.read_bytes()).hexdigest(),
    }
    audio_path.write_text(json.dumps(audio), encoding="utf-8")
    with pytest.raises(ValueError, match="absolute_path"):
        main([str(audio_path), "--trial-report", str(trial_path), "--output", str(output)])

    audio["source_report"] = {
        "absolute_path": str(trial_path.resolve()),
        "sha256": "wrong-hash",
    }
    audio_path.write_text(json.dumps(audio), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        main([str(audio_path), "--trial-report", str(trial_path), "--output", str(output)])


def test_frozen_selection_ids_and_original_cli(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps([
        {"start": 0, "end": 100, "text": "one two three"},
        {"start": 200, "end": 300, "text": "one two three"},
    ]), encoding="utf-8")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "sha256": source_hash,
        "planner": {"cps": 13.0, "gap_ms": 200, "max_speed": 1.5},
        "selection": {"selected_ids": ["cue-0"]},
    }), encoding="utf-8")
    output = tmp_path / "dry"
    assert shortening_main([str(source), "--output-dir", str(output), "--selection-manifest", str(selection)]) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selection"]["selected_ids"] == ["cue-0"]
