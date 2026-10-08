"""Tests for the isolated shortening audio timing verifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from utils.shortening_audio import verify_audio


def _report(*records: dict[str, Any]) -> dict[str, Any]:
    return {"records": list(records), "source_manifest": {"absolute_path": "/input.json", "sha256": "source"}}


def _record(
    position: int,
    status: str,
    *,
    original: str = "original",
    final_text: str | None = "final",
    available_ms: int = 1000,
    start: int = 100,
    end: int = 600,
    cue_id: str | None = None,
) -> dict[str, Any]:
    return {
        "cue_id": cue_id or f"cue-{position}",
        "position": position,
        "original": original,
        "candidate": final_text,
        "final_text": final_text,
        "status": status,
        "budget": {
            "available_ms": available_ms,
            "max_chars": 100,
            "start": start,
            "end": end,
        },
    }


def _synthesizer(durations: dict[str, float], calls: list[dict[str, Any]] | None = None):
    def synthesize(*, text: str, output_path: str, voice: int, speech_rate: float) -> Path:
        if calls is not None:
            calls.append({"text": text, "voice": voice, "speech_rate": speech_rate})
        duration = durations[text]
        sample_rate = 1000
        sf.write(output_path, np.zeros(max(1, round(duration * sample_rate)), dtype=np.float32), sample_rate, subtype="PCM_16")
        return Path(output_path)

    return synthesize


def test_boundary_overshoot_and_raw_metadata(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    report = _report(
        _record(0, "accepted", original="orig", final_text="boundary", available_ms=1000),
        _record(1, "accepted", original="too-long", final_text="over", available_ms=1000),
    )
    durations = {"orig": 0.4, "boundary": 1.0, "too-long": 0.3, "over": 1.001}
    result = verify_audio(report, tmp_path / "audio", model_path=model, synthesis_fn=_synthesizer(durations))

    assert result["records"][0]["fits"] is True
    assert result["records"][0]["overflow_sec"] == 0.0
    assert result["records"][1]["fits"] is False
    assert result["records"][1]["overflow_sec"] == pytest.approx(0.001, abs=0.002)
    assert result["summary"]["actual_fits"] == 1
    assert result["preprocessing"] == "vosk_punctuation_v1_no_ruaccent"
    assert result["synthetic_fallback"] is False
    assert result["speed_postprocess"] is False
    assert Path(result["records"][0]["original"]["path"]).is_relative_to(Path("."))
    assert report["records"][0]["status"] == "accepted"


def test_rate_is_capped_and_same_voice_rate_is_used(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    calls: list[dict[str, Any]] = []
    report = _report(_record(0, "accepted", original="a", final_text="b"))
    verify_audio(
        report,
        tmp_path / "audio",
        model_path=model,
        voice=8,
        speech_rate=2.0,
        synthesis_fn=_synthesizer({"a": 0.1, "b": 0.1}, calls),
    )
    assert calls == [
        {"text": "a", "voice": 8, "speech_rate": 1.5},
        {"text": "b", "voice": 8, "speech_rate": 1.5},
    ]


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_rate_is_rejected(tmp_path: Path, rate: float) -> None:
    model = tmp_path / "model"
    model.mkdir()
    with pytest.raises(ValueError):
        verify_audio(_report(), tmp_path / "audio", model_path=model, speech_rate=rate, synthesis_fn=lambda **_: None)


def test_short_timing_synthesizes_original_only_and_counts_overlap(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    calls: list[dict[str, Any]] = []
    report = _report(_record(0, "short_timing", original="short", final_text=None, available_ms=500))
    result = verify_audio(
        report,
        tmp_path / "audio",
        model_path=model,
        synthesis_fn=_synthesizer({"short": 0.6}, calls),
    )
    row = result["records"][0]
    assert [call["text"] for call in calls] == ["short"]
    assert row["final"] is None
    assert "final_words_count" not in row
    assert "final_chars_count" not in row
    assert row["fits"] is False
    assert row["timing_action"] == "short_phrase_overlap_pending"
    assert result["summary"]["short_overlap"] == 1


def test_short_final_overflow_is_not_counted_as_a_fit(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    result = verify_audio(
        _report(_record(0, "accepted", original="original", final_text="one two", available_ms=100)),
        tmp_path / "audio",
        model_path=model,
        synthesis_fn=_synthesizer({"original": 0.01, "one two": 0.1209}),
    )

    row = result["records"][0]
    assert row["status"] == "accepted"
    assert row["final_words_count"] == 2
    assert row["final_chars_count"] == len("one two")
    assert row["final_fits"] is False
    assert row["fits"] is False
    assert row["timing_action"] == "short_phrase_overlap_pending"
    assert result["summary"]["actual_fits"] == 0
    assert result["summary"]["short_final_overlap_count"] == 1
    assert result["summary"]["long_final_overflow_count"] == 0


def test_long_final_overflow_needs_timing_revision(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    result = verify_audio(
        _report(_record(0, "accepted", original="original", final_text="one two three four", available_ms=100)),
        tmp_path / "audio",
        model_path=model,
        synthesis_fn=_synthesizer({"original": 0.01, "one two three four": 0.1209}),
    )

    row = result["records"][0]
    assert row["final_words_count"] == 4
    assert row["timing_action"] == "needs_timing_revision"
    assert row["final_fits"] is False
    assert result["summary"]["actual_fits"] == 0
    assert result["summary"]["short_final_overlap_count"] == 0
    assert result["summary"]["long_final_overflow_count"] == 1


def test_four_word_repetition_is_not_short_overlap(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    result = verify_audio(
        _report(_record(0, "accepted", original="original", final_text="go go go go", available_ms=100)),
        tmp_path / "audio",
        model_path=model,
        synthesis_fn=_synthesizer({"original": 0.01, "go go go go": 0.1209}),
    )

    row = result["records"][0]
    assert row["final_words_count"] == 4
    assert row["timing_action"] == "needs_timing_revision"
    assert result["summary"]["short_final_overlap_count"] == 0
    assert result["summary"]["long_final_overflow_count"] == 1


@pytest.mark.parametrize(
    ("final_text", "action"),
    [("go go go", "short_phrase_overlap_pending"), ("go go go go", "needs_timing_revision")],
)
def test_repetition_overlap_boundary_is_three_words(tmp_path: Path, final_text: str, action: str) -> None:
    model = tmp_path / "model"
    model.mkdir()
    result = verify_audio(
        _report(_record(0, "accepted", original="original", final_text=final_text, available_ms=100)),
        tmp_path / "audio",
        model_path=model,
        synthesis_fn=_synthesizer({"original": 0.01, final_text: 0.1209}),
    )
    assert result["records"][0]["timing_action"] == action


def test_excludes_manual_and_caps_by_source_position(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    calls: list[dict[str, Any]] = []
    report = _report(
        _record(10, "accepted", original="late", final_text="late-f"),
        _record(2, "manual", original="manual", final_text="manual-f"),
        _record(5, "accepted", original="first", final_text="first-f"),
        _record(1, "pending", original="pending", final_text="pending-f"),
    )
    result = verify_audio(
        report,
        tmp_path / "audio",
        model_path=model,
        limit=1,
        synthesis_fn=_synthesizer({"first": 0.1, "first-f": 0.1, "late": 0.1, "late-f": 0.1}, calls),
    )
    assert [row["position"] for row in result["records"]] == [5]
    assert len(calls) == 2


def test_output_dir_must_be_new(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "audio"
    output.mkdir()
    with pytest.raises(FileExistsError):
        verify_audio(_report(), output, model_path=model, synthesis_fn=lambda **_: None)


def test_exception_is_type_only_and_checkpoint_keeps_prior_record(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "audio"
    calls = 0

    def synthesize(*, text: str, output_path: str, **_: Any) -> Path:
        nonlocal calls
        calls += 1
        if text == "bad":
            raise RuntimeError("secret details must not be logged")
        sf.write(output_path, np.zeros(100, dtype=np.float32), 1000, subtype="PCM_16")
        return Path(output_path)

    result = verify_audio(
        _report(
            _record(0, "accepted", original="good", final_text="good-final"),
            _record(1, "accepted", original="bad", final_text="bad-final"),
        ),
        output,
        model_path=model,
        synthesis_fn=synthesize,
    )
    checkpoint = json.loads((output / "audio_report.json").read_text(encoding="utf-8"))
    assert calls == 3
    assert checkpoint["records"][0]["fits"] is True
    assert checkpoint["records"][1]["error"] == {
        "type": "RuntimeError",
        "message": "secret details must not be logged",
        "cause_type": None,
        "cause_message": None,
    }
    assert checkpoint["records"][1]["error"]["message"] == "secret details must not be logged"
    assert "Traceback" not in json.dumps(checkpoint)
    assert result["summary"]["fail"] == 1


def test_same_original_and_candidate_reuse_one_physical_wav(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    calls: list[dict[str, Any]] = []
    report = _report(_record(0, "accepted", original="same", final_text="same"))
    result = verify_audio(
        report,
        tmp_path / "audio",
        model_path=model,
        synthesis_fn=_synthesizer({"same": 0.2}, calls),
    )
    row = result["records"][0]
    assert len(calls) == 1
    assert row["original"]["path"] == row["final"]["path"]
    assert row["final"]["reused"] is True
    assert result["summary"]["physical_file_count"] == 1


def test_punctuation_mode_preserves_display_budget_and_records_tts_text(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    original = "  Привет  123…  "
    final = "Привет 123..."
    report = _report(_record(0, "accepted", original=original, final_text=final))
    calls: list[dict[str, Any]] = []
    result = verify_audio(
        report,
        tmp_path / "audio",
        model_path=model,
        synthesis_fn=_synthesizer({"Привет 123...": 0.1}, calls),
    )

    row = result["records"][0]
    assert row["original_text"] == original
    assert row["final_text"] == final
    assert row["original"]["display_text"] == original
    assert row["original"]["tts_text"] == "Привет 123..."
    assert row["original"]["was_normalized"] is True
    assert row["final"]["display_text"] == final
    assert row["final"]["tts_text"] == final
    assert row["final"]["was_normalized"] is False
    assert row["available_ms"] == 1000.0
    assert len(calls) == 1
    assert result["summary"]["physical_file_count"] == 1


def test_raw_mode_keeps_legacy_preprocessing_and_text(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    display = "Original… text"
    calls: list[dict[str, Any]] = []
    result = verify_audio(
        _report(_record(0, "short_timing", original=display, final_text=None)),
        tmp_path / "audio",
        model_path=model,
        text_mode="raw",
        synthesis_fn=_synthesizer({display: 0.1}, calls),
    )

    row = result["records"][0]
    assert result["preprocessing"] == "raw_no_ruaccent"
    assert calls[0]["text"] == display
    assert row["original"]["tts_text"] == display
    assert row["original"]["was_normalized"] is False


def test_final_error_keeps_original_audio_and_cause_details(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    cause = ValueError("Bearer abc sk-test-secret")

    def synthesize(*, text: str, output_path: str, **_: Any) -> Path:
        if text == "final":
            raise RuntimeError("synthesis failed") from cause
        sf.write(output_path, np.zeros(100, dtype=np.float32), 1000, subtype="PCM_16")
        return Path(output_path)

    result = verify_audio(
        _report(_record(0, "accepted", original="original", final_text="final")),
        tmp_path / "audio",
        model_path=model,
        synthesis_fn=synthesize,
    )

    row = result["records"][0]
    assert row["original"]["duration_sec"] == pytest.approx(0.1)
    assert row["final"] is None
    assert row["error"] == {
        "type": "RuntimeError",
        "message": "synthesis failed",
        "cause_type": "ValueError",
        "cause_message": "Bearer [REDACTED] [REDACTED]",
    }
    assert row["timing_action"] == "synthesis_error"
    assert "abc" not in json.dumps(result)
    assert "sk-test-secret" not in json.dumps(result)


def test_source_input_metadata_is_separate_from_source_report(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    report = _report(_record(0, "short_timing", original="text", final_text=None))
    report["source_report"] = {"absolute_path": "/trial-report.json", "sha256": "trial"}
    result = verify_audio(
        report,
        tmp_path / "audio",
        model_path=model,
        synthesis_fn=_synthesizer({"text": 0.1}),
    )

    assert result["source_report"] == {
        "absolute_path": "/trial-report.json",
        "sha256": "trial",
    }
    assert result["source_input_path"] == "/input.json"
    assert result["source_input_sha256"] == "source"
