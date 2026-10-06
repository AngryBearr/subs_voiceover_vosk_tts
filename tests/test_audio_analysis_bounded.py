"""Offline tempo tests: mocked durations and ffmpeg, no audio encoding or TTS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from utils import audio_analysis as aa


def fixture_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duration: float) -> tuple[list[dict[str, Any]], list[float]]:
    items = [{"index": 7, "start": 0, "end": 1000, "text": "Да."}]
    (tmp_path / "7.wav").write_bytes(b"raw fixture, not an audio file")
    durations = {"7.wav": duration}
    factors: list[float] = []

    def measure(path: Path) -> float:
        return durations[path.name]

    def tempo(src: Path, dst: Path, *, speed_factor: float, **kwargs: Any) -> None:
        factors.append(speed_factor)
        durations[dst.name] = durations[src.name] / speed_factor
        dst.write_bytes(b"mocked tempo output")

    monkeypatch.setattr(aa, "_wav_duration_sec", measure)
    monkeypatch.setattr(aa, "_speedup_with_ffmpeg", tempo)
    monkeypatch.setattr(aa, "_ffmpeg_available", lambda: True)
    monkeypatch.setattr(aa, "_speedup_naive_resample", lambda *args, **kwargs: pytest.fail("Pitch fallback prohibited"))
    return items, factors


@pytest.mark.parametrize("duration,base,cap,status,factor", [
    (0.99, 1.25, 1.5, "fits", None),
    (1.05, 1.25, 1.5, "accelerated_fits", 1.05),
    (1.20, 1.25, 1.5, "accelerated_fits", 1.2),
    (1.21, 1.25, 1.5, "unresolved", None),
    (1.04, 1.25, 1.3, "accelerated_fits", 1.04),
    (1.05, 1.25, 1.3, "unresolved", None),
    (1.01, 1.5, 1.5, "unresolved", None),
])
def test_only_needed_factor_with_one_total_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duration: float, base: float,
    cap: float, status: str, factor: float | None,
) -> None:
    items, factors = fixture_audio(tmp_path, monkeypatch, duration)
    rows = aa.check_and_speedup_segments(items, tmp_path, synthesis_rate=base, max_total_speed=cap)
    assert rows[0].status == status
    assert rows[0].synthesis_rate * rows[0].effective_factor <= cap
    assert factors == ([] if factor is None else [factor])
    report = json.loads((tmp_path / "audio_timing.json").read_text())
    assert report["failed_count"] == int(status == "unresolved")
    assert report["segments"][0]["duration_after_sec"] == rows[0].duration_after_sec
    assert aa.check_and_speedup_segments(items, tmp_path, synthesis_rate=base, max_total_speed=cap) == rows
    assert factors == ([] if factor is None else [factor])  # No cumulative tempo.


def test_uses_exact_shared_gap_window_not_analyzer_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items, factors = fixture_audio(tmp_path, monkeypatch, 1.7)
    items[0]["analysis"] = {"extended_duration_sec": 99}
    items.append({"index": 12, "start": 1800, "end": 2000, "text": "Да."})
    rows = aa.check_and_speedup_segments(items, tmp_path, synthesis_rate=1.25, max_total_speed=1.5)
    assert rows[0].allowed_duration_sec == 1.6  # 1000 + (800 - 200), no extra reserve.
    assert factors == [1.7 / 1.6]
    assert rows[1].status == "missing"


@pytest.mark.parametrize("failure", ["missing_ffmpeg", "failed_ffmpeg", "nonfinite", "rounding"])
def test_failures_never_truncate_or_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    items, factors = fixture_audio(tmp_path, monkeypatch, 1.05)
    original = (tmp_path / "7.wav").read_bytes()
    if failure == "missing_ffmpeg":
        monkeypatch.setattr(aa, "_ffmpeg_available", lambda: False)
    elif failure == "failed_ffmpeg":
        def fail(*args: Any, **kwargs: Any) -> None:
            args[1].write_bytes(b"partial failed ffmpeg output")
            raise RuntimeError("ffmpeg test failure")
        monkeypatch.setattr(aa, "_speedup_with_ffmpeg", fail)
    elif failure == "nonfinite":
        monkeypatch.setattr(aa, "_wav_duration_sec", lambda path: float("nan"))
    else:
        monkeypatch.setattr(aa, "_wav_duration_sec", lambda path: 1.05 if path.name == "7.wav" else 1.00001)
    row = aa.check_and_speedup_segments(items, tmp_path, synthesis_rate=1.25, max_total_speed=1.5)[0]
    assert row.status == ("unresolved" if failure == "rounding" else "invalid")
    if failure != "rounding":
        assert (tmp_path / "7.wav").read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["7.wav", "audio_timing.json"]


def test_stale_hash_pending_report_and_changed_rate_reject_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    items, factors = fixture_audio(tmp_path, monkeypatch, 1.05)
    aa.check_and_speedup_segments(items, tmp_path, synthesis_rate=1.25, max_total_speed=1.5)
    with pytest.raises(ValueError, match="stale or interrupted"):
        aa.check_and_speedup_segments(items, tmp_path, synthesis_rate=1.5, max_total_speed=1.5)
    report_path = tmp_path / "audio_timing.json"
    report = json.loads(report_path.read_text())
    report["state"] = "pending"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="stale or interrupted"):
        aa.check_and_speedup_segments(items, tmp_path, synthesis_rate=1.25, max_total_speed=1.5)
    report["state"] = "complete"
    report_path.write_text(json.dumps(report))
    (tmp_path / "7.wav").write_bytes(b"externally tempo-tagged replacement")
    with pytest.raises(ValueError, match="stale or interrupted"):
        aa.check_and_speedup_segments(items, tmp_path, synthesis_rate=1.25, max_total_speed=1.5)
    assert factors == [1.05]


def test_short_window_frame_boundary_and_disabled_tempo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items, factors = fixture_audio(tmp_path, monkeypatch, 0.10001)
    items[0]["end"] = 100
    row = aa.check_and_speedup_segments(items, tmp_path, synthesis_rate=1.25, max_total_speed=1.5, allow_speedup=False)[0]
    assert row.status == "unresolved" and row.required_factor > 1
    assert factors == []


def test_invalid_timing_missing_rate_and_duplicate_indices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items, _ = fixture_audio(tmp_path, monkeypatch, 1)
    with pytest.raises(ValueError, match="actual synthesis_rate"):
        aa.check_and_speedup_segments(items, tmp_path, max_total_speed=1.5)
    with pytest.raises(ValueError, match="unique integer"):
        aa.check_and_speedup_segments(items * 2, tmp_path, synthesis_rate=1.25, max_total_speed=1.5)
    items[0]["end"] = 0
    assert aa.check_and_speedup_segments(items, tmp_path, synthesis_rate=1.25, max_total_speed=1.5)[0].status == "invalid"


def test_short_frame_overrun_gets_only_needed_tempo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items, factors = fixture_audio(tmp_path, monkeypatch, 0.10001)
    items[0]["end"] = 100
    row = aa.check_and_speedup_segments(items, tmp_path, synthesis_rate=1.25, max_total_speed=1.5)[0]
    assert row.status == "accelerated_fits" and factors == [0.10001 / 0.1]


def test_legacy_utility_still_uses_opt_in_fixed_factor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    items, factors = fixture_audio(tmp_path, monkeypatch, 1.2)
    items[0]["analysis"] = {"duration_sec": 1}
    monkeypatch.setattr(aa.sf, "info", lambda path: SimpleNamespace(samplerate=22050))
    result = aa.check_and_speedup_segments(items, tmp_path)[0]
    assert result.speedup_applied and factors == [1.1]


@pytest.mark.parametrize("outputs,expected_status,expected_calls", [
    ([0.999], "accelerated_fits", 1),
    ([1.010, 0.989], "accelerated_fits", 2),
    ([1.010, 1.001], "unresolved", 2),
    ([1.021], "unresolved", 1),
])
def test_bounded_render_measures_and_reuses_only_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outputs: list[float],
    expected_status: str, expected_calls: int,
) -> None:
    items, _ = fixture_audio(tmp_path, monkeypatch, 1.1)
    source = tmp_path / "7.wav"
    original = source.read_bytes()
    calls: list[tuple[Path, float]] = []

    def render(src: Path, dst: Path, *, speed_factor: float) -> None:
        assert src == source and src.read_bytes() == original
        calls.append((src, speed_factor))
        dst.write_bytes(b"processed output")

    monkeypatch.setattr(aa, "_speedup_with_ffmpeg", render)
    monkeypatch.setattr(aa, "_wav_duration_sec", lambda p: 1.1 if p == source else outputs[len(calls) - 1])
    row = aa.check_and_speedup_segments(items, tmp_path, synthesis_rate=1.25, max_total_speed=1.5)[0]
    assert row.status == expected_status
    assert row.render_attempt_count == len(calls) == expected_calls
    assert row.duration_after_sec == outputs[-1]
    assert row.effective_factor == calls[-1][1]  # Whole factor, NOT product of two attempts.
    assert row.effective_total_speed == 1.25 * row.effective_factor <= 1.5
    assert row.render_guard_sec == pytest.approx(0.020 if expected_calls == 2 else 0)
    assert row.allowed_duration_sec == 1  # Guard is not a physical window change.
    assert aa.check_and_speedup_segments(items, tmp_path, synthesis_rate=1.25, max_total_speed=1.5)[0] == row
    assert len(calls) == expected_calls


@pytest.mark.parametrize("failure", ["exception", "missing_output"])
def test_failed_second_render_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    items, _ = fixture_audio(tmp_path, monkeypatch, 1.1)
    source = tmp_path / "7.wav"
    original = source.read_bytes()
    calls = 0

    def render(src: Path, dst: Path, *, speed_factor: float) -> None:
        nonlocal calls
        calls += 1
        assert src.read_bytes() == original
        if calls == 1:
            dst.write_bytes(b"first processed output")
        elif failure == "exception":
            dst.write_bytes(b"partial failed output")
            raise RuntimeError("second ffmpeg failure")

    def measure(path: Path) -> float:
        if path == source:
            return 1.1
        if not path.exists():
            raise FileNotFoundError(path)
        return 1.010

    monkeypatch.setattr(aa, "_speedup_with_ffmpeg", render)
    monkeypatch.setattr(aa, "_wav_duration_sec", measure)
    row = aa.check_and_speedup_segments(items, tmp_path, synthesis_rate=1.25, max_total_speed=1.5)[0]
    assert row.status == "invalid" and row.error
    assert row.render_attempt_count == calls == 2
    assert row.render_attempts[-1]["status"] == "failed"
    assert not row.speedup_applied and row.effective_factor == 1
    assert source.read_bytes() == original


@pytest.mark.parametrize("allowed,source_duration,cap", [(1.0, 1.19, 1.2), (0.030, 0.031, 1.2)])
def test_correction_clamps_cap_and_short_window_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allowed: float, source_duration: float, cap: float,
) -> None:
    source, output = tmp_path / "source.wav", tmp_path / "output.wav"
    source.write_bytes(b"unchanged source")
    factors: list[float] = []

    def render(src: Path, dst: Path, *, speed_factor: float) -> None:
        assert src == source
        factors.append(speed_factor)
        dst.write_bytes(b"output")

    monkeypatch.setattr(aa, "_ffmpeg_available", lambda: True)
    monkeypatch.setattr(aa, "_speedup_with_ffmpeg", render)
    monkeypatch.setattr(aa, "_wav_duration_sec", lambda p: source_duration if p == source else allowed + 0.010)
    attempts: list[dict[str, Any]] = []
    factor, after = aa.render_bounded_tempo(source, output, allowed_duration_sec=allowed,
                                          max_factor=cap, attempts=attempts)
    assert len(factors) == 2 and factor == cap and 1.25 * factor <= 1.5
    assert after > allowed  # No forced success at the cap.
    assert attempts[-1]["target_headroom_sec"] == pytest.approx(min(0.020, allowed * 0.5))
    assert source.read_bytes() == b"unchanged source"
