from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile

from utils import calibrate_tts_budget as calibrator
from utils import reduce_silence
from utils.analyze_text import join_text_lines
from utils.calibrate_tts_budget import (
    build_parser,
    cache_key,
    calibrate_records,
    count_ascii_punctuation,
    default_duration_reader,
    fit_duration_model,
    post_silence_duration_reader,
    percentile,
    sanitized_voice_directory,
    summarize_records,
    validate_items,
    validate_duration_parameters,
)
from utils.text_sanitize import count_symbols_for_sps


def test_join_and_count_conventions_differ_for_spaces() -> None:
    text = join_text_lines(["one", "two"])
    assert text == "one two"
    assert len(text) == 7
    assert count_symbols_for_sps(text) == 6


def test_cache_key_and_voice_directory_are_stable() -> None:
    first = cache_key("text", "ru-RU-DmitryNeural", "+0%", "+0%", "+0Hz")
    assert first == cache_key("text", "ru-RU-DmitryNeural", "+0%", "+0%", "+0Hz")
    assert first != cache_key("other", "ru-RU-DmitryNeural", "+0%", "+0%", "+0Hz")
    assert first != cache_key("text", "en-US-Test", "+0%", "+0%", "+0Hz")
    assert first != cache_key("text", "ru-RU-DmitryNeural", "+10%", "+0%", "+0Hz")
    assert sanitized_voice_directory("ru/RU voice?") == "ru_RU_voice"


def test_percentiles_and_summary_formulas() -> None:
    records = [
        {"voice": "v", "budget_chars": n, "sps_chars": n - 1, "duration_sec": 1.0,
         "budget_cps": float(n), "sps_cps": float(n - 1)}
        for n in (1, 2, 3, 4)
    ]
    summary = summarize_records(records)["v"]
    assert summary["budget_cps"]["arithmetic_mean"] == 2.5
    assert summary["budget_cps"]["pooled_cps"] == 10.0 / 4
    assert summary["budget_cps"]["median"] == 2.5
    assert summary["budget_cps"]["p10"] == 1.3
    assert summary["budget_cps"]["p25"] == 1.75
    assert summary["sps_cps"]["p25"] == 0.75
    assert summary["exploratory_budget_cps_p25"] == 1.75
    assert summary["duration_models"]["chars_plus_punctuation"] is None
    assert "recommended_duration_profile" not in summary
    assert "exploratory_duration_profile" in summary

    thirty = [{**record, "voice": "v2"} for record in records] * 7 + [{**records[0], "voice": "v2"}]
    assert summarize_records(thirty)["v2"]["count"] == 29
    thirty.append({**records[1], "voice": "v2"})
    assert summarize_records(thirty)["v2"]["exploratory_budget_cps_p25"] is not None


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_percentile_rejects_empty_values_and_out_of_range_fraction(fraction: float) -> None:
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        percentile([1.0], fraction)


def test_default_duration_reader_reads_known_wav(tmp_path: Path) -> None:
    path = tmp_path / "known.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\0\0" * 8000)
    assert default_duration_reader(path) == 1.0


def test_post_silence_duration_reader_reduces_long_silence(tmp_path: Path) -> None:
    path = tmp_path / "silence.wav"
    sample_rate = 8000
    tone = np.full(1600, 0.25, dtype=np.float32)
    silence = np.zeros(6400, dtype=np.float32)
    soundfile.write(path, np.concatenate((tone, silence, tone)), sample_rate)
    assert post_silence_duration_reader(path) == pytest.approx(0.6, abs=0.011)


def test_post_silence_duration_reader_keeps_short_silence(tmp_path: Path) -> None:
    path = tmp_path / "short-silence.wav"
    sample_rate = 8000
    tone = np.full(1600, 0.25, dtype=np.float32)
    silence = np.zeros(3200, dtype=np.float32)
    soundfile.write(path, np.concatenate((tone, silence, tone)), sample_rate)
    assert post_silence_duration_reader(path) == pytest.approx(0.8, abs=0.011)


def test_post_silence_duration_reader_passes_pcm16_quantized_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "quantized.wav"
    source = np.array([[0.123456, -0.234567], [0.5, -0.5]], dtype=np.float32)
    soundfile.write(path, source, 8000, format="WAV", subtype="PCM_32")
    captured: list[np.ndarray] = []

    def capture(audio: np.ndarray, sample_rate: int, **kwargs: object) -> tuple[np.ndarray, int]:
        captured.append(audio.copy())
        return audio, 0

    monkeypatch.setattr(reduce_silence, "reduce_silence_audio", capture)
    assert post_silence_duration_reader(path) == pytest.approx(2 / 8000)
    assert len(captured) == 1
    assert captured[0].shape == source.shape
    assert not np.array_equal(captured[0], source)
    assert np.allclose(captured[0], np.rint(captured[0] * 32768.0) / 32768.0)


def test_ascii_punctuation_counter_counts_each_character() -> None:
    assert count_ascii_punctuation("...Hello, world! Really?;:") == 8


def _duration_record(chars: int, punctuation: int, duration: float) -> dict[str, object]:
    return {"budget_chars": chars, "punctuation_count": punctuation, "duration_sec": duration}


def test_chars_only_model_exact_recovery() -> None:
    records = [_duration_record(chars, 0, 1.75 + 0.4 * chars) for chars in (3, 7, 12, 19, 28)]
    model = fit_duration_model(records)
    assert model is not None
    assert model["model"] == "chars_only"
    assert model["formula"] == "duration_sec = intercept_sec + seconds_per_budget_char * budget_chars"
    assert model["sample_count"] == 5
    assert model["coefficients"] == pytest.approx({"intercept_sec": 1.75, "seconds_per_budget_char": 0.4})
    assert model["r_squared"] == pytest.approx(1.0)
    assert model["mae_sec"] == pytest.approx(0.0, abs=1e-12)
    assert model["rmse_sec"] == pytest.approx(0.0, abs=1e-12)


def test_chars_plus_punctuation_model_exact_recovery() -> None:
    pairs = ((3, 1), (7, 4), (12, 2), (19, 7), (28, 3), (35, 9))
    records = [_duration_record(chars, punctuation, 2.25 + 0.3 * chars + 0.17 * punctuation)
               for chars, punctuation in pairs]
    model = fit_duration_model(records, include_punctuation=True)
    assert model is not None
    assert model["model"] == "chars_plus_punctuation"
    assert model["formula"] == (
        "duration_sec = intercept_sec + seconds_per_budget_char * budget_chars"
        " + seconds_per_punctuation * punctuation_count"
    )
    assert model["sample_count"] == 6
    assert model["coefficients"] == pytest.approx({
        "intercept_sec": 2.25, "seconds_per_budget_char": 0.3, "seconds_per_punctuation": 0.17,
    })
    assert model["r_squared"] == pytest.approx(1.0)
    assert model["mae_sec"] == pytest.approx(0.0, abs=1e-12)
    assert model["rmse_sec"] == pytest.approx(0.0, abs=1e-12)


def test_noisy_residual_sign_quantiles_and_in_sample_margin() -> None:
    errors = (-0.4, 0.2, 0.8, -0.1, 0.5, 0.0, 0.3, -0.2, 0.7, 0.1,
              -0.3, 0.6, 0.4, -0.05, 0.25, 0.9, -0.15, 0.35, 0.05, 0.45,
              -0.25, 0.15, 0.55, -0.35, 0.65, 0.12, -0.08, 0.28, 0.48, -0.18)
    records = [_duration_record(chars, punctuation, 2.0 + 0.25 * chars + 0.1 * punctuation + error)
               for chars, punctuation, error in
               ((index + 3, index % 4, error) for index, error in enumerate(errors))]
    model = fit_duration_model(records, include_punctuation=True)
    assert model is not None
    coefficients = model["coefficients"]
    expected = [
        float(record["duration_sec"])
        - (coefficients["intercept_sec"] + coefficients["seconds_per_budget_char"] * float(record["budget_chars"])
           + coefficients["seconds_per_punctuation"] * float(record["punctuation_count"]))
        for record in records
    ]
    quantiles = model["residual_quantiles_sec"]
    for key, fraction in (("p10", 0.10), ("p25", 0.25), ("median", 0.50), ("p75", 0.75), ("p90", 0.90)):
        assert quantiles[key] == pytest.approx(percentile(expected, fraction))
    assert quantiles["min"] == min(expected)
    assert quantiles["max"] == max(expected)
    assert quantiles["p10"] <= quantiles["p25"] <= quantiles["median"] <= quantiles["p75"] <= quantiles["p90"]
    summary = summarize_records([{
        **record, "voice": "noisy", "budget_cps": record["budget_chars"] / record["duration_sec"],
        "sps_chars": record["budget_chars"], "sps_cps": record["budget_chars"] / record["duration_sec"],
    } for record in records])["noisy"]
    profile = summary["exploratory_duration_profile"]
    assert profile is not None
    assert profile["in_sample_p75_margin_sec"] == pytest.approx(max(0.0, percentile(expected, 0.75)))
    assert profile["margin_quantile"] == (
        "signed in-sample residual (observed_duration_sec - predicted_duration_sec) p75"
    )
    assert "recommended_duration_profile" not in summary
    assert "safety_margin_sec" not in profile
    assert "safety_quantile" not in profile
    assert "same sample" in summary["duration_profile_warning"]
    assert "not cross-validated" in summary["duration_profile_warning"]
    assert "not a production recommendation" in summary["duration_profile_warning"]
    assert "do not guarantee coverage" in summary["duration_profile_warning"]


def test_duration_profile_requires_thirty_valid_samples_and_prefers_punctuation() -> None:
    def records(count: int) -> list[dict[str, object]]:
        result = []
        for index in range(count):
            record = _duration_record(index + 2, index % 5,
                                      1.5 + 0.2 * (index + 2) + 0.08 * (index % 5))
            result.append({"voice": "v", **record, "budget_cps": record["budget_chars"] / record["duration_sec"],
                           "sps_chars": record["budget_chars"], "sps_cps": record["budget_chars"] / record["duration_sec"]})
        return result

    summary_29 = summarize_records(records(29))["v"]
    assert summary_29["exploratory_duration_profile"] is None
    summary_30 = summarize_records(records(30))["v"]
    profile = summary_30["exploratory_duration_profile"]
    plus = summary_30["duration_models"]["chars_plus_punctuation"]
    assert profile is not None and plus is not None
    assert profile["source_model"] == "chars_plus_punctuation"
    assert profile["coefficients"] == plus["coefficients"]
    assert profile["margin_quantile"] == (
        "signed in-sample residual (observed_duration_sec - predicted_duration_sec) p75"
    )
    assert "recommended_duration_profile" not in summary_30
    assert "safety_margin_sec" not in profile
    assert "safety_quantile" not in profile


def test_duration_profile_is_null_when_no_physical_model_is_valid() -> None:
    records = []
    for index in range(30):
        chars = index + 1
        punctuation = index % 5
        duration = 40.0 - chars
        records.append({
            "voice": "invalid",
            **_duration_record(chars, punctuation, duration),
            "budget_cps": chars / duration,
            "sps_chars": chars,
            "sps_cps": chars / duration,
        })

    summary = summarize_records(records)["invalid"]
    assert summary["duration_models"]["chars_only"] is not None
    assert summary["duration_models"]["chars_plus_punctuation"] is not None
    assert summary["exploratory_duration_profile"] is None
    assert "No physically valid exploratory duration model is available." in summary["duration_profile_warning"]
    assert "recommended_duration_profile" not in summary


def test_duration_profile_falls_back_to_physically_valid_chars_only_model() -> None:
    records = []
    for index in range(30):
        chars = index + 10
        punctuation = index % 5
        duration = 8.0 + 0.5 * chars - 0.1 * punctuation
        records.append({
            "voice": "fallback",
            **_duration_record(chars, punctuation, duration),
            "budget_cps": chars / duration,
            "sps_chars": chars,
            "sps_cps": chars / duration,
        })

    summary = summarize_records(records)["fallback"]
    chars_only = summary["duration_models"]["chars_only"]
    chars_plus = summary["duration_models"]["chars_plus_punctuation"]
    profile = summary["exploratory_duration_profile"]
    assert chars_only is not None and chars_plus is not None and profile is not None
    assert chars_plus["coefficients"]["seconds_per_punctuation"] < 0
    assert profile["source_model"] == "chars_only"
    assert profile["coefficients"] == chars_only["coefficients"]
    assert profile["in_sample_p75_margin_sec"] == pytest.approx(
        max(0.0, chars_only["residual_quantiles_sec"]["p75"])
    )


def test_rank_deficiency_missing_punctuation_and_invalid_rows_are_filtered() -> None:
    deficient = [_duration_record(5, punctuation, 2.0 + punctuation) for punctuation in (0, 1, 2, 3)]
    assert fit_duration_model(deficient, include_punctuation=True) is None
    missing = [_duration_record(chars, 0, 1.0 + chars * 0.2) for chars in (2, 4, 7, 11)]
    for record in missing:
        del record["punctuation_count"]
    assert fit_duration_model(missing) is not None
    assert fit_duration_model(missing, include_punctuation=True) is None
    valid = [_duration_record(2, 1, 1.4), _duration_record(4, 2, 1.8), _duration_record(7, 1, 2.4)]
    invalid = [
        {**valid[0], "duration_sec": float("nan")}, {**valid[0], "duration_sec": float("inf")},
        {**valid[0], "duration_sec": 0.0}, {**valid[0], "duration_sec": -1.0},
        {**valid[0], "budget_chars": True}, {**valid[0], "budget_chars": -1},
        {**valid[0], "budget_chars": 2.5}, {**valid[0], "punctuation_count": True},
        {**valid[0], "punctuation_count": -1}, {**valid[0], "punctuation_count": 1.5},
    ]
    model = fit_duration_model(valid + invalid, include_punctuation=True)
    assert model is not None
    assert model["sample_count"] == len(valid)


def test_runner_cache_and_multiple_voices(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []
    durations: dict[str, float] = {}
    corrupt_once = {"value": False}

    async def synth(text: str, voice: str, path: Path, rate: str, volume: str, pitch: str,
                    connect: float, receive: float) -> None:
        calls.append((text, voice))
        path.write_bytes(b"mp3")

    def duration(path: Path) -> float:
        if path.name.startswith("1-") and not calls and not corrupt_once["value"]:
            corrupt_once["value"] = True
            raise ValueError("corrupt cache")
        return durations.setdefault(str(path), 2.0)

    items = [{"index": 1, "text": ["one", "two"]}]
    first = asyncio.run(calibrate_records(items, tmp_path, ["v1", "v2"], duration_reader=duration, synthesizer=synth))
    assert len(first) == 2 and all(not item["cache_hit"] for item in first)
    calls.clear()
    second = asyncio.run(calibrate_records(items, tmp_path, ["v1", "v2"], duration_reader=duration, synthesizer=synth))
    assert len(second) == 2 and not second[0]["cache_hit"] and second[1]["cache_hit"]
    calls.clear()
    third = asyncio.run(calibrate_records(items, tmp_path, ["v1", "v2"], duration_reader=duration, synthesizer=synth))
    assert len(third) == 2 and all(item["cache_hit"] for item in third)
    assert calls == []
    assert {"index", "voice", "audio_path", "cache_hit", "text", "budget_chars", "sps_chars",
             "duration_sec", "budget_cps", "sps_cps", "punctuation_count"} <= set(third[0])
    assert all(item["raw_duration_sec"] == item["duration_sec"] for item in third)


def test_runner_separates_raw_and_selected_measurements_on_cache_hit(tmp_path: Path) -> None:
    async def synth(text: str, voice: str, path: Path, rate: str, volume: str, pitch: str,
                    connect: float, receive: float) -> None:
        path.write_bytes(b"mp3")

    raw_calls = 0
    measured_calls = 0

    def raw(path: Path) -> float:
        nonlocal raw_calls
        raw_calls += 1
        return 4.0

    def measured(path: Path) -> float:
        nonlocal measured_calls
        measured_calls += 1
        return 2.0

    items = [{"index": 1, "text": "hello"}]
    first = asyncio.run(calibrate_records(items, tmp_path, ["voice"], duration_reader=raw,
                                          measurement_reader=measured, synthesizer=synth))
    second = asyncio.run(calibrate_records(items, tmp_path, ["voice"], duration_reader=raw,
                                           measurement_reader=measured, synthesizer=synth))
    assert first[0]["raw_duration_sec"] == second[0]["raw_duration_sec"] == 4.0
    assert first[0]["duration_sec"] == second[0]["duration_sec"] == 2.0
    assert first[0]["budget_cps"] == second[0]["budget_cps"] == 2.5
    assert raw_calls == 2 and measured_calls == 2


def test_runner_retries_until_synthesizer_succeeds(tmp_path: Path) -> None:
    calls = 0

    async def synth(text: str, voice: str, path: Path, rate: str, volume: str, pitch: str,
                    connect: float, receive: float) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise OSError("offline")
        path.write_bytes(b"mp3")

    records = asyncio.run(calibrate_records(
        [{"index": 1, "text": "hello"}],
        tmp_path,
        ["voice"],
        retries=3,
        duration_reader=lambda path: 2.0,
        synthesizer=synth,
    ))

    assert calls == 3
    assert records[0]["cache_hit"] is False
    assert Path(records[0]["audio_path"]).is_file()


def test_runner_exhausted_retries_raise_and_leave_no_cache(tmp_path: Path) -> None:
    calls = 0

    async def synth(text: str, voice: str, path: Path, rate: str, volume: str, pitch: str,
                    connect: float, receive: float) -> None:
        nonlocal calls
        calls += 1
        path.write_bytes(b"partial")
        raise OSError("offline")

    with pytest.raises(RuntimeError, match="failed to synthesize index 1, voice voice"):
        asyncio.run(calibrate_records(
            [{"index": 1, "text": "hello"}],
            tmp_path,
            ["voice"],
            retries=3,
            duration_reader=lambda path: 2.0,
            synthesizer=synth,
        ))

    assert calls == 3
    assert list(tmp_path.rglob("*.mp3")) == []
    assert list(tmp_path.rglob("*.tmp")) == []


@pytest.mark.parametrize("raw", [[{"index": 1}, {"index": 1, "text": "x"}], [{"index": 1, "text": " "}]])
def test_validation_errors(raw: object) -> None:
    with pytest.raises(ValueError):
        validate_items(raw)
    with pytest.raises(ValueError):
        validate_items([{"index": 1, "text": "x"}], limit=0)


def test_cli_parser_defaults_and_flags() -> None:
    args = build_parser().parse_args(["input.json", "--output-dir", "out"])
    assert args.voices == ["ru-RU-DmitryNeural"]
    assert (args.rate, args.volume, args.pitch) == ("+0%", "+0%", "+0Hz")
    assert (args.retries, args.connect_timeout, args.receive_timeout) == (3, 30, 300)
    assert (args.duration_mode, args.silence_dbfs, args.silence_threshold_sec,
            args.silence_target_sec, args.silence_frame_ms) == ("raw_mp3", -35.0, 0.5, 0.2, 10)
    args = build_parser().parse_args(["input.json", "--output-dir", "out", "--voices", "a", "b",
                                      "--limit", "2", "--report", "report.json"])
    assert args.voices == ["a", "b"] and args.limit == 2 and args.report == Path("report.json")


def test_cli_parser_post_options_and_duration_validation() -> None:
    args = build_parser().parse_args(["input.json", "--output-dir", "out", "--duration-mode", "post_silence",
                                      "--silence-dbfs", "-40", "--silence-threshold-sec", "1",
                                      "--silence-target-sec", "0.3", "--silence-frame-ms", "20"])
    assert (args.duration_mode, args.silence_dbfs, args.silence_threshold_sec,
            args.silence_target_sec, args.silence_frame_ms) == ("post_silence", -40.0, 1.0, 0.3, 20)
    validate_duration_parameters("post_silence", -35.0, 0.5, 0.2, 10)
    validate_duration_parameters("post_silence", 0.0, 0.5, 0.2, 10)
    with pytest.raises(ValueError):
        validate_duration_parameters("post_silence", 0.1, 0.5, 0.2, 10)
    with pytest.raises(ValueError):
        validate_duration_parameters("post_silence", -35.0, 0.5, 0.6, 10)
    with pytest.raises(ValueError):
        validate_duration_parameters("post_silence", float("nan"), 0.5, 0.2, 10)


@pytest.mark.parametrize("mode, label", [
    ("raw_mp3", "raw_edge_mp3_before_ffmpeg_and_silence_reduction"),
    ("post_silence", "edge_audio_decoded_with_libsndfile_pcm16_wav_emulation_measured_after_production_silence_reduction"),
])
def test_run_report_duration_mode_schema_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, label: str,
) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps([{"index": 1, "text": "hello"}]), encoding="utf-8")

    async def fake_calibrate(*args: object, **kwargs: object) -> list[dict[str, object]]:
        return [{"voice": "voice", "budget_chars": 5, "sps_chars": 5, "duration_sec": 2.0,
                 "raw_duration_sec": 4.0, "budget_cps": 2.5, "sps_cps": 2.5}]

    monkeypatch.setattr(calibrator, "calibrate_records", fake_calibrate)
    args = build_parser().parse_args([str(input_path), "--output-dir", str(tmp_path / "out"),
                                      "--duration-mode", mode])
    report = asyncio.run(calibrator.run(args))
    assert report["schema_version"] == 3
    assert report["measurement"] == label
    assert report["parameters"]["duration_mode"] == mode
    assert report["parameters"]["silence_dbfs"] == -35.0
    assert report["parameters"]["silence_threshold_sec"] == 0.5
    assert report["parameters"]["silence_target_sec"] == 0.2
    assert report["parameters"]["silence_frame_ms"] == 10
    if mode == "post_silence":
        assert "libsndfile" in report["measurement_caveat"]
    else:
        assert "measurement_caveat" not in report
