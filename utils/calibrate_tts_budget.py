"""Offline-friendly Edge TTS budget calibration utility."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
import os
import re
import sys
import inspect
from pathlib import Path
from statistics import mean, median
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Sequence

from utils.analyze_text import join_text_lines
from utils.shortening_domain import POST_SILENCE_MEASUREMENT_LABEL, count_ascii_punctuation
from utils.text_sanitize import count_symbols_for_sps

DEFAULT_VOICE = "ru-RU-DmitryNeural"
CURRENT_DEFAULT_BUDGET_CPS = 13.0
DurationReader = Callable[[Path], float]
Synthesizer = Callable[[str, str, Path, str, str, str, float, float], Awaitable[None] | None]
DurationModel = Dict[str, Any]
RAW_MEASUREMENT_LABEL = "raw_edge_mp3_before_ffmpeg_and_silence_reduction"
def sanitized_voice_directory(voice: str) -> str:
    """Return a stable, filesystem-safe directory name for a voice."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", voice).strip("._")
    return safe or "voice"


def cache_key(text: str, voice: str, rate: str, volume: str, pitch: str) -> str:
    payload = json.dumps(
        {"pitch": pitch, "rate": rate, "text": text, "voice": voice, "volume": volume},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear interpolation at fraction, using positions 0 through n-1."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _stats(values: Sequence[float], chars: int, duration: float) -> Dict[str, float | int]:
    return {
        "count": len(values),
        "arithmetic_mean": mean(values),
        "pooled_cps": chars / duration,
        "median": median(values),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "min": min(values),
        "max": max(values),
    }


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fit_linear_system(matrix: List[List[float]], target: List[float]) -> List[float] | None:
    """Solve a small square system, returning None for rank deficiency."""
    size = len(target)
    augmented = [row[:] + [target[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    left - factor * right for left, right in zip(augmented[row], augmented[column])
                ]
    return [augmented[index][-1] for index in range(size)]


def fit_duration_model(records: Sequence[Mapping[str, Any]], include_punctuation: bool = False) -> DurationModel | None:
    """Fit an ordinary least-squares duration model to valid measurement records."""
    rows: List[tuple[List[float], float]] = []
    for record in records:
        duration = _finite_number(record.get("duration_sec"))
        chars = _finite_number(record.get("budget_chars"))
        if (duration is None or duration <= 0 or isinstance(record.get("budget_chars"), bool)
                or chars is None or chars < 0 or not chars.is_integer()):
            continue
        predictors = [1.0, chars]
        if include_punctuation:
            punctuation = _finite_number(record.get("punctuation_count"))
            if (isinstance(record.get("punctuation_count"), bool) or punctuation is None
                    or punctuation < 0 or not punctuation.is_integer()):
                continue
            predictors.append(punctuation)
        rows.append((predictors, duration))
    parameter_count = 3 if include_punctuation else 2
    if len(rows) < parameter_count:
        return None
    matrix = [[sum(row[0][left] * row[0][right] for row in rows) for right in range(parameter_count)]
              for left in range(parameter_count)]
    target = [sum(row[0][column] * row[1] for row in rows) for column in range(parameter_count)]
    coefficients = _fit_linear_system(matrix, target)
    if coefficients is None or not all(math.isfinite(value) for value in coefficients):
        return None
    residuals = [duration - sum(coefficient * value for coefficient, value in zip(coefficients, predictors))
                 for predictors, duration in rows]
    observed = [duration for _, duration in rows]
    observed_mean = mean(observed)
    total_sum_squares = sum((value - observed_mean) ** 2 for value in observed)
    residual_sum_squares = sum(value * value for value in residuals)
    model_name = "chars_plus_punctuation" if include_punctuation else "chars_only"
    coefficient_values: Dict[str, float] = {
        "intercept_sec": coefficients[0],
        "seconds_per_budget_char": coefficients[1],
    }
    if include_punctuation:
        coefficient_values["seconds_per_punctuation"] = coefficients[2]
    residual_quantiles = {
        "p10": percentile(residuals, 0.10), "p25": percentile(residuals, 0.25),
        "median": percentile(residuals, 0.50), "p75": percentile(residuals, 0.75),
        "p90": percentile(residuals, 0.90), "min": min(residuals), "max": max(residuals),
    }
    return {
        "model": model_name,
        "formula": "duration_sec = intercept_sec + seconds_per_budget_char * budget_chars"
        + (" + seconds_per_punctuation * punctuation_count" if include_punctuation else ""),
        "sample_count": len(rows),
        "coefficients": coefficient_values,
        "r_squared": None if total_sum_squares == 0 else 1.0 - residual_sum_squares / total_sum_squares,
        "mae_sec": mean([abs(value) for value in residuals]),
        "rmse_sec": math.sqrt(mean([value * value for value in residuals])),
        "residual_quantiles_sec": residual_quantiles,
    }


def _exploratory_duration_profile(records: Sequence[Mapping[str, Any]]) -> DurationModel | None:
    chars_only = fit_duration_model(records)
    chars_plus = fit_duration_model(records, include_punctuation=True)
    candidates = [chars_plus, chars_only]
    for model in candidates:
        if model is None or model["sample_count"] < 30:
            continue
        coefficients = model["coefficients"]
        if (coefficients["intercept_sec"] < 0 or coefficients["seconds_per_budget_char"] <= 0
                or ("seconds_per_punctuation" in coefficients and coefficients["seconds_per_punctuation"] < 0)):
            continue
        safety = max(0.0, model["residual_quantiles_sec"]["p75"])
        return {
            "source_model": model["model"], "coefficients": dict(coefficients),
            "in_sample_p75_margin_sec": safety,
            "margin_quantile": "signed in-sample residual (observed_duration_sec - predicted_duration_sec) p75",
        }
    return None


def summarize_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize records by voice in both character-count conventions."""
    result: Dict[str, Any] = {}
    voices = sorted({str(record["voice"]) for record in records})
    for voice in voices:
        subset = [record for record in records if record["voice"] == voice]
        budget_values = [float(record["budget_cps"]) for record in subset]
        sps_values = [float(record["sps_cps"]) for record in subset]
        duration = sum(float(record["duration_sec"]) for record in subset)
        chars_only = fit_duration_model(subset)
        chars_plus = fit_duration_model(subset, include_punctuation=True)
        profile = _exploratory_duration_profile(subset)
        profile_warning = (
            "Coefficients and the in-sample p75 margin are fitted and evaluated on the same sample; "
            "they are not cross-validated, are not a production recommendation, and do not guarantee coverage."
        )
        if profile is None:
            if chars_only is None or chars_only["sample_count"] < 30:
                profile_warning = (
                    "Exploratory duration profile is absent because at least 30 valid samples are required. "
                    + profile_warning
                )
            else:
                profile_warning = "No physically valid exploratory duration model is available. " + profile_warning
        result[voice] = {
            "count": len(subset),
            "budget_cps": _stats(budget_values, sum(int(r["budget_chars"]) for r in subset), duration),
            "sps_cps": _stats(sps_values, sum(int(r["sps_chars"]) for r in subset), duration),
            "exploratory_budget_cps_p25": percentile(budget_values, 0.25) if budget_values else None,
            "duration_models": {"chars_only": chars_only, "chars_plus_punctuation": chars_plus},
            "exploratory_duration_profile": profile,
            "scalar_cps_warning": "Scalar CPS is length-sensitive and exploratory; it is not a production recommendation.",
            "duration_profile_warning": profile_warning,
        }
    return result


def default_duration_reader(path: Path) -> float:
    import soundfile

    info = soundfile.info(str(path))
    duration = float(info.duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid audio duration: {path}")
    return duration


def post_silence_duration_reader(
    path: Path,
    *,
    silence_dbfs: float = -35.0,
    threshold_sec: float = 0.5,
    target_sec: float = 0.2,
    frame_ms: int = 10,
) -> float:
    """Measure libsndfile-decoded audio after PCM16 WAV emulation and reduction."""
    import soundfile

    from utils.reduce_silence import reduce_silence_audio

    audio, sample_rate = soundfile.read(str(path), always_2d=True, dtype="float32")
    if not math.isfinite(float(sample_rate)) or float(sample_rate) <= 0:
        raise ValueError(f"invalid audio sample rate: {path}")
    wav_buffer = io.BytesIO()
    soundfile.write(wav_buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    wav_buffer.seek(0)
    audio, roundtrip_sample_rate = soundfile.read(wav_buffer, always_2d=True, dtype="float32")
    if roundtrip_sample_rate != sample_rate:
        raise ValueError(f"PCM16 WAV round-trip changed sample rate: {path}")
    reduced_audio, _ = reduce_silence_audio(
        audio,
        sample_rate,
        silence_dbfs=silence_dbfs,
        threshold_sec=threshold_sec,
        target_sec=target_sec,
        frame_ms=frame_ms,
    )
    duration = float(reduced_audio.shape[0]) / float(sample_rate)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid reduced audio duration: {path}")
    return duration


async def default_synthesizer(
    text: str, voice: str, output_path: Path, rate: str, volume: str, pitch: str,
    connect_timeout: float, receive_timeout: float,
) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(
        text,
        voice=voice,
        rate=rate,
        volume=volume,
        pitch=pitch,
        connect_timeout=int(connect_timeout),
        receive_timeout=int(receive_timeout),
    )
    await asyncio.wait_for(communicate.save(str(output_path)), timeout=connect_timeout + receive_timeout)


async def calibrate_records(
    items: Sequence[Mapping[str, Any]], output_dir: Path, voices: Sequence[str], rate: str = "+0%",
    volume: str = "+0%", pitch: str = "+0Hz", retries: int = 3, connect_timeout: float = 30,
     receive_timeout: float = 300, limit: int | None = None, duration_reader: DurationReader = default_duration_reader,
     synthesizer: Synthesizer = default_synthesizer, measurement_reader: DurationReader | None = None,
) -> List[Dict[str, Any]]:
    """Sequentially synthesize/cache all requested measurements."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = list(items[:limit] if limit is not None else items)
    records: List[Dict[str, Any]] = []
    for item in selected:
        index = int(item["index"])
        text = join_text_lines(item.get("text", "")).strip()
        for voice in voices:
            digest = cache_key(text, voice, rate, volume, pitch)
            voice_dir = output_dir / sanitized_voice_directory(voice)
            audio_path = voice_dir / f"{index}-{digest[:16]}.mp3"
            cache_hit = False
            raw_duration: float | None = None
            if audio_path.exists():
                try:
                    raw_duration = duration_reader(audio_path)
                    cache_hit = True
                except Exception:
                    audio_path.unlink(missing_ok=True)
            if not cache_hit:
                voice_dir.mkdir(parents=True, exist_ok=True)
                temporary = audio_path.with_name(f".{audio_path.name}.{os.getpid()}.tmp")
                temporary.unlink(missing_ok=True)
                try:
                    error: Exception | None = None
                    for attempt in range(retries):
                        try:
                            result = synthesizer(text, voice, temporary, rate, volume, pitch, connect_timeout, receive_timeout)
                            if inspect.isawaitable(result):
                                await result
                            raw_duration = duration_reader(temporary)
                            error = None
                            break
                        except Exception as exc:
                            error = exc
                            temporary.unlink(missing_ok=True)
                            if attempt + 1 < retries:
                                await asyncio.sleep(min(2.0, 0.25 * (2**attempt)))
                    if error is not None or raw_duration is None:
                        raise RuntimeError(f"failed to synthesize index {index}, voice {voice}") from error
                    os.replace(temporary, audio_path)
                finally:
                    temporary.unlink(missing_ok=True)
            assert raw_duration is not None
            duration = measurement_reader(audio_path) if measurement_reader is not None else raw_duration
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"invalid measured audio duration: {audio_path}")
            budget_chars = len(text)
            sps_chars = count_symbols_for_sps(text)
            punctuation_count = count_ascii_punctuation(text)
            records.append({
                "index": index, "voice": voice, "audio_path": str(audio_path), "cache_hit": cache_hit,
                "text": text, "budget_chars": budget_chars, "sps_chars": sps_chars,
                "raw_duration_sec": raw_duration, "duration_sec": duration,
                "budget_cps": budget_chars / duration, "sps_cps": sps_chars / duration,
                "punctuation_count": punctuation_count,
            })
    return records


def calibrate_records_sync(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
    """Synchronous adapter for callers that do not already own an event loop."""
    return asyncio.run(calibrate_records(*args, **kwargs))


def validate_items(raw: Any, limit: int | None = None) -> List[Mapping[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("input JSON must be a list")
    seen: set[int] = set()
    valid: List[Mapping[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or isinstance(item.get("index"), bool) or not isinstance(item.get("index"), int):
            raise ValueError("each item must have a unique integer index")
        index = int(item["index"])
        if index in seen:
            raise ValueError(f"duplicate index: {index}")
        text = join_text_lines(item.get("text", "")).strip()
        if not text:
            raise ValueError(f"empty text at index: {index}")
        seen.add(index)
        valid.append(item)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    return valid


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate raw and post-silence Edge TTS durations.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voices", nargs="+", default=[DEFAULT_VOICE])
    parser.add_argument("--rate", default="+0%")
    parser.add_argument("--volume", default="+0%")
    parser.add_argument("--pitch", default="+0Hz")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--connect-timeout", type=float, default=30)
    parser.add_argument("--receive-timeout", type=float, default=300)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--duration-mode", choices=("raw_mp3", "post_silence"), default="raw_mp3")
    parser.add_argument("--silence-dbfs", type=float, default=-35.0)
    parser.add_argument("--silence-threshold-sec", type=float, default=0.5)
    parser.add_argument("--silence-target-sec", type=float, default=0.2)
    parser.add_argument("--silence-frame-ms", type=int, default=10)
    return parser


def validate_duration_parameters(
    duration_mode: str,
    silence_dbfs: float,
    silence_threshold_sec: float,
    silence_target_sec: float,
    silence_frame_ms: int,
) -> None:
    """Validate duration measurement mode and silence reducer settings."""
    if duration_mode not in {"raw_mp3", "post_silence"}:
        raise ValueError("duration-mode must be raw_mp3 or post_silence")
    values = (silence_dbfs, silence_threshold_sec, silence_target_sec, float(silence_frame_ms))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("silence parameters must be finite")
    if silence_dbfs > 0:
        raise ValueError("silence_dbfs must be less than or equal to zero")
    if silence_threshold_sec <= 0 or silence_target_sec <= 0 or silence_frame_ms <= 0:
        raise ValueError("silence threshold, target, and frame must be positive")
    if silence_target_sec > silence_threshold_sec:
        raise ValueError("silence target must be less than or equal to threshold")


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.voices or any(not voice for voice in args.voices):
        raise ValueError("voices must be nonempty")
    if (args.retries <= 0 or not math.isfinite(args.connect_timeout) or args.connect_timeout <= 0
            or not math.isfinite(args.receive_timeout) or args.receive_timeout <= 0):
        raise ValueError("retries and timeouts must be positive")
    validate_duration_parameters(args.duration_mode, args.silence_dbfs, args.silence_threshold_sec,
                                 args.silence_target_sec, args.silence_frame_ms)
    with args.input.open(encoding="utf-8") as handle:
        items = validate_items(json.load(handle), args.limit)
    measurement_reader: DurationReader | None = None
    if args.duration_mode == "post_silence":
        def measure(path: Path) -> float:
            return post_silence_duration_reader(
                path,
                silence_dbfs=args.silence_dbfs,
                threshold_sec=args.silence_threshold_sec,
                target_sec=args.silence_target_sec,
                frame_ms=args.silence_frame_ms,
            )
        measurement_reader = measure
    records = await calibrate_records(items, args.output_dir, args.voices, args.rate, args.volume, args.pitch,
                                      args.retries, args.connect_timeout, args.receive_timeout, args.limit,
                                      measurement_reader=measurement_reader)
    measurement = (RAW_MEASUREMENT_LABEL if args.duration_mode == "raw_mp3"
                   else POST_SILENCE_MEASUREMENT_LABEL)
    report = {
        "schema_version": 3,
        "source": str(args.input),
        "parameters": {"voices": args.voices, "rate": args.rate, "volume": args.volume, "pitch": args.pitch,
                        "retries": args.retries, "connect_timeout": args.connect_timeout,
                        "receive_timeout": args.receive_timeout, "limit": args.limit,
                        "duration_mode": args.duration_mode, "silence_dbfs": args.silence_dbfs,
                        "silence_threshold_sec": args.silence_threshold_sec,
                        "silence_target_sec": args.silence_target_sec,
                        "silence_frame_ms": args.silence_frame_ms},
        "measurement": measurement,
        "current_default_budget_cps": CURRENT_DEFAULT_BUDGET_CPS,
        "records": records,
        "summaries": summarize_records(records),
    }
    if args.duration_mode == "post_silence":
        report["measurement_caveat"] = (
            "Post-silence audio is decoded with libsndfile rather than production ffmpeg; "
            "this does not claim waveform equivalence or a production guarantee."
        )
    report_path = args.report or args.output_dir / "calibration.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_rounded(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(run(args))
    except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
