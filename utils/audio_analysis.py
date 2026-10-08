"""Post-synthesis audio analysis and optional speed-up.

Goal (per spec):
    If an audio segment is longer than the allowed subtitle duration by >= 10%,
    speed it up by +10% (or use another faster method).

This module provides a lightweight, batch-friendly implementation.
The standalone legacy path retains the rule above. The CLI/UI Vosk pipeline
opts into bounded measured tempo via synthesis_rate and max_total_speed;
that path uses shared planner windows and never changes pitch as a fallback.

Allowed duration rule:
    - Prefer `analysis.extended_duration_sec` if present (gap extension).
    - Otherwise use `analysis.duration_sec`.
"""

from __future__ import annotations

import argparse
import json
import logging
import hashlib
import math
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import soundfile as sf


@dataclass
class SegmentTimingResult:
    index: int
    audio_path: Path
    allowed_duration_sec: float
    audio_duration_sec: float
    needs_speedup: bool
    speedup_applied: bool
    status: str = "unchecked"
    synthesis_rate: float | None = None
    required_factor: float | None = None
    effective_factor: float = 1.0
    duration_after_sec: float | None = None
    max_total_speed: float | None = None
    cue_id: str | None = None
    start: int | None = None
    end: int | None = None
    error: str | None = None
    available_ms: int | None = None
    effective_total_speed: float | None = None
    render_attempts: list[dict[str, Any]] = field(default_factory=list)
    render_attempt_count: int = 0
    render_guard_sec: float = 0.0


def _allowed_duration_sec(analysis: Any) -> Optional[float]:
    if not isinstance(analysis, dict):
        return None
    for key in ("extended_duration_sec", "duration_sec"):
        v = analysis.get(key)
        try:
            f = float(v)
        except Exception:
            continue
        if f > 0:
            return f
    return None


def _wav_duration_sec(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames / info.samplerate)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _speedup_with_ffmpeg(
    input_path: Path,
    output_path: Path,
    *,
    speed_factor: float,
    sample_rate: Optional[int] = None,
) -> None:
    if speed_factor <= 0:
        raise ValueError("speed_factor must be > 0")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-filter:a",
        f"atempo={speed_factor}",
        "-acodec",
        "pcm_s16le",
    ]
    if sample_rate:
        cmd.extend(["-ar", str(int(sample_rate))])
    cmd.extend(["-y", str(output_path)])

    subprocess.run(cmd, check=True)


def render_bounded_tempo(
    input_path: Path, output_path: Path, *, allowed_duration_sec: float,
    max_factor: float, attempts: list[dict[str, Any]],
) -> tuple[float, float]:
    """Render once, with at most one small-overrun correction from the same source.

    A 20 ms internal render target is not a changed cue window or fit tolerance.
    Caller replaces its source only after successful return and owns rate provenance.
    """
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Bounded rendering requires a distinct output; never compound tempo")
    if not math.isfinite(allowed_duration_sec) or allowed_duration_sec <= 0:
        raise ValueError("Allowed duration must be finite and positive")
    if not math.isfinite(max_factor) or max_factor < 1:
        raise ValueError("Remaining tempo cap must be finite and at least one")
    actual = _wav_duration_sec(input_path)
    if not math.isfinite(actual) or actual <= 0:
        raise ValueError("Source audio duration must be finite and positive")
    factor = max(1.0, actual / allowed_duration_sec)
    if factor > max_factor:
        raise ValueError("Required tempo exceeds remaining cap; futile rendering prohibited")
    if not _ffmpeg_available():
        raise RuntimeError("Bounded tempo requires ffmpeg atempo; pitch-changing fallback is prohibited")

    def render(speed: float, headroom: float) -> float:
        attempt: dict[str, Any] = {"factor": speed, "source_path": str(input_path),
                                   "target_headroom_sec": headroom, "status": "started"}
        attempts.append(attempt)
        try:
            # Remove any earlier output: a missing second render must not reuse it.
            output_path.unlink(missing_ok=True)
            _speedup_with_ffmpeg(input_path, output_path, speed_factor=speed)
            duration = _wav_duration_sec(output_path)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Invalid ffmpeg output duration")
            attempt.update(status="complete", duration_after_sec=duration)
            return duration
        except Exception as exc:
            attempt.update(status="failed", error=str(exc))
            raise

    after = render(factor, 0.0)
    guard = 0.020
    if 0 < after - allowed_duration_sec <= guard and factor < max_factor:
        target = max(allowed_duration_sec - guard, allowed_duration_sec * 0.5)
        corrected = min(max_factor, factor * after / target)
        if corrected > factor:
            after = render(corrected, allowed_duration_sec - target)
            factor = corrected
    return factor, after


def _speedup_naive_resample(
    input_path: Path,
    output_path: Path,
    *,
    speed_factor: float,
) -> None:
    """Fallback: naive time compression using resampling.

    This changes pitch. Use only if ffmpeg is unavailable.
    """

    data, sr = sf.read(str(input_path), always_2d=True, dtype="float32")
    if sr <= 0 or data.shape[0] == 0:
        sf.write(str(output_path), data, sr)
        return

    n = data.shape[0]
    new_n = max(1, int(round(n / float(speed_factor))))
    x_old = np.linspace(0.0, 1.0, n, endpoint=False)
    x_new = np.linspace(0.0, 1.0, new_n, endpoint=False)

    out = np.stack([
        np.interp(x_new, x_old, data[:, ch]) for ch in range(data.shape[1])
    ], axis=1).astype(np.float32)

    sf.write(str(output_path), out, sr)


def check_and_speedup_segments(
    items: List[Dict[str, Any]],
    audio_dir: Path,
    *,
    file_prefix: str = "",
    ext: str = ".wav",
    overage_threshold_pct: float = 0.10,
    speedup_factor: float = 1.10,
    in_place: bool = True,
    prefer_ffmpeg: bool = True,
    synthesis_rate: float | None = None,
    max_total_speed: float | None = None,
    report_path: Path | None = None,
    allow_speedup: bool = True,
) -> List[SegmentTimingResult]:
    """Check each segment duration and speed up when it exceeds the allowed slot.

    The expected naming is `<file_prefix><index><ext>`.
    """

    if synthesis_rate is not None and max_total_speed is None:
        raise ValueError("synthesis_rate requires a bounded max_total_speed")
    if max_total_speed is not None:
        if synthesis_rate is None:
            raise ValueError("Bounded audio requires the actual synthesis_rate")
        if not in_place:
            raise ValueError("Bounded audio requires in_place=True")
        return _check_bounded_segments(
            items, audio_dir, file_prefix=file_prefix, ext=ext,
            synthesis_rate=synthesis_rate, max_total_speed=max_total_speed,
            report_path=report_path or audio_dir / "audio_timing.json",
            allow_speedup=allow_speedup, prefer_ffmpeg=prefer_ffmpeg,
        )

    results: List[SegmentTimingResult] = []

    if not audio_dir.exists() or not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    use_ffmpeg = prefer_ffmpeg and _ffmpeg_available()

    for item in items:
        try:
            idx = int(item.get("index"))
        except Exception:
            continue

        analysis = item.get("analysis")
        allowed = _allowed_duration_sec(analysis)
        if allowed is None:
            continue

        wav_path = audio_dir / f"{file_prefix}{idx}{ext}"
        if not wav_path.exists():
            continue

        actual = _wav_duration_sec(wav_path)
        needs = actual >= (allowed * (1.0 + float(overage_threshold_pct)))

        applied = False
        if needs:
            if in_place:
                out_path = wav_path
            else:
                out_path = audio_dir / f"{file_prefix}{idx}.speedup{ext}"

            # Write through a temp file on the SAME drive to avoid WinError 17
            # (Temp defaults to system drive C:, while repo may be on D:).
            with tempfile.TemporaryDirectory(dir=str(audio_dir)) as td:
                tmp = Path(td) / f"tmp{ext}"
                try:
                    if use_ffmpeg:
                        info = sf.info(str(wav_path))
                        _speedup_with_ffmpeg(
                            wav_path,
                            tmp,
                            speed_factor=float(speedup_factor),
                            sample_rate=int(info.samplerate) if info.samplerate else None,
                        )
                    else:
                        logging.warning(
                            "ffmpeg not available; using naive speedup (pitch will change) for %s",
                            wav_path.name,
                        )
                        _speedup_naive_resample(
                            wav_path,
                            tmp,
                            speed_factor=float(speedup_factor),
                        )

                    tmp.replace(out_path)
                    applied = True
                except Exception as e:
                    logging.error("Failed to speed up %s: %s", wav_path.name, e)

        results.append(
            SegmentTimingResult(
                index=idx,
                audio_path=wav_path,
                allowed_duration_sec=float(allowed),
                audio_duration_sec=float(actual),
                needs_speedup=bool(needs),
                speedup_applied=bool(applied),
            )
        )

    return results


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_timing_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _check_bounded_segments(
    items: List[Dict[str, Any]], audio_dir: Path, *, file_prefix: str, ext: str,
    synthesis_rate: float, max_total_speed: float, report_path: Path,
    allow_speedup: bool, prefer_ffmpeg: bool,
) -> List[SegmentTimingResult]:
    """Use shared windows, never resample pitch or compound a completed run.

    Futile acceleration (required factor above the remaining cap) is not applied.
    A pending report after interruption is rejected rather than guessing WAV rate.
    """
    from utils.pipeline_text import SpeedPolicy, json_sha256
    from utils.shortening_plan import PlannerConfig, plan_subtitles

    policy = SpeedPolicy(float(synthesis_rate), float(max_total_speed))
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    indices = [item.get("index") for item in items]
    if any(not isinstance(idx, int) or isinstance(idx, bool) for idx in indices) or len(set(indices)) != len(indices):
        raise ValueError("Bounded audio requires unique integer source indices")
    plans = plan_subtitles(items, PlannerConfig(max_speed=policy.max_total_speed))
    context = json_sha256({"items": items, "synthesis_rate": synthesis_rate,
                           "max_total_speed": max_total_speed, "file_prefix": file_prefix,
                           "ext": ext, "allow_speedup": allow_speedup, "prefer_ffmpeg": prefer_ffmpeg})
    paths = [audio_dir / f"{file_prefix}{idx}{ext}" for idx in indices]
    hashes = {str(idx): _file_sha256(path) for idx, path in zip(indices, paths)}
    if report_path.exists():
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        if saved.get("state") != "complete" or saved.get("context_sha256") != context or saved.get("wav_sha256") != hashes:
            raise ValueError("Audio timing report is stale or interrupted; use fresh synthesis in a new output directory")
        return [SegmentTimingResult(**{**row, "audio_path": Path(row["audio_path"])}) for row in saved["segments"]]

    report: Dict[str, Any] = {"schema": "pipeline-audio-timing-v1", "state": "pending",
                              "context_sha256": context, "wav_sha256": hashes, "segments": []}
    # Persist before the first possible WAV mutation, so interrupted runs cannot compound tempo.
    _write_timing_report(report_path, report)
    remaining = policy.max_total_speed / policy.base_speech_rate
    ffmpeg_ready = prefer_ffmpeg and _ffmpeg_available()
    results: List[SegmentTimingResult] = []
    for item, plan, path in zip(items, plans, paths):
        allowed = plan.available_ms / 1000.0
        result = SegmentTimingResult(
            index=item["index"], audio_path=path, allowed_duration_sec=allowed,
            audio_duration_sec=0.0, needs_speedup=False, speedup_applied=False,
            status="invalid", synthesis_rate=synthesis_rate, max_total_speed=max_total_speed,
            cue_id=plan.cue_id, start=plan.start, end=plan.end,
            available_ms=plan.available_ms, effective_total_speed=synthesis_rate,
        )
        try:
            if plan.status == "invalid" or allowed <= 0:
                result.error = "Invalid source timing/text"
            elif not path.exists():
                result.status = "missing"
            else:
                actual = _wav_duration_sec(path)
                if not math.isfinite(actual) or actual <= 0:
                    raise ValueError("Audio duration must be finite and positive")
                result.audio_duration_sec = actual
                result.duration_after_sec = actual
                required = max(1.0, actual / allowed)
                if not math.isfinite(required):
                    raise ValueError("Nonfinite required tempo factor")
                result.required_factor = required
                result.needs_speedup = actual > allowed
                result.status = "fits" if actual <= allowed else "unresolved"
                if (result.needs_speedup and required <= remaining
                        and synthesis_rate * required <= max_total_speed and allow_speedup):
                    if not ffmpeg_ready:
                        raise RuntimeError("Bounded tempo requires ffmpeg atempo; pitch-changing fallback is prohibited")
                    with tempfile.TemporaryDirectory(dir=str(audio_dir)) as td:
                        tmp = Path(td) / f"tempo{ext}"
                        applied, after = render_bounded_tempo(
                            path, tmp, allowed_duration_sec=allowed, max_factor=remaining,
                            attempts=result.render_attempts,
                        )
                        tmp.replace(path)
                    result.effective_factor = applied
                    result.effective_total_speed = synthesis_rate * applied
                    result.render_guard_sec = result.render_attempts[-1]["target_headroom_sec"]
                    result.speedup_applied = True
                    result.duration_after_sec = after
                    result.status = "accelerated_fits" if after <= allowed else "unresolved"
        except Exception as exc:
            result.status = "invalid"
            result.error = str(exc)
            logging.error("Bounded audio %s: %s", path.name, exc)
        result.render_attempt_count = len(result.render_attempts)
        results.append(result)
        report["segments"].append({**asdict(result), "audio_path": str(path)})
        report["wav_sha256"][str(result.index)] = _file_sha256(path)
    report["state"] = "complete"
    report["failed_count"] = sum(r.status not in {"fits", "accelerated_fits"} for r in results)
    _write_timing_report(report_path, report)
    logging.info("Measured audio: %d cues; unresolved/invalid/missing: %d; report: %s",
                 len(results), report["failed_count"], report_path)
    return results


def _load_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("JSON must be a list")
    return data  # type: ignore[return-value]


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze synthesized wav durations and speed up if needed")
    p.add_argument("json", help="JSON subtitles with analysis field")
    p.add_argument("audio_dir", help="Folder with <index>.wav files")
    p.add_argument("--file-prefix", default="", help="Prefix before <index> in file names")
    p.add_argument("--ext", default=".wav", help="Audio extension")
    p.add_argument("--overage-threshold-pct", type=float, default=0.10)
    p.add_argument("--speedup-factor", type=float, default=1.10)
    p.add_argument("--no-ffmpeg", action="store_true", help="Disable ffmpeg even if available")
    p.add_argument("--in-place", action="store_true", help="Overwrite wav files")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="[%(levelname)s] %(message)s")

    json_path = Path(args.json)
    audio_dir = Path(args.audio_dir)

    items = _load_json(json_path)
    results = check_and_speedup_segments(
        items,
        audio_dir,
        file_prefix=str(args.file_prefix),
        ext=str(args.ext),
        overage_threshold_pct=float(args.overage_threshold_pct),
        speedup_factor=float(args.speedup_factor),
        in_place=bool(args.in_place),
        prefer_ffmpeg=(not bool(args.no_ffmpeg)),
    )

    total = len(results)
    need = sum(1 for r in results if r.needs_speedup)
    applied = sum(1 for r in results if r.speedup_applied)

    logging.info("Checked %d segments; need speedup: %d; applied: %d", total, need, applied)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
