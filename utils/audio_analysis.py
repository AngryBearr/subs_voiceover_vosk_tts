"""Post-synthesis audio analysis and optional speed-up.

Goal (per spec):
    If an audio segment is longer than the allowed subtitle duration by >= 10%,
    speed it up by +10% (or use another faster method).

This module provides a lightweight, batch-friendly implementation.

Allowed duration rule:
    - Prefer `analysis.extended_duration_sec` if present (gap extension).
    - Otherwise use `analysis.duration_sec`.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
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
) -> List[SegmentTimingResult]:
    """Check each segment duration and speed up when it exceeds the allowed slot.

    The expected naming is `<file_prefix><index><ext>`.
    """

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
