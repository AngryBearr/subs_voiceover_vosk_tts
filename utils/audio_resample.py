"""
High-quality audio resampling utilities.

Uses SciPy's polyphase resampler when available.
"""
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable

from typing import Optional
import math
import numpy as np

try:  # Optional high-quality resampling for float audio
    import soxr  # type: ignore
except Exception:  # pragma: no cover
    soxr = None

try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover
    sf = None


class ResampleUnavailableError(RuntimeError):
    """Raised when SciPy-based resampling is requested but SciPy is not installed."""


def resample_int16_scipy(audio_i16: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """
    Resample mono int16 PCM from `from_sr` to `to_sr` using SciPy's resample_poly.

    Args:
        audio_i16: int16 numpy array (mono) at from_sr.
        from_sr: source sample rate (Hz)
        to_sr: target sample rate (Hz)

    Returns:
        int16 numpy array resampled to `to_sr`.

    Raises:
        ResampleUnavailableError: if SciPy is not installed.
    """
    if from_sr == to_sr:
        return audio_i16

    try:
        from scipy.signal import resample_poly  # type: ignore
    except Exception as e:
        raise ResampleUnavailableError(
            "SciPy is required for resampling but is not installed."
        ) from e

    # Convert to float [-1, 1]
    x = audio_i16.astype(np.float32) / 32767.0

    # Use rational up/down derived from gcd for stability/perf
    g = math.gcd(from_sr, to_sr)
    up = to_sr // g
    down = from_sr // g

    y = resample_poly(x, up=up, down=down)

    # Back to int16 with clipping
    y = np.clip(y, -1.0, 1.0)
    return (y * 32767.0).astype(np.int16)


def _resample_float(data: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample float audio (samples, channels) from from_sr to to_sr."""

    if from_sr == to_sr:
        return data

    if data.ndim == 1:
        data = data[:, None]

    if soxr is not None:
        return soxr.resample(data, from_sr, to_sr).astype(np.float32)

    try:
        from scipy.signal import resample_poly  # type: ignore

        g = math.gcd(from_sr, to_sr)
        up = to_sr // g
        down = from_sr // g

        out_ch = []
        for ch in range(data.shape[1]):
            out_ch.append(resample_poly(data[:, ch], up=up, down=down))
        out = np.stack(out_ch, axis=1)
        return out.astype(np.float32)
    except Exception:
        # Linear interpolation fallback
        ratio = to_sr / from_sr
        new_len = int(round(data.shape[0] * ratio))
        if new_len <= 1:
            return data[:1].astype(np.float32)
        x_old = np.linspace(0.0, 1.0, data.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
        out = np.stack(
            [np.interp(x_new, x_old, data[:, ch]) for ch in range(data.shape[1])],
            axis=1,
        )
        return out.astype(np.float32)


@dataclass
class ResampleFileStats:
    input_path: Path
    output_path: Path
    from_sr: int
    to_sr: int
    old_duration_sec: float
    new_duration_sec: float


def resample_wav_file(
    input_path: Path,
    output_path: Path,
    *,
    target_sr: int = 44100,
) -> ResampleFileStats:
    """Resample a wav (or any libsndfile-supported file) to target sample rate."""

    if sf is None:
        raise ResampleUnavailableError("soundfile is required for file resampling")

    data, sr = sf.read(str(input_path), always_2d=True, dtype="float32")
    from_sr = int(sr)
    old_dur = float(data.shape[0] / from_sr) if from_sr else 0.0

    if from_sr != int(target_sr):
        data = _resample_float(data, from_sr, int(target_sr))
        sr = int(target_sr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), data, int(sr))
    new_dur = float(data.shape[0] / int(sr)) if sr else 0.0

    return ResampleFileStats(
        input_path=input_path,
        output_path=output_path,
        from_sr=int(from_sr),
        to_sr=int(target_sr),
        old_duration_sec=old_dur,
        new_duration_sec=new_dur,
    )


def resample_wav_dir(
    input_dir: Path,
    *,
    target_sr: int = 44100,
    pattern: str = "*.wav",
    in_place: bool = True,
    output_dir: Optional[Path] = None,
) -> list[ResampleFileStats]:
    """Resample all matching files in a directory."""

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    out_dir = input_dir if in_place else (output_dir or (input_dir / f"resampled_{target_sr}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: list[ResampleFileStats] = []
    for p in sorted(input_dir.glob(pattern)):
        if not p.is_file():
            continue
        out_path = p if in_place else (out_dir / p.name)
        stats.append(resample_wav_file(p, out_path, target_sr=target_sr))
    return stats


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resample wav files to a target sample rate")
    p.add_argument("path", help="Input file or folder")
    p.add_argument("--target-sr", type=int, default=44100)
    p.add_argument("--pattern", default="*.wav")
    p.add_argument("--output", default=None, help="Output file (file mode) or folder (dir mode)")
    p.add_argument("--in-place", action="store_true", help="Overwrite files in-place (dir mode)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    import logging

    logging.basicConfig(level=args.log_level.upper(), format="[%(levelname)s] %(message)s")

    src = Path(args.path)
    if not src.exists():
        raise FileNotFoundError(f"Path not found: {src}")

    if src.is_file():
        out = Path(args.output) if args.output else (src.parent / f"{src.stem}.{args.target_sr}hz{src.suffix}")
        resample_wav_file(src, out, target_sr=int(args.target_sr))
        logging.info("Resampled %s -> %s", src.name, out.name)
        return 0

    out_dir = Path(args.output) if (args.output and not args.in_place) else None
    stats = resample_wav_dir(
        src,
        target_sr=int(args.target_sr),
        pattern=str(args.pattern),
        in_place=bool(args.in_place),
        output_dir=out_dir,
    )
    logging.info("Resampled %d files to %d Hz", len(stats), int(args.target_sr))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
