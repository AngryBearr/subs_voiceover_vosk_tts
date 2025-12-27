"""Reduce long silent pauses in WAV files.

Purpose:
    After TTS synthesis, pauses may be too long for comfortable listening.
    This utility shortens silent regions longer than a threshold.

Default behavior (per spec):
    - Reduce pauses longer than 0.5 seconds down to 0.2 seconds.

Notes:
    - Works on WAV/any libsndfile-supported formats via `soundfile`.
    - Uses a simple peak-based silence detector on short frames.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import soundfile as sf


@dataclass
class ReduceSilenceStats:
    input_path: Path
    output_path: Path
    sr: int
    old_duration_sec: float
    new_duration_sec: float
    reduced_regions: int


def _dbfs_to_amp(dbfs: float) -> float:
    # Full scale assumed 1.0
    return float(10 ** (dbfs / 20.0))


def reduce_silence_audio(
    audio: np.ndarray,
    sr: int,
    *,
    silence_dbfs: float = -35.0,
    threshold_sec: float = 0.5,
    target_sec: float = 0.2,
    frame_ms: int = 10,
) -> tuple[np.ndarray, int]:
    """Return audio with long silent regions shortened.

    Args:
        audio: float array, shape (samples, channels) or (samples,)
        sr: sample rate
        silence_dbfs: threshold in dBFS below which a frame is considered silent
        threshold_sec: only silence longer than this is reduced
        target_sec: silence segments above threshold are replaced by this length
        frame_ms: analysis frame size

    Returns:
        (new_audio, reduced_regions_count)
    """

    if sr <= 0:
        raise ValueError("sr must be > 0")
    if threshold_sec < 0:
        raise ValueError("threshold_sec must be >= 0")
    if target_sec < 0:
        raise ValueError("target_sec must be >= 0")

    data = np.asarray(audio)
    if data.ndim == 1:
        data = data[:, None]

    if data.size == 0:
        return data, 0

    frame = max(1, int(round(sr * (frame_ms / 1000.0))))
    amp_thr = _dbfs_to_amp(silence_dbfs)

    n = data.shape[0]
    n_frames = int(np.ceil(n / frame))

    silent_frames = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        a = i * frame
        b = min(n, (i + 1) * frame)
        peak = float(np.max(np.abs(data[a:b, :]))) if b > a else 0.0
        silent_frames[i] = peak < amp_thr

    threshold_frames = int(np.ceil(threshold_sec / (frame_ms / 1000.0))) if threshold_sec > 0 else 0
    target_frames = int(np.round(target_sec / (frame_ms / 1000.0))) if target_sec > 0 else 0

    # Find runs of silent frames
    reduced = 0
    out_parts: list[np.ndarray] = []

    cur = 0
    while cur < n_frames:
        if not silent_frames[cur]:
            # non-silent: copy one frame
            a = cur * frame
            b = min(n, (cur + 1) * frame)
            out_parts.append(data[a:b, :])
            cur += 1
            continue

        # silent run
        run_start = cur
        while cur < n_frames and silent_frames[cur]:
            cur += 1
        run_end = cur
        run_len = run_end - run_start

        a = run_start * frame
        b = min(n, run_end * frame)
        original = data[a:b, :]

        if threshold_frames and run_len > threshold_frames and target_frames < run_len:
            # Replace with shorter silence
            new_len = min(b - a, target_frames * frame)
            out_parts.append(np.zeros((new_len, data.shape[1]), dtype=data.dtype))
            reduced += 1
        else:
            out_parts.append(original)

    out = np.concatenate(out_parts, axis=0) if out_parts else data[:0, :]
    return out, reduced


def reduce_silence_file(
    input_path: Path,
    output_path: Path,
    *,
    silence_dbfs: float = -35.0,
    threshold_sec: float = 0.5,
    target_sec: float = 0.2,
    frame_ms: int = 10,
) -> ReduceSilenceStats:
    """Reduce silence in one file."""

    data, sr = sf.read(str(input_path), always_2d=True, dtype="float32")
    old_dur = float(data.shape[0] / sr) if sr else 0.0

    new_data, reduced = reduce_silence_audio(
        data,
        sr,
        silence_dbfs=float(silence_dbfs),
        threshold_sec=float(threshold_sec),
        target_sec=float(target_sec),
        frame_ms=int(frame_ms),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), new_data, sr)

    new_dur = float(new_data.shape[0] / sr) if sr else 0.0
    return ReduceSilenceStats(
        input_path=input_path,
        output_path=output_path,
        sr=int(sr),
        old_duration_sec=old_dur,
        new_duration_sec=new_dur,
        reduced_regions=int(reduced),
    )


def reduce_silence_in_dir(
    input_dir: Path,
    *,
    output_dir: Optional[Path] = None,
    pattern: str = "*.wav",
    in_place: bool = False,
    silence_dbfs: float = -35.0,
    threshold_sec: float = 0.5,
    target_sec: float = 0.2,
    frame_ms: int = 10,
) -> list[ReduceSilenceStats]:
    """Reduce silence for all matching files in a directory."""

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    if in_place:
        out_dir = input_dir
    else:
        out_dir = output_dir or (input_dir / "reduced_silence")

    stats: list[ReduceSilenceStats] = []
    for p in sorted(input_dir.glob(pattern)):
        if not p.is_file():
            continue
        out_path = p if in_place else (out_dir / p.name)
        st = reduce_silence_file(
            p,
            out_path,
            silence_dbfs=silence_dbfs,
            threshold_sec=threshold_sec,
            target_sec=target_sec,
            frame_ms=frame_ms,
        )
        stats.append(st)

    return stats


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reduce long silent pauses in WAV files")
    parser.add_argument("path", help="Input file or folder")
    parser.add_argument("--pattern", default="*.wav", help="Glob pattern for directory mode")
    parser.add_argument("--output", default=None, help="Output file (file mode) or folder (dir mode)")
    parser.add_argument("--in-place", action="store_true", help="Overwrite files in-place (dir mode)")

    parser.add_argument("--threshold-sec", type=float, default=0.5, help="Reduce pauses longer than this")
    parser.add_argument("--target-sec", type=float, default=0.2, help="Replace long pauses with this length")
    parser.add_argument("--silence-dbfs", type=float, default=-35.0, help="Silence threshold (dBFS)")
    parser.add_argument("--frame-ms", type=int, default=10, help="Analysis frame size in ms")

    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="[%(levelname)s] %(message)s")

    src = Path(args.path)
    if not src.exists():
        raise FileNotFoundError(f"Path not found: {src}")

    if src.is_file():
        out = Path(args.output) if args.output else (src.parent / f"{src.stem}.reduced{src.suffix}")
        st = reduce_silence_file(
            src,
            out,
            silence_dbfs=args.silence_dbfs,
            threshold_sec=args.threshold_sec,
            target_sec=args.target_sec,
            frame_ms=args.frame_ms,
        )
        logging.info(
            "Reduced %s -> %s (%.3fs -> %.3fs, regions=%d)",
            st.input_path.name,
            st.output_path.name,
            st.old_duration_sec,
            st.new_duration_sec,
            st.reduced_regions,
        )
        return 0

    # Directory mode
    out_dir = Path(args.output) if (args.output and not args.in_place) else None
    stats = reduce_silence_in_dir(
        src,
        output_dir=out_dir,
        pattern=args.pattern,
        in_place=bool(args.in_place),
        silence_dbfs=args.silence_dbfs,
        threshold_sec=args.threshold_sec,
        target_sec=args.target_sec,
        frame_ms=args.frame_ms,
    )

    total = len(stats)
    reduced = sum(1 for s in stats if s.reduced_regions > 0)
    logging.info("Processed %d files; reduced silence in %d files", total, reduced)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
