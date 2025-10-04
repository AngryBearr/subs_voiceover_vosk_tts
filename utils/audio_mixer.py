"""Efficient audio segment mixer for subtitle-aligned TTS outputs.

This utility mixes a large number of short WAV (or other libsndfile-supported)
segments onto a single timeline defined by a JSON subtitles file. Compared to
the iterative pydub overlay approach, this performs a single preallocation and
linear accumulation pass, yielding substantially lower CPU and memory overhead
for thousands of segments.

Expected JSON structure (list of dicts):
    {
        "index": int,        # numeric id matching <index>.wav file name
        "start": int,        # start time in milliseconds
        "end": int,          # end time in milliseconds (used to derive total length)
        ...                   # other fields ignored
    }

Key features:
    * Single allocation mix buffer (int64 accumulator for int16 output or float32 path)
    * Optional resampling (uses python-soxr if installed, else a simple linear fallback)
    * Optional normalization / peak limiting to avoid clipping
    * Optional transcode step (ffmpeg) for final compressed output
    * CLI entry point for quick usage

Usage (module):
    from pathlib import Path
    from utils.audio_mixer import mix_segments
    mix_segments(Path('subs.json'), Path('tts_out'), Path('mixed/output'))

CLI example:
    python -m utils.audio_mixer subs_chunk.json tts_out output/mix --sr 48000 --channels 1

Notes:
    * For very long durations that exceed comfortable RAM, a streaming/windowed
      variant can be added later (two-pass or chunk-based). For typical subtitle
      voice-over sessions (< 1 hour, mono 48 kHz) this approach is efficient.
    * The function auto-extends the buffer if a segment exceeds previously
      calculated total length (defensive against timing inconsistencies).

Author: Automated assistant (initial implementation)
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Literal, Optional, List

import numpy as np
import soundfile as sf  # pip install soundfile

try:  # Optional high-quality resampling
    import soxr  # type: ignore
except Exception:  # pragma: no cover - optional
    soxr = None  # fallback handled below


def _resample_if_needed(data: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate if required.

    Args:
        data: (samples, channels) float64 array.
        sr: Original sample rate.
        target_sr: Desired sample rate.

    Returns:
        Possibly resampled array (float64, always 2D).
    """
    if sr == target_sr:
        return data
    if soxr is not None:  # High quality path
        return soxr.resample(data, sr, target_sr)
    # Simple linear interpolation fallback per channel
    ratio = target_sr / sr
    new_len = int(round(data.shape[0] * ratio))
    if new_len <= 1:
        return data[:1]
    x_old = np.linspace(0.0, 1.0, data.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
    return np.stack([
        np.interp(x_new, x_old, data[:, ch]) for ch in range(data.shape[1])
    ], axis=1)


def _match_channels(data: np.ndarray, target_channels: int) -> np.ndarray:
    """Adapt channel count (mix down / duplicate / pad) to target."""
    cur = data.shape[1]
    if cur == target_channels:
        return data
    if target_channels == 1:  # Mix down
        return data.mean(axis=1, keepdims=True)
    if cur == 1 and target_channels > 1:  # Duplicate mono
        return np.repeat(data, target_channels, axis=1)
    # General pad or truncate
    if cur > target_channels:
        return data[:, :target_channels]
    pad = np.zeros((data.shape[0], target_channels - cur), dtype=data.dtype)
    return np.concatenate([data, pad], axis=1)


def _read_audio(path: Path, target_sr: int, target_channels: int) -> np.ndarray:
    """Read an audio file to float64 2D array at desired sample rate/channels."""
    data, sr = sf.read(str(path), always_2d=True)
    data = _resample_if_needed(data, sr, target_sr)
    data = _match_channels(data, target_channels)
    return data  # float64


def mix_segments(
    subtitles_json: Path,
    segments_folder: Path,
    output_audio: Path,
    sample_rate: int = 48000,
    channels: int = 1,
    dtype: Literal["int16", "float32"] = "int16",
    normalize: bool = True,
    limiter_margin_db: float = 0.5,
    expected_index_ext: str = ".wav",
    progress_every: int = 250,
    mp3_bitrate: str = "192k",
    encode_format: str = "wav",
    transcode_to: Optional[str] = "mp3",
    base_audio_path: Optional[Path] = None,
    base_gain_db: float = 0.0,
    speech_gain_db: float = 0.0,
    total_duration_ms: Optional[int] = None,
    missing_policy: Literal["skip", "warn", "error"] = "skip",
) -> Path:
    """Efficiently mix segmented audio onto a single timeline.

    Args:
        subtitles_json: Path to JSON with timing info.
        segments_folder: Folder containing <index><ext> files.
        output_audio: Basename (path without final extension) for output.
        sample_rate: Target sample rate.
        channels: Target channel count.
        dtype: Internal accumulation/output format (int16 or float32).
        normalize: Whether to normalize/limit peaks.
        limiter_margin_db: Headroom below 0 dBFS after normalization.
        expected_index_ext: File extension for segment files (e.g. ".wav").
        progress_every: Print progress every N segments (0 disables).
        mp3_bitrate: Bitrate used if transcoding to mp3.
        encode_format: Container/format for initial PCM write ("wav").
    transcode_to: Optional final format ("mp3", "ogg", "opus", None to skip).
    base_audio_path: Optional existing bed / silence / music track to include.
    base_gain_db: Gain in dB applied to base track before mixing.
    speech_gain_db: Gain in dB applied to each speech segment before mixing.
    total_duration_ms: Force minimum total output duration (pad with silence).
    missing_policy: Handling for missing segments: skip | warn | error.

    Returns:
        Path to final written audio file.
    """
    with open(subtitles_json, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not entries:
        raise ValueError("Subtitle JSON is empty – nothing to mix.")

    # Defensive validate starts <= ends
    for e in entries:
        if e["start"] > e["end"]:
            raise ValueError(f"Entry index {e.get('index')} has start > end.")

    max_end_ms = max(e["end"] for e in entries)

    # Optional base track
    base_data = None
    base_len_samples = 0
    if base_audio_path:
        bdata, bsr = sf.read(str(base_audio_path), always_2d=True)
        if bsr != sample_rate:
            if soxr is not None:
                bdata = soxr.resample(bdata, bsr, sample_rate)
            else:
                ratio = sample_rate / bsr
                new_len = int(round(bdata.shape[0] * ratio))
                x_old = np.linspace(0, 1, bdata.shape[0], endpoint=False)
                x_new = np.linspace(0, 1, new_len, endpoint=False)
                bdata = np.stack([
                    np.interp(x_new, x_old, bdata[:, ch]) for ch in range(bdata.shape[1])
                ], axis=1)
        # Channel adapt
        bdata = _match_channels(bdata, channels)
        if base_gain_db:
            bdata *= 10 ** (base_gain_db / 20.0)
        base_data = bdata
        base_len_samples = bdata.shape[0]

    # Determine final duration
    chosen_ms = max_end_ms
    if total_duration_ms is not None:
        chosen_ms = max(chosen_ms, total_duration_ms)
    if base_data is not None:
        base_ms = int(round((base_len_samples / sample_rate) * 1000.0))
        chosen_ms = max(chosen_ms, base_ms)

    total_samples = math.ceil(sample_rate * (chosen_ms / 1000.0))
    if dtype == "int16":
        mix_buf = np.zeros((total_samples, channels), dtype=np.int64)
    else:
        mix_buf = np.zeros((total_samples, channels), dtype=np.float32)

    # Copy base first
    if base_data is not None:
        if dtype == "int16":
            mix_buf[:base_data.shape[0], :] += np.round(np.clip(base_data, -1, 1) * 32767).astype(np.int64)
        else:
            mix_buf[:base_data.shape[0], :] += base_data.astype(np.float32)

    speech_scale = 10 ** (speech_gain_db / 20.0) if speech_gain_db else 1.0
    missing_indices: List[int] = []

    for i, entry in enumerate(entries, 1):
        idx = entry["index"]
        start_ms = entry["start"]
        seg_path = segments_folder / f"{idx}{expected_index_ext}"
        if not seg_path.exists():
            missing_indices.append(idx)
            if missing_policy == "error":
                raise FileNotFoundError(f"Missing segment: {seg_path}")
            if missing_policy == "warn" and len(missing_indices) <= 10:
                print(f"[mix] Warning: missing {seg_path}")
            if progress_every and i % progress_every == 0:
                print(f"[mix] {i}/{len(entries)} processed (missing)")
            continue
        data = _read_audio(seg_path, sample_rate, channels)
        if speech_scale != 1.0:
            data *= speech_scale
        start_sample = int(round(start_ms * sample_rate / 1000.0))
        end_sample = start_sample + data.shape[0]
        if end_sample > mix_buf.shape[0]:
            extra = end_sample - mix_buf.shape[0]
            mix_buf = np.vstack([
                mix_buf,
                np.zeros((extra, channels), dtype=mix_buf.dtype),
            ])
        if dtype == "int16":
            seg_int = np.round(np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int64)
            mix_buf[start_sample:end_sample, :] += seg_int
        else:
            mix_buf[start_sample:end_sample, :] += data.astype(np.float32)
        if progress_every and i % progress_every == 0:
            print(f"[mix] {i}/{len(entries)} segments mixed")

    if missing_policy == "warn" and len(missing_indices) > 10:
        print(f"[mix] Total missing segments: {len(missing_indices)}")

    if dtype == "int16":
        peak = int(np.max(np.abs(mix_buf))) if mix_buf.size else 0
        if peak == 0:
            out_int16 = np.zeros_like(mix_buf, dtype=np.int16)
        else:
            target_peak = int(32767 * (10 ** (-limiter_margin_db / 20))) if normalize else min(peak, 32767)
            if peak > target_peak:
                scale = target_peak / peak
                mix_buf = np.round(mix_buf * scale).astype(np.int64)
            mix_buf = np.clip(mix_buf, -32768, 32767)
            out_int16 = mix_buf.astype(np.int16)
        pcm_path = output_audio.with_suffix(f".{encode_format}")
        sf.write(str(pcm_path), out_int16, sample_rate, subtype="PCM_16")
        written_path = pcm_path
    else:  # float32 path
        if normalize and mix_buf.size:
            peak = float(np.max(np.abs(mix_buf)))
            if peak > 0:
                target_lin = 10 ** (-limiter_margin_db / 20)
                if peak > target_lin:
                    mix_buf *= (target_lin / peak)
        pcm_path = output_audio.with_suffix(f".{encode_format}")
        sf.write(str(pcm_path), mix_buf, sample_rate, subtype="PCM_32")
        written_path = pcm_path

    final_path = written_path
    if transcode_to:
        final_path = output_audio.with_suffix(f".{transcode_to}")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(written_path),
        ]
        if transcode_to == "mp3":
            cmd += ["-c:a", "libmp3lame", "-b:a", mp3_bitrate]
        elif transcode_to in ("ogg", "opus"):
            codec = "libopus" if transcode_to == "opus" else "libvorbis"
            cmd += ["-c:a", codec, "-b:a", "128k"]
        else:
            cmd += ["-c:a", "copy"]
        cmd.append(str(final_path))
        subprocess.run(cmd, check=True)
    print(f"[mix] Wrote {final_path}")
    return final_path


def _build_arg_parser():  # pragma: no cover - CLI convenience
    import argparse
    ap = argparse.ArgumentParser(description="Efficiently mix segmented WAVs based on JSON subtitle timing.")
    ap.add_argument("subs_json", type=Path, help="Path to subtitles JSON")
    ap.add_argument("segments_dir", type=Path, help="Directory containing <index>.wav files")
    ap.add_argument("output_basename", type=Path, help="Output path WITHOUT extension (e.g. output/mix)")
    ap.add_argument("--sr", type=int, default=48000, help="Target sample rate")
    ap.add_argument("--channels", type=int, default=1, help="Target channel count")
    ap.add_argument("--dtype", choices=["int16", "float32"], default="int16", help="Internal/output dtype")
    ap.add_argument("--no-normalize", action="store_true", help="Disable normalization / limiting")
    ap.add_argument("--limiter-margin", type=float, default=0.5, help="Headroom dB below 0 dBFS")
    ap.add_argument("--ext", default=".wav", help="Segment file extension")
    ap.add_argument("--progress", type=int, default=250, help="Progress print interval (0 disables)")
    ap.add_argument("--mp3-bitrate", default="192k", help="Bitrate for mp3 transcode")
    ap.add_argument("--encode-format", default="wav", help="Intermediate PCM container (wav)")
    ap.add_argument("--transcode", default="mp3", help="Final format: mp3|ogg|opus|wav|none")
    ap.add_argument("--base-audio", type=Path, help="Optional base/bed audio file", default=None)
    ap.add_argument("--base-gain", type=float, default=0.0, help="Gain dB applied to base track")
    ap.add_argument("--speech-gain", type=float, default=0.0, help="Gain dB applied to each speech segment")
    ap.add_argument("--total-duration-ms", type=int, default=None, help="Force minimum total duration in ms")
    ap.add_argument("--missing-policy", choices=["skip", "warn", "error"], default="skip", help="Handle missing segments")
    return ap


def main():  # pragma: no cover - CLI convenience
    parser = _build_arg_parser()
    args = parser.parse_args()
    trans = None if args.transcode == "none" else args.transcode
    mix_segments(
        subtitles_json=args.subs_json,
        segments_folder=args.segments_dir,
        output_audio=args.output_basename,
        sample_rate=args.sr,
        channels=args.channels,
        dtype=args.dtype,
        normalize=not args.no_normalize,
        limiter_margin_db=args.limiter_margin,
        expected_index_ext=args.ext,
        progress_every=args.progress,
        mp3_bitrate=args.mp3_bitrate,
        encode_format=args.encode_format,
        transcode_to=trans,
        base_audio_path=args.base_audio,
        base_gain_db=args.base_gain,
        speech_gain_db=args.speech_gain,
        total_duration_ms=args.total_duration_ms,
        missing_policy=args.missing_policy,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
