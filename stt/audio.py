"""Media decoding helpers (PyAV-based, no system ffmpeg required)."""

from __future__ import annotations

import logging
from pathlib import Path

import av
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000

log = logging.getLogger(__name__)


def load_audio(
    path: str | Path,
    *,
    start: float = 0.0,
    duration: float | None = None,
    sample_rate: int = SAMPLE_RATE,
    channels: int = 1,
) -> np.ndarray:
    """Decode the first audio stream of a media file to float32 (mono, or stereo with `channels=2`).

    `start` and `duration` (seconds) select an interval on the media presentation
    timeline, i.e. the same timeline subtitles use. Returned length is exact; stereo
    audio has shape (samples, 2).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Media not found: {path}")
    if start < 0 or (duration is not None and duration <= 0):
        raise ValueError("start must be >= 0 and duration must be > 0")

    first = int(round(start * sample_rate))
    last = None if duration is None else first + int(round(duration * sample_rate))
    if channels not in (1, 2):
        raise ValueError("channels must be 1 or 2")
    resampler = av.AudioResampler(format="flt", layout="mono" if channels == 1 else "stereo", rate=sample_rate)
    pieces: list[np.ndarray] = []
    origin: int | None = None  # target-rate sample index of the first decoded sample
    decoded = 0

    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError(f"No audio stream in {path}")
        if start > 0:
            container.seek(int(start / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if origin is None:
                origin = int(round((frame.time or 0.0) * sample_rate))
            for out in resampler.resample(frame):
                data = out.to_ndarray().reshape(-1, channels)
                pieces.append(data)
                decoded += len(data)
            if last is not None and origin + decoded >= last:
                break
        for out in resampler.resample(None):
            pieces.append(out.to_ndarray().reshape(-1, channels))

    if pieces:
        audio = np.concatenate(pieces).astype(np.float32, copy=False)
    else:
        audio = np.zeros((0, channels), np.float32)
    origin = origin or 0
    if origin > first:
        # Stream starts after the requested position: pad with silence.
        audio = np.concatenate([np.zeros((origin - first, channels), np.float32), audio])
        origin = first
    audio = audio[first - origin :]
    if last is not None:
        wanted = last - first
        if len(audio) < wanted:
            log.warning("Media ended %.2fs before the requested interval end", (wanted - len(audio)) / sample_rate)
        audio = audio[:wanted]
    return np.ascontiguousarray(audio[:, 0] if channels == 1 else audio)


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write mono float audio as 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16")


def read_wav(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read a mono WAV written by `save_wav` (sample rate must match)."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != sample_rate:
        raise ValueError(f"{path}: expected {sample_rate} Hz, got {sr} Hz")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)
