"""Speech activity detection and ASR chunking shared by all ASR backends."""

from __future__ import annotations

import numpy as np

from stt.audio import SAMPLE_RATE

Chunk = tuple[float, float]

# Sensitive defaults: on TV audio (music, cheering) the usual 0.5 threshold dropped 4-11% of
# subtitled speech before ASR; 0.15 + 0.5 s padding keeps ~99-100% at ~15% more audio.
VAD_THRESHOLD = 0.15
VAD_PAD_MS = 500


def speech_regions(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    threshold: float = VAD_THRESHOLD,
    min_silence_ms: int = 300,
    speech_pad_ms: int = VAD_PAD_MS,
    max_region_s: float = 28.0,
) -> list[Chunk]:
    """Detect speech regions (seconds) with Silero VAD (bundled with faster-whisper)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        max_speech_duration_s=max_region_s,
    )
    stamps = get_speech_timestamps(audio, options, sampling_rate=sample_rate)
    return [(s["start"] / sample_rate, s["end"] / sample_rate) for s in stamps]


def merge_regions(regions: list[Chunk], *, max_chunk_s: float = 28.0, max_gap_s: float = 2.0) -> list[Chunk]:
    """Merge neighbouring speech regions into ASR chunks of at most `max_chunk_s`.

    Regions are joined while the merged span stays within the limit and the silence
    between them is short, so models get sentence context without long music gaps.
    """
    chunks: list[Chunk] = []
    for start, end in regions:
        if chunks:
            prev_start, prev_end = chunks[-1]
            if end - prev_start <= max_chunk_s and start - prev_end <= max_gap_s:
                chunks[-1] = (prev_start, max(prev_end, end))
                continue
        chunks.append((start, end))
    return chunks


def asr_chunks(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    max_chunk_s: float = 28.0,
    threshold: float = VAD_THRESHOLD,
    speech_pad_ms: int = VAD_PAD_MS,
) -> list[Chunk]:
    """VAD-based chunks (seconds) suitable for every ASR backend (< 30 s each)."""
    regions = speech_regions(
        audio, sample_rate=sample_rate, max_region_s=max_chunk_s, threshold=threshold, speech_pad_ms=speech_pad_ms
    )
    return merge_regions(regions, max_chunk_s=max_chunk_s)


def vad_settings() -> dict[str, float]:
    """Current VAD parameters, stored next to cached chunks to invalidate them on change."""
    return {"threshold": VAD_THRESHOLD, "pad_ms": VAD_PAD_MS}
