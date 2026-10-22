"""Forced alignment: word start/end times for ASR text that came without them.

Uses Qwen3-ForcedAligner-0.6B (`qwen-asr` package, `stt_qwen_env`). Lets text-only models
(Cohere, LLM-fused text) feed the speaker pipeline.
"""

from __future__ import annotations

import difflib
import re

import numpy as np

from stt.audio import SAMPLE_RATE
from stt.types import Segment, Word

ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"

_KEY = re.compile(r"[^\w']+")


def _key(token: str) -> str:
    return _KEY.sub("", token.lower())


def attach_times(tokens: list[str], aligned: list[tuple[str, float, float]], offset: float) -> list[Word]:
    """Give each original token (with punctuation) the times of its aligned counterpart.

    Tokens without a match (e.g. pure punctuation, normalization differences) take the
    neighbouring words' boundaries so every token keeps a monotonic time.
    """
    keys = [_key(t) for t in tokens]
    akeys = [_key(a[0]) for a in aligned]
    times: list[tuple[float, float] | None] = [None] * len(tokens)
    matcher = difflib.SequenceMatcher(a=keys, b=akeys, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                _, start, end = aligned[j1 + k]
                times[i1 + k] = (start, end)
        elif tag == "replace" and j2 > j1:  # spread the aligned span over the tokens
            start, end = aligned[j1][1], aligned[j2 - 1][2]
            step = (end - start) / (i2 - i1)
            for k in range(i2 - i1):
                times[i1 + k] = (start + k * step, start + (k + 1) * step)
    words: list[Word] = []
    prev_end = 0.0
    for idx, token in enumerate(tokens):
        span = times[idx]
        if span is None:
            nxt = next((t[0] for t in times[idx + 1 :] if t is not None), prev_end)
            span = (prev_end, max(prev_end, nxt))
        start, end = max(span[0], prev_end), max(span[1], span[0], prev_end)
        words.append(Word(start=round(offset + start, 3), end=round(offset + end, 3), text=token))
        prev_end = end
    return words


class ForcedAligner:
    def __init__(self, device: str = "cuda", batch_size: int = 8, language: str = "English") -> None:
        self.device = device
        self.batch_size = batch_size
        self.language = language

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ForcedAligner

        self._aligner = Qwen3ForcedAligner.from_pretrained(
            ALIGNER_MODEL, dtype=torch.bfloat16, device_map=self.device
        )

    def align(self, audio: np.ndarray, segments: list[Segment]) -> list[Segment]:
        """Return copies of `segments` with `words` filled from forced alignment."""
        todo = [s for s in segments if s.text.strip()]
        out: list[Segment] = []
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i : i + self.batch_size]
            waves = [(audio[int(s.start * SAMPLE_RATE) : int(s.end * SAMPLE_RATE)], SAMPLE_RATE) for s in batch]
            results = self._aligner.align(
                audio=waves, text=[s.text for s in batch], language=[self.language] * len(batch)
            )
            for seg, result in zip(batch, results):
                aligned = [(str(it.text), float(it.start_time), float(it.end_time)) for it in result]
                words = attach_times(seg.text.split(), aligned, seg.start)
                out.append(Segment(start=seg.start, end=seg.end, text=seg.text, words=words, speaker=seg.speaker))
        return out
