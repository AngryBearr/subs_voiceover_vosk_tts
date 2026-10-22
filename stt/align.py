"""Forced alignment: word start/end times for ASR text that came without them.

Lets text-only models (Cohere, LLM-fused text) feed the speaker pipeline. Two aligners:

- `ctc` (default): character-level CTC alignment with the English wav2vec2 model
  WAV2VEC2_ASR_LARGE_LV60K_960H (torchaudio), as in WhisperX; runs in `subs_env`.
- `qwen`: Qwen3-ForcedAligner-0.6B (`qwen-asr` package, `stt_qwen_env`). On reality TV it gave
  7-11% zero-length words and moved ~8% of words by more than 0.5 s against Whisper's times,
  which made Russian lines start before the English speech.
"""

from __future__ import annotations

import difflib
import re

import numpy as np

from stt.audio import SAMPLE_RATE
from stt.types import Segment, Word

ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
CTC_MODEL = "WAV2VEC2_ASR_LARGE_LV60K_960H"

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


def _ctc_word(token: str, vocab: dict[str, int]) -> str:
    """A token as the characters the CTC model knows; numbers are spelled out ("39" -> "THIRTY NINE")."""
    text = token.replace("’", "'")
    if re.search(r"\d", text):
        from num2words import num2words

        text = re.sub(r"\d+(?:[.,]\d+)?", lambda m: " " + num2words(float(m.group(0).replace(",", ""))
                                                                       if "." in m.group(0) else int(m.group(0).replace(",", ""))) + " ", text)
    return "|".join(w for w in ("".join(c for c in part.upper() if c in vocab and c not in "|-")
                                for part in re.split(r"[\s-]+", text)) if w)


class CtcAligner:
    """Character CTC forced alignment of each segment's text to its audio (English wav2vec2)."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    def load(self) -> None:
        import torchaudio

        bundle = getattr(torchaudio.pipelines, CTC_MODEL)
        self.model = bundle.get_model().to(self.device).eval()
        self.vocab = {label: i for i, label in enumerate(bundle.get_labels())}
        self.rate = bundle.sample_rate

    def align_segment(self, audio: np.ndarray, text: str) -> list[tuple[str, float, float]]:
        """[(token, start, end)] in seconds from the segment start, for tokens the model can spell."""
        import torch
        import torchaudio.functional as F

        tokens = [(t, _ctc_word(t, self.vocab)) for t in text.split()]
        tokens = [(t, w) for t, w in tokens if w]
        if not tokens or len(audio) < self.rate // 10:
            return []
        targets: list[int] = []
        owner: list[int] = []
        for k, (_, spelled) in enumerate(tokens):
            if targets:
                targets.append(self.vocab["|"])
                owner.append(-1)
            for ch in spelled:
                targets.append(self.vocab[ch])
                owner.append(k if ch != "|" else -1)
        with torch.inference_mode():
            wave = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None].to(self.device)
            emission, _ = self.model(wave)
            log_probs = torch.log_softmax(emission.float(), dim=-1)
        if log_probs.shape[1] < len(targets):
            return []  # more letters than frames: leave this segment to the even spread
        aligned, scores = F.forced_align(log_probs, torch.tensor([targets], device=log_probs.device), blank=0)
        spans = F.merge_tokens(aligned[0], scores[0].exp())
        frame = len(audio) / self.rate / log_probs.shape[1]
        bounds: dict[int, list[float]] = {}
        for span, k in zip(spans, owner):
            if k < 0:
                continue
            b = bounds.setdefault(k, [span.start * frame, span.end * frame])
            b[0], b[1] = min(b[0], span.start * frame), max(b[1], span.end * frame)
        return [(tokens[k][0], round(b[0], 3), round(b[1], 3)) for k, b in sorted(bounds.items())]

    def align(self, audio: np.ndarray, segments: list[Segment]) -> list[Segment]:
        """Return copies of `segments` with `words` filled from forced alignment."""
        out: list[Segment] = []
        for seg in segments:
            if not seg.text.strip():
                continue
            piece = audio[int(seg.start * SAMPLE_RATE) : int(seg.end * SAMPLE_RATE)]
            aligned = self.align_segment(piece, seg.text)
            if not aligned:  # nothing alignable: spread the words evenly over the segment
                aligned = [("", 0.0, seg.end - seg.start)]
            words = attach_times(seg.text.split(), aligned, seg.start)
            out.append(Segment(start=seg.start, end=seg.end, text=seg.text, words=words, speaker=seg.speaker))
        return out
