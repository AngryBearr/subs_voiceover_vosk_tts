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


MAX_DRIFT_S = 1.5  # CTC times further than this from a matching reference word are treated as drifted
MIN_DRIFT_RUN = 3  # ... when at least this many matched words in a row drift


def anchor_to_reference(words: list[Word], reference: list[Word], max_drift: float = MAX_DRIFT_S,
                        min_run: int = MIN_DRIFT_RUN) -> list[Word]:
    """Replace drifted forced-alignment times by the times of the same words in a reference recognition.

    wav2vec2 CTC alignment loses its place in shouting, chanting and music: a whole stretch of words
    lands seconds away from where it was said (and the real speech ends up with no words). A word that
    also stands in the reference (Whisper's own word times, matched by text) and differs from it by more
    than `max_drift` seconds, in a run of at least `min_run` such words (a lone mismatch is more likely
    the reference's own timing error), takes the reference time; words without a match that no longer fit between
    their matched neighbours are spread evenly between those neighbours. Words are never dropped.
    """
    if not words or not reference:
        return words
    lo, hi = words[0].start - 2.0, words[-1].end + 2.0
    ref = [w for w in reference if w.end > lo and w.start < hi]
    keys = [_key(w.text) for w in words]
    matcher = difflib.SequenceMatcher(a=keys, b=[_key(w.text) for w in ref], autojunk=False)
    times: list[tuple[float, float]] = [(w.start, w.end) for w in words]
    matched = [False] * len(words)
    pairs = [(block.a + k, block.b + k) for block in matcher.get_matching_blocks() for k in range(block.size)
             if keys[block.a + k]]
    drifting = [abs(words[i].start - ref[j].start) > max_drift for i, j in pairs]
    changed = False
    for n, (i, j) in enumerate(pairs):
        matched[i] = True
        lo_n = n
        while lo_n > 0 and drifting[lo_n - 1]:
            lo_n -= 1
        hi_n = n
        while hi_n + 1 < len(pairs) and drifting[hi_n + 1]:
            hi_n += 1
        if drifting[n] and hi_n - lo_n + 1 >= min_run:
            times[i] = (ref[j].start, max(ref[j].end, ref[j].start))
            changed = True
    if not changed:
        return words
    i = 0
    while i < len(words):  # unmatched runs: keep their times only if they still fit between the neighbours
        if matched[i]:
            i += 1
            continue
        j = i
        while j < len(words) and not matched[j]:
            j += 1
        left = times[i - 1][1] if i > 0 else None
        right = times[j][0] if j < len(words) else None
        fits = all((left is None or times[k][0] >= left - 1e-6) and (right is None or times[k][1] <= right + 1e-6)
                   for k in range(i, j))
        if not fits and (left is not None or right is not None):
            a = left if left is not None else right - 0.2 * (j - i)  # type: ignore[operator]
            b = right if right is not None else a + 0.2 * (j - i)
            b = max(a, b)
            step = (b - a) / (j - i)
            for k in range(i, j):
                times[k] = (a + (k - i) * step, a + (k - i + 1) * step)
        i = j
    out: list[Word] = []
    prev_end = 0.0
    for word, (start, end) in zip(words, times):
        start = max(start, prev_end)
        end = max(end, start)
        out.append(Word(start=round(start, 3), end=round(end, 3), text=word.text))
        prev_end = end
    return out


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

    def align(self, audio: np.ndarray, segments: list[Segment], reference: list[Word] | None = None) -> list[Segment]:
        """Return copies of `segments` with `words` filled from forced alignment.

        `reference` (e.g. Whisper's timed words) corrects stretches where the CTC alignment drifted.
        """
        out: list[Segment] = []
        for seg in segments:
            if not seg.text.strip():
                continue
            piece = audio[int(seg.start * SAMPLE_RATE) : int(seg.end * SAMPLE_RATE)]
            aligned = self.align_segment(piece, seg.text)
            if not aligned:  # nothing alignable: spread the words evenly over the segment
                aligned = [("", 0.0, seg.end - seg.start)]
            words = attach_times(seg.text.split(), aligned, seg.start)
            if reference:
                words = anchor_to_reference(words, reference)
            out.append(Segment(start=seg.start, end=seg.end, text=seg.text, words=words, speaker=seg.speaker))
        return out
