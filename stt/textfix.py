"""Model-agnostic clean-up of ASR output: collapse hallucinated repetition loops."""

from __future__ import annotations

import re
from dataclasses import replace

from stt.types import Segment

_KEY = re.compile(r"[^\w']+")


def _key(token: str) -> str:
    return _KEY.sub("", token.lower())


def repeat_keep_indices(tokens: list[str], *, max_repeats: int = 4, max_n: int = 4) -> list[int]:
    """Indices of tokens to keep after cutting any n-gram (n <= max_n) repeated > max_repeats times in a row.

    Autoregressive decoders (Whisper, speech-LLMs) can loop on chants or noise
    ("go, go, go, ..." x500); real speech rarely repeats a phrase more than a few times.
    """
    keys = [_key(t) for t in tokens]
    keep: list[int] = []
    i, total = 0, len(tokens)
    while i < total:
        best: tuple[int, int] | None = None  # (repeats, n)
        for n in range(1, max_n + 1):
            gram = keys[i : i + n]
            if len(gram) < n or not all(gram):
                break
            repeats = 1
            while keys[i + repeats * n : i + (repeats + 1) * n] == gram:
                repeats += 1
            if repeats > max_repeats and (best is None or repeats * n > best[0] * best[1]):
                best = (repeats, n)
        if best:
            repeats, n = best
            keep.extend(range(i, i + max_repeats * n))
            i += repeats * n
        else:
            keep.append(i)
            i += 1
    return keep


def collapse_repeats(segments: list[Segment], *, max_repeats: int = 4) -> tuple[list[Segment], int]:
    """Apply the repetition guard to words (if timed) or text; returns (segments, removed token count)."""
    cleaned: list[Segment] = []
    removed = 0
    for seg in segments:
        if seg.words:
            keep = repeat_keep_indices([w.text for w in seg.words], max_repeats=max_repeats)
            if len(keep) < len(seg.words):
                words = [seg.words[i] for i in keep]
                removed += len(seg.words) - len(words)
                seg = replace(seg, words=words, text=" ".join(w.text for w in words))
        else:
            tokens = seg.text.split()
            keep = repeat_keep_indices(tokens, max_repeats=max_repeats)
            if len(keep) < len(tokens):
                removed += len(tokens) - len(keep)
                seg = replace(seg, text=" ".join(tokens[i] for i in keep))
        cleaned.append(seg)
    return cleaned, removed
