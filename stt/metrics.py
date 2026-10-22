"""ASR accuracy metrics (WER/CER) with Whisper's English text normalization."""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Any

import jiwer


@lru_cache(maxsize=1)
def _normalizer() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        from whisper_normalizer.english import EnglishTextNormalizer

    return EnglishTextNormalizer()


def normalize(text: str) -> str:
    """Lowercase, expand contractions, unify numbers/spelling, drop punctuation."""
    return " ".join(_normalizer()(text).split())


def error_rates(reference: str, hypothesis: str) -> dict[str, Any]:
    """WER/CER and edit counts after normalization of both texts."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        raise ValueError("Reference is empty after normalization")
    words = jiwer.process_words(ref, hyp or "<empty>")
    return {
        "wer": round(words.wer, 4),
        "cer": round(jiwer.cer(ref, hyp or "<empty>"), 4),
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "ref_words": len(ref.split()),
        "hyp_words": len(hyp.split()),
    }


def alignment_report(reference: str, hypothesis: str) -> str:
    """Human-readable word alignment (REF/HYP with S/D/I marks) for error inspection."""
    words = jiwer.process_words(normalize(reference), normalize(hypothesis) or "<empty>")
    return jiwer.visualize_alignment(words, show_measures=False, line_width=120)
