"""Which syllable a take stressed in the words whose stress was checked, heard by TeraAnaly.

TeraAnaly's stress head (TeraSpace/TeraAnaly, CTC-aligned to the given text) scores every vowel of
every word; the vowel scores of a word are turned into probabilities here. A word is heard wrong
when another vowel won and the marked one got less than ``WRONG_BELOW``. Only the words the stress
check looked at are listened to (homographs, names, guessed and disputed words): on all words the
voices follow the marks 98-99% of the time and TeraAnaly's own errors would outnumber theirs.
Words with "ё" or a single vowel are skipped (their stress is not in question).
"""

from __future__ import annotations

import json
import logging
import re
import sys
import unicodedata
from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("voiceover.stress_listen")

REPO = "TeraSpace/TeraAnaly"
REVISION = "d1840efbf94886681cb1b8e5d556b2e7074f8cc8"
VOWELS = "аеёиоуыэюя"
WRONG_BELOW = 0.25  # the marked vowel's probability under which another winning vowel counts
STRESS_SCORE = 0.3  # take score per listened word heard with another stress (as much as a garbled word)
_WORD = re.compile(r"[А-Яа-яЁё+]+")


def fold(word: str) -> str:
    return word.lower().replace("ё", "е").replace("+", "")


def listen_words(stress_result: dict[str, Any]) -> set[str]:
    """Words of a ``voiceover.stress_check`` result worth listening to (folded, without marks)."""
    words = {fold(v["word"]) for v in stress_result.get("verdicts", [])}
    words |= {fold(w) for w in stress_result.get("dictionary", {})}
    for fixes in stress_result.get("sentences", {}).values():
        words |= {fold(w) for w in fixes}
    return {w for w in words if sum(ch in VOWELS for ch in w) >= 2}


def words_in(marked: str, words: Collection[str]) -> list[str]:
    """The listened words that occur in an engine text, in order (repeats kept)."""
    return [fold(w) for w in _WORD.findall(marked) if fold(w) in words]


def _plain_with_stress(marked: str) -> tuple[str, set[int]]:
    """Text without '+' (NFKC, single spaces, as TeraAnaly aligns it) and the indices of marked vowels."""
    marked = " ".join(unicodedata.normalize("NFKC", marked).split())
    plain, stressed = [], set()
    for i, ch in enumerate(marked):
        if ch == "+":
            continue
        if i and marked[i - 1] == "+" and ch.lower() in VOWELS:
            stressed.add(len(plain))
        plain.append(ch)
    return "".join(plain), stressed


class StressListener:
    """TeraAnaly loaded once; call with a take, its engine text and the words to listen to."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.model: Any = None
        self._module: Any = None

    def load(self) -> None:
        if self.model is not None:
            return
        from huggingface_hub import snapshot_download
        from transformers import AutoModel

        source = snapshot_download(REPO, revision=REVISION)
        self.model = AutoModel.from_pretrained(source, trust_remote_code=True, device=self.device)
        self._module = sys.modules[type(self.model).__module__]

    def __call__(self, audio: np.ndarray, rate: int, marked: str, words: Collection[str]) -> list[dict[str, Any]]:
        """[{"word", "want", "heard", "p_want", "wrong"}] for each listened word of the text (vowel indices)."""
        import torch

        plain, stressed = _plain_with_stress(marked)
        if not any(fold(w) in words for w in _WORD.findall(plain)):
            return []
        self.load()
        head = self.model.stress_head
        with torch.inference_mode():
            states = self.model.encoder(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)), rate)
            aligned = head.ctc_head.align(states, plain)
            features, found = self._module.wav2vec_vowel_features(aligned, states["hidden_states"][head.layer],
                                                                  hidden_size=1024)
            scores = head.ranker(features).squeeze(-1).float().cpu().numpy()
        out = []
        for word in found:
            text = word["word"]
            if fold(text) not in words or "ё" in text.lower() or len(word["vowels"]) < 2:
                continue
            marks = [n for n, index in enumerate(word["vowels"]) if index in stressed]
            if len(marks) != 1:
                continue
            row = scores[word["row_start"]:word["row_end"]]
            probs = np.exp(row - row.max())
            probs /= probs.sum()
            want, heard = marks[0], int(probs.argmax())
            p_want = float(probs[want])
            out.append({"word": fold(text), "want": want, "heard": heard, "p_want": round(p_want, 3),
                        "wrong": heard != want and p_want < WRONG_BELOW})
        return out


def hear(listener: StressListener, audio: np.ndarray, rate: int, marked: str, words: Collection[str]
         ) -> list[dict[str, Any]]:
    """``listener(...)``, or nothing when TeraAnaly cannot align the take (garbled audio): the speech
    check judges such a take."""
    try:
        return listener(audio, rate, marked, words)
    except Exception as exc:  # noqa: BLE001 - alignment failure on one take must not stop the run
        log.warning("stress not heard (%s): %s", str(exc)[:120], marked[:60])
        return []


def wrong_words(heard: list[dict[str, Any]]) -> int:
    return sum(bool(w["wrong"]) for w in heard)


def record(words: Collection[str], heard: list[dict[str, Any]]) -> dict[str, Any]:
    """What a cache entry keeps of the listening: the words, the ones heard wrong, the details."""
    return {"words": sorted(set(words)), "wrong": [w["word"] for w in heard if w["wrong"]], "heard": heard}


def listened(entry: dict[str, Any], words: Collection[str]) -> bool:
    """The cached take was already listened to for these words."""
    return (entry.get("stress") or {}).get("words") == sorted(set(words))


def load_listen_words(path: str | Path | None) -> set[str]:
    """``listen_words`` of a stress check result file; empty when there is none."""
    if not path or not Path(str(path)).is_file():
        return set()
    return listen_words(json.loads(Path(str(path)).read_text(encoding="utf-8")))
