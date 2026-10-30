"""Word-level speaker relabelling: a Viterbi pass over per-word speaker embeddings.

The diarizer misses speaker changes without a pause (a one-second shout inside a long turn) and puts
boundaries a word or two off. Every word gets an embedding of a fixed window around its middle; each
diarizer speaker gets a centroid from its "clean" words (far from any change of the diarizer label).
A word's score for a speaker is its cosine to that speaker's centroid plus a bonus for the diarizer's
own label; switching speaker between consecutive words costs a penalty that fades with the pause
between them. Viterbi picks the best label sequence.

Optionally the season's voice bank adds states: a bank person with no cluster of their own in the
episode (merged into another person's cluster) becomes an extra state ``bank:<name>`` with the person's
clip centroid, so a word can go to them instead of the cluster they were merged into.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from stt.refine_speakers import RATE, Embed
from stt.types import Word

log = logging.getLogger("stt.boundaries")

_FORBIDDEN = -1e9
BANK_PREFIX = "bank:"  # label prefix of the voice bank states


@dataclass
class RelabelParams:
    model: str = "resnet293"  # stt.refine_speakers.make_embedder name
    window_s: float = 1.0  # audio window centred on each word's middle (fixed length: windows batch)
    switch_penalty: float = 0.3  # cost of a different speaker on consecutive words
    gap_relief_s: float = 0.6  # the penalty fades with the pause: penalty * exp(-gap / gap_relief_s)
    prior: float = 0.0  # bonus for keeping the diarizer's own label of the word
    change_prior: float = 0.15  # bonus for a speaker change where the diarizer has one (removing it costs this)
    clean_margin_s: float = 1.0  # words closer than this to a diarizer speaker change do not feed centroids
    min_clean_words: int = 3  # clean words a speaker needs to get a centroid
    passes: int = 1  # extra passes re-estimate the centroids from the new labels
    bank_margin: float = 0.05  # subtracted from a bank state's emission: it must beat a cluster clearly
    bank_represented: float = 0.5  # cosine from which a cluster counts as the voice of its closest bank person
    bank_min_s: float = 0.0  # a run of a bank label shorter than this goes back to the words' own labels


def word_windows(words: list[Word], audio: np.ndarray, window_s: float) -> np.ndarray:
    """(n, L) windows of ``audio`` centred on each word's middle; zero-padded at the audio edges."""
    length = int(round(window_s * RATE))
    out = np.zeros((len(words), length), dtype=np.float32)
    for i, w in enumerate(words):
        first = int(round((w.start + w.end) / 2 * RATE)) - length // 2
        lo, hi = max(first, 0), min(first + length, len(audio))
        if hi > lo:
            out[i, lo - first:hi - first] = audio[lo:hi]
    return out


def embed_windows(windows: np.ndarray, embedder: Any) -> np.ndarray:
    """Unit embeddings (n, d) of the windows: ``embedder.batch`` when it exists, else per item."""
    if not len(windows):
        return np.zeros((0, 0), dtype=np.float32)
    if hasattr(embedder, "batch"):
        return np.asarray(embedder.batch(windows), dtype=np.float32)
    return np.stack([np.asarray(embedder(w), dtype=np.float32) for w in windows])


def _load_cache(cache: Path, words: list[Word], window_s: float) -> np.ndarray | None:
    try:
        with np.load(cache) as data:
            same = (np.array_equal(data["starts"], _times(words, "start"))
                    and np.array_equal(data["ends"], _times(words, "end"))
                    and float(data["window_s"]) == window_s)
            return data["emb"] if same else None
    except (OSError, KeyError, ValueError):
        return None


def _times(words: list[Word], attr: str) -> np.ndarray:
    return np.array([round(getattr(w, attr), 3) for w in words], dtype=np.float64)


def _clean_mask(words: list[Word], labels: list[str | None], margin: float) -> np.ndarray:
    """Words that may feed centroids: not OCR, labelled, and not near a change of the label."""
    changes = np.array([words[i].start for i in range(1, len(words)) if labels[i] != labels[i - 1]])
    mask = np.zeros(len(words), dtype=bool)
    for i, w in enumerate(words):
        if w.source == "ocr" or labels[i] is None:
            continue
        if len(changes):
            k = np.searchsorted(changes, w.start - margin, side="left")
            if k < len(changes) and changes[k] <= w.end + margin:
                continue
        mask[i] = True
    return mask


def _centroids(words: list[Word], labels: list[str | None], emb: np.ndarray,
               params: RelabelParams) -> dict[str, np.ndarray]:
    clean = _clean_mask(words, labels, params.clean_margin_s)
    out: dict[str, np.ndarray] = {}
    for label in {labels[i] for i in np.flatnonzero(clean)}:
        if str(label).startswith(BANK_PREFIX):  # bank vectors stay as given
            continue
        idx = [i for i in np.flatnonzero(clean) if labels[i] == label]
        if len(idx) >= params.min_clean_words:
            mean = emb[idx].sum(axis=0)
            out[str(label)] = mean / (np.linalg.norm(mean) + 1e-9)
    return out


def _viterbi(words: list[Word], own: list[str | None], emb: np.ndarray, centroids: dict[str, np.ndarray],
             params: RelabelParams, bank: dict[str, np.ndarray] | None = None) -> list[str | None]:
    # states: cluster labels with a centroid, bank states, then the other labels seen (they only keep
    # their own words); the first len(cent) states are the ones with a vector
    bank = {BANK_PREFIX + name: vec for name, vec in (bank or {}).items()}
    states: list[str | None] = sorted(centroids) + sorted(bank)
    states += [s for s in dict.fromkeys(own) if s not in centroids and s not in bank and s is not None]
    n, k = len(words), len(states)
    index = {s: j for j, s in enumerate(states)}
    cent = np.stack([centroids[s] if s in centroids else bank[s] for s in states[:len(centroids) + len(bank)]])
    sims = emb @ cent.T  # (n, centred states)
    sims[:, len(centroids):] -= params.bank_margin  # bank states: no prior, no change bonus, a margin
    emis = np.full((n, k), _FORBIDDEN)
    for i, w in enumerate(words):
        if w.source == "ocr" and own[i] is None:  # no label to keep: any state, restored below
            emis[i, :] = 0.0
            continue
        if w.source == "ocr" or (own[i] is not None and own[i] not in centroids):
            emis[i, index[own[i]]] = params.prior  # fixed to its label
            continue
        emis[i, :len(cent)] = sims[i]
        if own[i] is not None:
            emis[i, index[own[i]]] += params.prior
    dp = emis[0].copy()
    back = np.zeros((n, k), dtype=np.int64)
    for i in range(1, n):
        gap = max(words[i].start - words[i - 1].end, 0.0)
        if own[i] != own[i - 1] and own[i] is not None and own[i - 1] is not None:
            penalty = -params.change_prior  # the diarizer changes speaker here: a switch is rewarded
        else:
            penalty = params.switch_penalty * np.exp(-gap / params.gap_relief_s)
        stay = dp  # same state
        order = np.argsort(-dp)  # best and second-best predecessor for a switch (never the state itself)
        first, second = int(order[0]), int(order[1]) if k > 1 else int(order[0])
        switch_from = np.where(np.arange(k) == first, second, first)
        switch = dp[switch_from] - penalty
        take_stay = stay >= switch
        back[i] = np.where(take_stay, np.arange(k), switch_from)
        dp = np.where(take_stay, stay, switch) + emis[i]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmax(dp))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    result: list[str | None] = [states[j] for j in path]
    for i, w in enumerate(words):
        if w.source == "ocr":
            result[i] = own[i]
    return result


def _changes(labels: list[str | None]) -> int:
    return sum(a != b for a, b in zip(labels, labels[1:]))


def _drop_short_bank_runs(words: list[Word], labels: list[str | None], own: list[str | None],
                          min_s: float) -> list[str | None]:
    """Runs of one bank label shorter than ``min_s`` seconds fall back to each word's own label."""
    out = list(labels)
    i = 0
    while i < len(out):
        j = i
        while j + 1 < len(out) and out[j + 1] == out[i]:
            j += 1
        if str(out[i]).startswith(BANK_PREFIX) and words[j].end - words[i].start < min_s:
            out[i:j + 1] = own[i:j + 1]
        i = j + 1
    return out


def bank_centroids(references: Path, embedder: Any, window_s: float) -> dict[str, np.ndarray]:
    """Unit mean embedding per person of the clips in ``<references>/references.json`` (voice bank export).

    Clips are resampled to 16 kHz and cut into ``window_s`` windows every ``window_s / 2``; nearly
    silent windows (the clip's trailing silence) are skipped; the windows are embedded like the words.
    """
    import soundfile as sf
    import soxr

    index = json.loads((references / "references.json").read_text(encoding="utf-8"))
    length = int(round(window_s * RATE))
    out: dict[str, np.ndarray] = {}
    for ref in index:
        audio, rate = sf.read(str(references / ref["wav"]), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != RATE:
            audio = soxr.resample(audio, rate, RATE).astype(np.float32)
        loud = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        starts = range(0, max(len(audio) - length, 0) + 1, max(length // 2, 1))
        windows = [np.pad(audio[a:a + length], (0, max(a + length - len(audio), 0))) for a in starts]
        windows = [w for w in windows if loud > 0 and float(np.sqrt(np.mean(w ** 2))) >= 0.2 * loud]
        if not windows:
            log.warning("Voice bank clip of %s has no usable windows", ref["name"])
            continue
        mean = embed_windows(np.stack(windows), embedder).sum(axis=0)
        out[ref["name"]] = (mean / (np.linalg.norm(mean) + 1e-9)).astype(np.float32)
    return out


def cached_bank_centroids(references: Path, embedder: Any, window_s: float, cache: Path) -> dict[str, np.ndarray]:
    """``bank_centroids`` cached in ``cache`` (an .npz outside the references folder)."""
    index = json.loads((references / "references.json").read_text(encoding="utf-8"))
    key = json.dumps([[r["name"], r["wav"], (references / r["wav"]).stat().st_size] for r in index]
                     + [round(window_s, 3)])
    try:
        with np.load(cache) as data:
            if str(data["key"]) == key:
                return {str(n): v for n, v in zip(data["names"], data["vectors"])}
    except (OSError, KeyError, ValueError):
        pass
    found = bank_centroids(references, embedder, window_s)
    np.savez(cache, key=np.array(key), names=np.array(list(found)), vectors=np.array(list(found.values())))
    return found


def missing_people(bank: dict[str, np.ndarray], centroids: dict[str, np.ndarray],
                   represented: float) -> dict[str, np.ndarray]:
    """The bank people without a cluster of their own in the episode.

    A cluster represents the bank person it is closest to; a person is represented when they are that
    closest person for at least one cluster and the cosine is at least ``represented``.
    """
    if not centroids:
        return dict(bank)
    if not bank:
        return {}
    names = list(bank)
    matrix = np.stack([bank[n] for n in names])
    have: set[str] = set()
    for vec in centroids.values():
        sims = matrix @ vec
        best = int(np.argmax(sims))
        if sims[best] >= represented:
            have.add(names[best])
    return {n: bank[n] for n in names if n not in have}


def relabel_words(words: list[Word], audio: np.ndarray, embedder: Any, params: RelabelParams,
                  cache: Path | None = None,
                  bank: dict[str, np.ndarray] | None = None) -> tuple[list[Word], dict[str, Any]]:
    """Copies of ``words`` with Viterbi-refined speakers, and statistics.

    ``bank`` (name -> unit vector) adds the people missing from the clusters as ``bank:<name>`` states.
    """
    stats: dict[str, Any] = {"words": len(words), "relabelled": 0, "changes_before": 0, "changes_after": 0,
                             "speakers_with_centroid": 0, "embed_s": 0.0}
    if bank is not None:
        stats.update({"bank_states": [], "bank_words": 0})
    own = [w.speaker for w in words]
    stats["changes_before"] = stats["changes_after"] = _changes(own)
    if not words:
        return [], stats
    t0 = time.perf_counter()
    emb = _load_cache(cache, words, params.window_s) if cache is not None and cache.exists() else None
    if emb is not None:
        log.info("Word embeddings loaded from %s", cache)
    else:
        emb = embed_windows(word_windows(words, audio, params.window_s), embedder)
        stats["embed_s"] = round(time.perf_counter() - t0, 2)
        if cache is not None:
            np.savez(cache, starts=_times(words, "start"), ends=_times(words, "end"),
                     emb=emb.astype(np.float32), window_s=np.float64(params.window_s))
    labels = list(own)
    centroids: dict[str, np.ndarray] = {}
    for _ in range(max(params.passes, 1)):
        centroids = _centroids(words, labels, emb, params)
        stats["speakers_with_centroid"] = len(centroids)
        if not centroids:
            break
        extra = missing_people(bank, centroids, params.bank_represented) if bank else {}
        if bank is not None:
            stats["bank_states"] = sorted(extra)
        new = _viterbi(words, own, emb, centroids, params, extra)
        if extra and params.bank_min_s > 0:
            new = _drop_short_bank_runs(words, new, own, params.bank_min_s)
        if new == labels:
            break
        labels = new
    out = [replace(w, speaker=s) for w, s in zip(words, labels)]
    stats["relabelled"] = sum(a != b for a, b in zip(own, labels))
    stats["changes_after"] = _changes(labels)
    if bank is not None:
        stats["bank_words"] = sum(str(s).startswith(BANK_PREFIX) for s in labels)
    return out, stats
