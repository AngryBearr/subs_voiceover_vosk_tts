"""Word-level speaker relabelling: a Viterbi pass over per-word speaker embeddings.

The diarizer misses speaker changes without a pause (a one-second shout inside a long turn) and puts
boundaries a word or two off. Every word gets an embedding of a fixed window around its middle; each
diarizer speaker gets a centroid from its "clean" words (far from any change of the diarizer label).
A word's score for a speaker is its cosine to that speaker's centroid plus a bonus for the diarizer's
own label; switching speaker between consecutive words costs a penalty that fades with the pause
between them. Viterbi picks the best label sequence.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from stt.refine_speakers import RATE, Embed
from stt.types import Word

log = logging.getLogger("stt.boundaries")

_FORBIDDEN = -1e9


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
    turn_bonus: float = 0.3  # bonus for a speaker change where the subtitles open another voice


@dataclass
class WordHints:
    """What existing subtitles say about the words (``stt.subtitle_hints``). ``persons``: person -> indices of the words
    the subtitles give to that person; ``turns``: indices of words where the subtitles open another voice.
    ``state_of`` and ``fixed`` are filled by ``plan_states``."""

    persons: dict[str, list[int]] = field(default_factory=dict)
    turns: set[int] = field(default_factory=set)
    min_words: int = 10  # hinted words a person needs for a state of their own
    purity: float = 0.8  # share of a cluster's hinted seconds one person needs to keep that cluster
    state_of: dict[str, str] = field(default_factory=dict)  # person -> cluster label, or "P:<NAME>"
    fixed: dict[int, str] = field(default_factory=dict)  # word index -> state its hinted person has


OWN_STATE = "P:"  # label prefix of a person with a state of their own (not a cluster of the diarizer)


def plan_states(words: list[Word], own: list[str | None], hints: WordHints, available: set[str],
                emb: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Decide the state of every hinted person and fill ``hints.state_of`` / ``hints.fixed``. Returns the centroids of the
    new own states (empty without ``emb``).

    A person alone holding at least ``purity`` of a cluster's hinted seconds keeps that cluster. Otherwise, with at least
    ``min_words`` hinted words, the person gets a state "P:<NAME>" (centroid from the hinted words); else the cluster
    where they have most hinted seconds. Only clusters in ``available`` (those with a centroid) can be kept.
    """
    seconds: dict[str, Counter[str]] = defaultdict(Counter)  # cluster -> person -> hinted seconds
    of_person: dict[str, Counter[str]] = defaultdict(Counter)
    usable = {person: [i for i in idx if words[i].source != "ocr"] for person, idx in hints.persons.items()}
    for person, idx in usable.items():
        for i in idx:
            if own[i] is not None:
                seconds[str(own[i])][person] += words[i].end - words[i].start
                of_person[person][str(own[i])] += words[i].end - words[i].start
    owner = {cluster: votes.most_common(1)[0][0] for cluster, votes in seconds.items()}
    extra: dict[str, np.ndarray] = {}
    hints.state_of, hints.fixed = {}, {}
    for person in sorted(usable):
        if person not in of_person:
            continue
        cluster = of_person[person].most_common(1)[0][0]
        clean = owner.get(cluster) == person and seconds[cluster][person] / sum(seconds[cluster].values()) >= hints.purity
        if clean and cluster in available:
            hints.state_of[person] = cluster
        elif len(usable[person]) >= hints.min_words:
            state = OWN_STATE + person
            hints.state_of[person] = state
            if emb is not None:
                mean = emb[usable[person]].sum(axis=0)
                extra[state] = mean / (np.linalg.norm(mean) + 1e-9)
        elif cluster in available:
            hints.state_of[person] = cluster
    hints.fixed = {i: hints.state_of[person] for person, idx in usable.items() if person in hints.state_of for i in idx}
    return extra


def restore_fixed(words: list[Word], fixed: dict[int, str]) -> list[Word]:
    """Copies of ``words`` where the hinted words have their person's state again (smoothing must not undo a named line)."""
    out = list(words)
    for i, state in fixed.items():
        out[i] = replace(out[i], speaker=state)
    return out


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
        idx = [i for i in np.flatnonzero(clean) if labels[i] == label]
        if len(idx) >= params.min_clean_words:
            mean = emb[idx].sum(axis=0)
            out[str(label)] = mean / (np.linalg.norm(mean) + 1e-9)
    return out


def _viterbi(words: list[Word], own: list[str | None], emb: np.ndarray, centroids: dict[str, np.ndarray],
             params: RelabelParams, fixed: dict[int, str] | None = None, turns: set[int] | None = None) -> list[str | None]:
    """The best speaker per word. ``fixed``: word index -> state it must take (a subtitle names the speaker).
    ``turns``: words where the subtitles open another voice; a switch there is rewarded like a diarizer change."""
    # states: labels with a centroid, then the other labels seen (they only keep their own words)
    states: list[str | None] = sorted(centroids)
    states += [s for s in dict.fromkeys(own) if s not in centroids and s is not None]
    n, k = len(words), len(states)
    index = {s: j for j, s in enumerate(states)}
    cent = np.stack([centroids[s] for s in states if s in centroids])
    sims = emb @ cent.T  # (n, centred states)
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
    for i, state in (fixed or {}).items():
        emis[i, :] = _FORBIDDEN
        emis[i, index[state]] = 0.0
    dp = emis[0].copy()
    back = np.zeros((n, k), dtype=np.int64)
    for i in range(1, n):
        gap = max(words[i].start - words[i - 1].end, 0.0)
        if turns is not None and i in turns:
            penalty = -params.turn_bonus
        elif own[i] != own[i - 1] and own[i] is not None and own[i - 1] is not None:
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


def relabel_words(words: list[Word], audio: np.ndarray, embedder: Any, params: RelabelParams,
                  cache: Path | None = None, hints: WordHints | None = None) -> tuple[list[Word], dict[str, Any]]:
    """Copies of ``words`` with Viterbi-refined speakers, and statistics. ``hints`` (subtitles): hinted words take their
    person's state (``plan_states``; ``hints.fixed`` is filled) and a switch at a turn mark is rewarded."""
    stats: dict[str, Any] = {"words": len(words), "relabelled": 0, "changes_before": 0, "changes_after": 0,
                             "speakers_with_centroid": 0, "embed_s": 0.0}
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
    extra: dict[str, np.ndarray] = {}
    for turn in range(max(params.passes, 1)):
        centroids = _centroids(words, labels, emb, params)
        stats["speakers_with_centroid"] = len(centroids)
        if not centroids:
            break
        if hints is not None:
            if turn == 0:
                extra = plan_states(words, own, hints, set(centroids), emb)
                stats["own_states"] = len(extra)
            centroids = {**centroids, **extra}
        new = _viterbi(words, own, emb, centroids, params, fixed=hints.fixed if hints else None,
                       turns=(hints.turns or None) if hints else None)
        if new == labels:
            break
        labels = new
    out = [replace(w, speaker=s) for w, s in zip(words, labels)]
    stats["relabelled"] = sum(a != b for a, b in zip(own, labels))
    stats["changes_after"] = _changes(labels)
    return out, stats
