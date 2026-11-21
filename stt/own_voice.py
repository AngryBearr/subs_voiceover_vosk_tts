"""Own voices for speakers without a voice-bank clip: a clean prompt, then a Russian sample made from it.

A speaker without a voice-bank clip is voiced by the generic voice of the gender. With
``voiceover.engines.f5.own_voice.enabled`` he gets his own voice the way the people with a clip do: a short clean
English prompt is cut from that speaker's speech in the same episode, X-Voice says the Russian text of those
lines in that voice (a "Russian sample", ``stt.ru_samples``), and ESpeech then clones the sample like any
reference clip. This module makes the prompt and the sample; nothing special happens inside ``voiceover``.

The prompt has to be CLEAN: no laughter, music, silence or other people.

ASR word times are the weak point: the last word of a line is often stretched over the laughter or music that
follows it ("happened?" 624.1-626.9 s, "time." 452.2-459.0 s), so a cue cut by its boundaries is a clip of laughter.
Hence the prompt is built from words, not cues: a word longer than its letters allow is cut short
(``capped_end``), laughter tokens are dropped, words close in time are joined into spans, each span is embedded
(WeSpeaker resnet293-ft) and the spans far from the speaker's centroid (another person, a cry) are dropped. The
best spans (nearest to the centroid) make up to ``max_s`` seconds; the vocal separator of ``stt.manual_clip`` takes
the music away. A speaker whose spans do not agree with one another (a mixed cluster) gets no prompt and stays on
the generic voice.

Files in ``<episode>/own_voice/``: ``<sid>.wav`` + ``<sid>.json`` (the prompt and what it was cut from; the
``sample`` block describes the Russian sample), ``<sid>.ru.wav`` + ``<sid>.ru.txt`` (the sample and its text).

    from stt.own_voice import build_voices
    voices = build_voices(episode_dir, ["S09"], {"max_s": 10.0, "min_s": 1.5, "takes": 6})   # {sid: <ep>/own_voice/S09.ru.wav}
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from utils.atomic_io import write_text_atomic

log = logging.getLogger(__name__)

RATE = 16_000  # audio.wav
BASE_S = 0.30  # a word is at least this long (0.15 s + 0.15 s of slack) ...
PER_LETTER_S = 0.11  # ... plus this much per letter: the longest a word may be
MERGE_GAP_S = 0.25  # words closer than this make one span
MIN_SPAN_S = 0.30  # shorter spans are dropped
MIN_COSINE = 0.45  # a span less similar to the speaker's centroid is another voice or a cry
SPLIT_MAX_COSINE = 0.5  # two parts of a speaker's spans less similar than this are two people
SPLIT_MIN_SHARE = 0.25  # ...when the smaller part holds this share of the speech
MAJOR_RATIO = 2.0  # the larger of two voices is taken when it has this many times the speech of the other
MIN_EMBED_S = 1.0  # shorter spans are padded (by repeating them) for the embedder
CUT_PAD_S = 0.05  # context kept around a span when cut from the media
JOIN_GAP_S = 0.5  # chosen spans closer than this are cut as one range
MIN_RANGE_S = 0.5  # the vocal separator skips shorter ranges
PAUSE_S = 0.3  # silence ``make_manual_clip`` puts between pieces
OTHER_OVERLAP_S = 0.1  # a span this long under another speaker's word is not clean
FILE_VERSION = 1  # bump when the prompt's making changes

_WORD = re.compile(r"[^a-z']")
Span = tuple[float, float]


def _token(text: str) -> str:
    return _WORD.sub("", str(text).lower())


def _laughter() -> frozenset[str]:
    from stt.fill import _LAUGHTER

    return _LAUGHTER


def capped_end(word: dict[str, Any]) -> float:
    """End of a word as far as its letters allow: ASR stretches the last word over laughter or music."""
    letters = sum(ch.isalpha() for ch in str(word.get("text", "")))
    start, end = float(word["start"]), float(word["end"])
    return min(end, start + BASE_S + PER_LETTER_S * letters)


def speech_spans(words: list[dict[str, Any]], speaker: str, start: float, end: float, *,
                 laughter: frozenset[str] | None = None) -> list[Span]:
    """Clean speech of ``speaker`` inside the cue ``start``-``end`` (seconds): word spans, nothing else.

    Laughter-only tokens are dropped, each word is cut to what its letters allow (``capped_end``), words closer
    than ``MERGE_GAP_S`` make one span and spans under ``MIN_SPAN_S`` are dropped."""
    stop = laughter if laughter is not None else _laughter()
    spans: list[Span] = []
    for word in sorted((w for w in words if w.get("speaker") == speaker), key=lambda w: float(w["start"])):
        w_start = float(word["start"])
        if not start - 0.05 <= w_start < end or float(word["end"]) <= w_start or _token(word.get("text", "")) in stop:
            continue
        w_end = capped_end(word)
        if spans and w_start - spans[-1][1] < MERGE_GAP_S:
            spans[-1] = (spans[-1][0], max(spans[-1][1], w_end))
        else:
            spans.append((w_start, w_end))
    return [s for s in spans if s[1] - s[0] >= MIN_SPAN_S]


def _under_others(span: Span, others: list[Span]) -> bool:
    return any(min(span[1], b) - max(span[0], a) > OTHER_OVERLAP_S for a, b in others)


def collect_spans(words: list[dict[str, Any]], cues: list[dict[str, Any]], speaker: str) -> list[Span]:
    """The speaker's spans over the episode: plain cues only (not overlapping, filled or on-screen ones), and not
    under another speaker's word. Cue times are in milliseconds, word times in seconds."""
    others = [(float(w["start"]), float(w["end"])) for w in words
              if w.get("speaker") != speaker and float(w["end"]) > float(w["start"])]
    spans: list[Span] = []
    for cue in cues:
        if cue.get("speaker") != speaker or cue.get("overlap") or cue.get("fill") or cue.get("ocr"):
            continue
        spans += [s for s in speech_spans(words, speaker, float(cue["start"]) / 1000, float(cue["end"]) / 1000)
                  if not _under_others(s, others)]
    return sorted(set(spans))


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-9)


def embed_spans(audio: np.ndarray, spans: list[Span], embed: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Unit embeddings (n, d) of the spans of a 16 kHz signal; a span under ``MIN_EMBED_S`` is repeated to fill it."""
    vectors = []
    for a, b in spans:
        piece = np.asarray(audio[int(a * RATE):int(b * RATE)], dtype=np.float32)
        if 0 < len(piece) < MIN_EMBED_S * RATE:
            piece = np.tile(piece, int(np.ceil(MIN_EMBED_S * RATE / len(piece))))[:int(MIN_EMBED_S * RATE)]
        vectors.append(_unit(np.asarray(embed(piece), dtype=np.float64)))
    return np.stack(vectors) if vectors else np.zeros((0, 0))


@dataclass
class Selection:
    kept: list[tuple[Span, float]] = field(default_factory=list)  # (span, cosine), best first
    dropped: list[tuple[Span, float]] = field(default_factory=list)
    reason: str | None = None  # why there is no prompt (None: there is one)

    @property
    def seconds(self) -> float:
        return sum(b - a for (a, b), _ in self.kept)


def _centroid(vectors: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return _unit((vectors[mask] * weights[mask, None]).sum(0))


def split_voices(vectors: np.ndarray, weights: np.ndarray, *, max_between: float = SPLIT_MAX_COSINE,
                 min_share: float = SPLIT_MIN_SHARE) -> tuple[np.ndarray, float] | None:
    """Two voices inside one speaker's spans: a 2-means split (seeded by the least similar pair) whose centroids are
    less similar than ``max_between`` and whose smaller part holds at least ``min_share`` of the speech. Returns
    (mask of the larger part, cosine between the parts), or None for one voice. Two evenly mixed voices both sit
    near the joint centroid, so the centroid test of ``select_spans`` alone does not see them."""
    if len(vectors) < 4:
        return None
    sims = vectors @ vectors.T
    i, j = np.unravel_index(np.argmin(sims), sims.shape)
    centres = np.stack([vectors[i], vectors[j]])
    side = np.zeros(len(vectors), dtype=bool)
    for _ in range(10):
        side = (vectors @ centres[0]) >= (vectors @ centres[1])  # True: nearer the first centre
        if side.all() or not side.any():
            return None
        new = np.stack([_centroid(vectors, weights, side), _centroid(vectors, weights, ~side)])
        if np.allclose(new, centres):
            break
        centres = new
    between = float(centres[0] @ centres[1])
    first_s, second_s = float(weights[side].sum()), float(weights[~side].sum())
    if between >= max_between or min(first_s, second_s) < min_share * (first_s + second_s):
        return None
    return (side if first_s >= second_s else ~side), between


def select_spans(spans: list[Span], vectors: np.ndarray, *, min_s: float = 1.5, max_s: float = 10.0,
                 min_cosine: float = MIN_COSINE) -> Selection:
    """The spans of one voice: the centroid is the duration-weighted mean of the unit embeddings; spans under
    ``min_cosine`` to it are dropped and the centroid is made again from the rest (once). No selection (``reason``)
    when less than ``min_s`` of speech is left or more than half of the speech was dropped (a mixed cluster).
    The nearest spans are taken until ``max_s`` (pauses between pieces counted)."""
    if not spans:
        return Selection(reason="no clean speech in the plain cues")
    weights = np.array([b - a for a, b in spans])
    everything = np.ones(len(spans), dtype=bool)
    first = vectors @ _centroid(vectors, weights, everything)
    keep = first >= min_cosine
    if not keep.any():
        return Selection(dropped=[(s, float(c)) for s, c in zip(spans, first)], reason="spans do not agree with each other")
    cosines = vectors @ _centroid(vectors, weights, keep)
    keep = cosines >= min_cosine
    result = Selection(dropped=[(s, float(c)) for s, c, k in zip(spans, cosines, keep) if not k])
    dropped_s = float(weights[~keep].sum())
    kept_s = float(weights[keep].sum())
    if kept_s < min_s:
        result.reason = f"only {kept_s:.1f} s of clean speech (need {min_s:.1f} s)"
        return result
    if dropped_s > 0.5 * float(weights.sum()):
        result.reason = f"mixed voices: {dropped_s:.1f} of {weights.sum():.1f} s are far from the centroid"
        return result
    two = split_voices(vectors[keep], weights[keep])
    if two is not None:  # two voices that agree with the centroid alike: keep the clearly larger one
        major, between = two
        idx = np.flatnonzero(keep)
        minor_s = float(weights[idx[~major]].sum())
        if float(weights[idx[major]].sum()) < MAJOR_RATIO * minor_s:
            result.reason = (f"mixed voices: two voices of {kept_s - minor_s:.1f} and {minor_s:.1f} s "
                             f"(cosine {between:.2f} between them)")
            return result
        result.dropped += [(spans[i], float(cosines[i])) for i in idx[~major]]
        keep[idx[~major]] = False
        if float(weights[keep].sum()) < min_s:
            result.reason = f"only {float(weights[keep].sum()):.1f} s of the main voice (need {min_s:.1f} s)"
            return result
        cosines = vectors @ _centroid(vectors, weights, keep)
    ranked = sorted(((s, float(c)) for s, c, k in zip(spans, cosines, keep) if k), key=lambda item: -item[1])
    total = 0.0
    for span, cosine in ranked:
        gap = PAUSE_S if result.kept else 0.0
        if total + gap + (span[1] - span[0]) <= max_s:
            result.kept.append((span, cosine))
            total += gap + span[1] - span[0]
    if not result.kept:  # every span is longer than max_s: the best one, cut to the limit
        (a, _), cosine = ranked[0]
        result.kept.append(((a, a + max_s), cosine))
    return result


def cut_ranges(spans: list[Span]) -> list[Span]:
    """Chosen spans in time order, padded by ``CUT_PAD_S``, those closer than ``JOIN_GAP_S`` joined, ranges
    under ``MIN_RANGE_S`` dropped (the separator skips them)."""
    ranges: list[Span] = []
    for a, b in sorted(spans):
        a, b = max(0.0, a - CUT_PAD_S), b + CUT_PAD_S
        if ranges and a - ranges[-1][1] <= JOIN_GAP_S:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], b))
        else:
            ranges.append((a, b))
    return [(round(a, 3), round(b, 3)) for a, b in ranges if b - a >= MIN_RANGE_S]


def _media_offset(folder: Path) -> tuple[Path, float]:
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    return Path(source["media"]), float(source.get("start") or 0.0)


def _strip_tail(wav: Path) -> float:
    """Take the 1 s of silence ``make_manual_clip`` leaves at the end off a prompt; the seconds of speech left."""
    import soundfile as sf

    data, rate = sf.read(str(wav), dtype="float32")
    loud = np.flatnonzero(np.abs(data) > 1e-3)
    end = int(loud[-1]) + 1 + int(0.05 * rate) if loud.size else len(data)
    sf.write(str(wav), data[:min(end, len(data))], rate, subtype="PCM_16")
    return min(end, len(data)) / rate


def _reusable(meta: Path, wav: Path, ranges: list[Span]) -> bool:
    try:
        old = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (wav.is_file() and wav.stat().st_size > 44 and old.get("version") == FILE_VERSION
            and [tuple(r) for r in old.get("ranges", [])] == [tuple(r) for r in ranges])


def build_prompts(episode_dir: Path, speakers: list[str], config: dict[str, Any], device: str = "cuda",
                  reasons: dict[str, str] | None = None, embedder: Any = None) -> dict[str, Path]:
    """``{speaker id: prompt WAV}`` for the speakers that have a clean prompt, in ``<episode>/own_voice/``.

    ``config`` is ``voiceover.engines.f5.own_voice`` (``min_s``, ``max_s``). A speaker without a prompt gets its reason
    in ``reasons`` (when given) and is left out. ``<speaker>.json`` beside a prompt keeps what it was cut from; the
    same ranges later reuse the WAV. Models are loaded once, only when something has to be embedded or cut (an
    ``embedder`` given by the caller is used instead of loading one)."""
    import soundfile as sf

    from stt.manual_clip import ClipModels, make_manual_clip
    from stt.refine_speakers import make_embedder

    min_s, max_s = float(config.get("min_s", 1.5)), float(config.get("max_s", 10.0))
    words = json.loads((episode_dir / "words.json").read_text(encoding="utf-8"))
    cues = json.loads((episode_dir / "subs.en.json").read_text(encoding="utf-8"))
    media, offset = _media_offset(episode_dir)
    out_dir = episode_dir / "own_voice"
    audio: np.ndarray | None = None
    models: Any = None
    prompts: dict[str, Path] = {}
    why = reasons if reasons is not None else {}
    for speaker in speakers:
        spans = collect_spans(words, cues, speaker)
        if audio is None and spans:
            audio, rate = sf.read(str(episode_dir / "audio.wav"), dtype="float32")
            audio = audio.mean(axis=1) if audio.ndim > 1 else audio
            if rate != RATE:
                raise ValueError(f"{episode_dir / 'audio.wav'}: expected {RATE} Hz, got {rate}")
            embedder = embedder or make_embedder("resnet293-ft", device)
        selection = select_spans(spans, embed_spans(audio, spans, embedder) if spans else np.zeros((0, 0)),
                                 min_s=min_s, max_s=max_s)
        wav, meta = out_dir / f"{speaker}.wav", out_dir / f"{speaker}.json"
        if selection.reason:
            why[speaker] = selection.reason
            continue
        ranges = cut_ranges([s for s, _ in selection.kept])
        if not ranges:
            why[speaker] = "no range long enough to cut"
            continue
        media_ranges = [(a + offset, b + offset) for a, b in ranges]
        info: dict[str, Any] = {"version": FILE_VERSION, "ranges": media_ranges, "speaker": speaker,
                                "seconds": round(selection.seconds, 2),
                                "cosines": [round(c, 3) for _, c in selection.kept],
                                "spans_total": len(spans), "spans_chosen": len(selection.kept),
                                "spans_dropped": len(selection.dropped),
                                "dropped_cosines": [round(c, 3) for _, c in selection.dropped], "reason": None}
        if not _reusable(meta, wav, media_ranges):
            models = models or ClipModels(device)
            try:
                clip = make_manual_clip(media, media_ranges, wav, device=device, text="", models=models)
            except (ValueError, FileNotFoundError) as error:
                why[speaker] = f"the cut failed: {error}"
                continue
            info["seconds"] = round(_strip_tail(wav), 2)
            info["warnings"] = clip.warnings
            if info["seconds"] < min_s:
                why[speaker] = f"only {info['seconds']:.1f} s of speech after the vocal separation (need {min_s:.1f} s)"
                wav.unlink(missing_ok=True)
                continue
            write_text_atomic(meta, json.dumps(info, ensure_ascii=False, indent=1) + "\n")
        prompts[speaker] = wav
        log.info("own-voice prompt %s: %.1f s from %d of %d spans (%d dropped as another voice), cosines %.2f-%.2f", speaker,
                 info["seconds"], len(selection.kept), len(spans), len(selection.dropped),
                 min(c for _, c in selection.kept), max(c for _, c in selection.kept))
    return prompts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def russian_sample(episode_dir: Path, speaker: str, prompt_wav: Path, *, takes: int, recognizer: Any, embedder: Any,
                   run_worker: Callable[[list[dict[str, Any]], Path], list[dict[str, Any]]] | None = None,
                   reasons: dict[str, str] | None = None) -> Path | None:
    """The Russian sample of a speaker's prompt: ``<prompt>.ru.wav`` (+ ``.ru.txt``), None when it cannot be made.

    The text is the Russian translation (``subs.ru.json``) of the lines the prompt was cut from. X-Voice makes
    ``takes`` takes of it in the prompt's voice (``run_worker``, default ``stt.ru_samples.run_worker``); the best one
    (``select_take``: few recognition errors, then the most similar voice) is kept. The ``sample`` block of
    ``<prompt>.json`` holds the text, the take's scores and the prompt's hash: the same prompt and text later reuse the
    sample without running X-Voice. The reason of a failure goes to ``reasons[speaker]`` (when given)."""
    import librosa
    import soundfile as sf

    from stt import ru_samples

    run_worker = run_worker or ru_samples.run_worker

    def fail(reason: str) -> None:
        if reasons is not None:
            reasons[speaker] = reason

    meta_path = prompt_wav.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    offset = float(json.loads((episode_dir / "source.json").read_text(encoding="utf-8")).get("start") or 0.0)
    rows = json.loads((episode_dir / "subs.ru.json").read_text(encoding="utf-8"))  # on the decoded-audio timeline
    text = ru_samples.russian_text(rows, {"pieces": [(a - offset, b - offset) for a, b in meta["ranges"]],
                                          "speaker": speaker})
    if not text:
        fail("no Russian text")
        return None
    sample, text_path = prompt_wav.with_suffix(".ru.wav"), prompt_wav.with_suffix(".ru.txt")
    prompt_hash = _sha256(prompt_wav)
    old = meta.get("sample") or {}
    if (sample.is_file() and text_path.is_file() and old.get("prompt_sha256") == prompt_hash
            and old.get("text") == text):
        log.info("own-voice sample %s: reused", speaker)
        return sample
    with tempfile.TemporaryDirectory(dir=prompt_wav.parent) as tmp_name:
        tmp = Path(tmp_name)
        items = [{"ref_wav": str(prompt_wav.resolve()), "text": text, "out_wav": str(tmp / f"{seed}.wav"), "seed": seed}
                 for seed in range(takes)]
        try:
            for result in run_worker(items, tmp):
                if result.get("error"):
                    log.warning("own-voice sample %s: %s: %s", speaker, Path(result["out_wav"]).name, result["error"])
        except (SystemExit, RuntimeError) as error:
            fail(f"X-Voice failed: {error}")
            return None
        english, _ = librosa.effects.trim(ru_samples._read_mono(prompt_wav, 16_000), top_db=40)
        reference_print = embedder.batch(ru_samples._windows(english)).mean(axis=0)
        reference_print = reference_print / np.linalg.norm(reference_print)
        scored: list[dict[str, Any]] = []
        for seed in range(takes):
            path = tmp / f"{seed}.wav"
            if path.exists() and (one := ru_samples.score_take(path, text, recognizer, embedder, reference_print)):
                scored.append({"take": seed, **one})
    log.debug("own-voice sample %s takes: %s", speaker,
              [(t["take"], t["cer"], t["similarity"], t["speech_s"]) for t in scored])
    best = ru_samples.select_take(scored)
    if best is None:
        sample.unlink(missing_ok=True)  # a sample of another prompt would clone the wrong voice
        fail("no usable X-Voice take")
        return None
    sf.write(str(sample), ru_samples.finish(best.pop("audio")), ru_samples.RATE, subtype="PCM_16")
    write_text_atomic(text_path, text + "\n")
    meta["sample"] = {"text": text, "take": best["take"], "cer": best["cer"], "similarity": best["similarity"],
                      "speech_s": best["speech_s"], "takes": len(scored), "prompt_sha256": prompt_hash}
    write_text_atomic(meta_path, json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    log.info("own-voice sample %s: take %d of %d, CER %.1f%%, similarity %.3f, %.1f s: %s", speaker, best["take"],
             len(scored), 100 * best["cer"], best["similarity"], best["speech_s"], text)
    return sample


def build_voices(episode_dir: Path, speakers: list[str], config: dict[str, Any], device: str = "cuda",
                 reasons: dict[str, str] | None = None) -> dict[str, Path]:
    """``{speaker id: Russian sample WAV}`` for the speakers that got a clean prompt and a usable sample.

    ``config`` is ``voiceover.engines.f5.own_voice`` (``min_s``, ``max_s``, ``takes``). The speech recognizer and the
    voice embedder are loaded once, and only when a prompt is there. Speakers left out get their reason in ``reasons``."""
    from stt.refine_speakers import make_embedder
    from voiceover.speech_check import Recognizer

    why = reasons if reasons is not None else {}
    embedder = make_embedder("resnet293-ft", device)
    prompts = build_prompts(episode_dir, speakers, config, device, why, embedder)
    recognizer: Any = None
    voices: dict[str, Path] = {}
    for speaker, prompt in prompts.items():
        recognizer = recognizer or Recognizer(device)
        sample = russian_sample(episode_dir, speaker, prompt, takes=int(config.get("takes", 6)),
                                recognizer=recognizer, embedder=embedder, reasons=why)
        if sample is not None:
            voices[speaker] = sample
    return voices
