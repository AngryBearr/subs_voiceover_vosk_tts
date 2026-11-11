"""Second recognition pass ("fill"): words for stretches where somebody speaks but the transcript has none.

At challenges the main recognition often misses shouts under music and crowd noise: the voice activity
detector (VAD) hears speech there, yet the final word stream has no words. This module finds such stretches,
lets Whisper listen to each stretch alone (that recovers about half of such lines) and keeps the result only
if it passes guards against text the model invents on noise ("Thank you.", a repeat of the neighbouring
speech, very unsure words). The accepted words carry ``source="fill"``; they join the word stream before the
speakers are assigned, and the dub voices them with the neutral voice.

When the episode has subtitles in sync (``stt.transcribe --subtitles``), their lines come first (``from_subtitles``):
a line the audio did not confirm whose time holds no recognised word is inserted as it is written, its words
spread over the line's time (the captioner heard it; S01E01 2:37-2:41: "The rod, Sean!", "Listen up!", "Grab your
backpacks!" where both models hear nothing). A line with a speaker label keeps that person (``source="subtitle"``;
the hints pin it to the person like any named line), an unlabelled one is a ``fill`` line like Whisper's. Whisper
then listens only to what is still free.

Experiments on an existing episode folder (does not touch ``words.json``):
    subs_env/bin/python -m stt.fill output/stt/ep01 --model whisper-large-v3 --device cuda

Files in the folder: ``fill.chunks.json`` (stretches of the last run), ``asr.fill.json`` (the worker output),
``fill.json`` (stats, accepted lines and rejected ones with the reason, for listening and evaluation).
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import itertools
import logging
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from stt.audio import SAMPLE_RATE
from stt.runner import load_json, save_json
from stt.textfix import collapse_repeats
from stt.types import Segment, Word
from stt.vad import speech_regions

log = logging.getLogger(__name__)

# Texts Whisper invents on noise and silence (normalised: lowercase words separated by one space).
STOCK_PHRASES: frozenset[str] = frozenset({
    "thank you", "thanks for watching", "thank you for watching", "thank you very much", "thank you so much",
    "thanks", "bye", "bye bye", "goodbye", "you", "the end", "we'll be right back", "see you", "see you next time",
    "see you later", "subtitles by the amara org community", "subscribe", "so", "oh", "uh", "um", "hmm", "mm", "okay",
    "music", "applause", "laughter",
})
_NOISE_WORD = re.compile(r"(.)\1{3,}")  # "AAAAAAAA", "hahahahaha": a letter repeated four or more times in a row
_LAUGHTER = frozenset({"ha", "haha", "hah", "heh", "hehe", "hee", "ho", "hoho", "ah", "aah", "ahh", "oh", "ooh", "ow",
                       "woo", "whoo", "wow", "yay", "aw", "aww", "mm", "hmm", "uh", "um"})  # laughter and cries alone
_STOCK_PREFIXES = ("subtitles by", "subtitled by", "transcribed by", "captions by", "copyright")  # credits
_CREDITS = re.compile(r"captioning (sponsored|provided|made possible)|captioned by|closed captions?|subtitles? by|"
                      r"transcribed by|www\.|\.com\b|\.org\b|\bcbs\b.*\btoyota\b")  # a captioner's credits line


def is_credits(text: str) -> bool:
    return bool(_CREDITS.search(text.lower()))

ZERO_WORD_S = 0.05  # length given to a word the model timed with no duration
OUTSIDE_TOLERANCE_S = 0.5  # a word may start this far outside its stretch
ECHO_WINDOW_S = 3.0  # final words this close to a stretch edge are the "neighbours"
EDGE_TOL_S = 1.0  # missing words of a partly recognised line may sit this far outside its (shifted) cue time
EDGE_MIN_WORD_S = 0.15  # room needed per missing word
EDGE_MAX_WORD_S = 0.5  # the longest a placed word may be
EDGE_GAP_S = 0.02  # clearance from the neighbouring words
EDGE_MIN_SHARE = 0.34  # share of a line's tokens the words must match for the edge pass (pinning needs 0.6)
EDGE_MIN_RUN = 3  # the longest matched block of such a line must have at least this many tokens
EDGE_MAX_WORDS = 8  # the longest run of missing words taken

Stretch = tuple[float, float]
Runner = Callable[..., dict[str, Any]]


def normalize(text: str) -> list[str]:
    """Lowercase word tokens of ``text``: letters, digits and apostrophes."""
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text.lower().replace("’", "'"))


def free_stretches(audio: np.ndarray, words: list[Word], *, sample_rate: int = SAMPLE_RATE, min_s: float = 1.0,
                   pad_s: float = 0.3, context_s: float = 0.25, max_s: float = 28.0) -> list[Stretch]:
    """Speech regions of the VAD that no word covers (words widened by ``pad_s``).

    Pieces shorter than ``min_s`` are dropped, the rest is widened by ``context_s`` on both sides (inside the
    audio), pieces that then touch are joined and pieces longer than ``max_s`` are split in equal parts.
    """
    total = len(audio) / sample_rate
    spans = sorted((w.start - pad_s, w.end + pad_s) for w in words)
    known: list[list[float]] = []
    for start, end in spans:
        if known and start <= known[-1][1]:
            known[-1][1] = max(known[-1][1], end)
        else:
            known.append([start, end])

    pieces: list[Stretch] = []
    for r_start, r_end in sorted(speech_regions(audio, sample_rate=sample_rate)):
        cursor = r_start
        for k_start, k_end in known:
            if k_end <= cursor:
                continue
            if k_start >= r_end:
                break
            if k_start > cursor:
                pieces.append((cursor, k_start))
            cursor = max(cursor, k_end)
        if cursor < r_end:
            pieces.append((cursor, r_end))

    widened: list[list[float]] = []
    for start, end in pieces:
        if end - start < min_s:
            continue
        start, end = max(0.0, start - context_s), min(total, end + context_s)
        if widened and start <= widened[-1][1]:
            widened[-1][1] = max(widened[-1][1], end)
        else:
            widened.append([start, end])

    result: list[Stretch] = []
    for start, end in widened:
        parts = max(1, math.ceil((end - start) / max_s))
        step = (end - start) / parts
        result += [(round(start + i * step, 3), round(start + (i + 1) * step, 3)) for i in range(parts)]
    return result


def _longest_run(a: list[str], b: list[str]) -> int:
    """Length of the longest contiguous run of tokens that ``a`` and ``b`` share."""
    best, prev = 0, [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def accept(segment: Segment, stretch: Stretch, neighbours: list[Word], *, min_score: float = 0.7,
           stock: frozenset[str] = STOCK_PHRASES) -> tuple[list[Word], str | None]:
    """Guards for one recognized segment of a stretch: ``(words, None)`` if it is kept, else ``([], reason)``.

    Reasons: "outside" (the words lie outside the stretch), "empty" (no letters, or a letter repeated four times in a
    row "AAAAAA", or laughter and cries alone "Ha ha ha"), "score" (mean word
    probability too low), "stock" (a phrase the model invents on noise), "echo" (at least half of the words
    repeat, in order, the known speech next to the stretch).
    """
    start, end = stretch
    kept = [w for w in segment.words if start - OUTSIDE_TOLERANCE_S <= (w.start + w.end) / 2 <= end + OUTSIDE_TOLERANCE_S]
    if not kept:
        return [], "outside"
    text = " ".join(w.text for w in kept)
    if not any(ch.isalpha() for ch in text) or any(_NOISE_WORD.search(w.text.lower()) for w in kept):
        return [], "empty"
    tokens = normalize(text)
    if tokens and all(t in _LAUGHTER for t in tokens):
        return [], "empty"  # "Ha ha ha", "Oh! Oh!": laughter or cries, what the captioner writes as [laughing]
    if sum(1.0 if w.score is None else w.score for w in kept) / len(kept) < min_score:
        return [], "score"
    joined = " ".join(tokens)
    if joined in stock or joined.startswith(_STOCK_PREFIXES) or "amara" in joined or "subtitle workshop" in joined:
        return [], "stock"
    near: list[str] = []
    for w in neighbours:
        if w.end >= start - ECHO_WINDOW_S and w.start <= end + ECHO_WINDOW_S:
            near += normalize(w.text)
    if tokens and _longest_run(tokens, near) * 2 >= len(tokens):
        return [], "echo"
    words = []
    for w in kept:
        stop = w.end if w.end > w.start else w.start + ZERO_WORD_S
        words.append(dataclasses.replace(w, end=stop, source="fill"))
    return words, None

class _Blockers:
    """Words in the way, sorted by start, with a running maximum of the ends (``add`` rebuilds lazily)."""

    def __init__(self, words: list[Word]) -> None:
        self._pairs = [(w.start, w.end) for w in words]
        self._dirty = True
        self.starts: list[float] = []
        self.ends: list[float] = []
        self.reach: list[float] = []  # reach[i]: the largest end among the first i + 1 words

    def add(self, words: list[Word]) -> None:
        self._pairs += [(w.start, w.end) for w in words]
        self._dirty = True

    def _build(self) -> None:
        if self._dirty:
            self._pairs.sort()
            self.starts = [a for a, _ in self._pairs]
            self.ends = [b for _, b in self._pairs]
            self.reach = list(itertools.accumulate(self.ends, max))
            self._dirty = False

    def last_end_before(self, t: float) -> float:
        """The largest end among the words that start before ``t`` (-inf if none)."""
        self._build()
        i = bisect.bisect_left(self.starts, t)
        return self.reach[i - 1] if i else -math.inf

    def first_start_after(self, t: float) -> float:
        """The smallest start among the words that start after ``t`` (inf if none)."""
        self._build()
        i = bisect.bisect_right(self.starts, t)
        return self.starts[i] if i < len(self.starts) else math.inf


def _edge_words(piece: Any, pin: Any, words: list[Word], index: Any, blockers: _Blockers, *,
                total: float) -> tuple[list[list[Word]], list[str]]:
    """Words for the parts of a pinned subtitle line that the recognition dropped: ``(runs of words, skip reasons)``.

    The matched parts of the line stay as they are. Each run of missing parts at the start, at the end or between two
    matched parts is placed in the free time where it belongs (next to the matched word, at most ``EDGE_TOL_S``
    outside the shifted cue), evenly spread, with ``source="edge"``. One reason is returned per run left out.
    """
    from stt.subtitles import tokens as subtitle_tokens

    if pin.match >= 1.0 or not pin.blocks or (len(pin.blocks) == 1 and pin.blocks[0][2] >= len(piece.tokens)):
        return [], []
    parts = [part.strip("-:>") or part for part in piece.text.split()]
    sized = [(part, len(subtitle_tokens(part))) for part in parts if part]
    sized = [(part, n) for part, n in sized if n]
    if sum(n for _, n in sized) != len(piece.tokens):
        return [], ["edge tokens"]
    slot = {a + k: wi_pos + k for a, wi_pos, size in
            ((a, bisect.bisect_left(index.index, wi), size) for a, wi, size in pin.blocks) for k in range(size)}

    def word_at(token_pos: int) -> Word:
        return words[index.index[slot[token_pos]]]

    ranges: list[tuple[str, int, int]] = []  # (part, first token, end token)
    cursor = 0
    for part, n in sized:
        ranges.append((part, cursor, cursor + n))
        cursor += n
    matched = [any(k in slot for k in range(a, b)) for _, a, b in ranges]
    if not any(matched):
        return [], []
    runs: list[list[int]] = []
    for i, ok in enumerate(matched):
        if ok:
            continue
        if runs and runs[-1][-1] == i - 1:
            runs[-1].append(i)
        else:
            runs.append([i])

    lo_cue = max(piece.cue_start + pin.offset - EDGE_TOL_S, 0.0)
    hi_cue = min(piece.cue_end + pin.offset + EDGE_TOL_S, total)
    out: list[list[Word]] = []
    reasons: list[str] = []
    for run in runs:
        first, end = ranges[run[0]][1], ranges[run[-1]][2]
        before = [k for k in slot if k < first]
        after = [k for k in slot if k >= end]
        n = len(run)
        if n > EDGE_MAX_WORDS:
            reasons.append("edge too long")
            continue
        run_tokens = [t for i in run for t in subtitle_tokens(ranges[i][0])]
        if all(t in _LAUGHTER for t in run_tokens):
            reasons.append("edge cries")
            continue
        if all(len(t) == 1 and t not in ("a", "i") for t in run_tokens):
            reasons.append("edge noise")  # a cut word ("three s.") or a letter: nothing to voice
            continue
        if not before:  # prefix: ends where the first matched word starts
            hi = word_at(min(after)).start - EDGE_GAP_S
            lo = max(lo_cue, blockers.last_end_before(hi) + EDGE_GAP_S)
            where = "prefix"
        elif not after:  # suffix: starts where the last matched word ends
            last = word_at(max(before))
            lo = last.end + EDGE_GAP_S
            hi = min(hi_cue, blockers.first_start_after(last.start) - EDGE_GAP_S)
            where = "suffix"
        else:  # between two matched parts: nothing recognised may sit there
            w1, w2 = word_at(max(before)), word_at(min(after))
            lo, hi = w1.end + EDGE_GAP_S, w2.start - EDGE_GAP_S
            if blockers.last_end_before(w2.start) > lo:
                reasons.append("edge substituted")
                continue
            where = "middle"
        room = hi - lo
        if room < n * EDGE_MIN_WORD_S:
            reasons.append("edge no room")
            continue
        span = min(room, n * EDGE_MAX_WORD_S)
        start = hi - span if where == "prefix" else lo if where == "suffix" else lo + (room - span) / 2
        step = span / n
        out.append([Word(start=round(start + k * step, 3), end=round(start + (k + 1) * step, 3), text=ranges[i][0],
                         source="edge") for k, i in enumerate(run)])
    return out, reasons


def from_subtitles(path: Path, words: list[Word], *, start_s: float = 0.0, end_s: float | None = None,
                   pad_s: float = 0.3, min_pinned_share: float = 0.5, max_line_s: float = 12.0,
                   edges: bool = True) -> tuple[list[Word], dict[str, Any]]:
    """Words for the subtitle lines the recognised ``words`` do not confirm and whose time is free of words.

    The file must be in sync: at least ``min_pinned_share`` of its lines with text must pin to the words
    (``stt.subtitles.pin_pieces``), else nothing is taken. Every line is shifted by the local offset the pinning found
    around it. ``end_s``: the audio's length (lines after it are not taken; default: 30 s past the last word).
    Returns the words (``source`` "subtitle" for a labelled line, "fill" for an unlabelled one) and stats
    ``{"lines", "pinned", "share", "inserted", "words", "edges", "edge_words", "edge_candidates", "skipped": {reason: n}}``.
    With ``edges``, a line the words confirm only in part gets its missing first/last/middle words too (``source``
    "edge": they join the recognised phrase, they are not a ``fill`` line), where the time for them is free.
    """
    from stt.subtitles import WordIndex, local_offsets, pin_piece, pin_pieces, read_pieces

    stats: dict[str, Any] = {"lines": 0, "pinned": 0, "share": 0.0, "inserted": 0, "words": 0, "edges": 0,
                             "edge_words": 0, "edge_candidates": 0, "skipped": {}}
    if not words:
        return [], stats
    total = end_s if end_s is not None else max(w.end for w in words) + 30.0
    pieces = []
    for piece in read_pieces(path):  # only the lines of the audio at hand (a test piece: --start/--duration)
        piece.cue_start -= start_s
        piece.cue_end -= start_s
        if piece.tokens and piece.cue_end > 0.0 and piece.cue_start < total:
            pieces.append(piece)
    stats["lines"] = len(pieces)
    if not pieces:
        return [], stats
    index = WordIndex(words)
    pinned, _offset = pin_pieces(pieces, words)
    confirmed = {id(p.piece) for p in pinned}
    stats["pinned"], stats["share"] = len(pinned), round(len(pinned) / len(pieces), 3)
    if stats["share"] < min_pinned_share:
        stats["skipped"]["not in sync"] = len(pieces) - len(pinned)
        return [], stats
    mids = sorted((w.start + w.end) / 2 for w in words)
    def skip(reason: str) -> None:
        stats["skipped"][reason] = stats["skipped"].get(reason, 0) + 1

    out: list[Word] = []
    inserted: set[int] = set()
    taken: dict[tuple[float, float], float] = {}  # a cue shared by several voices: each piece takes its part in turn
    all_shifts = local_offsets(pieces, index)
    for piece, shifts in zip(pieces, all_shifts):
        if id(piece) in confirmed:
            continue
        shift = shifts[0] if shifts else 0.0
        start, end = piece.cue_start + shift, piece.cue_end + shift
        if end - start > max_line_s or end <= start or end < 0 or end > total:
            skip("odd time")
            continue
        parts = [part.strip("-:>") or part for part in piece.text.split()]
        parts = [part for part in parts if part]
        tokens = normalize(piece.text)
        if not parts or (tokens and all(t in _LAUGHTER for t in tokens)):
            skip("cries")  # "Aah!", "Whoo!", "Mm.": not worth a voice
            continue
        if is_credits(piece.text):
            skip("credits")
            continue
        if bisect.bisect_right(mids, end + pad_s) - bisect.bisect_left(mids, start - pad_s):
            skip("words there")  # the recognition heard something else here: not ours to replace
            continue
        key = (piece.cue_start, piece.cue_end)
        siblings = [q for q in pieces if (q.cue_start, q.cue_end) == key and id(q) not in confirmed]
        if len(siblings) > 1:  # the cue's time shared in proportion to the words of each voice
            total_words = sum(len(q.text.split()) for q in siblings)
            length = (end - start) * len(parts) / max(total_words, 1)
            start = taken.get(key, start)
            end = start + length
            taken[key] = end
        step = (end - start) / len(parts)
        source = "subtitle" if piece.person else "fill"
        out += [Word(start=round(start + i * step, 3), end=round(start + (i + 1) * step, 3), text=part, source=source)
                for i, part in enumerate(parts)]
        stats["inserted"] += 1
        stats["words"] += len(parts)
        inserted.add(id(piece))
    if edges:
        candidates = list(pinned)  # plus the lines that match only in part, but with a solid run of words
        for piece, shifts in zip(pieces, all_shifts):
            if id(piece) in confirmed or id(piece) in inserted:
                continue
            hit = next((p for p in (pin_piece(piece, index, s, min_share=EDGE_MIN_SHARE) for s in shifts)
                        if p is not None), None)
            if hit is not None and max(size for _, _, size in hit.blocks) >= EDGE_MIN_RUN:
                candidates.append(hit)
                stats["edge_candidates"] += 1
        blockers = _Blockers(words + out)
        for pin in candidates:
            got, reasons = _edge_words(pin.piece, pin, words, index, blockers, total=total)
            for reason in reasons:
                skip(reason)
            for run in got:
                blockers.add(run)
                out += run
                stats["edges"] += 1
                stats["edge_words"] += len(run)
    return out, stats


def _stretch_of(segment: Segment, stretches: list[Stretch]) -> Stretch:
    """The stretch that holds the middle of ``segment``, else the nearest one."""
    mid = (segment.start + segment.end) / 2
    return min(stretches, key=lambda s: 0.0 if s[0] <= mid <= s[1] else min(abs(mid - s[0]), abs(mid - s[1])))


def run(folder: Path, audio: np.ndarray, words: list[Word], *, model: str, device: str, logs: Path, force: bool,
        min_s: float, min_score: float, runner: Runner | None = None) -> tuple[list[Word], dict[str, Any]]:
    """Recognize the free stretches of an episode folder; returns (accepted words sorted by time, stats).

    ``runner`` runs a worker stage like ``stt.transcribe.run_stage`` (the default); the worker output is reused
    when the stretches are the same as in the last run.
    """
    from stt.asr import ASR_MODELS

    if runner is None:
        from stt.transcribe import run_stage as runner

    chunks_path, out, report = folder / "fill.chunks.json", folder / "asr.fill.json", folder / "fill.json"
    stretches = free_stretches(audio, words, min_s=min_s)
    previous = load_json(chunks_path) if chunks_path.exists() else None
    stats: dict[str, Any] = {"stretches": len(stretches), "stretch_s": round(sum(e - s for s, e in stretches), 1),
                             "segments": 0, "accepted_segments": 0, "words": 0, "rejected": {}, "fill_s": None}
    save_json(chunks_path, [[s, e] for s, e in stretches])
    if not stretches:
        out.unlink(missing_ok=True)
        save_json(report, {"stats": stats, "lines": [], "rejected": []})
        return [], stats

    cached = out.exists() and not force and previous == [[s, e] for s, e in stretches]
    args = ["asr", "--audio", str(folder / "audio.wav"), "--chunks", str(chunks_path), "--model", model,
            "--out", str(out), "--device", device]
    data = load_json(out) if cached else runner("asr.fill", ASR_MODELS[model].env, args, out, logs, True)
    stats["fill_s"] = data.get("transcribe_s")

    segments, _ = collapse_repeats([Segment.from_dict(s) for s in data["segments"]])
    final = sorted(words, key=lambda w: w.start)
    accepted: list[Word] = []
    lines: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for seg in segments:
        stats["segments"] += 1
        got, reason = accept(seg, _stretch_of(seg, stretches), final, min_score=min_score)
        if reason is None:
            scores = [1.0 if w.score is None else w.score for w in got]
            lines.append({"start": round(got[0].start, 3), "end": round(got[-1].end, 3),
                          "text": " ".join(w.text for w in got), "score": round(sum(scores) / len(scores), 3)})
            accepted += got
        else:
            stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
            rejected.append({"start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text,
                             "reason": reason})
    accepted.sort(key=lambda w: w.start)
    stats["accepted_segments"], stats["words"] = len(lines), len(accepted)
    save_json(report, {"stats": stats, "lines": lines, "rejected": rejected})
    return accepted, stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fill pass on an episode folder (audio.wav + words.json)")
    parser.add_argument("folder", help="Episode folder written by stt.transcribe")
    parser.add_argument("--model", default="whisper-large-v3")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--min-s", type=float, default=1.0, help="Shortest stretch to listen to, seconds")
    parser.add_argument("--min-score", type=float, default=0.7, help="Lowest mean word probability to keep")
    parser.add_argument("--force", action="store_true", help="Recognize again even if the stretches are unchanged")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from stt.audio import read_wav

    folder = Path(args.folder)
    fields = {f.name for f in dataclasses.fields(Word)}
    offset = float((load_json(folder / "source.json").get("start") or 0.0) if (folder / "source.json").exists() else 0.0)
    words = [Word(**{k: v for k, v in w.items() if k in fields}) for w in load_json(folder / "words.json")
             if w.get("source") != "fill"]  # the words of an earlier fill pass are not known speech
    for w in words:  # words.json is on the episode timeline, audio.wav starts at the cut (--start)
        w.start, w.end = w.start - offset, w.end - offset
    found, stats = run(folder, read_wav(folder / "audio.wav"), words, model=args.model, device=args.device,
                       logs=folder / "logs", force=args.force, min_s=args.min_s, min_score=args.min_score)
    for line in load_json(folder / "fill.json")["lines"]:
        print(f"{line['start']:8.2f} - {line['end']:8.2f}  {line['score']:.2f}  {line['text']}")
    print(stats)


if __name__ == "__main__":
    main()
