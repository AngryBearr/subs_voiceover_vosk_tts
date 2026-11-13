"""Combine ASR words with diarization into speaker-attributed subtitle cues."""

from __future__ import annotations

import bisect
import difflib
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

from stt.diarize import Diarization, Turn
from stt.types import Segment, Word

_SENTENCE_END = (".", "?", "!", "…")
_CLAUSE_END = (",", ";", ":", "—", "-")


@dataclass
class CueOptions:
    """Cue segmentation limits (tuned for dubbing: one utterance per cue)."""

    max_gap_s: float = 0.8  # pause that starts a new cue after a finished sentence
    max_gap_mid_s: float = 1.0  # pause that splits an unfinished sentence (the dub must not run ahead of it)
    # ...except a pause up to max_join_s between words without punctuation that would leave a phrase
    # of at most fragment_words words on one side: voiced alone, "Google | it." became "Проверь |
    # в Гугле" a few seconds apart. A pause after a comma ("First vote, | Sugar.") still splits.
    fragment_words: int = 2
    max_join_s: float = 4.0
    max_duration_s: float = 7.0
    max_chars: int = 120
    sentence_split_s: float = 2.5  # close a cue at sentence end once it is this long
    nearest_turn_s: float = 0.5  # attach words in diarization gaps to a turn this close
    snap_words: int = 3  # move a mid-sentence speaker change to a sentence boundary this many words away
    majority_words: int = 15  # sentences up to this long get a single (majority) speaker


@dataclass
class Cue:
    start: float
    end: float
    speaker: str | None
    words: list[Word] = field(default_factory=list)
    overlap: bool = False

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


class _TurnIndex:
    """Fast lookup of the speaker talking at a given time span."""

    def __init__(self, turns: list[Turn]) -> None:
        self.turns = sorted(turns, key=lambda t: t.start)
        self.starts = [t.start for t in self.turns]
        self.max_len = max((t.end - t.start for t in self.turns), default=0.0)

    def overlapping(self, start: float, end: float) -> list[Turn]:
        lo = bisect.bisect_left(self.starts, start - self.max_len)
        hi = bisect.bisect_right(self.starts, end)
        return [t for t in self.turns[lo:hi] if t.end > start and t.start < end]

    def best_speaker(self, start: float, end: float, nearest_s: float) -> str | None:
        overlap: dict[str, float] = {}
        for t in self.overlapping(start, end):
            overlap[t.speaker] = overlap.get(t.speaker, 0.0) + min(end, t.end) - max(start, t.start)
        if overlap:
            return max(overlap, key=overlap.__getitem__)
        mid = (start + end) / 2
        near = self.overlapping(mid - nearest_s, mid + nearest_s)
        if not near:
            return None
        return min(near, key=lambda t: min(abs(mid - t.start), abs(mid - t.end))).speaker


def collect_words(segments: list[Segment]) -> list[Word]:
    """All timed words in chronological order; fails if the ASR model gave no word times."""
    words = [w for s in segments for w in s.words]
    if segments and not words:
        raise ValueError(
            "This ASR output has no word timestamps; run it through stt.align (stt.transcribe does it automatically)"
        )
    return sorted(words, key=lambda w: (w.start, w.end))


def assign_speakers(words: list[Word], diarization: Diarization, options: CueOptions) -> list[Word]:
    """Label each word with the speaker that overlaps it most (exclusive turns preferred)."""
    index = _TurnIndex(diarization.exclusive or diarization.turns)
    return [replace(w, speaker=index.best_speaker(w.start, w.end, options.nearest_turn_s)) for w in words]


def _ends_sentence(word: Word) -> bool:
    return word.text.rstrip('"\')»').endswith(_SENTENCE_END)


def snap_speaker_boundaries(words: list[Word], max_shift: int = 3) -> list[Word]:
    """Move speaker changes that fall inside a sentence to the nearest sentence boundary.

    Diarization turn edges are off by a few hundred milliseconds, so the first/last words of
    a sentence often land on the neighbouring speaker. A change is moved only if a sentence
    boundary lies within `max_shift` words; real exchanges ("Yes." / "No.") stay untouched.
    """
    words = list(words)
    i = 1
    while i < len(words):
        prev, cur = words[i - 1], words[i]
        if cur.speaker == prev.speaker or _ends_sentence(prev) or prev.speaker is None or cur.speaker is None:
            i += 1
            continue
        # Forward: the previous speaker finishes the sentence within max_shift words.
        forward = next((j for j in range(i, min(i + max_shift, len(words))) if _ends_sentence(words[j])), None)
        if forward is not None and all(w.speaker == cur.speaker for w in words[i : forward + 1]):
            for j in range(i, forward + 1):
                words[j] = replace(words[j], speaker=prev.speaker)
            i = forward + 1
            continue
        # Backward: the new speaker's sentence started a few words earlier.
        back = next((k for k in range(i - 2, max(-1, i - 1 - max_shift), -1) if _ends_sentence(words[k])), None)
        start = 0 if back is None and i <= max_shift else (back + 1 if back is not None else None)
        if start is not None and all(w.speaker == prev.speaker for w in words[start:i]):
            for j in range(start, i):
                words[j] = replace(words[j], speaker=cur.speaker)
        i += 1
    return words


def sentence_majority(words: list[Word], max_words: int = 15) -> list[Word]:
    """Give every short sentence one speaker: the one owning most of its words.

    Word-level assignment flickers between speakers at turn edges (`What's`=A `your`=B
    `game`=B `plan,`=A `then?`=B); a sentence of up to `max_words` words is almost always
    said by one person. Words outside diarization turns take the sentence's speaker too.
    Longer sentences are left to `snap_speaker_boundaries`.
    """
    words = list(words)
    start = 0
    for i, word in enumerate(words):
        if not _ends_sentence(word) and i < len(words) - 1:
            continue
        sentence = words[start : i + 1]
        votes = Counter(w.speaker for w in sentence if w.speaker is not None)
        mixed = len(votes) > 1 or (votes and any(w.speaker is None for w in sentence))
        if mixed and len(sentence) <= max_words:
            winner = votes.most_common(1)[0][0]
            for j in range(start, i + 1):
                words[j] = replace(words[j], speaker=winner)
        start = i + 1
    return words


def smooth_speakers(words: list[Word], options: CueOptions) -> list[Word]:
    """Sentence-level speaker clean-up: majority for short sentences, boundary snapping otherwise."""
    return snap_speaker_boundaries(sentence_majority(words, options.majority_words), options.snap_words)


_TOKEN = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _caption_groups(captions: list[dict[str, Any]], max_gap_s: float) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for cap in sorted(captions, key=lambda c: c["start"]):
        if groups and cap["start"] - groups[-1][-1]["end"] <= max_gap_s:
            groups[-1].append(cap)
        else:
            groups.append([cap])
    return groups


def _spread(texts: list[str], start: float, end: float) -> list[Word]:
    """Caption words over [start, end], each taking time proportional to its length."""
    weights = [len(t) + 1 for t in texts]
    step = (end - start) / sum(weights)
    words, t = [], start
    for text, weight in zip(texts, weights):
        words.append(Word(start=round(t, 3), end=round(t + weight * step, 3), text=text, source="ocr", caption=True))
        t += weight * step
    return words


def merge_captions(
    words: list[Word],
    captions: list[dict[str, Any]],
    *,
    keep_ratio: float = 0.7,
    before_s: float = 2.5,
    after_s: float = 1.0,
    group_gap_s: float = 1.5,
    lead_s: float = 0.25,
    tail_s: float = 0.6,
    min_word_s: float = 0.12,
    max_shift_s: float = 1.5,
    min_similarity: float = 0.6,
    max_rewrite_extra_s: float = 2.0,
    decisions: list[dict[str, Any]] | None = None,
) -> tuple[list[Word], dict[str, int]]:
    """Put on-screen captions where the ASR missed or garbled them.

    Captions appear about `lead_s` before the speech and stay `tail_s` after it, while ASR
    word times may be off by a second or two. Back-to-back captions are therefore aligned as
    one token stream against the ASR words from `before_s` before to `after_s` after them, so
    every caption finds its own ASR words without taking its neighbours'. Single-word matches
    count only near the caption's speech time, runs of 2+ words within the same margins.

    A caption with at least `keep_ratio` of its words found keeps the ASR words (exact timing), but
    the caption is the truth about its own utterance: if its words differ from the matched ASR words,
    their text is replaced by the caption's (spread over the same time span, `source="ocr"`). Speech
    that is not the caption's utterance is never touched. Otherwise its ASR version is replaced in
    place by the caption text (`source="ocr"`): the matched words, the words between them and the rest of their ASR
    sentences; when none of its words was found, the ASR sentences said entirely while the caption was shown (its
    misheard version). A whole sentence with none of the caption's words next to the ones that have them is someone
    else's ("I know, me too." after "I've been wanting to talk to you alone.") and stays: S48/S49 official subtitles
    have 32 of the 46 such sentences the replacement used to swallow. A caption the ASR missed completely goes to the
    nearest ASR sentence boundary, unless that is more than `max_shift_s` away: then it is a
    quieter voice under transcribed speech and stays only in the original audio.

    The replacement is made only if the caption text is similar (`min_similarity`, character
    match ratio) to the ASR words it would replace; otherwise the caption is another speaker's
    line, a name or a title that merely shares a word with the speech, and the ASR words stay.

    Every word of a captioned utterance (kept, rewritten, replaced or added) gets `caption=True`.
    If `decisions` is given it is filled, in the order of `captions`, with one dict per caption:
    `index`, `start`, `end`, `text`, `action` (kept_asr | replaced | added | under_speech | rejected |
    empty), `changed` (the text of ASR words was replaced), `asr` (the ASR text it replaced) and
    `voiced` (the caption became part of the transcript).
    """
    stats = {"captions": len(captions), "kept_asr": 0, "replaced": 0, "added": 0, "under_speech": 0,
             "rejected": 0}
    verdicts: list[dict[str, Any] | None] = [None] * len(captions)
    order = {id(cap): n for n, cap in enumerate(captions)}

    def decide(cap: dict[str, Any], action: str, asr: str = "", changed: bool = False) -> None:
        verdicts[order[id(cap)]] = {
            "index": order[id(cap)], "start": cap["start"], "end": cap["end"], "text": cap["text"],
            "action": action, "changed": changed, "asr": asr,
            "voiced": action in ("kept_asr", "replaced", "added"),
        }

    words = sorted(words, key=lambda w: (w.start, w.end))
    mids = [(w.start + w.end) / 2 for w in words]
    sentence, first, last = [], {}, {}  # sentence id per word; first/last word index per sentence
    for i, word in enumerate(words):
        sid = sentence[-1] + _ends_sentence(words[i - 1]) if i else 0
        sentence.append(sid)
        first.setdefault(sid, i)
        last[sid] = i
    dropped: set[int] = set()
    marked: set[int] = set()  # kept ASR words of captioned utterances
    inserts: list[tuple[int, float, float, list[str]]] = []  # (position, speech start, end, words)
    for group in _caption_groups(captions, group_gap_s):
        lo, hi = group[0]["start"] - before_s, group[-1]["end"] + after_s
        near = [i for i, w in enumerate(words) if lo <= mids[i] <= hi and w.source != "ocr" and i not in dropped]
        asr_owner = [i for i in near for _ in _tokens(words[i].text)]
        asr_tokens = [t for i in near for t in _tokens(words[i].text)]
        cap_tokens = [_tokens(cap["text"]) for cap in group]
        cap_owner = [k for k, tokens in enumerate(cap_tokens) for _ in tokens]
        speech = [(cap["start"] - lead_s, max(cap["start"] - lead_s + 0.25 * len(cap["text"].split()),
                                              cap["end"] - tail_s)) for cap in group]
        matched: list[set[int]] = [set() for _ in group]
        offsets = [sum(len(ts) for ts in cap_tokens[:k]) for k in range(len(group))]  # first token of each caption
        positions: list[set[int]] = [set() for _ in group]  # matched token positions inside each caption
        found = [0] * len(group)
        matcher = difflib.SequenceMatcher(a=[t for ts in cap_tokens for t in ts], b=asr_tokens, autojunk=False)
        for block in matcher.get_matching_blocks():
            owners = [cap_owner[block.a + j] for j in range(block.size)]
            for j, k in enumerate(owners):
                i = asr_owner[block.b + j]
                t0, t1 = speech[k]
                if (t0 - 0.5 <= mids[i] <= t1 + 0.5
                        or ((owners.count(k) >= 2 or len(cap_tokens[k]) == 1) and t0 - before_s <= mids[i] <= t1 + after_s)):
                    matched[k].add(i)
                    positions[k].add(block.a + j - offsets[k])
                    found[k] += 1
        territory = [set(range(min(m), max(m) + 1)) if m else set() for m in matched]
        for k, cap in enumerate(group):
            texts = cap["text"].split()
            if not cap_tokens[k] or not texts:
                decide(cap, "empty")
                continue
            others = set().union(*(t for j, t in enumerate(territory) if j != k))
            if found[k] / len(cap_tokens[k]) >= keep_ratio:
                stats["kept_asr"] += 1
                own_words = sorted(i for i in territory[k] if i not in others and i not in dropped
                                   and words[i].source != "ocr")
                asr_tokens_own = [t for i in own_words for t in _tokens(words[i].text)]
                changed = False
                if own_words and positions[k]:
                    # ASR words next to the matched ones stand for the caption words that found no match
                    # (a garbled first/last word): same ASR sentence, around the time of the caption's speech.
                    lead, trail = min(positions[k]), len(cap_tokens[k]) - 1 - max(positions[k])
                    t0, t1 = speech[k]
                    for count, step, edge in ((lead, -1, own_words[0]), (trail, 1, own_words[-1])):
                        for _ in range(count):
                            i = edge + step
                            if (i not in near or i in others or i in dropped or sentence[i] != sentence[edge]
                                    or not t0 - 1.0 <= mids[i] <= t1 + 1.0):
                                break
                            own_words.insert(0 if step < 0 else len(own_words), i)
                            edge = i
                    asr_tokens_own = [t for i in own_words for t in _tokens(words[i].text)]
                if own_words:
                    s0, s1 = words[own_words[0]].start, max(words[i].end for i in own_words)
                    if asr_tokens_own != cap_tokens[k] and s1 - s0 <= cap["end"] - cap["start"] + max_rewrite_extra_s:
                        changed = True  # the caption's text wins over the heard words
                        dropped |= set(own_words)
                        end = re.search(r"[.!?…,;:]+$", words[own_words[-1]].text)
                        if end and not texts[-1].endswith(_SENTENCE_END + _CLAUSE_END):  # captions often drop the final stop
                            texts = [*texts[:-1], texts[-1] + end.group()]
                        inserts.append((own_words[0], s0, s1, texts))
                    else:
                        marked |= set(own_words)
                decide(cap, "kept_asr", " ".join(words[i].text for i in own_words) if changed else "", changed)
                continue
            t0, t1 = speech[k]
            own = {sentence[i] for i in territory[k]}
            span = {
                i for i in near if i not in others and (
                    i in territory[k]
                    or (sentence[i] in own and t0 - 1.5 <= mids[i] <= t1 + 1.5)
                    or (not territory[k]  # nothing matched: the sentences said while it was shown are its copy
                        and words[first[sentence[i]]].start >= t0 - 0.5 and words[last[sentence[i]]].end <= cap["end"] + 0.3))
            }
            if span:
                # Replace the ASR words only when the caption really is a version of them: the caption of
                # another speaker, a name or a title shares a word or two with the speech but not its text.
                asr_text = "".join(t for i in sorted(span) for t in _tokens(words[i].text))
                if difflib.SequenceMatcher(None, "".join(cap_tokens[k]), asr_text).ratio() < min_similarity:
                    stats["rejected"] += 1
                    decide(cap, "rejected")
                    continue
            if span:
                t0, t1 = min(t0, min(words[i].start for i in span)), max(t1, max(words[i].end for i in span))
                position = min(span)
            else:
                position = bisect.bisect_left(mids, t0)
                if 0 < position < len(words) and sentence[position - 1] == sentence[position]:
                    sid = sentence[position]  # said under an ASR sentence: move to its nearer edge
                    shift_back, shift_on = t1 - words[first[sid]].start, words[last[sid]].end - t0
                    if min(shift_back, shift_on) > max_shift_s:
                        stats["under_speech"] += 1  # a quieter voice under the transcribed one
                        decide(cap, "under_speech")
                        continue
                    position = first[sid] if shift_back <= shift_on else last[sid] + 1
            stats["replaced" if span else "added"] += 1
            decide(cap, "replaced" if span else "added", " ".join(words[i].text for i in sorted(span)), bool(span))
            dropped |= span
            inserts.append((position, t0, t1, texts))

    # Place captions in sequence order; their times never run back past the previous word.
    out: list[Word] = []
    pending = sorted(inserts, key=lambda x: (x[0], x[1]))
    kept = [i for i in range(len(words)) if i not in dropped]
    for position in range(len(words) + 1):
        while pending and pending[0][0] == position:
            _, t0, t1, texts = pending.pop(0)
            j = bisect.bisect_left(kept, position)
            prev_end = out[-1].end if out else t0
            t0, t1 = max(t0, prev_end), max(t1, prev_end)
            if j < len(kept) and t1 > words[kept[j]].start:  # shift back before the next word
                t0, t1 = max(prev_end, t0 - (t1 - words[kept[j]].start)), words[kept[j]].start
            out += _spread(texts, t0, max(t1, t0 + min_word_s * len(texts)))
        if position < len(words) and position not in dropped:
            out.append(replace(words[position], caption=True) if position in marked else words[position])
    if decisions is not None:
        decisions[:] = [v for v in verdicts if v is not None]
    return out, stats


ROLL_GAP_S = 6.0  # cards of a title roll (opening roster, end credits) follow each other within this
ROLL_MIN_CARDS = 4  # different first words that make a roll (OCR variants of one card share it)
ROLL_MAX_WORDS = 3  # median words spoken under a card of a roll: nobody on it is talking
PLACE_WORDS = frozenset({"tribal", "council", "survivor", "camp", "previously"})


def title_roll_cards(cards: list[dict[str, Any]], words: list[Word]) -> set[int]:
    """Indices of the cards of a title roll: a run of cards of at least ``ROLL_MIN_CARDS`` different first words (one card
    read three ways, "BIANCA" / "BIANCA PR CONSULTAN" / "PR CONSULTANT", is not a roll), each shown within
    ``ROLL_GAP_S`` of the one before, with at most ``ROLL_MAX_WORDS`` words spoken under a card (median). The opening
    roster of S01E01 shows 16 names in 37 s over the narration (0-3 words each), the end credits the crew; a card of an
    interview has the person talking under it (10-25 words) and the cards of a vote come 9-16 s apart."""
    order = sorted(range(len(cards)), key=lambda k: cards[k]["start"])
    middles = sorted((w.start + w.end) / 2 for w in words)
    runs: list[list[int]] = []
    end = float("-inf")
    for k in order:
        if not runs or cards[k]["start"] - end > ROLL_GAP_S:
            runs.append([])
            end = float("-inf")
        runs[-1].append(k)
        end = max(end, cards[k]["end"])
    out: set[int] = set()
    for run in runs:
        firsts = {str(cards[k]["text"]).split()[0].upper() for k in run if str(cards[k]["text"]).split()}
        if len(firsts) < ROLL_MIN_CARDS:
            continue
        spoken = sorted(bisect.bisect_right(middles, cards[k]["end"]) - bisect.bisect_left(middles, cards[k]["start"])
                        for k in run)
        if spoken[len(spoken) // 2] <= ROLL_MAX_WORDS:
            out.update(run)
    return out


def place_card(card: dict[str, Any], tribes: set[str]) -> bool:
    """A card of a tribe, day or place ("TAGI TRIBE", "VULA (DAY4)", "TRIBAL COUNCIL"): it says where the scene is, not
    who speaks. ``tribes``: the tribe names of the episode (``stt.onscreen._tribe_words``)."""
    from stt.onscreen import _generic_line

    text = str(card.get("text") or "")
    return _generic_line(text, tribes) or bool({w.lower() for w in re.findall(r"[^\W_]+", text)} & PLACE_WORDS)


def all_cards(ocr: dict[str, Any], screen_items: list[dict[str, Any]] | None = None, offset: float = 0.0) -> list[dict[str, Any]]:
    """The OCR's cards plus the name cards ``stt.onscreen`` assembles from several lines ("Rudy" over "Retired Navy SEAL -
    Tagi Tribe"): S01 shows its interview cards only that way, in mixed case, and the OCR's own list had none of them.
    ``screen_items`` (``onscreen.json``) are on the media timeline: ``offset`` (the audio's start) is taken off. A name
    card the OCR list already has at that time (same first word) is not added twice."""
    cards = list(ocr.get("cards", []))
    for item in screen_items or []:
        text = str(item.get("text") or "").strip()
        if item.get("kind") != "name_card" or not text:
            continue
        start, end = float(item["start"]) - offset, float(item["end"]) - offset
        first = text.split()[0].upper()
        if any(str(c.get("text") or "").split()[:1] == [first] and min(end, c["end"]) - max(start, c["start"]) > 0
               for c in cards):
            continue
        cards.append({"start": start, "end": end, "kind": "name_card", "text": text, "descriptor": item.get("descriptor")})
    return cards


def on_screen_hints(ocr: dict[str, Any], words: list[Word], diarization: Diarization,
                    max_hints: int = 8, after_s: float = 10.0, screen_items: list[dict[str, Any]] | None = None,
                    offset: float = 0.0) -> dict[str, list[str]]:
    """Per diarizer speaker: names shown on screen while they talked and caption labels of their lines.

    A card goes to the speaker talking most while it is shown; a card shown in a pause (no diarized
    speech under it) goes to the speaker of most of the words in the `after_s` seconds after it
    (on S51E01-E02 the four such cards all belonged to that speaker). Repeated hints are counted
    ("title: SHAUN x3"); OCR spacing variants count as one; the `max_hints` most frequent are kept.
    The cards of a title roll (``title_roll_cards``) and of a tribe, day or place (``place_card``) name nobody talking:
    on S01E01 the opening roster gave six names to one voice, which the naming then called mixed voices. With
    ``screen_items`` the multi-line name cards of ``stt.onscreen`` count too (``all_cards``).
    """
    from stt.onscreen import _tribe_words

    index = _TurnIndex(diarization.exclusive or diarization.turns)
    found: dict[str, list[str]] = {}
    cards = all_cards(ocr, screen_items, offset)
    tribes = _tribe_words([{"text": f"{c.get('text') or ''} {c.get('descriptor') or ''}"} for c in cards])
    roll = title_roll_cards(cards, words)
    for k, card in enumerate(cards):
        if k in roll or place_card(card, tribes):
            continue
        talk: dict[str, float] = {}
        for t in index.overlapping(card["start"], card["end"]):
            talk[t.speaker] = talk.get(t.speaker, 0.0) + min(card["end"], t.end) - max(card["start"], t.start)
        if not talk:
            after = [w.speaker for w in words
                     if w.speaker and card["start"] <= (w.start + w.end) / 2 <= card["end"] + after_s]
            talk = {s: float(n) for s, n in Counter(after).items()}
        if talk:
            label = card["text"] + (f" ({card['descriptor']})" if card.get("descriptor") else "")
            found.setdefault(max(talk, key=talk.__getitem__), []).append(f"{card['kind']}: {label}")
    for cap in ocr.get("captions", []):
        if not cap.get("label"):
            continue
        owners = [w.speaker for w in words if w.speaker and cap["start"] <= (w.start + w.end) / 2 <= cap["end"]]
        if owners:
            found.setdefault(max(set(owners), key=owners.count), []).append(f"caption label: {cap['label']}")
    hints: dict[str, list[str]] = {}
    for speaker, items in found.items():
        counts: Counter[str] = Counter()
        shown: dict[str, str] = {}
        for item in items:
            key = re.sub(r"\s+", "", item)
            counts[key] += 1
            shown.setdefault(key, " ".join(item.split()))
        hints[speaker] = [shown[k] + (f" x{n}" if n > 1 else "") for k, n in counts.most_common(max_hints)]
    return hints


def _split_point(words: list[Word]) -> int:
    """Best index to cut an over-long word run.

    Prefers the last sentence end, then the last clause end (both in the second half),
    then the longest pause between words in the last two thirds.
    """
    half = len(words) // 2
    for marks in (_SENTENCE_END, _CLAUSE_END):
        for i in range(len(words) - 1, half - 1, -1):
            if words[i].text.endswith(marks):
                return i + 1
    first = max(1, len(words) // 3)
    if first >= len(words):
        return len(words)
    gaps = [(words[i].start - words[i - 1].end, i) for i in range(first, len(words))]
    return max(gaps)[1]


def _fragment_pause(words: list[Word], i: int, current: list[Word], options: CueOptions) -> bool:
    """A pause before words[i] inside a phrase (no punctuation before it) that would cut off a
    fragment of at most ``options.fragment_words`` words on either side."""
    prev, word = current[-1], words[i]
    if word.speaker != prev.speaker or word.start - prev.end > options.max_join_s:
        return False
    marks = _SENTENCE_END + _CLAUSE_END
    if prev.text.endswith(marks):
        return False
    head = 0  # words of the phrase before the pause
    for w in reversed(current):
        if w.text.endswith(marks):
            break
        head += 1
    tail = 0  # words of the phrase after it, up to its punctuation or another speaker
    for w in words[i:]:
        if w.speaker != word.speaker:
            break
        tail += 1
        if w.text.endswith(marks) or tail > options.fragment_words:
            break
    return min(head, tail) <= options.fragment_words


def build_cues(words: list[Word], options: CueOptions) -> list[Cue]:
    """Group speaker-labelled words into cues: speaker change, pauses, sentence ends, length limits."""
    cues: list[Cue] = []
    current: list[Word] = []

    def flush(run: list[Word]) -> None:
        if run:
            cues.append(Cue(run[0].start, run[-1].end, run[0].speaker, list(run)))

    for i, word in enumerate(words):
        if current:
            prev = current[-1]
            duration = word.end - current[0].start
            chars = sum(len(w.text) + 1 for w in current) + len(word.text)
            gap = word.start - prev.end
            limit = options.max_gap_s if _ends_sentence(prev) else options.max_gap_mid_s
            if word.speaker != prev.speaker or word.caption != prev.caption \
                    or (word.source == "fill") != (prev.source == "fill") or (gap > limit and not _fragment_pause(words, i, current, options)):
                flush(current)
                current = []
            elif _ends_sentence(prev) and prev.end - current[0].start >= options.sentence_split_s:
                flush(current)
                current = []
            elif duration > options.max_duration_s or chars > options.max_chars:
                cut = _split_point(current)
                flush(current[:cut])
                current = current[cut:]
        current.append(word)
    flush(current)
    return cues


def mark_overlaps(cues: list[Cue], diarization: Diarization, min_share: float = 0.3) -> None:
    """Flag cues where two or more diarized speakers talk at the same time for at least `min_share` of the cue.

    Counted on the turns' timeline, not against the cue's label: a cue the word-level relabelling moved
    to another speaker sits inside the diarizer's turn of the old one, and that is a disagreement about
    who talks, not a second voice.
    """
    index = _TurnIndex(diarization.turns)
    for cue in cues:
        span = cue.end - cue.start
        if span <= 0:
            continue
        events: list[tuple[float, int, str]] = []  # (time, +1 / -1, speaker)
        for t in index.overlapping(cue.start, cue.end):
            events.append((max(t.start, cue.start), 1, t.speaker))
            events.append((min(t.end, cue.end), -1, t.speaker))
        events.sort(key=lambda e: (e[0], e[1]))  # an end before a start at the same time
        active: Counter[str] = Counter()
        shared = 0.0
        last = cue.start
        for time, delta, speaker in events:
            if sum(1 for n in active.values() if n > 0) >= 2:
                shared += time - last
            active[speaker] += delta
            last = time
        cue.overlap = shared / span >= min_share


def speaker_ids(cues: list[Cue]) -> dict[str, str]:
    """Stable public IDs (S01, S02, ...) by first appearance in the cues."""
    mapping: dict[str, str] = {}
    for cue in cues:
        if cue.speaker is not None and cue.speaker not in mapping:
            mapping[cue.speaker] = f"S{len(mapping) + 1:02d}"
    return mapping


def subtitle_records(
    cues: list[Cue],
    ids: dict[str, str],
    genders: dict[str, str],
    *,
    offset_s: float = 0.0,
) -> list[dict[str, Any]]:
    """Cues in the project subtitle schema (ms ints, `text` as list[str]) plus speaker fields."""
    records = []
    for i, cue in enumerate(cues, start=1):
        start = int(round((cue.start + offset_s) * 1000))
        end = max(start + 1, int(round((cue.end + offset_s) * 1000)))
        speaker = ids.get(cue.speaker) if cue.speaker else None
        records.append(
            {
                "index": i,
                "start": start,
                "end": end,
                "duration": end - start,
                "text": [cue.text],
                "gender": genders.get(speaker, "unknown") if speaker else "unknown",
                "speaker": speaker,
                "overlap": cue.overlap,
                "ocr": any(w.source == "ocr" for w in cue.words),
                **({"on_screen": True} if any(w.caption for w in cue.words) else {}),
                **({"fill": True} if any(w.source == "fill" for w in cue.words) else {}),
            }
        )
    return records
