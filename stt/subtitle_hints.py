"""Existing subtitles as hints for who speaks (used by ``stt.transcribe --subtitles``).

``labels`` files ("SANDRA: ...") give the words of named lines to a person; ``turns`` files (">>") reward a speaker
change where a new voice starts. Plain files give nothing. The words come from ``stt.subtitles.pin_pieces`` (lines the
audio does not confirm are dropped); the states of the persons and the Viterbi pass are in ``stt.boundaries``.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stt.boundaries import WordHints
from stt.subtitles import HOST, labelled, names_person, pin_pieces, read_pieces, subtitle_kind, turn_marks, turn_starts
from stt.types import Word

log = logging.getLogger("stt.subtitle_hints")

MIN_TURN_MARKS = 20  # ">>" marks a file needs to count as marking the changes of voice
MIN_EVIDENCE_LINES = 2  # named lines a state needs before the person is passed on as evidence for its name
MIN_EVIDENCE_SHARE = 0.7  # share of the state's named lines that must name this person


@dataclass
class SubtitleInfo:
    """Facts about the subtitle file and its pinning; the report ``subtitle_hints.json`` is built from these and the hints."""

    kind: str
    offset_s: float
    pinned_lines: int  # labelled lines the audio confirmed
    not_confirmed: int  # labelled lines it did not
    line_persons: list[str]  # person of every confirmed labelled line


def build_hints(path: Path, words: list[Word], start_s: float = 0.0) -> tuple[WordHints, SubtitleInfo]:
    """Hints from a subtitle file for the recognized ``words`` (times from ``start_s`` on the media timeline: the
    subtitles are shifted back by it). A "plain" file gives empty hints."""
    pieces = read_pieces(path)
    kind = subtitle_kind(pieces)
    hints = WordHints()
    if kind == "plain":
        return hints, SubtitleInfo(kind, 0.0, 0, 0, [])
    for piece in pieces:
        piece.cue_start -= start_s
        piece.cue_end -= start_s
    pinned, offset = pin_pieces(pieces, words)
    named = labelled(pinned)
    if kind == "labels":
        persons: dict[str, list[int]] = defaultdict(list)
        for p in named:
            persons[p.piece.person] += [i for i in range(p.first, p.last + 1) if words[i].source != "ocr"]
        hints.persons = {person: sorted(set(idx)) for person, idx in persons.items()}
        # Broadcast captions name mostly the host ("labels") and mark every other voice with ">>": both count.
        marks = turn_marks(pinned)
        if len(marks) >= MIN_TURN_MARKS:
            hints.turns = set(marks)
    else:
        hints.turns = set(turn_starts(pinned))
    in_file = sum(names_person(p.person) for p in pieces)
    info = SubtitleInfo(kind, round(offset, 2), len(named), in_file - len(named), [p.piece.person for p in named])
    log.info("[subtitles] %s: %d lines pinned, %d not confirmed, %d persons, %d turn marks", kind, info.pinned_lines,
             info.not_confirmed, len(hints.persons), len(hints.turns))
    return hints, info


def state_persons(hints: WordHints, info: SubtitleInfo) -> dict[str, tuple[str, int]]:
    """State -> (person, named lines) for the states whose named lines are mostly one person's."""
    lines: dict[str, Counter[str]] = defaultdict(Counter)
    for person in info.line_persons:
        state = hints.state_of.get(person)
        if state is not None:
            lines[state][person] += 1
    out = {}
    for state, votes in lines.items():
        person, n = votes.most_common(1)[0]
        if n >= MIN_EVIDENCE_LINES and n / sum(votes.values()) >= MIN_EVIDENCE_SHARE:
            out[state] = (person, n)
    return out


def evidence(hints: WordHints, info: SubtitleInfo) -> dict[str, list[str]]:
    """State -> naming evidence for ``speakers.json`` ``on_screen`` (the naming LLM reads it like an OCR caption label).
    The host gets "host per subtitles", not a name."""
    return {state: [f"host per subtitles x{n}" if person == HOST else f"caption label: {person} x{n}"]
            for state, (person, n) in state_persons(hints, info).items()}


def subtitle_persons(hints: WordHints, info: SubtitleInfo) -> dict[str, str]:
    """State -> the one person (never the host) that the subtitles name for it."""
    return {state: person for state, (person, _) in state_persons(hints, info).items() if person != HOST}


def report(hints: WordHints, info: SubtitleInfo, words: list[Word], own: list[str | None]) -> dict[str, Any]:
    """The content of ``subtitle_hints.json``. ``words``: the final words, ``own``: their diarizer labels before the hints."""
    return {
        "kind": info.kind,
        "offset_s": info.offset_s,
        "pinned_lines": info.pinned_lines,
        "not_confirmed": info.not_confirmed,
        "fixed_words": len(hints.fixed),
        "turns": len(hints.turns),
        "persons": {person: {"label": hints.state_of.get(person), "words": len(idx)}
                    for person, idx in sorted(hints.persons.items())},
        "relabelled": sum(words[i].speaker != own[i] for i in hints.fixed),
    }
