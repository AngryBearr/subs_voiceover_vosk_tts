"""Gender of the lines of speakers whose gender stayed unknown, from the lines' own voices.

The voice estimate abstains on little clean speech and the naming LLM may find no evidence, so some
speakers stay "unknown" (voiced as male, translated in the masculine). Each of their lines of 1.2+ s
is classified by the same age/gender model (audeering) and keeps its own confident gender (p >= 0.9):
such speakers are mostly mixed clusters ("camp voices", "challenge voices"); a speaker whose confident
lines mostly agree (3+ lines, 70%) also gets that gender, for its short and unclear lines. Runs after the naming, before the translation, so both the voice and the Russian
grammar follow it. On the S10 SDH-labelled lines (E01/E12/E16, 243 lines) the right gender went from
78.2% to 84.4%; on S29E01/E05-E08 it touched 6 of 2756 lines (there are few unknown speakers).
Trusting the line's voice for speakers of known gender did not pay: the classifier is often the one
that is wrong (shouting men at challenges heard as women).

    subs_env/bin/python -m stt.line_gender <episode folder> [--device cuda]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stt.runner import load_json, save_json

log = logging.getLogger("stt.line_gender")

MIN_MS = 1200  # shorter lines carry too little voice
MAX_MS = 8000  # longer lines are classified on their first 8 s
CONFIDENT = 0.9  # p(female) >= CONFIDENT is female, <= 1 - CONFIDENT male
VOTE_LINES, VOTE_SHARE = 3, 0.7  # a speaker's gender from its lines: at least this many, this share agreeing


def heard(p_female: float) -> str | None:
    return "female" if p_female >= CONFIDENT else "male" if p_female <= 1 - CONFIDENT else None


def decide(cues: list[dict[str, Any]], speakers: list[dict[str, Any]], p_female: dict[int, float]
           ) -> tuple[dict[str, str], dict[int, str]]:
    """({speaker id: gender} voted from the lines, {cue index: gender} for lines of still-unknown speakers)."""
    unknown = {s["id"] for s in speakers if s.get("gender") not in ("male", "female")}
    votes: dict[str, Counter] = defaultdict(Counter)
    for cue in cues:
        g = heard(p_female[cue["index"]]) if cue["index"] in p_female else None
        if cue.get("speaker") in unknown and g:
            votes[cue["speaker"]][g] += 1
    voted = {}
    for sid, counts in votes.items():
        g, n = counts.most_common(1)[0]
        if n >= VOTE_LINES and n / sum(counts.values()) >= VOTE_SHARE:
            voted[sid] = g
    lines = {cue["index"]: g for cue in cues  # a confident line keeps its own voice: these clusters often mix people
             if cue.get("speaker") in unknown and cue["index"] in p_female and (g := heard(p_female[cue["index"]]))}
    return voted, lines


def apply(cues: list[dict[str, Any]], speakers: list[dict[str, Any]], voted: dict[str, str],
          lines: dict[int, str]) -> int:
    """Write the decisions into the speaker cards and the cues; returns the number of cues changed."""
    for speaker in speakers:
        if speaker["id"] in voted:
            speaker.update(gender=voted[speaker["id"]], gender_source="line vote")
    changed = 0
    for cue in cues:
        gender = lines.get(cue["index"]) or voted.get(cue.get("speaker") or "")
        if gender and cue.get("gender") != gender:
            cue["gender"] = gender
            changed += 1
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gender of the lines of speakers of unknown gender, from their voices")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    subs_path, speakers_path = args.folder / "subs.en.json", args.folder / "speakers.json"
    cues, speakers = load_json(subs_path), load_json(speakers_path)
    unknown = {s["id"] for s in speakers if s.get("gender") not in ("male", "female")}
    todo = [c for c in cues if c.get("speaker") in unknown and c["duration"] >= MIN_MS]
    p_female: dict[int, float] = {}
    if todo:
        from stt.audio import SAMPLE_RATE, read_wav
        from stt.voice import AgeGenderEstimator

        audio = read_wav(args.folder / "audio.wav")
        source = args.folder / "source.json"  # a test piece: audio.wav starts at its "start" (seconds)
        offset_ms = float(load_json(source).get("start") or 0.0) * 1000 if source.exists() else 0.0
        estimator = AgeGenderEstimator(device=args.device)
        estimator.load()
        for cue in todo:
            begin, end = cue["start"] - offset_ms, min(cue["end"], cue["start"] + MAX_MS) - offset_ms
            piece = audio[max(0, int(begin / 1000 * SAMPLE_RATE)):max(0, int(end / 1000 * SAMPLE_RATE))]
            if len(piece) < SAMPLE_RATE // 2:
                continue
            _, probs = estimator.predict(piece.astype(np.float32))
            adult = probs[:2] / max(float(probs[:2].sum()), 1e-9)
            p_female[cue["index"]] = round(float(adult[0]), 3)
    voted, lines = decide(cues, speakers, p_female)
    changed = apply(cues, speakers, voted, lines)
    save_json(speakers_path, speakers)
    save_json(subs_path, cues)
    save_json(args.folder / "line_gender.json", {"unknown_speakers": sorted(unknown), "voted": voted,
                                                 "lines": {str(k): v for k, v in lines.items()},
                                                 "p_female": {str(k): v for k, v in p_female.items()}})
    log.info("line gender: %d unknown speakers, %d voted %s, %d more lines from their own voice, %d cues changed",
             len(unknown), len(voted), voted, len(lines), changed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
