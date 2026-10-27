"""Speaker changes of a transcript against the '>>' turn marks of CBS closed captions.

CBS captions start a line with '>>' whenever another person speaks (and name only the host).
The captions' words are aligned to ours (text, not times: caption timing drifts), then:

- recall: share of the captions' turn changes where our speaker changes too (within one word);
- precision: share of our speaker changes that the captions mark as a turn (within one word).

Only changes whose neighbouring words are aligned on both sides count. Roll-up captions (each
cue repeats the previous cue's last line, S29 WEB-DL) are de-duplicated first. Challenge
commentary with players shouting over the host keeps recall below 100% even when we are right
about the main speaker, so the numbers are for comparing variants, not absolute.

    subs_env/bin/python -m stt.turn_eval --srt ep.en.srt output/stt/ep/words.json [other/words.json ...]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import jiwer

from stt.metrics import normalize
from stt.reference import clean_caption_line
from utils.srt_to_json import parse_srt_file


def _key(line: str) -> str:
    return re.sub(r"[^a-z0-9']+", " ", line.lower()).strip()


def caption_lines(srt_path: str | Path) -> list[str]:
    """Caption lines in order, with the lines that roll-up captions repeat dropped."""
    lines: list[str] = []
    last = ""
    for item in parse_srt_file(str(srt_path)):
        cue = [line.strip() for line in item["text"] if line.strip()]
        if cue and last and _key(cue[0]) == last:
            cue = cue[1:]
        if cue:
            last = _key(cue[-1])
            lines += cue
    return lines


def reference_tokens(lines: list[str]) -> tuple[list[str], list[bool]]:
    """Normalized words of the captions and whether each one starts a new speaker turn."""
    tokens: list[str] = []
    turns: list[bool] = []
    for line in lines:
        turn = line.lstrip().startswith(">>")
        text = re.sub(r"^\s*>>\s*[a-zA-Z][a-zA-Z .'&-]{0,30}:\s+", ">> ", line)  # '>> jeff: ' -> '>> '
        for k, token in enumerate(normalize(clean_caption_line(text)).split()):
            tokens.append(token)
            turns.append(turn and k == 0)
    return tokens, turns


def hypothesis_tokens(words: list[dict[str, Any]]) -> tuple[list[str], list[str | None]]:
    """Normalized words of our transcript (words.json) with their speaker."""
    tokens: list[str] = []
    speakers: list[str | None] = []
    for word in words:
        for token in normalize(word["text"]).split():
            tokens.append(token)
            speakers.append(word.get("speaker"))
    return tokens, speakers


def score(ref: list[str], turns: list[bool], hyp: list[str], speakers: list[str | None]) -> dict[str, Any]:
    out = jiwer.process_words(" ".join(ref), " ".join(hyp) or "<empty>")
    r2h: dict[int, int] = {}
    for chunk in out.alignments[0]:
        if chunk.type in ("equal", "substitute"):
            for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                r2h[chunk.ref_start_idx + k] = chunk.hyp_start_idx + k
    h2r = {h: r for r, h in r2h.items()}
    found = marked = 0
    for r, is_turn in enumerate(turns):
        if not is_turn or r == 0 or r not in r2h or r - 1 not in r2h:
            continue
        marked += 1
        a, b = r2h[r - 1], r2h[r]
        found += any(speakers[k] != speakers[k + 1] for k in range(max(0, a - 1), min(len(speakers) - 1, b + 1)))
    ours = right = 0
    for h in range(1, len(speakers)):
        if speakers[h] == speakers[h - 1] or h not in h2r or h - 1 not in h2r:
            continue
        ours += 1
        r = h2r[h]
        right += any(0 <= q < len(turns) and turns[q] for q in (r - 1, r, r + 1))
    recall, precision = found / max(marked, 1), right / max(ours, 1)
    return {"caption_turns": marked, "our_changes": ours, "recall": round(recall, 4),
            "precision": round(precision, 4), "f1": round(2 * recall * precision / max(recall + precision, 1e-9), 4)}


def evaluate(srt_path: str | Path, words: list[dict[str, Any]]) -> dict[str, Any]:
    ref, turns = reference_tokens(caption_lines(srt_path))
    hyp, speakers = hypothesis_tokens(words)
    return score(ref, turns, hyp, speakers)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Speaker changes vs the '>>' marks of CBS captions")
    parser.add_argument("--srt", required=True, help="CBS closed captions of the episode")
    parser.add_argument("words", nargs="+", help="words.json of one or more runs of the episode")
    args = parser.parse_args(argv)
    for path in args.words:
        words = json.loads(Path(path).read_text(encoding="utf-8"))
        print(path, evaluate(args.srt, words))


if __name__ == "__main__":
    main()
