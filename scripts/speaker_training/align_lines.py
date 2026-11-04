"""Training-data build, step 4: pin every SDH-labelled line to the words really spoken (stt.subtitles).

Output per episode: <season>/<code>/lines.json = {"offset_s", "stats", "lines": [{person, start, end, text, tokens, match,
cue_start, cue_end, cluster, cluster_share, other_voice_s, words: [[start, end], ...]}]}. Episodes without a diarization
(the seasons added only for their subtitles) get cluster None: no pseudo segments, gold lines only.

    subs_env/bin/python scripts/speaker_training/align_lines.py [--only S48E01] [--seasons S41,S42]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path.cwd() / "output" / "subs_train"  # the build folder (a symlink to the data drive here)
sys.path.insert(0, str(Path.cwd()))
from stt.subtitles import labelled, names_person, pin_pieces, read_pieces  # noqa: E402

ASR_FILES = ("asr.whisper-large-v3.json", "asr.whisper-large-v3-turbo.json")


def load_words(folder: Path) -> list[SimpleNamespace] | None:
    for name in ASR_FILES:
        path = folder / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            words = [SimpleNamespace(start=float(w["start"]), end=float(w["end"]), text=w["text"], source=None)
                     for s in data["segments"] for w in s.get("words") or [] if w["end"] > w["start"]]
            words.sort(key=lambda w: w.start)
            return words
    return None


def cluster_of(turns: list[dict[str, Any]], start: float, end: float) -> tuple[str | None, float, float]:
    """(main diarizer label of the span, its share of the voiced time, seconds with a second voice at once)."""
    cover: Counter[str] = Counter()
    for t in turns:
        overlap = min(end, t["end"]) - max(start, t["start"])
        if overlap > 0:
            cover[t["speaker"]] += overlap
    if not cover:
        return None, 0.0, 0.0
    label, sec = cover.most_common(1)[0]
    return label, round(sec / sum(cover.values()), 3), round(sum(cover.values()) - sec, 3)


def process(folder: Path, srt: Path) -> dict[str, Any] | None:
    words = load_words(folder)
    if words is None:
        return None
    refined = folder / "diarization.diarizen-v2.refined.resnet293.json"
    turns = json.loads(refined.read_text(encoding="utf-8"))["turns"] if refined.exists() else []
    pieces = read_pieces(srt)
    pinned, offset = pin_pieces(pieces, words)
    named = labelled(pinned)
    lines: list[dict[str, Any]] = []
    for p in named:
        label, share, other = cluster_of(turns, p.start, p.end)
        lines.append({"person": p.piece.person, "start": round(p.start, 3), "end": round(p.end, 3), "text": p.piece.text.strip(),
                      "tokens": len(p.piece.tokens), "match": p.match, "cue_start": p.piece.cue_start, "cue_end": p.piece.cue_end,
                      "cluster": label, "cluster_share": share, "other_voice_s": other,
                      "words": [[round(words[i].start, 3), round(words[i].end, 3)] for i in range(p.first, p.last + 1)]})
    labelled_total = sum(names_person(x.person) for x in pieces)
    if labelled_total >= 20 and len(lines) < 0.3 * labelled_total:
        # The subtitles belong to another cut of the episode (S50E01/E02 pinned 6 of 491 and 4 of 207 named lines; pin_pieces then
        # reports offset 0, so the cue lines below would carry wrong people): nothing from this episode goes to training.
        stats = {"pieces": len(pieces), "labelled": labelled_total, "kept": 0, "not_confirmed": labelled_total, "cue_lines": 0,
                 "out_of_sync": True, "pinned": len(lines)}
        result = {"offset_s": round(offset, 3), "stats": stats, "lines": [], "cue_lines": []}
        (folder / "lines.json").write_text(json.dumps(result, ensure_ascii=False, indent=0), encoding="utf-8")
        return {"code": folder.name, "offset_s": result["offset_s"], **stats, "people": 0, "seconds": 0.0}
    # Lines Whisper did not hear: mostly shouts at challenges, the hard speech. Kept with the cue's own times when the
    # cue holds one voice and the file is in sync.
    pinned_ids = {id(p.piece) for p in pinned}
    voices = Counter(round(x.cue_start, 3) for x in pieces)
    cue_lines: list[dict[str, Any]] = []
    if abs(offset) < 1.0:
        for x in pieces:
            if (names_person(x.person) and id(x) not in pinned_ids and voices[round(x.cue_start, 3)] == 1
                    and 0.4 <= x.cue_end - x.cue_start <= 6.0 and x.tokens):
                label, share, other = cluster_of(turns, x.cue_start, x.cue_end)
                cue_lines.append({"person": x.person, "start": round(x.cue_start, 3), "end": round(x.cue_end, 3), "text": x.text.strip(),
                                  "tokens": len(x.tokens), "cluster": label, "cluster_share": share, "other_voice_s": other})
    stats = {"pieces": len(pieces), "labelled": labelled_total, "kept": len(lines), "not_confirmed": labelled_total - len(lines),
             "cue_lines": len(cue_lines)}
    result = {"offset_s": round(offset, 3), "stats": stats, "lines": lines, "cue_lines": cue_lines}
    (folder / "lines.json").write_text(json.dumps(result, ensure_ascii=False, indent=0), encoding="utf-8")
    return {"code": folder.name, "offset_s": result["offset_s"], **stats, "people": len({x["person"] for x in lines}),
            "seconds": round(sum(x["end"] - x["start"] for x in lines), 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only")
    parser.add_argument("--seasons")
    args = parser.parse_args()
    for wav in sorted(HERE.glob("S*/S*E*/audio.wav")):
        folder = wav.parent
        srt = folder.parent / f"{folder.name}.sdh.srt"
        if (args.only and args.only != folder.name) or not srt.exists():
            continue
        if args.seasons and folder.parent.name not in args.seasons.split(","):
            continue
        row = process(folder, srt)
        if row:
            print(row, flush=True)


if __name__ == "__main__":
    main()
