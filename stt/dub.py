"""Dubbing script: which translated lines are voiced, and by which TTS voice.

    subs_env/bin/python -m stt.dub output/stt/ep01 [--voices female=49,male=56] [--interjections dense]
    subs_env/bin/python -m stt.dub output/stt/ep01 --voices female=ru_f1,male=ru_m5 --output dub.ru.teratts.json

Simultaneous speech is never voiced simultaneously: the ASR already yields one stream of
words (secondary voices under the main one are mostly not transcribed), cues follow each
other, and the voice-over timeline solver (`voiceover` package) keeps spoken lines from
colliding. This step removes what should not compete for that single stream: short
interjections/backchannels ("Yeah", "Wow", "Go, go!") that sit on top of another speaker or
inside a rapid exchange stay audible in the ducked original instead of being dubbed.

Writes `dub.ru.json` (or `--output`; voiced lines only, `voiceover` input format plus `speaker`,
`gender`, `voice`), `dub.ru.srt` and `dub.report.json` (skipped lines with reasons).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from stt.runner import load_json, save_json

log = logging.getLogger("stt.dub")

# Lexical content of a line made only of these words is an interjection or backchannel.
INTERJECTIONS = {
    "yeah", "yep", "yup", "uh", "um", "umm", "mm", "mmm", "mhm", "hmm", "huh", "oh", "ooh", "ah", "aah",
    "aw", "aww", "wow", "whoa", "woah", "woo", "woohoo", "yay", "yeehaw", "ha", "haha", "hahaha", "hey",
    "go", "come", "on", "lets", "let's", "my", "god", "gosh", "oops", "ugh", "ow", "ouch", "phew",
}  # fmt: skip
_TOKEN = re.compile(r"[a-z']+")


def is_interjection(text_en: str, max_words: int = 4) -> bool:
    words = _TOKEN.findall(text_en.lower())
    return 0 < len(words) <= max_words and all(w in INTERJECTIONS for w in words)


def parse_voices(spec: str) -> dict[str, int | str]:
    """'female=49,male=56[,unknown=56]' (Vosk ids) or 'female=ru_f1,male=ru_m5' (TeraTTS names) -> mapping;
    unknown defaults to the male voice."""
    voices: dict[str, int | str] = {k.strip(): int(v) if v.strip().isdigit() else v.strip()
                                    for k, v in (item.split("=") for item in spec.split(",") if item.strip())}
    if not {"female", "male"} <= voices.keys():
        raise ValueError("--voices needs female=<id>,male=<id>")
    voices.setdefault("unknown", voices["male"])
    return voices


def plan(
    records: list[dict[str, Any]],
    voices: dict[str, int | str],
    *,
    interjections: str = "dense",
    dense_gap_ms: int = 1000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (voiced lines, skipped lines with reasons)."""
    voiced: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for pos, rec in enumerate(records):
        text_en = " ".join(rec.get("text_en") or [])
        reason = None
        if not " ".join(rec["text"]).strip():
            reason = "empty"
        elif interjections != "keep" and is_interjection(text_en):
            prev_gap = rec["start"] - records[pos - 1]["end"] if pos > 0 else None
            next_gap = records[pos + 1]["start"] - rec["end"] if pos + 1 < len(records) else None
            dense = any(g is not None and g < dense_gap_ms for g in (prev_gap, next_gap))
            if interjections == "all":
                reason = "interjection"
            elif rec.get("overlap"):
                reason = "interjection over another speaker"
            elif dense:
                reason = "interjection in a rapid exchange"
        if reason:
            skipped.append({"index": rec["index"], "start": rec["start"], "end": rec["end"],
                            "speaker": rec.get("speaker"), "gender": rec.get("gender"),
                            "text_en": text_en, "text": rec["text"], "reason": reason})
            continue
        gender = rec.get("gender", "unknown")
        voiced.append({**rec, "voice": voices.get(gender, voices["unknown"])})
    for i, rec in enumerate(voiced, start=1):  # contiguous numbering for downstream tools
        rec["source_index"] = rec["index"]
        rec["index"] = i
    return voiced, skipped


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the dubbing script (voiced lines and voices)")
    parser.add_argument("folder", help="Output folder of stt.translate")
    parser.add_argument("--input", default="subs.ru.json", help="Translated subtitles inside the folder")
    parser.add_argument("--voices", default="female=49,male=56",
                        help="TTS voice per gender: female=49,male=56[,unknown=56] (Vosk), female=ru_f1,male=ru_m5 (TeraTTS)")
    parser.add_argument("--output", default="dub.ru.json", help="Dubbing script file name inside the folder")
    parser.add_argument("--single-voice", type=int, default=None, help="One voice for every line")
    parser.add_argument("--interjections", default="dense", choices=["dense", "all", "keep"],
                        help="Skip interjections over other speech / in rapid exchanges (dense), always, or never")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    folder = Path(args.folder)
    records = load_json(folder / args.input)
    if args.single_voice is not None:
        voices = {"female": args.single_voice, "male": args.single_voice, "unknown": args.single_voice}
    else:
        voices = parse_voices(args.voices)
    voiced, skipped = plan(records, voices, interjections=args.interjections)
    save_json(folder / args.output, voiced)
    save_json(folder / "dub.report.json", {"voices": voices, "interjections": args.interjections,
                                           "voiced": len(voiced), "skipped": skipped})
    from stt.transcribe import write_srt

    write_srt(folder / "dub.ru.srt", voiced, with_speaker=False)
    by_voice = Counter(r["voice"] for r in voiced)
    log.info("%d lines voiced (voices %s), %d skipped -> %s", len(voiced), dict(by_voice), len(skipped),
             folder / args.output)


if __name__ == "__main__":
    main()
