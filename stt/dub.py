"""Dubbing script: which translated lines are voiced, and by which TTS voice.

    subs_env/bin/python -m stt.dub output/stt/ep01 [--voices female=49,male=56] [--interjections dense]
    subs_env/bin/python -m stt.dub output/stt/ep01 --voices female=ru_f1,male=ru_m5 --output dub.ru.teratts.json

Simultaneous speech is never voiced simultaneously: the ASR already yields one stream of
words (secondary voices under the main one are mostly not transcribed), cues follow each
other, and the voice-over timeline solver (`voiceover` package) keeps spoken lines from
colliding. This step removes what should not compete for that single stream: short
interjections/backchannels ("Yeah", "Wow", "Go, go!") on top of another speaker (`--interjections overlap`,
the default; `stt.crowd` can say them on a layer of their own) or also inside a rapid exchange (`dense`)
stay audible in the ducked original instead of being dubbed; `keep` voices them all, `all` none.

Lines the second recognition pass added where the first heard nothing (`fill` in the records, `stt.fill`:
shouts under music and crowd noise) are voiced by the generic voice of `--neutral-gender` (`--fill-voice
neutral`, the default: who shouts is a guess) or by their speaker's voice like any line (`--fill-voice speaker`).

Manual fixes from `corrections.json` in the folder (`stt.review` lists lines worth checking)
are applied first: `{"123": {"gender": "female"}, "130": {"speaker": "Julie McGee"},
"140": {"text": "..."}, "141": {"skip": true}}` (line numbers as in `subs.ru.srt`). A gender that differs
from the speaker's (a correction, or `gender_by_voice` from `stt.line_gender`) is voiced by the voice of that
gender, not by the speaker's own voice; a corrected gender wins over `gender_by_voice`.

Lines captioned on screen (`on_screen` in the records, set by `stt.transcribe`) are voiced by the engine's generic
voice of `--neutral-gender` (`--on-screen-voice neutral`), never by a speaker's own or the host's voice: the text
on screen is not the hero's utterance. `--on-screen-voice text` (the default) does not voice them at all: the Russian
on-screen text shows them and the original audio stays audible. `--on-screen-voice hero` voices them like any line.
A manual `speaker` or `gender` fix in `corrections.json` wins over `text` and `neutral`: the person decided.
In `text` mode an ASR line mostly under a caption is not voiced either: it is usually the caption's own speech
misheard. In `neutral` mode such a line is compared with the caption (`onscreen.ru.json`): when the caption has a
line of its own and the ASR text looks like it (similarity 0.5+), the ASR copy is left out; when the caption has no
line of its own, the ASR line says the caption's text instead of what the ASR heard ("I don't like being in the snow
bathroom" -> "I really needed this so badly", `caption_text` in the record), unless the ASR line is much longer than
the caption (then the caption was a quieter voice under other speech, which stays as heard).

Writes `dub.ru.json` (or `--output`; voiced lines only, `voiceover` input format plus `speaker`,
`gender`, `voice`), `dub.ru.srt` and `dub.report.json` (skipped lines with reasons).
"""

from __future__ import annotations

import argparse
import difflib
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


CAPTION_PAD_S = 0.4  # a burned-in caption is shown a little before and after its speech
UNDER_CAPTION_SHARE = 0.3  # "text" mode: a line this much under captions on screen is not voiced
CAPTION_COPY_SIMILARITY = 0.65  # an ASR line this similar to the caption it sits under is the caption misheard
CAPTION_LONGER_WORDS = 2  # an ASR line over twice the caption's words (+2) is other speech over a quiet caption


def caption_items(folder: Path) -> list[dict[str, Any]]:
    """The burned-in captions: ``{start, end, text, text_ru}`` (seconds on the episode timeline, like the cues) from
    ``onscreen.ru.json`` (translated by the translate stage) or, without it, ``onscreen.json`` (``text_ru`` None)."""
    path = folder / "onscreen.ru.json"
    if not path.exists():
        path = folder / "onscreen.json"
    if not path.exists():
        return []
    data = load_json(path)
    items = data.get("items", []) if isinstance(data, dict) else data
    return [{"start": float(i["start"]), "end": float(i["end"]), "text": str(i.get("text") or ""),
             "text_ru": i.get("text_ru")} for i in items if i.get("kind") == "caption"]


def caption_spans(folder: Path) -> list[tuple[float, float]]:
    """(start, end) seconds of the burned-in captions (see ``caption_items``)."""
    return [(c["start"], c["end"]) for c in caption_items(folder)]


def _similarity(a: str, b: str) -> float:
    norm = lambda t: " ".join(_TOKEN.findall(t.lower()))  # noqa: E731
    return difflib.SequenceMatcher(None, norm(a), norm(b), autojunk=False).ratio()


def _caption_under(start_ms: float, end_ms: float, items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The caption (``caption_items``) that shares the most time with a line (±CAPTION_PAD_S), if any."""
    start, end = start_ms / 1000, end_ms / 1000
    best, shared = None, 0.0
    for item in items:
        got = min(end, item["end"] + CAPTION_PAD_S) - max(start, item["start"] - CAPTION_PAD_S)
        if got > shared:
            best, shared = item, got
    return best


def under_captions(start_ms: float, end_ms: float, captions: list[tuple[float, float]],
                   share: float = UNDER_CAPTION_SHARE, pad_s: float = CAPTION_PAD_S) -> bool:
    """Whether at least ``share`` of a line falls while a caption is on screen: then it is the caption's own speech as
    the ASR heard it (often garbled: "I don't like being in the snow bathroom" under "I really needed this so badly")
    or talk over it. A long line that only touches a caption (a confessional over a whispered exchange) is not."""
    start, end = start_ms / 1000, end_ms / 1000
    covered = sum(max(0.0, min(end, ce + pad_s) - max(start, cs - pad_s)) for cs, ce in captions)
    return covered > 0 and covered / max(end - start, 1e-3) >= share


def plan(
    records: list[dict[str, Any]],
    voices: dict[str, int | str],
    *,
    interjections: str = "overlap",
    dense_gap_ms: int = 1000,
    skip: set[int] | frozenset[int] = frozenset(),
    speaker_voices: dict[str, int | str] | None = None,
    on_screen_voice: str = "text",
    neutral_gender: str = "male",
    corrected: set[int] | frozenset[int] = frozenset(),
    captions: list[tuple[float, float]] | None = None,
    fill_voice: str = "neutral",
    caption_texts: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (voiced lines, skipped lines with reasons).

    `corrected`: line numbers with a manual speaker/gender fix; these keep the voice that fix implies even
    when the line is captioned on screen. `captions`: (start, end) seconds of the burned-in captions (taken from
    `caption_texts` when not given); in "text" mode a line mostly under them (`under_captions`) is not voiced either:
    the caption's Russian text shows that moment. In "neutral" mode such a line is the caption's speech misheard: with
    `caption_texts` (``caption_items``: start, end, text, text_ru) it is left out when the caption has a line of its own
    and the texts look alike, or says the caption's text when the caption has no line of its own (`caption_text` set);
    a line much longer than the caption, or with no caption texts at hand, stays as heard.
    `fill_voice`: lines of the second recognition pass (`fill`) get the neutral voice ("neutral") or their own ("speaker").
    """
    voiced: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    if captions is None and caption_texts:
        captions = [(c["start"], c["end"]) for c in caption_texts]
    captioned = [(r["start"] / 1000, r["end"] / 1000) for r in records
                 if r.get("on_screen") and r["index"] not in skip]  # the captions' own lines, voiced in "neutral" mode
    said: set[int] = set()  # captions whose text a line under them already says
    for pos, rec in enumerate(records):
        text_en = " ".join(rec.get("text_en") or [])
        reason = None
        if rec["index"] in skip:
            reason = "skipped in corrections.json"
        elif rec.get("on_screen") and on_screen_voice == "text" and rec["index"] not in corrected:
            reason = "on-screen caption: shown as Russian text"
        elif (captions and on_screen_voice == "text" and rec["index"] not in corrected
              and under_captions(rec["start"], rec["end"], captions)):
            reason = "under an on-screen caption: shown as Russian text"
        elif (captions and on_screen_voice == "neutral" and not rec.get("on_screen") and rec["index"] not in corrected
              and under_captions(rec["start"], rec["end"], captions)
              and (cap := _caption_under(rec["start"], rec["end"], caption_texts or [])) is not None):
            own = any(a < cap["end"] + CAPTION_PAD_S and b > cap["start"] - CAPTION_PAD_S for a, b in captioned)
            n_line, n_cap = len(_TOKEN.findall(text_en.lower())), len(_TOKEN.findall(cap["text"].lower()))
            if own:  # the copy, or a fragment of it, is left out; other speech next to the caption is kept
                if _similarity(text_en, cap["text"]) >= CAPTION_COPY_SIMILARITY or n_line <= 2:
                    reason = "under an on-screen caption: the caption's own line is voiced"
            elif id(cap) in said:
                reason = "under an on-screen caption: the caption's text is said by the line before"
            elif cap.get("text_ru") and n_cap and n_line <= CAPTION_LONGER_WORDS * n_cap + 2:
                rec = {**rec, "text": [str(cap["text_ru"])], "text_en": [cap["text"]], "on_screen": True,
                       "caption_text": True}  # the caption's words at the time the ASR heard them
                text_en = cap["text"]
                said.add(id(cap))
        elif not " ".join(rec["text"]).strip():
            reason = "empty"
        elif interjections != "keep" and is_interjection(text_en):
            prev_gap = rec["start"] - records[pos - 1]["end"] if pos > 0 else None
            next_gap = records[pos + 1]["start"] - rec["end"] if pos + 1 < len(records) else None
            dense = any(g is not None and g < dense_gap_ms for g in (prev_gap, next_gap))
            if interjections == "all":
                reason = "interjection"
            elif rec.get("overlap"):
                reason = "interjection over another speaker"
            elif dense and interjections == "dense":
                reason = "interjection in a rapid exchange"
        if reason:
            skipped.append({"index": rec["index"], "start": rec["start"], "end": rec["end"],
                            "speaker": rec.get("speaker"), "gender": rec.get("gender"),
                            "text_en": text_en, "text": rec["text"], "reason": reason})
            continue
        gender = rec.get("gender", "unknown")
        own_voice = None if rec.get("gender_changed") or rec.get("gender_by_voice") \
            else (speaker_voices or {}).get(rec.get("speaker") or "")  # a changed gender needs that gender's voice
        voice = own_voice or voices.get(gender, voices["unknown"])
        if rec.get("on_screen") and on_screen_voice == "neutral" and rec["index"] not in corrected:
            voice = voices.get(neutral_gender, voices["unknown"])
        if rec.get("fill") and fill_voice == "neutral" and rec["index"] not in corrected:
            voice = voices.get(neutral_gender, voices["unknown"])
        voiced.append({**rec, "voice": voice})
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
    parser.add_argument("--speaker-voices", default=None,
                        help="JSON {speaker id: voice} (e.g. cloned voices of the F5 engine); others by gender")
    parser.add_argument("--interjections", default="overlap", choices=["overlap", "dense", "all", "keep"],
                        help="Skip interjections over other speech (overlap), also in rapid exchanges (dense), "
                             "always (all), or never (keep)")
    parser.add_argument("--on-screen-voice", default="text", choices=["text", "neutral", "hero"],
                        help="Lines captioned on screen: not voiced, the Russian on-screen text shows them (text), "
                             "the generic voice of --neutral-gender (neutral) or the speaker's usual one (hero)")
    parser.add_argument("--neutral-gender", default="male", choices=["male", "female"],
                        help="Which generic voice (of --voices) speaks the lines captioned on screen and the filled ones")
    parser.add_argument("--fill-voice", default="neutral", choices=["neutral", "speaker"],
                        help="Lines of the second recognition pass (stt.fill): the generic voice of --neutral-gender "
                             "(neutral) or the speaker's own voice (speaker)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    folder = Path(args.folder)
    records = load_json(folder / args.input)
    skip: set[int] = set()
    corrected: set[int] = set()
    if (folder / "corrections.json").exists():
        from stt.review import apply_corrections

        corrections = load_json(folder / "corrections.json")
        records, skip = apply_corrections(records, corrections, load_json(folder / "speakers.json"))
        log.info("corrections.json: %d lines corrected", len(corrections))
        corrected = {int(k) for k, fix in corrections.items() if "speaker" in fix or "gender" in fix}
    if args.single_voice is not None:
        voices = {"female": args.single_voice, "male": args.single_voice, "unknown": args.single_voice}
    else:
        voices = parse_voices(args.voices)
    speaker_voices = load_json(Path(args.speaker_voices)) if args.speaker_voices else None
    voiced, skipped = plan(records, voices, interjections=args.interjections, skip=skip, speaker_voices=speaker_voices,
                           on_screen_voice=args.on_screen_voice, neutral_gender=args.neutral_gender,
                           corrected=corrected, caption_texts=caption_items(folder), fill_voice=args.fill_voice)
    save_json(folder / args.output, voiced)
    save_json(folder / "dub.report.json", {"voices": voices, "interjections": args.interjections,
                                           "on_screen_voice": args.on_screen_voice, "neutral_gender": args.neutral_gender,
                                           "fill_voice": args.fill_voice, "voiced": len(voiced),
                                           "caption_text": sum(1 for r in voiced if r.get("caption_text")),
                                           "skipped": skipped})
    from stt.transcribe import write_srt

    write_srt(folder / "dub.ru.srt", voiced, with_speaker=False)
    by_voice = Counter(r["voice"] for r in voiced)
    log.info("%d lines voiced (voices %s), %d skipped -> %s", len(voiced), dict(by_voice), len(skipped),
             folder / args.output)


if __name__ == "__main__":
    main()
