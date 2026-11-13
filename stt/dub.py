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
An ASR line mostly under a caption is compared with the caption's text (`onscreen.ru.json`, `caption_copy`): when it
is the caption's own speech as the ASR heard it (a fragment of the caption's words, a near-verbatim second copy, or a
look-alike of a caption that has no line of its own), `text` mode does not voice it (the caption's Russian text shows
that moment) and `neutral` mode leaves it out when the caption has a line of its own, or says the caption's text
instead of what the ASR heard when it has none ("Should we do like being an angel?" -> "Should we do, like, name and
age and...", `caption_text` in the record). Any other line under a caption is another person talking next to it or
over it ("I appreciate that." after "You struck me as a Ryan.") and is voiced as heard in both modes: the official
subtitles of S48/S49 (which leave burned-in captions out) have 579 of the 925 lines `text` mode used to silence.
When the episode has official subtitles that fit it (`subs.source.en.srt`, `--subtitle-check`, on by default) they
decide instead of the text: a line under a caption that they have is voiced, a line they lack is the caption's
("I don't like being in the snow bathroom" under "I really needed this so badly"); `dub.report.json` says whether
they were used (`subtitles`).

Writes `dub.ru.json` (or `--output`; voiced lines only, `voiceover` input format plus `speaker`,
`gender`, `voice`), `dub.ru.srt` and `dub.report.json` (skipped lines with reasons).
"""

from __future__ import annotations

import argparse
import difflib
import logging
import re
import sys
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

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
UNDER_CAPTION_SHARE = 0.3  # a line this much under captions on screen is compared with them (`caption_copy`)
# A line under a caption is the caption's own speech as the ASR heard it ("copy") when it is this similar to it
# (character similarity, `_similarity`). Measured on the lines of S48E01-E13 and S49E01-E10 under captions, labelled
# by the official subtitles (which leave burned-in captions out): these thresholds keep 571 of 579 lines the subtitles
# have and still find 89 of the 285 they lack (the rest are mostly backchannels the subtitles skip).
COPY_OF_OWN_SIMILARITY = 0.8  # the caption has a line of its own: only a near-verbatim second copy
COPY_SIMILARITY = 0.6  # the caption has no line of its own (a quieter voice the merge put aside): a look-alike...
COPY_ALIGNED_SIMILARITY = 0.4  # ...or a half look-alike said at the caption's time:
COPY_ALIGNED_IOU = 0.2  # sharing this much of their joint time span
CAPTION_LONGER_WORDS = 2  # an ASR line over twice the caption's words (+2) is other speech over a quiet caption
# The episode's official subtitles (`subs.source.en.srt`) leave the burned-in captions out, so they tell a line under a
# caption apart better than its text: a line they have is someone else's speech, a line they lack is the caption's own
# speech as the ASR heard it ("I don't like being in the snow bathroom" under "I really needed this so badly"), a
# fragment, laughter heard as words, or a backchannel the captioner skipped.
SUBTITLE_FOUND = 0.6  # this share of the line's words in the subtitles around it: they have the line
SUBTITLE_MISSING = 0.34  # below this share they lack it; in between the text decides (`caption_copy`)
# ...except a line of this many words that is unlike every caption it shares time with (character similarity below
# SUBTITLE_KEEP_UNLIKE): broadcast captions skip background talk ("Wes, can you bring that over here when you're done
# with it?", S29E07), so the text decides there too. S29E07 (fresh run, 56 lines under captions, read one by one):
# real lines lost 7 -> 3 (one-word pieces), copies voiced 0 -> 1 of 18; S48/S49: real lost 0 -> 0, lines the
# subtitles lack left out 280 -> 256 of 285 (the user's choice 13.11: a misheard copy now and then rather than lost
# context).
SUBTITLE_KEEP_WORDS = 4
SUBTITLE_KEEP_UNLIKE = 0.35
SUBTITLE_MARGIN_S = 1.5  # around the line, after moving the subtitles onto the recognized words
SUBTITLE_HAS_CAPTIONS = 0.3  # a file that writes out this share of the captions (3+ words) has their text too: S21E01
# archive 85% of them; S01, S02, S10, S48, S49 0-6%. Then a subtitle of 2+ words with SUBTITLE_CAPTION_SHARE of them in
# a caption it overlaps is that caption's text and proves nothing; in the other files such a subtitle is an answer
# that echoes the caption ("Yes, I definitely want to talk you." after "I want to talk to you.") and stays.
SUBTITLE_CAPTION_SHARE = 0.7
SUBTITLE_MIN_FIT = 0.7  # used only when they have this share of the other lines: S48/S49 0.94-0.99, S01 archive
# 0.91-0.99, online 0.87-0.93, subtitles shifted by 20 s still 0.98, another episode's subtitles 0.00.
# S48E01-E13 + S49E01-E10 with them: 0 of the 579 lines the subtitles have left out (text alone: 8), 282 of the 285
# they lack (text alone: 89).


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


def _shared_s(start: float, end: float, item: dict[str, Any]) -> float:
    """Seconds a line (start, end in seconds) shares with a caption shown ±CAPTION_PAD_S."""
    return min(end, item["end"] + CAPTION_PAD_S) - max(start, item["start"] - CAPTION_PAD_S)


def _has_own_line(item: dict[str, Any], captioned: list[tuple[float, float]]) -> bool:
    """Whether a caption has a line of its own: a captioned record (seconds) while it is on screen (±CAPTION_PAD_S)."""
    return any(a < item["end"] + CAPTION_PAD_S and b > item["start"] - CAPTION_PAD_S for a, b in captioned)


def caption_copy(text_en: str, start_ms: float, end_ms: float, items: list[dict[str, Any]],
                 captioned: list[tuple[float, float]], in_subtitles: float | None = None) -> dict[str, Any] | None:
    """The caption (``caption_items``) whose speech an ASR line is, as the ASR heard it; None when it is other speech.

    Only the captions the line shares time with (±CAPTION_PAD_S) count. ``in_subtitles``: the share of the line's
    words the official subtitles have around it (``subtitle_shares``), None without usable subtitles. A line they have
    (``SUBTITLE_FOUND``) is other speech; a line they lack (below ``SUBTITLE_MISSING``) is a copy of the caption it
    looks like or else shares the most time with, unless it is a longer line unlike every caption
    (``SUBTITLE_KEEP_WORDS``, ``SUBTITLE_KEEP_UNLIKE``). Otherwise the text decides: the line is a copy when every word of it
    is a word of those captions (a fragment, "to be" under "But we have to be..."), when it is a near-verbatim copy
    (``COPY_OF_OWN_SIMILARITY``), or, for a caption with no line of its own (``captioned``: seconds of the captioned
    records), when it looks like it (``COPY_SIMILARITY``) or half like it while said at its time
    (``COPY_ALIGNED_SIMILARITY``, ``COPY_ALIGNED_IOU``). Anything else is someone else talking next to the caption or
    over it: an answer ("I appreciate that." after "You struck me as a Ryan."), the host, a confessional."""
    start, end = start_ms / 1000, end_ms / 1000
    touching = sorted((c for c in items if _shared_s(start, end, c) > 0), key=lambda c: -_shared_s(start, end, c))
    words = _TOKEN.findall(text_en.lower())
    if not touching or not words or (in_subtitles is not None and in_subtitles >= SUBTITLE_FOUND):
        return None
    if in_subtitles is not None and in_subtitles < SUBTITLE_MISSING:
        alike = {id(c): _similarity(text_en, str(c["text"])) for c in touching}
        if len(words) < SUBTITLE_KEEP_WORDS or max(alike.values()) >= SUBTITLE_KEEP_UNLIKE:
            return next((c for c in touching if alike[id(c)] >= COPY_ALIGNED_SIMILARITY), touching[0])
    shown = {w for c in touching for w in _TOKEN.findall(str(c["text"]).lower())}
    if all(w in shown for w in words):
        return touching[0]
    for item in touching:
        similarity = _similarity(text_en, str(item["text"]))
        if similarity >= COPY_OF_OWN_SIMILARITY:
            return item
        if _has_own_line(item, captioned):
            continue
        joint = max(end, item["end"]) - min(start, item["start"])
        iou = max(0.0, min(end, item["end"]) - max(start, item["start"])) / max(joint, 1e-3)
        if similarity >= COPY_SIMILARITY or (similarity >= COPY_ALIGNED_SIMILARITY and iou >= COPY_ALIGNED_IOU):
            return item
    return None


_SPOKEN = {"gonna": "going to", "wanna": "want to", "gotta": "got to", "kinda": "kind of", "outta": "out of",
           "'cause": "because", "yup": "yep", "ok": "okay", "'em": "them"}


def spoken_tokens(text: str) -> list[str]:
    """Words as said, to compare a line with the subtitles: numbers spelled out ("100%" -> "one hundred percent"),
    "gonna" -> "going to"."""
    from num2words import num2words

    from stt.subtitles import tokens

    def spell(match: re.Match[str]) -> str:
        number = match.group(0).replace(",", "")
        try:
            return f" {num2words(float(number) if '.' in number else int(number))} "
        except (ValueError, OverflowError, NotImplementedError):
            return f" {number} "

    text = re.sub(r"\d+(?:[.,]\d+)*", spell, text.replace("%", " percent ").replace("’", "'"))
    return [part for word in tokens(text) for part in _SPOKEN.get(word, word).split()]


def subtitle_shares(records: list[dict[str, Any]], path: Path, words: Sequence[Any], offset_s: float = 0.0,
                    captions: Sequence[dict[str, Any]] = ()) -> tuple[dict[int, float], float, int]:
    """Per record index, the share of its English words (``text_en``) that the official subtitles at ``path`` have
    within SUBTITLE_MARGIN_S of it; the fit: the share of the lines of 3+ words, not captions, that they have
    (SUBTITLE_FOUND); and, when the file writes out the burned-in ``captions`` too (``caption_items``;
    SUBTITLE_HAS_CAPTIONS), how many of its subtitles were set aside as their text: only a subtitle that is not a
    caption shows someone else's line. Each subtitle is first moved onto the recognized ``words``
    (``stt.subtitles.local_offsets``: another cut of the episode moves them by a different amount per act);
    ``offset_s``: where the words' audio starts in the video."""
    from stt.subtitles import WordIndex, local_offsets, read_pieces

    pieces = [p for p in read_pieces(path) if p.tokens]
    shifts = local_offsets(pieces, WordIndex(words))
    spans = sorted((p.cue_start + tries[0] + offset_s, p.cue_end + tries[0] + offset_s, set(spoken_tokens(p.text)))
                   for p, tries in zip(pieces, shifts))
    shown = [(c["start"] - SUBTITLE_MARGIN_S, c["end"] + SUBTITLE_MARGIN_S, spoken_tokens(str(c["text"])))
             for c in captions]
    written = [sum(w in set().union(*(t for s, e, t in spans if s < b and e > a)) for w in said) / len(said)
               for a, b, said in shown if len(said) >= 3]  # how much of each caption the file writes out
    dropped = 0
    if written and sum(x >= SUBTITLE_FOUND for x in written) / len(written) >= SUBTITLE_HAS_CAPTIONS:
        # a subtitle may hold two captions at once (rolling broadcast captions: "come on, baby. That didn't work.")
        kept = [(s, e, t) for s, e, t in spans
                if len(t) < 2 or len(t & set().union(*(set(said) for a, b, said in shown if a < e and b > s)))
                < SUBTITLE_CAPTION_SHARE * len(t)]
        dropped, spans = len(spans) - len(kept), kept
    starts = [s for s, _, _ in spans]
    longest = max((e - s for s, e, _ in spans), default=0.0)
    shares: dict[int, float] = {}
    for rec in records:
        line = spoken_tokens(" ".join(rec.get("text_en") or []))
        if not line:
            continue
        lo, hi = rec["start"] / 1000 - SUBTITLE_MARGIN_S, rec["end"] / 1000 + SUBTITLE_MARGIN_S
        pool: set[str] = set().union(*(t for _, e, t in spans[bisect_left(starts, lo - longest):bisect_left(starts, hi)]
                                       if e > lo))
        shares[rec["index"]] = sum(t in pool for t in line) / len(line)
    lines = [r["index"] for r in records
             if not r.get("on_screen") and len(_TOKEN.findall(" ".join(r.get("text_en") or []).lower())) >= 3]
    fit = sum(shares.get(i, 0.0) >= SUBTITLE_FOUND for i in lines) / len(lines) if lines else 0.0
    return shares, fit, dropped


def folder_subtitle_shares(folder: Path, records: list[dict[str, Any]],
                           captions: Sequence[dict[str, Any]] = ()) -> tuple[dict[int, float] | None, dict[str, Any]]:
    """``subtitle_shares`` of an episode folder (``subs.source.en.srt`` on the recognized ``words.json``) and what was
    found for ``dub.report.json``; the shares are None without subtitles or when they do not fit (SUBTITLE_MIN_FIT)."""
    path, words_path = folder / "subs.source.en.srt", folder / "words.json"
    if not path.exists() or not words_path.exists():
        return None, {"used": False, "why": "no subtitles" if not path.exists() else "no words.json"}
    from stt.types import Word

    source = load_json(folder / "source.json") if (folder / "source.json").exists() else {}
    fields = Word.__dataclass_fields__
    words = [Word(**{k: v for k, v in w.items() if k in fields}) for w in load_json(words_path)]
    shares, fit, dropped = subtitle_shares(records, path, words, float(source.get("start") or 0.0), captions)
    info: dict[str, Any] = {"used": fit >= SUBTITLE_MIN_FIT, "fit": round(fit, 3), "caption_subtitles": dropped}
    if not info["used"]:
        info["why"] = f"they have only {fit:.0%} of the lines (another cut or episode?)"
    return (shares if info["used"] else None), info


def under_captions(start_ms: float, end_ms: float, captions: list[tuple[float, float]],
                   share: float = UNDER_CAPTION_SHARE, pad_s: float = CAPTION_PAD_S) -> bool:
    """Whether at least ``share`` of a line falls while a caption is on screen: then it may be the caption's own speech
    as the ASR heard it, which ``caption_copy`` decides. A long line that only touches a caption is never one."""
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
    subtitle_share: dict[int, float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (voiced lines, skipped lines with reasons).

    `corrected`: line numbers with a manual speaker/gender fix; these keep the voice that fix implies even
    when the line is captioned on screen. `caption_texts` (``caption_items``: start, end, text, text_ru): the burned-in
    captions; a line mostly under them (`under_captions`) that is a caption's speech as the ASR heard it
    (`caption_copy`) is not voiced in "text" mode (the caption's Russian text shows that moment); in "neutral" mode it
    is left out when the caption has a line of its own, or says the caption's text when it has none (`caption_text`
    set; a line much longer than the caption stays as heard). Other lines under a caption are voiced as heard.
    `captions`: only the (start, end) seconds of the captions, without their text: no line is a copy, all are voiced.
    `subtitle_share` (``folder_subtitle_shares``): per line, the share of its words the official subtitles have; when
    given, they decide which line under a caption is its copy (see `caption_copy`).
    `fill_voice`: lines of the second recognition pass (`fill`) get the neutral voice ("neutral") or their own ("speaker").
    """
    voiced: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    if caption_texts is None:  # times only: no text to compare with, so no line is a caption's copy
        caption_texts = [{"start": s, "end": e, "text": "", "text_ru": None} for s, e in captions or []]
    spans = [(c["start"], c["end"]) for c in caption_texts]
    captioned = [(r["start"] / 1000, r["end"] / 1000) for r in records
                 if r.get("on_screen") and r["index"] not in skip]  # the captions' own lines
    said: set[int] = set()  # captions whose text a line under them already says
    for pos, rec in enumerate(records):
        text_en = " ".join(rec.get("text_en") or [])
        reason = None
        if rec["index"] in skip:
            reason = "skipped in corrections.json"
        elif rec.get("on_screen") and on_screen_voice == "text" and rec["index"] not in corrected:
            reason = "on-screen caption: shown as Russian text"
        elif (on_screen_voice in ("text", "neutral") and not rec.get("on_screen") and rec["index"] not in corrected
              and under_captions(rec["start"], rec["end"], spans)
              and (cap := caption_copy(text_en, rec["start"], rec["end"], caption_texts, captioned,
                                       None if subtitle_share is None else subtitle_share.get(rec["index"])))
              is not None):
            # the caption's own speech as the ASR heard it; other speech next to a caption is voiced as heard
            n_line, n_cap = len(_TOKEN.findall(text_en.lower())), len(_TOKEN.findall(str(cap["text"]).lower()))
            if on_screen_voice == "text":
                reason = "under an on-screen caption: shown as Russian text"
            elif _has_own_line(cap, captioned):
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
    parser.add_argument("--subtitle-check", action=argparse.BooleanOptionalAction, default=True,
                        help="Tell the lines under a burned-in caption by the official subtitles (subs.source.en.srt, "
                             "which leave captions out): a line they have is voiced, a line they lack is the caption's")
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
    captions = caption_items(folder)
    shares: dict[int, float] | None = None
    subtitles: dict[str, Any] = {"used": False,
                                 "why": "--no-subtitle-check" if not args.subtitle_check else "no captions"}
    if args.subtitle_check and captions:
        shares, subtitles = folder_subtitle_shares(folder, records, captions)
        log.info("official subtitles for the lines under captions: %s", subtitles)
    voiced, skipped = plan(records, voices, interjections=args.interjections, skip=skip, speaker_voices=speaker_voices,
                           on_screen_voice=args.on_screen_voice, neutral_gender=args.neutral_gender,
                           corrected=corrected, caption_texts=captions, fill_voice=args.fill_voice,
                           subtitle_share=shares)
    save_json(folder / args.output, voiced)
    save_json(folder / "dub.report.json", {"voices": voices, "interjections": args.interjections,
                                           "on_screen_voice": args.on_screen_voice, "neutral_gender": args.neutral_gender,
                                           "fill_voice": args.fill_voice, "voiced": len(voiced),
                                           "caption_text": sum(1 for r in voiced if r.get("caption_text")),
                                           "subtitles": subtitles,
                                           "skipped": skipped})
    from stt.transcribe import write_srt

    write_srt(folder / "dub.ru.srt", voiced, with_speaker=False)
    by_voice = Counter(r["voice"] for r in voiced)
    log.info("%d lines voiced (voices %s), %d skipped -> %s", len(voiced), dict(by_voice), len(skipped),
             folder / args.output)


if __name__ == "__main__":
    main()
