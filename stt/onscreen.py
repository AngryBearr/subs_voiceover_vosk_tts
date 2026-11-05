"""On-screen text to show in Russian: captions, name cards and lower thirds.

    subs_env/bin/python -m stt.onscreen translate output/stt/ep01 [--glossary glossary.txt] [--model ...]

`collect` (called by `stt.transcribe`) turns the OCR scan (`ocr.json`) and the per-caption decisions of
`stt.assemble.merge_captions` into display items and writes `onscreen.json`; `translate` adds Russian text
and writes `onscreen.ru.json`. Item schema (times in seconds on the media timeline, like `words.json`):

    {"id": int, "kind": "caption" | "name_card" | "lower_third", "start": float, "end": float,
     "text": str, "descriptor": str | None, "box": [x0, y0, x1, y1] | None (normalised, top-left origin),
     "voiced": bool}            # captions may also have "label" (the speaker prefix "Shonee:")

A card or title block that the screen shows as several lines ("CHASE" + "PRO RACE CAR JACKMAN" + "ESPADA - Older
Tribe", or "ESPADA - Older Tribe" + "DAY 1") is ONE item with `lines`: [{"text", "box"}] for every line read from
the screen. A name_card has the name in `text` and the other lines in `descriptor` (joined with "\n", the job
first); a lower_third block keeps all its lines in `text` joined with "\n".

`voiced` is true only for captions that became part of the transcript (kept, replaced or added);
the other captions, cards and lower thirds are only shown on screen. The translated file adds
`text_ru` (and `descriptor_ru` for cards) and, when it is known, the `gender` ("male" | "female") of the
person who speaks the caption or is named on the card: it drives first-person and past-tense forms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from stt.ocr_repair import fix_graphic_text
from stt.llm import DEFAULT_MODEL, OpenRouterClient
from stt.runner import load_json, save_json

log = logging.getLogger("stt.onscreen")

KINDS = ("caption", "name_card", "lower_third")
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)

SYSTEM_PROMPT = """You translate English on-screen text of a TV show (Survivor-style reality series) into Russian
subtitles that are shown on top of the original text.

Each item has "kind":
- "caption": burned-in subtitle of someone's speech ("label" is the speaker prefix, do not translate it into the
  text). Natural, short Russian subtitle text: same meaning and tone, no more characters than needed, keep
  the line breaks out. Use the gender of the speaker card when it is clear who speaks.
  A caption can have "context": "cue" are the dialogue lines (English and the Russian of the dub, made in
  context and proofread) that are said while the caption is on screen, "before" / "after" the neighbouring
  lines. When the caption covers the same words as a cue, follow that cue's Russian (same wording, cut to the
  caption's part if the cue is longer); otherwise translate in the context of the dialogue (who speaks to whom,
  what is meant: "Where do I put it on?" is not "Where do I put it?").
- "name_card": the name of the person on screen ("en") and, in "descriptor", what they do ("ACTOR",
  "LIVESTOCK FARMER") and often the tribe on the next line. Names: use "label_ru" / the glossary spelling of the same
  person (the cast cards), otherwise transliterate the way Russian subtitles do. The descriptor is a short job or
  origin in Russian ("АКТЁР", "ФЕРМЕР"; "АКТРИСА" for a "female" person); OCR may have damaged it
  ("VOIGEACTRESS" = "VOICE ACTRESS"): translate the intended meaning. The descriptor has one line per line of the
  card: keep the line structure, joined with "\n" in "descriptor_ru" (job first, tribe after it).
- "lower_third": a graphic caption (place, "DAY 3", "THIRTY NINE DAYS", "CURRENTLY IN 7TH PLACE", a tribe and a day
  on two lines): short Russian phrase, numbers as digits or words the way a Russian caption would write them.
  A text of several lines ("\n") is translated line by line and keeps its "\n".
If the English text is in upper case, write the Russian in upper case too. Keep show-specific terms from the glossary.
OCR can have dropped or swapped letters: translate the intended text.

Answer with JSON only: {"items": [{"i": <id>, "ru": "<Russian text>", "descriptor_ru": "<Russian descriptor or empty>"}, ...]}
One output item per input item, same "i"."""


def _norm(text: str) -> str:
    return " ".join(w.lower() for w in _WORD.findall(text))


def _contains(a: list[str], b: list[str]) -> bool:
    """Token list `b` occurs inside `a` as a contiguous run."""
    return bool(b) and any(a[i : i + len(b)] == b for i in range(len(a) - len(b) + 1))


def _same_text(a: str, b: str) -> bool:
    """OCR fragments of one text: equal, one inside the other (word-wise), or a typo apart."""
    ta, tb = _norm(a).split(), _norm(b).split()
    if not ta or not tb:
        return False
    return _contains(ta, tb) or _contains(tb, ta) or SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio() >= 0.85


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def _union(boxes: list[list[float]]) -> list[float]:
    return [round(min(b[0] for b in boxes), 4), round(min(b[1] for b in boxes), 4),
            round(max(b[2] for b in boxes), 4), round(max(b[3] for b in boxes), 4)]


def _caption_items(ocr: dict[str, Any], voiced: dict[int, bool]) -> list[dict[str, Any]]:
    lines = [e for e in ocr.get("events", []) if e.get("kind") == "caption"]
    items = []
    for n, cap in enumerate(ocr.get("captions", [])):
        full = _norm(f"{cap.get('label') or ''} {cap['text']}")
        boxes = [e["box"] for e in lines
                 if _overlap(cap["start"], cap["end"], e["start"], e["end"]) > 0.5 * min(cap["end"] - cap["start"], e["end"] - e["start"])
                 and _norm(e["text"]) and (_norm(e["text"]) in full or SequenceMatcher(None, _norm(e["text"]), full).ratio() >= 0.8)]
        item = {"kind": "caption", "start": cap["start"], "end": cap["end"], "text": cap["text"], "descriptor": None,
                "box": _union(boxes) if boxes else None, "voiced": voiced.get(n, False)}
        if cap.get("label"):
            item["label"] = cap["label"]
        items.append(item)
    return items


def _merge_fragments(events: list[dict[str, Any]], gap_s: float) -> list[dict[str, Any]]:
    """Consecutive OCR fragments of one text (an animated lower third is read in several frames,
    sometimes cut: "SPIRIT OF THE PACIFIC" -> "OF THE PACIFIC"): the most complete text wins, times are united."""
    clusters: list[dict[str, Any]] = []
    for ev in sorted(events, key=lambda e: (e["start"], e["end"])):
        for cluster in reversed(clusters):
            if ev["start"] - cluster["end"] <= gap_s and _same_text(cluster["text"], ev["text"]):
                cluster["end"] = max(cluster["end"], ev["end"])
                if len(_norm(ev["text"])) > len(_norm(cluster["text"])):
                    cluster.update(text=ev["text"], box=ev["box"])
                if ev.get("descriptor") and len(ev["descriptor"]) > len(cluster.get("descriptor") or ""):
                    cluster["descriptor"] = ev["descriptor"]
                break
        else:
            clusters.append(dict(ev))
    return clusters


def _card_descriptor(card: dict[str, Any], events: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Lines under a name card and the events they came from.

    A card shows its person's job and often the tribe on the next line ("GAME DESIGNER" / "TOKA TRIBE"); the OCR
    calls them descriptors or lower thirds. They are the simultaneous events right under the card; without any,
    the descriptor variant the OCR classifier noted is used.
    """
    below = sorted((o for o in events if o["kind"] in ("name_card_descriptor", "lower_third")
                    and abs(o["start"] - card["start"]) <= 1.0 and 0.0 < o["box"][1] - card["box"][1] < 0.2
                    and card["box"][0] - 0.08 <= o["box"][0] <= card["box"][2] + 0.35), key=lambda o: o["box"][1])
    if below:
        return "\n".join(dict.fromkeys(o["text"] for o in below)), below
    return next((v.split(": ", 1)[1] for v in card.get("variants", []) if v.startswith("descriptor: ")), None), []


def _letters(text: str) -> int:
    return sum(c.isalpha() for c in text)


def _is_job_word(word: str, job_words: set[str], names: set[str]) -> bool:
    """A word of a job line; a cut animation frame ("OCIOLOGY") of a job word counts too."""
    return word not in names and (word in job_words or any(len(j) >= 5 and word.endswith(j) for j in job_words))


def _split_card(card: dict[str, Any], names: set[str], job_words: set[str], *, trust_first: bool = False) -> None:
    """"DEVIN ACTOR" (two card lines read as one): a known name first and a job after it becomes the name
    and a descriptor; a junk tail ("JENNA VE") is dropped. `trust_first`: the OCR itself called the line a
    name card, so an unknown first word is a name too. Changes the card in place."""
    tokens = card["text"].split()
    if len(tokens) < 2:
        return
    first = tokens[0].strip(".,").lower()
    if first not in names and not (trust_first and first not in job_words):
        return
    cut = 1
    while cut < len(tokens) - 1 and re.fullmatch(r"[^\W\d_]\.", tokens[cut]):  # "JIMMY T. FISHERMAN": initials
        cut += 1
    name = " ".join(tokens[:cut])
    rest = " ".join(tokens[cut:])
    words = [w.lower() for w in _WORD.findall(rest)]
    if _letters(rest) <= 2:
        card["text"] = name
    elif words and all(_is_job_word(w, job_words, names) for w in words):
        card["text"] = name
        old = card.get("descriptor") or ""
        if rest.lower() not in old.lower():
            card["descriptor"] = f"{rest}\n{old}" if old else rest


def _boxes_touch(a: list[float] | None, b: list[float] | None, pad: float = 0.05) -> bool:
    if not a or not b:
        return True
    return a[0] - pad < b[2] and b[0] - pad < a[2] and a[1] - pad < b[3] and b[1] - pad < a[3]


def _is_fragment(short: str, full: str, kind: str) -> bool:
    """`short` is `full` without its first or last 1-3 letters (a truncated animation frame) or, for cards
    and graphics, without its first or last 1-3 whole words."""
    a, b = " ".join(short.lower().split()), " ".join(full.lower().split())
    if len(a) < 3 or len(a) >= len(b):
        return False
    for tail in (False, True):  # `short` is the beginning or the end of `full`
        if not (b.endswith(a) if tail else b.startswith(a)):
            continue
        cut = b[: len(b) - len(a)] if tail else b[len(a) :]
        if (cut[-1] if tail else cut[0]).isalpha() and (a[0] if tail else a[-1]).isalpha():  # a word is cut
            if _letters(cut.split()[-1] if tail else cut.split()[0]) <= 3:
                return True
        elif kind != "caption" and 1 <= len(cut.split()) <= 3:
            return True
    return False


def _merge_truncated(items: list[dict[str, Any]], gap_s: float = 3.0) -> list[dict[str, Any]]:
    """An item that is a truncated read of another of the same kind close in time and place ("ELCOME TO" next
    to "WELCOME TO") is folded into the full one: its text stays, the times are united."""
    absorbed: set[int] = set()
    for n, short in enumerate(items):
        for m, full in enumerate(items):
            if (n == m or m in absorbed or short["kind"] != full["kind"]
                    or _overlap(short["start"], short["end"], full["start"] - gap_s, full["end"] + gap_s) < 0
                    or not _boxes_touch(short.get("box"), full.get("box"))
                    or not _is_fragment(short["text"], full["text"], short["kind"])):
                continue
            full["start"], full["end"] = min(full["start"], short["start"]), max(full["end"], short["end"])
            full["voiced"] = bool(full["voiced"] or short["voiced"])
            if len(short.get("descriptor") or "") > len(full.get("descriptor") or ""):
                full["descriptor"] = short["descriptor"]
            absorbed.add(n)
            break
    for n, third in enumerate(items):  # a graphic read from the job line of a name card ("OCIOLOGY PROFESSOR")
        if third["kind"] != "lower_third" or n in absorbed:
            continue
        for card in (i for m, i in enumerate(items) if i["kind"] == "name_card" and m not in absorbed):
            lines = (card.get("descriptor") or "").split("\n")
            if (_overlap(third["start"], third["end"], card["start"] - gap_s, card["end"] + gap_s) >= 0
                    and _boxes_touch(third.get("box"), card.get("box"))
                    and any(_norm(third["text"]) == _norm(line) or _is_fragment(third["text"], line, "lower_third")
                            for line in lines if line)):
                absorbed.add(n)
                break
    return [item for n, item in enumerate(items) if n not in absorbed]


# ---- card blocks: the lines of one name card or title graphic are one item

BLOCK_KINDS = ("name_card", "name_card_descriptor", "lower_third", "scene")
LOWER_Y = 0.55  # `scene` text (shirts, signs) is taken into a block only in the lower part of the frame


def _compact(text: str) -> str:
    return "".join(w.lower() for w in _WORD.findall(text))


def _area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _v_overlap(a: list[float], b: list[float]) -> float:
    return min(a[3], b[3]) - max(a[1], b[1])


def _same_block_place(a: list[float], b: list[float]) -> bool:
    """Boxes of one graphic: lines stacked and left-aligned, words side by side on one baseline, or the
    same text read twice (heavily overlapping boxes)."""
    top, low = (a, b) if a[1] <= b[1] else (b, a)
    if abs(a[0] - b[0]) <= 0.03 and low[1] - top[1] > 0.03 and -0.06 <= low[1] - top[3] <= 0.05:
        return True
    left, right = (a, b) if a[0] <= b[0] else (b, a)
    if _v_overlap(a, b) >= 0.5 * min(a[3] - a[1], b[3] - b[1]) and -0.04 <= right[0] - left[2] <= 0.06:
        return True
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, _v_overlap(a, b))
    return inter >= 0.5 * min(_area(a), _area(b))


def _together(a: dict[str, Any], b: dict[str, Any], start_s: float = 1.5, *, strict: bool = False) -> bool:
    """Shown at the same time: starts within `start_s` and the intervals overlap (touch, or with `strict`
    overlap by a real part: one card that follows another is not the same card)."""
    shared = _overlap(a["start"], a["end"], b["start"], b["end"])
    need = min(0.3, 0.25 * min(a["end"] - a["start"], b["end"] - b["start"])) if strict else -0.05
    return abs(a["start"] - b["start"]) <= start_s and shared >= need


def _cut_inside_word(short: str, full: str) -> bool:
    """`short` (compact) is the beginning or the end of `full` and starts or ends inside a word of `full`."""
    words = [_compact(w) for w in full.split()]
    ends = {sum(len(w) for w in words[: n + 1]) for n in range(len(words))}
    starts = {0} | ends
    whole = "".join(words)
    if whole.endswith(short):
        return len(whole) - len(short) not in starts
    return whole.startswith(short) and len(short) not in ends


def _fold_pieces(events: list[dict[str, Any]], link_s: float = 1.5
                 ) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Animated lines are read in pieces: a line read as one ("HOLLY SWIMCOACH" over the two lines "HOLLY" and
    "SWIM COACH") and a frame cut inside a word ("OCIOLOGY PROFESSOR" before "JELLY SOCIOLOGY PROFESSOR") are
    folded into the full lines; their time is added to the lines they fold into. Events carry their position in
    "_n". Returns (new dicts without the folded ones, {folded "_n": "_n" of a line it folded into})."""
    events = [dict(e) for e in events]
    gone: set[int] = set()
    into: dict[int, int] = {}
    for n, merged in enumerate(events):  # one read over several lines
        inside = [o for m, o in enumerate(events) if m != n and m not in gone and _together(merged, o, 60.0)
                  and _area(o["box"]) > 0 and _area(_intersection(merged["box"], o["box"])) >= 0.5 * _area(o["box"])
                  and len(_compact(o["text"])) < len(_compact(merged["text"]))]
        inside.sort(key=lambda o: (round(o["box"][1] / 0.03), o["box"][0]))
        glued = "".join(_compact(o["text"]) for o in inside)
        read = _compact(merged["text"])
        if (len(inside) >= 2 and (read in glued or SequenceMatcher(None, read, glued[: len(read) + 2]).ratio() >= 0.85)
                and not any(read in _compact(o["text"]) for o in inside)):
            for o in inside:
                o["start"], o["end"] = min(o["start"], merged["start"]), max(o["end"], merged["end"])
            gone.add(n)
            into[merged["_n"]] = inside[0]["_n"]
    for n, piece in enumerate(events):  # a frame cut inside a word
        short = _compact(piece["text"])
        for m, full in enumerate(events):
            if (m == n or m in gone or n in gone or len(short) < 4 or len(short) >= len(_compact(full["text"]))
                    or _overlap(piece["start"], piece["end"], full["start"] - link_s, full["end"] + link_s) < 0
                    or not _boxes_touch(piece["box"], full["box"]) or not _cut_inside_word(short, full["text"])):
                continue
            full["start"], full["end"] = min(full["start"], piece["start"]), max(full["end"], piece["end"])
            gone.add(n)
            into[piece["_n"]] = full["_n"]
            break
    return [e for n, e in enumerate(events) if n not in gone], into


def _intersection(a: list[float], b: list[float]) -> list[float]:
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


_DAY_LINE = re.compile(r"(day|night)s?\W*\w{0,2}", re.IGNORECASE)


def _tribe_words(events: list[dict[str, Any]]) -> set[str]:
    """Words before "TRIBE" in the on-screen texts of the episode ("TOKA TRIBE", "ESPADA - Older Tribe"):
    the tribe names."""
    found: set[str] = set()
    for e in events:
        words = [w.lower() for w in _WORD.findall(e["text"])]
        for n, word in enumerate(words):
            if word == "tribe" and n:
                found.add(words[n - 1])
    return found


def _generic_line(text: str, tribes: frozenset[str] | set[str] = frozenset()) -> bool:
    """A tribe, day or place line ("TOKA TRIBE", "TOKA", "DAY 1"): never the name of a person."""
    words = [w.lower() for w in _WORD.findall(text)]
    return (not words or "tribe" in words or any(c.isdigit() for c in text) or bool(_DAY_LINE.fullmatch(text.strip()))
            or (len(words) == 1 and words[0] in tribes))


def _reading_order(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Top to bottom; lines on one baseline (name and job side by side) left to right."""
    rows: list[list[dict[str, Any]]] = []
    for line in sorted(lines, key=lambda e: (e["box"][1] + e["box"][3], e["box"][0])):
        for row in rows:
            ref = row[0]["box"]
            if _v_overlap(ref, line["box"]) >= 0.5 * min(ref[3] - ref[1], line["box"][3] - line["box"][1]):
                row.append(line)
                break
        else:
            rows.append([line])
    return [line for row in rows for line in sorted(row, key=lambda e: e["box"][0])]


def _block_groups(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Groups of 2+ text events that are one graphic: shown together and placed as one block."""
    parent = list(range(len(events)))

    def find(n: int) -> int:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for n, a in enumerate(events):
        for m in range(n + 1, len(events)):
            if _together(a, events[m], strict=True) and _same_block_place(a["box"], events[m]["box"]):
                parent[find(n)] = find(m)
    groups: dict[int, list[dict[str, Any]]] = {}
    for n, event in enumerate(events):
        groups.setdefault(find(n), []).append(event)
    return [g for g in groups.values() if len(g) >= 2]


def _dedupe_lines(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same line read twice (typos apart) is one line: the longer text, united time and box."""
    out: list[dict[str, Any]] = []
    for ev in sorted(group, key=lambda e: -len(_compact(e["text"]))):
        twin = next((o for o in out if _compact(o["text"]) == _compact(ev["text"])
                     or (_boxes_touch(o["box"], ev["box"], 0.0) and (
                         _same_text(o["text"], ev["text"])
                         or SequenceMatcher(None, _compact(o["text"]), _compact(ev["text"])).ratio() >= 0.8))), None)
        if twin is None:
            out.append(dict(ev))
            continue
        twin["start"], twin["end"] = min(twin["start"], ev["start"]), max(twin["end"], ev["end"])
        twin["box"] = _union([twin["box"], ev["box"]])
    return out


def _accept_block(lines: list[dict[str, Any]]) -> bool:
    """A block of graphic lines (a card has 2-4 lines on a band of the frame) is a card; a block of `scene`
    text only when it has the shape of a title block (lines on the left margin, a second or more), so that
    shirts, signs and logos stay out."""
    if len(lines) > 4 or _union([e["box"] for e in lines])[3] - _union([e["box"] for e in lines])[1] > 0.25:
        return False  # a list (the credits roll) or text on the whole frame
    if any(e["kind"] != "scene" for e in lines):
        return True
    return (len(lines) >= 2 and min(e["box"][0] for e in lines) <= 0.3
            and max(e["end"] for e in lines) - min(e["start"] for e in lines) >= 1.0
            and all(e["box"][3] - e["box"][1] >= 0.05 for e in lines))


def _block_item(group: list[dict[str, Any]], names: set[str], job_words: set[str],
                tribes: set[str]) -> dict[str, Any]:
    """One item for the lines of a card: name + descriptor lines, or a block of lines without a name."""
    lines = _reading_order([{**e, "text": fix_graphic_text(e["text"])} for e in group])
    plain = [e for e in lines if not _generic_line(e["text"], tribes)]
    name = None
    if not any(_DAY_LINE.fullmatch(e["text"].strip()) for e in lines):  # "TOKA" + "DAY 1" is a title, not a person
        name = (next((e for e in plain if e["kind"] == "name_card"), None)
                or next((e for e in plain if e["text"].split()[0].strip(".,").lower() in names), None))
        first = lines[0]
        if (name is None and first in plain and first["kind"] == "scene" and len(first["text"].split()) == 1
                and any(e["kind"] in ("lower_third", "name_card_descriptor") for e in lines)):
            name = first  # the name is set in a font that the OCR takes for a scene text
    item: dict[str, Any] = {"start": min(e["start"] for e in lines), "end": max(e["end"] for e in lines),
                            "box": _union([e["box"] for e in lines]), "voiced": False,
                            "lines": [{"text": e["text"], "box": [round(x, 4) for x in e["box"]]} for e in lines]}
    if name is None:
        return {**item, "kind": "lower_third", "text": "\n".join(e["text"] for e in lines), "descriptor": None}
    card = {"text": name["text"], "descriptor": None}
    _split_card(card, names, job_words, trust_first=name["kind"] == "name_card")
    rest = [e["text"] for e in lines if e is not name]
    parts = ([card["descriptor"]] if card["descriptor"] else []) + rest
    unique = list(dict.fromkeys(p for p in parts if _letters(p) >= 2))
    return {**item, "kind": "name_card", "text": card["text"], "descriptor": "\n".join(unique) or None}


def _merge_block_items(items: list[dict[str, Any]], gap_s: float) -> list[dict[str, Any]]:
    """The same card read in two runs (a gap in the OCR): one item, the fuller one wins, times are united."""
    out: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda i: (i["start"], i["end"])):
        twin = next((o for o in reversed(out) if o["kind"] == item["kind"] and item["start"] - o["end"] <= gap_s
                     and _same_text(o["text"], item["text"])), None)
        if twin is None:
            out.append(item)
            continue
        start, end = min(twin["start"], item["start"]), max(twin["end"], item["end"])
        if len(item["lines"]) > len(twin["lines"]) or (len(item["lines"]) == len(twin["lines"])
                                                       and len(item["text"]) > len(twin["text"])):
            twin.update(item)
        twin["start"], twin["end"] = start, end
    return out


def _absorb_into_blocks(items: list[dict[str, Any]], gap_s: float) -> list[dict[str, Any]]:
    """A name read alone just before or after its card ("JENNA" and then "JENNA" with the job) is part of the
    card: single-line items equal to the name of a multi-line card next to them in time and place are dropped,
    their time goes to the card."""
    cards = [i for i in items if i.get("lines")]
    out = []
    for item in items:
        host = None if item.get("lines") or item["kind"] == "caption" else next(
            (c for c in cards if c["kind"] == item["kind"] and _same_text(c["text"], item["text"])
             and _overlap(item["start"], item["end"], c["start"] - gap_s, c["end"] + gap_s) >= 0
             and _boxes_touch(item.get("box"), c["box"])), None)
        if host is None:
            out.append(item)
        else:
            host["start"], host["end"] = min(host["start"], item["start"]), max(host["end"], item["end"])
            known = {_norm(line) for line in (host.get("descriptor") or "").split("\n")}
            extra = [line for line in (item.get("descriptor") or "").split("\n") if line and _norm(line) not in known]
            if extra:  # the job read with the name goes before the tribe line
                host["descriptor"] = "\n".join([*extra, *([host["descriptor"]] if host.get("descriptor") else [])])
    return out


def card_blocks(events: list[dict[str, Any]], names: set[str], job_words: set[str], gap_s: float = 1.0,
                tribes: set[str] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(card items, the events that are not in any card). The lines of a name card or title graphic that are
    shown together and placed as one block (stacked left-aligned, or side by side) become one item."""
    cand = [{**e, "_n": n} for n, e in enumerate(events) if e["kind"] in BLOCK_KINDS and e.get("box")
            and _WORD.search(e["text"]) and (e["kind"] != "scene" or (e["box"][1] + e["box"][3]) / 2 >= LOWER_Y)
            and e["box"][2] - e["box"][0] < 0.9 and e["box"][3] - e["box"][1] < 0.3]
    tribes = _tribe_words(events) if tribes is None else tribes
    folded, into = _fold_pieces(cand)
    items: list[dict[str, Any]] = []
    used: set[int] = set()
    for group in _block_groups(folded):
        lines = _dedupe_lines(group)
        if len(lines) >= 2 and _accept_block(lines):
            items.append(_block_item(lines, names, job_words, tribes))
            used |= {e["_n"] for e in group}
    used |= {piece for piece, line in into.items() if line in used}
    return _merge_block_items(items, gap_s), [e for n, e in enumerate(events) if n not in used]


def collect(ocr: dict[str, Any], decisions: list[dict[str, Any]] | None = None, *, offset: float = 0.0,
            gap_s: float = 1.0, min_s: float = 0.3, names: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Display items of an OCR scan, in time order, ids 1..n.

    `decisions` are those of `merge_captions` (caption `index` -> `voiced`); without them no caption is voiced.
    `offset` is added to the times (the audio starts at that point of the media, like in `words.json`).
    `names` (cast, speakers, proper names of the speech) help to split card lines read as one ("DEVIN ACTOR").
    """
    voiced = {d["index"]: bool(d["voiced"]) for d in decisions or []}
    items = _caption_items(ocr, voiced)
    events = [e for e in ocr.get("events", []) if e.get("box")]
    known_names = {n.lower() for name in names for n in _WORD.findall(name)} | {
        e["text"].strip().lower() for e in events if e["kind"] == "name_card" and len(e["text"].split()) == 1}
    job_words = {w.lower() for e in events if e["kind"] in ("lower_third", "name_card_descriptor")
                 for w in _WORD.findall(e["text"])}
    job_words |= {w.lower() for e in events for v in e.get("variants", []) if v.startswith("descriptor: ")
                  for w in _WORD.findall(v)}
    blocks, events = card_blocks(events, known_names, job_words, gap_s)  # multi-line cards and title blocks
    consumed: set[int] = set()  # lower thirds that are the descriptor of a name card
    cards = []
    for ev in (e for e in events if e["kind"] == "name_card"):
        descriptor, sources = _card_descriptor(ev, events)
        consumed |= {id(o) for o in sources}
        cards.append({"start": ev["start"], "end": ev["end"], "text": ev["text"], "descriptor": descriptor,
                      "box": ev["box"], "variants": ev.get("variants", [])})
        _split_card(cards[-1], known_names, job_words)
    for card in _merge_fragments(cards, gap_s):
        card["kind"], card["voiced"] = "name_card", False
        items.append(card)
    thirds = [{"start": e["start"], "end": e["end"], "text": e["text"], "descriptor": None, "box": e["box"]}
              for e in events if e["kind"] == "lower_third" and id(e) not in consumed]
    merged = _merge_fragments(thirds, gap_s)
    for third in merged:
        third["kind"], third["voiced"] = "lower_third", False
    graphics = merged + blocks
    for third in graphics:
        # a roll of 5+ "lower thirds" within a few seconds is the end credits (names of the crew), not graphics
        if sum(abs(o["start"] - third["start"]) <= 3.0 for o in graphics) >= 5:
            continue
        items.append(third)
    items = _absorb_into_blocks(items, gap_s)
    out = []
    for item in sorted(_merge_truncated(items), key=lambda i: (i["start"], i["end"])):
        if item["end"] - item["start"] < min_s or not _WORD.search(item["text"]):
            continue
        out.append({"id": len(out) + 1, "kind": item["kind"], "start": round(item["start"] + offset, 3),
                    "end": round(item["end"] + offset, 3), "text": item["text"].strip(),
                    "descriptor": (item.get("descriptor") or "").strip() or None,
                    "box": item["box"] and [round(x, 4) for x in item["box"]], "voiced": item["voiced"],
                    **({"lines": item["lines"]} if item.get("lines") else {}),
                    **({"label": item["label"]} if item.get("label") else {})})
    return out


# ---- translation

def _key(item: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """What the translation of an item depends on; a change of any part drops its cached translation.
    `context_hash` is the hash of the dialogue context a caption was translated with."""
    return (item["kind"], item["text"], item.get("descriptor") or "", item.get("gender") or "",
            item.get("context_hash") or "")


def _cue_line(cue: dict[str, Any], russian: dict[int, str]) -> dict[str, Any]:
    line: dict[str, Any] = {"t": [round(cue["start"] / 1000, 1), round(cue["end"] / 1000, 1)],
                            "en": " ".join(cue.get("text") or [])}
    if russian.get(cue.get("index", -1)):
        line["ru"] = russian[cue["index"]]
    return line


def caption_context(item: dict[str, Any], cues: list[dict[str, Any]], russian: dict[int, str], near: int = 3
                    ) -> dict[str, Any] | None:
    """The dialogue around a caption: "cue" are the lines of `subs.en.json` said while it is on screen, "before" /
    "after" the `near` neighbours, each with the Russian of `subs.ru.json` (the dub translation, made in
    context and proofread). None without the Russian dub text: then a caption is translated alone."""
    if not russian or not cues:
        return None
    over = [n for n, c in enumerate(cues)
            if _overlap(item["start"], item["end"], c["start"] / 1000, c["end"] / 1000) > 0]
    first = over[0] if over else sum(c["start"] / 1000 <= item["start"] for c in cues)
    last = over[-1] if over else first - 1
    context: dict[str, Any] = {}
    if over:
        context["cue"] = [_cue_line(cues[n], russian) for n in over[:4]]
    context["before"] = [_cue_line(c, russian) for c in cues[max(0, first - near) : first]]
    context["after"] = [_cue_line(c, russian) for c in cues[last + 1 : last + 1 + near]]
    return context


def _known_gender(value: Any) -> str | None:
    return value if value in ("male", "female") else None


def _speaker_by_label(label: str, speakers: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The one speaker whose name starts with the caption label ("Shonee:")."""
    first = label.strip().lower()
    found = [s for s in speakers if first and first in {(s.get("name") or "").split(" ")[0].lower(),
                                                        (s.get("subtitle_person") or "").lower()}]
    return found[0] if len(found) == 1 else None


def caption_speaker(item: dict[str, Any], cues: list[dict[str, Any]],
                    speakers: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """(gender, Russian name) of whoever says a caption: the speaker named by its label, else the cue
    of `subs.en.json` that overlaps it most (a voiced caption has an `on_screen` cue of its own, which wins
    on a tie); (None, None) when no one speaks."""
    by_id = {s["id"]: s for s in speakers}
    person = _speaker_by_label(item["label"], speakers) if item.get("label") else None
    cue: dict[str, Any] | None = None
    if person is None:
        best = 0.0
        for c in cues:
            overlap = _overlap(item["start"], item["end"], c["start"] / 1000, c["end"] / 1000)
            bonus = 1e-3 if c.get("on_screen") and item.get("voiced") else 0.0
            if overlap > 0 and overlap + bonus > best:
                best, cue = overlap + bonus, c
        person = by_id.get(cue["speaker"]) if cue and cue.get("speaker") else None
    gender = _known_gender(cue.get("gender") if cue else None) or _known_gender((person or {}).get("gender"))
    named = bool((person or {}).get("name"))  # an unnamed speaker has only a role in place of a name
    name = ((cue.get("speaker_name") if cue else None) or (person or {}).get("label_ru")) if named else None
    return gender, name


def card_gender(item: dict[str, Any], speakers: list[dict[str, Any]]) -> str | None:
    """Gender of the person a name card names (for "ACTRESS" / "АКТРИСА"): the speaker that has this card, or
    the one with this first name, when all candidates agree."""
    text = item["text"].strip().lower()
    found = []
    for s in speakers:
        first = (s.get("name") or "").split(" ")[0].lower()
        cards = [c.lower() for c in s.get("on_screen") or [] if c.lower().startswith("name_card: ")]
        if first == text or any(c.startswith(f"name_card: {text} ") for c in cards):
            found.append(_known_gender(s.get("gender")))
    return found[0] if found and found[0] and len(set(found)) == 1 else None


def add_speaker_genders(items: list[dict[str, Any]], cues: list[dict[str, Any]],
                        speakers: list[dict[str, Any]]) -> None:
    """Set `gender` (and `speaker_ru` for captions) on the items where it is known; clears stale values."""
    for item in items:
        item.pop("gender", None)
        item.pop("speaker_ru", None)
        if item["kind"] == "caption":
            gender, name = caption_speaker(item, cues, speakers)
            if name and gender:
                item["speaker_ru"] = name
        elif item["kind"] == "name_card":
            gender = card_gender(item, speakers)
        else:
            gender = None
        if gender:
            item["gender"] = gender


def translate_items(items: list[dict[str, Any]], speakers: list[dict[str, Any]], client: Any, *,
                    glossary: str = "", batch_size: int = 80, cached: list[dict[str, Any]] | None = None,
                    workers: int = 4, cues: list[dict[str, Any]] | None = None,
                    cues_ru: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Items with `text_ru` (and `descriptor_ru`). Identical texts by the same kind of speaker are translated
    once; `cached` (the previous `onscreen.ru.json`) supplies translations of unchanged items.

    `cues` (`subs.en.json`) tell who says each caption; every item gets the speaker's `gender`
    when known, so that "I was waiting" is "Я ждала" for a woman. `cues_ru` (`subs.ru.json`) gives each
    caption the Russian of the dialogue around it (`context`), whose hash is part of the cache key."""
    from stt.translate import _prompt, speaker_cards

    items = [dict(item) for item in items]
    add_speaker_genders(items, cues or [], speakers)
    russian = {c["index"]: " ".join(c.get("text") or []) for c in cues_ru or [] if "index" in c}
    contexts: dict[int, dict[str, Any]] = {}
    by_time = sorted(cues or [], key=lambda c: c["start"])
    for item in items:
        item.pop("context_hash", None)
        context = caption_context(item, by_time, russian) \
            if item["kind"] == "caption" else None
        if context:
            contexts[item["id"]] = context
            item["context_hash"] = hashlib.sha1(json.dumps(context, sort_keys=True, ensure_ascii=False)
                                                .encode("utf-8")).hexdigest()[:12]
    known = {_key(c): (c["text_ru"], c.get("descriptor_ru")) for c in cached or [] if c.get("text_ru")}
    todo: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for item in items:
        if _key(item) not in known:
            todo.setdefault(_key(item), item)
    cards = speaker_cards(speakers)
    pending = list(todo.values())
    for _attempt in range(3):
        if not pending:
            break
        batches = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]

        def ask(batch: list[dict[str, Any]]) -> Any:
            lines = [{"i": it["id"], "kind": it["kind"], "en": it["text"],
                      **({"descriptor": it["descriptor"]} if it.get("descriptor") else {}),
                      **({"label": it["label"]} if it.get("label") else {}),
                      **({"gender": it["gender"]} if it.get("gender") else {}),
                      **({"speaker": it["speaker_ru"]} if it.get("speaker_ru") else {}),
                      **({"context": contexts[it["id"]]} if it["id"] in contexts else {})} for it in batch]
            return client.chat_json(SYSTEM_PROMPT, _prompt(cards, glossary, lines))

        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(batches)))) as pool:  # independent requests
            answers = list(pool.map(ask, batches))  # in batch order; the first error is raised
        for batch, answer in zip(batches, answers):
            rows = answer.get("items", []) if isinstance(answer, dict) else answer
            by_id = {r["i"]: r for r in rows if isinstance(r, dict) and str(r.get("ru", "")).strip()}
            for it in batch:
                row = by_id.get(it["id"])
                if row:
                    known[_key(it)] = (str(row["ru"]).strip(), str(row.get("descriptor_ru") or "").strip() or None)
        pending = [it for it in pending if _key(it) not in known]
        if pending:
            log.warning("%d on-screen items not translated, retrying", len(pending))
    if pending:
        raise RuntimeError(f"On-screen items not translated: {[it['id'] for it in pending]}")
    out = []
    for item in items:
        text_ru, descriptor_ru = known[_key(item)]
        out.append({**item, "text_ru": text_ru, **({"descriptor_ru": descriptor_ru} if item.get("descriptor") else {})})
    return out


def translate_folder(folder: Path, client: Any, *, glossary: str = "", batch_size: int = 80) -> list[dict[str, Any]]:
    """Read `onscreen.json` and, when they exist, `speakers.json`, `subs.en.json` and `subs.ru.json` of an episode
    folder, write and return `onscreen.ru.json`."""
    items = load_json(folder / "onscreen.json")
    path = folder / "onscreen.ru.json"
    cached = load_json(path) if path.exists() else None
    speakers = load_json(folder / "speakers.json") if (folder / "speakers.json").exists() else []
    cues = load_json(folder / "subs.en.json") if (folder / "subs.en.json").exists() else []
    cues_ru = load_json(folder / "subs.ru.json") if (folder / "subs.ru.json").exists() else []  # dub text: context
    translated = translate_items(items, speakers, client, glossary=glossary, batch_size=batch_size, cached=cached,
                                 cues=cues, cues_ru=cues_ru)
    save_json(path, translated)
    log.info("Wrote %s (%d items)", path, len(translated))
    return translated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="On-screen text: translate onscreen.json to Russian")
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("translate", help="onscreen.json -> onscreen.ru.json")
    tr.add_argument("folder", help="Episode folder (with onscreen.json, speakers.json, subs.en.json and subs.ru.json)")
    tr.add_argument("--model", default=DEFAULT_MODEL, help="LLM id, as for stt.translate")
    tr.add_argument("--glossary", default=None, help="Glossary text file (default: glossary.txt in the folder)")
    tr.add_argument("--reasoning", default="medium", choices=["default", "off", "minimal", "low", "medium", "high"])
    tr.add_argument("--batch-size", type=int, default=80)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    folder = Path(args.folder)
    glossary_path = Path(args.glossary) if args.glossary else folder / "glossary.txt"
    glossary = glossary_path.read_text(encoding="utf-8") if glossary_path.exists() else ""
    client = OpenRouterClient(args.model, reasoning=None if args.reasoning == "default" else args.reasoning)
    translate_folder(folder, client, glossary=glossary, batch_size=args.batch_size)
    log.info("usage %s", client.usage.to_dict())


if __name__ == "__main__":
    main()
