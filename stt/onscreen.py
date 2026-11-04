"""On-screen text to show in Russian: captions, name cards and lower thirds.

    subs_env/bin/python -m stt.onscreen translate output/stt/ep01 [--glossary glossary.txt] [--model ...]

`collect` (called by `stt.transcribe`) turns the OCR scan (`ocr.json`) and the per-caption decisions of
`stt.assemble.merge_captions` into display items and writes `onscreen.json`; `translate` adds Russian text
and writes `onscreen.ru.json`. Item schema (times in seconds on the media timeline, like `words.json`):

    {"id": int, "kind": "caption" | "name_card" | "lower_third", "start": float, "end": float,
     "text": str, "descriptor": str | None, "box": [x0, y0, x1, y1] | None (normalised, top-left origin),
     "voiced": bool}            # captions may also have "label" (the speaker prefix "Shonee:")

`voiced` is true only for captions that became part of the transcript (kept, replaced or added);
the other captions, cards and lower thirds are only shown on screen. The translated file adds
`text_ru` (and `descriptor_ru` for cards) and, when it is known, the `gender` ("male" | "female") of the
person who speaks the caption or is named on the card: it drives first-person and past-tense forms.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from difflib import SequenceMatcher
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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
- "name_card": the name of the person on screen ("en") and, in "descriptor", what they do ("ACTOR",
  "LIVESTOCK FARMER"). Names: use "label_ru" / the glossary spelling of the same person, otherwise
  transliterate the way Russian subtitles do. The descriptor is a short job or origin in Russian
  ("АКТЁР", "ФЕРМЕР"; "АКТРИСА" for a "female" person); OCR may have damaged it ("VOIGEACTRESS" = "VOICE ACTRESS"): translate the intended meaning.
- "lower_third": a graphic caption (place, "DAY 3", "THIRTY NINE DAYS", "CURRENTLY IN 7TH PLACE"): short
  Russian phrase, numbers as digits or words the way a Russian caption would write them.
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


def _split_card(card: dict[str, Any], names: set[str], job_words: set[str]) -> None:
    """"DEVIN ACTOR" (two card lines read as one): a known name first and a job after it becomes the name
    and a descriptor; a junk tail ("JENNA VE") is dropped. Changes the card in place."""
    tokens = card["text"].split()
    if len(tokens) < 2 or tokens[0].strip(".,").lower() not in names:
        return
    rest = " ".join(tokens[1:])
    words = [w.lower() for w in _WORD.findall(rest)]
    if _letters(rest) <= 2:
        card["text"] = tokens[0]
    elif words and all(w in job_words and w not in names for w in words):
        card["text"] = tokens[0]
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
        # a roll of 5+ "lower thirds" within a few seconds is the end credits (names of the crew), not graphics
        if sum(abs(o["start"] - third["start"]) <= 3.0 for o in merged) >= 5:
            continue
        third["kind"], third["voiced"] = "lower_third", False
        items.append(third)
    out = []
    for item in sorted(_merge_truncated(items), key=lambda i: (i["start"], i["end"])):
        if item["end"] - item["start"] < min_s or not _WORD.search(item["text"]):
            continue
        out.append({"id": len(out) + 1, "kind": item["kind"], "start": round(item["start"] + offset, 3),
                    "end": round(item["end"] + offset, 3), "text": item["text"].strip(),
                    "descriptor": (item.get("descriptor") or "").strip() or None,
                    "box": item["box"] and [round(x, 4) for x in item["box"]], "voiced": item["voiced"],
                    **({"label": item["label"]} if item.get("label") else {})})
    return out


# ---- translation

def _key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    """What the translation of an item depends on; a change of any part drops its cached translation."""
    return item["kind"], item["text"], item.get("descriptor") or "", item.get("gender") or ""


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
                    cues: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Items with `text_ru` (and `descriptor_ru`). Identical texts by the same kind of speaker are translated
    once; `cached` (the previous `onscreen.ru.json`) supplies translations of unchanged items.

    `cues` (`subs.en.json`) tell who says each caption; every item gets the speaker's `gender`
    when known, so that "I was waiting" is "Я ждала" for a woman."""
    from stt.translate import _prompt, speaker_cards

    items = [dict(item) for item in items]
    add_speaker_genders(items, cues or [], speakers)
    known = {_key(c): (c["text_ru"], c.get("descriptor_ru")) for c in cached or [] if c.get("text_ru")}
    todo: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for item in items:
        if _key(item) not in known:
            todo.setdefault(_key(item), item)
    cards = speaker_cards(speakers)
    pending = list(todo.values())
    for _attempt in range(3):
        if not pending:
            break
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            lines = [{"i": it["id"], "kind": it["kind"], "en": it["text"],
                      **({"descriptor": it["descriptor"]} if it.get("descriptor") else {}),
                      **({"label": it["label"]} if it.get("label") else {}),
                      **({"gender": it["gender"]} if it.get("gender") else {}),
                      **({"speaker": it["speaker_ru"]} if it.get("speaker_ru") else {})} for it in batch]
            answer = client.chat_json(SYSTEM_PROMPT, _prompt(cards, glossary, lines))
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
    """Read `onscreen.json` + `speakers.json` of an episode folder, write and return `onscreen.ru.json`."""
    items = load_json(folder / "onscreen.json")
    path = folder / "onscreen.ru.json"
    cached = load_json(path) if path.exists() else None
    speakers = load_json(folder / "speakers.json") if (folder / "speakers.json").exists() else []
    cues = load_json(folder / "subs.en.json") if (folder / "subs.en.json").exists() else []
    translated = translate_items(items, speakers, client, glossary=glossary, batch_size=batch_size, cached=cached,
                                 cues=cues)
    save_json(path, translated)
    log.info("Wrote %s (%d items)", path, len(translated))
    return translated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="On-screen text: translate onscreen.json to Russian")
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("translate", help="onscreen.json -> onscreen.ru.json")
    tr.add_argument("folder", help="Episode folder (with onscreen.json, speakers.json and subs.en.json)")
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
