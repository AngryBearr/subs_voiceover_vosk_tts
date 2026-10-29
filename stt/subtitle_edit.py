"""Edit an episode's Russian lines (text, speaker, gender, not voiced) outside or inside the UI.

The lines are ``subs.ru.json`` with ``corrections.json`` applied (``stt.review.apply_corrections``).
An edit is saved as a correction (``corrections.json``, keyed by the line numbers of
``subs.ru.srt``), and the next ``stt.run`` of the episode redoes the dubbing script, the stress
check, the voice-over, the checks and the mix; only the lines whose text or voice changed are
synthesised again (every other take comes from the TTS cache), so an episode takes minutes.

Subtitle Edit round trip: ``export`` writes the lines as ASS (the speaker in "Actor", the gender
as the style "Мужской" / "Женский", "Не озвучивать" for a line not to voice, the line number in
"Effect"; ``--review``: only the lines of ``review.json`` with what was heard as a hidden
``{...}`` note) or as SRT ("[Speaker] text"); ``import`` reads the edited file back, matches its
lines by number or start time, and turns the differences into corrections. A stress mark typed
as ``+`` before the vowel ("зам+ок") is kept by the engines.

    subs_env/bin/python -m stt.subtitle_edit export output/stt/<episode> [--review] [--srt]
    subs_env/bin/python -m stt.subtitle_edit import output/stt/<episode> edited.ass [--dry-run]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stt.runner import REPO_ROOT, load_json, save_json

log = logging.getLogger("stt.subtitle_edit")

STYLE_MALE, STYLE_FEMALE, STYLE_SKIP = "Мужской", "Женский", "Не озвучивать"
STYLE_GENDER = {STYLE_MALE: "male", STYLE_FEMALE: "female"}
MATCH_MS = 500  # an edited line belongs to the line starting this close
_NOTE_RE = re.compile(r"\{[^}]*\}")
_SPEAKER_RE = re.compile(r"^\s*\[([^\]]+)\]\s*")
_SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


@dataclass
class Line:
    """One Russian line as it will be voiced (corrections applied)."""

    index: int
    start: int
    end: int
    speaker: str | None  # speaker id
    speaker_name: str
    gender: str
    text: str
    skip: bool
    text_en: str
    window_ms: int


@dataclass
class Edited:
    """A line of an edited subtitle file."""

    start: int
    end: int
    text: str
    speaker: str | None = None  # name as typed; None: not given (SRT without "[...]")
    gender: str | None = None
    skip: bool = False
    number: int | None = None
    marks_skip: bool = True  # the file can say "not voiced" (ASS style); SRT cannot: a line stays as it was


@dataclass
class Fit:
    syllables: int
    seconds: float
    rate: float
    verdict: str  # ok | fast | limit | over
    excess: int  # syllables to remove for the ceiling tempo
    text: str  # Russian explanation


def speaker_label(card: dict[str, Any]) -> str:
    return str(card.get("label_ru") or card.get("name") or card["id"])


def find_speaker(speakers: list[dict[str, Any]], wanted: str) -> dict[str, Any] | None:
    """The speaker card a name refers to: id, Russian label or a spelling of the English name."""
    from stt.voice_bank import same_person

    wanted = wanted.strip()
    return next((s for s in speakers if wanted in (s["id"], s.get("label_ru"))
                 or (s.get("name") and same_person(s["name"], wanted))), None)


def _text(value: Any) -> str:
    return " ".join(value) if isinstance(value, list) else str(value or "")


def _same(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


def current_lines(folder: Path) -> list[Line]:
    """The episode's lines with ``corrections.json`` applied."""
    from stt.review import apply_corrections

    records = load_json(folder / "subs.ru.json")
    path = folder / "corrections.json"
    skip: set[int] = set()
    if path.exists():
        records, skip = apply_corrections(copy.deepcopy(records), load_json(path), load_json(folder / "speakers.json"))
    return [Line(r["index"], r["start"], r["end"], r.get("speaker"), str(r.get("speaker_name") or r.get("speaker") or ""),
                 str(r.get("gender") or "unknown"), _text(r.get("text")), r["index"] in skip,
                 _text(r.get("text_en")), int(r.get("window_ms") or (r["end"] - r["start"])))
            for r in records]


def _policy() -> dict[str, float]:
    """Tempo limits of the voice-over: base, comfort and ceiling syllables per second."""
    from stt.run import load_config

    policy = dict(load_json(REPO_ROOT / "voiceover_config.json").get("policy") or {})
    policy.update((load_config(REPO_ROOT / "stt_config.json").get("voiceover") or {}).get("policy") or {})
    return {key: float(policy.get(key, default)) for key, default in
            (("base_syl_per_sec", 5.0), ("comfort_syl_per_sec", 6.3), ("ceiling_syl_per_sec", 6.7))}


def fit(text: str, window_ms: int, policy: dict[str, float] | None = None) -> Fit:
    """Will the text fit the time of its line? (the solver may also shift it a little)"""
    from voiceover.text import syllables

    policy = policy or _policy()
    count = syllables(text.replace("+", ""))
    seconds = max(window_ms, 1) / 1000
    rate = count / seconds
    ceiling = policy["ceiling_syl_per_sec"]
    excess = max(0, math.ceil(count - ceiling * seconds))
    if rate <= policy["base_syl_per_sec"]:
        verdict, note = "ok", "влезет"
    elif rate <= policy["comfort_syl_per_sec"]:
        verdict, note = "fast", "влезет, чуть быстрее обычного"
    elif rate <= ceiling:
        verdict, note = "limit", "на пределе скорости"
    else:
        verdict, note = "over", f"не влезет: уберите примерно {excess} слог."
    return Fit(count, seconds, rate, verdict, excess,
               f"{count} слог. на {seconds:.1f} с ({rate:.1f} слог/с; обычно до {policy['base_syl_per_sec']:g}, "
               f"предел {ceiling:g}): {note}")


# ---------------------------------------------------------------- files

def _ass_time(ms: int) -> str:
    cs = max(0, round(ms / 10))
    hours, rest = divmod(cs, 360000)
    minutes, rest = divmod(rest, 6000)
    seconds, cs = divmod(rest, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{cs:02d}"


def _parse_ass_time(value: str) -> int:
    hours, minutes, seconds = value.strip().split(":")
    return round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000)


def _srt_time(ms: int) -> str:
    hours, rest = divmod(max(0, ms), 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _selection(folder: Path, review: bool) -> tuple[list[Line], dict[int, str]]:
    lines = current_lines(folder)
    if not review:
        return lines, {}
    notes = {}
    for item in load_json(folder / "review.json"):
        heard = [f"слышно {item['heard']}" if item.get("heard") else "",
                 f"похоже на {item['sounds_like']}" if item.get("sounds_like") else ""]
        notes[int(item["line"])] = "; ".join(h for h in heard if h)
    return [line for line in lines if line.index in notes], notes


def export_ass(folder: Path, path: Path, *, review: bool = False) -> Path:
    """The lines for Subtitle Edit: speaker as Actor, gender as style, the line number in Effect."""
    lines, notes = _selection(folder, review)
    speakers = load_json(folder / "speakers.json")
    names = ", ".join(sorted({speaker_label(s) for s in speakers if s.get("cues")}))
    style = "{name},Arial,56,{colour},&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,60,60,50,204"
    head = [
        "[Script Info]",
        f"; {folder.name}: русские реплики для правки (stt.subtitle_edit)",
        "; Говорящий - поле «Актёр» (Actor); пол - стиль «Мужской» / «Женский»; «Не озвучивать» - реплику не озвучивать.",
        "; Ударение: + перед ударной гласной (зам+ок). Время и поле «Эффект» не меняйте: по ним реплики сверяются.",
        f"; Говорящие этой серии: {names}",
        "ScriptType: v4.00+", "PlayResX: 1920", "PlayResY: 1080", "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        "Style: " + style.format(name=STYLE_MALE, colour="&H00FFFFFF"),
        "Style: " + style.format(name=STYLE_FEMALE, colour="&H0080FFFF"),
        "Style: " + style.format(name=STYLE_SKIP, colour="&H00909090"), "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    events = []
    for line in lines:
        kind = STYLE_SKIP if line.skip else STYLE_FEMALE if line.gender == "female" else STYLE_MALE
        note = "{" + notes[line.index].replace("{", "(").replace("}", ")") + "}" if notes.get(line.index) else ""
        actor = line.speaker_name.replace(",", " ")
        events.append(f"Dialogue: 0,{_ass_time(line.start)},{_ass_time(line.end)},{kind},{actor},0,0,0,"
                      f"#{line.index},{note}{line.text}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(head + events) + "\n", encoding="utf-8-sig")
    return path


def export_srt(folder: Path, path: Path, *, review: bool = False) -> Path:
    """The lines as SRT, "[Speaker] text"; the gender and "not voiced" are not in SRT."""
    lines, _ = _selection(folder, review)
    blocks = [f"{line.index}\n{_srt_time(line.start)} --> {_srt_time(line.end)}\n[{line.speaker_name}] {line.text}\n"
              for line in lines]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def read_ass(text: str) -> list[Edited]:
    fields: list[str] = []
    out: list[Edited] = []
    section = ""
    for raw in text.splitlines():
        row = raw.strip()
        if row.startswith("["):
            section = row.lower()
            continue
        if section != "[events]":
            continue
        if row.lower().startswith("format:"):
            fields = [f.strip().lower() for f in row.split(":", 1)[1].split(",")]
            continue
        if not row.lower().startswith("dialogue:") or not fields:
            continue  # "Comment:" lines are left out
        values = dict(zip(fields, row.split(":", 1)[1].split(",", len(fields) - 1)))
        body = _NOTE_RE.sub("", values.get("text", "")).replace("\\N", " ").replace("\\n", " ")
        style = values.get("style", "").strip()
        effect = re.search(r"#(\d+)", values.get("effect", ""))
        out.append(Edited(_parse_ass_time(values["start"]), _parse_ass_time(values["end"]), " ".join(body.split()),
                          values.get("name", "").strip() or None, STYLE_GENDER.get(style), style == STYLE_SKIP,
                          int(effect.group(1)) if effect else None))
    return out


def read_srt(text: str) -> list[Edited]:
    out: list[Edited] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        rows = block.strip().split("\n")
        at = next((i for i, row in enumerate(rows) if _SRT_TIME.search(row)), None)
        if at is None:
            continue
        h1, m1, s1, f1, h2, m2, s2, f2 = _SRT_TIME.search(rows[at]).groups()  # type: ignore[union-attr]
        start = ((int(h1) * 60 + int(m1)) * 60 + int(s1)) * 1000 + int(f1.ljust(3, "0")[:3])
        end = ((int(h2) * 60 + int(m2)) * 60 + int(s2)) * 1000 + int(f2.ljust(3, "0")[:3])
        body = " ".join(" ".join(rows[at + 1:]).split())
        body = re.sub(r"</?[a-zA-Z][^>]*>", "", body)  # <i>, <font ...> of other editors
        speaker = None
        match = _SPEAKER_RE.match(body)
        if match:
            speaker, body = match.group(1).strip(), body[match.end():]
        number = int(rows[at - 1]) if at > 0 and rows[at - 1].strip().isdigit() else None
        out.append(Edited(start, end, body.strip(), speaker, None, not body.strip(), number, marks_skip=False))
    return out


def read_edited(path: Path) -> list[Edited]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return read_ass(text) if "[events]" in text.lower() else read_srt(text)


# ---------------------------------------------------------------- changes

@dataclass
class Change:
    index: int
    time: str
    what: list[str]
    fix: dict[str, Any]  # the line's whole correction after the edit ({} = none)
    fit: Fit | None = None


def match(lines: list[Line], edited: list[Edited]) -> tuple[list[tuple[Line, Edited]], list[Edited]]:
    """Edited lines paired with the episode's lines: by the number in Effect (ASS), else by start time."""
    by_index = {line.index: line for line in lines}
    used: set[int] = set()
    pairs: list[tuple[Line, Edited]] = []
    unmatched: list[Edited] = []
    for item in edited:
        line = by_index.get(item.number) if item.number is not None else None
        if line is not None and abs(line.start - item.start) > 5000:
            line = None  # an SRT number renumbered by the editor
        if line is None or line.index in used:
            near = [ln for ln in lines if ln.index not in used and abs(ln.start - item.start) <= MATCH_MS]
            line = min(near, key=lambda ln: abs(ln.start - item.start)) if near else None
        if line is None:
            unmatched.append(item)
            continue
        used.add(line.index)
        pairs.append((line, item))
    return pairs, unmatched


def changes(folder: Path, edited: list[Edited]) -> tuple[list[Change], list[str]]:
    """What the edited lines change, as the corrections they need (relative to subs.ru.json)."""
    lines = current_lines(folder)
    original = {r["index"]: r for r in load_json(folder / "subs.ru.json")}
    speakers = load_json(folder / "speakers.json")
    path = folder / "corrections.json"
    existing = load_json(path) if path.exists() else {}
    policy = _policy()
    pairs, unmatched = match(lines, edited)
    warnings = [f"{_srt_time(e.start)[:8]} «{e.text[:40]}»: нет такой реплики в серии (новые реплики не добавляются)"
                for e in unmatched]
    out: list[Change] = []
    for line, item in pairs:
        base = original[line.index]
        fix = dict(existing.get(str(line.index), {}))
        what: list[str] = []
        skip = item.skip or not item.text or (line.skip and not item.marks_skip)
        if skip != line.skip:
            what.append("не озвучивать" if skip else "снова озвучивать")
        if skip:
            fix["skip"] = True
        else:
            fix.pop("skip", None)
        text_fit = None
        if item.text and not _same(item.text, line.text):
            what.append(f"текст: «{line.text}» → «{item.text}»")
            text_fit = fit(item.text, line.window_ms, policy)
        if item.text:
            if _same(item.text, _text(base.get("text"))):
                fix.pop("text", None)
            elif not _same(item.text, line.text) or "text" in fix:
                fix["text"] = item.text
        if item.speaker and item.speaker != line.speaker_name:
            card = find_speaker(speakers, item.speaker)
            if card is None:
                warnings.append(f"реплика {line.index}: говорящего «{item.speaker}» нет в серии, не изменено")
            elif card["id"] != line.speaker:
                what.append(f"говорящий: {line.speaker_name} → {speaker_label(card)}")
                if card["id"] == base.get("speaker"):
                    fix.pop("speaker", None)
                else:
                    fix["speaker"] = card["id"]
                fix.pop("gender", None)  # the new speaker's gender, unless set below
        gender = "female" if line.gender == "female" else "male"  # as exported (an unknown one as male)
        if item.gender and item.gender != gender:
            what.append(f"пол: {'женский' if item.gender == 'female' else 'мужской'}")
            card = find_speaker(speakers, str(fix["speaker"])) if "speaker" in fix else None
            implied = card.get("gender") if card else base.get("gender")
            if item.gender == implied:
                fix.pop("gender", None)
            else:
                fix["gender"] = item.gender
        # a style left at the old speaker's gender: the new speaker's own gender wins (popped above)
        if what:
            out.append(Change(line.index, _srt_time(line.start)[:8], what, fix, text_fit))
    return out, warnings


def apply(folder: Path, found: list[Change]) -> dict[str, Any]:
    """Write the changes into corrections.json (a line back to its original loses its correction)."""
    path = folder / "corrections.json"
    corrections = load_json(path) if path.exists() else {}
    for change in found:
        if change.fix:
            corrections[str(change.index)] = change.fix
        else:
            corrections.pop(str(change.index), None)
    save_json(path, dict(sorted(corrections.items(), key=lambda kv: int(kv[0]))))
    return corrections


def set_line(folder: Path, index: int, *, text: str | None = None, speaker: str | None = None,
             gender: str | None = None, skip: bool | None = None) -> list[Change]:
    """One line edited in the UI: the same rules as an imported file."""
    line = next(ln for ln in current_lines(folder) if ln.index == index)
    edited = Edited(line.start, line.end, line.text if text is None else " ".join(text.split()),
                    speaker, gender, line.skip if skip is None else skip, index)
    found, warnings = changes(folder, [edited])
    if warnings:
        raise ValueError("; ".join(warnings))
    apply(folder, found)
    return found


def reset_line(folder: Path, index: int) -> None:
    path = folder / "corrections.json"
    if path.exists():
        corrections = load_json(path)
        corrections.pop(str(index), None)
        save_json(path, corrections)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export / import an episode's Russian lines for editing")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Write the lines as ASS (Subtitle Edit) or SRT")
    export.add_argument("folder", type=Path)
    export.add_argument("--review", action="store_true", help="Only the lines of review.json")
    export.add_argument("--srt", action="store_true", help="SRT ([Speaker] text) instead of ASS")
    export.add_argument("--out", type=Path, default=None, help="Default: <folder>/edit/<name>.ru[.review].ass|srt")
    load = sub.add_parser("import", help="Read an edited ASS / SRT and write the differences to corrections.json")
    load.add_argument("folder", type=Path)
    load.add_argument("file", type=Path)
    load.add_argument("--dry-run", action="store_true", help="Only list the changes")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if args.command == "export":
        suffix = (".review" if args.review else "") + (".srt" if args.srt else ".ass")
        out = args.out or args.folder / "edit" / f"{args.folder.name}.ru{suffix}"
        (export_srt if args.srt else export_ass)(args.folder, out, review=args.review)
        log.info("%s", out)
        return 0
    found, warnings = changes(args.folder, read_edited(args.file))
    for change in found:
        log.info("%d %s: %s%s", change.index, change.time, "; ".join(change.what),
                 f" ({change.fit.text})" if change.fit and change.fit.verdict != "ok" else "")
    for warning in warnings:
        log.warning("! %s", warning)
    if not args.dry_run and found:
        apply(args.folder, found)
        log.info("%d lines -> %s; voice them again: stt.run <video> --engine ...", len(found),
                 args.folder / "corrections.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
