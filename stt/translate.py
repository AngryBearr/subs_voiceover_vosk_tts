"""EN -> RU subtitle translation for voice-over with an LLM (speaker-, gender- and timing-aware).

    subs_env/bin/python -m stt.translate output/stt/ep01 [--model ...] [--cps 14 | --tts-profile p.json]

Reads `subs.en.json` + `speakers.json` (after `stt.speakers`), writes `subs.ru.json`
in the project subtitle schema (ready for utils.text_normalizer -> TTS) and `subs.ru.srt`.

Timing: every cue gets a character budget for its time window (cue start until the next cue,
at most `--extend` seconds past its own end). The budget comes either from a flat
characters-per-second rate or from a linear TTS duration profile
(`seconds = intercept + slope * chars`, e.g. the Vosk calibration of the shortening branch).
Lines still over budget after translation get up to `--shorten-rounds` shortening passes.
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stt.llm import DEFAULT_MODEL, OpenRouterClient
from stt.runner import load_json, save_json

log = logging.getLogger("stt.translate")

MIN_BUDGET = 8  # characters; short interjections always fit

# First-person past tense right after "я": "я устал" (male) / "я устала" (female).
_ADVERBS = (r"(?:не|уже|так|просто|тоже|ведь|даже|бы|еще|ещё|всегда|никогда|сразу|правда|реально|точно|сам|сама|"
            r"и|вот|наконец|только|честно|лично)")
_FIRST_PERSON_PAST = re.compile(rf"\bя(?:\s+{_ADVERBS})*\s+([а-яё]{{2,}}?)(лась|лся|ла|л)\b", re.IGNORECASE)

SYSTEM_PROMPT = """You translate English TV subtitles into Russian for a voice-over dub.

Rules:
- Natural, idiomatic spoken Russian; keep the meaning, tone, humor and register of each line.
- Gender agreement is essential. Use the speaker card: a female speaker says "я устала", "я была
  готова"; a male speaker - "я устал", "я был готов". When a line addresses or describes another
  person, infer who it is from context and the speaker cards and agree with THEIR gender
  ("ты сделала" to a woman). Plural "you" to a group -> "вы".
- Names: transliterate consistently (use "label_ru" when it is a name). Keep show-specific terms
  from the glossary.
- Write numbers (ranges too: "три-четыре года"), units and symbols in words as they should be spoken,
  in the right case ("сто десять килограммов", "с восемнадцатью участниками"); write "ё" where it is said.
- Timing: the dub must fit the original timing. Each line has "max" = maximum number of Russian
  characters that fit its time window. Stay within it by rephrasing concisely: drop filler words,
  repetitions and redundant phrases first, never essential meaning. Very short interjections may
  stay short ("Да!", "Что?").
- One output line per input line, same "i". Do not merge or split lines, do not add notes.
- Lines marked "ctx" are already translated context - do not output them.

Answer with JSON only: {"items": [{"i": <index>, "ru": "<Russian text>"}, ...]}"""

SHORTEN_PROMPT = """You shorten Russian voice-over lines so they fit their time windows.

Each line has the English original ("en"), the current Russian translation ("ru"), the speaker
card id ("s") and "max" = maximum number of characters. Rewrite "ru" to at most "max" characters:
keep the meaning, tone and gender agreement of the current translation; remove fillers,
repetitions and secondary details; prefer shorter synonyms and simpler syntax. If the essential
meaning cannot fit, return the shortest natural version you can. Lines marked "ctx" are context.

Answer with JSON only: {"items": [{"i": <index>, "ru": "<shorter Russian text>"}, ...]}"""


GENDER_FIX_PROMPT = """You fix grammatical gender in Russian voice-over lines.
Each line is spoken by a speaker of the given gender ("g"), but some first-person forms
(past tense verbs, short adjectives: "я устал/устала", "я готов/готова") do not agree with it.
Rewrite the line so that every form referring to the speaker agrees with "g". Change nothing else:
same meaning, words, punctuation and length. Forms about other people stay as they are.

Answer with JSON only: {"items": [{"i": <index>, "ru": "<corrected Russian text>"}, ...]}"""


def gender_mismatches(text: str, gender: str) -> list[str]:
    """First-person past-tense forms after "я" whose ending disagrees with the speaker gender."""
    if gender not in ("male", "female"):
        return []
    wrong = []
    for match in _FIRST_PERSON_PAST.finditer(text):
        feminine = match.group(2).lower() in ("ла", "лась")
        if feminine != (gender == "female"):
            wrong.append(match.group(0))
    return wrong


@dataclass(frozen=True)
class DurationModel:
    """Spoken duration of a Russian line: `intercept + slope * chars` seconds."""

    intercept: float = 0.0
    slope: float = 1.0 / 14.0

    @classmethod
    def from_cps(cls, cps: float) -> DurationModel:
        return cls(0.0, 1.0 / cps)

    @classmethod
    def from_profile(cls, path: Path) -> DurationModel:
        """JSON with `intercept` and `slope` (seconds, seconds per character)."""
        data = load_json(path)
        return cls(float(data["intercept"]), float(data["slope"]))

    def max_chars(self, window_s: float) -> int:
        return max(MIN_BUDGET, math.floor((window_s - self.intercept) / self.slope))


def cue_windows(records: list[dict[str, Any]], extend_s: float) -> list[float]:
    """Seconds available for each cue: until the next cue starts, at most `extend_s` past its end."""
    starts = sorted(r["start"] for r in records)
    windows = []
    for r in records:
        pos = bisect.bisect_right(starts, r["start"])
        limit = r["end"] + extend_s * 1000
        if pos < len(starts):  # never cut the cue itself, only the extension
            limit = min(limit, max(starts[pos], r["end"]))
        windows.append((limit - r["start"]) / 1000)
    return windows


def speaker_cards(speakers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": s["id"],
            "name": s.get("name"),
            "label_ru": s.get("label_ru"),
            "role": s.get("role"),
            "gender": s.get("gender", "unknown"),
        }
        for s in speakers
        if s.get("cues")
    ]


def _prompt(cards: list[dict[str, Any]], glossary: str, lines: list[dict[str, Any]]) -> str:
    parts = ["Speaker cards:", json.dumps(cards, ensure_ascii=False)]
    if glossary:
        parts += ["Glossary:", glossary]
    parts += ["Lines:", "\n".join(json.dumps(x, ensure_ascii=False) for x in lines)]
    return "\n".join(parts)


def _ask(client: OpenRouterClient, system: str, prompt: str, wanted: set[int]) -> dict[int, str]:
    answer = client.chat_json(system, prompt)
    items = answer.get("items", []) if isinstance(answer, dict) else answer
    return {
        item["i"]: str(item["ru"]).strip()
        for item in items
        if isinstance(item, dict) and item.get("i") in wanted and str(item.get("ru", "")).strip()
    }


def translate_records(
    records: list[dict[str, Any]],
    speakers: list[dict[str, Any]],
    client: OpenRouterClient,
    *,
    budgets: dict[int, int],
    batch_size: int,
    context_size: int,
    glossary: str = "",
) -> dict[int, str]:
    """Translate in batches with preceding translated lines as context; retries missing lines."""
    cards = speaker_cards(speakers)
    translated: dict[int, str] = {}
    done: list[dict[str, Any]] = []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        pending = batch
        for _attempt in range(3):
            lines = [{"ctx": True, "s": c["speaker"], "en": " ".join(c["text"]), "ru": c["ru"]}
                     for c in done[-context_size:]]
            lines += [
                {"i": r["index"], "s": r.get("speaker"), "en": " ".join(r["text"]), "max": budgets[r["index"]]}
                for r in pending
            ]
            translated.update(_ask(client, SYSTEM_PROMPT, _prompt(cards, glossary, lines),
                                   {r["index"] for r in pending}))
            pending = [r for r in pending if r["index"] not in translated]
            if not pending:
                break
            log.warning("Batch at %d: %d lines missing, retrying", i, len(pending))
        if pending:
            raise RuntimeError(f"Lines not translated: {[r['index'] for r in pending]}")
        done.extend({**r, "ru": translated[r["index"]]} for r in batch)
        log.info("Translated %d/%d lines of a part (cost so far $%.4f)", len(translated), len(records),
                 client.usage.cost)
    return translated


def split_parts(records: list[dict[str, Any]], parts: int, min_size: int = 120) -> list[list[dict[str, Any]]]:
    """Split chronological records into up to `parts` contiguous runs at the longest pauses."""
    parts = max(1, min(parts, len(records) // min_size))
    if parts == 1:
        return [records]
    gaps = sorted(
        range(1, len(records)), key=lambda i: records[i]["start"] - records[i - 1]["end"], reverse=True
    )
    target = len(records) / parts
    cuts: list[int] = []
    for i in gaps:  # longest pauses first, keeping parts reasonably balanced
        if all(abs(i - c) >= target * 0.5 for c in cuts) and target * 0.5 <= i <= len(records) - target * 0.5:
            cuts.append(i)
        if len(cuts) == parts - 1:
            break
    bounds = [0, *sorted(cuts), len(records)]
    return [records[a:b] for a, b in zip(bounds, bounds[1:])]


def over_budget(translated: dict[int, str], budgets: dict[int, int], tolerance: float) -> list[int]:
    return [i for i, text in translated.items() if len(text) > budgets[i] * tolerance]


def shorten_pass(
    records: list[dict[str, Any]],
    speakers: list[dict[str, Any]],
    translated: dict[int, str],
    client: OpenRouterClient,
    *,
    budgets: dict[int, int],
    tolerance: float,
    batch_size: int = 40,
) -> int:
    """Ask for shorter versions of over-budget lines; keeps a new version only if it is shorter."""
    cards = speaker_cards(speakers)
    by_index = {r["index"]: (pos, r) for pos, r in enumerate(records)}
    targets = over_budget(translated, budgets, tolerance)
    improved = 0
    for i in range(0, len(targets), batch_size):
        lines: list[dict[str, Any]] = []
        for idx in targets[i : i + batch_size]:
            pos, rec = by_index[idx]
            if pos > 0:  # previous line as context for pronouns and addressees
                prev = records[pos - 1]
                lines.append({"ctx": True, "s": prev.get("speaker"), "ru": translated[prev["index"]]})
            lines.append({"i": idx, "s": rec.get("speaker"), "en": " ".join(rec["text"]),
                          "ru": translated[idx], "max": budgets[idx]})
        answers = _ask(client, SHORTEN_PROMPT, _prompt(cards, "", lines), set(targets[i : i + batch_size]))
        for idx, text in answers.items():
            if len(text) < len(translated[idx]):
                translated[idx] = text
                improved += 1
    return improved


def gender_fix_pass(
    records: list[dict[str, Any]], translated: dict[int, str], client: OpenRouterClient
) -> tuple[int, int]:
    """Re-ask the LLM for lines whose first-person forms disagree with the speaker gender.

    Returns (flagged, fixed); a correction is kept only if the mismatch is gone.
    """
    flagged = [r for r in records if gender_mismatches(translated[r["index"]], r.get("gender", "unknown"))]
    if not flagged:
        return 0, 0
    lines = [{"i": r["index"], "g": r["gender"], "en": " ".join(r["text"]), "ru": translated[r["index"]],
              "wrong": gender_mismatches(translated[r["index"]], r["gender"])} for r in flagged]
    answers = _ask(client, GENDER_FIX_PROMPT, "Lines:\n" + "\n".join(json.dumps(x, ensure_ascii=False) for x in lines),
                   {r["index"] for r in flagged})
    fixed = 0
    by_index = {r["index"]: r for r in flagged}
    for idx, text in answers.items():
        if not gender_mismatches(text, by_index[idx]["gender"]):
            translated[idx] = text
            fixed += 1
    return len(flagged), fixed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Translate speaker-attributed subtitles EN -> RU for voice-over")
    parser.add_argument("folder", help="Output folder of stt.transcribe (+ stt.speakers)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id")
    parser.add_argument("--cps", type=float, default=14.0, help="Russian characters per second (flat rate)")
    parser.add_argument("--tts-profile", default=None,
                        help="JSON {intercept, slope} TTS duration profile; overrides --cps")
    parser.add_argument("--extend", type=float, default=0.6,
                        help="Seconds a line may run past its cue end into a following pause")
    parser.add_argument("--shorten-rounds", type=int, default=2, help="Shortening passes for over-budget lines")
    parser.add_argument("--tolerance", type=float, default=1.15, help="Over-budget threshold (x budget)")
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--parallel", type=int, default=4,
                        help="Translate this many contiguous parts of the episode concurrently (split at pauses)")
    parser.add_argument("--context", type=int, default=12, help="Previously translated lines sent as context")
    parser.add_argument("--glossary", default=None, help="Text file with show-specific terms/names")
    parser.add_argument("--suffix", default="", help="Output name suffix, e.g. a model tag for comparisons")
    parser.add_argument(
        "--reasoning",
        default="medium",
        choices=["default", "off", "minimal", "low", "medium", "high"],
        help="Reasoning effort; 'off' disables it (some providers ignore effort and think until the token limit)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    folder = Path(args.folder)
    records, speakers = load_json(folder / "subs.en.json"), load_json(folder / "speakers.json")
    glossary = Path(args.glossary).read_text(encoding="utf-8") if args.glossary else ""
    if args.tts_profile:
        duration = DurationModel.from_profile(Path(args.tts_profile))
    else:
        duration = DurationModel.from_cps(args.cps)
    windows = cue_windows(records, args.extend)
    budgets = {r["index"]: duration.max_chars(w) for r, w in zip(records, windows)}

    client = OpenRouterClient(args.model, reasoning=None if args.reasoning == "default" else args.reasoning)
    t0 = time.perf_counter()
    translated: dict[int, str] = {}
    chunks = split_parts(records, args.parallel)
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:  # independent runs, each with its own context
        futures = [
            pool.submit(translate_records, part, speakers, client, budgets=budgets, batch_size=args.batch_size,
                        context_size=args.context, glossary=glossary)
            for part in chunks
        ]
        for future in futures:
            translated.update(future.result())
    over_initial = len(over_budget(translated, budgets, args.tolerance))
    for round_no in range(args.shorten_rounds):
        if not over_budget(translated, budgets, args.tolerance):
            break
        improved = shorten_pass(records, speakers, translated, client, budgets=budgets, tolerance=args.tolerance)
        log.info("Shortening round %d: %d lines shortened, %d still over budget", round_no + 1, improved,
                 len(over_budget(translated, budgets, args.tolerance)))
    over_final = over_budget(translated, budgets, args.tolerance)
    gender_flagged, gender_fixed = gender_fix_pass(records, translated, client)
    if gender_flagged:
        log.info("Gender check: %d lines with first-person forms of the wrong gender, %d fixed",
                 gender_flagged, gender_fixed)

    out_records = [
        {**r, "text": [translated[r["index"]]], "text_en": r["text"], "budget_chars": budgets[r["index"]],
         "window_ms": round(w * 1000)}
        for r, w in zip(records, windows)
    ]
    name = f"subs.ru{args.suffix}"
    save_json(folder / f"{name}.json", out_records)

    from stt.transcribe import write_srt

    write_srt(folder / f"{name}.srt", out_records, with_speaker=False)
    usage = client.usage.to_dict()
    save_json(
        folder / f"{name}.usage.json",
        {
            "model": args.model,
            "reasoning": args.reasoning,
            "duration_model": {"intercept": duration.intercept, "slope": duration.slope},
            "extend_s": args.extend,
            "tolerance": args.tolerance,
            "usage": usage,
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "over_budget_initial": over_initial,
            "over_budget": len(over_final),
            "gender_flagged": gender_flagged,
            "gender_fixed": gender_fixed,
        },
    )
    log.info("Wrote %s (%d lines; over budget x%.2f: %d after translation, %d after shortening), usage %s",
             folder / f"{name}.json", len(out_records), args.tolerance, over_initial, len(over_final), usage)


if __name__ == "__main__":
    main()
