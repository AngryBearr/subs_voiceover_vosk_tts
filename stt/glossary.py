"""Show glossary from a transcript: canonical names/terms with Russian renderings (LLM).

    subs_env/bin/python -m stt.glossary output/stt/ep01 [--season output/stt/survivor_s10.glossary.json]

Writes `glossary.json`, `glossary.txt` (EN -> RU lines for `stt.translate --glossary`) and
`hotwords.txt` (English names/terms for ASR biasing, `stt.bench asr --hotwords-file`).
With `--season`, the season file is extended and used as the starting point, so names and
renderings stay consistent across episodes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from stt.llm import DEFAULT_MODEL, OpenRouterClient
from stt.runner import load_json, save_json

log = logging.getLogger("stt.glossary")

SYSTEM_PROMPT = """You build a glossary for dubbing an English TV show into Russian.
From the transcript (automatic speech recognition, so names may be misspelled or spelled
inconsistently) extract:
- people: every named person (cast, host, family members mentioned);
- places, teams/tribes, show-specific terms and catchphrases (e.g. "Tribal Council",
  "immunity idol", the show title).
For each entry give:
- "en": canonical English spelling (choose the most likely correct spelling when the transcript varies);
- "variants": misspellings seen in the transcript (may be empty);
- "ru": how it should sound in the Russian dub (names transliterated, terms translated idiomatically,
  a show title in its established Russian form or a natural translation);
- "type": person | place | team | term | title;
- "note": optional short note (gender for people if evident, e.g. "female contestant").
Keep entries already present in the existing glossary and their "ru" unless clearly wrong.
Answer with JSON only: {"entries": [{"en": ..., "variants": [...], "ru": ..., "type": ..., "note": ...}]}"""


def merge(existing: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union by canonical English form; existing renderings win, variants are combined."""
    merged = {e["en"].casefold(): dict(e) for e in existing}
    for entry in new:
        if not isinstance(entry, dict) or not entry.get("en"):
            continue
        key = entry["en"].casefold()
        if key in merged:
            variants = set(merged[key].get("variants") or []) | set(entry.get("variants") or [])
            merged[key]["variants"] = sorted(variants)
        else:
            merged[key] = entry
    return sorted(merged.values(), key=lambda e: (e.get("type", ""), e["en"]))


def write_outputs(folder: Path, entries: list[dict[str, Any]]) -> None:
    save_json(folder / "glossary.json", entries)
    lines = [f"{e['en']} -> {e.get('ru', '')}" + (f" ({e['note']})" if e.get("note") else "") for e in entries]
    (folder / "glossary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Short list for ASR biasing: long prompts dilute it (Whisper keeps ~224 prompt tokens).
    names = [e["en"] for e in entries if e.get("type") in ("person", "team", "title")]
    (folder / "hotwords.txt").write_text(", ".join(names) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract a show glossary (names, terms, Russian renderings)")
    parser.add_argument("folder", help="Output folder of stt.transcribe")
    parser.add_argument("--season", default=None, help="Season glossary JSON to start from and extend")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id")
    parser.add_argument(
        "--reasoning", default="medium", choices=["default", "off", "minimal", "low", "medium", "high"]
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    folder = Path(args.folder)
    records = load_json(folder / "subs.en.json")
    season_path = Path(args.season) if args.season else None
    existing = load_json(season_path) if season_path and season_path.exists() else []
    transcript = "\n".join(f"{r.get('speaker') or '??'}: {' '.join(r['text'])}" for r in records)
    client = OpenRouterClient(args.model, reasoning=None if args.reasoning == "default" else args.reasoning)
    prompt = "Existing glossary:\n" + json.dumps(existing, ensure_ascii=False) + "\n\nTranscript:\n" + transcript
    answer = client.chat_json(SYSTEM_PROMPT, prompt)
    entries = merge(existing, answer.get("entries", []) if isinstance(answer, dict) else answer)
    write_outputs(folder, entries)
    if season_path:
        save_json(season_path, entries)
    log.info("%d glossary entries (%d people), usage %s", len(entries),
             sum(e.get("type") == "person" for e in entries), client.usage.to_dict())


if __name__ == "__main__":
    main()
