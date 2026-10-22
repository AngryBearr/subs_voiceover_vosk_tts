"""Speaker naming and gender reconciliation with an LLM over the diarized transcript.

    subs_env/bin/python -m stt.speakers output/stt/ep01 [--model anthropic/claude-sonnet-5.5]

Reads `speakers.json` + `subs.en.json` from the transcription folder, asks the LLM for
real names (only when evident), a short Russian label, role and gender (voice estimate
combined with textual evidence), then applies manual corrections from
`speakers.override.json` ({"S03": {"name": "Jeff", "gender": "male"}, ...}) and writes
the results back into `speakers.json` and `subs.en.json` (`gender`, `speaker_name`).
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

log = logging.getLogger("stt.speakers")

SYSTEM_PROMPT = """You analyze a diarized English transcript of a TV show / film to identify its speakers.
Speaker IDs (S01, S02, ...) come from automatic voice clustering: usually one ID = one person,
but clustering can split a person into two IDs or merge similar voices.

For EVERY speaker ID return:
- "name": the person's real name ONLY if the transcript makes it evident (self-introduction,
  "my name is", being addressed by name and answering, others clearly referring to them) or the
  speaker's "on_screen" evidence names them: a name card/title shown while they talked or a caption
  label of their line (OCR, may contain misreadings, props or place names). Else null.
  Never guess celebrities from style alone.
- "name_evidence": short quote/reason for the name, or null.
- "role": short English role (host, narrator, contestant, interviewer, interviewee, reporter, ...).
- "label_ru": short unique Russian label for display. If the name is known: the name in Russian
  transliteration (e.g. "Джефф"). Otherwise a vivid descriptive nickname from role/traits/content,
  1-3 words (e.g. "Ведущий", "Рассказчик", "Пожилой рыбак", "Девушка-юрист").
- "gender": "male", "female" or "unknown" - the person's gender as needed for Russian grammar.
  Combine the acoustic voice estimate (given with confidence) with textual evidence: pronouns others
  use about them (he/she/him/her), gendered self-references, gendered words (wife, husband, guys),
  the name. Strong textual evidence overrides a low-confidence or conflicting voice estimate.
- "gender_evidence": short explanation, e.g. "voice 0.97 + 'she' in S04 line 12".
- "same_as": ID of another speaker that is very likely the same person, else null.
- "confidence": 0..1 overall confidence in this card.

Answer with JSON only: {"speakers": [{"id": "S01", ...}, ...]}"""


def _clock(ms: int) -> str:
    seconds = ms // 1000
    return f"{seconds // 3600:d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def build_prompt(
    speakers: list[dict[str, Any]], records: list[dict[str, Any]], max_lines: int = 4000, glossary: str = ""
) -> str:
    table = [
        {
            "id": s["id"],
            "voice_gender": s.get("gender"),
            "voice_gender_conf": s.get("gender_conf"),
            "voice_age": s.get("age"),
            "child_prob": s.get("p_child"),
            "speech_seconds": s.get("speech_s"),
            "on_screen": s.get("on_screen") or [],
            "lines": s.get("cues"),
        }
        for s in speakers
    ]
    lines = [f"[{_clock(r['start'])}] {r['speaker'] or '??'}: {' '.join(r['text'])}" for r in records[:max_lines]]
    head = f"Show glossary (use these spellings and Russian renderings):\n{glossary}\n\n" if glossary else ""
    return (
        head
        + "Speakers (acoustic estimates):\n"
        + json.dumps(table, ensure_ascii=False, indent=1)
        + "\n\nTranscript:\n"
        + "\n".join(lines)
    )


def apply_cards(speakers: list[dict[str, Any]], cards: list[dict[str, Any]]) -> None:
    """Merge LLM cards into speaker dicts, tracking where the final gender came from."""
    by_id = {c.get("id"): c for c in cards if isinstance(c, dict)}
    for speaker in speakers:
        card = by_id.get(speaker["id"])
        if not card:
            continue
        voice_gender = speaker.get("voice_gender", speaker.get("gender"))
        speaker["voice_gender"] = voice_gender
        llm_gender = card.get("gender") if card.get("gender") in ("male", "female") else "unknown"
        speaker.update(
            {
                "name": card.get("name") or None,
                "name_evidence": card.get("name_evidence"),
                "role": card.get("role"),
                "label_ru": card.get("label_ru"),
                "same_as": card.get("same_as"),
                "llm_confidence": card.get("confidence"),
                "gender_evidence": card.get("gender_evidence"),
                "gender": llm_gender,
                "gender_source": (
                    "unknown" if llm_gender == "unknown"
                    else "voice+text" if llm_gender == voice_gender
                    else "text"
                ),
            }
        )


def apply_overrides(speakers: list[dict[str, Any]], overrides: dict[str, dict[str, Any]]) -> None:
    for speaker in speakers:
        override = overrides.get(speaker["id"])
        if override:
            speaker.update(override)
            if "gender" in override:
                speaker["gender_source"] = "manual"


def update_records(records: list[dict[str, Any]], speakers: list[dict[str, Any]]) -> None:
    by_id = {s["id"]: s for s in speakers}
    for record in records:
        speaker = by_id.get(record.get("speaker"))
        if speaker:
            record["gender"] = speaker.get("gender", "unknown")
            record["speaker_name"] = speaker.get("label_ru") or speaker.get("name")


def name_speakers(
    folder: Path, model: str, *, use_llm: bool = True, reasoning: str | None = None, glossary: str = ""
) -> list[dict[str, Any]]:
    speakers_path, subs_path = folder / "speakers.json", folder / "subs.en.json"
    speakers, records = load_json(speakers_path), load_json(subs_path)
    if use_llm:
        client = OpenRouterClient(model, reasoning=reasoning)
        prompt = build_prompt([s for s in speakers if s.get("cues")], records, glossary=glossary)
        answer = client.chat_json(SYSTEM_PROMPT, prompt)
        cards = answer.get("speakers", []) if isinstance(answer, dict) else answer
        save_json(folder / "speakers.llm.json", {"model": model, "usage": client.usage.to_dict(), "cards": cards})
        apply_cards(speakers, cards)
        log.info("LLM naming: %s", client.usage.to_dict())
    override_path = folder / "speakers.override.json"
    if override_path.exists():
        apply_overrides(speakers, load_json(override_path))
    update_records(records, speakers)
    save_json(speakers_path, speakers)
    save_json(subs_path, records)
    for s in speakers:
        if s.get("cues"):
            log.info(
                "%s %-22s %-7s (%s) %s",
                s["id"], s.get("label_ru") or "-", s.get("gender"), s.get("gender_source"), s.get("name") or "",
            )
    return speakers


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Name speakers and reconcile gender with an LLM")
    parser.add_argument("folder", help="Output folder of stt.transcribe")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id")
    parser.add_argument("--no-llm", action="store_true", help="Only apply speakers.override.json")
    parser.add_argument("--glossary", default=None, help="glossary.txt from stt.glossary (canonical names)")
    parser.add_argument(
        "--reasoning",
        default="medium",
        choices=["default", "off", "minimal", "low", "medium", "high"],
        help="Reasoning effort; 'off' disables it (some providers ignore effort and think until the token limit)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    reasoning = None if args.reasoning == "default" else args.reasoning
    glossary = Path(args.glossary).read_text(encoding="utf-8") if args.glossary else ""
    name_speakers(Path(args.folder), args.model, use_llm=not args.no_llm, reasoning=reasoning, glossary=glossary)


if __name__ == "__main__":
    main()
