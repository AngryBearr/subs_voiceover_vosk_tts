"""Speaker naming and gender reconciliation with an LLM over the diarized transcript.

    subs_env/bin/python -m stt.speakers output/stt/ep01 [--model anthropic/claude-sonnet-5.5]

Reads `speakers.json` + `subs.en.json` from the transcription folder, asks the LLM for
real names (only when evident), a short Russian label, role and gender (voice estimate
combined with textual evidence), then applies manual corrections from
`speakers.override.json` ({"S03": {"name": "Jeff", "gender": "male"}, ...}) and writes
the results back into `speakers.json` and `subs.en.json` (`gender`, `speaker_name`).

With ``--voice-bank`` the season voice bank (``stt.voice_bank``) adds voice matches with the
other episodes as evidence, names clusters the LLM left unnamed when the match is strong, and
takes this episode's named clusters as new prints.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from stt import progress, voice_bank
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
  label of their line (OCR, may contain misreadings, props or place names), or a strong
  "voice_bank" match does (see below). Else null. Never guess celebrities from style alone.
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

"voice_bank" (when present) lists people named in earlier episodes of the same season whose voice
matches this speaker, with cosine similarity 0..1 (0.75 and more: strong, 0.6-0.75: weak). Use
it as evidence for the name, like on_screen. It can repeat an earlier naming mistake and relatives
can sound alike (twins, parent and child): clear transcript evidence for another name wins.

Answer with JSON only: {"speakers": [{"id": "S01", ...}, ...]}"""


def _clock(ms: int) -> str:
    seconds = ms // 1000
    return f"{seconds // 3600:d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def build_prompt(
    speakers: list[dict[str, Any]], records: list[dict[str, Any]], max_lines: int = 4000, glossary: str = "",
    cast: str = "",
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
            **({"voice_bank": [f"{m['name']} ({m['score']:.2f})" for m in s["voice_bank"]]}
               if s.get("voice_bank") else {}),
            "lines": s.get("cues"),
        }
        for s in speakers
    ]
    lines = [f"[{_clock(r['start'])}] {r['speaker'] or '??'}: {' '.join(r['text'])}" for r in records[:max_lines]]
    head = f"Show glossary (use these spellings and Russian renderings):\n{glossary}\n\n" if glossary else ""
    if cast:
        head = f"Cast of the season (from the web: names, gender, relations):\n{cast}\n\n" + head
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
                "name_source": "llm" if card.get("name") else None,
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


def russian_names(folder: Path) -> dict[str, str]:
    """Lower-case English name or variant -> Russian spelling, from the episode's glossary (``glossary.json``, people)."""
    path = folder / "glossary.json"
    try:
        entries = load_json(path) if path.exists() else []
    except (OSError, ValueError):
        return {}
    out: dict[str, str] = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or entry.get("type") != "person" or not entry.get("ru") or not entry.get("en"):
            continue
        for spelling in [entry["en"], *(entry.get("variants") or [])]:
            out.setdefault(str(spelling).lower(), str(entry["ru"]))
    return out


def apply_subtitle_names(speakers: list[dict[str, Any]], cast: dict[str, Any] | None = None,
                         ru_names: dict[str, str] | None = None) -> None:
    """A speaker whose lines the subtitles mostly name ("SANDRA:", ``subtitle_person`` from stt.transcribe) gets that name
    when the LLM left it open; with the season ``cast`` a first name, nickname or initials ("B.B") become the cast
    member's name. No LLM call: the card's own name, a voice-bank name or an override win. The Russian label becomes
    the name's glossary spelling (``ru_names``, see ``russian_names``); without one it is dropped, so the English name
    shows instead of the LLM's description of an unnamed voice (S01E01: Sean Kenniff shown as «Мастер костра»)."""
    from stt.cast import member_for_label

    for speaker in speakers:
        person = speaker.get("subtitle_person")
        if person and not speaker.get("name"):
            member = member_for_label(person, cast) if cast else None
            name = member["name"] if member else person.title()
            known = ru_names or {}
            label_ru = known.get(name.lower()) or known.get(person.lower()) or known.get(name.split()[0].lower())
            speaker.update({"name": name, "name_source": "subtitles", "label_ru": label_ru,
                            "name_evidence": f"the subtitles name this voice: {person}"})


HOST_ROLE = re.compile(r"\b(host|narrator)\b", re.I)
NOT_HOST_ROLE = re.compile(
    r"contestant|guest|reunion|co-?host|former|merged|mixed|multiple|several|background|occasional|voices", re.I)
MIXED_ROLE = re.compile(r"merged|mixed|multiple|several|different|background|crowd|chorus|voices", re.I)


def _is_mixed(speaker: dict[str, Any]) -> bool:
    """A card whose role the LLM wrote as a group of voices, or that a ``{"mixed": true}`` override marks: no one person."""
    return bool(speaker.get("mixed")) or bool(MIXED_ROLE.search(str(speaker.get("role") or "")))


def apply_cast_host(speakers: list[dict[str, Any]], cast: dict[str, Any] | None,
                    protected: frozenset[str] = frozenset()) -> None:
    """A card whose role is the show's host or narrator ("host", "host and narrator") and that has no name gets the
    season host from the cast (name and gender). A name already given (LLM, subtitles, a person) stays."""
    from stt.cast import host_member

    host = host_member(cast) if cast else None
    if host is None or not host.get("name"):
        return
    for speaker in speakers:
        # the role's first clause says who it is; a note after it ("Host and narrator; cluster also includes contestant
        # dialogue", S01E01 without subtitle names) left the host unnamed: 168 lines in a generic voice
        role = re.split(r"[;(]", str(speaker.get("role") or ""))[0]
        if (speaker.get("name") or not speaker.get("cues") or speaker["id"] in protected
                or not HOST_ROLE.search(role) or NOT_HOST_ROLE.search(role)):
            continue
        speaker.update({"name": host["name"], "name_source": "cast_host",
                        "name_evidence": f"role '{role}': the season host in the cast"})
        if host.get("gender") in ("male", "female"):
            speaker.update({"gender": host["gender"], "gender_source": "cast"})
        log.info("cast host: %s (%s) is %s", speaker["id"], role, host["name"])


def _same_as_target(speaker: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """The first named card along the ``same_as`` pointers from ``speaker`` (None: a cycle, a dead end or an unknown id)."""
    seen = {speaker["id"]}
    target_id = speaker.get("same_as")
    while target_id and target_id not in seen:
        seen.add(target_id)
        target = by_id.get(target_id)
        if target is None or target.get("name"):
            return target
        target_id = target.get("same_as")
    return None


def apply_same_as(speakers: list[dict[str, Any]], protected: frozenset[str] = frozenset()) -> None:
    """A nameless card the LLM marked ``same_as`` another card takes the name and gender of that card's person (the
    first named card along a chain). A cycle, an unknown id, a group of voices (mixed card) on either side, a name already
    on the card and a gender conflict leave the card as it is."""
    from stt.voice_bank import sure_name

    by_id = {s["id"]: s for s in speakers}
    for speaker in speakers:
        if not speaker.get("same_as") or speaker.get("name") or speaker["id"] in protected or _is_mixed(speaker):
            continue
        target = _same_as_target(speaker, by_id)
        if target is None or _is_mixed(target):
            log.info("same_as: %s -> %s ignored (cycle, unknown id or no named person)", speaker["id"], speaker["same_as"])
            continue
        mine, theirs = speaker.get("gender"), target.get("gender")
        if mine in ("male", "female") and theirs in ("male", "female") and mine != theirs:
            log.info("same_as: %s -> %s ignored (gender %s vs %s)", speaker["id"], target["id"], mine, theirs)
            continue
        speaker.update({"name": target["name"], "name_source": "same_as",
                        "name_evidence": f"the LLM marked this voice as the same person as {target['id']}"})
        if theirs in ("male", "female"):
            speaker.update({"gender": theirs, "gender_source": target.get("gender_source") or "text"})
        if sure_name(target):  # the name is as sure as the card it was copied from
            speaker["llm_confidence"] = max(speaker.get("llm_confidence") or 0, target.get("llm_confidence") or 0, 0.9)
        log.info("same_as: %s merged into %s (%s)", speaker["id"], target["id"], target["name"])


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
    folder: Path, model: str, *, use_llm: bool = True, reasoning: str | None = None, glossary: str = "",
    bank_path: Path | None = None, cast_path: Path | None = None, saved_llm: bool = False,
    identities: bool = True,
) -> list[dict[str, Any]]:
    speakers_path, subs_path = folder / "speakers.json", folder / "subs.en.json"
    speakers, records = load_json(speakers_path), load_json(subs_path)
    bank = voice_bank.VoiceBank.load(bank_path) if bank_path else None
    if bank is not None:
        voice_bank.attach_matches(speakers, bank, folder.name)
    cast = load_json(cast_path) if cast_path and cast_path.exists() else None
    saved_path = folder / "speakers.llm.json"
    if saved_llm and saved_path.exists():  # apply the earlier answer again, no request
        apply_cards(speakers, load_json(saved_path).get("cards", []))
        log.info("LLM naming: the saved answer %s", saved_path.name)
    elif use_llm:
        client = OpenRouterClient(model, reasoning=reasoning)
        from stt.cast import cast_text

        prompt = build_prompt([s for s in speakers if s.get("cues")], records, glossary=glossary,
                              cast=cast_text(cast) if cast else "")
        progress.tick(0, 1, "requests")
        answer = client.chat_json(SYSTEM_PROMPT, prompt)
        progress.tick(1, 1, "requests")
        cards = answer.get("speakers", []) if isinstance(answer, dict) else answer
        save_json(saved_path, {"model": model, "usage": client.usage.to_dict(), "cards": cards})
        apply_cards(speakers, cards)
        log.info("LLM naming: %s", client.usage.to_dict())
    override_path = folder / "speakers.override.json"
    overrides = load_json(override_path) if override_path.exists() else {}
    protected = frozenset(i for i, o in overrides.items() if "name" in o)  # a person's name is never replaced
    apply_subtitle_names(speakers, cast, russian_names(folder))
    if bank is not None:
        for note in voice_bank.resolve(speakers):
            log.info("voice bank: %s", note)
    apply_cast_host(speakers, cast, protected)
    apply_overrides(speakers, overrides)  # before the merge: a card a person named is the one others point to
    apply_same_as(speakers, protected)
    if cast:  # cast spellings; names not in the cast and gender conflicts listed
        from stt.cast import check_speakers

        for note in check_speakers(speakers, cast):
            log.info("cast: %s", note)
    apply_overrides(speakers, overrides)
    if bank is not None and identities:  # voices without a name: season identities (V01, ...)
        for note in voice_bank.resolve_identities(speakers, bank, folder.name, records):
            log.info("voice bank: %s", note)
    if bank is not None:
        added = bank.add_episode(folder.name, speakers, records)
        bank.save()
        log.info("voice bank: %d prints from this episode, %d in %s", added, len(bank.prints), bank.path)
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
    parser.add_argument("--voice-bank", type=Path, default=None, help="Season voice bank (stt.voice_bank)")
    parser.add_argument("--cast", type=Path, default=None, help="Season cast from the web (stt.cast)")
    parser.add_argument("--no-identities", action="store_true",
                        help="Do not give unnamed voices season identities (V01, ...) in the voice bank")
    parser.add_argument("--saved-llm", action="store_true",
                        help="Apply the saved speakers.llm.json instead of asking the LLM again")
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
    name_speakers(Path(args.folder), args.model, use_llm=not args.no_llm, reasoning=reasoning, glossary=glossary,
                  bank_path=args.voice_bank, cast_path=args.cast, saved_llm=args.saved_llm,
                  identities=not args.no_identities)


if __name__ == "__main__":
    main()
