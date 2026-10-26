"""The cast of a season from the web: names, spelling, gender and relations of everyone who speaks.

A cheap LLM with web search (OpenRouter ":online", another vendor's model if it fails) reads the show and season off the folder and
file names, once per season (``<out-root>/<folder>.cast.json``). The glossary takes the cast's
spellings, the speaker naming gets the cast as evidence, and after naming every speaker is
checked against it (``check_speakers``):

- a name that is another spelling of a cast member's name becomes the cast's spelling ("John
  Misch" -> "Jon Misch"); nicknames stay as they are;
- a name not in the cast is listed (likely an ASR mistake or a one-off guest);
- a gender that contradicts the cast is listed, never changed: the web can be wrong too (S10:
  Stevie Khouw came back "female" from luna, S29 Reed Kelly "female" from Haiku; the voice and the
  transcript said male, correctly).

S29 (San Juan del Sur): 18 contestants + host with their relations (twins, mother and daughter,
couples) in 14-20 s for $0.01-0.02; Australian Survivor S10: 25 people, $0.01-0.03. On the S10
SDH speaker labels (E01, E12, E16) the names were as right with the cast as without it (gpt-6.1-sol
and gpt-6-luna, three runs each); on S29E01-E04 the names were already right.

    subs_env/bin/python -m stt.cast output/stt/<episode> --cast output/stt/<folder>.cast.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from stt.llm import OpenRouterClient
from stt.runner import load_json, save_json

log = logging.getLogger("stt.cast")

# tried in order: another vendor answers when one refuses or fails (Claude Haiku: S29 as complete)
DEFAULT_MODELS = ["openrouter:openai/gpt-6-luna:online", "openrouter:anthropic/claude-haiku-5.5:online"]
MIN_PEOPLE = 3  # fewer is a refusal or a wrong show, not a cast
SYSTEM_PROMPT = """You compile the cast of one season of a TV show for a dubbing pipeline. Use web search.
The user gives the video folder and file names (they name the show and season) and names heard
in the episodes (automatic speech recognition: may be misspelled).
Return every person who speaks in the season: hosts, narrators, contestants (all of them, also
those voted out early) and regular guests. For each give the name the person goes by on the show
in its most common spelling ("Dale Wentworth", not "Bradley \"Dale\" Wentworth"), other
spellings/nicknames, gender ("male"/"female"), role, and relations to other
cast members (couple, twins, mother of, ...). Do not invent people; omit what you cannot verify.
Answer JSON only: {"show": "...", "season": "...", "cast": [{"name": "...", "aliases": [...],
"gender": "...", "role": "...", "relations": "..."}], "sources": ["<url>", ...]}"""


def fetch_cast(client: OpenRouterClient, folder_name: str, file_name: str, heard: list[str]) -> dict[str, Any]:
    prompt = f"Folder: {folder_name}\nFile: {file_name}\nNames heard: {', '.join(heard) or '-'}"
    answer = client.chat_json(SYSTEM_PROMPT, prompt)
    cast = [p for p in (answer.get("cast") or []) if isinstance(p, dict) and p.get("name")]
    return {"show": answer.get("show"), "season": answer.get("season"), "cast": cast,
            "sources": answer.get("sources") or [], "model": client.model, "usage": client.usage.to_dict()}


def find_cast(models: list[str], folder_name: str, file_name: str, heard: list[str]) -> dict[str, Any]:
    """The first answer with a real cast; a failure or refusal moves on to the next model."""
    errors = []
    for model in models:
        try:
            cast = fetch_cast(OpenRouterClient(model, reasoning="low"), folder_name, file_name, heard)
        except Exception as error:  # noqa: BLE001 - the next model may answer
            errors.append(f"{model}: {error}"[:300])
            log.warning("cast: %s failed: %s", model, error)
            continue
        if len(cast["cast"]) >= MIN_PEOPLE:
            return {**cast, **({"failed": errors} if errors else {})}
        errors.append(f"{model}: {len(cast['cast'])} people")
        log.warning("cast: %s found %d people, trying the next model", model, len(cast["cast"]))
    raise RuntimeError("no model found the cast: " + "; ".join(errors))


def cast_text(cast: dict[str, Any]) -> str:
    """The cast as prompt lines: name (aliases) - gender, role; relations."""
    lines = []
    for p in cast.get("cast", []):
        aliases = f" ({', '.join(p['aliases'])})" if p.get("aliases") else ""
        relations = f"; {p['relations']}" if str(p.get("relations") or "").lower() not in ("", "none", "null", "-") else ""
        lines.append(f"- {p['name']}{aliases} - {p.get('gender') or '?'}, {p.get('role') or '?'}{relations}")
    return f"{cast.get('show') or ''} {cast.get('season') or ''}".strip() + ":\n" + "\n".join(lines)


def find_member(name: str, cast: dict[str, Any]) -> dict[str, Any] | None:
    """The one cast member a name refers to (spellings, aliases, first name only); None if none or
    ambiguous ("John" with a John Rocker and a Jon Misch in the cast)."""
    from stt.voice_bank import same_person

    found = [person for person in cast.get("cast", [])
             if any(same_person(name, candidate) for candidate in [person["name"], *(person.get("aliases") or [])])]
    exact = [person for person in found if person["name"].lower() == name.lower()]
    return exact[0] if exact else found[0] if len(found) == 1 else None


def check_speakers(speakers: list[dict[str, Any]], cast: dict[str, Any]) -> list[str]:
    """Cast spellings for named speakers; notes on names not in the cast and gender conflicts."""
    from stt.voice_bank import same_person

    notes = []
    for s in speakers:
        if not s.get("name") or not s.get("cues"):
            continue
        member = find_member(s["name"], cast)
        if member is None:
            s["cast_check"] = "not in cast"
            notes.append(f"{s['id']} {s['name']}: not in the cast (or ambiguous)")
            continue
        # the cast's spelling of the same name; a nickname ("Felicity" / "Flick") stays as the LLM chose
        # it, so the voice bank does not get two names for one person
        if s["name"] != member["name"] and same_person(s["name"], member["name"]):
            notes.append(f"{s['id']} {s['name']} -> {member['name']} (cast spelling)")
            s["name"] = member["name"]
        if member.get("gender") in ("male", "female") and s.get("gender") in ("male", "female") \
                and member["gender"] != s["gender"]:
            s["cast_check"] = f"cast says {member['gender']}"
            notes.append(f"{s['id']} {s['name']}: voice/text say {s['gender']}, the cast {member['gender']}")
        else:
            s.pop("cast_check", None)
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find the cast of a season on the web (once per season)")
    parser.add_argument("folder", type=Path, help="Episode output folder (its source.json names the video)")
    parser.add_argument("--cast", type=Path, required=True, help="Season cast JSON to write (kept if it exists)")
    parser.add_argument("--model", default=",".join(DEFAULT_MODELS),
                        help="Models with web search, comma-separated: the next one if one fails or refuses")
    parser.add_argument("--force", action="store_true", help="Look up again even if the cast file exists")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    if args.cast.exists() and not args.force:
        log.info("cast: %s exists", args.cast)
        return 0
    media = Path(load_json(args.folder / "source.json")["media"])
    glossary = args.folder / "glossary.json"
    heard = [e["en"] for e in load_json(glossary) if e.get("type") == "person"] if glossary.exists() else []
    cast = find_cast([m for m in args.model.split(",") if m], media.parent.name, media.name, heard)
    save_json(args.cast, cast)
    log.info("cast: %s %s, %d people -> %s (%s)", cast["show"], cast["season"], len(cast["cast"]), args.cast,
             cast["usage"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
