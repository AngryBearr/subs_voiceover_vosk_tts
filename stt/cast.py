"""The cast of a season from the web: names, spelling, gender and relations of everyone who speaks.

Once per season (``<out-root>/<folder>.cast.json``) a model reads the show and season off the folder
and file names and lists the cast from one of two sources, per model of ``--model`` (tried in
order, the next one when one fails or refuses):

- a model id ending in ``:online`` searches the web itself (OpenRouter's web search; the shared
  default: gpt-6-luna, Claude Haiku if it fails);
- any other model gets the English Wikipedia article of the season: it names up to three
  searches, picks the article among the hits, and lists the cast from the article's text (the
  program fetches the pages; the model never browses). Through the OpenCode login of a
  subscription this costs nothing. gpt-6.1-sol this way (29.10, 3 requests, 20-30 s a season):
  S51 (fall 2026, after the model's training) the same 21 castaways + host as the web search;
  S29 the same 19 ("Dale Wentworth" where the web search wrote "Bradley "Dale" Wentworth"); S20
  the same 21 + the 6 loved ones of the family visit; Australian Survivor S10 "Heroes V Villains",
  24 + host. A gender the article does not give is left out (S51 Ori Jean-Charles; the web search
  said female, the voice says male).

The glossary takes the cast's spellings, the speaker naming gets the cast as evidence, and after
naming every speaker is checked against it (``check_speakers``):

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
import re
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stt.llm import OpenRouterClient
from stt.runner import load_json, save_json
from utils.llm_route import split_route

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


WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_CHARS = 60000  # article text given to the model (a season article without references: 20-40k)
WIKI_FIND_PROMPT = """You find the English Wikipedia article about one season of a TV show.
The user gives the video folder and file names (they name the show and season). Give up to three
Wikipedia search queries, the most likely first; the article's exact title if you know it
("Survivor: San Juan del Sur", "Australian Survivor: Heroes V Villains"), else the title such
articles usually have ("Survivor 51"). Answer JSON only: {"queries": ["...", ...]}"""
WIKI_PICK_PROMPT = """You pick the English Wikipedia article about one season of a TV show.
The user gives the video folder and file names and the hits of a Wikipedia search (title: snippet).
Answer JSON only: {"title": "<exact title of the season's own article>"} or {"title": null} when no
hit is that season's article (a list of contestants or the show's main article is not)."""
WIKI_CAST_PROMPT = SYSTEM_PROMPT.replace(
    "Use web search.",
    "Use the Wikipedia article the user gives (wikitext); list only people it names, gender from\n"
    "the article (pronouns) or from the name when clear.").replace(
    ', "sources": ["<url>", ...]}', "}")


def _wiki_get(params: dict[str, Any]) -> dict[str, Any]:
    url = WIKI_API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    request = urllib.request.Request(url, headers={"User-Agent": "subs-voiceover-stt/1.0 (season cast lookup)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def article_text(wikitext: str, limit: int = WIKI_CHARS) -> str:
    """Wikitext without references and comments (most of an article's size), cut to ``limit``."""
    text = re.sub(r"<!--.*?-->", "", wikitext, flags=re.S)
    text = re.sub(r"<ref[^>/]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    return text[:limit]


def wiki_cast(client: OpenRouterClient, folder_name: str, file_name: str, heard: list[str],
              wiki: Callable[[dict[str, Any]], dict[str, Any]] = _wiki_get) -> dict[str, Any]:
    """The cast from the season's Wikipedia article: the model names searches, picks the hit, reads it."""
    names = f"Folder: {folder_name}\nFile: {file_name}"
    queries = [str(q) for q in (client.chat_json(WIKI_FIND_PROMPT, names).get("queries") or []) if q][:3]
    hits: dict[str, str] = {}
    for query in queries:
        for hit in wiki({"action": "query", "list": "search", "srsearch": query, "srlimit": 5})["query"]["search"]:
            hits.setdefault(hit["title"], re.sub(r"<[^>]+>", "", hit.get("snippet", "")))
    if not hits:
        raise RuntimeError(f"no Wikipedia hits for {queries}")
    listing = "\n".join(f"- {title}: {snippet}" for title, snippet in hits.items())
    title = client.chat_json(WIKI_PICK_PROMPT, f"{names}\nHits:\n{listing}").get("title")
    if title not in hits:
        raise RuntimeError(f"no Wikipedia article of the season among {list(hits)}")
    page = wiki({"action": "parse", "page": title, "prop": "wikitext", "redirects": 1})["parse"]
    prompt = (f"{names}\nNames heard: {', '.join(heard) or '-'}\n\nWikipedia article "
              f"\"{page['title']}\":\n{article_text(page['wikitext']['*'])}")
    answer = client.chat_json(WIKI_CAST_PROMPT, prompt)
    cast = [p for p in (answer.get("cast") or []) if isinstance(p, dict) and p.get("name")]
    url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(page["title"].replace(" ", "_"))
    return {"show": answer.get("show"), "season": answer.get("season"), "cast": cast, "sources": [url],
            "model": client.model, "usage": client.usage.to_dict()}


def find_cast(models: list[str], folder_name: str, file_name: str, heard: list[str]) -> dict[str, Any]:
    """The first answer with a real cast; a failure or refusal moves on to the next model.

    A model id ending in ``:online`` searches the web itself; any other reads the Wikipedia article.
    """
    errors = []
    for model in models:
        try:
            lookup = fetch_cast if split_route(model)[1].endswith(":online") else wiki_cast
            cast = lookup(OpenRouterClient(model, reasoning="low"), folder_name, file_name, heard)
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
    parser = argparse.ArgumentParser(description="Find the cast of a season (web search or Wikipedia, once per season)")
    parser.add_argument("folder", type=Path, help="Episode output folder (its source.json names the video)")
    parser.add_argument("--cast", type=Path, required=True, help="Season cast JSON to write (kept if it exists)")
    parser.add_argument("--model", default=",".join(DEFAULT_MODELS),
                        help="Models, comma-separated, the next one if one fails or refuses: ids ending in "
                             "':online' search the web, others read the season's Wikipedia article")
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
