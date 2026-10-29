"""Stress check of the Russian text before synthesis.

RUAccent takes stress from a 3.2-million-word dictionary, picks one of the variants of a
homograph (замок/замок) with a model and guesses words it does not know (names, slang, rare
forms) with another model. Errors come from the last two, and the TTS engines say exactly the
stress they are given. Telling the stress from the synthesized audio did not work (three
detectors tried on 160 phrases with deliberately wrong stress: ~40% accuracy, chance level), so
the check runs on the text: guessed words and homographs are re-checked in their sentence by
several LLMs; a correction needs a majority of them to agree on the same other stress.

Names of the show's people (``--names``: the episode glossary) are checked too, in every case
form, even when RUAccent knows the word: it stresses English names the French way or as Russian
words (Жакл+ин, Натал+и, Мисс+и, Ал+ек, кит+а for Keith's "Кита"), ~20% of the 2462 name uses of
S29E01-E09. The models get the English original and skip the screener.

A second stress placer (silero-stress, if installed) reads the same lines; where it disagrees
with RUAccent on a word, that sentence also goes to the strong models without the screener:
on S29E03, of 53 such words RUAccent was right in ~13 of 22 kinds, silero in ~9 (п+осле, пор+а,
понял+а, вых+одит, п+олосу), all of which the screener had let through.

One stress per season (``--season``: ``<season>.stress.json`` next to the season glossary): a
name or guessed word the models decided by a majority (a correction or RUAccent's stress
confirmed) is written there with the votes, and later episodes take it from there without asking,
so a name sounds the same all season and a hand edit of the file applies to every episode.
Homographs stay per sentence. Without the file a name could get its correction in one episode
and not in the next, where the vote fell short.

Corrections go to ``stress.json``, which the engines load through
``engine.normalization.stress_dict``: ``dictionary`` holds guessed words (the same stress
everywhere), ``sentences`` holds homograph fixes per sentence (два го́лоса / нужны голоса́),
applied to exactly those sentences after RUAccent.

On S29E01 cheap models alone do more harm than good: gpt-6-luna with high reasoning found 27 of
the 31 corrections of three strong models (gpt-6.1-sol, claude-sonnet-5.5, gemini-4-argon; right
in 8-9 of 10 checked by hand) but added 36 of its own, mostly wrong (волос -> во́лос, гвозди ->
гвозди́); deepseek-v4.1-flash found 11 and added 75. So a cheap screener (``screen``) proposes,
and the strong models only vote on its proposals: ~$0.02 + the strong models on ~1/4 of the
sentences instead of ~$0.3 for all of them, at the cost of the few corrections the screener misses.

    uv run -m voiceover.stress_check dub.ru.json --out stress.json [--config voiceover_config.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voiceover.engine import build_engine
from voiceover.llm import LLMClient, LLMConfig, LLMError
from voiceover.text import join_lines

log = logging.getLogger("voiceover.stress_check")

PROMPT_VERSION = "stress-v4"
SEASON_SCHEMA = "season-stress-v1"
VOWELS = set("аеёиоуыэюя")
_WORD = re.compile(r"[А-Яа-яЁё+]+")
DEFAULT_MODELS = ("openai/gpt-6.1-sol", "anthropic/claude-sonnet-5.5", "google/gemini-4-argon")
DEFAULT_SCREEN = {"model": "openai/gpt-6-luna", "reasoning": "high"}

SYSTEM = """You are an expert in Russian pronunciation (orthoepy) checking word stress for a
text-to-speech voice-over. Each item has an id, a word with its stress marked by "+" before the
stressed vowel, and the sentence the word is used in; some items also list "variants" (the
possible stresses of a homograph).
For each item first determine the word's meaning and grammatical form in this sentence (part
of speech, case and number, or tense/person/aspect), then give the stress of standard Russian
for exactly that form. For names and borrowed words use their usual Russian pronunciation.
"variants" (if given) are the readings the dictionary knows; answer with one of them unless the
correct stress of this form is another one (поняла: п+оняла / пон+яла known, correct понял+а).
Answer JSON: {"items": [{"id": <id>, "form": "<meaning and form, a few words>",
"stressed": "<the word with + before the stressed vowel>"}]}, one entry per item in the same
order. Keep the letters exactly as given; only the "+" may move.
Items with "name_en" are the Russian spelling of an English name of a person in the show (a case
form of it): stress the syllable that is stressed in the English original as the person's
name is said in English (Jaclyn -> ж+аклин, Natalie -> н+атали, Alec -> +алек, Keith -> к+ита,
к+итом), the same syllable in every case form, not a French or Russian-word reading."""

# Case endings a Russian name form can add to the name (Кит -> Кита, Китом; Надия -> Надию).
NAME_ENDINGS = ("", "а", "я", "у", "ю", "ом", "ем", "ём", "ой", "ей", "е", "и", "ы", "ам", "ям", "ах", "ях")


@dataclass
class Suspect:
    """A word whose stress RUAccent did not take from its dictionary."""

    word: str  # lower case, no stress mark
    accented: str  # as RUAccent marked it (lower case)
    source: str  # guess | homograph
    variants: list[str] = field(default_factory=list)  # homograph stresses known to RUAccent
    contexts: list[str] = field(default_factory=list)
    count: int = 0
    name_en: str | None = None  # for a name of the show's people: the English original
    disputed: list[str] = field(default_factory=list)  # sentences where the second stress placer disagrees


def glossary_names(glossary: list[dict[str, Any]]) -> dict[str, str]:
    """Lower-case Russian name parts of the glossary's people -> their English original."""
    names: dict[str, str] = {}
    for entry in glossary:
        if entry.get("type") != "person" or not entry.get("ru") or not entry.get("en"):
            continue
        ru, en = entry["ru"].split(), entry["en"].split()
        for k, part in enumerate(ru):
            if sum(ch in VOWELS for ch in part.lower()) >= 1 and len(part) >= 3:
                names.setdefault(part.lower(), en[k] if len(ru) == len(en) else entry["en"])
    return names


def name_of(token: str, names: dict[str, str]) -> str | None:
    """The English original when `token` (as written, no marks) is a case form of a name."""
    if not token[:1].isupper():
        return None
    word = token.lower()
    for name, english in names.items():
        stem = name[:-1] if name[-1] in "аяйь" else name
        if word.startswith(stem) and word[len(stem):] in NAME_ENDINGS + ((name[-1],) if stem != name else ()):
            return english
    return None


def classify(word: str, accentizer: Any) -> str:
    """Where RUAccent took the stress of `word` (lower case, no marks) from."""
    if sum(ch in VOWELS for ch in word) < 2 or "ё" in word:
        return "trivial"
    if word in accentizer.custom_dict:
        return "project"
    if word in accentizer.omographs:
        return "homograph"
    if word in accentizer.accents:
        return "dictionary"
    return "guess"


def _stress(marked: str) -> tuple[str, int | None]:
    """(lower-case letters, index of the stressed vowel among the vowels or None) of a marked word."""
    letters, index, vowel = "", None, 0
    for i, ch in enumerate(marked.lower()):
        if ch == "+":
            continue
        if ch in VOWELS:
            if i and marked[i - 1] == "+":
                index = vowel
            vowel += 1
        letters += ch
    return letters, index


def disagreements(ours: str, theirs: str) -> set[int]:
    """Indexes of the words of `ours` (both "+"-marked texts of one line) whose stress `theirs` puts
    elsewhere; words with "ё", one vowel or other letters are left out."""
    a, b = _WORD.findall(ours), _WORD.findall(theirs)
    if len(a) != len(b):
        return set()
    out = set()
    for k, (x, y) in enumerate(zip(a, b)):
        (lx, ix), (ly, iy) = _stress(x), _stress(y)
        if lx == ly and "ё" not in lx and sum(ch in VOWELS for ch in lx) >= 2 and None not in (ix, iy) and ix != iy:
            out.add(k)
    return out


def second_opinion(texts: list[str]) -> list[str] | None:
    """The lines (plain text) as silero-stress marks them; None when it is not installed."""
    try:
        from silero_stress import load_accentor
    except ImportError:
        log.warning("stress: silero-stress is not installed; no second stress placer")
        return None
    accentor = load_accentor()
    return [accentor(text) for text in texts]


def collect(texts: list[str], accentizer: Any, max_contexts: int = 8,
            names: dict[str, str] | None = None, others: list[str] | None = None) -> dict[str, Suspect]:
    """Guessed words, homographs, names and words a second stress placer (``others``: the same
    lines marked by it) disagrees on, with a few example sentences."""
    found: dict[str, Suspect] = {}
    for n, text in enumerate(texts):
        disputed = disagreements(text, others[n]) if others else set()
        for k, match in enumerate(_WORD.finditer(text)):
            if text[match.end():match.end() + 1] == "-" or text[max(0, match.start() - 1):match.start()] == "-":
                continue  # parts of hyphenated words (кое-что, альфа-самец) are stressed separately
            accented = match.group(0).lower()
            word = accented.replace("+", "")
            source = classify(word, accentizer)
            english = name_of(match.group(0).replace("+", ""), names) if names and source != "trivial" else None
            if english:
                source = "name"
            elif source == "dictionary" and k in disputed:
                source = "disputed"  # RUAccent's dictionary and the second placer disagree in this sentence
            elif source not in ("guess", "homograph"):
                continue
            variants = list(accentizer.omographs.get(word, [])) if source == "homograph" else []
            item = found.setdefault(word, Suspect(word, accented, source, variants, name_en=english))
            item.count += 1
            sentence = text.replace("+", "")
            if len(item.contexts) < max_contexts and sentence not in item.contexts:
                item.contexts.append(sentence)
            if k in disputed and sentence in item.contexts and sentence not in item.disputed:
                item.disputed.append(sentence)
    return found


def valid_stress(suspect: Suspect, stressed: str) -> bool:
    """One "+" right before a vowel and the same letters (a homograph may get a stress outside the
    dictionary's variants: поняла has п+оняла / пон+яла there, but is понял+а)."""
    position = stressed.find("+")
    return (stressed.replace("+", "") == suspect.word and stressed.count("+") == 1
            and position + 1 < len(stressed) and stressed[position + 1] in VOWELS)


def queries(suspects: list[Suspect]) -> list[tuple[Suspect, str]]:
    """What to ask: a guessed word or a name once, a homograph or a disputed word in each of its sentences."""
    return [(s, context) for s in suspects
            for context in (s.contexts if s.source in ("homograph", "disputed") else s.contexts[:1])]


def ask(pairs: list[tuple[Suspect, str]], client: LLMClient, batch: int = 40, workers: int = 4) -> list[str | None]:
    """One model's stress for every (word, sentence) pair; None where the answer is invalid or missing.

    Batches go out ``workers`` at a time (S29: the screener's 7 batches took 6 min one by one)."""
    offsets = range(0, len(pairs), batch)
    with ThreadPoolExecutor(max(1, min(workers, len(offsets)))) as pool:
        parts = pool.map(lambda offset: _ask_batch(pairs[offset:offset + batch], client, f"stress:{offset}"), offsets)
    return [answer for part in parts for answer in part]


def _ask_batch(chunk: list[tuple[Suspect, str]], client: LLMClient, tag: str) -> list[str | None]:
    items = [{"id": i, "word": s.accented, "sentence": context, **({"variants": s.variants} if s.variants else {}),
              **({"name_en": s.name_en} if s.name_en else {})} for i, (s, context) in enumerate(chunk)]
    try:
        result = client.complete_json(SYSTEM, json.dumps({"items": items}, ensure_ascii=False),
                                      prompt_version=PROMPT_VERSION, tag=tag)
    except LLMError as error:  # a broken answer: ask again in halves, give up on single items
        if len(chunk) == 1:
            log.warning("stress: %s gave no usable answer for %s: %s", client.config.model, tag, str(error)[:120])
            return [None]
        half = len(chunk) // 2
        return _ask_batch(chunk[:half], client, tag + "a") + _ask_batch(chunk[half:], client, tag + "b")
    except Exception as error:  # noqa: BLE001 - the model is not reachable (e.g. a subscription login): no votes
        log.warning("stress: %s did not answer %s (%s): the others vote", client.config.model, tag, str(error)[:160])
        return [None] * len(chunk)
    answers = {a.get("id"): a for a in result.content.get("items", []) if isinstance(a, dict)}
    out: list[str | None] = []
    for i, (suspect, _) in enumerate(chunk):
        stressed = str((answers.get(i) or {}).get("stressed", "")).strip().lower()
        out.append(stressed if valid_stress(suspect, stressed) else None)
    return out


def vote(suspect: Suspect, answers: list[str | None]) -> str | None:
    """The corrected stress if a majority of all models agree on the same other stress."""
    other = Counter(a for a in answers if a and a != suspect.accented)
    if not other:
        return None
    stressed, votes = other.most_common(1)[0]
    return stressed if votes > len(answers) / 2 else None


def lines_sha256(lines: list[dict[str, Any]]) -> str:
    """Fingerprint of the lines' texts: another engine's script with the same lines needs no new check."""
    import hashlib

    texts = [join_lines(line.get("text")) for line in lines]
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode("utf-8")).hexdigest()


def decided(answers: list[str | None]) -> str | None:
    """The stress a majority of all models gave (a correction or the one RUAccent had)."""
    counts = Counter(a for a in answers if a)
    if not counts:
        return None
    stressed, votes = counts.most_common(1)[0]
    return stressed if votes > len(answers) / 2 else None


def load_season(path: Path | None) -> dict[str, dict[str, Any]]:
    """{word: {"stressed", "source", ...}} of a season's stress file; empty when there is none."""
    if not path or not path.is_file():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")).get("words", {}))


def save_season(path: Path, new: dict[str, dict[str, Any]], episode: str) -> int:
    """Add the words decided in an episode; words already there (decided earlier or edited by hand)
    are never replaced. Returns the number added."""
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"schema": SEASON_SCHEMA, "words": {}}
    words = data.setdefault("words", {})
    added = 0
    for word, entry in sorted(new.items()):
        if word not in words:
            words[word] = {**entry, "episode": episode}
            added += 1
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)
    return added


def _client(llm_config: dict[str, Any], model: str, env_file: Path | None, **extra: Any) -> LLMClient:
    # reasoning models spend part of max_tokens on thinking before the JSON answer
    return LLMClient(LLMConfig.from_dict({**llm_config, "model": model, **extra,
                                          "max_tokens": max(16000, int(llm_config.get("max_tokens", 0)))}),
                     env_file=env_file)


def check(lines: list[dict[str, Any]], engine_config: dict[str, Any], llm_config: dict[str, Any],
          models: list[str], env_file: Path | None = None, screen: dict[str, Any] | None = None,
          names: dict[str, str] | None = None, second: bool = True,
          season: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Accent the lines exactly as the engine will, then re-check guessed words and homographs
    (names and guessed words in ``season`` are taken from there, not asked)."""
    engine = build_engine({**engine_config, "normalization": {
        **engine_config.get("normalization", {}), "stress_dict": None}})
    items = [{"cue_id": str(i), "index": i, "text": join_lines(line.get("text"))} for i, line in enumerate(lines)]
    prepared = engine.prepare(items)
    accentizer = engine._accentizer  # type: ignore[attr-defined]
    texts = [p.tts_text for p in prepared]
    others = second_opinion([t.replace("+", "") for t in texts]) if second else None
    suspects = sorted(collect(texts, accentizer, names=names, others=others).values(),
                      key=lambda s: (s.source, s.word))
    counts = Counter(s.source for s in suspects)
    log.info("stress: %d lines, %d guessed words, %d homographs, %d name forms, %d disputed words "
             "(%d sentences where the second placer disagrees) to check with %d models", len(lines), counts["guess"],
             counts["homograph"], counts["name"], counts["disputed"], sum(len(s.disputed) for s in suspects),
             len(models))
    season = season or {}
    known = [x for x in suspects if x.source in ("guess", "name") and x.word in season]
    suspects = [x for x in suspects if x not in known]
    if known:
        log.info("stress: %d names and guessed words decided earlier in the season", len(known))
    clients = [_client(llm_config, model, env_file) for model in models]
    pairs = queries(suspects)
    screened: list[str | None] = [None] * len(pairs)
    stats: dict[str, Any] = {}
    if screen:  # the cheap model proposes, the strong ones vote only on its proposals
        screener = _client(llm_config, str(screen["model"]), env_file,
                           **{k: v for k, v in screen.items() if k != "model"})
        screened = ask(pairs, screener)
        stats[f"screen:{screen['model']}"] = screener.stats()
        todo = [n for n, (s, context) in enumerate(pairs)  # names and disputed sentences: always
                if s.source == "name" or context in s.disputed or (screened[n] and screened[n] != s.accented)]
        log.info("stress: the screener proposes %d changes in %d sentences", len(todo), len(pairs))
    else:
        todo = list(range(len(pairs)))
    with ThreadPoolExecutor(len(clients)) as pool:
        partial = list(pool.map(lambda client: ask([pairs[n] for n in todo], client), clients))
    answers: list[list[str | None]] = [[None] * len(pairs) for _ in clients]
    for a, part in zip(answers, partial):
        for n, value in zip(todo, part):
            a[n] = value
    checked = set(todo)
    verdicts: dict[str, dict[str, Any]] = {}
    guessed: dict[str, str] = {}
    new_words: dict[str, dict[str, Any]] = {}  # names and guessed words decided here, for the season
    sentences: dict[str, dict[str, str]] = {}
    for n, (s, context) in enumerate(pairs):
        votes = [a[n] for a in answers]
        corrected = vote(s, votes) if n in checked else None
        entry = verdicts.setdefault(s.word, {"word": s.word, "accented": s.accented, "source": s.source,
                                             "count": s.count, "variants": s.variants, "contexts": [],
                                             **({"name_en": s.name_en} if s.name_en else {})})
        entry["contexts"].append({"sentence": context, "answers": dict(zip(models, votes)), "screen": screened[n],
                                  "corrected": corrected, **({"disputed": True} if context in s.disputed else {})})
        if corrected and s.source in ("guess", "name"):
            guessed[s.word] = corrected
        elif corrected:
            sentences.setdefault(context, {})[s.word] = corrected
        if n in checked and s.source in ("guess", "name") and (stressed := decided(votes)):
            new_words[s.word] = {"stressed": stressed, "source": s.source, "votes": dict(zip(models, votes)),
                                 **({"name_en": s.name_en} if s.name_en else {})}
    for s in known:
        stressed = str(season[s.word]["stressed"])
        verdicts[s.word] = {"word": s.word, "accented": s.accented, "source": s.source, "count": s.count,
                            "variants": [], "contexts": [], "season": stressed,
                            **({"name_en": s.name_en} if s.name_en else {})}
        if stressed != s.accented and valid_stress(s, stressed):
            guessed[s.word] = stressed
    return {
        "schema": "voiceover-stress-v3",
        "prompt_version": PROMPT_VERSION,
        "models": models,
        "dictionary": dict(sorted(guessed.items())),
        "sentences": sentences,
        "checked": dict(counts),
        "second_placer": "silero-stress" if others is not None else None,
        "corrected": {"guess": sum(v["source"] == "guess" for k, v in verdicts.items() if k in guessed),
                      "name": sum(v["source"] == "name" for k, v in verdicts.items() if k in guessed),
                      "homograph_sentences": sum(len(v) for v in sentences.values())},
        "season": {"used": len(known), "decided": new_words},
        "verdicts": list(verdicts.values()),
        "llm": {**stats, **{model: client.stats() for model, client in zip(models, clients)}},
    }


def restress(token: str, stressed: str) -> str:
    """`token` (any case, maybe with "+") with the stress of `stressed` (lower case, one "+")."""
    plain = token.replace("+", "")
    position = stressed.index("+")
    return plain[:position] + "+" + plain[position:]


def apply_sentence_fixes(text: str, fixes: dict[str, dict[str, str]], key: str) -> str:
    """Re-stress the words of one accented text listed for its sentence (`key`: plain text)."""
    words = fixes.get(key)
    if not words:
        return text
    return _WORD.sub(lambda m: restress(m.group(0), words[m.group(0).lower().replace("+", "")])
                     if m.group(0).lower().replace("+", "") in words else m.group(0), text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-check RUAccent stress of guessed words and homographs (LLMs)")
    parser.add_argument("source", type=Path, help="JSON list of Russian lines (text: str | list[str])")
    parser.add_argument("--out", type=Path, required=True, help="stress.json to write")
    parser.add_argument("--config", type=Path, default=Path("voiceover_config.json"))
    parser.add_argument("--engine", default=None, help="Override engine.name (vosk | teratts)")
    parser.add_argument("--models", default=None, help="Comma-separated OpenRouter models (default: config/built-in)")
    parser.add_argument("--no-screen", action="store_true", help="Let the strong models check every sentence")
    parser.add_argument("--screen", default=None, help="Cheap model that proposes changes (default: config/built-in)")
    parser.add_argument("--no-second", action="store_true", help="No second stress placer (silero-stress)")
    parser.add_argument("--names", type=Path, default=None,
                        help="Glossary JSON (stt.glossary): its people's names are checked in every case form")
    parser.add_argument("--season", type=Path, default=None,
                        help="<season>.stress.json: names / guessed words decided in earlier episodes (read and extended)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    engine_config = {**config["engine"], **({"name": args.engine} if args.engine else {})}
    stress = config.get("stress", {})
    llm_config = stress.get("llm") or config.get("verify", {}).get("llm") or config["llm"]
    models = args.models.split(",") if args.models else list(stress.get("models", DEFAULT_MODELS))
    screen = None if args.no_screen else ({**DEFAULT_SCREEN, "model": args.screen} if args.screen
                                          else stress.get("screen", DEFAULT_SCREEN))
    lines = json.loads(args.source.read_text(encoding="utf-8"))
    names = glossary_names(json.loads(args.names.read_text(encoding="utf-8"))) if args.names else None
    result = check(lines, engine_config, llm_config, models, env_file=Path(".env"), screen=screen, names=names,
                   second=not args.no_second, season=load_season(args.season))
    result["lines_sha256"] = lines_sha256(lines)
    tmp = args.out.with_suffix(".tmp")  # engines of the other lane may be reading the old one
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(args.out)
    if args.season:
        added = save_season(args.season, result["season"]["decided"], args.source.resolve().parent.name)
        log.info("stress: %d words taken from %s, %d added to it", result["season"]["used"], args.season.name, added)
    log.info("stress: corrected %d guessed words, %d name forms, %d homographs in their sentences -> %s",
             result["corrected"]["guess"], result["corrected"]["name"], result["corrected"]["homograph_sentences"],
             args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
