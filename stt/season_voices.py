"""Season voices: every word is assigned again over the people the pipeline named in the whole season.

The word-level speaker assignment (``stt.boundaries``) compares each word with the average voice of the diarizer's
clusters of one episode, and a cluster often mixes people. After the first pass of every episode (transcript and naming
done) this optional stage

1. sums the word embeddings under each named person over the season (``<season>.season_voices.npz``, beside the voice
   bank; a person needs ``min_words`` words; words of named subtitle lines count for the person too),
2. assigns the words of every episode again with the Viterbi pass of ``stt.boundaries`` over the season's people and the
   episode's clusters that carry no name (a cluster change of the diarizer is a cheap switch), smooths like the first pass,
3. rewrites ``words.json``, ``subs.en.json``, ``subs.en.srt``, ``subs.en.speakers.srt`` and ``speakers.json`` (one speaker
   per person) and writes ``season_voices.json`` (what moved).

The first-pass files are kept once as ``*.before_voices.json``; the stage always starts from them, so a rerun (also with
more episodes in the season) gives the same result as a first run. No LLM and no GPU: the word embeddings of the first
pass are reused (``words.emb.<model>.npz``). A voice without a name that the voice bank gave a season identity
("V03", ``stt.voice_bank``) is a person of the season too, keyed by that id (``"name": null, "identity": "V03"`` on its
card). Off by default (``season_voices`` in ``stt_config.json``).

    subs_env/bin/python -m stt.season_voices output/stt [--episodes "*S51*"] [--dry-run]
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import json
import logging
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from stt.assemble import Cue, CueOptions, assign_speakers, smooth_speakers
from stt.boundaries import OWN_STATE, RelabelParams, WordHints, _centroids, _viterbi, restore_fixed
from stt.diarize import Diarization
from stt.runner import load_json, save_json
from stt.types import Word
from stt.voice_bank import is_identity

log = logging.getLogger("stt.season_voices")

SCHEMA = "season-voices-v1"
STATE_PREFIX = "V:"  # state / ``diarizer_label`` of a person: "V:<name>"
BEFORE = ".before_voices"  # words.before_voices.json, speakers.before_voices.json, subs.en.before_voices.json
REPORT = "season_voices.json"
MIN_WORDS = 20  # words a person needs over the season
SWITCH_PENALTY = 0.3  # the penalties of the first pass (x1.0); the diarizer's cluster change is a cheap switch
TURN_BONUS = 0.15
FILES = ("words", "speakers", "subs.en")  # the files the stage replaces (all .json)


# ---------------------------------------------------------------- names -> people
def clean_name(name: str) -> str:
    return " ".join(name.split())


def group_names(counts: dict[str, int]) -> dict[str, str]:
    """Raw speaker name (or identity) -> person key. Spellings of one name (``voice_bank.same_person``: "Jon"/"John Misch", "Jeff"/"Jeff
    Probst") are one person, keyed by the spelling with most words (the longer one on a tie)."""
    from stt.voice_bank import same_person

    keys: list[str] = []
    out: dict[str, str] = {}
    for name in sorted(counts, key=lambda n: (-counts[n], -len(n), n)):
        if is_identity(name):  # a voice identity is its own person
            out[name] = name
            continue
        key = next((k for k in keys if k == name or same_person(name, k)), None)
        if key is None:
            keys.append(name)
            key = name
        out[name] = key
    return out


@dataclass
class SeasonVoices:
    """The season's people: unit voice vectors, word counts and one speaker card each (``cards``, the largest one)."""

    names: list[str]
    vectors: np.ndarray  # (people, d), unit
    counts: list[int]
    model: str
    episodes: list[str] = field(default_factory=list)
    fingerprint: str = ""
    cards: dict[str, dict[str, Any]] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)  # lower-case alias -> name (the season's cast list)
    cast: dict[str, Any] = field(default_factory=dict)  # the season's cast (``<season>.cast.json``): resolves subtitle labels

    def __len__(self) -> int:
        return len(self.names)

    def person_of(self, name: str | None) -> str | None:
        """The person (key) a speaker name or identity belongs to, or None when the season has no such person.
        An identity matches exactly only."""
        if not name:
            return None
        from stt.voice_bank import same_person

        name = clean_name(name)
        if name in self.names:
            return name
        if is_identity(name):
            return None
        order = sorted(range(len(self.names)), key=lambda j: -(self.counts[j] if self.counts else 0))  # no counts while building
        return next((self.names[j] for j in order if same_person(name, self.names[j])), None)

    def by_first_name(self, label: str) -> str | None:
        """The one person whose first name is ``label``'s (case-insensitive); None when none or several."""
        from stt.cast import member_for_label
        from stt.subtitles import HOST, HOST_LABELS

        member = member_for_label(label, self.cast) if self.cast else None
        if member is not None:  # the cast says who the label is ("B.B" = B.B. Andersen, "_HOST" = the host)
            person = self.person_of(member["name"])
            if person is not None:
                return person
        if label == HOST:
            found = [n for n in self.names if not is_identity(n)
                     and (n.upper() in HOST_LABELS or n.upper().split()[0] in HOST_LABELS)]
        else:
            first = label.lower().split()[0] if label.split() else ""
            found = [n for n in self.names if not is_identity(n) and n.lower().split()[0] == first] if first else []
        if not found and label != HOST:  # subtitles also use surnames and nicknames: "KILBY", "JELLY"
            wanted = label.lower()
            person = self.aliases.get(wanted)
            found = [person] if person in self.names else [n for n in self.names
                                                           if not is_identity(n) and wanted in n.lower().split()]
        return found[0] if len(found) == 1 else None

    def own_member(self, label: str) -> dict[str, Any] | None:
        """The cast member a subtitle label stands for (None without a cast or a clear member)."""
        from stt.cast import member_for_label

        return member_for_label(label, self.cast) if self.cast else None

    def own_name(self, label: str) -> str:
        """Name of a person the subtitles name but the first pass never did: the cast's spelling ("SEAN" -> "Sean Kenniff",
        "_HOST" -> "Jeff Probst"), else the label in title case."""
        member = self.own_member(label)
        return member["name"] if member else label.title()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as handle:
            np.savez(handle, names=np.array(self.names), vectors=self.vectors.astype(np.float32),
                     counts=np.array(self.counts, dtype=np.int64), model=np.array(self.model),
                     episodes=np.array(json.dumps(self.episodes)), fingerprint=np.array(self.fingerprint),
                     cards=np.array(json.dumps(self.cards, ensure_ascii=False)), schema=np.array(SCHEMA))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> SeasonVoices | None:
        try:
            with np.load(path) as data:
                if str(data["schema"]) != SCHEMA:
                    return None
                return cls(names=[str(n) for n in data["names"]], vectors=data["vectors"].astype(np.float64),
                           counts=[int(c) for c in data["counts"]], model=str(data["model"]),
                           episodes=json.loads(str(data["episodes"])), fingerprint=str(data["fingerprint"]),
                           cards=json.loads(str(data["cards"])))
        except (OSError, KeyError, ValueError):
            return None


# ---------------------------------------------------------------- first-pass data of an episode
def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def before_path(folder: Path, name: str) -> Path:
    return folder / f"{name}{BEFORE}.json"


def written_by_stage(folder: Path) -> bool:
    """Whether the transcript files in ``folder`` are still the ones this stage wrote (its report says so)."""
    report, words = folder / REPORT, folder / "words.json"
    if not (report.exists() and words.exists()):
        return False
    try:
        return load_json(report).get("written", {}).get("words") == _sha(words)
    except (OSError, ValueError, AttributeError):
        return False


def restore_first_pass(folder: Path) -> bool:
    """Move the saved first pass (``*.before_voices.json``) back in place of this stage's output, before the naming
    stage runs again: the naming then works on the first-pass speakers (the ids ``speakers.override.json`` uses), and
    this stage starts from the new names instead of the saved old ones. False when there is nothing to restore."""
    saved = {n: before_path(folder, n) for n in FILES}
    if not all(p.exists() for p in saved.values()) or not written_by_stage(folder):
        return False
    for name, path in saved.items():
        path.replace(folder / f"{name}.json")
    return True


def first_pass_ids(folder: Path) -> dict[str, str]:
    """Speaker id in ``speakers.json`` now -> the id of the same speaker in the first pass (the ids the naming stage and
    ``speakers.override.json`` use). The same ids while the files are the first pass. After this stage, a cluster
    without a name keeps its diarizer label, which finds its first-pass card; people (one card per person, made by
    this stage) have no first-pass id and are left out."""
    try:
        current = load_json(folder / "speakers.json")
    except (OSError, ValueError):
        return {}
    if not isinstance(current, list):
        return {}
    saved = before_path(folder, "speakers")
    if not (written_by_stage(folder) and saved.exists()):
        return {c["id"]: c["id"] for c in current if isinstance(c, dict) and c.get("id")}
    by_label: dict[Any, list[str]] = defaultdict(list)
    for card in load_json(saved):
        if isinstance(card, dict) and card.get("id"):
            by_label[card.get("diarizer_label")].append(card["id"])
    mapping: dict[str, str] = {}
    for card in current:
        if not isinstance(card, dict) or not card.get("id"):
            continue
        label = card.get("diarizer_label")
        ids = by_label.get(label) or []
        if label and not str(label).startswith(STATE_PREFIX) and len(ids) == 1:
            mapping[card["id"]] = ids[0]
    return mapping


def first_pass_files(folder: Path) -> tuple[dict[str, Path], bool]:
    """The first-pass files to read (words, speakers, subs.en) and whether they are the saved originals.

    The saved ``*.before_voices.json`` are the first pass as long as the files in the folder are still what this stage wrote;
    otherwise the transcript was made again since (the files in the folder are the new first pass)."""
    current = {n: folder / f"{n}.json" for n in FILES}
    saved = {n: before_path(folder, n) for n in FILES}
    report = folder / REPORT
    if all(p.exists() for p in saved.values()) and report.exists() and current["words"].exists():
        try:
            if load_json(report).get("written", {}).get("words") == _sha(current["words"]):
                return saved, True
        except (OSError, ValueError):
            pass
    return current, False


@dataclass
class EpisodeData:
    folder: Path
    words: list[Word]  # on the audio timeline (the media offset taken off)
    raw_words: list[dict[str, Any]]
    cards: list[dict[str, Any]]
    records: list[dict[str, Any]]
    offset: float
    emb: np.ndarray
    original: bool  # read from the saved ``*.before_voices.json``
    sources: dict[str, Path]

    def card_of(self, speaker_id: str | None) -> dict[str, Any] | None:
        return next((c for c in self.cards if c["id"] == speaker_id), None)

    def word_names(self) -> list[str | None]:
        """The name of the speaker of every word (None: no speaker, or one without a name)."""
        names = {c["id"]: clean_name(c["name"]) for c in self.cards if c.get("name")}
        return [names.get(w.get("speaker")) for w in self.raw_words]

    def word_keys(self) -> list[str | None]:
        """The person key (name, else identity) of the speaker of every word (None: no speaker, or neither)."""
        from stt.voice_bank import person_key

        keys = {c["id"]: person_key(c) for c in self.cards if person_key(c)}
        return [keys.get(w.get("speaker")) for w in self.raw_words]


def _offset(folder: Path) -> float:
    source = folder / "source.json"
    try:
        return float(load_json(source).get("start") or 0.0) if source.exists() else 0.0
    except (OSError, ValueError):
        return 0.0


def embedding_path(folder: Path, model: str) -> Path:
    from stt.refine_speakers import model_cache_key

    return folder / f"words.emb.{model_cache_key(model)}.npz"


def load_episode(folder: Path, model: str) -> EpisodeData | None:
    """The first pass of an episode, or None when it is not done (no words, speakers or word embeddings) or the word
    embeddings do not belong to the words."""
    sources, original = first_pass_files(folder)
    emb_path = embedding_path(folder, model)
    missing = [p.name for p in (*sources.values(), emb_path) if not p.exists()]
    if missing:
        log.info("[%s] no first pass yet (missing %s)", folder.name, ", ".join(missing))
        return None
    raw = load_json(sources["words"])
    offset = _offset(folder)
    words = [Word(w["start"] - offset, w["end"] - offset, w["text"], w.get("score"), None, w.get("source"),
                  bool(w.get("caption"))) for w in raw]
    try:
        with np.load(emb_path) as data:
            emb, starts, ends = data["emb"], data["starts"], data["ends"]
    except (OSError, KeyError, ValueError) as error:
        log.warning("[%s] word embeddings unreadable (%s)", folder.name, error)
        return None
    if not _times_match(starts, ends, words) or len(emb) != len(words):
        log.warning("[%s] %s does not belong to words.json (different words); skipped", folder.name, emb_path.name)
        return None
    return EpisodeData(folder, words, raw, load_json(sources["speakers"]), load_json(sources["subs.en"]), offset,
                       emb.astype(np.float64), original, sources)


def _times_match(starts: np.ndarray, ends: np.ndarray, words: list[Word]) -> bool:
    """Whether stored word embeddings (their word start/end times) belong to these words."""
    return (len(starts) == len(words) and len(ends) == len(words)
            and bool(np.allclose(starts, [w.start for w in words], atol=2e-3))
            and bool(np.allclose(ends, [w.end for w in words], atol=2e-3)))


def _stale_embeddings(path: Path, words: list[Word]) -> bool:
    """True when ``path`` holds word embeddings of other words (the transcript was rebuilt since): recompute them."""
    try:
        with np.load(path) as data:
            return not _times_match(data["starts"], data["ends"], words) or len(data["emb"]) != len(words)
    except (OSError, KeyError, ValueError):
        return True


def find_diarization(folder: Path, cards: list[dict[str, Any]]) -> Diarization | None:
    """The diarization the first pass used: a refined one before a plain one, whose speakers cover the cards' labels."""
    labels = {str(c["diarizer_label"]) for c in cards
              if c.get("cues") and not str(c["diarizer_label"]).startswith((STATE_PREFIX, OWN_STATE))}  # not clusters of a diarizer
    candidates = sorted(folder.glob("diarization.*.refined.*.json"), key=lambda p: -p.stat().st_mtime)
    candidates += sorted((p for p in folder.glob("diarization.*.json") if ".refined." not in p.name),
                         key=lambda p: -p.stat().st_mtime)
    for path in candidates:
        try:
            diarization = Diarization.from_dict(load_json(path))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if labels <= set(diarization.speakers):
            return diarization
    return None


def subtitle_hints(data: EpisodeData) -> Any:
    """What the episode's named subtitle lines say (``stt.subtitle_hints``): WordHints with ``persons``; None without."""
    path = data.folder / "subs.source.en.srt"
    if not path.exists():
        return None
    try:
        from stt.subtitle_hints import build_hints

        hints, _ = build_hints(path, data.words, data.offset)
    except Exception as error:  # noqa: BLE001 - subtitles are a bonus
        log.warning("[%s] subtitle hints failed: %s", data.folder.name, error)
        return None
    return hints if hints.persons or hints.turns else None


# ---------------------------------------------------------------- build
def fingerprint(folders: list[Path], model: str, min_words: int, cast: dict[str, Any] | None = None,
                source: str = "all") -> str:
    """What the voices depend on: the first-pass files, the word embeddings and the subtitles of every episode, and the
    season's cast (it turns subtitle labels into people)."""
    from stt.refine_speakers import model_cache_key

    digest = hashlib.sha256(f"{SCHEMA}|{model_cache_key(model)}|{min_words}{'' if source == 'all' else '|' + source}".encode())
    digest.update(json.dumps(cast or {}, sort_keys=True, ensure_ascii=False).encode())
    for folder in sorted(folders):
        sources, _ = first_pass_files(folder)
        parts = [sources["words"], sources["speakers"], embedding_path(folder, model), folder / "subs.source.en.srt",
                 folder / "ocr.json", folder / "onscreen.json"]
        digest.update(folder.name.encode())
        for path in parts:
            digest.update((_sha(path) if path.exists() else "-").encode())
    return digest.hexdigest()


def card_words(episode: EpisodeData, voices: SeasonVoices) -> dict[str, list[int]]:
    """Person -> the words spoken while a name card of that person was on screen (``ocr.json``): an interview card shows
    the person talking. The cards of title rolls and of tribes, days and places are left out (``stt.assemble``); a card
    counts only when the season's cast says whose it is ("SUSAN" -> Susan Hawk), so this needs the cast."""
    from stt.assemble import all_cards, place_card, title_roll_cards
    from stt.onscreen import _tribe_words

    path, screen = episode.folder / "ocr.json", episode.folder / "onscreen.json"
    if not voices.cast or not path.exists():
        return {}
    try:
        items = load_json(screen) if screen.exists() else []
        cards = all_cards(load_json(path), items if isinstance(items, list) else [], episode.offset)
    except (OSError, ValueError, AttributeError, KeyError, TypeError):
        return {}
    cards = [c for c in cards if c.get("kind") == "name_card" and c.get("text")]
    roll = title_roll_cards(cards, episode.words)
    tribes = _tribe_words([{"text": f"{c.get('text') or ''} {c.get('descriptor') or ''}"} for c in cards])
    middles = [(w.start + w.end) / 2 for w in episode.words]
    out: dict[str, list[int]] = defaultdict(list)
    for k, card in enumerate(cards):
        if k in roll or place_card(card, tribes):
            continue
        text = str(card["text"])
        member = voices.own_member(text) or voices.own_member(text.split()[0])
        if member is None:
            continue
        key = voices.person_of(member["name"]) or clean_name(member["name"])
        out[key] += [i for i, m in enumerate(middles) if card["start"] <= m <= card["end"]
                     and episode.words[i].source != "ocr"]
    return {key: sorted(set(idx)) for key, idx in out.items() if idx}


def build_voices(folders: list[Path], model: str = "resnet293", min_words: int = MIN_WORDS,
                 aliases: dict[str, str] | None = None, cast: dict[str, Any] | None = None, source: str = "all") -> SeasonVoices:
    """The season's people from the named speakers (and named subtitle lines) of the episodes with a first pass.
    ``source`` "clean": a person with ``min_words`` words under their name cards and on their named subtitle lines gets a
    voice of those words only, not of the whole clusters named after them (S02E05: a 151-cue cluster of several people
    named Kimmi after one card made her voice an average of them all); "all": every word under the person's name."""
    episodes = [e for e in (load_episode(f, model) for f in folders) if e is not None]
    totals: Counter[str] = Counter()
    for episode in episodes:
        for name, n in Counter(n for n, w in zip(episode.word_keys(), episode.words) if n and w.source != "ocr").items():
            totals[name] += n
    key_of = group_names(dict(totals))
    keys = SeasonVoices(sorted(set(key_of.values())), np.zeros((0, 0)), [], model, aliases=dict(aliases or {}),
                       cast=cast or {})
    sums: dict[str, np.ndarray] = {}
    counts: Counter[str] = Counter()
    card_sums: dict[str, np.ndarray] = {}  # words under name cards: a voice only for people with too few words otherwise
    card_counts: Counter[str] = Counter()
    label_sums: dict[str, np.ndarray] = {}  # words of named subtitle lines (for ``source`` "clean")
    label_counts: Counter[str] = Counter()
    cards: dict[str, tuple[int, dict[str, Any]]] = {}
    from stt.speakers import _is_mixed
    from stt.voice_bank import person_key

    for episode in episodes:
        person: dict[int, str] = {}
        # a card the naming itself calls mixed ("contestant; mixed-speaker cluster", S02E05: 151 cues of several people
        # named Kimmi) keeps its name but gives the person's voice no words: they would pull other people's words to it
        mixed = {c["id"] for c in episode.cards if _is_mixed(c)}
        for i, (name, word, raw) in enumerate(zip(episode.word_keys(), episode.words, episode.raw_words)):
            if name and word.source != "ocr" and raw.get("speaker") not in mixed:
                person[i] = key_of[name]
        hints = subtitle_hints(episode)
        labelled: dict[int, str] = {}
        for label, idx in (hints.persons.items() if hints else ()):
            key = keys.by_first_name(label)
            if key is None:  # a person only the subtitles name ("DIRK:" on S01E01): a voice of their own when the cast
                member = keys.own_member(label)  # says who it is (labels like "MAN" or "BOTH" are nobody's)
                key = clean_name(member["name"]) if member else None
            if key is None:
                continue
            for i in idx:
                if episode.words[i].source != "ocr":
                    person[i] = labelled[i] = key  # a named subtitle line beats the speaker's name
        for key, idx in _group(person).items():
            sums[key] = sums.get(key, 0.0) + episode.emb[idx].sum(axis=0)
            counts[key] += len(idx)
        for key, idx in card_words(episode, keys).items():
            card_sums[key] = card_sums.get(key, 0.0) + episode.emb[idx].sum(axis=0)
            card_counts[key] += len(idx)
        for key, idx in _group(labelled).items():
            label_sums[key] = label_sums.get(key, 0.0) + episode.emb[idx].sum(axis=0)
            label_counts[key] += len(idx)
        for card in episode.cards:
            key = key_of.get(person_key(card) or "")
            weight = (card.get("words") or 0) - (10 ** 9 if card["id"] in mixed else 0)  # a clean card first
            if key and weight > cards.get(key, (-(10 ** 10), {}))[0]:
                cards[key] = (weight, card)
    # People the names and subtitles give too few words (S01E01: Susan's interview fell into a mixed cluster) get a voice
    # from the words under their name cards; the Viterbi pass then decides which words are theirs.
    for key in sorted(set(card_counts) | set(label_counts)):
        if card_counts[key] and counts[key] < min_words <= counts[key] + card_counts[key]:
            sums[key] = sums.get(key, 0.0) + card_sums[key]
            counts[key] += card_counts[key]
            log.info("season voices: %s from name cards (%d words)", key, card_counts[key])
        elif source == "clean" and card_counts[key] + label_counts[key] >= min_words:
            sums[key] = card_sums.get(key, 0.0) + label_sums.get(key, 0.0)
            counts[key] = card_counts[key] + label_counts[key]
            log.info("season voices: %s from name cards and named lines only (%d words)", key, counts[key])
    kept = sorted((k for k in counts if counts[k] >= min_words), key=lambda k: (-counts[k], k))
    vectors = np.stack([sums[k] / (np.linalg.norm(sums[k]) + 1e-9) for k in kept]) if kept else np.zeros((0, 0))
    log.info("season voices: %d people from %d episodes (%d with fewer than %d words left out)", len(kept), len(episodes),
             len(counts) - len(kept), min_words)
    return SeasonVoices(kept, vectors, [counts[k] for k in kept], model, [e.folder.name for e in episodes],
                        fingerprint(folders, model, min_words, cast, source),
                        {k: {f: v for f, v in cards[k][1].items() if f != "voice_bank"} for k in kept if k in cards},
                        aliases=dict(aliases or {}), cast=cast or {})


def load_cast(voices_path: Path | None) -> dict[str, Any]:
    """The season's cast (``<season>.cast.json`` beside the voices); empty without one."""
    if voices_path is None:
        return {}
    path = voices_path.with_name(voices_path.name.replace(".season_voices.npz", ".cast.json"))
    if not path.exists():
        return {}
    try:
        cast = load_json(path)
    except (OSError, ValueError):
        return {}
    return cast if isinstance(cast, dict) and isinstance(cast.get("cast"), list) else {}


def cast_aliases(voices_path: Path | None) -> dict[str, str]:
    """Lower-case alias -> full name from the season's cast list (``<season>.cast.json`` beside the voices)."""
    cast = load_cast(voices_path).get("cast") or []
    return {str(a).lower(): str(p["name"]) for p in cast if p.get("name") for a in p.get("aliases") or []}


def _group(person: dict[int, str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for i, key in person.items():
        out[key].append(i)
    return out


def ensure_voices(folders: list[Path], path: Path | None, model: str, min_words: int, *, save: bool = True,
                  source: str = "all") -> SeasonVoices:
    """The saved season voices when they were made from exactly these inputs, else built (and saved to ``path``)."""
    aliases, cast = cast_aliases(path), load_cast(path)
    wanted = fingerprint(folders, model, min_words, cast, source)
    if path is not None and path.exists():
        saved = SeasonVoices.load(path)
        if saved is not None and saved.fingerprint == wanted and saved.model == model:
            log.info("season voices: %s is up to date (%d people)", path.name, len(saved))
            saved.aliases, saved.cast = aliases, cast
            return saved
    voices = build_voices(folders, model, min_words, aliases, cast, source)
    voices.aliases, voices.cast = aliases, cast
    if save and path is not None:
        voices.save(path)
        log.info("season voices saved to %s", path)
    return voices


# ---------------------------------------------------------------- apply
def _weighted_embedding(cards: list[dict[str, Any]]) -> list[float] | None:
    """Mean of the cards' voice prints (by words), scaled like the largest card's print."""
    prints = [(max(c.get("words") or 0, 1), np.asarray(c["embedding"], dtype=np.float64)) for c in cards if c.get("embedding")]
    if not prints:
        return None
    units = [(w, p / (np.linalg.norm(p) + 1e-9)) for w, p in prints]
    mean = sum(w * u for w, u in units) / sum(w for w, _ in units)
    scale = float(np.linalg.norm(max(prints, key=lambda wp: wp[0])[1]))
    return [float(x) for x in mean / (np.linalg.norm(mean) + 1e-9) * scale]


def russian_names(folder: Path) -> dict[str, str]:
    """Lower-case English name -> Russian spelling from the episode's glossary (``stt.speakers.russian_names``)."""
    from stt.speakers import russian_names as names

    return names(folder)


def _cast_gender(voices: SeasonVoices, name: str) -> str | None:
    """The cast's gender of a person named only by the subtitles (None when the cast does not say, or for an identity)."""
    if is_identity(name):
        return None
    member = voices.own_member(name)
    gender = member.get("gender") if member else None
    return gender if gender in ("male", "female") else None


def _card_for(label: str, ids: dict[str, str], cues: list[Cue], old: EpisodeData, voices: SeasonVoices,
              labels_of: dict[str, list[str]], card_person: dict[str, str | None],
              own_labels: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """The speaker card of a new speaker (a person "V:<name>" or an unnamed cluster)."""
    own = [c for c in cues if c.speaker == label]
    person = label[len(STATE_PREFIX):] if label.startswith(STATE_PREFIX) else None
    if person is not None:
        mine = [c for c in old.cards if card_person[c["id"]] == person]
        base = max(mine, key=lambda c: c.get("words") or 0) if mine else voices.cards.get(person, {"gender": "unknown"})
        card = copy.deepcopy(base)
        identity = is_identity(person)
        card.update({"name": None if identity else person, "diarizer_label": label,
                     "diarizer_labels": labels_of.get(person, []), "season_voice": True})
        if identity:  # a voice without a name: the id is its only key
            card.update(identity=person, label_ru=None, name_source=None)
        if not identity and not mine and person not in voices.cards:  # named only by subtitles or name cards: no speaker card in the season
            gender = _cast_gender(voices, person)
            if gender and card.get("gender") not in ("male", "female"):
                card.update(gender=gender, gender_source="cast")
            card.update(name_source="season_voices", label_ru=russian_names(old.folder).get(person.lower()),
                        name_evidence="a season voice from named subtitle lines or name cards")
        card["on_screen"] = list(dict.fromkeys(s for c in mine for s in c.get("on_screen") or [])) or card.get("on_screen", [])
        embedding = _weighted_embedding(mine)
        if embedding is not None:
            card["embedding"] = embedding
    else:
        wanted = [label, *((own_labels or {}).get(label[len(OWN_STATE):], []) if label.startswith(OWN_STATE) else [])]
        base = next((c for c in old.cards if c.get("diarizer_label") in wanted), None)
        card = copy.deepcopy(base) if base else {"name": None, "gender": "unknown", "gender_source": None, "on_screen": []}
        card.update({"diarizer_label": label, "diarizer_labels": [label]})
        if label.startswith(OWN_STATE):  # a person only the subtitles name (the name is the cast's spelling when known)
            if not card.get("name"):
                name = label[len(OWN_STATE):]
                card.update(name=name, name_source="subtitles", label_ru=russian_names(old.folder).get(name.lower()),
                            name_evidence="the subtitles name this voice: " + ", ".join(
                                x[len(OWN_STATE):] for x in wanted[1:] or wanted))
            gender = _cast_gender(voices, label[len(OWN_STATE):])
            if gender and card.get("gender") not in ("male", "female"):
                card.update(gender=gender, gender_source="cast")
    card.update({
        "id": ids[label], "same_as": None, "cues": len(own), "words": sum(len(c.words) for c in own),
        "first_seen": round(own[0].start + old.offset, 3) if own else None, "sample": own[0].text[:120] if own else None,
    })
    return card


def _tribe_bias(old: EpisodeData, voices: SeasonVoices, prior: float) -> dict[str, np.ndarray]:
    """State -> per-word score change of the season's people at the camps of other tribes (``stt.tribes``)."""
    if prior <= 0 or not voices.cast or not (old.folder / "ocr.json").exists():
        return {}
    from stt.tribes import tribe_penalties

    try:
        cards = list(load_json(old.folder / "ocr.json").get("cards") or [])
    except (OSError, ValueError, AttributeError):
        return {}
    members = {voices.person_of(str(p.get("name"))): str(p.get("name")) for p in voices.cast.get("cast") or [] if p.get("name")}
    people = {members[name]: name for name in voices.names if name in members}  # cast name -> season person
    changes = tribe_penalties(old.words, cards, voices.cast, list(people), prior)
    if changes:
        log.info("[%s] tribe camps: %d people less likely at other tribes' camps", old.folder.name, len(changes))
    return {STATE_PREFIX + people[name]: vec for name, vec in changes.items()}


def _overlap_s(a: tuple[float, float], b: tuple[float, float]) -> float:
    return min(a[1], b[1]) - max(a[0], b[0])


def _line_genders(records: list[dict[str, Any]], old_records: list[dict[str, Any]], cards: list[dict[str, Any]]) -> None:
    """Cues of a speaker whose gender stays unknown keep the gender ``stt.line_gender`` heard for the old cue there
    (it needs the GPU; the lines are the same audio)."""
    unknown = {c["id"] for c in cards if c.get("gender") not in ("male", "female")}
    heard = [(r["start"], r["end"], r["gender"]) for r in old_records if r.get("gender") in ("male", "female")]
    for record in records:
        if record.get("speaker") not in unknown:
            continue
        best = max(heard, key=lambda h: _overlap_s((record["start"], record["end"]), (h[0], h[1])), default=None)
        if best and _overlap_s((record["start"], record["end"]), (best[0], best[1])) > 0.5 * record["duration"]:
            record["gender"] = best[2]


@dataclass
class Result:
    status: str  # applied | unchanged | skipped
    reason: str = ""
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.status == "applied"


def _tally(states: list[str | None], words: list[Word]) -> dict[str, int]:
    count: Counter[str] = Counter(s for s, w in zip(states, words) if s is not None)
    return {(s[len(STATE_PREFIX):] if s.startswith(STATE_PREFIX) else f"cluster {s}"): n for s, n in count.most_common()}


def apply_episode(folder: Path, voices: SeasonVoices, *, model: str = "resnet293", dry_run: bool = False,
                  options: CueOptions | None = None, caption_bonus: float = TURN_BONUS, tribe_prior: float = 0.0,
                  dissolve: str = "none") -> Result:
    """Assign the words of one episode again over the season's people and rewrite its transcript (see the module doc).
    ``caption_bonus``: the reward of a switch at a ">>" caption mark (a diarizer change gets ``TURN_BONUS``).
    ``tribe_prior``: how much less likely a person is at the camp of a tribe they were never in (``stt.tribes``; 0: off).
    ``dissolve``: unnamed clusters that are no state of their own, so their words go to the season's people: "mixed" (the
    ones the naming calls a mix of voices), "unnamed" (all of them) or "none"."""
    from stt.speakers import update_records
    from stt.transcribe import assemble_cues, write_transcript

    options = options or CueOptions()
    if not len(voices):
        return Result("skipped", "no named people in the season")
    old = load_episode(folder, model)
    if old is None:
        return Result("skipped", "no first pass (words, speakers, word embeddings)")
    diarization = find_diarization(folder, old.cards)
    if diarization is None:
        return Result("skipped", "no diarization that matches the speaker cards")
    n = len(old.words)
    if n == 0:
        return Result("skipped", "no words")

    # the first pass's state of every word: a person with a voice in the season, else the diarizer cluster
    card_person = {c["id"]: voices.person_of(c.get("name")) or voices.person_of(c.get("identity")) for c in old.cards}
    state_of_card: dict[str, str] = {}
    for c in old.cards:
        person = card_person[c["id"]]
        state_of_card[c["id"]] = STATE_PREFIX + person if person else str(c["diarizer_label"])
    labels_of: dict[str, list[str]] = defaultdict(list)
    named_clusters: set[str] = set()
    for card in old.cards:
        person = card_person[card["id"]]
        if person and card.get("cues"):
            named_clusters.add(str(card["diarizer_label"]))
            labels_of[person].append(str(card["diarizer_label"]))
    old_state: list[str | None] = [state_of_card.get(raw.get("speaker")) for raw in old.raw_words]

    own = [w.speaker for w in assign_speakers(old.words, diarization, options)]
    clusters = _centroids(old.words, own, old.emb, RelabelParams())
    states: dict[str, np.ndarray] = {STATE_PREFIX + name: voices.vectors[j] for j, name in enumerate(voices.names)}
    from stt.speakers import _is_mixed

    gone = {str(c["diarizer_label"]) for c in old.cards if not card_person[c["id"]]
            and (dissolve == "unnamed" or (dissolve == "mixed" and _is_mixed(c)))}
    states.update({label: vec for label, vec in clusters.items() if label not in named_clusters and label not in gone})
    fixed: dict[int, str] = {}
    hints = subtitle_hints(old)
    own_lines: dict[str, list[int]] = defaultdict(list)  # a name only the subtitles give -> its words
    own_labels: dict[str, list[str]] = defaultdict(list)  # ... and the first pass's "P:<LABEL>" cards of it
    for label, idx in (hints.persons.items() if hints else ()):
        who = voices.by_first_name(label)
        usable = [i for i in idx if old.words[i].source != "ocr"]
        if who is not None:
            fixed.update({i: STATE_PREFIX + who for i in usable})
        else:  # spellings of one person ("SUE", "SUSAN") share the state, named by the cast when it knows them
            name = voices.own_name(label)
            own_lines[name] += usable
            own_labels[name].append(OWN_STATE + label)
    for name, usable in own_lines.items():
        if len(usable) >= WordHints.min_words:  # the subtitles name someone the first pass never did: they keep their lines
            total = old.emb[usable].sum(axis=0)
            states[OWN_STATE + name] = total / (np.linalg.norm(total) + 1e-9)
            fixed.update({i: OWN_STATE + name for i in usable})
    turns = {i: TURN_BONUS for i in range(1, n) if own[i] and own[i - 1] and own[i] != own[i - 1]}
    for i in (hints.turns if hints else ()):  # the ">>" marks of captions; the larger bonus where both say so
        turns[i] = max(turns.get(i, 0.0), caption_bonus)
    params = RelabelParams(switch_penalty=SWITCH_PENALTY, turn_bonus=TURN_BONUS)
    bias = _tribe_bias(old, voices, tribe_prior)
    path = _viterbi(old.words, [None] * n, old.emb, states, params, fixed=fixed, turns=turns, bias=bias)
    for i, word in enumerate(old.words):
        if word.source == "ocr":
            path[i] = old_state[i]  # on-screen captions keep the speaker they had (the viterbi pass leaves them open)
    words = [replace(w, speaker=s) for w, s in zip(old.words, path)]
    words = smooth_speakers(words, options)
    words = restore_fixed(words, fixed)  # smoothing must not undo a named line

    def gender_of(label: str) -> str:
        person = label[len(STATE_PREFIX):] if label.startswith(STATE_PREFIX) else None
        mine = [c for c in old.cards if person and card_person[c["id"]] == person]
        base = (max(mine, key=lambda c: c.get("words") or 0) if mine
                else voices.cards.get(person or "") or next((c for c in old.cards if c.get("diarizer_label") == label), {}))
        gender = str(base.get("gender") or "unknown")
        if gender not in ("male", "female") and label.startswith(OWN_STATE):  # the cast knows the person's gender
            gender = _cast_gender(voices, label[len(OWN_STATE):]) or gender
        if gender not in ("male", "female") and person:
            gender = _cast_gender(voices, person) or gender
        return gender

    cues, ids, records = assemble_cues(words, diarization, options, offset=old.offset, gender_of=gender_of)
    cards = [_card_for(label, ids, cues, old, voices, labels_of, card_person, own_labels) for label, _ in sorted(ids.items(), key=lambda kv: kv[1])]
    update_records(records, cards)
    _line_genders(records, old.records, cards)

    final = [w.speaker for w in words]
    before, after = _tally(old_state, old.words), _tally(final, words)
    old_majority = _old_cue_states(cues, words, old_state)
    report = {
        "schema": SCHEMA,
        "model": model,
        "voices": {"people": len(voices), "episodes": voices.episodes, "fingerprint": voices.fingerprint},
        "words": n,
        "words_moved": sum(a != b for a, b in zip(old_state, final)),
        "cues_before": len(old.records),
        "cues_after": len(records),
        "cues_changed_speaker": sum(a != c.speaker for a, c in zip(old_majority, cues)),
        "speakers_before": len([c for c in old.cards if c.get("cues")]),
        "speakers_after": len(cards),
        "people": sum(c.get("season_voice", False) for c in cards),
        "per_person": {name: {"before": before.get(name, 0), "after": after.get(name, 0)} for name in sorted({*before, *after})},
    }
    if _same_as_current(folder, words, ids, records, cards):
        return Result("unchanged", "the transcript is already what the stage gives", report)
    if dry_run:
        return Result("would-apply", report=report)
    result = Result("applied", report=report)
    if (folder / "corrections.json").exists():  # speakers.override.json uses first-pass ids on purpose (naming stage)
        log.warning("[%s] corrections.json refers to the line numbers of the first pass; check it", folder.name)
    if not old.original:  # the files in the folder are the first pass: keep them once
        for name in FILES:
            shutil.copy2(old.sources[name], before_path(folder, name))
    write_transcript(folder, records, words, ids, old.offset)
    save_json(folder / "speakers.json", cards)
    result.report["written"] = {n: _sha(folder / f"{n}.json") for n in FILES}
    save_json(folder / REPORT, result.report)
    return result


def _old_cue_states(cues: list[Cue], words: list[Word], old_state: list[str | None]) -> list[str | None]:
    """For each new cue, the first-pass state most of its words had."""
    out: list[str | None] = []
    position = {id(w): i for i, w in enumerate(words)}
    for cue in cues:
        votes = Counter(old_state[position[id(w)]] for w in cue.words if id(w) in position)
        out.append(votes.most_common(1)[0][0] if votes else None)
    return out


def _same_as_current(folder: Path, words: list[Word], ids: dict[str, str], records: list[dict[str, Any]],
                     cards: list[dict[str, Any]]) -> bool:
    """Whether the files in the folder already hold exactly this transcript (nothing to write, no marks to touch)."""
    try:
        return (load_json(folder / "subs.en.json") == json.loads(json.dumps(records))
                and load_json(folder / "speakers.json") == json.loads(json.dumps(cards))
                and [w.get("speaker") for w in load_json(folder / "words.json")]
                == [ids.get(w.speaker) if w.speaker else None for w in words])
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------- season
def season_folders(root: Path, pattern: str | None = None) -> list[Path]:
    """Episode folders under ``root`` (the ones with a transcript); ``pattern``: a glob of the folder names."""
    found = sorted(p for p in root.iterdir() if p.is_dir() and (p / "words.json").exists() and (p / "speakers.json").exists())
    return [p for p in found if not pattern or fnmatch.fnmatch(p.name, pattern)]


def prepare_embeddings(folders: list[Path], model: str, device: str = "cuda") -> None:
    """Word embeddings of ``model`` for the episodes that lack them (another model than the first pass: uses the GPU,
    30-45 s per episode). The model is loaded only when something is missing."""
    from stt.audio import read_wav
    from stt.boundaries import RelabelParams, _times, embed_windows, word_windows
    from stt.refine_speakers import make_embedder

    embedder: Any = None
    window = RelabelParams().window_s
    for folder in folders:
        sources, _ = first_pass_files(folder)
        path, audio = embedding_path(folder, model), folder / "audio.wav"
        if not sources["words"].exists() or not audio.exists():
            continue
        offset = _offset(folder)
        words = [Word(w["start"] - offset, w["end"] - offset, w["text"]) for w in load_json(sources["words"])]
        if path.exists():
            if not _stale_embeddings(path, words):
                continue
            log.info("[%s] %s belongs to an earlier transcript: computed again", folder.name, path.name)
        embedder = embedder or make_embedder(model, device)
        emb = embed_windows(word_windows(words, read_wav(audio), window), embedder)
        np.savez(path, starts=_times(words, "start"), ends=_times(words, "end"), emb=emb.astype(np.float32),
                 window_s=np.float64(window))
        log.info("[%s] word embeddings of %s written", folder.name, model)


def apply_season(all_folders: list[Path], targets: list[Path], voices_path: Path | None, *, model: str = "resnet293",
                 min_words: int = MIN_WORDS, dry_run: bool = False, caption_bonus: float = TURN_BONUS,
                 tribe_prior: float = 0.0, dissolve: str = "none", voice_source: str = "all") -> dict[str, Result]:
    """Build (or reuse) the season voices from ``all_folders`` and apply them to ``targets``. Returns a Result per folder.
    A dry run computes no word embeddings (episodes without them are left out) and writes nothing."""
    if not dry_run:
        prepare_embeddings(all_folders, model)
    voices = ensure_voices(all_folders, voices_path, model, min_words, save=not dry_run, source=voice_source)
    results: dict[str, Result] = {}
    for folder in targets:
        try:
            results[folder.name] = apply_episode(folder, voices, model=model, dry_run=dry_run, caption_bonus=caption_bonus,
                                                 tribe_prior=tribe_prior, dissolve=dissolve)
        except Exception as error:  # noqa: BLE001 - one episode must not stop the season; the first pass stays
            log.exception("[%s] season voices failed", folder.name)
            results[folder.name] = Result("skipped", f"error: {error}")
    return results


def summary(name: str, result: Result) -> str:
    r = result.report
    if not r:
        return f"{name}: {result.status} ({result.reason})"
    return (f"{name}: {result.status}; {r['words_moved']} of {r['words']} words moved, {r['cues_changed_speaker']} cues changed "
            f"speaker, cues {r['cues_before']} -> {r['cues_after']}, speakers {r['speakers_before']} -> {r['speakers_after']} "
            f"({r['people']} people)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Assign the words of a season again over the season's named people")
    parser.add_argument("root", type=Path, help="Work folder of the season (one sub-folder per episode)")
    parser.add_argument("--episodes", default=None, help="Glob of the episode folder names to rewrite (voices: all)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would move; write nothing")
    parser.add_argument("--model", default="resnet293", help="Word embedding model (words.emb.<model>.npz)")
    parser.add_argument("--min-words", type=int, default=MIN_WORDS, help="Words a person needs over the season")
    parser.add_argument("--caption-bonus", type=float, default=TURN_BONUS,
                        help="Reward of a speaker switch at a '>>' caption mark (a diarizer change gets %(default)s)")
    parser.add_argument("--voice-source", choices=["all", "clean"], default="all",
                        help="clean: a person's voice from the words under their name cards and named lines when enough")
    parser.add_argument("--dissolve", choices=["none", "mixed", "unnamed"], default="none",
                        help="Unnamed clusters whose words go to the season's people: the mixed ones, all, or none")
    parser.add_argument("--tribe-prior", type=float, default=0.0,
                        help="How much less likely a person is at another tribe's camp (stt.tribes); 0: off")
    parser.add_argument("--voices", type=Path, default=None, help="Where the season voices are saved (default: <root>/<root name>.season_voices.npz)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    root = args.root
    if not root.is_dir():
        raise SystemExit(f"Not a folder: {root}")
    everything = season_folders(root)
    targets = season_folders(root, args.episodes)
    if not targets:
        raise SystemExit(f"No episode folders with words.json and speakers.json in {root}")
    path = args.voices or root / f"{root.resolve().name}.season_voices.npz"
    for name, result in apply_season(everything, targets, path, model=args.model, min_words=args.min_words,
                                     dry_run=args.dry_run, caption_bonus=args.caption_bonus,
                                     tribe_prior=args.tribe_prior, dissolve=args.dissolve,
                                     voice_source=args.voice_source).items():
        print(summary(name, result))  # CLI output
    if args.dry_run:
        print("dry run: nothing written")


if __name__ == "__main__":
    main()
