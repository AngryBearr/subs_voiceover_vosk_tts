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
pass are reused (``words.emb.<model>.npz``). Off by default (``season_voices`` in ``stt_config.json``).

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
    """Raw speaker name -> person key. Spellings of one name (``voice_bank.same_person``: "Jon"/"John Misch", "Jeff"/"Jeff
    Probst") are one person, keyed by the spelling with most words (the longer one on a tie)."""
    from stt.voice_bank import same_person

    keys: list[str] = []
    out: dict[str, str] = {}
    for name in sorted(counts, key=lambda n: (-counts[n], -len(n), n)):
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

    def __len__(self) -> int:
        return len(self.names)

    def person_of(self, name: str | None) -> str | None:
        """The person (key) a speaker name belongs to, or None when the season has no such person."""
        if not name:
            return None
        from stt.voice_bank import same_person

        name = clean_name(name)
        if name in self.names:
            return name
        order = sorted(range(len(self.names)), key=lambda j: -self.counts[j])
        return next((self.names[j] for j in order if same_person(name, self.names[j])), None)

    def by_first_name(self, label: str) -> str | None:
        """The one person whose first name is ``label``'s (case-insensitive); None when none or several."""
        from stt.subtitles import HOST, HOST_LABELS

        if label == HOST:
            found = [n for n in self.names if n.upper() in HOST_LABELS or n.upper().split()[0] in HOST_LABELS]
        else:
            first = label.lower().split()[0] if label.split() else ""
            found = [n for n in self.names if n.lower().split()[0] == first] if first else []
        if not found and label != HOST:  # subtitles also use surnames and nicknames: "KILBY", "JELLY"
            wanted = label.lower()
            person = self.aliases.get(wanted)
            found = [person] if person in self.names else [n for n in self.names if wanted in n.lower().split()]
        return found[0] if len(found) == 1 else None

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
def fingerprint(folders: list[Path], model: str, min_words: int) -> str:
    """What the voices depend on: the first-pass files, the word embeddings and the subtitles of every episode."""
    from stt.refine_speakers import model_cache_key

    digest = hashlib.sha256(f"{SCHEMA}|{model_cache_key(model)}|{min_words}".encode())
    for folder in sorted(folders):
        sources, _ = first_pass_files(folder)
        parts = [sources["words"], sources["speakers"], embedding_path(folder, model), folder / "subs.source.en.srt"]
        digest.update(folder.name.encode())
        for path in parts:
            digest.update((_sha(path) if path.exists() else "-").encode())
    return digest.hexdigest()


def build_voices(folders: list[Path], model: str = "resnet293", min_words: int = MIN_WORDS,
                 aliases: dict[str, str] | None = None) -> SeasonVoices:
    """The season's people from the named speakers (and named subtitle lines) of the episodes with a first pass."""
    episodes = [e for e in (load_episode(f, model) for f in folders) if e is not None]
    totals: Counter[str] = Counter()
    for episode in episodes:
        for name, n in Counter(n for n, w in zip(episode.word_names(), episode.words) if n and w.source != "ocr").items():
            totals[name] += n
    key_of = group_names(dict(totals))
    keys = SeasonVoices(sorted(set(key_of.values())), np.zeros((0, 0)), [], model, aliases=dict(aliases or {}))
    sums: dict[str, np.ndarray] = {}
    counts: Counter[str] = Counter()
    cards: dict[str, tuple[int, dict[str, Any]]] = {}
    for episode in episodes:
        person: dict[int, str] = {}
        for i, (name, word) in enumerate(zip(episode.word_names(), episode.words)):
            if name and word.source != "ocr":
                person[i] = key_of[name]
        hints = subtitle_hints(episode)
        for label, idx in (hints.persons.items() if hints else ()):
            key = keys.by_first_name(label)
            if key is None:
                continue
            for i in idx:
                if episode.words[i].source != "ocr":
                    person[i] = key  # a named subtitle line beats the speaker's name
        for key, idx in _group(person).items():
            sums[key] = sums.get(key, 0.0) + episode.emb[idx].sum(axis=0)
            counts[key] += len(idx)
        for card in episode.cards:
            key = key_of.get(clean_name(card["name"])) if card.get("name") else None
            if key and (card.get("words") or 0) > cards.get(key, (-1, {}))[0]:
                cards[key] = (card.get("words") or 0, card)
    kept = sorted((k for k in counts if counts[k] >= min_words), key=lambda k: (-counts[k], k))
    vectors = np.stack([sums[k] / (np.linalg.norm(sums[k]) + 1e-9) for k in kept]) if kept else np.zeros((0, 0))
    log.info("season voices: %d people from %d episodes (%d with fewer than %d words left out)", len(kept), len(episodes),
             len(counts) - len(kept), min_words)
    return SeasonVoices(kept, vectors, [counts[k] for k in kept], model, [e.folder.name for e in episodes],
                        fingerprint(folders, model, min_words),
                        {k: {f: v for f, v in cards[k][1].items() if f != "voice_bank"} for k in kept if k in cards})


def cast_aliases(voices_path: Path | None) -> dict[str, str]:
    """Lower-case alias -> full name from the season's cast list (``<season>.cast.json`` beside the voices)."""
    if voices_path is None:
        return {}
    path = voices_path.with_name(voices_path.name.replace(".season_voices.npz", ".cast.json"))
    if not path.exists():
        return {}
    try:
        cast = load_json(path).get("cast") or []
    except (OSError, ValueError, AttributeError):
        return {}
    return {str(a).lower(): str(p["name"]) for p in cast if p.get("name") for a in p.get("aliases") or []}


def _group(person: dict[int, str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for i, key in person.items():
        out[key].append(i)
    return out


def ensure_voices(folders: list[Path], path: Path | None, model: str, min_words: int, *, save: bool = True) -> SeasonVoices:
    """The saved season voices when they were made from exactly these inputs, else built (and saved to ``path``)."""
    wanted = fingerprint(folders, model, min_words)
    aliases = cast_aliases(path)
    if path is not None and path.exists():
        saved = SeasonVoices.load(path)
        if saved is not None and saved.fingerprint == wanted and saved.model == model:
            log.info("season voices: %s is up to date (%d people)", path.name, len(saved))
            saved.aliases = aliases
            return saved
    voices = build_voices(folders, model, min_words, aliases)
    voices.aliases = aliases
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


def _card_for(label: str, ids: dict[str, str], cues: list[Cue], old: EpisodeData, voices: SeasonVoices,
              labels_of: dict[str, list[str]], card_person: dict[str, str | None]) -> dict[str, Any]:
    """The speaker card of a new speaker (a person "V:<name>" or an unnamed cluster)."""
    own = [c for c in cues if c.speaker == label]
    person = label[len(STATE_PREFIX):] if label.startswith(STATE_PREFIX) else None
    if person is not None:
        mine = [c for c in old.cards if card_person[c["id"]] == person]
        base = max(mine, key=lambda c: c.get("words") or 0) if mine else voices.cards.get(person, {"gender": "unknown"})
        card = copy.deepcopy(base)
        card.update({"name": person, "diarizer_label": label, "diarizer_labels": labels_of.get(person, []),
                     "season_voice": True})
        card["on_screen"] = list(dict.fromkeys(s for c in mine for s in c.get("on_screen") or [])) or card.get("on_screen", [])
        embedding = _weighted_embedding(mine)
        if embedding is not None:
            card["embedding"] = embedding
    else:
        base = next((c for c in old.cards if c.get("diarizer_label") == label), None)
        card = copy.deepcopy(base) if base else {"name": None, "gender": "unknown", "gender_source": None, "on_screen": []}
        card.update({"diarizer_label": label, "diarizer_labels": [label]})
        if label.startswith(OWN_STATE) and not card.get("name"):
            card["name"] = label[len(OWN_STATE):].title()
    card.update({
        "id": ids[label], "same_as": None, "cues": len(own), "words": sum(len(c.words) for c in own),
        "first_seen": round(own[0].start + old.offset, 3) if own else None, "sample": own[0].text[:120] if own else None,
    })
    return card


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
                  options: CueOptions | None = None) -> Result:
    """Assign the words of one episode again over the season's people and rewrite its transcript (see the module doc)."""
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
    card_person = {c["id"]: voices.person_of(c.get("name")) for c in old.cards}
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
    states.update({label: vec for label, vec in clusters.items() if label not in named_clusters})
    fixed: dict[int, str] = {}
    hints = subtitle_hints(old)
    for label, idx in (hints.persons.items() if hints else ()):
        who = voices.by_first_name(label)
        usable = [i for i in idx if old.words[i].source != "ocr"]
        if who is not None:
            fixed.update({i: STATE_PREFIX + who for i in usable})
        elif len(usable) >= WordHints.min_words:  # the subtitles name someone the first pass never did: they keep their lines
            total = old.emb[usable].sum(axis=0)
            states[OWN_STATE + label] = total / (np.linalg.norm(total) + 1e-9)
            fixed.update({i: OWN_STATE + label for i in usable})
    turns = {i for i in range(1, n) if own[i] and own[i - 1] and own[i] != own[i - 1]}
    turns |= set(hints.turns) if hints else set()  # the ">>" marks of captions
    params = RelabelParams(switch_penalty=SWITCH_PENALTY, turn_bonus=TURN_BONUS)
    path = _viterbi(old.words, [None] * n, old.emb, states, params, fixed=fixed, turns=turns)
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
        return str(base.get("gender") or "unknown")

    cues, ids, records = assemble_cues(words, diarization, options, offset=old.offset, gender_of=gender_of)
    cards = [_card_for(label, ids, cues, old, voices, labels_of, card_person) for label, _ in sorted(ids.items(), key=lambda kv: kv[1])]
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
    for extra in ("corrections.json", "speakers.override.json"):
        if (folder / extra).exists():
            log.warning("[%s] %s refers to the numbering of the first pass (line numbers / speaker ids); check it", folder.name, extra)
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
                 min_words: int = MIN_WORDS, dry_run: bool = False) -> dict[str, Result]:
    """Build (or reuse) the season voices from ``all_folders`` and apply them to ``targets``. Returns a Result per folder."""
    prepare_embeddings(all_folders, model)
    voices = ensure_voices(all_folders, voices_path, model, min_words, save=not dry_run)
    results: dict[str, Result] = {}
    for folder in targets:
        try:
            results[folder.name] = apply_episode(folder, voices, model=model, dry_run=dry_run)
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
                                     dry_run=args.dry_run).items():
        print(summary(name, result))  # CLI output
    if args.dry_run:
        print("dry run: nothing written")


if __name__ == "__main__":
    main()
