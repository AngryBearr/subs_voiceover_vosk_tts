"""«Кто это?»: speaker groups of an episode that have no name, short snippets to listen to, and the answer written to
``speakers.override.json`` (the file ``stt.speakers`` applies on the next naming run).

Pure functions over an episode work folder (``speakers.json``, ``speakers.llm.json``, ``subs.en.json``, ``audio.wav``)
and the season's ``<season>.cast.json``. The override entry of a person is ``{"name": ..., "gender": ...}``; a mixed
group is ``{"mixed": true}`` (no name: it keeps the generic voice but is not offered again).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from utils.atomic_io import write_text_atomic

MIN_CUES = 5  # smaller groups are noise, not worth a question
MIN_SNIPPET_S, MAX_SNIPPET_S = 2.0, 6.0
SNIPPETS = 5
APART_S = 15.0  # snippets of one group start at least this far from each other when possible
GENDERS = ("male", "female")
OVERRIDE = "speakers.override.json"
MIXED = "mixed"


@dataclass
class Snippet:
    start: float  # seconds in audio.wav
    end: float
    text: str
    wav: Path | None = None


@dataclass
class Group:
    episode: str  # work folder name
    speaker: str  # "S02"
    cues: int
    seconds: float
    role: str = ""
    gender: str = "unknown"
    gender_evidence: str = ""
    snippets: list[Snippet] = field(default_factory=list)


def _load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def load_overrides(folder: Path) -> dict[str, dict[str, Any]]:
    data = _load(folder / OVERRIDE, {})
    return data if isinstance(data, dict) else {}


def cast_names(cast_path: Path) -> list[str]:
    """Names of the season cast, the host first (the host speaks in every episode)."""
    from stt.cast import host_member

    cast = _load(cast_path, {})
    if not isinstance(cast, dict):
        return []
    members = [str(p["name"]) for p in cast.get("cast", []) if isinstance(p, dict) and p.get("name")]
    host = host_member(cast)
    if host and host.get("name") in members:
        members.remove(host["name"])
        members.insert(0, str(host["name"]))
    return members


def cast_gender(cast_path: Path, name: str) -> str | None:
    cast = _load(cast_path, {})
    for person in cast.get("cast", []) if isinstance(cast, dict) else []:
        if isinstance(person, dict) and person.get("name") == name and person.get("gender") in GENDERS:
            return str(person["gender"])
    return None


def _record_seconds(record: dict[str, Any]) -> float:
    return max(0.0, (record.get("end", 0) - record.get("start", 0)) / 1000.0)


def pick_snippets(records: list[dict[str, Any]], speaker: str, count: int = SNIPPETS) -> list[Snippet]:
    """The longest clean cues of ``speaker`` (2-6 s, no overlap, not on-screen text, nobody else talking at the same
    time), spread over the episode; cues longer than 6 s are cut to 6 s when too few fit."""
    others = [(r["start"], r["end"]) for r in records if r.get("speaker") != speaker]
    mine = [r for r in records if r.get("speaker") == speaker and not r.get("overlap") and not r.get("ocr")]

    def clean(r: dict[str, Any]) -> bool:
        return not any(s < r["end"] and e > r["start"] for s, e in others)

    def text_of(r: dict[str, Any]) -> str:
        t = r.get("text")
        return " ".join(t) if isinstance(t, list) else str(t or "")

    def best(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(pool, key=lambda r: (-min(_record_seconds(r), MAX_SNIPPET_S), r["start"]))

    fit = [r for r in mine if clean(r) and MIN_SNIPPET_S <= _record_seconds(r) <= MAX_SNIPPET_S and text_of(r).strip()]
    long = [r for r in mine if clean(r) and _record_seconds(r) > MAX_SNIPPET_S and text_of(r).strip()]
    chosen: list[dict[str, Any]] = []
    for pool in (best(fit), best(long)):
        for r in pool:  # first spread out, then anything left
            if len(chosen) < count and all(abs(r["start"] - c["start"]) / 1000 >= APART_S for c in chosen):
                chosen.append(r)
        for r in pool:
            if len(chosen) < count and r not in chosen:
                chosen.append(r)
    chosen.sort(key=lambda r: r["start"])
    return [Snippet(r["start"] / 1000.0, r["start"] / 1000.0 + min(_record_seconds(r), MAX_SNIPPET_S), text_of(r))
            for r in chosen]


def unnamed_groups(folder: Path, min_cues: int = MIN_CUES) -> list[Group]:
    """Groups of the episode with no name, enough lines and no answer yet (no override entry), biggest first;
    snippets are chosen but not cut."""
    speakers = _load(folder / "speakers.json", [])
    if not isinstance(speakers, list):
        return []
    records = _load(folder / "subs.en.json", [])
    llm = _load(folder / "speakers.llm.json", {})
    cards = {c.get("id"): c for c in llm.get("cards", []) if isinstance(c, dict)} if isinstance(llm, dict) else {}
    overrides = load_overrides(folder)
    groups = []
    for s in speakers:
        if s.get("name") or (s.get("cues") or 0) < min_cues or s.get("id") in overrides:
            continue
        mine = [r for r in records if r.get("speaker") == s["id"]]
        card = cards.get(s["id"], {})
        groups.append(Group(
            folder.name, s["id"], int(s.get("cues") or 0), round(sum(_record_seconds(r) for r in mine), 1),
            str(card.get("role") or s.get("role") or ""), str(s.get("gender") or "unknown"),
            str(card.get("gender_evidence") or s.get("gender_evidence") or ""), pick_snippets(records, s["id"])))
    return sorted(groups, key=lambda g: -g.seconds)


def cut_snippets(group: Group, folder: Path, out_dir: Path) -> None:
    """Writes each snippet of ``group`` from ``folder/audio.wav`` as a 16-bit wav in ``out_dir`` and sets ``wav``
    (a snippet that cannot be read keeps ``wav`` None)."""
    audio = folder / "audio.wav"
    if not audio.is_file():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.-]+", "_", folder.name)
    try:
        with sf.SoundFile(str(audio)) as src:
            rate = src.samplerate
            for n, snip in enumerate(group.snippets, 1):
                src.seek(min(int(snip.start * rate), src.frames))
                data = src.read(int((snip.end - snip.start) * rate), dtype="float32", always_2d=True)
                if not len(data):
                    continue
                target = out_dir / f"{safe}.{group.speaker}.{n}.wav"
                sf.write(str(target), np.clip(data.mean(axis=1), -1.0, 1.0), rate, subtype="PCM_16")
                snip.wav = target
    except (OSError, RuntimeError):
        return


def episode_groups(results: Path, episodes: list[str] | None = None, cut_to: Path | None = None) -> list[Group]:
    """Unnamed groups of the season's episodes (all work folders with a ``speakers.json``, or the listed ones);
    with ``cut_to`` the snippets are cut there."""
    found: list[Group] = []
    for speakers_json in sorted(results.glob("*/speakers.json")):
        folder = speakers_json.parent
        if episodes and folder.name not in episodes:
            continue
        for group in unnamed_groups(folder):
            if cut_to is not None:
                cut_snippets(group, folder, cut_to)
            found.append(group)
    return found


def save_answer(folder: Path, speaker: str, name: str | None = None, gender: str | None = None, mixed: bool = False) -> Path:
    """Writes the answer into ``speakers.override.json`` (other entries stay): a person -> ``{"name", "gender"?}``,
    a mixed group -> ``{"mixed": true}``. Raises ValueError for nothing to write."""
    if gender not in GENDERS:
        gender = None
    if mixed:
        entry: dict[str, Any] = {MIXED: True}
        if gender:
            entry["gender"] = gender
    elif name and name.strip():
        entry = {"name": name.strip()}
        if gender:
            entry["gender"] = gender
    else:
        raise ValueError("no name and not marked as mixed")
    overrides = load_overrides(folder)
    overrides[speaker] = entry
    path = folder / OVERRIDE
    write_text_atomic(path, json.dumps(overrides, ensure_ascii=False, indent=2) + "\n")
    return path


def episode_stage(results: Path, folder: Path) -> str:
    """"voiced" (a dub exists), "translated" or "early" (not translated yet)."""
    if any(folder.glob("*.ru.*.m4a")) or (results / f"{folder.name}.ru.mkv").exists() \
            or any(results.glob(f"{folder.name}.ru.*mkv")):
        return "voiced"
    return "translated" if (folder / "subs.ru.json").exists() else "early"


def redo_command(video: Path, results: Path) -> str:
    return (f'subs_env/bin/python -m stt.run "{video}" --out-root "{results}" --preset upload '
            f"--engine f5,teratts,vosk --force speakers")
