"""Season voice bank: voice prints of the named people of a season, shared by its episodes.

Each episode adds one print per named speaker cluster: the pyannote (WeSpeaker) embedding of
the cluster with its name, Russian label and gender. A later episode compares its clusters
with the prints of the other episodes:

- the closest names (cosine similarity) go to the speaker-naming LLM as evidence: people
  who never say their name in an episode can still be named; the LLM keeps the last word,
  because a print can carry an earlier naming mistake;
- a cluster the LLM leaves unnamed takes a name that matches strongly and clearly;
- a strong match the LLM contradicts is reported as a conflict (one of the two names is
  wrong, or the cluster holds two voices).

Same person across S29E01/E02: similarity 0.67-0.98 (a host's narration and on-location voice
are two clusters, 0.70-0.82); different people mostly under 0.5, twins 0.67. The bank also
lists, per person, the cleanest stretches of solo speech (6-12 s): reference clips for voice
cloning.

    uv run -m stt.voice_bank output/stt_s29/<season>.voices.json   # what the bank holds
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

SCHEMA = "stt-voice-bank-v1"
EVIDENCE = 0.6  # matches from this similarity on are shown to the LLM
STRONG = 0.75  # an unnamed cluster takes the name from this similarity on ...
MARGIN = 0.1  # ... when the next other name is at least this much lower
MIN_SPEECH_S = 20.0  # clusters with less speech give unreliable prints
MIN_CONFIDENCE = 0.6  # LLM naming confidence needed to add a print
SAMPLE_MIN_S, SAMPLE_MAX_S, SAMPLE_GAP_S = 6.0, 12.0, 0.4
SAMPLES_PER_EPISODE = 3


def _similar(a: str, b: str, threshold: float) -> bool:
    return difflib.SequenceMatcher(a=a, b=b).ratio() >= threshold


def same_person(a: str, b: str) -> bool:
    """Spellings of one name: "Jon Misch" / "John Misch", "Baylor" / "Baylor Wilson" (not "Nadiya" / "Natalie")."""
    ta, tb = re.findall(r"[^\W\d_]+", a.lower()), re.findall(r"[^\W\d_]+", b.lower())
    if not ta or not tb or not _similar(ta[0], tb[0], 0.75):
        return False
    if len(ta) == 1 or len(tb) == 1:
        return True
    return _similar(ta[-1], tb[-1], 0.85)


def _unit(vector: list[float]) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def reference_samples(speaker: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stretches of one speaker's solo speech (consecutive cues, no overlap, no burned-in caption)."""
    out: list[dict[str, Any]] = []
    run: list[dict[str, Any]] = []

    def close() -> None:
        if run:
            span = (run[-1]["end"] - run[0]["start"]) / 1000
            if SAMPLE_MIN_S <= span <= SAMPLE_MAX_S:
                out.append({"start": run[0]["start"] / 1000, "end": run[-1]["end"] / 1000,
                            "text": " ".join(" ".join(r["text"]) for r in run), "cues": len(run)})
        run.clear()

    for record in records:
        solo = record.get("speaker") == speaker and not record.get("overlap") and not record.get("ocr")
        if solo and run and (record["start"] - run[-1]["end"]) / 1000 > SAMPLE_GAP_S:
            close()
        if not solo:
            close()
            continue
        if run and (record["end"] - run[0]["start"]) / 1000 > SAMPLE_MAX_S:
            close()
        run.append(record)
    close()
    # long, few cues (fewer turn edges), early in the list on ties
    out.sort(key=lambda s: (-(s["end"] - s["start"]) + 0.5 * s["cues"], s["start"]))
    return out[:SAMPLES_PER_EPISODE]


class VoiceBank:
    def __init__(self, path: Path, data: dict[str, Any] | None = None) -> None:
        self.path = path
        self.data = data or {"schema": SCHEMA, "prints": [], "conflicts": []}

    @classmethod
    def load(cls, path: Path) -> "VoiceBank":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema") != SCHEMA:
                raise ValueError(f"{path}: unknown voice bank schema {data.get('schema')!r}")
            return cls(path, data)
        return cls(path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    @property
    def prints(self) -> list[dict[str, Any]]:
        return self.data["prints"]

    def match(self, embedding: list[float], episode: str) -> list[dict[str, Any]]:
        """Names of other episodes' prints, best similarity first (one entry per person)."""
        probe = _unit(embedding)
        best: list[dict[str, Any]] = []
        for p in self.prints:
            if p["episode"] == episode:
                continue
            score = round(float(probe @ _unit(p["embedding"])), 3)
            entry = next((b for b in best if same_person(b["name"], p["name"])), None)
            if entry is None:
                best.append({"name": p["name"], "label_ru": p.get("label_ru"), "gender": p.get("gender"),
                             "score": score, "episode": p["episode"], "speaker": p["speaker"]})
                continue
            if len(p["name"]) > len(entry["name"]):  # the fullest spelling names the person
                entry.update(name=p["name"], label_ru=p.get("label_ru") or entry["label_ru"])
            if score > entry["score"]:
                entry.update(score=score, episode=p["episode"], speaker=p["speaker"])
        return sorted(best, key=lambda m: -m["score"])

    def add_episode(self, episode: str, speakers: list[dict[str, Any]], records: list[dict[str, Any]]) -> int:
        """Replace this episode's prints by its named, well-heard clusters; returns their number."""
        self.data["prints"] = [p for p in self.prints if p["episode"] != episode]
        self.data["conflicts"] = [c for c in self.data.get("conflicts", []) if c["episode"] != episode]
        added = 0
        for s in speakers:
            if not s.get("name") or not s.get("embedding") or (s.get("speech_s") or 0) < MIN_SPEECH_S:
                continue
            confident = s.get("name_source") == "voice_bank" or (s.get("llm_confidence") or 0) >= MIN_CONFIDENCE
            if not confident:
                continue
            self.prints.append({
                "episode": episode, "speaker": s["id"], "name": s["name"], "label_ru": s.get("label_ru"),
                "gender": s.get("gender"), "speech_s": s.get("speech_s"), "source": s.get("name_source", "llm"),
                "confidence": s.get("llm_confidence"), "embedding": s["embedding"],
                "samples": reference_samples(s["id"], records),
            })
            added += 1
        for s in speakers:
            if s.get("voice_bank_conflict"):
                self.data["conflicts"].append({"episode": episode, "speaker": s["id"], "name": s.get("name"),
                                               **s["voice_bank_conflict"]})
        return added

    def people(self) -> list[dict[str, Any]]:
        """Per name: episodes, total speech and the reference stretches, longest first."""
        people: list[dict[str, Any]] = []
        for p in self.prints:
            person = next((x for x in people if same_person(x["name"], p["name"])), None)
            if person is None:
                person = {"name": p["name"], "label_ru": p.get("label_ru"), "gender": p.get("gender"),
                          "episodes": [], "speech_s": 0.0, "samples": []}
                people.append(person)
            elif len(p["name"]) > len(person["name"]):
                person.update(name=p["name"], label_ru=p.get("label_ru") or person["label_ru"])
            person["episodes"].append(p["episode"])
            person["speech_s"] += p.get("speech_s") or 0
            person["samples"] += [{"episode": p["episode"], **x} for x in p.get("samples", [])]
        for person in people:
            person["episodes"] = sorted(set(person["episodes"]))
            person["samples"].sort(key=lambda x: -(x["end"] - x["start"]))
        return sorted(people, key=lambda x: -x["speech_s"])


def attach_matches(speakers: list[dict[str, Any]], bank: VoiceBank, episode: str) -> None:
    """Store the evidence-level matches of every cluster in ``voice_bank`` (top 2 names)."""
    for s in speakers:
        if not s.get("embedding"):
            continue
        matches = [m for m in bank.match(s["embedding"], episode) if m["score"] >= EVIDENCE][:2]
        if matches:
            s["voice_bank"] = matches
        else:
            s.pop("voice_bank", None)


def resolve(speakers: list[dict[str, Any]]) -> list[str]:
    """After LLM naming: fill unnamed clusters from strong matches, flag contradicted ones."""
    notes = []
    for s in speakers:
        matches = s.get("voice_bank") or []
        s.pop("voice_bank_conflict", None)
        if not matches or matches[0]["score"] < STRONG:
            continue
        top = matches[0]
        clear = len(matches) < 2 or top["score"] - matches[1]["score"] >= MARGIN
        if not s.get("name"):
            if clear:
                s.update(name=top["name"], label_ru=top.get("label_ru") or s.get("label_ru"),
                         name_source="voice_bank",
                         name_evidence=f"voice bank: {top['score']:.2f} with {top['speaker']} of {short(top['episode'])}")
                if s.get("gender") not in ("male", "female") and top.get("gender") in ("male", "female"):
                    s.update(gender=top["gender"], gender_source="voice_bank")
                notes.append(f"{s['id']} named {top['name']} by voice ({top['score']:.2f})")
        elif not same_person(s["name"], top["name"]) and clear:
            s["voice_bank_conflict"] = {"bank_name": top["name"], "score": top["score"],
                                        "bank_episode": top["episode"], "bank_speaker": top["speaker"]}
            notes.append(f"{s['id']} is {s['name']} by the transcript but sounds like {top['name']} "
                         f"of {short(top['episode'])} ({top['score']:.2f})")
    return notes


def short(episode: str) -> str:
    """S29E01 out of a long episode folder name."""
    found = re.search(r"S\d+E\d+", episode, re.IGNORECASE)
    return found.group(0) if found else episode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show a season voice bank")
    parser.add_argument("bank", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    bank = VoiceBank.load(args.bank)
    for person in bank.people():
        best = person["samples"][0] if person["samples"] else None
        sample = f"{short(best['episode'])} {best['start']:.1f}-{best['end']:.1f}s" if best else "-"
        print(f"{person['name']:24} {person.get('gender') or '?':7} {person['speech_s']:7.0f}s "
              f"{len(person['episodes'])} ep  best sample: {sample}")
    for c in bank.data.get("conflicts", []):
        print(f"conflict: {short(c['episode'])} {c['speaker']} {c['name']} sounds like {c['bank_name']} "
              f"({c['score']:.2f}, {short(c['bank_episode'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
