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
lists, per person, stretches of solo speech (6-12 s). ``export`` picks the cleanest one per
person (no other diarized voice inside, every 3 s window sounding like the person, the least
music) and writes its separated vocals (Hybrid Demucs) as a voice-cloning reference: 24 kHz
mono, 1 s of silence at the end, transcript by Whisper from the clip itself (F5 speaks any
mismatch between clip and transcript into the cloned line).

Once several episodes are in, ``--recheck`` looks for prints whose voice other episodes
know under another name (S29E01's "Jaclyn" cluster is Baylor's voice in most lines) and can
write the fix as ``speakers.override.json`` of that episode (``--write-overrides``); then
``stt.run --force speakers`` redoes its naming and translation.

    uv run -m stt.voice_bank output/stt_s29/<season>.voices.json            # what the bank holds
    uv run -m stt.voice_bank output/stt_s29/<season>.voices.json --export output/voices/s29
    uv run -m stt.voice_bank output/stt_s29/<season>.voices.json --recheck [--write-overrides]
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import subprocess
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
SAMPLE_MIN_S, SAMPLE_MAX_S, SAMPLE_GAP_S = 5.0, 9.0, 0.4  # F5 clips references over 12 s (with 1 s pad): keep them shorter
SAMPLES_PER_EPISODE = 5
CLONE_RATE = 24_000
MIN_SAMPLE_SCORE = 0.6  # a reference clip must sound at least this much like the person's prints
MIN_WINDOW_SCORE = 0.45  # ... and so must each 3 s window of it (another voice inside drops one)
MAX_OTHER_VOICE_S = 0.15  # seconds of another diarized speaker allowed inside a reference stretch
MUSIC_DB = -20.0  # accompaniment below the voice by more than this counts as clean
SEPARATE_TOP = 8  # candidates per person that go through vocal separation
MIN_SUPPORT = 2  # a rename needs the voice under the other name in at least this many other episodes
RECHECK_SCORE = 0.85  # ... and this similarity (S29 twins Nadiya/Natalie: 0.80)


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


def relatives(a: str, b: str) -> bool:
    """Different first names, one surname: family members sound alike (Nadiya/Natalie Anderson)."""
    ta, tb = re.findall(r"[^\W\d_]+", a.lower()), re.findall(r"[^\W\d_]+", b.lower())
    return len(ta) > 1 and len(tb) > 1 and ta[-1] == tb[-1] and not same_person(a, b)


def _unit(vector: list[float]) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def reference_samples(speaker: str, records: list[dict[str, Any]], limit: int | None = SAMPLES_PER_EPISODE
                      ) -> list[dict[str, Any]]:
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
    return out[:limit] if limit else out


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


def recheck(bank: VoiceBank) -> list[dict[str, Any]]:
    """Prints whose voice more other episodes know under another name (in at least MIN_SUPPORT, with a
    similarity of RECHECK_SCORE or more); relatives with one surname are never renamed into each other."""
    suggestions = []
    for p in bank.prints:
        probe = _unit(p["embedding"])
        groups: list[dict[str, Any]] = []
        for q in bank.prints:
            if q["episode"] == p["episode"]:
                continue
            score = float(probe @ _unit(q["embedding"]))
            group = next((g for g in groups if same_person(g["name"], q["name"])), None)
            if group is None:
                group = {"name": q["name"], "label_ru": q.get("label_ru"), "episodes": set(), "score": 0.0}
                groups.append(group)
            elif len(q["name"]) > len(group["name"]):
                group.update(name=q["name"], label_ru=q.get("label_ru") or group["label_ru"])
            if score >= STRONG:
                group["episodes"].add(q["episode"])
            group["score"] = max(group["score"], score)
        own = next((len(g["episodes"]) for g in groups if same_person(g["name"], p["name"])), 0)
        others = [g for g in groups if not same_person(g["name"], p["name"])]
        if not others:
            continue
        best = max(others, key=lambda g: (len(g["episodes"]), g["score"]))
        if relatives(p["name"], best["name"]) or best["score"] < RECHECK_SCORE:
            continue
        if len(best["episodes"]) >= MIN_SUPPORT and len(best["episodes"]) > own:
            suggestions.append({"episode": p["episode"], "speaker": p["speaker"], "name": p["name"],
                                "suggested": best["name"], "label_ru": best["label_ru"],
                                "support": len(best["episodes"]), "own_support": own,
                                "score": round(best["score"], 3)})
    return suggestions


def write_overrides(bank: VoiceBank, suggestions: list[dict[str, Any]]) -> list[Path]:
    """Add the renames to each episode's speakers.override.json (manual entries win)."""
    written = []
    for episode in sorted({s["episode"] for s in suggestions}):
        path = bank.path.parent / episode / "speakers.override.json"
        overrides = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        for s in suggestions:
            if s["episode"] == episode and s["speaker"] not in overrides:
                overrides[s["speaker"]] = {"name": s["suggested"], "label_ru": s["label_ru"],
                                           "name_source": "voice_bank_recheck"}
        path.write_text(json.dumps(overrides, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        written.append(path)
    return written


class SegmentEmbedder:
    """WeSpeaker embedding of an audio stretch, in the space of pyannote community-1's cluster prints."""

    def __init__(self, device: str = "cuda") -> None:
        import glob as globlib

        import torch
        from pyannote.audio import Inference, Model

        cache = Path.home() / ".cache/huggingface/hub/models--pyannote--speaker-diarization-community-1"
        found = sorted(globlib.glob(str(cache / "snapshots" / "*" / "embedding")))
        if not found:
            raise RuntimeError("pyannote community-1 is not downloaded; run a transcription first")
        self._torch = torch
        self.inference = Inference(Model.from_pretrained(found[-1]), window="whole", device=torch.device(device))

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        vector = self.inference({"waveform": self._torch.from_numpy(np.ascontiguousarray(audio))[None],
                                 "sample_rate": sample_rate})
        return _unit(list(np.asarray(vector).ravel()))


def _words(episode_dir: Path, start: float, end: float) -> str | None:
    """Aligned words whose middle falls inside the stretch (exact text for a voice-cloning reference)."""
    path = episode_dir / "words.json"
    if not path.exists():
        return None
    words = json.loads(path.read_text(encoding="utf-8"))
    picked = [w["text"] for w in words if start <= (w["start"] + w["end"]) / 2 <= end]
    return " ".join(picked) or None


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("_") or "speaker"


def _read_media(media: str, start: float, duration: float, rate: int, channels: int) -> np.ndarray:
    """(channels, n) float32 samples of a stretch of the original video's audio."""
    from utils.ffmpeg import ffmpeg_exe

    raw = subprocess.run([ffmpeg_exe(), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", media,
                          "-vn", "-ac", str(channels), "-ar", str(rate), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, channels).T


def other_voice_seconds(episode_dir: Path, speaker: str, start: float, end: float) -> float:
    """Seconds of other diarized speakers inside the stretch (overlapping speech included)."""
    found = sorted(episode_dir.glob("diarization.*.json"))
    speakers_path = episode_dir / "speakers.json"
    if not found or not speakers_path.exists():
        return 0.0
    labels = {s["id"]: s.get("diarizer_label") for s in json.loads(speakers_path.read_text(encoding="utf-8"))}
    own = labels.get(speaker)
    turns = json.loads(found[0].read_text(encoding="utf-8"))["turns"]
    return sum(max(0.0, min(end, t["end"]) - max(start, t["start"])) for t in turns if t["speaker"] != own)


def _window_scores(embed: "SegmentEmbedder", audio: np.ndarray, rate: int, prints: list[np.ndarray],
                   window: float = 3.0, hop: float = 1.5) -> list[float]:
    size, step = int(window * rate), int(hop * rate)
    starts = range(0, max(1, len(audio) - size + 1), step)
    return [max(float(embed(audio[i:i + size], rate) @ p) for p in prints) for i in starts]


def load_rejected(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "rejected.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def reject_reference(out_dir: Path, name: str) -> dict[str, Any] | None:
    """Mark a person's current clip as rejected (``rejected.json``); the next export picks another."""
    index = json.loads((out_dir / "references.json").read_text(encoding="utf-8"))
    current = next((r for r in index if r["name"] == name), None)
    if current is None:
        return None
    rejected = load_rejected(out_dir)
    rejected.append({"name": name, "episode": current["episode"], "start": current["start"]})
    (out_dir / "rejected.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return current


def export_references(bank: VoiceBank, out_dir: Path, device: str = "cuda", only: str | None = None,
                      missing: bool = False) -> list[dict[str, Any]]:
    """Cut the cleanest reference stretch of every person (or only ``only``) into ``out_dir``.

    Returns the index (``references.json``; other people's entries are kept when ``only`` is set).

    Candidates: the person's solo stretches in every episode of the bank, without another
    diarized speaker inside. The closest to the person's prints go through vocal separation;
    a candidate needs every 3 s window of its vocals to sound like the person. Clean ones
    (accompaniment more than 20 dB under the voice) win, then the most similar. The clip is
    the separated vocals, 24 kHz mono, with 1 s of silence at the end.
    """
    import soundfile as sf
    import soxr

    from stt.separate import VocalSeparator

    embed = SegmentEmbedder(device)
    separate = VocalSeparator(device)
    transcriber: Any = None
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "references.json"
    index: list[dict[str, Any]] = []
    if (only or missing) and index_path.exists():  # keep the other people's clips
        index = [r for r in json.loads(index_path.read_text(encoding="utf-8")) if r["name"] != only]
    have = {r["name"] for r in index} if missing else set()
    rejected = {(r["episode"], round(r["start"], 2)) for r in load_rejected(out_dir)}
    for person in bank.people():
        if (only and person["name"] != only) or person["name"] in have:
            continue
        mine = [p for p in bank.prints if same_person(p["name"], person["name"])]
        prints = [_unit(p["embedding"]) for p in mine]
        candidates = []
        for p in mine:
            episode_dir = bank.path.parent / p["episode"]
            if not (episode_dir / "audio.wav").exists() or not (episode_dir / "subs.en.json").exists():
                continue
            records = json.loads((episode_dir / "subs.en.json").read_text(encoding="utf-8"))
            if p["episode"] not in audio_cache:
                audio_cache[p["episode"]] = sf.read(str(episode_dir / "audio.wav"), dtype="float32")
            audio, sr = audio_cache[p["episode"]]
            source = episode_dir / "source.json"  # a test piece's audio and diarization start mid-episode
            offset = float(json.loads(source.read_text(encoding="utf-8")).get("start") or 0.0) if source.exists() else 0.0
            for sample in reference_samples(p["speaker"], records, limit=None):
                if (p["episode"], round(sample["start"], 2)) in rejected:
                    continue
                start, end = sample["start"] - offset, sample["end"] - offset
                if start < 0 or other_voice_seconds(episode_dir, p["speaker"], start, end) > MAX_OTHER_VOICE_S:
                    continue
                vector = embed(audio[int(start * sr):int(end * sr)], sr)
                candidates.append({**sample, "episode": p["episode"], "speaker": p["speaker"],
                                   "score": round(max(float(vector @ q) for q in prints), 3)})
        candidates.sort(key=lambda c: -c["score"])
        best: dict[str, Any] | None = None
        for c in candidates[:SEPARATE_TOP]:
            if c["score"] < MIN_SAMPLE_SCORE:
                break
            media = json.loads((bank.path.parent / c["episode"] / "source.json").read_text(encoding="utf-8"))["media"]
            stereo = _read_media(media, c["start"], c["end"] - c["start"], separate.rate, 2)
            vocals, music_db = separate(stereo)
            mono16 = soxr.resample(vocals.mean(0), separate.rate, 16_000)
            windows = _window_scores(embed, mono16, 16_000, prints)
            if min(windows) < MIN_WINDOW_SCORE:
                continue
            c = {**c, "music_db": music_db, "min_window": round(min(windows), 3), "_vocals": vocals}
            if best is None or (c["music_db"] < MUSIC_DB, c["score"]) > (best["music_db"] < MUSIC_DB, best["score"]):
                best = c
        if best is None:
            log.info("%s: no clean reference among %d candidates", person["name"], len(candidates))
            (out_dir / f"{_safe_name(person['name'])}.wav").unlink(missing_ok=True)
            continue
        voice = soxr.resample(best.pop("_vocals").mean(0), separate.rate, CLONE_RATE).astype(np.float32)
        fade = int(0.05 * CLONE_RATE)
        voice[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        voice[-2 * fade:] *= np.linspace(1, 0, 2 * fade, dtype=np.float32)
        voice = np.concatenate([voice, np.zeros(CLONE_RATE, dtype=np.float32)])
        stem = _safe_name(person["name"])
        sf.write(str(out_dir / f"{stem}.wav"), voice, CLONE_RATE, subtype="PCM_16")
        if transcriber is None:  # what is actually said in the clip: its edges may cut a word of the cue
            from stt.asr import create_backend

            transcriber = create_backend("whisper-large-v3", device=device)
            transcriber.load()
        heard = " ".join(seg.text for seg in transcriber.transcribe(
            soxr.resample(voice, CLONE_RATE, 16_000).astype(np.float32), [(0.0, len(voice) / CLONE_RATE)]))
        text = heard.strip() or _words(bank.path.parent / best["episode"], best["start"], best["end"]) or best["text"]
        (out_dir / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
        index.append({"name": person["name"], "label_ru": person.get("label_ru"), "gender": person.get("gender"),
                      "wav": f"{stem}.wav", "text": text, "candidates": len(candidates),
                      **{k: best[k] for k in ("episode", "speaker", "start", "end", "score", "min_window", "music_db")}})
        log.info("%-24s %s %.1f-%.1fs score %.2f, windows >= %.2f, music %+.0f dB (%d candidates)", person["name"],
                 short(best["episode"]), best["start"], best["end"], best["score"], best["min_window"],
                 best["music_db"], len(candidates))
    index.sort(key=lambda r: r["name"])
    write_if_changed(index_path, json.dumps(index, ensure_ascii=False, indent=1) + "\n")
    return index


def write_if_changed(path: Path, text: str) -> bool:
    """Write ``text`` unless the file already holds it: an unchanged clip index keeps its time, so
    the episodes voiced with these clips are not voiced again (stt.run compares the times)."""
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def short(episode: str) -> str:
    """S29E01 out of a long episode folder name."""
    found = re.search(r"S\d+E\d+", episode, re.IGNORECASE)
    return found.group(0) if found else episode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show a season voice bank, or export voice-cloning references")
    parser.add_argument("bank", type=Path)
    parser.add_argument("--export", type=Path, default=None, help="Cut one reference clip per person into this folder")
    parser.add_argument("--person", default=None, help="With --export: only this person's clip")
    parser.add_argument("--missing", action="store_true",
                        help="With --export: only people without a clip yet (a person keeps one voice)")
    parser.add_argument("--reject", action="store_true",
                        help="With --export and --person: reject the current clip and pick another")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--recheck", action="store_true", help="List prints other episodes know by another name")
    parser.add_argument("--write-overrides", action="store_true",
                        help="With --recheck: write the renames into the episodes' speakers.override.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    bank = VoiceBank.load(args.bank)
    if args.export:
        if args.reject and args.person:
            reject_reference(args.export, args.person)
        export_references(bank, args.export, args.device, only=args.person, missing=args.missing)
        return 0
    if args.recheck:
        suggestions = recheck(bank)
        for x in suggestions:
            print(f"{short(x['episode'])} {x['speaker']} {x['name']} -> {x['suggested']} "
                  f"(voice in {x['support']} other episodes under that name, {x['own_support']} under its own; "
                  f"best {x['score']:.2f})")
        if not suggestions:
            print("no renames suggested")
        if args.write_overrides and suggestions:
            for path in write_overrides(bank, suggestions):
                print(f"wrote {path}; redo with: stt.run <video> --force speakers")
        return 0
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
