"""Lines worth checking by ear: voiced in the wrong gender or as the wrong speaker?

Each voiced line of ``subs.ru.json`` is heard twice:

- the age/gender model (the one that profiles the speakers) on the line's own audio: a
  confident (0.9+) other gender than the voice used, when the line does not clearly sound like
  its own speaker (print similarity under 0.35);
- its WeSpeaker voice print against the episode's speakers: clearly closer (0.15+) to another
  speaker than to its own.

The gender model hears the voices separated from music and effects (Hybrid Demucs, as in the
mix): music under narration made it hear the host as "female" (S29E01-E04: 30 of 72 gender flags
before). Shouts and crowd noise still fool it, so this is a list for a person, not an automatic
fix. Writes:

- ``review.json``: one entry per suspicious line - its number in ``subs.ru.srt``, time,
  speaker, the voice used, what the audio sounds like, English and Russian text, and a
  ready ``fix`` to copy into ``corrections.json``;
- ``review.srt``: the same lines only (numbers as in ``subs.ru.srt``), for Subtitle Edit next to
  the original video.

``corrections.json`` (``{"123": {"gender": "female"}, "130": {"speaker": "Julie McGee"},
"140": {"text": "..."}, "141": {"skip": true}}``) is applied by ``stt.dub``; ``stt.run`` redoes the
dubbing script and the voice-over when the file is newer than the last dub.

    subs_env/bin/python -m stt.review output/stt/<episode>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from stt.runner import load_json, save_json

log = logging.getLogger("stt.review")

GENDER_P = 0.9  # the gender model must be this sure of the other gender
MIN_GENDER_S = 0.7  # lines shorter than this are not judged by gender
MAX_GENDER_S = 10.0  # longer lines: their first seconds are enough
VOICE_MIN = 0.6  # another speaker's print must be at least this close ...
VOICE_MARGIN = 0.15  # ... and this much closer than the line's own speaker
MIN_VOICE_S = 1.0
OWN_VOICE_OK = 0.35  # a line this close to its own speaker's print is not flagged for gender alone
# (S29: the host's shouts heard as "female" matched his print at 0.36-0.76; other gender flags under 0.31)


def _clock(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def suspects(records: list[dict[str, Any]], speakers: list[dict[str, Any]], audio: np.ndarray, rate: int,
             gender_of: Any, embed: Any, skip: set[int] = frozenset(), offset: float = 0.0,
             voices: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Suspicious voiced lines; ``gender_of(audio) -> (p_female, p_male)``, ``embed(audio, rate) -> unit vector``.

    ``offset``: media time of the first sample of ``audio`` (a test piece starts mid-episode);
    ``voices``: the audio's separated voices (same rate and length), heard by the gender model."""
    from stt.voice_bank import _unit

    cards = {s["id"]: s for s in speakers}
    prints = {s["id"]: _unit(s["embedding"]) for s in speakers if s.get("embedding")}
    out = []
    for r in records:
        if r["index"] in skip or not " ".join(r["text"]).strip():
            continue
        start, end = r["start"] / 1000, r["end"] / 1000
        span = slice(max(0, int((start - offset) * rate)), max(0, int((end - offset) * rate)))
        segment = audio[span]
        if len(segment) < rate // 10:  # outside the decoded audio
            continue
        voice_only = voices[span] if voices is not None else segment
        voice = r.get("gender")
        entry: dict[str, Any] = {}
        own = r.get("speaker")
        sims: dict[str, float] = {}
        if end - start >= MIN_VOICE_S and own in prints and len(prints) > 1:
            vector = embed(segment, rate)
            sims = {sid: float(vector @ p) for sid, p in prints.items()}
        if end - start >= MIN_GENDER_S and voice in ("male", "female") and sims.get(own, 0.0) < OWN_VOICE_OK:
            p_female, p_male = gender_of(voice_only[: int(MAX_GENDER_S * rate)])
            heard, p = ("female", p_female) if p_female > p_male else ("male", p_male)
            if heard != voice and p >= GENDER_P:
                entry.update(heard=f"{heard} ({p:.2f})", fix={"gender": heard})
        if sims:
            best = max(sims, key=lambda sid: sims[sid])
            if best != own and sims[best] >= VOICE_MIN and sims[best] - sims[own] >= VOICE_MARGIN:
                other = cards[best]
                name = other.get("name") or other.get("label_ru") or best
                entry["sounds_like"] = f"{name} ({sims[best]:.2f}; own speaker {sims[own]:.2f})"
                entry["fix"] = {"speaker": name}
        if entry:
            card = cards.get(own, {})
            out.append({
                "line": r["index"], "time": _clock(start), "end_time": _clock(end),
                "speaker": r.get("speaker_name") or card.get("name") or own, "speaker_id": own,
                "voice": voice, **entry, "overlap": bool(r.get("overlap")),
                "both_signals": "heard" in entry and "sounds_like" in entry,
                "text_en": " ".join(r.get("text_en") or []), "text_ru": " ".join(r["text"]),
            })
    return out


def write_review(folder: Path, items: list[dict[str, Any]]) -> None:
    save_json(folder / "review.json", items)
    blocks = []
    sign = {"male": "M", "female": "F"}
    for item in items:
        what = []
        if "heard" in item:
            what.append(f"voice {sign.get(item['voice'], '?')}, heard {item['heard']}")
        if "sounds_like" in item:
            what.append(f"sounds like {item['sounds_like']}")
        blocks.append(f"{item['line']}\n{item['time']} --> {item['end_time']}\n"
                      f"#{item['line']} {item['speaker']}: {'; '.join(what)}\n{item['text_ru']}\n")
    (folder / "review.srt").write_text("\n".join(blocks), encoding="utf-8")


def separated_voices(audio: np.ndarray, rate: int, device: str = "cuda") -> np.ndarray:
    """The voices of a mono recording without music and effects (same rate and length)."""
    import soxr

    from stt.separate import VocalSeparator

    separator = VocalSeparator(device)
    up = soxr.resample(audio.astype(np.float32), rate, separator.rate)
    vocals = separator.separate_long(np.stack([up, up])).mean(0)
    return soxr.resample(vocals, separator.rate, rate)[: len(audio)].astype(np.float32)


def review(folder: Path, device: str = "cuda") -> list[dict[str, Any]]:
    import soundfile as sf

    from stt.voice import AgeGenderEstimator
    from stt.voice_bank import SegmentEmbedder

    records, speakers = load_json(folder / "subs.ru.json"), load_json(folder / "speakers.json")
    report = folder / "dub.report.json"
    skip = {s["index"] for s in load_json(report).get("skipped", [])} if report.exists() else set()
    audio, rate = sf.read(str(folder / "audio.wav"), dtype="float32")
    source = folder / "source.json"
    offset = float(load_json(source).get("start") or 0.0) if source.exists() else 0.0
    voices = separated_voices(audio, rate, device)
    estimator = AgeGenderEstimator(device)
    estimator.load()

    def gender_of(segment: np.ndarray) -> tuple[float, float]:
        _, probs = estimator.predict(segment)
        return float(probs[0]), float(probs[1])

    items = suspects(records, speakers, audio, rate, gender_of, SegmentEmbedder(device), skip, offset, voices)
    write_review(folder, items)
    log.info("review: %d of %d lines to check (%d with both signals) -> %s", len(items), len(records),
             sum(i["both_signals"] for i in items), folder / "review.json")
    return items


def apply_corrections(records: list[dict[str, Any]], corrections: dict[str, dict[str, Any]],
                      speakers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[int]]:
    """Apply ``corrections.json`` to translated lines: (corrected records, line numbers to skip)."""
    from stt.voice_bank import same_person

    by_index = {r["index"]: r for r in records}
    skip: set[int] = set()
    for key, fix in corrections.items():
        index = int(key)
        record = by_index.get(index)
        if record is None:
            log.warning("corrections: no line %s", key)
            continue
        if fix.get("skip"):
            skip.add(index)
        if "speaker" in fix:
            wanted = str(fix["speaker"])
            card = next((s for s in speakers if wanted in (s["id"], s.get("label_ru"))
                         or (s.get("name") and same_person(s["name"], wanted))), None)
            if card is None:
                raise ValueError(f"corrections.json line {key}: no speaker {wanted!r} in speakers.json")
            record.update(speaker=card["id"], speaker_name=card.get("label_ru") or card.get("name"),
                          gender=card.get("gender", record.get("gender")))
        if "gender" in fix:
            if fix["gender"] not in ("male", "female"):
                raise ValueError(f"corrections.json line {key}: gender must be male or female")
            record["gender"] = fix["gender"]
        if "text" in fix:
            record["text"] = fix["text"] if isinstance(fix["text"], list) else [str(fix["text"])]
    return records, skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List voiced lines that may have the wrong gender or speaker")
    parser.add_argument("folder", type=Path, help="Episode folder (after translate)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    review(args.folder, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
