"""Crowd layer (experimental): cheers and chants said by several voices at once.

ASR follows one voice at a time, so a team shouting "Go, go, go!" arrives as a short line of one
speaker, and `stt.dub` leaves such interjections out of the voice-over when they sit on other
speech or in a rapid exchange (one voice-over track cannot say them all). Here they come back
as an ambient layer: skipped interjections less than `group_gap_s` apart form a group; a group
is a crowd if two or more speakers shout or the line is a chant (a repeated word: "Go, go, go!").
Its Russian line is then said by several TTS voices at once with small offsets, quieter than the
voice-over. One person's "Oh my God!" stays in the original only. `stt.mix --extra` adds the layer after ducking, so it does not lower the original.

    subs_env/bin/python -m stt.crowd output/stt/ep01 --engine vosk
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import re

import numpy as np
import soundfile as sf

from stt.runner import REPO_ROOT, load_json, save_json

log = logging.getLogger("stt.crowd")

# Extra voices per gender, first = the dub's voice (Vosk ids by speaker name in the model config).
CROWD_VOICES: dict[str, dict[str, list[int | str]]] = {
    "vosk": {"female": [49, 38, 15], "male": [56, 12, 27]},
    "teratts": {"female": ["ru_f1", "ru_f2"], "male": ["ru_m5", "ru_m1"]},
}


def groups(skipped: list[dict[str, Any]], group_gap_s: float = 2.0) -> list[dict[str, Any]]:
    """Skipped interjections close in time -> [{start, end, text, genders, lines}] (times in ms)."""
    out: list[dict[str, Any]] = []
    for line in sorted(skipped, key=lambda x: x["start"]):
        text = " ".join(line["text"]) if isinstance(line["text"], list) else str(line["text"])
        end = int(line.get("end", line["start"] + 800))
        if out and line["start"] - out[-1]["end"] <= group_gap_s * 1000:
            group = out[-1]
            group["end"] = max(group["end"], end)
            group["lines"].append(text)
        else:
            group = {"start": int(line["start"]), "end": end, "lines": [text], "genders": []}
            out.append(group)
        if line.get("gender") in ("female", "male"):
            group["genders"].append(line["gender"])
        group.setdefault("speakers", set()).add(line.get("speaker"))
    for group in out:  # the longest line speaks for the group ("Вперёд! Вперёд!" over "Да!")
        group["text"] = max(group["lines"], key=len)
        group["speakers"] = sorted(str(x) for x in group["speakers"] if x)
        group["crowd"] = len(group["speakers"]) >= 2 or is_chant(" ".join(group["lines"]))
    return out


# Repeated backchannels ("Да, да, да", "Ага, ага") are one person agreeing, not a crowd.
_BACKCHANNEL = {"да", "ага", "угу", "нет", "ну", "ой", "ох", "ах", "ай", "мм", "хм", "так", "yes", "yeah", "no", "oh"}


def is_chant(text: str) -> bool:
    """A repeated word that is not a backchannel: "Go, go, go!", "Давай, давай!", "Вперёд, вперёд!"."""
    words = re.findall(r"\w+", text.lower())
    return any(words.count(w) >= 2 for w in set(words) if len(w) >= 2 and w not in _BACKCHANNEL)


def pick_voices(genders: list[str], pool: dict[str, list[int | str]], count: int = 3) -> list[int | str]:
    """Voices for a group: the genders heard first, then the other one, `count` in total."""
    order = sorted(("female", "male"), key=lambda g: -genders.count(g))
    picked: list[int | str] = []
    for k in range(max(len(v) for v in pool.values())):
        for gender in order:
            if k < len(pool[gender]) and len(picked) < count:
                picked.append(pool[gender][k])
    return picked


def render(groups_: list[dict[str, Any]], clips: dict[tuple[int, int | str], np.ndarray], sample_rate: int,
           length_s: float, *, offset_ms: int = 70, gain_db: float = -3.0) -> np.ndarray:
    """Sum the voices of every group with small offsets into one mono track."""
    track = np.zeros(int(length_s * sample_rate) + sample_rate, dtype=np.float32)
    gain = 10 ** (gain_db / 20)
    for n, group in enumerate(groups_):
        for k, voice in enumerate(group["voices"]):
            clip = clips.get((n, voice))
            if clip is None:
                continue
            at = int((group["start"] + k * offset_ms) / 1000 * sample_rate)
            end = min(len(track), at + len(clip))
            track[at:end] += gain * clip[: end - at]
    return track


def build(folder: Path, engine_name: str, count: int = 3) -> tuple[list[dict[str, Any]], np.ndarray, int]:
    from voiceover.engine import build_engine
    from voiceover.render import stretch_clip

    report = load_json(folder / "dub.report.json")
    groups_ = [g for g in groups(report.get("skipped", [])) if g["crowd"]]
    pool = CROWD_VOICES[engine_name]
    for group in groups_:
        group["voices"] = pick_voices(group["genders"], pool, count)
    vo_dir = folder / f"voiceover.{engine_name}"
    config = load_json(folder / f"voiceover.{engine_name}.config.json")
    calibration = load_json(vo_dir / "calibration.json")["calibration"]
    engine = build_engine({**config["engine"], "base_rate": calibration["rate"]})
    items = [{"cue_id": f"{n}|{voice}", "index": n, "text": group["text"], "voice": voice}
             for n, group in enumerate(groups_) for voice in group["voices"]]
    prepared = engine.prepare(items)
    cache = REPO_ROOT / config.get("synthesis", {}).get("cache_dir", "output/tts_cache")
    results = engine.synthesize_many(prepared, cache, int(config.get("synthesis", {}).get("workers", 1)))
    sample_rate = int(config.get("export", {}).get("sample_rate", 22050))
    clips: dict[tuple[int, int | str], np.ndarray] = {}
    for item, result in zip(items, results):
        if not result.error:
            path = Path(result.path) if Path(result.path).is_absolute() else REPO_ROOT / result.path
            clips[(int(item["index"]), item["voice"])] = stretch_clip(path, 1.0, sample_rate)
    length = sf.info(str(vo_dir / "export" / "track.wav")).duration
    return groups_, render(groups_, clips, sample_rate, length), sample_rate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ambient crowd layer from skipped interjections (experimental)")
    parser.add_argument("folder", help="Episode folder of stt.run (after the voiceover stage)")
    parser.add_argument("--engine", default="vosk", choices=sorted(CROWD_VOICES))
    parser.add_argument("--voices", type=int, default=3, help="Voices per group")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    folder = Path(args.folder)
    groups_, track, sample_rate = build(folder, args.engine, args.voices)
    out = folder / f"crowd.{args.engine}.wav"
    sf.write(str(out), track, sample_rate, subtype="PCM_16")
    save_json(folder / f"crowd.{args.engine}.json", groups_)
    log.info("crowd: %d groups from %d skipped lines -> %s", len(groups_), sum(len(g["lines"]) for g in groups_), out)


if __name__ == "__main__":
    main()
