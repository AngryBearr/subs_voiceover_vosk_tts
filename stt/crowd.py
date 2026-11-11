"""Shout layer: interjections the voice-over left out, said on a layer of their own.

ASR follows one voice at a time, so a team shouting "Go, go, go!" arrives as a short line of one
speaker, and `stt.dub` leaves such interjections out of the voice-over when they sit on other
speech (or, with `--interjections dense`, in a rapid exchange): one voice-over track cannot say them all.
Here they come back as an ambient layer: skipped interjections less than `group_gap_s` apart form a group,
whose longest Russian line is said by TTS and added to the mix after the ducking (`stt.mix --extra`), quieter
than the voice-over, so the original is not lowered for it. By default (`--solo`) every group is said, by one
voice: the engine's generic voice of the group's gender, or of `--neutral-gender` when nobody's gender is
known; a lone backchannel ("Да.", "Ага.", "Мм": one person agreeing under another's line, no exclamation
mark) is not a shout and stays in the original only. Without `--solo` only crowds are said (two or more
speakers, or a chant: a repeated word), and
`--voices 3` says them by three voices at once with small offsets. The voices come from `dub.report.json`
(the engine's own generic voices: Vosk ids, TeraTTS names, the ESpeech track's TeraTTS fallback) plus
`EXTRA_VOICES` for the second and third voice.

    subs_env/bin/python -m stt.crowd output/stt/ep01 --engine vosk [--solo] [--voices 1]
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

# Second and third voice per gender and engine for a group said by several voices; the first voice is the
# dub's own generic voice of that gender (``voices`` of dub.report.json), so every engine works without a table.
EXTRA_VOICES: dict[str, dict[str, list[int | str]]] = {
    "vosk": {"female": [38, 15], "male": [12, 27]},
    "teratts": {"female": ["ru_f2"], "male": ["ru_m1"]},
    "f5": {"female": ["teratts:ru_f2"], "male": ["teratts:ru_m1"]},
}
GENDERS = ("female", "male")


def voice_pool(engine_name: str, dub_voices: dict[str, Any]) -> dict[str, list[int | str]]:
    """{gender: [the dub's generic voice, extra voices...]} for an engine."""
    extras = EXTRA_VOICES.get(engine_name, {})
    pool = {g: [dub_voices[g]] + [v for v in extras.get(g, []) if v != dub_voices[g]] for g in GENDERS if g in dub_voices}
    if not pool:
        raise ValueError(f"dub.report.json has no generic voices for {engine_name}: {dub_voices}")
    return pool


def groups(skipped: list[dict[str, Any]], group_gap_s: float = 2.0) -> list[dict[str, Any]]:
    """Skipped interjections close in time -> [{start, end, text, genders, lines}] (times in ms).
    Other skipped lines (captions shown as text, corrections, empty lines) are not interjections and stay out."""
    out: list[dict[str, Any]] = []
    interjections = [x for x in skipped if str(x.get("reason", "")).startswith("interjection")]
    for line in sorted(interjections, key=lambda x: x["start"]):
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


def is_backchannel(text: str) -> bool:
    """One person agreeing or hesitating ("Да.", "Ага, ага", "Мм"): backchannel words only, no exclamation mark."""
    words = re.findall(r"\w+", text.lower())
    return bool(words) and "!" not in text and all(w in _BACKCHANNEL for w in words)


def is_chant(text: str) -> bool:
    """A repeated word that is not a backchannel: "Go, go, go!", "Давай, давай!", "Вперёд, вперёд!"."""
    words = re.findall(r"\w+", text.lower())
    return any(words.count(w) >= 2 for w in set(words) if len(w) >= 2 and w not in _BACKCHANNEL)


def pick_voices(genders: list[str], pool: dict[str, list[int | str]], count: int = 3,
                neutral_gender: str = "male") -> list[int | str]:
    """Voices for a group: the genders heard first (`neutral_gender` when nobody's is known), then the other one,
    `count` in total."""
    order = sorted(GENDERS, key=lambda g: (-genders.count(g), g != neutral_gender))
    picked: list[int | str] = []
    for k in range(max(len(v) for v in pool.values())):
        for gender in order:
            if k < len(pool.get(gender, ())) and len(picked) < count:
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


def build(folder: Path, engine_name: str, count: int = 1, *, solo: bool = True, neutral_gender: str = "male",
          generic_voices: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], np.ndarray, int]:
    """`generic_voices`: the engine's voice per gender; without it the ones of `dub.report.json`, which the last
    engine's dub wrote (the same for every engine only when one engine runs)."""
    from voiceover.engine import build_engine
    from voiceover.render import stretch_clip

    report = load_json(folder / "dub.report.json")
    groups_ = [g for g in groups(report.get("skipped", []))
               if g["crowd"] or (solo and not all(is_backchannel(line) for line in g["lines"]))]
    pool = voice_pool(engine_name, generic_voices or report.get("voices") or {})
    for group in groups_:
        group["voices"] = pick_voices(group["genders"], pool, count, neutral_gender)
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
    parser = argparse.ArgumentParser(description="Shout layer: skipped interjections said on a layer of their own")
    parser.add_argument("folder", help="Episode folder of stt.run (after the voiceover stage)")
    parser.add_argument("--engine", default="vosk", choices=sorted(EXTRA_VOICES))
    parser.add_argument("--voices", type=int, default=1, help="Voices per group")
    parser.add_argument("--solo", action="store_true",
                        help="Say every group, not only crowds (two or more speakers, or a chant)")
    parser.add_argument("--neutral-gender", default="male", choices=list(GENDERS),
                        help="The generic voice of a group whose speakers' gender is unknown")
    parser.add_argument("--generic-voices", default=None,
                        help="The engine's voice per gender, female=<voice>,male=<voice> (as stt.dub --voices); "
                             "by default the voices of dub.report.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    from stt.dub import parse_voices

    folder = Path(args.folder)
    generic = parse_voices(args.generic_voices) if args.generic_voices else None
    groups_, track, sample_rate = build(folder, args.engine, args.voices, solo=args.solo,
                                        neutral_gender=args.neutral_gender, generic_voices=generic)
    out = folder / f"crowd.{args.engine}.wav"
    sf.write(str(out), track, sample_rate, subtype="PCM_16")
    save_json(folder / f"crowd.{args.engine}.json", groups_)
    log.info("shout layer: %d groups from %d skipped lines (%d crowds) -> %s", len(groups_),
             sum(len(g["lines"]) for g in groups_), sum(1 for g in groups_ if g["crowd"]), out)


if __name__ == "__main__":
    main()
