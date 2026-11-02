"""Speaker model training set: the SDH-labelled lines (lines.json) as audio clips in one file on the SSD.

Gold segments = lines pinned to words, the person named by the subtitles. Cue segments = labelled lines Whisper did not hear
(shouts at challenges), with the cue's times, only cues of one voice. Pseudo segments = other diarized turns of a
cluster that the labelled lines tie to one person (purity at least --purity); they only serve as supports (the centroid
side), never as queries, so the model cannot just learn the diarizer's mistakes as hard examples. Noise = stretches with
neither words nor diarized speech. Held out and never read here: S51 (its cast is the test cast), S20, S29, Australian S10;
S46 is the development season (scored each epoch).

    subs_env/bin/python -I scripts/speaker_training/build_dataset.py --out output/exp_spk/data [--seasons S40,S48,S49,S50,SAU1]
Output: audio.i16 (16 kHz int16), index.json {"segments": [...], "noise": [...], "classes": {...}}.
"""
from __future__ import annotations

import argparse
import json
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

RATE = 16000
PAD_S = 0.7  # audio kept around a segment: a 1 s window centred on a word near the edge reaches the neighbour's speech
TRAIN = "output/subs_train"


def read_i16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def cluster_owner(lines: list[dict[str, Any]], purity: float, min_lines: int) -> dict[str, str]:
    """Diarizer cluster -> the one person the labelled lines say it is, if they say so clearly."""
    seconds: dict[str, Counter[str]] = defaultdict(Counter)
    count: Counter[tuple[str, str]] = Counter()
    for x in lines:
        if x["cluster"] is not None:
            seconds[x["cluster"]][x["person"]] += x["end"] - x["start"]
            count[(x["cluster"], x["person"])] += 1
    owners: dict[str, str] = {}
    for cluster, by_person in seconds.items():
        person, sec = by_person.most_common(1)[0]
        if sec / sum(by_person.values()) >= purity and count[(cluster, person)] >= min_lines:
            owners[cluster] = person
    # a person with two owned clusters is fine; two people on one cluster never reach here
    return owners


def pseudo_turns(turns: list[dict[str, Any]], owners: dict[str, str], cap_s: float, min_s: float, max_s: float,
                 rng: np.random.Generator) -> list[dict[str, Any]]:
    """Clean turns (no second voice) of the owned clusters, cut to at most max_s, capped per person."""
    spans = sorted(turns, key=lambda t: t["start"])
    out: list[dict[str, Any]] = []
    for i, t in enumerate(spans):
        if t["speaker"] not in owners or t["end"] - t["start"] < min_s:
            continue
        busy = any(o is not t and o["speaker"] != t["speaker"] and o["start"] < t["end"] and o["end"] > t["start"]
                   for o in spans[max(i - 8, 0):i + 8])
        if busy:
            continue
        a = t["start"]
        while a + min_s <= t["end"]:
            b = min(a + max_s, t["end"])
            out.append({"person": owners[t["speaker"]], "start": a, "end": b})
            a = b
    rng.shuffle(out)
    used: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    for seg in out:
        if used[seg["person"]] < cap_s:
            kept.append(seg)
            used[seg["person"]] += seg["end"] - seg["start"]
    return kept


def noise_spans(turns: list[dict[str, Any]], words: list[dict[str, Any]], duration: float, total_s: float,
                rng: np.random.Generator) -> list[tuple[float, float]]:
    """Stretches of at least 1 s, 0.3 s away from any diarized turn or word."""
    busy = sorted([(t["start"], t["end"]) for t in turns] + [(w["start"], w["end"]) for w in words])
    gaps: list[tuple[float, float]] = []
    last = 0.0
    for a, b in busy:
        if a - last >= 1.6:
            gaps.append((last + 0.3, a - 0.3))
        last = max(last, b)
    if duration - last >= 1.6:
        gaps.append((last + 0.3, duration - 0.3))
    order = rng.permutation(len(gaps))
    out: list[tuple[float, float]] = []
    got = 0.0
    for k in order:
        a, b = gaps[int(k)]
        b = min(b, a + 6.0)
        out.append((a, b))
        got += b - a
        if got >= total_s:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="output/exp_spk/data")
    parser.add_argument("--seasons", default="S40,S48,S49,S50,SAU1")
    parser.add_argument("--purity", type=float, default=0.8)
    parser.add_argument("--min-owner-lines", type=int, default=3)
    parser.add_argument("--min-person-lines", type=int, default=8)
    parser.add_argument("--pseudo-cap-s", type=float, default=150.0, help="pseudo seconds per person per episode")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    train = Path(TRAIN)
    episodes = [p for s in args.seasons.split(",") for p in sorted((train / s).glob(f"{s}E*/lines.json"))]
    per_season: dict[str, Counter[str]] = defaultdict(Counter)
    for path in episodes:
        for x in json.loads(path.read_text(encoding="utf-8"))["lines"]:
            per_season[path.parent.parent.name][x["person"]] += 1
    keep = {(s, p) for s, c in per_season.items() for p, n in c.items() if n >= args.min_person_lines}
    classes: dict[str, int] = {}

    def class_id(season: str, person: str) -> int:
        name = "_HOST" if person == "_HOST" else f"{season}:{person}"
        return classes.setdefault(name, len(classes))

    segments: list[dict[str, Any]] = []
    noise: list[dict[str, Any]] = []
    offset = 0
    stats: Counter[str] = Counter()
    with open(out / "audio.i16", "wb") as sink:
        def put(audio: np.ndarray, start: float, end: float) -> tuple[int, int, float]:
            nonlocal offset
            first = max(int((start - PAD_S) * RATE), 0)
            last = min(int((end + PAD_S) * RATE), len(audio))
            clip = audio[first:last]
            sink.write(clip.tobytes())
            where = offset
            offset += len(clip)
            return where, len(clip), first / RATE

        for path in episodes:
            folder = path.parent
            season, code = folder.parent.name, folder.name
            pinned_data = json.loads(path.read_text(encoding="utf-8"))
            lines = pinned_data["lines"]
            refined = folder / "diarization.diarizen-v2.refined.resnet293.json"
            turns = json.loads(refined.read_text(encoding="utf-8"))["turns"] if refined.exists() else []
            asr_path = next((folder / n for n in ("asr.whisper-large-v3.json", "asr.whisper-large-v3-turbo.json") if (folder / n).exists()))
            asr = json.loads(asr_path.read_text(encoding="utf-8"))
            words = [w for s in asr["segments"] for w in s.get("words") or []]
            audio = read_i16(folder / "audio.wav")
            for x in lines:
                if (season, x["person"]) not in keep:
                    stats["gold_rare_person"] += 1
                    continue
                where, n, t0 = put(audio, x["start"], x["end"])
                segments.append({"cls": class_id(season, x["person"]), "season": season, "ep": code, "kind": "gold", "off": where,
                                 "n": n, "t0": round(t0, 3), "start": x["start"], "end": x["end"],
                                 "words": x["words"], "other_voice_s": x["other_voice_s"]})
                stats["gold"] += 1
                stats["gold_s"] += int(x["end"] - x["start"])
            for x in pinned_data.get("cue_lines", []):  # not heard by Whisper: the cue's times, no word to centre on
                if (season, x["person"]) not in keep:
                    continue
                where, n, t0 = put(audio, x["start"], x["end"])
                segments.append({"cls": class_id(season, x["person"]), "season": season, "ep": code, "kind": "cue", "off": where,
                                 "n": n, "t0": round(t0, 3), "start": x["start"], "end": x["end"],
                                 "words": [], "other_voice_s": x["other_voice_s"]})
                stats["cue"] += 1
                stats["cue_s"] += int(x["end"] - x["start"])
            owners = cluster_owner(lines, args.purity, args.min_owner_lines)
            for seg in pseudo_turns(turns, owners, args.pseudo_cap_s, 1.0, 8.0, rng):
                if (season, seg["person"]) not in keep:
                    continue
                where, n, t0 = put(audio, seg["start"], seg["end"])
                segments.append({"cls": class_id(season, seg["person"]), "season": season, "ep": code, "kind": "pseudo", "off": where,
                                 "n": n, "t0": round(t0, 3), "start": seg["start"], "end": seg["end"], "words": []})
                stats["pseudo"] += 1
                stats["pseudo_s"] += int(seg["end"] - seg["start"])
            for a, b in noise_spans(turns, words, len(audio) / RATE, 60.0, rng):
                first, last = int(a * RATE), int(b * RATE)
                sink.write(audio[first:last].tobytes())
                noise.append({"off": offset, "n": last - first, "ep": code})
                offset += last - first
            print(f"=== {code} owned clusters {len(owners)}/{len({t['speaker'] for t in turns})}, segments {len(segments)}, "
                  f"{offset / RATE / 3600:.2f} h", flush=True)
    (out / "index.json").write_text(json.dumps({"segments": segments, "noise": noise, "classes": classes, "rate": RATE,
                                                "args": vars(args), "stats": dict(stats)}, ensure_ascii=False), encoding="utf-8")
    print(dict(stats), "classes", len(classes), "DONE", flush=True)


if __name__ == "__main__":
    main()
