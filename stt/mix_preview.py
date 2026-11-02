"""Hear the voice-over mix on a short piece of an episode before mixing the whole thing.

    subs_env/bin/python -m stt.mix_preview EPISODE_DIR MEDIA --engine f5 --start 600 --out piece.wav

``render_preview`` mixes the piece the way ``stt.mix`` does (same ducking, voice gain, loudness target and
peak limiter) from the episode's ``voiceover.<engine>/export/track.wav``; ``loudest_spot`` finds the place where
the voice-over has the hardest time against music and effects, the best piece to judge the settings on.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from stt.audio import load_audio
from stt.mix import _resample, finish_mix, mix
from utils.ffmpeg import ffmpeg_exe

log = logging.getLogger("stt.mix_preview")

RATE = 48000
SPOT_RATE = 16000
FRAME_S = 0.4
SPEECH_DB = -45.0  # a frame of the voice-over louder than this counts as speech
MIN_FREE_GB = 4.0  # GPU memory the voice separation of a short piece needs


def voice_track(episode_dir: Path, engine: str) -> Path:
    return episode_dir / f"voiceover.{engine}" / "export" / "track.wav"


def _read_voice(path: Path, start_s: float, seconds: float, rate: int) -> np.ndarray:
    """Mono voice-over of the piece (the track is on the episode timeline from 0) at ``rate``."""
    with sf.SoundFile(str(path)) as f:
        first = min(int(start_s * f.samplerate), f.frames)
        f.seek(first)
        voice = f.read(min(int(seconds * f.samplerate), f.frames - first), dtype="float32",
                       always_2d=True).mean(axis=1)
        return _resample(voice, f.samplerate, rate) if f.samplerate != rate else voice


def _separator_device() -> str:
    """The graphics card when it has room next to a running job, else the processor."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / 2 ** 30 >= MIN_FREE_GB:
            return "cuda"
    except Exception:  # noqa: BLE001 - any trouble with the card: use the processor
        pass
    return "cpu"


def render_preview(episode_dir: Path, media: Path, engine: str, start_s: float, seconds: float,
                   voice_gain_db: float, duck_db: float, rest_duck_db: float, loudness_lufs: float | None,
                   out: Path, duck: str | None = None) -> Path:
    """Mix ``seconds`` of the episode from ``start_s`` and write them to the WAV ``out``.

    ``duck``: "voices" (only the original voices go down, needs the voice separation) or "all"; default
    ``mix.duck`` of the config."""
    track = voice_track(episode_dir, engine)
    if not track.is_file():
        raise FileNotFoundError(f"No voice-over track of {engine}: {track}")
    if duck is None:
        from stt.run import REPO_ROOT, load_config

        duck = str((load_config(REPO_ROOT / "stt_config.json").get("mix") or {}).get("duck", "voices"))
    original = load_audio(media, start=start_s, duration=seconds, sample_rate=RATE, channels=2)
    voice = _read_voice(track, start_s, seconds, RATE)
    stem = None
    if duck == "voices":
        from stt.separate import VocalSeparator

        separator = VocalSeparator(device=_separator_device())
        stem = _resample(separator.separate_long(_resample(original, RATE, separator.rate).T).T,
                         separator.rate, RATE)[: len(original)]
    mixed = mix(original, voice, RATE, duck_db=duck_db, voice_gain_db=voice_gain_db, voices=stem,
                rest_duck_db=rest_duck_db)
    mixed, measured, gain_db = finish_mix(mixed, RATE, loudness_lufs)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), mixed, RATE, subtype="PCM_16")
    log.info("Preview %.0f s from %.0f s -> %s (loudness %s, gain %+.1f dB)", seconds, start_s, out,
             "-" if measured is None else f"{measured:.1f} LUFS", gain_db)
    return out


def existing_piece(m4a: Path, start_s: float, seconds: float, out: Path) -> Path:
    """The same piece cut from the episode's finished voice-over track (what it sounds like now)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start_s:.3f}", "-t",
                    f"{seconds:.3f}", "-i", str(m4a), "-c:a", "pcm_s16le", str(out)], check=True)
    return out


def _frame_db(audio: np.ndarray) -> np.ndarray:
    frame = int(SPOT_RATE * FRAME_S)
    n = len(audio) // frame
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms)


def best_window(voice_db: np.ndarray, bed_db: np.ndarray, seconds: float) -> float:
    """Start (s) of the window where the voice speaks most and the original under it is loudest:
    the sum over speech frames of the original's level above -60 dB."""
    n = min(len(voice_db), len(bed_db))
    score = np.where(voice_db[:n] > SPEECH_DB, np.maximum(bed_db[:n] + 60.0, 0.0), 0.0)
    width = max(1, min(n, int(round(seconds / FRAME_S))))
    cumulative = np.concatenate([[0.0], np.cumsum(score)])
    sums = cumulative[width:] - cumulative[:-width]
    return round(float(np.argmax(sums)) * FRAME_S, 1)


def loudest_spot(episode_dir: Path, media: Path, engine: str, seconds: float = 45.0) -> float:
    """Start time (s) of the ``seconds`` window with the most speech over the loudest original sound.
    Cached in ``mix_spot.<engine>.json`` (per length, per voice track version)."""
    track = voice_track(episode_dir, engine)
    cache = episode_dir / f"mix_spot.{engine}.json"
    key, stamp = str(seconds), track.stat().st_mtime
    saved = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
    if saved.get("track_mtime") == stamp and key in saved.get("spots", {}):
        return float(saved["spots"][key])
    voice = _read_voice(track, 0.0, 1e9, SPOT_RATE)
    original = load_audio(media, sample_rate=SPOT_RATE, channels=1)
    spot = best_window(_frame_db(voice), _frame_db(original), seconds)
    spots = saved.get("spots", {}) if saved.get("track_mtime") == stamp else {}
    cache.write_text(json.dumps({"track_mtime": stamp, "spots": {**spots, key: spot}}), encoding="utf-8")
    return spot


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render a short preview of the voice-over mix")
    parser.add_argument("episode", help="Episode work folder")
    parser.add_argument("media", help="Original video")
    parser.add_argument("--engine", default="f5")
    parser.add_argument("--start", type=float, default=None, help="Seconds; default: the loudest spot")
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--voice-gain-db", type=float, default=3.0)
    parser.add_argument("--duck-db", type=float, default=12.0)
    parser.add_argument("--rest-duck-db", type=float, default=6.0)
    parser.add_argument("--loudness-lufs", type=float, default=None)
    parser.add_argument("--duck", choices=["voices", "all"], default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    episode, media = Path(args.episode), Path(args.media)
    start = args.start if args.start is not None else loudest_spot(episode, media, args.engine, args.seconds)
    render_preview(episode, media, args.engine, start, args.seconds, args.voice_gain_db, args.duck_db,
                   args.rest_duck_db, args.loudness_lufs, Path(args.out), args.duck)


if __name__ == "__main__":
    main()
