"""Mix the Russian voice-over track over the original audio, ducking the original under the voice.

    subs_env/bin/python -m stt.mix EPISODE.mkv VOICE.wav --out ep.ru.wav [--duck-db 12] [--mux ep.ru.mkv --srt ep.ru.srt]

The original stays audible (music, crowd, overlapping voices) and is lowered only while the
voice-over speaks, with a smooth attack/release. `--mux` writes a video copy (video stream is
not re-encoded) with the voice-over mix as the first audio track, the original as the second
and optional Russian subtitles; it needs an `ffmpeg` binary (PATH or `imageio-ffmpeg`).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from stt.audio import load_audio

log = logging.getLogger("stt.mix")


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return audio
    import torch
    import torchaudio

    tensor = torch.from_numpy(np.ascontiguousarray(audio.T if audio.ndim == 2 else audio[None]))
    out = torchaudio.functional.resample(tensor, src, dst).numpy()
    return out.T if audio.ndim == 2 else out[0]


def speech_envelope(voice: np.ndarray, sample_rate: int, *, threshold_db: float = -45.0,
                    attack_s: float = 0.05, release_s: float = 0.3, hold_s: float = 0.25) -> np.ndarray:
    """0..1 per sample: 1 where the voice-over speaks (with hold), smoothed attack/release."""
    hop = int(sample_rate * 0.01)
    frames = len(voice) // hop + 1
    padded = np.pad(voice, (0, frames * hop - len(voice)))
    rms = np.sqrt(np.mean(padded.reshape(frames, hop) ** 2, axis=1) + 1e-12)
    active = (20 * np.log10(rms) > threshold_db).astype(np.float32)
    hold = int(hold_s / 0.01)
    if hold:  # keep the duck through short gaps between words
        active = (np.convolve(active, np.ones(hold, np.float32))[: len(active)] > 0).astype(np.float32)
    env = np.zeros_like(active)
    up, down = 0.01 / attack_s, 0.01 / release_s
    level = 0.0
    for i, target in enumerate(active):
        level = min(target, level + up) if target > level else max(target, level - down)
        env[i] = level
    return np.repeat(env, hop)[: len(voice)]


def mix(original: np.ndarray, voice: np.ndarray, sample_rate: int, *, duck_db: float = 12.0,
        voice_gain_db: float = 0.0, extra: list[tuple[np.ndarray, float]] | None = None) -> np.ndarray:
    """Stereo mix: original * ducking gain + voice (centered) + `extra` layers (not ducking)."""
    length = max([len(original), len(voice)] + [len(e) for e, _ in extra or []])
    original = np.pad(original, ((0, length - len(original)), (0, 0)))
    voice = np.pad(voice, (0, length - len(voice)))
    env = speech_envelope(voice, sample_rate)
    duck = 1.0 - (1.0 - 10 ** (-duck_db / 20)) * env
    out = original * duck[:, None] + (voice * 10 ** (voice_gain_db / 20))[:, None]
    for layer, gain_db in extra or []:
        out += (np.pad(layer, (0, length - len(layer))) * 10 ** (gain_db / 20))[:, None]
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.99:  # simple safety scaling instead of hard clipping
        out *= 0.99 / peak
    return out.astype(np.float32)


def find_ffmpeg() -> str | None:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def mux(video: Path, mixed_wav: Path, out: Path, srt: Path | None) -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found: install it or `uv pip install imageio-ffmpeg` in subs_env")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-i", str(mixed_wav)]
    if srt:
        cmd += ["-i", str(srt)]
    cmd += ["-map", "0:v:0", "-map", "1:a:0", "-map", "0:a:0"]
    if srt:
        cmd += ["-map", "2:s:0", "-c:s", "srt", "-metadata:s:s:0", "language=rus"]
    cmd += ["-c:v", "copy", "-c:a:0", "aac", "-b:a:0", "192k", "-c:a:1", "copy",
            "-metadata:s:a:0", "language=rus", "-metadata:s:a:0", "title=Russian voice-over",
            "-metadata:s:a:1", "title=Original", "-disposition:a:0", "default", "-disposition:a:1", "0",
            str(out)]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mix a voice-over track over the ducked original audio")
    parser.add_argument("media", help="Original video/audio")
    parser.add_argument("voice", help="Voice-over WAV aligned to the media timeline (e.g. voiceover export/track.wav)")
    parser.add_argument("--out", required=True, help="Mixed WAV")
    parser.add_argument("--start", type=float, default=0.0, help="Media start of the voice track, seconds")
    parser.add_argument("--duck-db", type=float, default=12.0, help="How much to lower the original under speech")
    parser.add_argument("--voice-gain-db", type=float, default=0.0)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--mp3", action="store_true", help="Also write an MP3 next to the WAV")
    parser.add_argument("--mux", default=None, help="Write a video copy with the mix as the first audio track")
    parser.add_argument("--srt", default=None, help="Russian subtitles to add when muxing")
    parser.add_argument("--extra", action="append", default=[], metavar="WAV[:GAIN_DB]",
                        help="Another layer on the voice timeline, added after ducking (e.g. stt.crowd)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    voice, voice_sr = sf.read(args.voice, dtype="float32", always_2d=False)
    if voice.ndim == 2:
        voice = voice.mean(axis=1)
    voice = _resample(voice, voice_sr, args.sample_rate)
    duration = len(voice) / args.sample_rate
    original = load_audio(args.media, start=args.start, duration=duration, sample_rate=args.sample_rate, channels=2)
    layers: list[tuple[np.ndarray, float]] = []
    for spec in args.extra:
        path, _, gain = spec.partition(":")
        layer, layer_sr = sf.read(path, dtype="float32", always_2d=False)
        layers.append((_resample(layer.mean(axis=1) if layer.ndim == 2 else layer, layer_sr, args.sample_rate),
                       float(gain or 0.0)))
    out = mix(original, voice, args.sample_rate, duck_db=args.duck_db, voice_gain_db=args.voice_gain_db,
              extra=layers)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), out, args.sample_rate, subtype="PCM_16")
    if args.mp3:
        sf.write(str(out_path.with_suffix(".mp3")), out, args.sample_rate, format="MP3")
    log.info("Mixed %.1f s -> %s", duration, out_path)
    if args.mux:
        mux(Path(args.media), out_path, Path(args.mux), Path(args.srt) if args.srt else None)
        log.info("Muxed -> %s", args.mux)


if __name__ == "__main__":
    main()
