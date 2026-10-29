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
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from stt.audio import load_audio
from utils.ffmpeg import ffmpeg_exe

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
        voice_gain_db: float = 0.0, extra: list[tuple[np.ndarray, float]] | None = None,
        voices: np.ndarray | None = None, rest_duck_db: float = 3.0) -> np.ndarray:
    """Stereo mix: original * ducking gain + voice (centered) + `extra` layers (not ducking).

    With ``voices`` (the original's vocal stem, same shape) only the voices are ducked by
    ``duck_db``; music and effects (original - voices) only by ``rest_duck_db``.
    """
    length = max([len(original), len(voice)] + [len(e) for e, _ in extra or []])
    original = np.pad(original, ((0, length - len(original)), (0, 0)))
    voice = np.pad(voice, (0, length - len(voice)))
    env = speech_envelope(voice, sample_rate)
    duck = 1.0 - (1.0 - 10 ** (-duck_db / 20)) * env
    if voices is not None:
        voices = np.pad(voices[:length], ((0, length - min(length, len(voices))), (0, 0)))
        duck_rest = 1.0 - (1.0 - 10 ** (-rest_duck_db / 20)) * env
        out = (original - voices) * duck_rest[:, None] + voices * duck[:, None]
    else:
        out = original * duck[:, None]
    out = out + (voice * 10 ** (voice_gain_db / 20))[:, None]
    for layer, gain_db in extra or []:
        out += (np.pad(layer, (0, length - len(layer))) * 10 ** (gain_db / 20))[:, None]
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.99:  # simple safety scaling instead of hard clipping
        out *= 0.99 / peak
    return out.astype(np.float32)


def encode_track(wav: Path, out: Path, bitrate: str = "192k") -> None:
    """A mixed WAV as an AAC track (``.m4a``): a tenth of the size, muxed into videos without re-encoding.
    The file appears only when complete (another engine's lane may be muxing the episode's video)."""
    partial = out.with_name(out.stem + ".partial" + out.suffix)
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-c:a", "aac",
                    "-b:a", bitrate, str(partial)], check=True)
    partial.replace(out)


def mux_tracks(video: Path, tracks: list[tuple[Path, str]], out: Path, srt: Path | None, start: float = 0.0,
               duration: float | None = None, srt_title: str = "Русские (озвученный текст)") -> None:
    """Video copy with voice-over mixes as audio tracks (``(file, title)``, the first is the default),
    the original audio after them and optional Russian subtitles. WAV tracks are encoded to AAC,
    others copied; a test piece (``start``) is cut and its video re-encoded. The file appears only
    when complete."""
    piece = ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}"] if start and duration else []
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *piece, "-i", str(video)]
    for path, _ in tracks:
        cmd += ["-i", str(path)]
    if srt and not piece:  # subtitle times are on the episode timeline
        cmd += ["-i", str(srt)]
    else:
        srt = None
    cmd += ["-map", "0:v:0"]
    for n in range(len(tracks)):
        cmd += ["-map", f"{n + 1}:a:0"]
    cmd += ["-map", "0:a:0"]
    if srt:
        cmd += ["-map", f"{len(tracks) + 1}:s:0", "-c:s", "srt", "-metadata:s:s:0", "language=rus",
                "-metadata:s:s:0", f"title={srt_title}"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"] if piece else ["-c:v", "copy"]
    for n, (path, title) in enumerate(tracks):
        cmd += [f"-c:a:{n}", *(["aac", f"-b:a:{n}", "192k"] if path.suffix.lower() == ".wav" else ["copy"]),
                f"-metadata:s:a:{n}", "language=rus", f"-metadata:s:a:{n}", f"title={title}",
                f"-disposition:a:{n}", "default" if n == 0 else "0"]
    original = len(tracks)
    cmd += [f"-c:a:{original}", "aac" if piece else "copy", f"-metadata:s:a:{original}", "title=Original",
            f"-disposition:a:{original}", "0"]
    partial = out.with_name(out.stem + ".partial" + out.suffix)
    subprocess.run([*cmd, str(partial)], check=True)
    partial.replace(out)


def streams(path: Path) -> list[tuple[str, str]]:
    """``(kind, title)`` of every stream of a media file (kind: video, audio, subtitle), as ffmpeg lists them."""
    info = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    found: list[tuple[str, str]] = []
    for line in info.splitlines():
        kind = re.match(r"\s*Stream #\d+:\d+(?:\[\w+\])?(?:\(\w+\))?: (Video|Audio|Subtitle)", line)
        if kind:
            found.append((kind.group(1).lower(), ""))
        title = re.match(r"\s*title\s*:\s*(.*)$", line)
        if title and found:
            found[-1] = (found[-1][0], title.group(1).strip())
    return found


def mux(video: Path, mixed_wav: Path, out: Path, srt: Path | None, start: float = 0.0,
        duration: float | None = None) -> None:
    """Video copy with the mix as the first audio track; a test piece (``start``) is cut and re-encoded."""
    mux_tracks(video, [(mixed_wav, "Russian voice-over")], out, srt, start, duration)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mix a voice-over track over the ducked original audio")
    parser.add_argument("media", help="Original video/audio")
    parser.add_argument("voice", help="Voice-over WAV aligned to the media timeline (e.g. voiceover export/track.wav)")
    parser.add_argument("--out", required=True, help="Mixed WAV")
    parser.add_argument("--start", type=float, default=0.0, help="Media start of the voice track, seconds")
    parser.add_argument("--voice-from-zero", action="store_true",
                        help="The voice track is on the media timeline from 0 (a test piece of stt.run): "
                             "its first --start seconds are dropped")
    parser.add_argument("--duck-db", type=float, default=12.0, help="How much to lower the original under speech")
    parser.add_argument("--duck-voices-only", action="store_true",
                        help="Lower only the original voices (Hybrid Demucs), keep music and effects")
    parser.add_argument("--rest-duck-db", type=float, default=3.0,
                        help="With --duck-voices-only: how much music and effects go down under speech")
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
    if args.voice_from_zero and args.start:
        voice = voice[int(args.start * voice_sr):]
    voice = _resample(voice, voice_sr, args.sample_rate)
    duration = len(voice) / args.sample_rate
    original = load_audio(args.media, start=args.start, duration=duration, sample_rate=args.sample_rate, channels=2)
    layers: list[tuple[np.ndarray, float]] = []
    for spec in args.extra:
        path, _, gain = spec.partition(":")
        layer, layer_sr = sf.read(path, dtype="float32", always_2d=False)
        layers.append((_resample(layer.mean(axis=1) if layer.ndim == 2 else layer, layer_sr, args.sample_rate),
                       float(gain or 0.0)))
    voices = None
    if args.duck_voices_only:
        from stt.separate import VocalSeparator

        separator = VocalSeparator()
        stem = separator.separate_long(_resample(original, args.sample_rate, separator.rate).T)
        voices = _resample(stem.T, separator.rate, args.sample_rate)[: len(original)]
        log.info("Separated the original voices (%.1f min)", len(original) / args.sample_rate / 60)
    out = mix(original, voice, args.sample_rate, duck_db=args.duck_db, voice_gain_db=args.voice_gain_db,
              extra=layers, voices=voices, rest_duck_db=args.rest_duck_db)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), out, args.sample_rate, subtype="PCM_16")
    if args.mp3:
        sf.write(str(out_path.with_suffix(".mp3")), out, args.sample_rate, format="MP3")
    log.info("Mixed %.1f s -> %s", duration, out_path)
    if args.mux:
        mux(Path(args.media), out_path, Path(args.mux), Path(args.srt) if args.srt else None,
            args.start, duration if args.start else None)
        log.info("Muxed -> %s", args.mux)


if __name__ == "__main__":
    main()
