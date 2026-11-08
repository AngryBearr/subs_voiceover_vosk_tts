"""Mix the Russian voice-over track over the original audio, ducking the original under the voice.

    subs_env/bin/python -m stt.mix EPISODE.mkv VOICE.wav --out ep.ru.wav [--duck-db 12] [--mux ep.ru.mkv --srt ep.ru.srt]

The original stays audible (music, crowd, overlapping voices) and is lowered only while the
voice-over speaks, with a smooth attack/release. `--mux` writes a video copy (video stream is
not re-encoded) with the voice-over mix as the first audio track, the original as the second
and optional Russian subtitles; it needs an `ffmpeg` binary (PATH or `imageio-ffmpeg`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import soundfile as sf

from stt.audio import load_audio
from utils.ffmpeg import ffmpeg_exe

log = logging.getLogger("stt.mix")

# Part of the key of a cached vocal stem: change it with the separator, its chunking or the resampling
SEPARATION_VERSION = 1
SEPARATOR_ID = "torchaudio.HDEMUCS_HIGH_MUSDB_PLUS;chunk=30;overlap=1"
_SEPARATION_LOCKS: dict[Path, threading.Lock] = {}
_SEPARATION_GUARD = threading.Lock()


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
    return out.astype(np.float32)  # peaks above 1.0 are handled by ``limit_peaks`` (never by scaling the whole track)


def limit_peaks(audio: np.ndarray, sample_rate: int, ceiling: float = 0.97, lookahead_s: float = 0.005) -> np.ndarray:
    """Look-ahead peak limiter: samples never exceed ``ceiling``, everything else stays as it is.

    The gain is ``min(1, ceiling / |peak over channels|)``, lowered ``lookahead_s`` before and after each
    peak (minimum filter) and smoothed (moving average no wider than that), so it reaches the reduced
    value before the peak arrives. Audio that never exceeds the ceiling is returned unchanged.
    """
    if audio.size == 0:
        return audio
    peak = np.abs(audio).max(axis=1) if audio.ndim == 2 else np.abs(audio)
    if float(peak.max()) <= ceiling:
        return audio
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    gain = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9)).astype(np.float32)
    del peak
    reach = max(1, int(lookahead_s * sample_rate))
    gain = minimum_filter1d(gain, 2 * reach + 1, mode="nearest")
    gain = uniform_filter1d(gain, reach + 1, mode="nearest")
    gain = np.minimum(gain, 1.0)
    return (audio * (gain[:, None] if audio.ndim == 2 else gain)).astype(np.float32)


def integrated_loudness(audio: np.ndarray, sample_rate: int) -> float:
    """Integrated loudness in LUFS (EBU R128) of (samples, channels) float audio, measured by ffmpeg's
    ``ebur128`` filter fed through a pipe; ``-inf`` for silence."""
    if audio.ndim == 1:
        audio = audio[:, None]
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostats", "-f", "f32le", "-ar", str(sample_rate),
           "-ac", str(audio.shape[1]), "-i", "-", "-af", "ebur128=peak=none", "-f", "null", "-"]
    data = np.ascontiguousarray(audio, dtype=np.float32)
    with tempfile.TemporaryFile() as err:  # ebur128 logs a line every 100 ms: a pipe could fill up and stall
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
        assert proc.stdin is not None
        step = sample_rate * 30
        try:
            for start in range(0, len(data), step):
                proc.stdin.write(data[start:start + step].tobytes())
            proc.stdin.close()
        except BrokenPipeError:
            pass
        code = proc.wait()
        err.seek(0)
        text = err.read().decode("utf-8", errors="replace")
    found = re.findall(r"^\s+I:\s+(-?inf|-?[\d.]+) LUFS", text, flags=re.MULTILINE)
    if code != 0 or not found:
        raise RuntimeError(f"ffmpeg ebur128 failed (code {code}): {text[-300:]}")
    return float(found[-1])


def finish_mix(audio: np.ndarray, sample_rate: int, loudness_lufs: float | None,
               ceiling: float = 0.97) -> tuple[np.ndarray, float | None, float]:
    """The last step of a mix: a static gain to ``loudness_lufs`` (None: none), then the peak limiter.
    Returns ``(audio, loudness before, gain applied in dB)``."""
    measured: float | None = None
    gain_db = 0.0
    if loudness_lufs is not None:
        measured = integrated_loudness(audio, sample_rate)
        if np.isfinite(measured):
            gain_db = loudness_lufs - measured
            audio = (audio * np.float32(10 ** (gain_db / 20))).astype(np.float32)
    return limit_peaks(audio, sample_rate, ceiling), measured, gain_db


def separation_path(cache_dir: Path, media: Path, start: float, duration: float | None, sample_rate: int) -> Path:
    """File of the cached vocal stem of an interval of ``media`` (decoded at ``sample_rate``): the key holds the media's
    identity (path, size, time), the interval, the rate, the separator and ``SEPARATION_VERSION``."""
    media = Path(media).resolve()
    stat = media.stat()
    key = json.dumps([str(media), stat.st_size, stat.st_mtime_ns, round(start, 3),
                      None if duration is None else round(duration, 3), sample_rate, SEPARATOR_ID,
                      SEPARATION_VERSION])
    return cache_dir / f"voices.{hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]}.npy"


def separated_voices(original: np.ndarray, sample_rate: int, cache: Path | None,
                     peaks_file: Path | None = None) -> np.ndarray:
    """The original's vocal stem (Hybrid Demucs) at ``sample_rate``, same shape as ``original``. With ``cache`` (a file
    from ``separation_path``) it is computed once: engines mixing at the same time wait for the first one and read its
    result; the file is written whole or not at all. If the card runs out of memory the separation is tried once more
    and then done on the CPU. The peak VRAM goes to ``peaks_file`` (see ``stt.gpu_budget``)."""
    with _SEPARATION_GUARD:
        lock = _SEPARATION_LOCKS.setdefault(cache, threading.Lock()) if cache else None
    with lock or nullcontext():
        if cache is not None and cache.exists():
            try:
                voices = np.load(cache, mmap_mode="r")
                if voices.shape == original.shape and voices.dtype == np.float32:
                    log.info("Original voices: reused %s", cache.name)
                    return voices
            except (OSError, ValueError):
                pass
            log.warning("Cached voices %s unreadable or of another shape: separating again", cache.name)
        from stt import gpu_budget
        from stt.gpu import VramPeak
        from stt.separate import VocalSeparator

        def separate(device: str) -> tuple[int, np.ndarray]:
            separator = VocalSeparator(device=device)
            return separator.rate, separator.separate_long(_resample(original, sample_rate, separator.rate).T)

        started = time.monotonic()
        with VramPeak(interval_s=0.25) as peak:
            rate, stem = gpu_budget.run_with_cpu_fallback(separate, "Voice separation")
        gpu_budget.record_peak(peaks_file, "demucs", peak.peak_mb)
        voices = _resample(stem.T, rate, sample_rate)[: len(original)]
        log.info("Separated the original voices (%.1f min) in %.1f s", len(original) / sample_rate / 60,
                 time.monotonic() - started)
        if cache is not None and voices.shape == original.shape:
            cache.parent.mkdir(parents=True, exist_ok=True)
            partial = cache.with_name(f"{cache.stem}.{os.getpid()}.{threading.get_ident()}.partial")
            try:
                with partial.open("wb") as file:
                    np.save(file, np.ascontiguousarray(voices, dtype=np.float32))
                partial.replace(cache)
            finally:
                partial.unlink(missing_ok=True)
        return voices


def encode_track(wav: Path, out: Path, bitrate: str = "192k") -> None:
    """A mixed WAV as an AAC track (``.m4a``): a tenth of the size, muxed into videos without re-encoding.
    The file appears only when complete (another engine's lane may be muxing the episode's video)."""
    partial = out.with_name(out.stem + ".partial" + out.suffix)
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-c:a", "aac",
                    "-b:a", bitrate, str(partial)], check=True)
    partial.replace(out)


def mux_tracks(video: Path, tracks: list[tuple[Path, str]], out: Path, srt: Path | None, start: float = 0.0,
               duration: float | None = None, srt_title: str = "Русские (озвученный текст)",
               video_source: Path | None = None, original_audio_kbps: int | None = None,
               onscreen_ass: Path | None = None, fonts: list[Path] | None = None,
               onscreen_title: str = "Надписи на экране") -> None:
    """Video copy with voice-over mixes as audio tracks (``(file, title)``, the first is the default),
    the original audio after them and optional Russian subtitles. WAV tracks are encoded to AAC,
    others copied; a test piece (``start``) is cut and its video re-encoded. The file appears only
    when complete. ``video_source`` (a file with a smaller video, same timeline; not for a test piece) replaces the
    video stream, copied as it is; ``original_audio_kbps`` re-encodes the original audio to stereo AAC at that rate.
    ``onscreen_ass`` (translations of on-screen text) becomes a default+forced subtitle track and the SRT track stops
    being the default; ``fonts`` are attached so players without them still draw the Cyrillic text. The ASS times are
    on the output's timeline: for a test piece write it with ``write_ass(..., offset_s=start)`` (it is not shifted here)."""
    piece = ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}"] if start and duration else []
    # A separate video file keeps its own start time: without -copyts ffmpeg moves each input to zero on its own
    keep_times = ["-copyts"] if video_source and not piece else []
    # AVI (The Amazing Race S1) stores no time stamps for its MP3 packets: matroska refuses them unless ffmpeg makes them
    stamps = ["-fflags", "+genpts"] if video.suffix.lower() == ".avi" else []
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *keep_times, *piece, *stamps, "-i", str(video)]
    for path, _ in tracks:
        cmd += ["-i", str(path)]
    if srt and not piece:  # subtitle times are on the episode timeline
        cmd += ["-i", str(srt)]
    else:
        srt = None
    next_input = len(tracks) + 1
    srt_input = next_input if srt else None
    next_input += 1 if srt else 0
    ass_input = None
    if onscreen_ass:
        cmd += ["-i", str(onscreen_ass)]
        ass_input, next_input = next_input, next_input + 1
    video_input = 0
    if video_source and not piece:
        video_input = next_input
        cmd += ["-i", str(video_source)]
    cmd += ["-map", f"{video_input}:v:0"]
    for n in range(len(tracks)):
        cmd += ["-map", f"{n + 1}:a:0"]
    cmd += ["-map", "0:a:0"]
    sub = 0  # index among the output's subtitle streams
    if srt_input:
        cmd += ["-map", f"{srt_input}:s:0", f"-c:s:{sub}", "srt", f"-metadata:s:s:{sub}", "language=rus",
                f"-metadata:s:s:{sub}", f"title={srt_title}"]
        if ass_input:
            cmd += [f"-disposition:s:{sub}", "0"]
        sub += 1
    if ass_input:
        cmd += ["-map", f"{ass_input}:s:0", f"-c:s:{sub}", "ass", f"-metadata:s:s:{sub}", "language=rus",
                f"-metadata:s:s:{sub}", f"title={onscreen_title}", f"-disposition:s:{sub}", "default+forced"]
    for n, font in enumerate(fonts or []):
        cmd += ["-attach", str(font), f"-metadata:s:t:{n}", "mimetype=application/x-truetype-font"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"] if piece else ["-c:v", "copy"]
    for n, (path, title) in enumerate(tracks):
        cmd += [f"-c:a:{n}", *(["aac", f"-b:a:{n}", "192k"] if path.suffix.lower() == ".wav" else ["copy"]),
                f"-metadata:s:a:{n}", "language=rus", f"-metadata:s:a:{n}", f"title={title}",
                f"-disposition:a:{n}", "default" if n == 0 else "0"]
    original = len(tracks)
    if original_audio_kbps and not piece:
        cmd += [f"-c:a:{original}", "aac", f"-b:a:{original}", f"{original_audio_kbps}k", f"-ac:a:{original}", "2"]
    else:
        cmd += [f"-c:a:{original}", "aac" if piece else "copy"]
    cmd += [f"-metadata:s:a:{original}", "title=Original",
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
        duration: float | None = None, onscreen_ass: Path | None = None, fonts: list[Path] | None = None) -> None:
    """Video copy with the mix as the first audio track; a test piece (``start``) is cut and re-encoded."""
    mux_tracks(video, [(mixed_wav, "Russian voice-over")], out, srt, start, duration, onscreen_ass=onscreen_ass,
               fonts=fonts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mix a voice-over track over the ducked original audio")
    parser.add_argument("media", help="Original video/audio")
    parser.add_argument("voice", help="Voice-over WAV aligned to the media timeline (e.g. voiceover export/track.wav)")
    parser.add_argument("--out", required=True, help="Mixed WAV")
    parser.add_argument("--start", type=float, default=0.0, help="Media start of the voice track, seconds")
    parser.add_argument("--voice-from-zero", action="store_true",
                        help="The voice track is on the media timeline from 0 (a test piece of stt.run): "
                             "its first --start seconds are dropped")
    parser.add_argument("--full-duration", action="store_true",
                        help="Keep the original audio through the end, even after the last dubbed line")
    parser.add_argument("--duck-db", type=float, default=12.0, help="How much to lower the original under speech")
    parser.add_argument("--duck-voices-only", action="store_true",
                        help="Lower only the original voices (Hybrid Demucs), keep music and effects")
    parser.add_argument("--rest-duck-db", type=float, default=3.0,
                        help="With --duck-voices-only: how much music and effects go down under speech")
    parser.add_argument("--separation-cache", default=None, metavar="DIR",
                        help="With --duck-voices-only: keep the separated voices in this folder and reuse them "
                             "(other engines of the same episode, a rerun)")
    parser.add_argument("--vram-peaks", default=None, metavar="FILE",
                        help="With --duck-voices-only: JSON file that gets the peak video memory of the separation")
    parser.add_argument("--voice-gain-db", type=float, default=0.0, help="Raise (or lower) the voice-over, dB")
    parser.add_argument("--loudness-lufs", type=float, default=None,
                        help="Bring the whole mix to this integrated loudness (default: leave the level as mixed)")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--mp3", action="store_true", help="Also write an MP3 next to the WAV")
    parser.add_argument("--mux", default=None, help="Write a video copy with the mix as the first audio track")
    parser.add_argument("--srt", default=None, help="Russian subtitles to add when muxing")
    parser.add_argument("--onscreen-ass", default=None,
                        help="ASS with translated on-screen text (stt.onscreen_render.write_ass) to add when muxing; "
                             "for a test piece (--start) it must be written with offset_s=start")
    parser.add_argument("--extra", action="append", default=[], metavar="WAV[:GAIN_DB]",
                        help="Another layer on the voice timeline, added after ducking (e.g. stt.crowd)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    t0 = time.monotonic()
    voice, voice_sr = sf.read(args.voice, dtype="float32", always_2d=False)
    if voice.ndim == 2:
        voice = voice.mean(axis=1)
    if args.voice_from_zero and args.start:
        voice = voice[int(args.start * voice_sr):]
    voice = _resample(voice, voice_sr, args.sample_rate)
    duration = len(voice) / args.sample_rate
    original = load_audio(args.media, start=args.start, duration=None if args.full_duration else duration,
                          sample_rate=args.sample_rate, channels=2)
    layers: list[tuple[np.ndarray, float]] = []
    for spec in args.extra:
        path, _, gain = spec.partition(":")
        layer, layer_sr = sf.read(path, dtype="float32", always_2d=False)
        layers.append((_resample(layer.mean(axis=1) if layer.ndim == 2 else layer, layer_sr, args.sample_rate),
                       float(gain or 0.0)))
    t_load = time.monotonic()
    voices = None
    if args.duck_voices_only:
        cache = separation_path(Path(args.separation_cache), Path(args.media), args.start,
                                None if args.full_duration else duration, args.sample_rate) \
            if args.separation_cache else None
        voices = separated_voices(original, args.sample_rate, cache,
                                  Path(args.vram_peaks) if args.vram_peaks else None)
    t_separated = time.monotonic()
    out = mix(original, voice, args.sample_rate, duck_db=args.duck_db, voice_gain_db=args.voice_gain_db,
              extra=layers, voices=voices, rest_duck_db=args.rest_duck_db)
    t_mixed = time.monotonic()
    out, measured, gain_db = finish_mix(out, args.sample_rate, args.loudness_lufs)
    t_finished = time.monotonic()
    if measured is not None:
        log.info("Loudness %.1f LUFS -> gain %+.1f dB to reach %.1f LUFS (peaks limited at 0.97)", measured, gain_db,
                 args.loudness_lufs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), out, args.sample_rate, subtype="PCM_16")
    if args.mp3:
        sf.write(str(out_path.with_suffix(".mp3")), out, args.sample_rate, format="MP3")
    log.info("Mixed %.1f s -> %s", len(out) / args.sample_rate, out_path)
    log.info("Mix timing: load %.0f s, separation %.0f s, mixing %.0f s, loudness+limiter %.0f s, wav write %.0f s",
             t_load - t0, t_separated - t_load, t_mixed - t_separated, t_finished - t_mixed,
             time.monotonic() - t_finished)
    if args.mux:
        fonts: list[Path] = []
        if args.onscreen_ass:
            from stt.onscreen_render import font_files

            fonts = list(font_files() or [])
        mux(Path(args.media), out_path, Path(args.mux), Path(args.srt) if args.srt else None,
            args.start, duration if args.start else None,
            onscreen_ass=Path(args.onscreen_ass) if args.onscreen_ass else None, fonts=fonts)
        log.info("Muxed -> %s", args.mux)


if __name__ == "__main__":
    main()
