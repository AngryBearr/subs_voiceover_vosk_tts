"""Make the video stream smaller (same resolution) so an episode fits a size, e.g. 2 GB instead of 6.

    subs_env/bin/python -m stt.video_compress FINISHED.mkv [--target-gb 2] [--codec hevc_nvenc] [--out small.mkv]
    subs_env/bin/python -m stt.video_compress FINISHED.mkv --sample-minute 10 --sample-seconds 60   # a piece to judge

The picture is encoded through PyAV (by default on the graphics card: ``hevc_nvenc`` / ``h264_nvenc``) with the
source's own timestamps, so the audio stays in sync. Every audio and subtitle stream of the file is kept (titles,
languages, default flags); AAC tracks are copied, others (the 5.1 original) become stereo AAC. The input is never
changed: the result is ``<stem>.small.mkv`` next to it (or ``--out``).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Callable

import av

from utils.ffmpeg import ffmpeg_exe

log = logging.getLogger("stt.video_compress")

CONTAINER_OVERHEAD = 0.02  # mkv headers, cues: kept out of the video budget
KEEP_MARGIN = 0.15  # the video is left as it is unless the budget is this much under its own bitrate
NVENC_OPTIONS = {"preset": "p6", "tune": "hq", "rc": "vbr", "multipass": "fullres", "spatial-aq": "1",
                 "temporal-aq": "1", "rc-lookahead": "32"}
SOFT_CODECS = {"hevc_nvenc": "libx265", "h264_nvenc": "libx264", "av1_nvenc": "libx265"}
SOFT_OPTIONS = {"libx265": {"preset": "medium", "x265-params": "log-level=error"}, "libx264": {"preset": "slow"}}
AUDIO_COPY_KBPS = 192  # a copied AAC track without a known bitrate


def video_bitrate(duration_s: float, target_gb: float, audio_kbps_total: float, source_bitrate: float,
                  min_kbps: int = 800) -> int | None:
    """Video bitrate in kbps that makes a file of ``target_gb`` (1 GB = 10^9 bytes) with ``audio_kbps_total`` of
    audio (all tracks together). ``source_bitrate`` is the video's own bitrate in kbps. None means "keep the
    video as it is": the budget is not at least 15% under it. Never below ``min_kbps``."""
    if duration_s <= 0:
        raise ValueError("duration must be positive")
    total_kbps = target_gb * 1e9 * 8 / duration_s / 1000 * (1 - CONTAINER_OVERHEAD)
    kbps = max(int(total_kbps - audio_kbps_total), min_kbps)
    if kbps > source_bitrate * (1 - KEEP_MARGIN):
        return None
    return kbps


def probe(path: Path) -> dict[str, float]:
    """``duration`` (s), ``video_kbps``, ``width``, ``height`` of a file; the video bitrate is the stream's own when
    the container states it, else the file's minus its audio."""
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        if not duration and video.duration:
            duration = float(video.duration * video.time_base)
        audio_kbps = sum((s.bit_rate or AUDIO_COPY_KBPS * 1000) / 1000 for s in container.streams.audio)
        if video.bit_rate:
            video_kbps = video.bit_rate / 1000
        else:
            video_kbps = max(path.stat().st_size * 8 / max(duration, 1) / 1000 - audio_kbps, 0.0)
        return {"duration": duration, "video_kbps": video_kbps, "width": video.width, "height": video.height}


def _encode(src: Path, partial: Path, bitrate: int, codec: str, start_s: float | None, seconds: float | None,
            on_progress: Callable[[float], None] | None) -> tuple[int, int]:
    """One encoding attempt with ``codec``; (frames, bytes). Frames keep the source timestamps (a piece starts at 0)."""
    with av.open(str(src)) as source, av.open(str(partial), "w", format="matroska") as target:
        video = source.streams.video[0]
        video.thread_type = "AUTO"
        rate = video.average_rate or video.guessed_rate or Fraction(25)
        options = NVENC_OPTIONS if codec.endswith("_nvenc") else SOFT_OPTIONS.get(codec, {})
        out = target.add_stream(codec, rate=rate, options=dict(options))
        out.width, out.height, out.pix_fmt = video.width, video.height, "yuv420p"
        out.bit_rate = bitrate * 1000
        out.codec_context.time_base = video.time_base  # frames keep the source's pts unchanged
        total = float(video.duration * video.time_base) if video.duration else (source.duration or 0) / av.time_base
        begin = start_s or 0.0
        shift = int(round(begin / video.time_base))  # a piece starts at 0 (the audio of the piece does too)
        if seconds:
            total = seconds
        if begin:
            source.seek(int(begin / video.time_base), stream=video, backward=True)
        frames = 0
        for frame in source.decode(video):
            if frame.pts is None:
                continue
            at = float(frame.pts * video.time_base)
            if at < begin - 1e-6:
                continue
            if seconds and at >= begin + seconds:
                break
            frame = frame.reformat(format="yuv420p")
            frame.pts, frame.time_base = frame.pts - shift, video.time_base
            frame.pict_type = av.video.frame.PictureType.NONE
            for packet in out.encode(frame):
                target.mux(packet)
            frames += 1
            if on_progress and frames % 240 == 0 and total:
                on_progress(min((at - begin) / total, 1.0))
        for packet in out.encode():
            target.mux(packet)
    return frames, partial.stat().st_size


def encode_video(src: Path, out: Path, bitrate: int, codec: str = "hevc_nvenc", start_s: float | None = None,
                 seconds: float | None = None, on_progress: Callable[[float], None] | None = None) -> Path:
    """The video stream of ``src`` alone, same resolution, ``bitrate`` kbps, source timestamps (a piece cut with
    ``start_s`` / ``seconds`` is shifted to start at 0). Written as ``<out>.partial`` and renamed when complete.
    If the encoder (e.g. NVENC) fails to work, the software one (libx265 / libx264) is used and a warning logged."""
    partial = out.with_name(out.stem + ".partial" + out.suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        frames, size = _encode(src, partial, bitrate, codec, start_s, seconds, on_progress)
    except (av.error.FFmpegError, ValueError, OSError) as error:
        soft = SOFT_CODECS.get(codec)
        if soft is None:
            raise
        log.warning("%s failed (%s); encoding with %s on the processor instead (slower)", codec, error, soft)
        codec = soft
        frames, size = _encode(src, partial, bitrate, codec, start_s, seconds, on_progress)
    partial.replace(out)
    elapsed = time.perf_counter() - started
    log.info("%s: %d frames, %.0f fps, %.2f GB at %d kbps (%s)", out.name, frames, frames / max(elapsed, 1e-9),
             size / 1e9, bitrate, codec)
    return out


def keep_streams(path: Path, copy_aac: bool = True, audio_kbps: int = 160) -> tuple[list[str], int]:
    """ffmpeg arguments that keep every audio and subtitle stream of ``path`` (input 0) and the audio
    bitrate (kbps) the result will have. AAC tracks are copied (not the one titled Original), others
    become stereo AAC at ``audio_kbps``; titles, languages and flags travel with the streams."""
    args: list[str] = []
    kbps = 0
    with av.open(str(path)) as container:
        for n, stream in enumerate(container.streams.audio):
            title = (stream.metadata.get("title") or "").lower()
            if copy_aac and stream.codec_context.name == "aac" and title != "original":
                args += ["-map", f"0:a:{n}", f"-c:a:{n}", "copy"]
                kbps += int((stream.bit_rate or AUDIO_COPY_KBPS * 1000) / 1000)
            else:
                args += ["-map", f"0:a:{n}", f"-c:a:{n}", "aac", f"-b:a:{n}", f"{audio_kbps}k", f"-ac:a:{n}", "2"]
                kbps += audio_kbps
        for n in range(len(container.streams.subtitles)):
            args += ["-map", f"0:s:{n}"]
    return args + ["-c:s", "copy"], kbps


def shrink(src: Path, out: Path, target_gb: float = 2.0, codec: str = "hevc_nvenc", audio_kbps: int = 160,
           sample_minute: float | None = None, sample_seconds: float = 60.0,
           on_progress: Callable[[float], None] | None = None) -> Path | None:
    """A copy of ``src`` with a smaller video; None (nothing written) when the video is already small enough.
    With ``sample_minute`` only ``sample_seconds`` from that minute, to judge the quality. ``src`` stays as it is."""
    if out.resolve() == src.resolve():
        raise ValueError("the output must not be the input")
    info = probe(src)
    audio_args, audio_kbps_total = keep_streams(src, audio_kbps=audio_kbps)
    bitrate = video_bitrate(info["duration"], target_gb, audio_kbps_total, info["video_kbps"])
    if bitrate is None:
        log.info("%s: the video (%.0f kbps) is already close to %.1f GB; nothing to do", src.name, info["video_kbps"],
                 target_gb)
        return None
    log.info("%s: %.0fx%.0f, video %.0f -> %d kbps for %.1f GB (audio %d kbps)", src.name, info["width"],
             info["height"], info["video_kbps"], bitrate, target_gb, audio_kbps_total)
    begin = sample_minute * 60 if sample_minute is not None else None
    piece = ["-ss", f"{begin:.3f}", "-t", f"{sample_seconds:.3f}"] if begin is not None else []
    video = out.with_name(out.stem + f".video{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        encode_video(src, video, bitrate, codec, begin, sample_seconds if begin is not None else None, on_progress)
        partial = out.with_name(out.stem + ".partial" + out.suffix)
        # A whole file keeps every stream's own start time: without -copyts ffmpeg moves each input to zero on its own
        keep_times = [] if piece else ["-copyts"]
        subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *keep_times, *piece, "-i", str(src),
                        "-i", str(video), "-map", "1:v:0", "-c:v", "copy", *audio_args, "-map_metadata", "0",
                        str(partial)], check=True)
        partial.replace(out)
    finally:
        video.unlink(missing_ok=True)
    log.info("%s: %.2f GB (was %.2f GB)", out, out.stat().st_size / 1e9, src.stat().st_size / 1e9)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Shrink the video stream of a finished file, keeping its resolution")
    parser.add_argument("video", help="Finished video (never changed)")
    parser.add_argument("--target-gb", type=float, default=2.0, help="Size of the result, GB (10^9 bytes)")
    parser.add_argument("--codec", default="hevc_nvenc", choices=["hevc_nvenc", "h264_nvenc", "libx265", "libx264"])
    parser.add_argument("--audio-kbps", type=int, default=160, help="Bitrate of re-encoded (non-AAC) audio tracks")
    parser.add_argument("--out", default=None, help="Default: <stem>.small.mkv next to the input (<stem>.sample.mkv)")
    parser.add_argument("--sample-minute", type=float, default=None, help="Only a piece from this minute, to judge quality")
    parser.add_argument("--sample-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    src = Path(args.video)
    if not src.is_file():
        raise SystemExit(f"Video not found: {src}")
    suffix = "sample" if args.sample_minute is not None else "small"
    out = Path(args.out) if args.out else src.with_name(f"{src.stem}.{suffix}.mkv")
    last = [-1]

    def progress(fraction: float) -> None:
        if int(fraction * 20) != last[0]:
            last[0] = int(fraction * 20)
            log.info("encoded %d%%", int(fraction * 100))

    shrink(src, out, args.target_gb, args.codec, args.audio_kbps, args.sample_minute, args.sample_seconds, progress)


if __name__ == "__main__":
    main()
