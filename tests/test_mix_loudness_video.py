"""Voice level, loudness target, peak limiter, mix preview and video compression (synthetic data only)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import av
import numpy as np
import pytest
import soundfile as sf

from stt.mix import integrated_loudness, limit_peaks, mix, mux_tracks, streams
from stt.mix_preview import best_window, loudest_spot, render_preview
from stt.video_compress import encode_video, video_bitrate
from utils.ffmpeg import ffmpeg_exe


def _sine(seconds: float, rate: int, db: float, channels: int = 2) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    wave = (10 ** (db / 20) * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    return np.stack([wave] * channels, axis=1)


def test_limiter_never_exceeds_ceiling_and_keeps_quiet_audio() -> None:
    rate = 48000
    audio = _sine(1.0, rate, -6.0)
    audio[20000:20010] *= 3.0  # a short loud burst
    limited = limit_peaks(audio, rate, ceiling=0.97)
    assert float(np.abs(limited).max()) <= 0.97 + 1e-6
    assert np.allclose(limited[:15000], audio[:15000])  # far from the burst: untouched
    quiet = _sine(0.5, rate, -20.0)
    assert limit_peaks(quiet, rate) is quiet


def test_integrated_loudness_of_a_sine() -> None:
    # a stereo 1 kHz sine at -20 dBFS peak per channel: -20 - 3.01 (RMS) + 3.01 (two channels) - 0.69 (K-weighting) = -20.7
    measured = integrated_loudness(_sine(5.0, 48000, -20.0), 48000)
    assert abs(measured - (-20.7)) < 1.0


def test_voice_gain_raises_the_voice_without_scaling_the_track() -> None:
    rate = 1000
    n = 4 * rate
    original = np.full((n, 2), 0.9, dtype=np.float32)
    voice = np.zeros(n, dtype=np.float32)
    voice[rate:3 * rate] = 0.2
    plain = mix(original, voice, rate, duck_db=12.0)
    louder = mix(original, voice, rate, duck_db=12.0, voice_gain_db=6.0)
    mid = slice(int(1.8 * rate), int(2.2 * rate))
    assert float((louder[mid, 0] - plain[mid, 0]).mean()) == pytest.approx(0.2 * (10 ** (6 / 20) - 1), abs=0.01)
    assert np.allclose(louder[: rate // 2], original[: rate // 2])  # nothing scaled outside the speech
    loud_voice = mix(original, voice * 10, rate, duck_db=12.0)
    assert float(loud_voice.max()) > 1.0  # peaks are the limiter's job, not a whole-track scale


def test_video_bitrate_cases() -> None:
    hour = 3600.0
    # 2 GB for an hour with 1 Mb/s of audio: (4444 kbps - 2% overhead) - 1000
    assert video_bitrate(hour, 2.0, 1000, 10000) == pytest.approx(3355, abs=2)
    assert video_bitrate(hour, 2.0, 1000, 3800) is None  # not 15% under the source: keep it
    assert video_bitrate(hour, 2.0, 1000, 3355 / 0.85 + 5) is not None
    assert video_bitrate(hour, 0.1, 1000, 10000) == 800  # never below the floor
    assert video_bitrate(hour, 0.1, 1000, 900) is None
    with pytest.raises(ValueError):
        video_bitrate(0, 2.0, 100, 1000)


def _make_video(path: Path, seconds: int = 2, audio: bool = True) -> None:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
           f"testsrc=size=320x240:rate=25:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", *(["-c:a", "aac"] if audio else []), str(path)]
    subprocess.run(cmd, check=True)


def _video_times(path: Path) -> tuple[int, int, int, int, int]:
    with av.open(str(path)) as container:
        v = container.streams.video[0]
        pts = [p.pts for p in container.demux(v) if p.pts is not None]
        return v.width, v.height, len(pts), min(pts), max(pts)


def test_encode_video_keeps_size_frames_and_timestamps(tmp_path: Path) -> None:
    src = tmp_path / "src.mkv"
    _make_video(src, audio=False)
    out = encode_video(src, tmp_path / "small.mkv", 300, codec="libx265")
    assert out.exists() and not (tmp_path / "small.partial.mkv").exists()
    assert _video_times(out) == _video_times(src)


def test_encode_video_piece_starts_at_zero(tmp_path: Path) -> None:
    src = tmp_path / "src.mkv"
    _make_video(src, seconds=3, audio=False)
    out = encode_video(src, tmp_path / "piece.mkv", 300, codec="libx265", start_s=1.0, seconds=1.0)
    width, height, frames, first, last = _video_times(out)
    assert (width, height, frames, first) == (320, 240, 25, 0)
    assert last == 960


def test_mux_tracks_with_video_source_and_stereo_original(tmp_path: Path) -> None:
    src = tmp_path / "src.mkv"
    _make_video(src)
    small = encode_video(src, tmp_path / "small.mkv", 300, codec="libx265")
    voice = tmp_path / "voice.wav"
    sf.write(str(voice), _sine(2.0, 48000, -20.0), 48000)
    out = tmp_path / "out.mkv"
    mux_tracks(src, [(voice, "Русский")], out, None, video_source=small, original_audio_kbps=96)
    assert streams(out) == [("video", ""), ("audio", "Русский"), ("audio", "Original")]
    with av.open(str(out)) as container:
        assert container.streams.video[0].codec_context.name == "hevc"
        assert container.streams.audio[1].codec_context.name == "aac"
    made, used = _video_times(out), _video_times(small)
    assert made[2] == used[2]  # every frame
    assert made[4] - made[3] == used[4] - used[3]  # the same spacing; AAC's start may move all streams together
    with av.open(str(out)) as container:
        video, audio = container.streams.video[0], container.streams.audio[0]
        lead = float(video.start_time * video.time_base) - float(audio.start_time * audio.time_base)
    assert abs(lead - (used[3] / 1000)) < 0.03  # the video keeps its place against the audio


def test_best_window_prefers_speech_over_loud_bed() -> None:
    voice = np.full(100, -60.0)
    bed = np.full(100, -40.0)
    voice[60:80] = -20.0
    bed[60:80] = -10.0
    voice[10:30] = -20.0  # speech over a quieter bed
    assert best_window(voice, bed, seconds=8.0) == pytest.approx(60 * 0.4, abs=0.5)


def test_preview_on_a_synthetic_episode(tmp_path: Path) -> None:
    media = tmp_path / "ep.mkv"
    _make_video(media, seconds=6)
    episode = tmp_path / "work"
    track = episode / "voiceover.f5" / "export" / "track.wav"
    track.parent.mkdir(parents=True)
    voice = np.zeros(6 * 24000, dtype=np.float32)
    voice[2 * 24000:4 * 24000] = 0.1 * np.sin(np.arange(2 * 24000) * 0.1)
    sf.write(str(track), voice, 24000)
    spot = loudest_spot(episode, media, "f5", seconds=2.0)
    assert 1.5 <= spot <= 2.5
    assert (episode / "mix_spot.f5.json").exists() and loudest_spot(episode, media, "f5", seconds=2.0) == spot
    quiet = render_preview(episode, media, "f5", 1.0, 4.0, 0.0, 12.0, 3.0, None, tmp_path / "a.wav", duck="all")
    loud = render_preview(episode, media, "f5", 1.0, 4.0, 6.0, 12.0, 3.0, -16.0, tmp_path / "b.wav", duck="all")
    a, rate = sf.read(str(quiet))
    b, _ = sf.read(str(loud))
    assert rate == 48000 and a.shape == b.shape == (4 * 48000, 2)
    assert abs(integrated_loudness(b.astype(np.float32), rate) - (-16.0)) < 1.0
    assert float(np.abs(b).max()) <= 0.98
