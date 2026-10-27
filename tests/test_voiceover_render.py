from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from voiceover.render import RenderOptions, Segment, level_clip, parse_segments, render, speech_level_db, stretch_clip

SR = 8000


def _tone(path: Path, sec: float, amp: float = 0.3, freq: float = 300.0, sr: int = SR) -> Path:
    t = np.arange(int(sr * sec)) / sr
    sf.write(str(path), (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)
    return path


def fake_stretch(path: Path, tempo: float, sr: int) -> np.ndarray:
    data, _ = sf.read(str(path), dtype="float32")
    return data


def _p(uid: str, cue_start: int, start: int, **kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "unit_id": uid, "cue_ids": [uid], "cue_start_ms": cue_start, "start_ms": start,
        "shift_ms": start - cue_start, "tempo_factor": 1.0, "total_speed": 1.0, "removed_sec": 0.0,
        "end_ms": start + 500, "status": "fits", "merged": False,
    }
    base.update(kw)
    return base


def _row(uid: str, path: Path | None, text: str = "text", error: str | None = None) -> dict[str, Any]:
    return {"unit_id": uid, "path": str(path) if path else None, "display_text": text, "error": error}


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    return _tone(tmp_path / "a.wav", 0.5)


def _run(tmp_path, placements, rows, segments, **opt):
    opts = RenderOptions(sample_rate=SR, segment_marker=opt.pop("segment_marker", False), **opt)
    out = tmp_path / "o" / "d.wav"
    guide = tmp_path / "o" / "d_guide.md"
    s = render(placements, rows, [], segments, opts, out, guide, stretch=fake_stretch)
    data, sr = sf.read(str(out), dtype="float32")
    assert sr == SR
    return s, data, guide.read_text(encoding="utf-8")


def _energy(x: np.ndarray, start_ms: float, dur_ms: float) -> float:
    a = int(SR * start_ms / 1000)
    return float(np.abs(x[a : a + int(SR * dur_ms / 1000)]).max())


def test_placement_offsets(tmp_path, wav):
    s, data, _ = _run(tmp_path, [_p("u1", 1000, 1200)], [_row("u1", wav)], [Segment(1000, 3000)])
    assert _energy(data, 0, 190) < 1e-3
    assert _energy(data, 200, 100) > 0.1
    assert s.units_rendered == 1 and s.overlaps == 0
    assert abs(s.duration_sec - 2.0) < 0.01


def test_negative_start_clamped(tmp_path, wav):
    _, data, _ = _run(tmp_path, [_p("u1", 1000, 900)], [_row("u1", wav)], [Segment(1000, 3000)])
    assert _energy(data, 0, 100) > 0.1


def test_segments_concat_with_gap(tmp_path, wav):
    pl = [_p("u1", 1000, 1000), _p("u2", 5000, 5000), _p("u3", 9000, 9000)]
    rows = [_row(u, wav) for u in ("u1", "u2", "u3")]
    s, data, guide = _run(tmp_path, pl, rows, [Segment(1000, 2000), Segment(5000, 6000)])
    assert s.units_rendered == 2
    # seg1 = 1.0 s buffer but clip runs 0.5 s inside; gap 1.5 s; seg2 starts at 2.5 s
    assert abs(s.duration_sec - 3.5) < 0.01
    assert _energy(data, 1100, 300) < 1e-3
    assert _energy(data, 2500, 100) > 0.1
    assert "## Фрагмент 1" in guide and "## Фрагмент 2" in guide
    assert "00:02.5" in guide


def test_overrun_extends_buffer_and_overlap(tmp_path, wav):
    pl = [_p("u1", 0, 0), _p("u2", 400, 400)]
    s, data, _ = _run(tmp_path, pl, [_row("u1", wav), _row("u2", wav)], [Segment(0, 600)])
    assert s.overlaps == 1
    assert abs(s.duration_sec - 0.9) < 0.01
    assert _energy(data, 450, 40) > 0.4  # both tones added


def test_ticks(tmp_path, wav):
    pl = [_p("u1", 1000, 1000, merged=True, cue_ids=["a", "b"],
             members=[{"cue_id": "a", "start": 1000, "drift_ms": 0}, {"cue_id": "b", "start": 1800, "drift_ms": -300}])]
    seg = [Segment(0, 3000)]
    _, off, _ = _run(tmp_path, pl, [_row("u1", wav)], seg)
    _, on, guide = _run(tmp_path, pl, [_row("u1", wav)], seg, ticks=True)
    assert _energy(off, 1800, 25) < 1e-3
    assert _energy(on, 1800, 25) > 0.05
    assert "300" in guide


def test_marker(tmp_path, wav):
    _, data, _ = _run(tmp_path, [], [], [Segment(0, 1000)], segment_marker=True)
    assert _energy(data, 0, 60) > 0.1


def test_error_rows_skipped(tmp_path, wav):
    pl = [_p("u1", 0, 0), _p("u2", 100, 100), _p("u3", 200, 200)]
    rows = [_row("u1", wav), _row("u2", wav, error="boom"), _row("u3", None)]
    s, _, guide = _run(tmp_path, pl, rows, [Segment(0, 1000)])
    assert (s.units_rendered, s.units_skipped) == (1, 2)
    assert guide.count("| fits |") == 1


def test_peak_normalise_only_when_needed(tmp_path):
    loud = _tone(tmp_path / "l.wav", 0.5, amp=0.9)
    pl = [_p("u1", 0, 0), _p("u2", 0, 0)]
    rows = [_row("u1", loud), _row("u2", loud)]
    s, data, _ = _run(tmp_path, pl, rows, [Segment(0, 1000)])
    assert s.peak_before_normalise > 1.0
    assert abs(float(np.abs(data).max()) - 10 ** (-1 / 20)) < 0.01
    quiet = _tone(tmp_path / "q.wav", 0.5, amp=0.3)
    s, data, _ = _run(tmp_path, [_p("u1", 0, 0)], [_row("u1", quiet)], [Segment(0, 1000)])
    assert abs(float(np.abs(data).max()) - 0.3) < 0.01


def test_guide_columns(tmp_path, wav):
    pl = [_p("u1", 65000, 65000, removed_sec=1.234, status="needs_text")]
    _, _, guide = _run(tmp_path, pl, [_row("u1", wav, text="x" * 100)], [Segment(60000, 70000)])
    for col in ("время в файле", "время в эпизоде", "статус", "скорость", "сдвиг мс", "вырезать с", "слито", "дрейф мс", "текст"):
        assert col in guide
    assert "эпизод 01:00–01:10" in guide
    assert "01:05.0" in guide and "1.2" in guide and "needs_text" in guide
    assert "x" * 80 not in guide


def test_parse_segments():
    assert parse_segments("45-120,450.5-540") == [Segment(45000, 120000), Segment(450500, 540000)]
    with pytest.raises(ValueError):
        parse_segments("10-5")
    with pytest.raises(ValueError):
        parse_segments("abc")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_stretch_clip_real(tmp_path):
    p = _tone(tmp_path / "s.wav", 1.0, sr=22050)
    assert abs(len(stretch_clip(p, 1.25, 22050)) / 22050 - 0.8) < 0.05
    assert abs(len(stretch_clip(p, 1.0, 22050)) / 22050 - 1.0) < 0.01
    with pytest.raises(RuntimeError):
        stretch_clip(tmp_path / "missing.wav", 1.0, 22050)


def test_level_clip_brings_quiet_and_loud_lines_to_one_level() -> None:
    # cloned voices speak as loud as their reference: S20E08 lines ranged from -42 to -17 dBFS
    import numpy as np

    rate = 16000
    t = np.arange(rate) / rate
    speech = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    speech[rate // 2: rate // 2 + rate // 10] = 0.0  # a pause does not count
    for amplitude in (0.02, 0.05, 0.5):  # -37, -29 and -9 dBFS of speech
        out = level_clip(speech * amplitude, rate, -20.0)
        assert abs(speech_level_db(out, rate) - -20.0) < 0.2
    assert abs(speech_level_db(level_clip(speech * 0.0005, rate, -20.0, max_gain_db=20.0), rate)
               - speech_level_db(speech * 0.005, rate)) < 0.2  # at most +20 dB
    loud = level_clip(speech * 0.001, rate, 0.0, max_gain_db=60.0)
    assert np.abs(loud).max() <= 0.95 + 1e-6  # never over the peak limit
    assert not level_clip(np.zeros(rate, np.float32), rate, -20.0).any()


def test_level_track_levels_each_placed_line() -> None:
    import numpy as np

    from voiceover.render import level_track

    rate = 16000
    t = np.arange(rate) / rate
    tone = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    track = np.concatenate([tone * 0.02, np.zeros(rate, np.float32), tone * 0.4]).astype(np.float32)
    out = level_track(track, rate, [(0, 1000), (2000, 3000)], -20.0)
    for a, b in ((0, rate), (2 * rate, 3 * rate)):
        assert abs(speech_level_db(out[a + 400:b - 400], rate) - -20.0) < 0.3
    assert not out[rate + 400:2 * rate - 400].any()
