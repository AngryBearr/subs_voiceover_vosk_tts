"""Bandwidth extension of narrowband cloned voices (stt.widen) and its use in the mix stage."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from stt import widen
from stt.run import Episode

RATE = 24000


def _noise(edge_hz: float | None, seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal(int(RATE * seconds)).astype(np.float32) * 0.1
    return x if edge_hz is None else sosfiltfilt(butter(12, edge_hz, fs=RATE, output="sos"), x).astype(np.float32)


def test_band_edge_finds_where_the_recording_ends() -> None:
    assert 4800 < widen.band_edge(_noise(5200), RATE) < widen.MAX_BAND_HZ  # S01: references end at ~5.2 kHz
    assert widen.band_edge(_noise(None), RATE) > 11000


def test_references_edge_is_the_median_of_the_clips(tmp_path: Path) -> None:
    voices = {}
    for k, edge in enumerate([5000, 5200, 5400]):
        path = tmp_path / f"p{k}.wav"
        sf.write(path, _noise(edge, seed=k), RATE)
        voices[f"S0{k}"] = f"{path}#abc{k}"
    voices["S09"] = "teratts:ru_f1"  # people without a clip have no reference
    (tmp_path / "speaker_voices.f5.json").write_text(json.dumps(voices))
    assert 4800 < widen.references_edge(tmp_path / "speaker_voices.f5.json") < widen.MAX_BAND_HZ
    (tmp_path / "none.json").write_text(json.dumps({"S01": "teratts:ru_m1"}))
    assert widen.references_edge(tmp_path / "none.json") is None
    assert widen.references_edge(tmp_path / "missing.json") is None


def test_widen_track_without_its_environment_keeps_the_track(tmp_path: Path) -> None:
    assert widen.widen_track(tmp_path / "track.wav", tmp_path / "track.wide.wav", 5200, tmp_path / "nope" / "python") is False


def _episode(tmp_path: Path, engine: str = "f5", enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(config={"mix": {"widen": {"enabled": enabled, "engines": ["f5"], "max_band_hz": 7600}}},
                           engine=engine, out=tmp_path, media=Path("Survivor - S01E01.mp4"))


def test_mix_takes_the_widened_track_only_for_narrowband_clones(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    track = tmp_path / "track.wav"
    sf.write(track, _noise(5200), RATE)
    calls: list[float] = []

    def fake_widen(src: Path, out: Path, edge: float, python: Path | None = None) -> bool:
        calls.append(edge)
        out.write_bytes(src.read_bytes())
        return True

    monkeypatch.setattr(widen, "widen_track", fake_widen)
    monkeypatch.setattr(widen, "references_edge", lambda path: 5200.0)
    assert Episode.widened_track(_episode(tmp_path), track) == tmp_path / "track.wide.wav"
    assert Episode.widened_track(_episode(tmp_path), track) == tmp_path / "track.wide.wav" and calls == [5200.0]  # reused
    assert Episode.widened_track(_episode(tmp_path, engine="vosk"), track) == track  # not a cloned voice
    assert Episode.widened_track(_episode(tmp_path, enabled=False), track) == track
    monkeypatch.setattr(widen, "references_edge", lambda path: 10300.0)  # S48: the references are wide already
    (tmp_path / "track.wide.wav").unlink()
    assert Episode.widened_track(_episode(tmp_path), track) == track
