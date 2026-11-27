"""Reference quality checks without loading models or modifying real season files."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from stt import voice_bank

RATE = 16_000


def _tone(db: float, seconds: float = 6.0) -> np.ndarray:
    time = np.arange(round(seconds * RATE)) / RATE
    return (np.sqrt(2) * 10 ** (db / 20) * np.sin(2 * np.pi * 220 * time)).astype(np.float32)


@pytest.mark.parametrize("db, expected", [(-71.0, "quiet"), (-46.0, "quiet"), (-44.0, None), (-22.0, None)])
def test_vocal_level_gate(db: float, expected: str | None) -> None:
    assert voice_bank.vocal_rejection(_tone(db), RATE) == expected
    padded = np.concatenate([_tone(db), np.zeros(RATE)])
    assert voice_bank.vocal_rejection(np.stack([padded, padded]), RATE) == expected


@pytest.mark.parametrize("audio", [
    np.array([]), np.empty((0, RATE)), np.zeros(100), np.full(RATE, np.nan),
    np.full((2, RATE), np.inf), np.zeros((2, 2, RATE)),
])
def test_broken_vocals_are_rejected(audio: np.ndarray) -> None:
    assert voice_bank.vocal_rejection(audio, RATE) == "broken"


def test_silence_and_isolated_peak_do_not_pass() -> None:
    audio = np.zeros(6 * RATE, dtype=np.float32)
    assert voice_bank.vocal_rejection(audio, RATE) == "quiet"
    audio[RATE] = 1.0
    assert voice_bank.vocal_rejection(audio, RATE) == "quiet"
    assert voice_bank.vocal_rejection(_tone(-22), 0) == "broken"
    # The check is on the mono signal actually exported, not individual channels.
    assert voice_bank.vocal_rejection(np.stack([_tone(-22), -_tone(-22)]), RATE) == "quiet"


@pytest.fixture
def season(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[voice_bank.VoiceBank, Path, dict[str, Any]]:
    import stt.separate

    episode = tmp_path / "S01E01"
    episode.mkdir()
    sf.write(episode / "audio.wav", _tone(-22, 20), RATE)
    records = [{"start": 0, "end": 6000, "speaker": "S1", "text": ["words"]}]
    (episode / "subs.en.json").write_text(json.dumps(records), encoding="utf-8")
    (episode / "source.json").write_text('{"media": "unused.mkv"}', encoding="utf-8")
    bank = voice_bank.VoiceBank(tmp_path / "Season.voices.json")
    bank.prints.append({"episode": episode.name, "speaker": "S1", "name": "Amy Adams",
                        "embedding": [1.0, 0.0], "speech_s": 30.0})
    state: dict[str, Any] = {"vocals": _tone(-22), "music_db": -30.0, "separated": 0, "windows": 0}

    class Embedder:
        def __call__(self, audio: np.ndarray, rate: int) -> np.ndarray:
            assert np.isfinite(audio).all()
            return np.array([1.0, 0.0])

    class Separator:
        rate = RATE

        def __call__(self, stereo: np.ndarray) -> tuple[np.ndarray, float]:
            state["separated"] += 1
            return state.get("output", np.stack([state["vocals"], state["vocals"]])), state["music_db"]

    def windows(*args: Any) -> list[float]:
        state["windows"] += 1
        return [1.0]

    monkeypatch.setattr(voice_bank, "SegmentEmbedder", lambda device: Embedder())
    monkeypatch.setattr(stt.separate, "VocalSeparator", lambda device: Separator())
    monkeypatch.setattr(voice_bank, "_read_media", lambda *args: np.stack([_tone(-22), _tone(-22)]))
    monkeypatch.setattr(voice_bank, "_window_scores", windows)
    monkeypatch.setattr(voice_bank, "load_transcriber", lambda device: object())
    monkeypatch.setattr(voice_bank, "transcribe_clip", lambda *args: "words")
    return bank, tmp_path / "refs", state


@pytest.mark.parametrize("audio, music_db, reason", [
    (_tone(-71), -30.0, "quiet"),
    (np.array([], dtype=np.float32), -30.0, "broken"),
    (np.full(RATE, np.nan), -30.0, "broken"),
    (np.full(RATE, np.inf), -30.0, "broken"),
    (_tone(-22), float("nan"), "broken"),
])
def test_export_rejects_before_voice_windows_and_explains_in_russian(
    season: tuple[voice_bank.VoiceBank, Path, dict[str, Any]],
    audio: np.ndarray, music_db: float, reason: str,
) -> None:
    bank, refs, state = season
    state.update(vocals=audio, music_db=music_db)
    assert voice_bank.export_references(bank, refs, "cpu", stitch=False) == []
    why = voice_bank.load_missing(refs)["Amy Adams"]
    assert why[reason] == 1 and why["unchecked"] == 0 and why["noisy"] == 0
    text = voice_bank.reason_text(why)
    assert ("голос слишком тихий или отсутствует" if reason == "quiet" else "аудио повреждено или пустое") in text
    assert state["windows"] == 0
    assert not (refs / "Amy_Adams.wav").exists()


def test_export_tries_next_candidate_after_silent_separation(
    season: tuple[voice_bank.VoiceBank, Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank, refs, state = season
    path = bank.path.parent / "S01E01/subs.en.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    records.append({**records[0], "start": 10000, "end": 16000})
    path.write_text(json.dumps(records), encoding="utf-8")

    def read(media: str, start: float, *args: Any) -> np.ndarray:
        state["vocals"] = _tone(-71 if start == 0 else -22)
        return np.stack([_tone(-22), _tone(-22)])

    monkeypatch.setattr(voice_bank, "_read_media", read)
    index = voice_bank.export_references(bank, refs, "cpu", stitch=False)
    assert index[0]["start"] == 10.0
    assert state["separated"] == 2 and state["windows"] == 1
    assert voice_bank.load_missing(refs) == {}
    audio, rate = sf.read(refs / index[0]["wav"])
    assert voice_bank.vocal_rejection(audio, rate) is None


@pytest.mark.parametrize("error", [
    ValueError("broken shape"), OSError("unreadable audio"), subprocess.CalledProcessError(1, ["ffmpeg"]),
])
def test_media_decode_failure_is_a_broken_candidate(
    season: tuple[voice_bank.VoiceBank, Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch, error: Exception,
) -> None:
    bank, refs, state = season

    def read(*args: Any) -> np.ndarray:
        raise error

    monkeypatch.setattr(voice_bank, "_read_media", read)
    assert voice_bank.export_references(bank, refs, "cpu", stitch=False) == []
    assert voice_bank.load_missing(refs)["Amy Adams"]["broken"] == 1
    assert state["separated"] == 0


@pytest.mark.parametrize("kind", ["quiet", "corrupt", "absent", "nonfinite"])
def test_missing_export_rechecks_old_bad_wavs(
    season: tuple[voice_bank.VoiceBank, Path, dict[str, Any]], kind: str,
) -> None:
    bank, refs, state = season
    refs.mkdir()
    wav = refs / "Amy_Adams.wav"
    if kind == "quiet":
        sf.write(wav, _tone(-71), RATE)
    elif kind == "corrupt":
        wav.write_bytes(b"not a WAV")
    elif kind == "nonfinite":
        sf.write(wav, np.full(RATE, np.nan), RATE, subtype="FLOAT")
    (refs / "references.json").write_text(json.dumps([{"name": "Amy Adams", "wav": wav.name}]), encoding="utf-8")
    index = voice_bank.export_references(bank, refs, "cpu", missing=True, stitch=False)
    assert len(index) == 1 and index[0]["episode"] == "S01E01"
    assert state["separated"] == 1
    audio, rate = sf.read(wav)
    assert voice_bank.vocal_rejection(audio, rate) is None


def test_missing_export_drops_stale_bad_clip_if_replacement_is_quiet(
    season: tuple[voice_bank.VoiceBank, Path, dict[str, Any]],
) -> None:
    bank, refs, state = season
    refs.mkdir()
    wav = refs / "Amy_Adams.wav"
    sf.write(wav, _tone(-71), RATE)
    (refs / "references.json").write_text(json.dumps([{"name": "Amy Adams", "wav": wav.name}]), encoding="utf-8")
    state["vocals"] = _tone(-71)
    assert voice_bank.export_references(bank, refs, "cpu", missing=True, stitch=False) == []
    assert not wav.exists()
    assert voice_bank.load_missing(refs)["Amy Adams"]["quiet"] == 1


def test_resume_keeps_valid_auto_and_manual_without_changing_files(
    season: tuple[voice_bank.VoiceBank, Path, dict[str, Any]],
) -> None:
    bank, refs, state = season
    index = voice_bank.export_references(bank, refs, "cpu")
    times = {p: p.stat().st_mtime_ns for p in (refs / "Amy_Adams.wav", refs / "references.json")}
    state["vocals"] = _tone(-71)
    assert voice_bank.export_references(bank, refs, "cpu", missing=True) == index
    assert all(p.stat().st_mtime_ns == t for p, t in times.items())
    # Manual clips are owned by the user, even if quiet.
    manual = {**index[0], "source": "manual", "wav": "Amy_Adams.manual.wav"}
    (refs / manual["wav"]).write_bytes(b"user data")
    (refs / "manual.json").write_text(json.dumps([manual]), encoding="utf-8")
    assert voice_bank.export_references(bank, refs, "cpu", missing=True)[0]["source"] == "manual"
    assert (refs / manual["wav"]).read_bytes() == b"user data"
    assert state["separated"] == 1


def test_stitched_candidate_uses_the_same_quiet_gate(
    season: tuple[voice_bank.VoiceBank, Path, dict[str, Any]],
) -> None:
    bank, refs, state = season
    records = [{"start": s, "end": s + 3000, "speaker": "S1", "text": ["words"]} for s in (0, 4000)]
    (bank.path.parent / "S01E01/subs.en.json").write_text(json.dumps(records), encoding="utf-8")
    state["vocals"] = _tone(-71)
    assert voice_bank.export_references(bank, refs, "cpu") == []
    why = voice_bank.load_missing(refs)["Amy Adams"]
    assert why["stretches"] == 0 and why["quiet"] == 1 and why["stitched_failed"] == 1
    assert "голос слишком тихий или отсутствует" in voice_bank.reason_text(why)
    assert state["windows"] == 0


@pytest.mark.parametrize("audio", [np.empty((2, 0)), np.full((2, RATE), np.nan), np.ones(RATE)])
def test_broken_media_does_not_reach_separator(
    season: tuple[voice_bank.VoiceBank, Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch, audio: np.ndarray,
) -> None:
    bank, refs, state = season
    monkeypatch.setattr(voice_bank, "_read_media", lambda *args: audio)
    assert voice_bank.export_references(bank, refs, "cpu", stitch=False) == []
    assert voice_bank.load_missing(refs)["Amy Adams"]["broken"] == 1
    assert state["separated"] == 0


@pytest.mark.parametrize("audio", [_tone(-22), np.ones((1, RATE)), np.ones((2, 2, RATE))])
def test_invalid_separator_shape_is_a_broken_candidate(
    season: tuple[voice_bank.VoiceBank, Path, dict[str, Any]], audio: np.ndarray,
) -> None:
    bank, refs, state = season
    state["output"] = audio
    assert voice_bank.export_references(bank, refs, "cpu", stitch=False) == []
    assert voice_bank.load_missing(refs)["Amy Adams"]["broken"] == 1
    assert state["windows"] == 0


def test_targeted_export_preserves_other_peoples_old_entries(
    season: tuple[voice_bank.VoiceBank, Path, dict[str, Any]],
) -> None:
    bank, refs, state = season
    refs.mkdir()
    unrelated = {"name": "Bea Brown", "wav": "Bea.wav"}
    (refs / "Bea.wav").write_bytes(b"unrelated user file")
    (refs / "references.json").write_text(json.dumps([unrelated]), encoding="utf-8")
    index = voice_bank.export_references(bank, refs, "cpu", only="Amy Adams", missing=True)
    assert unrelated in index and len(index) == 2
    assert (refs / "Bea.wav").read_bytes() == b"unrelated user file"
