"""Clips made by hand: range parsing, cutting/joining, and that no export ever drops them.

The heavy models (vocal separator, Whisper, WeSpeaker) and the video reading are replaced by small fakes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from stt import manual_clip, voice_bank
from stt.manual_clip import parse_ranges
from stt.voice_bank import VoiceBank

RATE = 44_100


# --- range parsing ---------------------------------------------------------------------------------------------


def test_parse_ranges_formats() -> None:
    assert parse_ranges("12:30-12:41") == [(750.0, 761.0)]
    assert parse_ranges("1:02:05 – 1:02:14") == [(3725.0, 3734.0)]
    assert parse_ranges("750-761.5\n12:30—12:41") == [(750.0, 761.5), (750.0, 761.0)]
    assert parse_ranges("0:05-0:09, 1:00-1:08; 90-95") == [(5.0, 9.0), (60.0, 68.0), (90.0, 95.0)]


@pytest.mark.parametrize("text, fragment", [
    ("", "Не указан ни один отрезок"),
    ("  \n ", "Не указан ни один отрезок"),
    ("12:30", "Не понял отрезок"),
    ("12:30-abc", "Не понял отрезок"),
    ("12:41-12:30", "конец не позже начала"),
    ("5-5", "конец не позже начала"),
    ("0:10-1:20", "длиннее 60 секунд"),
    ("1:75-2:00", "больше 59"),
])
def test_parse_ranges_errors_are_russian(text: str, fragment: str) -> None:
    with pytest.raises(ValueError) as error:
        parse_ranges(text)
    assert fragment in str(error.value)


def test_format_time_and_reason_text() -> None:
    assert manual_clip.format_time(750) == "12:30" and manual_clip.format_time(3725.5) == "1:02:05.5"
    why = {"episodes": 3, "stretches": 55, "rejected": 0, "other_voices": 40, "low_score": 15, "noisy": 0,
           "unchecked": 0}
    assert voice_bank.reason_text(why) == "55 кандидатов: в 40 рядом звучат другие голоса, в 15 голос не похож на человека"
    assert "нет ни одного отрезка" in voice_bank.reason_text({"episodes": 2, "stretches": 0})
    assert voice_bank.reason_text({"episodes": 0, "stretches": 0}).startswith("в банке нет")
    assert voice_bank.reason_text({"episodes": 1, "stretches": 21, "noisy": 8}).startswith("21 кандидат:")


# --- fakes --------------------------------------------------------------------------------------------------------


def _dim(i: int) -> np.ndarray:
    v = np.zeros(256)
    v[i] = 1.0
    return v


class FakeSeparator:
    rate = RATE

    def __call__(self, stereo: np.ndarray) -> tuple[np.ndarray, float]:
        return stereo.copy(), -25.0


class FakeEmbedder:
    """Loud audio sounds like person 0, quiet audio like person 1."""

    def __init__(self, device: str = "cpu") -> None:
        pass

    def __call__(self, audio: np.ndarray, rate: int) -> np.ndarray:
        return _dim(0) if float(np.abs(audio).mean()) > 0.15 else _dim(1)


class FakeTranscriber:
    def transcribe(self, audio: np.ndarray, chunks: list[Any]) -> list[Any]:
        return [SimpleNamespace(text=" hello there ")]


class FakeModels:
    def __init__(self, device: str = "cpu") -> None:
        self.separator, self.embedder, self.transcriber = FakeSeparator(), FakeEmbedder(), FakeTranscriber()


def _tone(seconds: float, amp: float = 0.5, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _speech_with_silence(lead: float, speech: float, tail: float, amp: float = 0.5) -> np.ndarray:
    mono = np.concatenate([np.zeros(int(lead * RATE), np.float32), _tone(speech, amp),
                           np.zeros(int(tail * RATE), np.float32)])
    return np.stack([mono, mono])


@pytest.fixture
def media_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """The video has 1 s of silence, then 2 s of speech, around every requested stretch."""
    def read(media: str, start: float, duration: float, rate: int, channels: int) -> np.ndarray:
        speech = max(0.0, duration - 2.0)
        return _speech_with_silence(1.0, speech, 1.0)[:, :int(duration * rate)] if duration > 2 else \
            _speech_with_silence(0.0, duration, 0.0)

    monkeypatch.setattr(manual_clip, "_read_media", read)


def _video(tmp_path: Path) -> Path:
    path = tmp_path / "ep.mkv"
    path.write_bytes(b"x")
    return path


# --- cutting and joining ----------------------------------------------------------------------------------------


def test_clip_is_trimmed_joined_and_ended_like_an_automatic_one(tmp_path: Path, media_reader: None) -> None:
    out = tmp_path / "clip.wav"
    clip = manual_clip.make_manual_clip(_video(tmp_path), [(10.0, 15.0), (20.0, 25.0)], out, models=FakeModels())
    audio, rate = sf.read(str(out), dtype="float32")
    assert rate == voice_bank.CLONE_RATE and audio.ndim == 1
    # each range holds 3 s of speech between 1 s of silence: trimmed to 3 s + 2 x 0.1 s pad, joined with 0.3 s
    assert clip.duration == pytest.approx(2 * 3.2 + 0.3, abs=0.1)
    assert len(audio) / rate == pytest.approx(clip.duration + 1.0, abs=0.05)  # 1 s of silence at the end
    assert np.abs(audio[-rate:]).max() == 0 and abs(audio[0]) < 0.01  # fade in
    assert clip.text == "hello there" and clip.warnings == [] and len(clip.pieces) == 2


def test_clip_longer_than_the_limit_is_cut_at_a_pause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def read(media: str, start: float, duration: float, rate: int, channels: int) -> np.ndarray:
        return _speech_with_silence(0.0, 4.0, 0.0) if start < 100 else np.concatenate(
            [_speech_with_silence(0, 3.0, 0.6), _speech_with_silence(0, 3.0, 0.0)], axis=1)

    monkeypatch.setattr(manual_clip, "_read_media", read)
    clip = manual_clip.make_manual_clip(_video(tmp_path), [(0.0, 4.0), (10.0, 14.0), (200.0, 207.0)],
                                        tmp_path / "c.wav", models=FakeModels(), text="given")
    assert clip.text == "given"
    assert clip.duration <= manual_clip.MAX_CLIP_S
    assert any("Достаточно" in w for w in clip.warnings)


def test_short_and_silent_ranges_warn(tmp_path: Path, media_reader: None, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = manual_clip.make_manual_clip(_video(tmp_path), [(10.0, 13.0)], tmp_path / "c.wav", models=FakeModels())
    assert clip.duration < 3.0 and any("Речи всего" in w for w in clip.warnings)
    monkeypatch.setattr(manual_clip, "_read_media", lambda *a: np.zeros((2, RATE * 4), np.float32))
    with pytest.raises(ValueError, match="не нашлось голоса"):
        manual_clip.make_manual_clip(_video(tmp_path), [(1.0, 5.0)], tmp_path / "d.wav", models=FakeModels())


def test_low_voice_similarity_warns_but_does_not_block(tmp_path: Path, media_reader: None) -> None:
    bank = VoiceBank(tmp_path / "S.voices.json")
    bank.prints.append({"episode": "e1", "speaker": "S1", "name": "Sai Hughley", "gender": "female",
                        "embedding": list(_dim(2))})
    clip = manual_clip.make_manual_clip(_video(tmp_path), [(10.0, 16.0)], tmp_path / "c.wav", models=FakeModels(),
                                        bank=bank, person="Sai Hughley")
    assert clip.similarity is not None and clip.similarity < voice_bank.MIN_SAMPLE_SCORE
    assert any("слабо похож" in w for w in clip.warnings) and clip.wav.exists()


# --- exports never drop manual clips ----------------------------------------------------------------------------


def _season(tmp_path: Path) -> tuple[VoiceBank, Path]:
    """A (loud, matches her prints) has a clean stretch; B (quiet, prints elsewhere) cannot get a clip."""
    root = tmp_path / "season"
    ep = root / "ep1"
    ep.mkdir(parents=True)
    audio = np.concatenate([_tone(8.0, 0.5, 16_000), _tone(8.0, 0.05, 16_000)])
    sf.write(str(ep / "audio.wav"), audio, 16_000)
    cue = lambda s, e, who: {"start": int(s * 1000), "end": int(e * 1000), "speaker": who,  # noqa: E731
                             "text": ["words here"]}
    (ep / "subs.en.json").write_text(json.dumps([cue(0, 7, "S1"), cue(9, 15, "S2")]), encoding="utf-8")
    (ep / "source.json").write_text(json.dumps({"media": str(_video(tmp_path))}), encoding="utf-8")
    bank = VoiceBank(root / "S.voices.json")
    for sid, name, dim in (("S1", "Amy Adams", 0), ("S2", "Bea Brown", 2)):
        bank.prints.append({"episode": "ep1", "speaker": sid, "name": name, "label_ru": name.split()[0],
                            "gender": "female", "speech_s": 60.0, "embedding": list(_dim(dim)), "samples": []})
    return bank, root / "S.references"


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch, media_reader: None) -> None:
    import stt.separate

    monkeypatch.setattr(voice_bank, "SegmentEmbedder", FakeEmbedder)
    monkeypatch.setattr(stt.separate, "VocalSeparator", lambda device: FakeSeparator())
    monkeypatch.setattr(voice_bank, "load_transcriber", lambda device: FakeTranscriber())
    monkeypatch.setattr(manual_clip, "ClipModels", FakeModels)
    monkeypatch.setattr(voice_bank, "_read_media", lambda m, s, d, r, c: _speech_with_silence(0, d, 0, 0.5 if s < 8 else 0.05))


def _names(index: list[dict[str, Any]]) -> dict[str, str]:
    return {r["name"]: r.get("source", "auto") for r in index}


def test_export_records_why_a_person_has_no_clip(tmp_path: Path, fake_models: None) -> None:
    bank, refs = _season(tmp_path)
    index = voice_bank.export_references(bank, refs, "cpu")
    assert _names(index) == {"Amy Adams": "auto"}
    missing = json.loads((refs / "missing.json").read_text(encoding="utf-8"))
    assert missing["Bea Brown"]["episodes"] == 1 and missing["Bea Brown"]["stretches"] == 1
    assert missing["Bea Brown"]["low_score"] == 1 and "Amy Adams" not in missing
    bank.save()
    rows = {r["name"]: r for r in voice_bank.clip_states(bank.path, refs)}
    assert rows["Amy Adams"]["state"] == "auto" and rows["Amy Adams"]["score"] == 1.0
    assert rows["Bea Brown"]["state"] == "missing" and "голос не похож" in rows["Bea Brown"]["reason_ru"]


def test_missing_reasons_are_cleared_when_a_clip_appears(tmp_path: Path, fake_models: None) -> None:
    bank, refs = _season(tmp_path)
    voice_bank.export_references(bank, refs, "cpu")
    assert "Bea Brown" in voice_bank.load_missing(refs)
    bank.prints[1]["embedding"] = list(_dim(1))  # now B's quiet stretch sounds like her prints
    index = voice_bank.export_references(bank, refs, "cpu", only="Bea Brown")
    assert _names(index) == {"Amy Adams": "auto", "Bea Brown": "auto"}
    assert voice_bank.load_missing(refs) == {}


def _make_manual(tmp_path: Path, refs: Path, bank: VoiceBank, name: str = "Bea Brown", **extra: Any) -> dict[str, Any]:
    clip = manual_clip.make_manual_clip(_video(tmp_path), [(10.0, 16.0)], tmp_path / "tmp.wav", models=FakeModels(),
                                        bank=bank, person=name)
    return manual_clip.save_manual_clip(refs, name, clip, episode="ep1", ranges=[(10.0, 16.0)], bank=bank, **extra)


def test_manual_clip_survives_every_export(tmp_path: Path, fake_models: None) -> None:
    bank, refs = _season(tmp_path)
    voice_bank.export_references(bank, refs, "cpu")
    entry = _make_manual(tmp_path, refs, bank)
    assert entry["source"] == "manual" and entry["ranges"] == [[10.0, 16.0]] and entry["gender"] == "female"
    assert (refs / "Bea_Brown.manual.wav").exists() and (refs / "Bea_Brown.manual.txt").read_text().strip() == "hello there"
    assert "Bea Brown" not in voice_bank.load_missing(refs)
    manual_wav = (refs / "Bea_Brown.manual.wav").read_bytes()
    for kwargs in ({}, {"missing": True}, {"only": "Bea Brown"}, {"only": "Amy Adams"}):
        index = voice_bank.export_references(bank, refs, "cpu", **kwargs)
        assert _names(index) == {"Amy Adams": "auto", "Bea Brown": "manual"}, kwargs
        assert (refs / "Bea_Brown.manual.wav").read_bytes() == manual_wav
        assert "Bea Brown" not in voice_bank.load_missing(refs)
    assert voice_bank.reject_reference(refs, "Bea Brown") is None  # a manual clip is removed, never rejected
    assert voice_bank.load_rejected(refs) == []
    assert {r["name"]: r["state"] for r in voice_bank.clip_states(bank.path, refs)}["Bea Brown"] == "manual"


def test_manual_clip_for_a_person_outside_the_bank_needs_a_gender(tmp_path: Path, fake_models: None) -> None:
    bank, refs = _season(tmp_path)
    clip = manual_clip.make_manual_clip(_video(tmp_path), [(10.0, 16.0)], tmp_path / "t.wav", models=FakeModels())
    with pytest.raises(ValueError, match="пол"):
        manual_clip.save_manual_clip(refs, "Zed Zimmer", clip, bank=bank)
    entry = manual_clip.save_manual_clip(refs, "Zed Zimmer", clip, gender="male", bank=bank)
    assert entry["label_ru"] == "Zed Zimmer" and entry["gender"] == "male"
    index = voice_bank.export_references(bank, refs, "cpu")
    assert _names(index)["Zed Zimmer"] == "manual"


def test_remove_manual_returns_to_the_automatic_clip(tmp_path: Path, fake_models: None) -> None:
    bank, refs = _season(tmp_path)
    voice_bank.export_references(bank, refs, "cpu")
    auto = [r for r in json.loads((refs / "references.json").read_text(encoding="utf-8")) if r["name"] == "Amy Adams"][0]
    _make_manual(tmp_path, refs, bank, "Amy Adams")
    index = json.loads((refs / "references.json").read_text(encoding="utf-8"))
    assert _names(index) == {"Amy Adams": "manual"} and (refs / "Amy_Adams.wav").exists()
    assert manual_clip.remove_manual_clip(refs, "Amy Adams") is True
    index = json.loads((refs / "references.json").read_text(encoding="utf-8"))
    assert index == [auto] and not (refs / "Amy_Adams.manual.wav").exists()
    assert manual_clip.remove_manual_clip(refs, "Amy Adams") is False
    # a person who had no automatic clip is simply without one again
    _make_manual(tmp_path, refs, bank)
    assert manual_clip.remove_manual_clip(refs, "Bea Brown") is True
    assert _names(json.loads((refs / "references.json").read_text(encoding="utf-8"))) == {"Amy Adams": "auto"}
    assert json.loads((refs / "manual.json").read_text(encoding="utf-8")) == []


# --- command line ------------------------------------------------------------------------------------------------


def _tree(root: Path) -> dict[str, float]:
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


def test_cli_preview_writes_nothing_in_the_season(tmp_path: Path, fake_models: None, capsys: pytest.CaptureFixture[str]
                                                  ) -> None:
    bank, refs = _season(tmp_path)
    bank.save()
    before = _tree(bank.path.parent)
    preview = tmp_path / "out" / "preview.wav"
    code = voice_bank.main([str(bank.path), "--manual-clip", "--person", "Bea Brown", "--media", str(_video(tmp_path)),
                            "--range", "0:10-0:16", "--preview", str(preview), "--device", "cpu"])
    assert code == 0 and preview.exists() and not refs.exists() and _tree(bank.path.parent) == before
    shown = capsys.readouterr().out
    assert "text: hello there" in shown and "preview only" in shown


def test_cli_saves_and_removes(tmp_path: Path, fake_models: None, capsys: pytest.CaptureFixture[str]) -> None:
    bank, refs = _season(tmp_path)
    bank.save()
    args = [str(bank.path), "--person", "Bea Brown", "--device", "cpu"]
    assert voice_bank.main([*args, "--manual-clip", "--media", str(_video(tmp_path)), "--range", "10-16"]) == 0
    shown = capsys.readouterr().out
    assert str(refs / "Bea_Brown.manual.wav") in shown and "manual.json" in shown
    manual = json.loads((refs / "manual.json").read_text(encoding="utf-8"))
    assert manual[0]["episode"] == "ep1" and manual[0]["source"] == "manual"
    assert voice_bank.main([*args, "--remove-manual"]) == 0 and not (refs / "Bea_Brown.manual.wav").exists()
    assert voice_bank.main([*args, "--manual-clip", "--range", "bad", "--media", str(_video(tmp_path))]) == 2


def test_cli_json_is_one_machine_readable_line(tmp_path: Path, fake_models: None, capsys: pytest.CaptureFixture[str]
                                               ) -> None:
    bank, refs = _season(tmp_path)
    bank.save()
    preview = tmp_path / "out" / "preview.wav"
    args = [str(bank.path), "--manual-clip", "--person", "Bea Brown", "--preview", str(preview), "--device", "cpu",
            "--json"]
    assert voice_bank.main([*args, "--media", str(_video(tmp_path)), "--range", "0:10-0:16"]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1
    found = json.loads(lines[0])
    assert found["wav"] == str(preview) and found["text"] == "hello there" and preview.exists()
    assert found["pieces"] == [[10.0, 16.0]] and isinstance(found["warnings"], list) and found["duration"] > 0
    assert not refs.exists()
    # a mistake is {"error": <Russian message>} and exit code 2
    assert voice_bank.main([*args, "--media", str(_video(tmp_path)), "--range", "bad"]) == 2
    assert "Не понял отрезок" in json.loads(capsys.readouterr().out)["error"]
    assert voice_bank.main([*args, "--media", str(tmp_path / "none.mkv"), "--range", "0:10-0:16"]) == 2
    assert "Нет файла" in json.loads(capsys.readouterr().out)["error"]


# --- what stt.run does with it --------------------------------------------------------------------------------------


def test_manual_clip_revoices_only_that_person(tmp_path: Path, fake_models: None) -> None:
    from unittest import mock

    from stt.run import Episode
    from voiceover.engine import cache_key

    bank, refs = _season(tmp_path)
    bank.save()
    voice_bank.export_references(bank, refs, "cpu")
    episode = Episode.__new__(Episode)
    episode.media, episode.out, episode.voice_bank = Path("ep.mkv"), tmp_path / "work", bank.path
    episode.config = {"voiceover": {}}
    episode.out.mkdir()
    (episode.out / "speakers.json").write_text(json.dumps([{"id": "S1", "name": "Amy Adams"},
                                                           {"id": "S2", "name": "Bea Brown"}]), encoding="utf-8")

    def mapping() -> dict[str, str]:
        with mock.patch.object(Episode, "_module", lambda self, *a, **k: voice_bank.export_references(
                bank, refs, "cpu", missing=True)):  # what the dub stage runs first: --export --missing
            return json.loads(episode._speaker_voices().read_text(encoding="utf-8"))

    before = mapping()
    assert set(before) == {"S1"}  # Bea has no clip yet
    index_time = (refs / "references.json").stat().st_mtime_ns
    time.sleep(0.01)
    _make_manual(tmp_path, refs, bank)
    after = mapping()
    assert set(after) == {"S1", "S2"} and after["S1"] == before["S1"]  # Amy's lines keep their cache entries
    assert cache_key("same line", after["S1"]) == cache_key("same line", before["S1"])
    assert (refs / "references.json").stat().st_mtime_ns > index_time  # the dub stage's input changed

    other = tmp_path / "other.wav"
    sf.write(str(other), np.concatenate([_tone(5.0, 0.6, 24_000), np.zeros(24_000, np.float32)]), 24_000)
    manual_clip.save_manual_clip(refs, "Bea Brown", other, text="another clip", episode="ep1", bank=bank)
    third = mapping()
    assert third["S1"] == before["S1"] and third["S2"] != after["S2"]  # a new manual clip: only Bea's key changes
    assert cache_key("same line", third["S2"]) != cache_key("same line", after["S2"])

    # the dub stage is stale after the change, and unchanged files keep the index time (write_if_changed)
    episode.args = argparse.Namespace(force=None)
    episode.forced_from = None
    stamp = episode.out / ".stages" / "dub.f5"
    stamp.parent.mkdir()
    stamp.write_text("", encoding="utf-8")
    outputs = episode.out / "dub.ru.f5.json"
    outputs.write_text("{}", encoding="utf-8")
    os.utime(stamp, (time.time() - 100, time.time() - 100))
    episode.engine = "f5"
    assert episode.needed("dub", [outputs], None, [refs / "references.json"]) is True


def test_export_stitches_short_solo_pieces_when_no_stretch_is_long_enough(tmp_path: Path, fake_models: None) -> None:
    bank, refs = _season(tmp_path)
    ep = bank.path.parent / "ep1"
    cue = lambda s, e, who: {"start": int(s * 1000), "end": int(e * 1000), "speaker": who,  # noqa: E731
                             "text": ["words here"]}
    (ep / "subs.en.json").write_text(json.dumps([cue(0, 3, "S1"), cue(4, 7, "S1"), cue(9, 15, "S2")]),
                                     encoding="utf-8")
    assert _names(voice_bank.export_references(bank, refs, "cpu", stitch=False)) == {}  # gaps break the stretch
    assert "склейка" not in voice_bank.reason_text(json.loads((refs / "missing.json").read_text("utf-8"))["Amy Adams"])
    index = voice_bank.export_references(bank, refs, "cpu")
    amy = next(r for r in index if r["name"] == "Amy Adams")
    assert amy["stitched"] is True and amy["pieces"] == [[0.0, 3.0], [4.0, 7.0]] and (amy["start"], amy["end"]) == (0.0, 7.0)
    assert (refs / amy["wav"]).exists()
    reject = voice_bank.reject_reference(refs, "Amy Adams")
    assert reject is not None
    again = voice_bank.export_references(bank, refs, "cpu")
    assert all(r["name"] != "Amy Adams" for r in again)
    assert json.loads((refs / "missing.json").read_text("utf-8"))["Amy Adams"]["stitched_failed"] == 1
