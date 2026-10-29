from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from stt.dub import plan
from voiceover.engine import PreparedText
from voiceover.engine_f5 import F5Engine


def _ref(folder: Path, name: str) -> str:
    wav = folder / f"{name}.wav"
    sf.write(str(wav), np.zeros(24000, dtype=np.float32), 24000)
    wav.with_suffix(".txt").write_text("Hello there.")
    return str(wav)


def test_speaker_voices_win_over_gender() -> None:
    records = [{"index": 1, "start": 0, "end": 1000, "text": ["Привет"], "text_en": ["Hi"], "gender": "female",
                "speaker": "S01"},
               {"index": 2, "start": 2000, "end": 3000, "text": ["Пока"], "text_en": ["Bye"], "gender": "female",
                "speaker": "S02"}]
    voiced, _ = plan(records, {"female": "f.wav", "male": "m.wav", "unknown": "m.wav"},
                     speaker_voices={"S01": "julie.wav"})
    assert [v["voice"] for v in voiced] == ["julie.wav", "f.wav"]


def test_f5_engine_caches_by_text_and_voice_clip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    python = tmp_path / "python"
    python.write_text("")
    default, julie = _ref(tmp_path, "default"), _ref(tmp_path, "julie")
    engine = F5Engine({"voice": default, "python": str(python), "trim": {"pauses_enabled": False}})
    runs: list[list[dict[str, Any]]] = []

    def fake_worker(items: list[dict[str, Any]], cache_dir: Path) -> None:
        runs.append(items)
        for item in items:
            audio = np.zeros(int(24000 * item["duration"]), dtype=np.float32)
            audio[2400:-2400] = 0.3
            sf.write(item["out_wav"], audio, 24000)
            item["result"] = {"key": item["key"], "seed": 1234, "takes": 1, "error": None}

    monkeypatch.setattr(engine, "_run_worker", fake_worker)
    rows = [PreparedText("c1", "a", "Прив+ет.", "ok"), PreparedText("c2", "b", "Прив+ет.", "ok", voice=julie),
            PreparedText("c3", "c", "Прив+ет.", "ok")]
    results = engine.synthesize_many(rows, tmp_path / "cache")
    assert len(runs[0]) == 2  # one line per (text, voice)
    assert {item["ref_wav"] for item in runs[0]} == {default, julie} and runs[0][0]["ref_text"] == "Hello there."
    assert results[0].path == results[2].path != results[1].path and results[0].error is None
    engine.synthesize_many(rows, tmp_path / "cache")
    assert len(runs) == 1  # all cached
    sf.write(julie, np.ones(24000, dtype=np.float32) * 0.1, 24000)  # another clip under the same name
    engine.synthesize_many(rows, tmp_path / "cache")
    assert len(runs) == 2 and [item["ref_wav"] for item in runs[1]] == [julie]


def test_f5_engine_resumes_an_interrupted_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    python = tmp_path / "python"
    python.write_text("")
    engine = F5Engine({"voice": _ref(tmp_path, "default"), "python": str(python), "trim": {"pauses_enabled": False}})
    runs: list[list[str]] = []

    def worker(items: list[dict[str, Any]], cache_dir: Path, stop_after: int | None = None) -> None:
        runs.append([item["text"] for item in items])
        for n, item in enumerate(items):
            if stop_after is not None and n == stop_after:
                raise RuntimeError("F5 worker failed (exit -15)")  # killed: earlier lines are on disk
            audio = np.zeros(int(24000 * item["duration"]), dtype=np.float32)
            audio[2400:-2400] = 0.3
            sf.write(item["out_wav"], audio, 24000)
            take = {"key": item["key"], "seed": 1234, "takes": 1, "error": None}
            Path(item["out_wav"]).with_suffix(".take.json").write_text(json.dumps(take))
            item["result"] = take

    rows = [PreparedText(f"c{n}", "x", f"Строк+а {n}.", "ok") for n in range(4)]
    monkeypatch.setattr(engine, "_run_worker", lambda items, cache_dir: worker(items, cache_dir, stop_after=2))
    with pytest.raises(RuntimeError):
        engine.synthesize_many(rows, tmp_path / "cache")
    monkeypatch.setattr(engine, "_run_worker", worker)
    results = engine.synthesize_many(rows, tmp_path / "cache")
    assert runs[1] == ["Строк+а 2.", "Строк+а 3."]  # only the lines the killed worker had not finished
    assert all(r.error is None and Path(r.path).exists() for r in results)
    assert not list((tmp_path / "cache").rglob("*.take.json"))  # turned into cache entries
    engine.synthesize_many(rows, tmp_path / "cache")
    assert len(runs) == 2


def test_voice_id_names_the_clip_content(tmp_path: Path) -> None:
    from voiceover.engine_f5 import clip_path, voice_id

    clip = _ref(tmp_path, "julie")
    first = voice_id(clip)
    sf.write(clip, np.ones(24000, dtype=np.float32) * 0.1, 24000)
    assert first != voice_id(clip) and clip_path(first) == clip


def test_short_word_options_change_the_cache_and_reach_the_worker(tmp_path: Path) -> None:
    from voiceover.speech_check import missing_words

    python = tmp_path / "python"
    python.write_text("")
    default = _ref(tmp_path, "default")
    engine = F5Engine({"voice": default, "python": str(python)})
    plain = F5Engine({"voice": default, "python": str(python), "prefix": "", "slack_s": 0, "retry_lost_words": False})
    assert (engine.prefix, engine.slack_s, engine.retry_lost_words) == ("— ", 0.3, True)
    assert engine.synth_identity() != plain.synth_identity()
    assert "prefix" not in plain.synth_identity()  # the old cache stays valid without the options
    assert missing_words(["да", "так", "и", "есть"], ["так", "есть"]) == 2
    assert missing_words(["и", "и"], ["и"]) == 1


def test_lines_without_a_clip_are_spoken_by_teratts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from voiceover.engine import SynthResult

    python = tmp_path / "python"
    python.write_text("")
    julie = _ref(tmp_path, "julie")
    engine = F5Engine({"voice": "teratts:ru_m5", "python": str(python), "teratts": {"device": "cpu"}})
    asked: list[list[PreparedText]] = []

    class FakeTera:
        def synthesize_many(self, rows: list[PreparedText], cache_dir: Path) -> list[SynthResult]:
            asked.append(rows)
            return [SynthResult(row.cue_id, f"/tera/{row.cue_id}.wav", "", 1.0, 1.0, 0, 0) for row in rows]

    monkeypatch.setattr(engine, "_teratts_engine", lambda: FakeTera())

    def fake_worker(items: list[dict[str, Any]], cache_dir: Path) -> None:
        for item in items:
            audio = np.zeros(int(24000 * item["duration"]), dtype=np.float32)
            audio[2400:-2400] = 0.3
            sf.write(item["out_wav"], audio, 24000)
            item["result"] = {"key": item["key"], "seed": 1234, "takes": 1, "error": None}

    monkeypatch.setattr(engine, "_run_worker", fake_worker)
    rows = [PreparedText("c1", "a", "Прив+ет.", "ok"),  # the default voice: TeraTTS
            PreparedText("c2", "b", "Прив+ет.", "ok", voice=julie),
            PreparedText("c3", "c", "Пок+а.", "ok", voice="teratts:ru_f1+ru_f2")]
    results = engine.synthesize_many(rows, tmp_path / "cache")
    assert [r.cue_id for r in results] == ["c1", "c2", "c3"]
    assert [(row.cue_id, row.voice) for row in asked[0]] == [("c1", "ru_m5"), ("c3", "ru_f1+ru_f2")]
    assert results[0].path == "/tera/c1.wav" and results[1].path.endswith(".wav") and "/tera/" not in results[1].path


def test_fallback_voices_come_from_the_teratts_settings(tmp_path: Path) -> None:
    from unittest import mock

    from stt import run

    refs = {"female": _ref(tmp_path, "fallback_female"), "male": _ref(tmp_path, "fallback_male")}
    episode = mock.Mock(engine="f5", references=tmp_path / "season.references")
    episode.config = {"voiceover": {"engines": {
        "f5": {"voices": {"female": "auto", "male": "auto"}, "fallback": "teratts", "nfe": 32},
        "teratts": {"voices": {"female": "ru_f1+ru_f2", "male": "ru_m5"}, "device": "cuda", "sample_rate": 44100}}}}
    with mock.patch("voiceover.engine_f5.fallback_references", return_value=refs), \
            mock.patch.object(run, "load_json", return_value={"engine": {"model_path": "vosk"}}):
        engine = run.Episode._engine(episode)
    assert engine["voices"] == {"female": "teratts:ru_f1+ru_f2", "male": "teratts:ru_m5"}
    assert engine["teratts"] == {"device": "cuda", "voice": "ru_m5", "listen_stress": True} and "fallback" not in engine
    assert engine["voice"].startswith(refs["male"] + "#")  # F5 still clones a default voice (calibration)


def test_f5_engine_sends_listened_words_and_hears_cached_takes_once(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    python = tmp_path / "python"
    python.write_text("")
    stress = tmp_path / "stress.json"
    stress.write_text(json.dumps({"verdicts": [{"word": "Натали"}], "dictionary": {}, "sentences": {}}),
                      encoding="utf-8")
    engine = F5Engine({"voice": _ref(tmp_path, "default"), "python": str(python), "trim": {"pauses_enabled": False}})
    runs: list[list[dict[str, Any]]] = []

    def worker(items: list[dict[str, Any]], cache_dir: Path) -> None:
        runs.append([dict(item) for item in items])
        for item in items:
            audio = np.zeros(int(24000 * item["duration"]), dtype=np.float32)
            audio[2400:-2400] = 0.3
            sf.write(item["out_wav"], audio, 24000)
            item["result"] = {"key": item["key"], "seed": 1234, "takes": 1, "error": None,
                              **({"stress": {"words": sorted(set(item["listen"])), "wrong": [], "heard": []}}
                                 if item.get("listen") else {})}

    monkeypatch.setattr(engine, "_run_worker", worker)
    rows = [PreparedText("c1", "a", "Н+атали, прив+ет.", "ok"), PreparedText("c2", "b", "Прив+ет.", "ok")]
    engine.synthesize_many(rows, tmp_path / "cache")  # no stress check yet: nothing to hear
    assert all("listen" not in item for item in runs[0])
    engine.normalization["stress_dict"] = str(stress)
    engine._listen = None
    engine.synthesize_many(rows, tmp_path / "cache")  # the cached take with a checked name is heard once
    assert [(item["text"], item["listen"], item["take_seed"]) for item in runs[1]] == [("Н+атали, прив+ет.",
                                                                                         ["натали"], 1234)]
    assert runs[1][0]["take_wav"] == runs[1][0]["out_wav"]
    engine.synthesize_many(rows, tmp_path / "cache")
    assert len(runs) == 2  # heard: nothing left to do


def test_vosk_track_sends_teratts_lines_to_teratts(tmp_path: Path) -> None:
    from voiceover.engine import SynthResult, VoskEngine

    engine = VoskEngine.__new__(VoskEngine)
    engine.voice = 56

    def result(row: PreparedText, who: str) -> SynthResult:
        return SynthResult(cue_id=row.cue_id, path=f"{who}:{row.voice}", sha256="", raw_sec=1.0, speech_sec=1.0,
                           edge_trimmed_ms=0, reduced_pauses=0, error=None)

    class Tera:
        def synthesize_many(self, rows, cache_dir):  # noqa: ANN001
            return [result(row, "tera") for row in rows]

    engine._teratts = Tera()
    rows = [PreparedText("c1", "a", "Прив+ет.", "ok"), PreparedText("c2", "b", "Пок+а.", "ok", voice="teratts:ru_m1"),
            PreparedText("c3", "c", "Д+а.", "ok", voice=49)]
    out = engine._with_teratts(rows, tmp_path, lambda own: [result(row, "vosk") for row in own])
    assert [r.path for r in out] == ["vosk:None", "tera:ru_m1", "vosk:49"]


def test_the_host_gets_the_host_voice(tmp_path: Path) -> None:
    import argparse
    import json

    from stt.run import Episode

    out_root = tmp_path / "work"
    media = tmp_path / "Season 29" / "S29E01.mkv"
    media.parent.mkdir()
    args = argparse.Namespace(season_glossary=None, force=None, engine="vosk", until="mix", start=None)
    config = {"voiceover": {"host_voice": "ru_m1+eng_m3", "engines": {}}}
    episode = Episode(media, out_root, args, config)
    episode.out.mkdir(parents=True)
    (episode.out / "speakers.json").write_text(json.dumps([{"id": "S01", "name": "Jeff Probst", "role": "narrator"},
                                                           {"id": "S02", "name": "Jon Misch", "role": "host?"}]))
    episode.cast.write_text(json.dumps({"cast": [{"name": "Jeff Probst", "role": "Host"},
                                                 {"name": "Jon Misch", "role": "Contestant"}]}))
    assert json.loads(episode._host_voices().read_text()) == {"S01": "teratts:ru_m1+eng_m3"}
    episode.engine = "teratts"
    assert json.loads(episode._host_voices().read_text()) == {"S01": "ru_m1+eng_m3"}
    episode.engine = "f5"
    assert episode._host_voices() is None  # the host's own cloned voice there
