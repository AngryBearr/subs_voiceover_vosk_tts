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
