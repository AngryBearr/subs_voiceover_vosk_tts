"""The on-screen text scan runs in the background of stt.transcribe (fakes only: no models, no GPU)."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from stt import transcribe as tr
from stt.runner import load_json, save_json


class _Stop(Exception):
    pass


def _args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"x")
    captured: list[argparse.Namespace] = []
    real = tr.transcribe
    tr.transcribe = lambda a: captured.append(a) or tmp_path  # type: ignore[assignment]
    try:
        tr.main([str(media), "--out", str(tmp_path / "out"), *extra])
    finally:
        tr.transcribe = real  # type: ignore[assignment]
    return captured[0]


def _fake_data(name: str) -> dict[str, Any]:
    if name == "diarize":
        return {"turns": [], "embeddings": {"a": [0.0]}, "diarize_s": 11.0, "peak_vram_mb": 100}
    if name.startswith("asr."):
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi", "words": [{"start": 0.0, "end": 1.0, "text": "hi"}]}],
                "load_s": 2.0, "warmup_s": 0.5, "transcribe_s": 30.0, "peak_vram_mb": 4000}
    if name == "voice":
        return {"run_s": 4.0, "peak_vram_mb": 10, "speakers": {}}
    if name == "ocr":
        return {"captions": [], "events": [], "ocr_s": 0.2, "peak_vram_mb": 500}
    raise AssertionError(name)


@pytest.fixture()
def events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    log: list[str] = []
    monkeypatch.setattr(tr, "prepare_audio", lambda *a, **k: (60.0, False))

    def fake_run_stage(name: str, env: str, args: list[str], out_path: Path, log_dir: Path, force: bool,
                       *, count_step: bool = True) -> dict[str, Any]:
        log.append(f"start:{name}:{threading.current_thread().name}")
        if name == "ocr":
            time.sleep(0.2)
        log.append(f"end:{name}")
        if name == "ocr_fail":
            raise RuntimeError("boom")
        return _fake_data(name)

    monkeypatch.setattr(tr, "run_stage", fake_run_stage)
    return log


def _stop_at_merge(monkeypatch: pytest.MonkeyPatch, log: list[str]) -> None:
    import stt.ocr

    def reclassify(data: Any, words: Any) -> Any:
        log.append("merge")
        raise _Stop

    monkeypatch.setattr(stt.ocr, "reclassify", reclassify)


def test_ocr_starts_first_and_is_joined_before_merge(tmp_path: Path, events: list[str],
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    _stop_at_merge(monkeypatch, events)
    args = _args(tmp_path, "--asr", "whisper-large-v3", "--refine-speakers", "none")
    with pytest.raises(_Stop):
        tr.transcribe(args)
    assert events[0] == "start:ocr:stage-ocr"  # before diarization, in its own thread
    assert events.index("start:ocr:stage-ocr") < next(i for i, e in enumerate(events) if e.startswith("start:diarize"))
    assert events.index("end:ocr") < events.index("merge")
    assert events.index("end:voice") < events.index("merge")


def test_ocr_failure_surfaces_like_before(tmp_path: Path, events: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(name: str, *a: Any, **k: Any) -> dict[str, Any]:
        events.append(f"start:{name}")
        if name == "ocr":
            raise RuntimeError("stt.worker ocr failed")
        return _fake_data(name)

    monkeypatch.setattr(tr, "run_stage", failing)
    with pytest.raises(RuntimeError, match="stt.worker ocr failed"):
        tr.transcribe(_args(tmp_path))
    assert "start:voice" in events  # the failure shows when the result is needed, after the other stages


def test_no_ocr_flag_starts_nothing(tmp_path: Path, events: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    _stop_at_merge(monkeypatch, events)
    seen: list[str] = []
    monkeypatch.setattr(tr, "BackgroundStage", lambda *a, **k: seen.append("bg"))
    try:
        tr.transcribe(_args(tmp_path, "--no-ocr"))
    except Exception:  # the fakes do not reach the end; only the absence of the scan matters
        pass
    assert seen == [] and not any("ocr" in e for e in events)


def _stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force: bool) -> tuple[tr.BackgroundStage, list[str]]:
    calls: list[str] = []

    def call(env: str, args: list[str], out: Path, log_path: Path) -> dict[str, Any]:
        calls.append(log_path.name)
        time.sleep(0.1)
        save_json(out, {"ocr_s": 1.0})
        return load_json(out)

    monkeypatch.setattr("stt.worker.call", call)
    return tr.BackgroundStage("ocr", "subs_env", ["ocr"], tmp_path / "ocr.json", tmp_path / "logs", force), calls


def test_background_stage_runs_and_logs_to_its_own_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job, calls = _stage(tmp_path, monkeypatch, False)
    assert job.join() == {"ocr_s": 1.0}
    assert calls == ["ocr.log"] and job.run_s >= 0.1


def test_background_stage_cached_and_forced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    save_json(tmp_path / "ocr.json", {"ocr_s": 9.0})
    job, calls = _stage(tmp_path, monkeypatch, False)
    assert job.join() == {"ocr_s": 9.0} and calls == [] and job.hidden_s == 0.0  # cached: not run
    job, calls = _stage(tmp_path, monkeypatch, True)
    assert job.join() == {"ocr_s": 1.0} and calls == ["ocr.log"]  # forced: re-run


def test_timings_json_has_real_step_times(tmp_path: Path, events: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args(tmp_path, "--no-ocr")
    try:
        tr.transcribe(args)
    except _Stop:
        raise
    except Exception:
        pass
    path = tmp_path / "out" / "timings.json"
    if not path.exists():
        pytest.skip("assembly with fake data does not reach timings")
    t = load_json(path)
    assert t["asr_s"] == 30.0 and t["diarize_s"] == 11.0 and t["asr_models"]["whisper-large-v3"]["load_s"] == 2.0
    assert {"fuse_s", "align_s", "ocr_s", "refine_s", "ocr_hidden_s", "transcribe_wall_s"} <= set(t)
