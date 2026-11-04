"""Stop and continue: stage progress files, atomic files, broken caches, kept LLM answers, F5 progress."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from stt import llm, progress
from stt.llm import OpenRouterClient
from stt.runner import load_json, save_json
from utils.atomic_io import write_text_atomic
from voiceover import engine_f5
from voiceover.engine import PreparedText, read_cache_entry, write_cache_entry
from voiceover.engine_f5 import F5Engine


def _record(folder: Path, stage: str, engine: str | None = None) -> dict[str, Any]:
    return json.loads(progress.progress_path(folder, stage, engine).read_text(encoding="utf-8"))


def test_track_records_running_done_and_counts(tmp_path: Path) -> None:
    with progress.track(tmp_path, "translate"):
        assert _record(tmp_path, "translate")["state"] == "running"
        progress.tick(3, 10, "lines")
        progress.tick(10, 10, "lines")  # the last tick is always written
        record = _record(tmp_path, "translate")
        assert (record["done"], record["total"], record["unit"]) == (10, 10, "lines")
        assert record["pid"] == os.getpid()
    done = _record(tmp_path, "translate")
    assert done["state"] == "done" and done["finished"] and done["done"] == 10


def test_track_records_failure_and_engine_name(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        with progress.track(tmp_path, "voiceover", "f5"):
            raise RuntimeError("boom")
    record = _record(tmp_path, "voiceover", "f5")
    assert record["state"] == "failed" and record["engine"] == "f5"
    assert progress.progress_path(tmp_path, "voiceover", "f5").name == "voiceover.f5.json"
    progress.tick(1, 2, "lines")  # outside a stage: nothing happens, no error


def test_a_started_program_reports_into_the_stage_file(tmp_path: Path) -> None:
    code = "from stt import progress; progress.tick(7, 9, 'lines'); progress.current().update(9, 9, 'lines', force=True)"
    with progress.track(tmp_path, "voiceover", "vosk") as tracker:
        env = {**os.environ, **tracker.environ(), "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
        subprocess.run([sys.executable, "-c", code], env=env, check=True)
        record = _record(tmp_path, "voiceover", "vosk")
        assert (record["done"], record["total"]) == (9, 9) and record["pid"] == os.getpid()
    assert _record(tmp_path, "voiceover", "vosk")["done"] == 9  # kept by the final write of the stage
    assert not progress.interrupted(_record(tmp_path, "voiceover", "vosk"))


def test_running_record_of_a_dead_process_is_interrupted(tmp_path: Path) -> None:
    with progress.track(tmp_path, "mix") as tracker:
        record = {**tracker.record("running"), "pid": 2 ** 22 + 12345}
        assert progress.interrupted(record)
        assert not progress.interrupted(tracker.record("running"))


def test_save_json_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "sub" / "out.json"
    save_json(path, {"text": "Привет"})
    assert load_json(path) == {"text": "Привет"} and "Привет" in path.read_text(encoding="utf-8")
    assert [p.name for p in path.parent.iterdir()] == ["out.json"]  # no temporary file left

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        save_json(path, {"text": "new"})
    assert load_json(path) == {"text": "Привет"}  # the old file is whole
    assert [p.name for p in path.parent.iterdir()] == ["out.json"]


def test_corrupt_cache_entry_counts_as_missing(tmp_path: Path) -> None:
    meta, wav = tmp_path / "k.json", tmp_path / "k.wav"
    assert read_cache_entry(meta, wav) is None
    sf.write(str(wav), np.zeros(2400, dtype=np.float32), 24000)
    write_cache_entry(meta, {"path": str(wav), "tts_text": "Привет", "error": None})
    assert read_cache_entry(meta, wav)["tts_text"] == "Привет"  # type: ignore[index]
    meta.write_text('{"path": "x", "tts_', encoding="utf-8")  # cut off by a power loss
    assert read_cache_entry(meta, wav) is None
    write_text_atomic(meta, "[1, 2]")
    assert read_cache_entry(meta, wav) is None  # not a record
    write_cache_entry(meta, {"tts_text": "Привет"})
    wav.write_bytes(b"RIFF")  # a clip without audio
    assert read_cache_entry(meta, wav) is None


def _f5_engine(tmp_path: Path) -> tuple[F5Engine, str]:
    python = tmp_path / "python"
    python.write_text("")
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros(24000, dtype=np.float32), 24000)
    ref.with_suffix(".txt").write_text("Hello there.")
    return F5Engine({"voice": str(ref), "python": str(python), "trim": {"pauses_enabled": False}}), str(ref)


def test_f5_redoes_a_line_whose_cache_or_take_file_is_broken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = _f5_engine(tmp_path)
    seen: list[list[str]] = []

    def fake_worker(items: list[dict[str, Any]], cache_dir: Path) -> None:
        seen.append([item["text"] for item in items])
        for item in items:
            sf.write(item["out_wav"], np.full(4800, 0.3, dtype=np.float32), 24000)
            item["result"] = {"key": item["key"], "seed": 1234, "takes": 1, "error": None}

    monkeypatch.setattr(engine, "_run_worker", fake_worker)
    rows = [PreparedText("c1", "a", "Прив+ет.", "ok"), PreparedText("c2", "b", "Пок+а.", "ok")]
    cache = tmp_path / "cache"
    engine.synthesize_many(rows, cache)
    assert seen == [["Прив+ет.", "Пок+а."]]
    folder = next(p for p in cache.iterdir() if p.is_dir())
    metas = sorted(p for p in folder.glob("*.json") if not p.name.endswith(".take.json"))
    assert len(metas) == 2
    metas[0].write_text("{", encoding="utf-8")  # one record cut off
    engine.synthesize_many(rows, cache)
    assert len(seen) == 2 and len(seen[1]) == 1  # only that line is done again
    key = metas[1].stem
    metas[1].unlink()
    (folder / f"{key}.take.json").write_text("{", encoding="utf-8")  # a worker's note cut off
    engine.synthesize_many(rows, cache)
    assert len(seen) == 3 and len(seen[2]) == 1


FAKE_WORKER = '''
import json, sys, time
from pathlib import Path
job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if job.get("fail"):
    sys.stderr.write("CUDA out of memory (fake)\\n")
    sys.exit(3)
for item in job["items"]:
    time.sleep(0.35)
    wav = Path(item["out_wav"])
    wav.write_bytes(b"x" * 100)
    wav.with_suffix(".take.json").write_text(json.dumps({"key": item["key"], "error": None}))
Path(sys.argv[2]).write_text(json.dumps([{"key": i["key"], "error": None} for i in job["items"]]))
'''


def _fake_python(tmp_path: Path, engine: F5Engine, fail: bool) -> None:
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    wrapper = Path(engine.python)
    # `python -m voiceover.f5_worker job out`: the fake takes the job and result paths
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$3" "$4"\n')
    wrapper.chmod(0o755)
    if fail:  # make the fake fail through the job file
        original = engine_f5.json.dumps
        engine_f5.json.dumps = lambda obj, **kw: original({**obj, "fail": True} if "items" in obj else obj, **kw)  # type: ignore[assignment]


def test_f5_worker_progress_is_polled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, ref = _f5_engine(tmp_path)
    _fake_python(tmp_path, engine, fail=False)
    monkeypatch.setattr(engine_f5, "WORKER_POLL_S", 0.1)
    cache = tmp_path / "cache"
    cache.mkdir()
    items = [{"key": f"k{n}", "out_wav": str(cache / f"k{n}.wav")} for n in range(4)]
    engine._progress_base = (2, 6)  # two lines were cached before the worker started
    with progress.track(tmp_path / "episode", "voiceover", "f5"):
        engine._run_worker(items, cache)
        record = _record(tmp_path / "episode", "voiceover", "f5")
        assert (record["done"], record["total"], record["unit"]) == (6, 6, "lines")
    assert all(item["result"]["error"] is None for item in items)
    assert (cache / "f5_worker.log").exists()


def test_f5_worker_progress_ticks_between_start_and_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = _f5_engine(tmp_path)
    _fake_python(tmp_path, engine, fail=False)
    monkeypatch.setattr(engine_f5, "WORKER_POLL_S", 0.1)
    ticks: list[tuple[int, int | None]] = []
    monkeypatch.setattr(engine_f5.progress, "tick", lambda done, total=None, unit=None: ticks.append((done, total)))
    cache = tmp_path / "cache"
    cache.mkdir()
    items = [{"key": f"k{n}", "out_wav": str(cache / f"k{n}.wav")} for n in range(4)]
    engine._run_worker(items, cache)
    assert any(0 < done < 4 for done, _ in ticks) and ticks[-1] == (4, 4)


def test_f5_worker_failure_reports_the_end_of_its_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = _f5_engine(tmp_path)
    _fake_python(tmp_path, engine, fail=True)
    monkeypatch.setattr(engine_f5, "WORKER_POLL_S", 0.1)
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(RuntimeError, match="exit 3.*CUDA out of memory"):
        engine._run_worker([{"key": "k0", "out_wav": str(cache / "k0.wav")}], cache)


def _client(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]]) -> OpenRouterClient:
    def post(self: OpenRouterClient, body: dict[str, Any], router: str = "openrouter") -> dict[str, Any]:
        calls.append(body)
        content = json.dumps({"items": [{"i": 1, "ru": f"ответ {len(calls)}"}]}, ensure_ascii=False)
        return {"model": "m", "choices": [{"message": {"content": content}, "finish_reason": "stop"}]}

    monkeypatch.setattr(OpenRouterClient, "_post", post)
    monkeypatch.setattr(llm, "_KEPT_NOW", set())  # a new process: nothing was asked yet
    return OpenRouterClient("openrouter:vendor/model", api_key="test-key")


def test_llm_answers_survive_a_stop_and_go_after_the_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    with pytest.raises(KeyboardInterrupt):  # the stage is stopped after two answers
        with progress.track(tmp_path, "translate"):
            first = _client(monkeypatch, calls).chat_json("sys", "part 1")
            _client(monkeypatch, calls).chat_json("sys", "part 2")
            raise KeyboardInterrupt
    assert len(calls) == 2
    kept = list((tmp_path / ".partial" / "llm" / "translate").glob("*.json"))
    assert len(kept) == 2

    with progress.track(tmp_path, "translate"):  # started again with the same inputs
        client = _client(monkeypatch, calls)
        assert client.chat_json("sys", "part 1") == first  # no new request, no new cost
        assert client.chat_json("sys", "part 2")
        assert len(calls) == 2
        assert client.chat_json("sys", "part 3")  # the one that had no answer yet
        assert len(calls) == 3
        # asking the same thing again inside one run is a retry: the model is asked again
        client.chat_json("sys", "part 3")
        assert len(calls) == 4
        # a kept answer that turned out bad is asked again on its retry, not returned once more
        client.chat_json("sys", "part 1")
        assert len(calls) == 5
        progress.clear_partial(tmp_path, "translate")
    assert not (tmp_path / ".partial").exists()

    with progress.track(tmp_path, "translate"):  # `--force translate` on a finished episode asks afresh
        _client(monkeypatch, calls).chat_json("sys", "part 1")
    assert len(calls) == 6


def test_llm_answers_are_not_kept_outside_a_stage_or_for_another_stage(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    _client(monkeypatch, calls).chat_json("sys", "x")
    _client(monkeypatch, calls).chat_json("sys", "x")
    assert len(calls) == 2
    with progress.track(tmp_path, "glossary"):
        _client(monkeypatch, calls).chat_json("sys", "x")
    with progress.track(tmp_path, "speakers"):
        _client(monkeypatch, calls).chat_json("sys", "x")
    assert len(calls) == 4  # each stage has its own answers
    (tmp_path / ".partial" / "llm" / "glossary").mkdir(parents=True, exist_ok=True)
    for broken in (tmp_path / ".partial" / "llm" / "glossary").glob("*.json"):
        broken.write_text("{", encoding="utf-8")
    with progress.track(tmp_path, "glossary"):
        _client(monkeypatch, calls).chat_json("sys", "x")  # a half-written answer is asked again
    assert len(calls) == 5


def test_translation_counts_lines_and_threads_keep_the_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from stt import translate

    calls: list[dict[str, Any]] = []
    answers = {"items": [{"i": 1, "ru": "а"}, {"i": 2, "ru": "б"}]}

    def post(self: OpenRouterClient, body: dict[str, Any], router: str = "openrouter") -> dict[str, Any]:
        calls.append(body)
        return {"model": "m", "choices": [{"message": {"content": json.dumps(answers)}, "finish_reason": "stop"}]}

    monkeypatch.setattr(OpenRouterClient, "_post", post)
    monkeypatch.setattr(llm, "_KEPT_NOW", set())
    client = OpenRouterClient("openrouter:vendor/model", api_key="test-key")
    records = [{"index": i, "start": i * 1000, "end": i * 1000 + 800, "text": ["x"], "speaker": "S01"} for i in (1, 2)]
    budgets = {1: 50, 2: 50}
    seen: list[int] = []
    with progress.track(tmp_path, "translate"):
        out = translate.translate_records(records, [], client, budgets=budgets, batch_size=2, context_size=2,
                                          on_lines=seen.append)
    assert out == {1: "а", 2: "б"} and seen == [2] and len(calls) == 1
    assert len(list((tmp_path / ".partial" / "llm" / "translate").glob("*.json"))) == 1


def test_episode_run_tracks_stages_and_skips_finished_ones(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import argparse

    from stt.run import Episode

    episode = Episode.__new__(Episode)
    episode.media, episode.out, episode.lock = Path("ep.mkv"), tmp_path / "ep", None
    episode.args = argparse.Namespace(until="dub", force=None, start=None, duration=None)
    episode.config, episode.engine, episode.forced_from = {}, "vosk", None
    episode.vo_dir, episode.voice_bank = episode.out / "voiceover.vosk", tmp_path / "S.voices.json"
    episode._subtitles = None
    ran: list[str] = []

    def stage(name: str, output: str) -> Any:
        def run() -> None:
            ran.append(name)
            progress.tick(1, 1, "requests")
            (episode.out / output).write_text("{}", encoding="utf-8")
            kept = progress.partial_dir("llm")
            assert kept is not None
            kept.mkdir(parents=True, exist_ok=True)
            (kept / "answer.json").write_text("{}", encoding="utf-8")

        return run

    episode.out.mkdir()
    for name, output in (("transcribe", "subs.en.json"), ("glossary", "glossary.json"),
                         ("speakers", "speakers.llm.json"), ("translate", "subs.ru.json"),
                         ("dub", "dub.ru.vosk.json")):
        setattr(episode, name, stage(name, output))
    with caplog.at_level("INFO", logger="stt.run"):
        episode.run()
    assert ran == ["transcribe", "glossary", "speakers", "translate", "dub"]
    text = caplog.text
    assert "[ep] translate: finished in" in text and "[ep] dub (vosk): finished in" in text
    assert _record(episode.out, "translate")["state"] == "done"
    assert _record(episode.out, "dub", "vosk")["state"] == "done"
    assert not (episode.out / ".partial").exists()  # finished stages keep no paid answers
    assert (episode.out / ".stages" / "dub.vosk").exists()

    caplog.clear()
    ran.clear()
    episode.forced_from = None
    with caplog.at_level("INFO", logger="stt.run"):
        episode.run()
    assert ran == [] and "[ep] translate: done earlier" in caplog.text and "finished in" not in caplog.text


def test_two_threads_may_write_the_same_file(tmp_path: Path) -> None:
    import threading

    path = tmp_path / "shared.json"
    errors: list[BaseException] = []

    def write(n: int) -> None:
        try:
            for i in range(200):
                save_json(path, {"writer": n, "i": i})
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors and "writer" in load_json(path)
