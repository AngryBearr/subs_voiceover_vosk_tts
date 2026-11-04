"""The one background-task runner of the page, its status strip, and how every long button answers."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from stt import progress
from ui import dub_app, job_view


@pytest.fixture
def idle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(dub_app, "JOB", None)
    monkeypatch.setattr(dub_app, "LAST_JOB", tmp_path / "last.json")
    return tmp_path


def _task(tmp_path: Path, **kw: Any) -> dub_app.Job:
    values: dict[str, Any] = dict(files=[], out_root=tmp_path, engines=[], config_path=Path(), log_path=tmp_path / "t.log",
                                  title="Сбор образцов голосов", cmd=[sys.executable, "-m", "stt.voice_bank"])
    values.update(kw)
    return dub_app.Job(**values)


def _wait(job: dub_app.Job) -> None:
    for _ in range(300):
        if job.state != "идёт":
            return
        time.sleep(0.05)
    raise AssertionError("the task did not end")


# ---------------------------------------------------------------- one at a time

def test_a_second_task_is_refused_and_says_which_one_runs(idle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    running = _task(idle, started=time.time() - 3 * 60, state="идёт")
    monkeypatch.setattr(dub_app, "JOB", running)
    monkeypatch.setattr(dub_app, "job_alive", lambda job: True)
    answer = "Сейчас идёт «Сбор образцов голосов» (3 мин). Дождитесь или остановите её."
    assert dub_app.start_task("Сжатие видео", ["x"], idle) == answer
    assert dub_app.start_job("f", ["a.mkv"], "o", ["Vosk"], "m", "m", True, True, True, True, False, 0, 0) == answer
    assert dub_app.recount_season_voices("f", "o") == answer
    assert dub_app.JOB is running


def test_the_refusal_names_a_processing_job_too(idle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dub_app, "JOB", _task(idle, cmd=None, title=dub_app.DEFAULT_TITLE, state="идёт"))
    monkeypatch.setattr(dub_app, "job_alive", lambda job: True)
    assert dub_app.start_task("Сжатие видео", ["x"], idle).startswith("Сейчас идёт «Обработка серий» (1 мин).")


# ---------------------------------------------------------------- running a task

def test_a_task_runs_and_ends_with_the_right_state_and_strip(idle: Path) -> None:
    assert dub_app.start_task("Сбор образцов голосов", [sys.executable, "-c", "print('hi')"], idle) == \
        "⏳ Запущено: Сбор образцов голосов. Ход — в строке состояния вверху."
    job = dub_app.JOB
    assert job is not None and job.title == "Сбор образцов голосов" and not job.is_stt_run
    _wait(job)
    assert job.state == "готово" and job.log_path.parent == idle / ".ui" and "hi" in job.log_path.read_text(encoding="utf-8")
    assert dub_app.strip_text(job).startswith("✅ **Готово:** Сбор образцов голосов за ")
    assert json.loads((idle / "last.json").read_text(encoding="utf-8"))["state"] == "готово"
    text, stop = dub_app.strip_view()
    assert text.startswith("✅") and stop["visible"] is False


def test_a_failing_task_says_why_in_the_strip(idle: Path) -> None:
    code = "import sys\nraise ValueError('нет такого файла')"
    dub_app.start_task("Сжатие видео", [sys.executable, "-c", code], idle)
    job = dub_app.JOB
    assert job is not None
    _wait(job)
    assert job.state.startswith("ошибка (код 1)")
    assert dub_app.strip_text(job) == "❌ **Не получилось:** Сжатие видео — ValueError: нет такого файла"


def test_the_strip_while_running_shows_the_last_progress_line(idle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = _task(idle, state="идёт", started=time.time() - 180)
    job.log_path.write_text("=== 10:00:00 task: x\nJeff Probst   S48E03 57.1-63.2s score 0.90, windows >= 0.7, music -8 dB (3 candidates)\n"
                            "The local file (/x/hdemucs.pt) exists. Skipping the download.\n", encoding="utf-8")
    assert dub_app.strip_text(job) == "⏳ **Идёт:** Сбор образцов голосов · 3 мин · Образец готов: Jeff Probst (сходство 0.90)"
    monkeypatch.setattr(dub_app, "JOB", job)
    monkeypatch.setattr(dub_app, "job_alive", lambda j: True)
    text, stop = dub_app.strip_view()
    assert text.startswith("⏳") and stop["visible"] is True
    assert dub_app.status_text(job).startswith("**идёт** · Сбор образцов голосов")


def test_the_strip_of_a_stt_run_uses_the_running_stage(idle: Path, tmp_path: Path) -> None:
    media = tmp_path / "Survivor.S48E01.mkv"
    media.write_text("")
    cfg = tmp_path / "c.json"
    cfg.write_text("{}")
    job = dub_app.Job(files=[media], out_root=tmp_path / "out", engines=["f5"], config_path=cfg, log_path=tmp_path / "j.log",
                      title="Переозвучка ESpeech", started=time.time() - 300)
    folder = job.out_root / media.stem
    now = time.time()
    path = progress.progress_path(folder, "voiceover", "f5")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"stage": "voiceover", "engine": "f5", "state": "running", "done": 120, "total": 640,
                                "unit": "lines", "started": now - 30, "updated": now, "finished": None, "pid": os.getpid(),
                                "boot": progress.boot_id()}), encoding="utf-8")
    text = dub_app.strip_text(job)
    assert text.startswith("⏳ **Идёт:** Переозвучка ESpeech · 5 мин · S48E01 · ESpeech · Озвучка: 120 из 640 реплик")


@pytest.mark.parametrize("state,expected", [("остановлено", "⏹ **Остановлено:** T"), ("прервано", "⏸ **Прервано:** T")])
def test_strip_of_a_stopped_or_interrupted_task(state: str, expected: str) -> None:
    job = dub_app.Job(files=[], out_root=Path("."), engines=[], config_path=Path(), log_path=Path("x.log"), title="T",
                      cmd=["x"], state=state)
    assert dub_app.strip_text(job) == expected + "."
    assert dub_app.strip_text(None) == ""


def test_the_compress_result_is_shown_after_done(tmp_path: Path) -> None:
    out = tmp_path / "ep.small.mkv"
    out.write_bytes(b"x" * 1000)
    job = _task(tmp_path, state="готово", result=str(out), title="Сжатие видео", started=time.time() - 240, finished=time.time())
    assert dub_app.strip_text(job).startswith("✅ **Готово:** Сжатие видео за 4 мин · `") and "ep.small.mkv" in dub_app.strip_text(job)


# ---------------------------------------------------------------- persistence of the last task

def test_a_task_is_saved_found_again_and_never_resumed(idle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = _task(idle, state="идёт", result="/x/y.mkv", pid=2 ** 22 + 77, boot=progress.boot_id())
    dub_app.save_job(job)
    saved = json.loads((idle / "last.json").read_text(encoding="utf-8"))
    assert saved["title"] == "Сбор образцов голосов" and saved["cmd"][-1] == "stt.voice_bank" and saved["result"] == "/x/y.mkv"
    found = dub_app.restore_last_job()
    assert found is not None and found.title == job.title and found.cmd == job.cmd and not found.is_stt_run
    assert found.state == "прервано"  # its program is gone
    assert dub_app.resume_job().startswith("Продолжать нечего")
    assert dub_app.refresh()[-1]["interactive"] is False  # no "Продолжить" for a task that is not stt.run


def test_an_old_job_file_without_a_title_is_a_processing_job(idle: Path) -> None:
    old = {"files": ["/v/a.mkv"], "out_root": "/o", "engines": ["vosk"], "config_path": "/o/c.json", "log_path": "/o/j.log",
           "extra": [], "pid": None, "boot": "", "started": 1.0, "finished": 2.0, "state": "готово"}
    (idle / "last.json").write_text(json.dumps(old), encoding="utf-8")
    job = dub_app.restore_last_job()
    assert job is not None and job.title == dub_app.DEFAULT_TITLE and job.is_stt_run


def test_stop_names_the_task(idle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "-m", "stt.voice_bank"], start_new_session=True)
    try:
        job = _task(idle, state="идёт", pid=process.pid, boot=progress.boot_id())
        monkeypatch.setattr(dub_app, "JOB", job)
        assert dub_app.stop_job() == "⏹ Остановлено: Сбор образцов голосов."
        assert process.wait(timeout=10) != 0
    finally:
        process.kill()
    assert dub_app.stop_job().startswith("Нечего останавливать")


# ---------------------------------------------------------------- the log lines of a task

def test_task_log_lines_in_words_without_library_noise() -> None:
    text = "\n".join([
        "=== 12:00:00 task: python -m stt.voice_bank",
        "The local file (/x/hdemucs_high_trained.pt) exists. Skipping the download.",
        "/x/pyannote/reproducibility.py:74: ReproducibilityWarning: TensorFloat-32 (TF32) has been disabled",
        "Jeff Probst              S48E03 57.1-63.2s score 0.90, windows >= 0.70, music -8 dB (21 candidates)",
        "Saiounia Hughley: no clean reference among 55 candidates (74 кандидата: в 19 рядом звучат другие голоса)",
        "2026-11-03 12:00:05,100 INFO stt.video_compress: encoded 45%",
        "Traceback (most recent call last):", '  File "x.py", line 1, in <module>', "ValueError: no good"])
    assert job_view.task_log_lines(text) == [
        "Образец готов: Jeff Probst (сходство 0.90)",
        "Saiounia Hughley: чистого образца не нашлось (кандидатов: 55)",
        "Кодирование видео: 45 %",
        "❌ Ошибка: ValueError: no good"]
