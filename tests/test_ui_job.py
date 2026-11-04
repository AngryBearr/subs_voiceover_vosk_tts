"""The "Обработка" tab: progress in words, the readable log, the last job after a restart, resume."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from stt import progress
from ui import dub_app, job_view

NOW = 1_800_000_000.0


def _record(folder: Path, stage: str, engine: str | None = None, **fields: Any) -> None:
    data = {"stage": stage, "engine": engine, "state": "running", "done": None, "total": None, "unit": None,
            "started": NOW - 600, "updated": NOW - 1, "finished": None, "pid": os.getpid(), "boot": progress.boot_id()}
    data.update(fields)
    progress._write(progress.progress_path(folder, stage, engine), data)


def _marks(folder: Path, *names: str) -> None:
    (folder / ".stages").mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / ".stages" / name).write_text("")


def test_cells_for_every_state(tmp_path: Path) -> None:
    ep = tmp_path / "S51E01"
    _marks(ep, "transcribe", "glossary", "speakers", "translate", "stress", "dub.vosk", "voiceover.vosk", "check.vosk",
           "mix.vosk", "dub.f5")
    _record(ep, "voiceover", "f5", done=120, total=640, unit="lines", started=NOW - 300)
    headers, rows = job_view.progress_table([("S51E01", ep)], ["vosk", "f5", "teratts"], now=NOW)
    assert headers == ["Серия", "Общие этапы", "Vosk", "ESpeech", "TeraTTS"]
    assert rows == [["S51E01", "✅ готово", "✅ готово", "⏳ Озвучка · 120/640 реплик · ≈ 22 мин", "—"]]
    (ep / ".stages" / "dub.f5").unlink()
    progress.progress_path(ep, "voiceover", "f5").unlink()
    _record(ep, "dub", "f5", state="failed", finished=NOW)
    assert job_view.progress_table([("S51E01", ep)], ["f5"], now=NOW)[1][0][2] == "❌ ошибка: Сценарий"
    # a "running" record of a process that is gone: interrupted
    _record(ep, "voiceover", "f5", pid=2 ** 22 + 12345, done=120, total=640, unit="lines")
    _marks(ep, "dub.f5")
    assert job_view.progress_table([("S51E01", ep)], ["f5"], now=NOW)[1][0][2] == "⏸ прервано: Озвучка 120/640"


def test_shared_cell_counts_and_translate_progress(tmp_path: Path) -> None:
    ep = tmp_path / "ep"
    _marks(ep, "transcribe", "glossary")
    assert job_view.progress_table([("ep", ep)], [], now=NOW)[1] == [["ep", "✔ 2 из 5"]]
    _record(ep, "speakers")
    assert job_view.progress_table([("ep", ep)], [], now=NOW)[1] == [["ep", "⏳ Спикеры"]]
    _record(ep, "speakers", state="done")  # no mark but finished: the table does not claim a stage that is not saved
    _record(ep, "translate", done=3, total=12, unit="parts", started=NOW - 120)
    assert job_view.progress_table([("ep", ep)], [], now=NOW)[1][0][1] == "⏳ Перевод · 3/12 частей · ≈ 6 мин"
    assert job_view.progress_table([("ep", tmp_path / "none")], ["vosk"], now=NOW)[1] == [["ep", "—", "—"]]


def test_records_of_an_earlier_run_are_ignored(tmp_path: Path) -> None:
    ep = tmp_path / "ep"
    _record(ep, "translate", state="failed", updated=NOW - 5000)
    assert job_view.progress_table([("ep", ep)], [], since=NOW - 100, now=NOW)[1] == [["ep", "—"]]
    assert job_view.progress_table([("ep", ep)], [], since=NOW - 9000, now=NOW)[1] == [["ep", "❌ ошибка: Перевод"]]


def test_status_lines_eta_and_checkpoints(tmp_path: Path) -> None:
    ep = tmp_path / "S51E01"
    _marks(ep, "transcribe", "glossary")
    _record(ep, "voiceover", "f5", done=120, total=640, unit="lines", started=NOW - 300)
    _record(ep, "voiceover", "vosk", started=NOW - 90)
    lines = job_view.status_lines([("S51E01", ep)], ["vosk", "f5"], now=NOW)
    assert "S51E01 · ESpeech · Озвучка: 120 из 640 реплик (19 %), осталось ≈ 22 мин" in lines
    assert "S51E01 · Vosk · Озвучка: идёт 90 с" in lines
    text = job_view.checkpoints([("S51E01", ep)], ["vosk", "f5"], running=True)
    assert "сохранено: Распознавание, Словарь" in text
    assert progress.RESUME_NOTES["voiceover"] in text
    assert "**Остановить можно в любой момент:**" in text
    stopped = job_view.checkpoints([("S51E01", ep)], ["f5"], running=False)
    assert "Остановить можно" not in stopped


def test_eta_needs_a_count() -> None:
    assert job_view.eta_text({"done": 0, "total": 10, "started": NOW - 10}, NOW) is None
    assert job_view.eta_text({"done": 5, "total": None, "started": NOW - 10}, NOW) is None
    assert job_view.eta_text({"done": 5, "total": 10, "started": NOW - 10}, NOW) == "меньше минуты"
    assert job_view.eta_text({"done": 1, "total": 3, "started": NOW - 3600}, NOW) == "≈ 2 ч 00 мин"


LOG = """
=== 12:02:59 vosk,f5: /usr/bin/python -m stt.run /v/S51E01.mkv --out-root /o
2026-11-03 12:03:00,100 INFO stt.run: [Survivor.S51E01.720p] transcribe: done earlier
2026-11-03 12:03:01,100 INFO stt.run: [Survivor.S51E01.720p] translate ...
2026-11-03 12:10:01,100 INFO stt.run: [Survivor.S51E01.720p] translate: finished in 420 s
2026-11-03 12:10:02,100 INFO stt.run: [Survivor.S51E01.720p] voiceover (f5) ...
2026-11-03 12:11:02,100 INFO stt.run: [Survivor.S51E01.720p] voiceover (f5): 120/640 lines
2026-11-03 12:11:03,100 INFO stt.dub: 7 lines voiced
2026-11-03 12:11:04,100 WARNING stt.llm: request failed because of a very long reason """ + "x" * 300 + """
2026-11-03 12:11:05,100 ERROR stt.run: The ChatGPT subscription login of OpenCode does not work (run `opencode auth login`)
Traceback (most recent call last):
  File "x.py", line 1, in <module>
    boom()
ValueError: no good
2026-11-03 12:12:00,000 INFO stt.run: [Survivor.S51E01.720p] voiceover (f5): finished in 90 s
"""


def test_readable_log() -> None:
    lines = job_view.readable_log(LOG)
    assert lines[0] == "12:02 Запуск обработки (движки: vosk,f5)"
    assert lines[1:5] == ["S51E01 · Распознавание: сделано раньше", "12:03 S51E01 · Перевод: начат",
                          "12:10 S51E01 · Перевод: готово за 7 мин", "12:10 S51E01 · ESpeech · Озвучка: начата"]
    assert lines[5] == "12:11 S51E01 · ESpeech · Озвучка: 120 из 640 реплик"
    assert lines[6].startswith("12:11 ⚠ Предупреждение: request failed") and lines[6].endswith("…")
    assert len(lines[6]) < 200
    assert lines[7] == "12:11 ❌ Ошибка: вход в подписку ChatGPT (opencode) не работает: запросы идут через платный роутер"
    assert lines[8] == "❌ Ошибка: ValueError: no good"
    assert lines[9] == "12:12 S51E01 · ESpeech · Озвучка: готово за 90 с"
    assert len(lines) == 10 and len(job_view.readable_log(LOG, limit=3)) == 3


# ---------------------------------------------------------------- last job, reattach, resume

def _job(tmp_path: Path, **kw: Any) -> dub_app.Job:
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    media = tmp_path / "ep.mkv"
    media.write_text("")
    return dub_app.Job(files=[media], out_root=tmp_path / "out", engines=["vosk"], config_path=cfg,
                       log_path=tmp_path / "job.log", extra=["--force", "mix"], **kw)


@pytest.fixture
def last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "ui_last_job.json"
    monkeypatch.setattr(dub_app, "LAST_JOB", path)
    monkeypatch.setattr(dub_app, "JOB", None)
    return path


def test_job_is_saved_and_loaded(tmp_path: Path, last: Path) -> None:
    job = _job(tmp_path, state="остановлено", pid=2 ** 22 + 99, boot=progress.boot_id())
    dub_app.save_job(job)
    saved = json.loads(last.read_text(encoding="utf-8"))
    assert saved["state"] == "остановлено" and saved["extra"] == ["--force", "mix"] and saved["engines"] == ["vosk"]
    loaded = dub_app.restore_last_job()
    assert loaded is dub_app.JOB and loaded.files == job.files and loaded.state == "остановлено"
    assert not list(tmp_path.glob(".ui_last_job*"))  # the temporary file is gone


def test_running_job_with_a_dead_process_is_interrupted(tmp_path: Path, last: Path) -> None:
    dub_app.save_job(_job(tmp_path, pid=2 ** 22 + 99, boot=progress.boot_id()))
    job = dub_app.restore_last_job()
    assert job is not None and job.state == "прервано"
    assert json.loads(last.read_text(encoding="utf-8"))["state"] == "прервано"
    # a process id of an earlier boot means nothing
    dub_app.save_job(_job(tmp_path, pid=os.getpid(), boot="another boot"))
    assert dub_app.restore_last_job().state == "прервано"  # type: ignore[union-attr]


def test_finished_while_the_page_was_down(tmp_path: Path, last: Path) -> None:
    job = _job(tmp_path, pid=2 ** 22 + 99, boot=progress.boot_id())
    (job.out_root / "ep" / ".stages").mkdir(parents=True)
    (job.out_root / "ep" / ".stages" / "mix.vosk").write_text("")
    dub_app.save_job(job)
    assert dub_app.restore_last_job().state == "готово"  # type: ignore[union-attr]


def test_a_live_stt_run_is_attached_stopped_and_blocks_a_start(tmp_path: Path, last: Path) -> None:
    fake = tmp_path / "stt" / "run.py"  # a process whose command line has ``stt.run`` in it
    fake.parent.mkdir()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "-m", "stt.run"],
                               start_new_session=True)
    try:
        dub_app.save_job(_job(tmp_path, pid=process.pid, boot=progress.boot_id()))
        job = dub_app.restore_last_job()
        assert job is not None and job.state == "идёт" and dub_app.job_alive(job)
        busy = dub_app.start_job("f", ["a.mkv"], "o", ["Vosk"], "m", "m", True, True, True, True, False, 0, 0)
        assert busy.startswith("Сейчас идёт «Обработка серий» (") and busy.endswith("мин). Дождитесь или остановите её.")
        assert dub_app.resume_job().startswith("Продолжать нечего")
        assert dub_app.stop_job().startswith("⏹ Остановлено")
        assert process.wait(timeout=10) != 0
        assert job.state == "остановлено"
    finally:
        process.kill()
    assert not dub_app.job_alive(job)


def test_a_process_that_is_not_stt_run_is_not_the_job(tmp_path: Path) -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        job = _job(tmp_path, pid=process.pid, boot=progress.boot_id())
        assert not dub_app.job_alive(job)
    finally:
        process.kill()


def test_resume_reruns_the_same_command_with_a_new_log(tmp_path: Path, last: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old = _job(tmp_path, state="прервано")
    monkeypatch.setattr(dub_app, "JOB", old)
    launched: list[dub_app.Job] = []
    monkeypatch.setattr(dub_app.threading, "Thread", lambda target, args, daemon: type(
        "T", (), {"start": lambda self: launched.append(args[0])})())
    time.sleep(1.1)
    message = dub_app.resume_job()
    assert "сохранён" in message and len(launched) == 1
    new = launched[0]
    assert new is dub_app.JOB and new is not old
    assert (new.files, new.engines, new.config_path, new.extra) == (old.files, old.engines, old.config_path, old.extra)
    assert new.log_path != old.log_path and new.state == "идёт"
    assert json.loads(last.read_text(encoding="utf-8"))["log_path"] == str(new.log_path)


def test_resume_refuses_without_files_or_state(tmp_path: Path, last: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert dub_app.resume_job().startswith("Продолжать нечего")
    job = _job(tmp_path, state="прервано")
    job.files[0].unlink()
    monkeypatch.setattr(dub_app, "JOB", job)
    assert "Не найдены видео" in dub_app.resume_job()
    job.state = "готово"
    assert dub_app.resume_job().startswith("Продолжать нечего")


def test_refresh_without_a_running_job_shows_the_last_one(tmp_path: Path, last: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = _job(tmp_path, state="прервано")
    ep = job.out_root / "ep"
    _marks(ep, "transcribe")
    _record(ep, "translate", pid=2 ** 22 + 5, done=3, total=12, unit="parts", started=job.started)
    job.log_path.write_text(LOG.replace("Survivor.S51E01.720p", "ep"), encoding="utf-8")
    monkeypatch.setattr(dub_app, "JOB", job)
    status, progress_update, readable, _, checkpoints, raw, resume = dub_app.refresh()
    assert status.startswith("**прервано**") and "Перевод" in status and "3/12 частей" in status
    assert progress_update["value"] == [["ep", "⏸ прервано: Перевод 3/12", "—"]] or progress_update["value"][0][1].startswith("⏸")
    assert "Перевод: начат" in readable and "Остановлено на этапе" in checkpoints and raw
    assert resume["interactive"] is True
    monkeypatch.setattr(dub_app, "JOB", None)
    assert dub_app.refresh()[0] == "Нет запущенной обработки." and dub_app.refresh()[-1]["interactive"] is False
