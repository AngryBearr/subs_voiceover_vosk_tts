"""Job progress, results and status text for the dubbing page."""

from __future__ import annotations

import json
import time
from pathlib import Path

from stt.run import SUBSCRIPTION_DOWN, finished_folder
from ui import job_view
from ui.app_jobs import Job, RESUMABLE
from utils.llm_route import PAID_MARK, STOP_MARK, stop_message_in, stop_message_ru

STAGE_RU = job_view.STAGE_RU
SUBSCRIPTION_WARNING = ("⚠️ Вход в подписку ChatGPT (opencode) не работает: в этом запуске платить было разрешено, "
                        "поэтому шаги с языковой моделью шли через платный роутер. Войдите заново командой "
                        "`opencode auth login`.")


def _episodes(job: Job) -> list[tuple[str, Path]]:
    return [(job_view.short_name(media.stem), job.out_root / media.stem) for media in job.files]


def progress_view(job: Job | None) -> tuple[list[str], list[list[str]]]:
    """(headers, rows) of the progress table: Серия | Общие этапы | one column per engine."""
    if job is None:
        return ["Серия"], []
    return job_view.progress_table(_episodes(job), job.engines, since=job.started)


def progress_rows(job: Job | None) -> list[list[str]]:
    return progress_view(job)[1]


def result_file(job: Job, media: Path, engine: str) -> Path | None:
    """The finished video with the engine's voice-over (or, without a video, its mixed audio)."""
    work = job.out_root / media.stem
    try:
        config = json.loads(job.config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        config = {}
    finished = finished_folder(config, media)
    folders = [finished, work] if finished else [work]
    track = work / f"{media.stem}.ru.{engine}.m4a"
    for folder in folders:
        if track.exists() and (folder / f"{media.stem}.ru.mkv").exists():
            return folder / f"{media.stem}.ru.mkv"
        if (folder / f"{media.stem}.ru.{engine}.mkv").exists():
            return folder / f"{media.stem}.ru.{engine}.mkv"
    for path in (track, work / f"{media.stem}.ru.{engine}.wav"):
        if path.exists():
            return path
    return None


def results_rows(job: Job | None) -> list[list[str]]:
    if job is None:
        return []
    return [[media.stem, job_view.engine_label(engine), str(path)] for media in job.files for engine in job.engines
            if (path := result_file(job, media, engine))]


def log_tail(job: Job | None, lines: int = 40) -> str:
    """The raw log, the lines that matter."""
    if job is None or not job.log_path.exists():
        return ""
    text = job.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    keep = [line for line in text if " INFO " in line or "ERROR" in line or "Traceback" in line or line.startswith("===")
            or "Error" in line]
    return "\n".join((keep or text)[-lines:])


def current_run_log(job: Job) -> str:
    """The end of the job's log from its last start (a resumed job appends to the same file)."""
    if not job.log_path.exists():
        return ""
    text = job.log_path.read_text(encoding="utf-8", errors="replace")[-500_000:]
    return text[max(text.rfind("\n=== "), 0):]


def subscription_failed(job: Job) -> bool:
    """The job's log shows the subscription failing (at the start or mid-run) or the run stopping because of it."""
    text = current_run_log(job)
    return STOP_MARK in text or SUBSCRIPTION_DOWN in text or "via opencode failed" in text


def llm_note(job: Job) -> str:
    """What to tell about the model requests: the stop message (nothing paid, how to go on) or the old warning
    of a run that was allowed to pay, and the line about money when something was paid."""
    text = current_run_log(job)
    stop = stop_message_in(text)
    notes = []
    if stop:
        notes.append("⚠️ " + stop_message_ru(stop))
    elif subscription_failed(job):
        notes.append(SUBSCRIPTION_WARNING)
    paid = [line for line in text.splitlines() if PAID_MARK in line]
    if paid:
        notes.append("💳 " + paid[-1][paid[-1].index(PAID_MARK):])
    return "\n\n".join(notes)


def readable_log(job: Job | None) -> str:
    """"Ход работы": one plain line per event of the job's log, newest last."""
    if job is None or not job.log_path.exists():
        return ""
    text = job.log_path.read_text(encoding="utf-8", errors="replace")[-300_000:]
    return "\n".join((job_view.readable_log if job.is_stt_run else job_view.task_log_lines)(text))


def last_line(job: Job) -> str:
    """What the task does now, in one line: the running stage of ``stt.run``, else the last event of the log."""
    if job.is_stt_run:
        lines = job_view.status_lines(_episodes(job), job.engines, since=job.started)
        if lines:
            return lines[0]
    events = [e for e in readable_log(job).splitlines() if e]
    plain = [e for e in events if "Предупреждение" not in e]
    return (plain or events or ["подготовка"])[-1]


def failure_reason(job: Job) -> str:
    """A short reason for a task that did not work: the last error line of its log, else the exit state."""
    errors = [e for e in readable_log(job).splitlines() if e.startswith("❌") or "❌" in e[:12]]
    if errors:
        return errors[-1].split("❌", 1)[-1].replace("Ошибка:", "").strip(" :") or job.state
    return job.state


def strip_text(job: Job | None) -> str:
    """The line under the season bar, on every tab: what runs, what ended, how."""
    if job is None:
        return ""
    spent = job_view.fmt_duration((job.finished or time.time()) - job.started)
    if job.state == "идёт":
        return f"⏳ **Идёт:** {job.title} · {spent} · {last_line(job)}"
    if job.state == "готово":
        done = Path(job.result) if job.result else None
        size = f" · `{done}` ({done.stat().st_size / 1e9:.2f} ГБ)" if done and done.is_file() else ""
        return f"✅ **Готово:** {job.title} за {spent}{size}"
    if job.state == "остановлено":
        return f"⏹ **Остановлено:** {job.title}" + (". Можно продолжить на вкладке «Обработка»." if job.is_stt_run else ".")
    if job.state == "прервано":
        return f"⏸ **Прервано:** {job.title}" + (". Продолжить можно на вкладке «Обработка»." if job.is_stt_run else ".")
    return f"❌ **Не получилось:** {job.title} — {failure_reason(job)}"


def status_text(job: Job | None) -> str:
    """The state, the time and what happens now (one line per running stage)."""
    if job is None:
        return "Нет запущенной обработки."
    minutes = ((job.finished or time.time()) - job.started) / 60
    labels = ", ".join(job_view.engine_label(e) for e in job.engines)
    if not job.is_stt_run:
        return f"**{job.state}** · {job.title} · {minutes:.0f} мин. Ход — в окне «Ход работы» ниже и в строке состояния вверху."
    text = f"**{job.state}** · {job.title} · {minutes:.0f} мин · движки: {labels or '-'}"
    episodes = _episodes(job)
    if job.state == "идёт":
        lines = job_view.status_lines(episodes, job.engines, since=job.started) or ["подготовка следующего этапа"]
        text += "\n\n" + "\n".join(f"- {line}" for line in lines)
    elif job.state in RESUMABLE:
        lines = job_view.interrupted_lines(episodes, job.engines, since=job.started)
        if lines:
            text += "\n\nОстановлено на этапах:\n\n" + "\n".join(f"- {line}" for line in lines)
    note = llm_note(job)
    return text + ("\n\n" + note if note else "")
