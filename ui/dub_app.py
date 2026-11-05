"""Browser UI for the English video -> Russian voice-over pipeline (``stt.run``).

    subs_env/bin/python -m ui.dub_app [--port 7861]

The season bar above the tabs: one season (the folder with the original episodes + its results folder, remembered in
``ui.local.json`` by ``ui.state``) drives every tab; there are no other folder fields. Folders are picked with the
desktop dialog or an in-page browser (``ui.folder_pick``). The first line of each tab says what it writes where
(``ui.seasons.tab_notes``). The settings in effect for a season are layered by ``ui.season_settings``: standard,
global settings file, the season's ``season_settings.json``, then the form of the "Обработка" tab.

Tab "Обработка": pick episodes of the season, engines and options, start, watch the stages of every
episode (and a table of their states), find the finished mkv files. A job runs ``stt.run`` in a background process (one
job at a time, engines one after the other); its config (``ui.seasons.season_config``: everything in the results
folder) and log are kept in ``<results>/.ui/``. The last job is kept in
``output/ui_last_job.json``: after a restart of the page (or of the computer) it is found again, still running
(shown and stoppable), or "прервано" with the "▶ Продолжить" button, which runs the same command with the same
settings file; ``ui.job_view`` turns the stage marks, progress files and log into the words on the tab.

Tab "Голоса сезона" (``ui.voices``): the people of the season and the voice clips ESpeech clones: listen, find another
automatically, give a clip by hand from time ranges of an episode, re-voice ESpeech after a change, rebuild all clips,
count the season's voices again, check names across the season.

Every long action is a task of one background runner (``Job`` with a Russian title; one at a time); the status strip
under the season bar shows the running task, and how the last one ended, on every tab.

Tab "Проверка и правка реплик": every Russian line of an episode (filters: the lines ``stt.review``
found suspicious, edited ones, ones too long for their time), the original and the voice-over of
each, edits of text / speaker (with a voice sample) / gender / "not voiced" into ``corrections.json``
(``stt.subtitle_edit``) with a length check, the Subtitle Edit round trip (ASS / SRT out and back
in), and re-voicing only the changed lines.

Tab "Звук и размер видео": hear a 45-second piece of an episode with other voice levels (``stt.mix_preview``),
save the mix levels, mix the episode again without voicing it, shrink a finished video (``stt.video_compress``).

Tab "Справка": what the program does and how to work with it (``ui/help.md``).

Tab "Настройки": the defaults people change (recognition, models, voices, checks, mix), saved into the global
settings file (``stt_config.local.json`` unless another is chosen) or the season's file (``ui.settings``); the
"Обработка" form starts from the settings in effect for the season. Each voice has a
"▶ Послушать" button: a sample phrase in the voice chosen now (``ui.voice_preview``, made in a subprocess).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore

from stt import progress as stt_progress
from stt.refine_speakers import HUB_WEIGHTS, model_available
from stt.run import SUBSCRIPTION_DOWN, apply_preset, finished_folder
from ui import folder_pick, job_view, season_settings, seasons, settings, voice_preview, voices, who

REPO_ROOT = Path(__file__).resolve().parents[1]
HELP = Path(__file__).with_name("help.md")  # the "Справка" tab
VIDEO_EXT = seasons.VIDEO_EXT
ENGINES, MODES, MODELS = settings.ENGINES, settings.MODES, settings.MODELS
TRANSPORTS = {"Подписка (opencode)": "opencode", "API OpenRouter": "openrouter"}
STAGE_RU = job_view.STAGE_RU
LAST_JOB = REPO_ROOT / "output" / "ui_last_job.json"  # the last job, so it can be found again after a restart
RESUMABLE = ("прервано", "остановлено")


DEFAULT_TITLE = "Обработка серий"
TASK_MODULES = (b"stt.run", b"stt.voice_bank", b"stt.video_compress")  # programs the runner starts


@dataclass
class Job:
    """A background task of the page (one at a time): ``stt.run`` for ``files`` (``cmd`` is None), or any other
    command (``cmd``: the voice clips, the video compression). Every task has a Russian ``title`` for the status strip."""

    files: list[Path]
    out_root: Path
    engines: list[str]
    config_path: Path
    log_path: Path
    extra: list[str] = field(default_factory=list)
    process: subprocess.Popen[bytes] | None = None
    engine: str | None = None
    state: str = "идёт"
    started: float = field(default_factory=time.time)
    finished: float | None = None
    pid: int | None = None  # of the program: it keeps running when this page is restarted
    boot: str = ""
    title: str = DEFAULT_TITLE
    cmd: list[str] | None = None  # None: ``stt.run`` for files / engines / extra
    result: str = ""  # a file the task makes (shown after "Готово")

    @property
    def is_stt_run(self) -> bool:
        return self.cmd is None


log = logging.getLogger("ui.dub_app")
JOB: Job | None = None
LOCK = threading.Lock()


def save_job(job: Job | None, path: Path | None = None) -> None:
    """Keep the job in ``output/ui_last_job.json`` (atomic), so a restart of the page or the computer finds it."""
    if job is None:
        return
    path = path or LAST_JOB
    data = {"files": [str(f) for f in job.files], "out_root": str(job.out_root), "engines": job.engines,
            "config_path": str(job.config_path), "log_path": str(job.log_path), "extra": job.extra, "pid": job.pid,
            "boot": job.boot, "started": job.started, "finished": job.finished, "state": job.state,
            "title": job.title, "cmd": job.cmd, "result": job.result}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _is_task_process(pid: int) -> bool:
    """The process ``pid`` is one of the programs of the runner (a process id can be reused by something else after
    a restart)."""
    try:
        return any(module in TASK_MODULES for module in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0"))
    except OSError:
        return False


def job_alive(job: Job | None) -> bool:
    """The program of the job still runs (started by this page, or found again after a restart)."""
    if job is None:
        return False
    if job.process is not None:
        return job.process.poll() is None
    return bool(job.pid) and stt_progress.process_alive({"pid": job.pid, "boot": job.boot}) and _is_task_process(job.pid)


def _episodes(job: Job) -> list[tuple[str, Path]]:
    return [(job_view.short_name(media.stem), job.out_root / media.stem) for media in job.files]


def _gone_state(job: Job) -> str:
    """State of a job whose process is gone and whose exit code nobody saw: all mixes made -> done, else interrupted."""
    if not job.is_stt_run:
        return "прервано"
    complete = all((job.out_root / m.stem / ".stages" / f"mix.{engine}").exists()
                   for m in job.files for engine in job.engines)
    return "готово" if complete and job.files else "прервано"


def restore_last_job(path: Path | None = None) -> Job | None:
    """Load the last job after a start of the page. Still running: shown as running (and stoppable); gone while it
    was running: "прервано" (or "готово" when everything was made). Sets ``JOB``."""
    global JOB
    path = path or LAST_JOB
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        job = Job(files=[Path(f) for f in data["files"]], out_root=Path(data["out_root"]), engines=list(data["engines"]),
                  config_path=Path(data["config_path"]), log_path=Path(data["log_path"]),
                  extra=list(data.get("extra") or []), state=str(data.get("state") or "прервано"),
                  started=float(data.get("started") or time.time()), finished=data.get("finished"),
                  pid=data.get("pid"), boot=str(data.get("boot") or ""), title=str(data.get("title") or DEFAULT_TITLE),
                  cmd=list(data["cmd"]) if data.get("cmd") else None, result=str(data.get("result") or ""))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if job.state == "идёт":
        if job_alive(job):
            threading.Thread(target=_watch, args=(job, path), daemon=True).start()
        else:
            job.state, job.finished = _gone_state(job), time.time()
            save_job(job, path)
    JOB = job
    return job


def _watch(job: Job, path: Path | None = None) -> None:
    """For a job found again after a restart: wait until its process is gone, then settle the state."""
    while job_alive(job):
        time.sleep(2)
    if job.state == "идёт":
        job.state = _gone_state(job)
    job.finished = time.time()
    save_job(job, path)


def list_videos(folder: str) -> list[str]:
    return seasons.video_names(Path(folder).expanduser())


def base_config() -> dict[str, Any]:
    """stt_config.json with the global settings file on top (``stt_config.local.json`` unless another is chosen)."""
    return season_settings.global_config()


def season_base(out_root: str) -> dict[str, Any]:
    """The settings in effect for the season with this results folder: standard + global + the season's own file."""
    return season_settings.effective(base_config(), _root(out_root) if out_root else None)


def job_config(mode: str, model: str, subtitles: bool, duck_voices: bool, stress: bool, refine: bool,
               f5_fast: bool = False, transport: str | None = None, preset: str | None = None,
               base: dict[str, Any] | None = None) -> dict[str, Any]:
    """The settings in effect (``base``, else the global ones), the chosen set of settings on top, then the options
    of the form."""
    config = apply_preset(base_config() if base is None else base, preset)
    config["mode"] = MODES.get(mode, mode)
    config.setdefault("llm", {})["model"] = MODELS.get(model, model)
    if TRANSPORTS.get(transport or "") == "openrouter":  # pay per request instead of the subscription
        config["llm"].update(opencode=[], router="openrouter")
    config.setdefault("mix", {}).update(subtitles=subtitles, duck="voices" if duck_voices else "all")
    config.setdefault("voiceover", {})["stress_check"] = stress
    config["speaker_refine"] = {**(config.get("speaker_refine") or {}), "model": "resnet293" if refine else "none"}
    f5 = config["voiceover"].setdefault("engines", {}).setdefault("f5", {"voices": {"female": "auto", "male": "auto"}})
    nfe = int(f5.get("nfe", 32))
    f5["nfe"] = 7 if f5_fast else (32 if nfe == 7 else nfe)  # 7 steps: ~2x faster, by ear almost the same
    return config


def _run(job: Job) -> None:
    """The job's thread: one stt.run for all engines (Vosk next to the GPU engines, ``stt.run``), or the job's command."""
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    with job.log_path.open("ab") as log_file:
        engines = ",".join(job.engines)
        job.engine = engines
        cmd = job.cmd or [sys.executable, "-m", "stt.run", *map(str, job.files), "--out-root", str(job.out_root),
                          "--config", str(job.config_path), "--engine", engines, *job.extra]
        log_file.write(f"\n=== {time.strftime('%H:%M:%S')} {engines if job.is_stt_run else 'task'}: {' '.join(cmd)}\n".encode())
        log_file.flush()
        job.process = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        job.pid, job.boot = job.process.pid, stt_progress.boot_id()
        save_job(job)
        code = job.process.wait()
        if job.state != "остановлено":
            job.state = "готово" if code == 0 else f"ошибка (код {code}): смотрите лог"
    job.finished = time.time()
    save_job(job)


NO_SEASON = "Сначала выберите сезон вверху страницы (или добавьте его кнопкой «＋ Новый сезон»)."


def _busy() -> str | None:
    """Why nothing new can start: the last job is running (also one found again after a restart)."""
    job = JOB
    if job and (job.state == "идёт" or job_alive(job)):
        minutes = max(1, round((time.time() - job.started) / 60))
        return f"Сейчас идёт «{job.title}» ({minutes} мин). Дождитесь или остановите её."
    return None


def started_text(title: str, detail: str = "") -> str:
    """The answer of a button that started a task."""
    return f"⏳ Запущено: {title}{f' ({detail})' if detail else ''}. Ход — в строке состояния вверху."


def start_task(title: str, cmd: list[str], root: Path, result: str = "") -> str:
    """Run ``cmd`` (a program of ``TASK_MODULES``) as the background task ``title``; its log is in ``<root>/.ui/``.
    Answers at once; a task already running is a refusal that says which."""
    with LOCK:
        if busy := _busy():
            return busy
        stamp = time.strftime("%Y%m%d_%H%M%S")
        job = Job(files=[], out_root=root, engines=[], config_path=Path(), log_path=root / ".ui" / f"task_{stamp}.log",
                  title=title, cmd=cmd, result=result)
        _launch(job)
    return started_text(title)


def _launch(job: Job) -> None:
    global JOB
    JOB = job
    save_job(job)
    threading.Thread(target=_run, args=(job,), daemon=True).start()


def start_job(folder: str, names: list[str], out_root: str, engines: list[str], mode: str, model: str,
              subtitles: bool, duck_voices: bool, stress: bool, refine: bool, preview: bool,
              preview_from: float, preview_minutes: float, f5_fast: bool = False, transport: str | None = None,
              stt_args: list[str] | None = None, preset: str | None = None,
              download_voice_model: bool = False, title: str | None = None) -> str:
    global JOB
    with LOCK:
        if busy := _busy():
            return busy
        if not folder or not out_root:
            return NO_SEASON
        if not names:
            return "Выберите хотя бы одну серию."
        if not engines:
            return "Выберите хотя бы один движок озвучки."
        root = Path(out_root).expanduser()
        if not root.is_absolute():
            root = REPO_ROOT / root
        extra: list[str] = list(stt_args or [])
        if download_voice_model:
            extra.append("--download-voice-model")
        config = job_config(mode, model, subtitles, duck_voices, stress, refine, f5_fast, transport, preset,
                            base=season_base(out_root))
        config = seasons.season_config(config, seasons.Season(Path(folder).expanduser(), root))  # all in the results folder
        if preview:  # a short piece in its own folder, so it never mixes with full episodes
            root = root / "_проба"
            extra += ["--start", str(preview_from * 60), "--duration", str(preview_minutes * 60)]
            config["paths"] = {**(config.get("paths") or {}), "season": None, "finished": None}
        ui_dir = root / ".ui"
        ui_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        config_path = ui_dir / f"config_{stamp}.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        job = Job(files=[Path(folder).expanduser() / n for n in names], out_root=root,
                  engines=[ENGINES[e] for e in engines], config_path=config_path,
                  log_path=ui_dir / f"job_{stamp}.log", extra=extra, title=title or DEFAULT_TITLE)
        _launch(job)
        return started_text(job.title, f"{seasons.episodes_word(len(names))}, озвучка: {', '.join(engines)}")


def voice_model_missing() -> bool:
    """The trained voice model of the config can be downloaded and is not on this computer yet."""
    model = str((base_config().get("season_voices") or {}).get("model") or "")
    return model in HUB_WEIGHTS and not model_available(model)


RECOUNT_TITLE = "Уточнение голосов сезона"


def recount_season_voices(folder: str, out_root: str, download_voice_model: bool = False) -> str:
    """Count the voices of the season in ``folder`` again, from its episodes that already have a transcript. Translated
    episodes stay as they are: a new transcript would mean a new translation and voice-over, LLM costs included."""
    with LOCK:
        if busy := _busy():
            return busy
        if not folder or not out_root:
            return NO_SEASON
        names = list_videos(folder)
        root = _root(out_root)
        done = [n for n in names if (root / Path(n).stem / "speakers.json").exists()]
        if not done:
            return "В этой папке ещё нет серий с распознанной речью и именами: сначала запустите обработку."
        kept = sum((root / Path(n).stem / "subs.ru.json").exists() for n in done)
        if kept == len(done):
            return "Все обработанные серии этой папки уже переведены, пересчитывать нечего: их реплики не меняются."
        ui_dir = root / ".ui"
        ui_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        config_path = ui_dir / f"config_{stamp}.json"
        config_path.write_text(json.dumps(seasons.season_config(season_base(out_root), seasons.Season(Path(folder).expanduser(), root)),
                                                ensure_ascii=False, indent=2), encoding="utf-8")
        extra = ["--season-voices-only", *(["--download-voice-model"] if download_voice_model else [])]
        job = Job(files=[Path(folder).expanduser() / n for n in done], out_root=root, engines=["vosk"],
                  config_path=config_path, log_path=ui_dir / f"job_{stamp}.log", extra=extra, title=RECOUNT_TITLE)
        _launch(job)
        return (started_text(RECOUNT_TITLE, f"серий с именами {len(done)}, реплики изменятся у {len(done) - kept}, "
                                            f"уже переведённые ({kept}) останутся как есть"))


def stop_job() -> str:
    with LOCK:
        job = JOB
        if not job or job.state != "идёт" or not job_alive(job):
            return "Нечего останавливать: сейчас ничего не запущено."
        job.state = "остановлено"
        save_job(job)
        pid = job.process.pid if job.process is not None else job.pid
        try:
            if pid and os.getpgid(pid) == pid:  # ``stt.run`` leads its own process group with its children
                os.killpg(pid, signal.SIGTERM)
            elif pid:
                os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if not job.is_stt_run:
            return f"⏹ Остановлено: {job.title}."
        return "⏹ Остановлено. Готовые этапы сохранены: кнопка «▶ Продолжить» продолжит с места остановки."


def resume_job() -> str:
    """Run the last, interrupted job again: the same files, engines and settings file; finished stages are skipped
    by ``stt.run`` and voiced lines come from the cache, so only the rest is done. A new log is started."""
    with LOCK:
        old = JOB
        if old is None or old.state not in RESUMABLE or not old.is_stt_run:
            return "Продолжать нечего: последняя обработка не прервана."
        if busy := _busy():
            return busy
        if not old.config_path.exists():
            return f"Файла настроек прерванного запуска нет ({old.config_path}): начните обработку заново."
        missing = [m.name for m in old.files if not m.exists()]
        if missing:
            return f"Не найдены видео: {', '.join(missing)}. Подключите диск и повторите."
        stamp = time.strftime("%Y%m%d_%H%M%S")
        job = Job(files=list(old.files), out_root=old.out_root, engines=list(old.engines), config_path=old.config_path,
                  log_path=old.log_path.with_name(f"job_{stamp}.log"), extra=list(old.extra), title=old.title)
        _launch(job)
        return (started_text(old.title, "тот же запуск: те же серии, движки и настройки; готовые этапы пропускаются, "
                                        "озвученные реплики и ответы языковой модели берутся из сохранённого"))


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
    except (OSError, ValueError):  # no settings file (any more): look in the work folder only
        config = {}
    finished = finished_folder(config, media)
    folders = [finished, work] if finished else [work]
    track = work / f"{media.stem}.ru.{engine}.m4a"
    for folder in folders:
        if track.exists() and (folder / f"{media.stem}.ru.mkv").exists():  # all voice-overs in one video
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


SUBSCRIPTION_WARNING = ("⚠️ Вход в подписку ChatGPT (opencode) не работает: шаги с языковой моделью идут через "
                        "платный роутер. Войдите заново командой `opencode auth login`.")


def subscription_failed(job: Job) -> bool:
    """The job's log shows the subscription login failing (at the start or mid-run)."""
    if not job.log_path.exists():
        return False
    text = job.log_path.read_text(encoding="utf-8", errors="replace")[-500_000:]
    return SUBSCRIPTION_DOWN in text or "via opencode failed" in text


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
    plain = [e for e in events if "Предупреждение" not in e]  # a warning is not what the task is doing
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


def strip_view() -> tuple[str, Any]:
    """(text, "Остановить" button) of the status strip; the timer of the page calls this."""
    job = JOB
    return strip_text(job), gr.update(visible=bool(job and job.state == "идёт" and job_alive(job)))


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
    if subscription_failed(job):
        text += "\n\n" + SUBSCRIPTION_WARNING
    return text


def refresh() -> tuple[str, Any, str, Any, str, str, Any]:
    """Everything the timer updates: status, progress table, readable log, results, checkpoints, raw log, "Продолжить"."""
    job = JOB
    headers, rows = progress_view(job)
    checkpoints = job_view.checkpoints(_episodes(job), job.engines, job.state == "идёт", job.started) if job else ""
    can_resume = bool(job and job.is_stt_run and job.state in RESUMABLE and not job_alive(job))
    return (status_text(job), gr.update(value=rows, headers=headers), readable_log(job), gr.update(value=results_rows(job)),
            checkpoints, log_tail(job), gr.update(interactive=can_resume))


# ---------------------------------------------------------------- season folders

def _root(out_root: str) -> Path:
    root = Path(out_root).expanduser()
    return root if root.is_absolute() else REPO_ROOT / root


def _season_root(out_root: str) -> Path:
    """Folder of the season files (voice banks, voice clips): the results folder of the season."""
    return _root(out_root)


# ---------------------------------------------------------------- check and edit tab

SHOW = ["Все реплики", "Для проверки", "С правками", "Не влезают по времени"]
GENDER_RU = {"Мужской": "male", "Женский": "female"}
FIT_MARK = {"ok": "", "fast": "быстро", "limit": "на пределе", "over": "не влезет"}


def list_episodes(out_root: str) -> list[str]:
    """Episode folders with a translation (lines to check and edit)."""
    return sorted(p.parent.name for p in _root(out_root).glob("*/subs.ru.json"))


def list_reviewed(out_root: str) -> list[str]:
    return sorted(p.parent.name for p in _root(out_root).glob("*/review.json"))


def _episode(out_root: str, episode: str) -> Path:
    return _root(out_root) / episode


def _review(folder: Path) -> dict[int, dict[str, Any]]:
    path = folder / "review.json"
    return {int(i["line"]): i for i in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}


def _corrections(folder: Path) -> dict[str, Any]:
    path = folder / "corrections.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def review_note(item: dict[str, Any] | None) -> str:
    if not item:
        return ""
    what = item.get("heard") and f"слышно {item['heard']}" or ""
    if item.get("sounds_like"):
        what = (what + "; " if what else "") + f"похоже на {item['sounds_like']}"
    if item.get("note"):  # e.g. the gender changed by voice (stt.line_gender)
        what = (what + "; " if what else "") + item["note"]
    return what


def edit_rows(out_root: str, episode: str | None, show: str = SHOW[0], query: str = "") -> list[list[Any]]:
    """Rows of the line table: №, time, speaker, gender, text, fit, the correction, what was heard."""
    from stt.subtitle_edit import _policy, current_lines, fit

    if not episode:
        return []
    folder = _episode(out_root, episode)
    review, fixed, policy = _review(folder), _corrections(folder), _policy()
    rows = []
    for line in current_lines(folder):
        verdict = fit(line.text, line.window_ms, policy).verdict
        fix = fixed.get(str(line.index))
        if (show == SHOW[1] and line.index not in review) or (show == SHOW[2] and not fix) \
                or (show == SHOW[3] and verdict != "over"):
            continue
        if query and query.lower() not in f"{line.text} {line.text_en} {line.speaker_name}".lower():
            continue
        gender = "не озвучивать" if line.skip else {"female": "жен", "male": "муж"}.get(line.gender, "?")
        rows.append([line.index, _clock(line.start), line.speaker_name, gender, line.text, FIT_MARK[verdict],
                     json.dumps(fix, ensure_ascii=False) if fix else "", review_note(review.get(line.index))])
    return rows


def _clock(ms: int) -> str:
    seconds = ms // 1000
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def line_audio(out_root: str, episode: str, line: int, engines: list[str]) -> tuple[Any, Any, str]:
    """(original, voice-over) clips of a line as (rate, samples), and its text."""
    import soundfile as sf

    folder = _episode(out_root, episode)
    records = {r["index"]: r for r in json.loads((folder / "subs.ru.json").read_text(encoding="utf-8"))}
    record = records.get(int(line))
    if record is None:
        return None, None, "Нет такой реплики."
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    offset = float(source.get("start") or 0.0)
    start, end = record["start"] / 1000 - 0.5, record["end"] / 1000 + 1.0
    audio, rate = sf.read(str(folder / "audio.wav"), dtype="float32")
    original = (rate, audio[max(0, int((start - offset) * rate)):max(0, int((end - offset) * rate))])
    dub = None
    for engine in engines or ["f5", "teratts", "vosk"]:
        track = folder / f"voiceover.{engine}" / "export" / "track.wav"
        if track.exists():
            info = sf.info(str(track))
            samples, sr = sf.read(str(track), dtype="float32", start=int(max(0.0, start) * info.samplerate),
                                  stop=int((end + 0.5) * info.samplerate))
            dub = (sr, samples)
            break
    english = " ".join(record.get("text_en") or [])
    return original, dub, f"**{line}** · {record.get('speaker_name') or record.get('speaker')}\n\n" \
                          f"EN: {english}\n\nRU: {' '.join(record['text'])}"


def speaker_choices(out_root: str, episode: str) -> list[str]:
    if not episode:
        return []
    speakers = json.loads((_episode(out_root, episode) / "speakers.json").read_text(encoding="utf-8"))
    return [f"{s['id']}: {s.get('name') or s.get('label_ru') or '?'} ({s.get('gender')})"
            for s in speakers if s.get("cues")]


def line_form(out_root: str, episode: str, line: int) -> tuple[str, str, str | None, str, bool, str]:
    """(text, fit, speaker choice, gender, not voiced, hint) of a line as it will be voiced."""
    from stt.subtitle_edit import current_lines, fit

    folder = _episode(out_root, episode)
    current = next(ln for ln in current_lines(folder) if ln.index == int(line))
    choice = next((c for c in speaker_choices(out_root, episode) if c.split(":")[0] == current.speaker), None)
    note = review_note(_review(folder).get(current.index))
    hint = f"Подозрение: {note}." if note else ""
    fix = _corrections(folder).get(str(current.index))
    if fix:
        hint += f" Уже исправлено: `{json.dumps(fix, ensure_ascii=False)}`."
    return (current.text, fit(current.text, current.window_ms).text, choice,
            "Женский" if current.gender == "female" else "Мужской", current.skip, hint.strip())


def fit_text(out_root: str, episode: str, line: Any, text: str) -> str:
    from stt.subtitle_edit import current_lines, fit

    if not episode or line is None or not text:
        return ""
    current = next((ln for ln in current_lines(_episode(out_root, episode)) if ln.index == int(line)), None)
    return fit(text, current.window_ms).text if current else ""


def speaker_sample(out_root: str, episode: str, choice: str | None) -> Any:
    """A voice sample of a speaker: the season's voice clip, else their longest line in the episode."""
    import soundfile as sf

    from stt.voice_bank import same_person

    if not episode or not choice:
        return None
    folder = _episode(out_root, episode)
    sid = choice.split(":")[0]
    card = next((s for s in json.loads((folder / "speakers.json").read_text(encoding="utf-8")) if s["id"] == sid), None)
    if card is None:
        return None
    for index in [*folder.parent.glob("*.references/references.json"),
                  *_season_root(out_root).glob("*.references/references.json")]:
        ref = next((r for r in json.loads(index.read_text(encoding="utf-8"))
                    if card.get("name") and same_person(r["name"], card["name"])), None)
        if ref and (index.parent / ref["wav"]).exists():
            return str(index.parent / ref["wav"])
    records = [r for r in json.loads((folder / "subs.ru.json").read_text(encoding="utf-8")) if r.get("speaker") == sid]
    if not records:
        return None
    longest = max(records, key=lambda r: min(r["end"] - r["start"], 8000))
    offset = float(json.loads((folder / "source.json").read_text(encoding="utf-8")).get("start") or 0.0)
    info = sf.info(str(folder / "audio.wav"))
    begin = max(0.0, longest["start"] / 1000 - offset)
    samples, rate = sf.read(str(folder / "audio.wav"), dtype="float32", start=int(begin * info.samplerate),
                            stop=int((begin + min(longest["end"] - longest["start"], 8000) / 1000) * info.samplerate))
    return rate, samples


def save_line(out_root: str, episode: str, line: Any, text: str, choice: str | None, gender: str,
              skip: bool) -> str:
    """The line as edited in the form -> corrections.json (only what differs from the translation)."""
    from stt import subtitle_edit

    if not episode or line is None:
        return "Выберите серию и реплику."
    folder = _episode(out_root, episode)
    try:
        found = subtitle_edit.set_line(folder, int(line), text=text, speaker=choice.split(":")[0] if choice else None,
                                       gender=GENDER_RU.get(gender), skip=skip)
    except ValueError as error:
        return f"Не сохранено: {error}"
    if not found:
        return f"Реплика {int(line)}: ничего не изменилось."
    change = found[0]
    warn = f"\n\n⚠ {change.fit.text}" if change.fit and change.fit.verdict == "over" else ""
    return (f"Реплика {int(line)}: {'; '.join(change.what)}. Правок в серии: {len(_corrections(folder))}. "
            f"Когда закончите, нажмите «Переозвучить изменённое».{warn}")


def undo_line(out_root: str, episode: str, line: Any) -> str:
    from stt import subtitle_edit

    if not episode or line is None:
        return "Выберите серию и реплику."
    subtitle_edit.reset_line(_episode(out_root, episode), int(line))
    return f"Реплика {int(line)}: правка убрана, будет как в переводе."


def export_subtitles(out_root: str, episode: str, kind: str, scope: str) -> tuple[str | None, str]:
    """The episode's lines as a file for Subtitle Edit (ASS) or any editor (SRT)."""
    from stt import subtitle_edit

    if not episode:
        return None, "Выберите серию."
    folder = _episode(out_root, episode)
    review = scope != "Вся серия"
    if review and not (folder / "review.json").exists():
        return None, "У этой серии нет списка для проверки."
    srt = kind.startswith("SRT")
    out = folder / "edit" / f"{episode}.ru{'.review' if review else ''}{'.srt' if srt else '.ass'}"
    (subtitle_edit.export_srt if srt else subtitle_edit.export_ass)(folder, out, review=review)
    return str(out), f"Готово: `{out.name}`. Скачайте, поправьте и загрузите обратно ниже."


def import_rows(out_root: str, episode: str, path: str | None) -> tuple[list[list[Any]], str]:
    """What an edited file changes (nothing written yet)."""
    from stt import subtitle_edit

    if not episode or not path:
        return [], "Выберите серию и файл."
    found, warnings = subtitle_edit.changes(_episode(out_root, episode), subtitle_edit.read_edited(Path(path)))
    rows = [[c.index, c.time, "; ".join(c.what), FIT_MARK[c.fit.verdict] if c.fit else ""] for c in found]
    text = f"Изменённых реплик: {len(found)}." if found else "Отличий от текущих реплик нет."
    if warnings:
        text += "\n\nНе применится:\n" + "\n".join(f"- {w}" for w in warnings)
    return rows, text


def import_apply(out_root: str, episode: str, path: str | None) -> str:
    from stt import subtitle_edit

    if not episode or not path:
        return "Выберите серию и файл."
    folder = _episode(out_root, episode)
    found, warnings = subtitle_edit.changes(folder, subtitle_edit.read_edited(Path(path)))
    if not found:
        return "Отличий от текущих реплик нет: ничего не записано."
    subtitle_edit.apply(folder, found)
    over = sum(1 for c in found if c.fit and c.fit.verdict == "over")
    return (f"Записано правок: {len(found)} (`corrections.json`)." + (f" Не влезут по времени: {over}." if over else "")
            + (f" Пропущено: {len(warnings)}." if warnings else "") + " Нажмите «Переозвучить изменённое».")


def revoice(out_root: str, episode: str, engines: list[str]) -> str:
    if not episode:
        return "Выберите серию."
    folder = _episode(out_root, episode)
    try:
        media = Path(json.loads((folder / "source.json").read_text(encoding="utf-8"))["media"])
    except (OSError, ValueError, KeyError):
        return f"❌ Не нашёл данные серии {job_view.short_name(episode)}: сначала обработайте её."
    done = [e for e in ENGINES if (folder / f"voiceover.{ENGINES[e]}").exists()]
    return start_job(str(media.parent), [media.name], str(_root(out_root)), engines or done or ["Vosk"],
                     list(MODES)[0], default_model(season_base(out_root)), True, True, True, True, False, 0, 0,
                     title=f"Переозвучка {job_view.short_name(episode)}")


def default_model(config: dict[str, Any] | None = None) -> str:
    model = (base_config() if config is None else config).get("llm", {}).get("model")
    return next((label for label, value in MODELS.items() if value == model), list(MODELS)[0])


def preset_choices() -> list[tuple[str, str]]:
    """(label, name) of the sets of settings in the config."""
    return [(str(p.get("label") or name), name) for name, p in (base_config().get("presets") or {}).items()]


def preset_form(preset: str | None, out_root: str = "") -> list[Any]:
    """What the "Обработка" form shows for a set of settings (on top of the settings in effect for the season):
    engines, mode, subtitles, duck voices, stress, refine, fast ESpeech."""
    values = form_defaults(apply_preset(season_base(out_root), preset))
    return [values[0], values[1], *values[3:]]


def start_with_preset(*values: Any) -> str:
    """``start_job`` from the form; the last values are the voice model tick and the chosen set of settings."""
    return start_job(*values[:-2], download_voice_model=values[-2], preset=values[-1])


def form_defaults(config: dict[str, Any] | None = None) -> list[Any]:
    """Starting values of the "Обработка" form from the settings: engines, mode, model and the
    subtitles / duck voices / stress / refine / fast ESpeech boxes."""
    config = base_config() if config is None else config
    chosen = (config.get("ui") or {}).get("engines") or ["vosk"]
    f5 = ((config.get("voiceover") or {}).get("engines") or {}).get("f5") or {}
    return [[label for label, value in ENGINES.items() if value in chosen],
            next((label for label, value in MODES.items() if value == config.get("mode")), list(MODES)[0]),
            default_model(config),
            bool((config.get("mix") or {}).get("subtitles", True)), (config.get("mix") or {}).get("duck") == "voices",
            bool((config.get("voiceover") or {}).get("stress_check", True)),
            (config.get("speaker_refine") or {}).get("model", "resnet293") != "none", int(f5.get("nfe", 32)) == 7]


TARGET_GLOBAL, TARGET_SEASON = "global", "season"


def _season(folder: str, out_root: str) -> seasons.Season | None:
    return seasons.Season(Path(folder).expanduser(), _root(out_root)) if folder and out_root else None


def target_choices(season: seasons.Season | None) -> list[tuple[str, str]]:
    """The radio "Сохранять изменения в"."""
    choices = [("общие настройки", TARGET_GLOBAL)]
    if season is not None:
        choices.append((f"настройки сезона {season.name}", TARGET_SEASON))
    return choices


def effect_text(season: seasons.Season | None) -> str:
    """Which settings the season runs with, in words."""
    if season is None:
        return "Сезон не выбран: действуют общие настройки."
    if season_settings.has_own(season.results):
        return (f"Сейчас для сезона {season.name} действуют: настройки сезона "
                f"(`{season_settings.season_file(season.results)}`).")
    return f"Сейчас для сезона {season.name} действуют: общие настройки."


def save_settings(folder: str, out_root: str, target: str, *values: Any) -> str:
    """"Сохранить": the form goes to the global file or to the season's file, as the radio says."""
    season = _season(folder, out_root)
    try:
        if target == TARGET_SEASON and season is not None:
            path = season_settings.save_own(list(values), season.results, base_config())
            return (f"Сохранено в настройки сезона {season.name}: `{path}`. Действует со следующего запуска "
                    "и только для этого сезона.")
        path = season_settings.global_path()
        own, warnings = settings.save(list(values), local_path=path)
    except settings.SettingsError as exc:
        return "Не сохранено:\n\n" + "\n".join(f"- {p}" for p in exc.problems)
    text = f"Сохранено в общие настройки: `{path}` (своих значений: {own}). Действует со следующего запуска."
    if season is not None and season_settings.has_own(season.results):
        text += f" Сезон {season.name} использует свои настройки: на него это не влияет."
    return text + "".join(f"\n- {w}" for w in warnings)


def reset_settings(folder: str, out_root: str, target: str) -> str:
    """"Сбросить к стандартным": the chosen file gets the standard values."""
    season = _season(folder, out_root)
    if target == TARGET_SEASON and season is not None:
        path = season_settings.reset_own(season.results, settings.read_json(settings.DEFAULTS))
        return f"Настройки сезона {season.name} сброшены к стандартным: `{path}`."
    path = season_settings.global_path()
    settings.reset(path)
    return f"Общие настройки сброшены к стандартным (личные ключи вроде подписки остались): `{path}`."


def remove_season_settings(folder: str, out_root: str) -> str:
    season = _season(folder, out_root)
    if season is None:
        return NO_SEASON
    if not season_settings.remove_own(season.results):
        return f"У сезона {season.name} нет своих настроек: действуют общие."
    return (f"Файл настроек сезона {season.name} удалён (`{season_settings.season_file(season.results)}`): "
            "теперь для сезона действуют общие настройки.")


def apply_global(text: str) -> Any:
    """Use the file in the field as the global settings (made as a copy of the current ones when it is new)."""
    status, message = season_settings.choose_global(text)
    return gr.update() if status == "same" else message


def season_line_text(season: seasons.Season | None) -> str:
    if season is None:
        return seasons.summary(None)
    return f"{seasons.summary(season)} · действуют: {season_settings.in_effect(season.results)}"


def settings_view(folder: str, out_root: str, target: str | None = None) -> list[Any]:
    """Everything that depends on the season's settings, in the order of ``build``'s ``view_outputs``: the fields of
    the "Настройки" tab, the "Обработка" form, the sound sliders, then the texts (the line of what is in effect, the
    radio, the button "Убрать…", the season line, the global file path, the notes of the tabs)."""
    season = _season(folder, out_root)
    config = season_base(out_root)
    choices = target_choices(season)
    has_own = season is not None and season_settings.has_own(season.results)
    wanted = target if target in {value for _, value in choices} else (TARGET_SEASON if has_own else TARGET_GLOBAL)
    notes = seasons.tab_notes(season, season_settings.global_path())
    return [*settings.form_values(config), *form_defaults(config), *mix_form_defaults(out_root),
            effect_text(season), gr.update(choices=choices, value=wanted), gr.update(visible=has_own),
            season_line_text(season), str(season_settings.global_path()), *[notes[tab] for tab in seasons.NOTE_TABS]]


def tts_cache(delete: bool) -> str:
    """The TTS cache cleanup (``voiceover.cache_clean``) as a message: what goes, or what went."""
    from voiceover.cache_clean import apply, cache_dir, default_roots, plan

    cache = cache_dir()
    if not cache.is_dir():
        return "Кэша озвучки нет."
    with LOCK:
        if delete and JOB and JOB.state == "идёт":
            return "Идёт обработка: очистите кэш после неё."
    result = plan(cache, default_roots(), keep_days=3)
    gb = 2 ** 30
    text = (f"Кэш озвучки: {result.keep_bytes / gb:.1f} ГБ нужны сериям (или сделаны за последние 3 дня), "
            f"{result.drop_bytes / gb:.1f} ГБ ({len(result.drop)} файлов) не нужны.")
    if delete:
        apply(result)
        text = f"Удалено {result.drop_bytes / gb:.1f} ГБ из кэша озвучки; осталось {result.keep_bytes / gb:.1f} ГБ."
    return text


# ---------------------------------------------------------------- sound and video size tab

MIX_SECONDS = 45.0
MIX_SETTINGS = [next(s for s in settings.SETTINGS if s.path == path) for path in (
    ("mix", "voice_gain_db"), ("mix", "rest_duck_db"), ("mix", "duck_db"), ("mix", "loudness_lufs"),
    ("mix", "video", "compress"), ("mix", "video", "target_gb"), ("mix", "video", "codec"))]


def mix_form_defaults(out_root: str = "") -> list[Any]:
    """Values of the "Звук и размер видео" form from the settings in effect (same order as ``MIX_SETTINGS``)."""
    config = season_base(out_root)
    return [settings.to_form(s, settings.get(config, s.path)) for s in MIX_SETTINGS]


def episode_media(out_root: str, episode: str) -> Path:
    """The original video of a work folder (``source.json``)."""
    return Path(json.loads((_episode(out_root, episode) / "source.json").read_text(encoding="utf-8"))["media"])


def voiced_engines(out_root: str, episode: str) -> list[str]:
    """Labels of the engines that have a voice-over track in the episode."""
    folder = _episode(out_root, episode)
    return [label for label, value in ENGINES.items() if (folder / f"voiceover.{value}" / "export" / "track.wav").exists()]


def mix_episodes(out_root: str) -> Any:
    found = [p.parents[2].name for p in _root(out_root).glob("*/voiceover.*/export/track.wav")]
    return gr.update(choices=sorted(set(found)))


def mix_episodes_listed(out_root: str) -> tuple[Any, str]:
    """The list of the "Обновить список" button: (the dropdown, what was found)."""
    update = mix_episodes(out_root) if out_root else gr.update(choices=[])
    count = len(update["choices"])
    return update, (f"Серий с готовой озвучкой: {count}." if count else
                    "Серий с готовой озвучкой в папке результатов сезона пока нет (или сезон не выбран).")


def find_spot(out_root: str, episode: str | None, engine: str) -> tuple[Any, str]:
    """(minute, message) of the place where the voice-over has the hardest time against the music."""
    if not episode:
        return gr.update(), "Выберите серию."
    from stt import mix_preview

    try:
        spot = mix_preview.loudest_spot(_episode(out_root, episode), episode_media(out_root, episode),
                                        ENGINES[engine], MIX_SECONDS)
    except Exception as error:  # noqa: BLE001 - shown to the person, details in the log
        log.exception("loudest spot")
        return gr.update(), f"❌ Не получилось найти место: {error}"
    return round(spot / 60, 2), f"✅ Самое громкое место: с {int(spot // 60)}:{int(spot % 60):02d}. Минута подставлена выше."


def listen_mix(out_root: str, episode: str | None, engine: str, minute: float, gain: float, rest: float,
               duck_db: float, lufs: float) -> tuple[Any, Any, str]:
    """(new mix, the mix now in the episode, message) of a 45-second piece from ``minute``."""
    if not episode:
        return None, None, "Выберите серию."
    from stt import mix_preview

    folder, label = _episode(out_root, episode), engine
    start = max(0.0, float(minute or 0.0) * 60)
    out_dir = REPO_ROOT / "output" / "mix_preview"
    try:
        new = mix_preview.render_preview(folder, episode_media(out_root, episode), ENGINES[label], start, MIX_SECONDS,
                                        float(gain), float(duck_db), float(rest), float(lufs), out_dir / "new.wav",
                                        duck=season_base(out_root).get("mix", {}).get("duck"))
    except Exception as error:  # noqa: BLE001 - shown to the person, details in the log
        log.exception("mix preview")
        return None, None, f"❌ Не получилось собрать отрывок: {error}"
    track = folder / f"{episode}.ru.{ENGINES[label]}.m4a"
    now = None
    if track.exists():
        now = str(mix_preview.existing_piece(track, start, MIX_SECONDS, out_dir / "now.wav"))
    return str(new), now, ("✅ Слева: с новыми настройками; справа: как сейчас в серии." if now else
                           "✅ Готово. Серия ещё не собрана: сравнивать не с чем.")


def save_mix_settings(out_root: str, *values: Any) -> str:
    """Save the sound sliders where the season's settings live: its own file when it has one, else the global file."""
    results = _root(out_root) if out_root else None
    try:
        if results is not None and season_settings.has_own(results):
            path = season_settings.save_own(list(values), results, base_config(), only=MIX_SETTINGS)
            return f"Сохранено в настройки сезона: `{path}`. Действует со следующего сведения серии."
        path = season_settings.global_path()
        own, warnings = settings.save(list(values), local_path=path, only=MIX_SETTINGS)
    except settings.SettingsError as exc:
        return "Не сохранено:\n\n" + "\n".join(f"- {p}" for p in exc.problems)
    return f"Сохранено в общие настройки: `{path}` (своих значений: {own}). Действует со следующего сведения серии."


def remix(out_root: str, episode: str | None) -> str:
    """Mix the episode again (all its voice-overs) with the saved settings; nothing is voiced again."""
    if not episode:
        return "Выберите серию."
    engines = voiced_engines(out_root, episode)
    if not engines:
        return "У серии нет готовой озвучки."
    try:
        media = episode_media(out_root, episode)
    except (OSError, ValueError, KeyError):
        return f"❌ Не нашёл данные серии {job_view.short_name(episode)}: сначала обработайте её."
    config = season_base(out_root)
    return start_job(str(media.parent), [media.name], str(_root(out_root)), engines, list(MODES)[0], default_model(config),
                     bool((config.get("mix") or {}).get("subtitles", True)), (config.get("mix") or {}).get("duck") == "voices",
                     True, True, False, 0, 0, stt_args=["--force", "mix"],
                     title=f"Пересведение {job_view.short_name(episode)}")


def videos_listed(out_root: str) -> tuple[Any, str]:
    """The list of the "Обновить список" button of the compression: (the dropdown, what was found)."""
    update = finished_videos(out_root)
    count = len(update["choices"])
    return update, (f"Готовых видео: {count}." if count else
                    "Готовых видео (`*.ru.mkv`) в папке результатов пока нет: впишите путь к файлу вручную.")


def finished_videos(out_root: str) -> Any:
    """Finished videos (``*.ru.mkv``) at the top of the results folder (and in the work folders of older runs)."""
    found: list[Path] = []
    if out_root:
        found += _root(out_root).glob("*.ru.mkv")
        found += _root(out_root).glob("*/*.ru.mkv")
    return gr.update(choices=sorted(str(p) for p in dict.fromkeys(found)))


def start_compress(path: str | None, sample: bool, target_gb: float, codec: str, out_root: str = "") -> str:
    """Run ``stt.video_compress`` on a finished video as a task (the video itself is never changed)."""
    src = Path(path or "").expanduser()
    if not src.is_file():
        return "❌ Выберите готовое видео или впишите путь к файлу."
    out = src.with_name(f"{src.stem}.{'sample' if sample else 'small'}.mkv")
    cmd = [sys.executable, "-m", "stt.video_compress", str(src), "--target-gb", str(target_gb),
           "--codec", settings.VIDEO_CODECS[codec]]
    if sample:
        cmd += ["--sample-minute", "10", "--sample-seconds", "60"]
    root = _root(out_root) if out_root else REPO_ROOT / "output" / "mix_preview"
    title = "Сжатие видео (пробный кусок)" if sample else "Сжатие видео"
    answer = start_task(title, cmd, root, result=str(out))
    return f"{answer} Результат: `{out}`" if answer.startswith("⏳") else answer


VOICE_ENGINE = {("voiceover", "engines", "vosk", "voices", "female"): "vosk",
                ("voiceover", "engines", "vosk", "voices", "male"): "vosk",
                ("voiceover", "engines", "teratts", "voices", "female"): "teratts",
                ("voiceover", "engines", "teratts", "voices", "male"): "teratts",
                ("voiceover", "host_voice"): "teratts"}  # settings with a "▶ Послушать" button
FIELDS_PER_ROW = 3


def _setting_field(setting: settings.Setting, value: Any) -> Any:
    if setting.kind == "bool":
        return gr.Checkbox(label=setting.label, value=value, info=setting.info or None)
    if setting.kind == "choice":
        return gr.Dropdown(label=setting.label, choices=list(setting.choices), value=value, info=setting.info or None,
                           min_width=280)
    if setting.kind == "engines":
        return gr.CheckboxGroup(label=setting.label, choices=list(setting.choices), value=value)
    if setting.kind == "vosk_voice":
        return gr.Dropdown(label=setting.label, choices=settings.vosk_voice_choices(setting.path[-1]), value=value,
                           info=setting.info or None, min_width=280)
    if setting.kind in ("number", "integer"):
        return gr.Number(label=setting.label, value=value, precision=0 if setting.kind == "integer" else None,
                         info=setting.info or None, min_width=200)
    if setting.kind == "voice":
        voices = settings.TERATTS_VOICES + sorted(settings.teratts_styles() - set(settings.TERATTS_VOICES))
        return gr.Dropdown(label=setting.label, choices=voices, value=value, allow_custom_value=True,
                           info=setting.info or None, min_width=280)
    return gr.Textbox(label=setting.label, value=value, info=setting.info or None, min_width=280)


def listen_voice(path: tuple[str, ...], voice: Any) -> tuple[Any, str]:
    """(sound, message) of the sample phrase in the voice now chosen in the form (not necessarily saved yet)."""
    engine = VOICE_ENGINE[tuple(path)]
    if voice is None or str(voice).strip() == "":
        return None, "Сначала выберите голос."
    wav, message = voice_preview.preview(engine, voice)
    return (str(wav) if wav else None), message


def _settings_group(group: str, values: dict[tuple[str, ...], Any], fields: dict[tuple[str, ...], Any]) -> None:
    """The fields of one group: the others in rows of three, voices with their "▶ Послушать", boxes one per line,
    so every label and hint is whole. Components are collected in ``fields`` by path (the order of the form
    is ``settings.SETTINGS``, whatever the order on the page)."""
    members = [s for s in settings.SETTINGS if s.group == group]
    plain = [s for s in members if s.kind != "bool" and s.path not in VOICE_ENGINE]
    voices = [s for s in members if s.path in VOICE_ENGINE]
    boxes = [s for s in members if s.kind == "bool"]
    for start in range(0, len(plain), FIELDS_PER_ROW):
        with gr.Row(equal_height=False):
            for setting in plain[start:start + FIELDS_PER_ROW]:
                fields[setting.path] = _setting_field(setting, values[setting.path])
    for setting in voices:  # each voice with its own button, state line and player, right under it
        with gr.Group():
            with gr.Row(equal_height=False):
                fields[setting.path] = _setting_field(setting, values[setting.path])
                listen = gr.Button("▶ Послушать", scale=0, min_width=160)
            note = gr.Markdown(visible=False)
            player = gr.Audio(label="Образец голоса", type="filepath", autoplay=True, interactive=False,
                              visible=False, show_label=False)
        listen.click(lambda voice, path=setting.path: _listen_started(path, voice), fields[setting.path],
                     [listen, note, player], queue=False).then(
            lambda voice, path=setting.path: _listen_outputs(path, voice), fields[setting.path],
            [listen, note, player])
    for setting in boxes:
        fields[setting.path] = _setting_field(setting, values[setting.path])


def _listen_started(path: tuple[str, ...], voice: Any) -> tuple[Any, Any, Any]:
    """Shown at once on the click: the button waits, the line says what happens and how long it may take."""
    engine = VOICE_ENGINE[tuple(path)]
    if voice is not None and str(voice).strip() and voice_preview.preview_path(engine, voice).exists():
        text = "⏳ Включаю образец…"
    elif engine == "vosk" and voice_preview.VOSK_PREPARING.is_set():
        text = "⏳ Образцы голосов Vosk ещё готовятся после запуска программы (до 1,5 мин), дождитесь…"
    else:
        wait = "до 45 с" if engine == "vosk" else "около 15 с"
        text = f"⏳ Готовлю образец этого голоса ({wait}, только в первый раз)…"
    return (gr.update(value="⏳ Готовлю…", interactive=False), gr.update(value=text, visible=True),
            gr.update(value=None, visible=False))


def _listen_outputs(path: tuple[str, ...], voice: Any) -> tuple[Any, Any, Any]:
    sound, message = listen_voice(path, voice)
    text = f"✅ {message} Звучит фраза-образец." if sound else f"❌ {message}"
    return (gr.update(value="▶ Послушать", interactive=True), gr.update(value=text, visible=True),
            gr.update(value=sound, visible=sound is not None))


def wait_state(label: str, note: str) -> tuple[Any, str]:
    """First step of a medium action (a few seconds to a minute): the button waits, a line says what happens."""
    return gr.update(value=f"⏳ {label}", interactive=False), f"⏳ {note}"


def ready_button(label: str) -> Any:
    return gr.update(value=label, interactive=True)


def find_spot_done(out_root: str, episode: str | None, engine: str) -> tuple[Any, Any, str]:
    minute, message = find_spot(out_root, episode, engine)
    return ready_button("Найти самое громкое место"), minute, message


def listen_mix_done(*args: Any) -> tuple[Any, Any, Any, str]:
    new, now, message = listen_mix(*args)
    return ready_button("Послушать"), new, now, message


def cache_done(delete: bool, label: str) -> tuple[Any, str]:
    try:
        message = tts_cache(delete)
    except Exception as error:  # noqa: BLE001 - shown to the person, details in the log
        log.exception("tts cache")
        message = f"❌ Не получилось: {error}"
    return ready_button(label), message


# ---------------------------------------------------------------- the season bar

def season_picker(selected: seasons.Season | None = None) -> Any:
    """The picker's choices and value: the remembered seasons (or the ones found next to the configured folders)."""
    found = seasons.available(base_config())
    current = selected or seasons.pick_current(found)
    if current is not None and current.source not in {s.source for s in found}:
        found.append(current)
    return gr.update(choices=seasons.choices(found), value=str(current.source) if current else None)


def select_season(source: str | None, previous: str = "") -> tuple[Any, ...]:
    """The picker changed: (source, results, summary line, episode list, message, episode table, question panel,
    question text, button "settings of the previous season", the pair the question is about); the choice is saved.
    ``previous`` is the results folder of the season shown before: when the new season has no settings of its own and
    the previous one had some that differ from the global ones, the panel asks which to take."""
    season = seasons.choose(source, base_config()) if source else None
    no_question = (gr.update(visible=False), "", gr.update(), None)
    if season is None:
        return "", "", seasons.summary(None), gr.update(choices=[], value=[]), "", [], *no_question
    names = seasons.video_names(season.source)
    if not season.source.is_dir():
        note = f"⚠ Папка с оригиналами {season.source} недоступна (диск не подключён?)."
    elif not names:
        note = f"⚠ В папке {season.source} нет видео."
    else:
        note = f"Сезон «{season.name}»: {seasons.episodes_word(len(names))}."
    question = no_question
    before = _root(previous) if previous else None
    if season_settings.should_ask(before, season.results, base_config()):
        assert before is not None
        previous_name = _season_name(before)
        question = (gr.update(visible=True), f"**У сезона {season.name} нет своих настроек. Какие взять?**",
                    gr.update(value=f"Настройки сезона {previous_name}"), [str(before), str(season.results)])
    return (str(season.source), str(season.results), season_line_text(season),
            gr.update(choices=seasons.episode_choices(names), value=[]), note, seasons.table(season), *question)


def _season_name(results: Path) -> str:
    """The name of the season whose results folder is ``results`` (from the remembered seasons, else the folder name)."""
    found = next((s for s in seasons.remembered() if s.results == results), None)
    if found is not None:
        return found.name
    return results.name[len(seasons.RESULTS_PREFIX):] if results.name.startswith(seasons.RESULTS_PREFIX) else results.name


def answer_settings_question(pending: list[str] | None, copy: bool) -> tuple[Any, str]:
    """The answer to "Какие взять?": copy the previous season's file into the new season's folder, or leave the
    global settings in effect. Returns (the panel, the message)."""
    hidden = gr.update(visible=False)
    if not pending or len(pending) != 2:
        return hidden, ""
    before, new = Path(pending[0]), Path(pending[1])
    name = _season_name(new)
    if not copy:
        return hidden, f"Для сезона {name} действуют общие настройки."
    try:
        path = season_settings.copy_own(before, new)
    except OSError as error:
        return hidden, f"❌ Не получилось скопировать настройки: {error}. Действуют общие настройки."
    return hidden, f"Для сезона {name} действуют настройки сезона {_season_name(before)}, скопированные в `{path}`."


def season_table(source: str, results: str) -> list[list[str]]:
    """The state of every episode of the season (the timer calls this: folder listings only)."""
    if not source or not results:
        return []
    return seasons.table(seasons.Season(Path(source), Path(results)))


def source_changed(text: str) -> tuple[str, Any]:
    """The folder of the new season was typed: what was found there, and the suggested results folder."""
    ok, note = seasons.describe_source(text)
    return note, (gr.update(value=str(seasons.suggest_results(text))) if ok else gr.update())


def results_changed(source: str, results: str) -> str:
    return seasons.describe_results(source, results)[1]


def save_season(source: str, results: str) -> tuple[str, Any, Any]:
    """"Сохранить сезон": check both folders, create the results folder, remember and select the season."""
    season, problem = seasons.validate(source, results)
    if season is None:
        return problem, gr.update(), gr.update()
    try:
        season.results.mkdir(parents=True, exist_ok=True)
        seasons.remember(season, base_config())
    except OSError as error:
        return f"❌ Не получилось сохранить: {error}", gr.update(), gr.update()
    return f"✅ Сезон «{season.name}» сохранён и выбран.", season_picker(season), gr.update(visible=False)


def build() -> gr.Blocks:
    personal = base_config().get("ui", {})
    start_values = form_defaults()
    with gr.Blocks(title="Озвучка сериалов") as app:
        gr.Markdown("# Озвучка сериалов\nАнглийское видео → русская закадровая озвучка.")
        folder = gr.State("")  # the folder with the original episodes of the current season
        out_root = gr.State("")  # its results folder: work folders, season files, finished videos
        with gr.Row(equal_height=False):
            season_pick = gr.Dropdown(label="Сезон", choices=[], value=None, scale=5,
                                      info="Один сезон для всех вкладок: папка с оригиналами и папка результатов")
            new_season = gr.Button("＋ Новый сезон", scale=1, min_width=160)
        season_line = gr.Markdown(seasons.summary(None))
        with gr.Column(visible=False) as season_form:
            with gr.Row(equal_height=False):
                new_source = gr.Textbox(label="Папка с оригинальными сериями", placeholder="/media/.../Survivor/Season_48",
                                        scale=6)
                pick_source = gr.Button("📁 Выбрать…", scale=1, min_width=140)
            source_note = gr.Markdown()
            with gr.Row(equal_height=False):
                new_results = gr.Textbox(label="Папка результатов (можно изменить)",
                                         info="Здесь будет всё: папки серий, файлы сезона, готовые видео. "
                                              "Если папки нет, она создаётся при «Сохранить сезон»",
                                         scale=6)
                pick_results = gr.Button("📁 Выбрать…", scale=1, min_width=140)
            results_note = gr.Markdown()
            pick_message = gr.Markdown()
            season_browser = folder_pick.build_browser()
            with gr.Row():
                season_save = gr.Button("Сохранить сезон", variant="primary")
                season_cancel = gr.Button("Отмена")
        with gr.Column(visible=False) as ask_panel:  # which settings a season without its own takes
            ask_text = gr.Markdown()
            with gr.Row():
                ask_copy = gr.Button("Настройки сезона", variant="primary")
                ask_global = gr.Button("Общие настройки")
        pending = gr.State(None)  # [results of the previous season, results of the new one] while the question is shown
        season_message = gr.Markdown()
        with gr.Row(equal_height=True):  # the running task, on every tab
            strip = gr.Markdown(strip_text(JOB), elem_id="task-strip")
            strip_stop = gr.Button("Остановить", variant="stop", size="sm", scale=0, min_width=140,
                                   visible=bool(JOB and JOB.state == "идёт"))
        note_boxes: dict[str, Any] = {}  # one line at the top of each tab: what it writes where
        with gr.Tab("Обработка"):
            note_boxes["process"] = gr.Markdown(seasons.NO_SEASON_NOTE)
            show = gr.Button("Обновить список серий")
            episodes = gr.CheckboxGroup(label="Серии сезона", choices=[])
            with gr.Row():
                select_all = gr.Button("Выбрать все", size="sm")
                select_none = gr.Button("Снять выбор", size="sm")
            season_status = gr.Dataframe(headers=["Серия", "Состояние", "Готовое видео"], label="Серии сезона: состояние",
                                         interactive=False, wrap=True)
            preset = gr.Radio(label="Набор настроек (ниже можно поправить отдельные пункты)", choices=preset_choices(), value="main")
            with gr.Row():
                engines = gr.CheckboxGroup(label="Озвучка", choices=list(ENGINES), value=start_values[0])
                mode = gr.Radio(label="Распознавание", choices=list(MODES), value=start_values[1])
                with gr.Column():
                    model = gr.Dropdown(label="Языковая модель", choices=list(MODELS), value=start_values[2])
                    transport = gr.Radio(label="Запросы к модели", choices=list(TRANSPORTS),
                                         value=list(TRANSPORTS)[0], visible=bool(personal.get("llm_transport_choice")))
            with gr.Row():
                subtitles = gr.Checkbox(label="Вшить субтитры", value=start_values[3])
                duck_voices = gr.Checkbox(label="Приглушать только голоса (музыку оставить)", value=start_values[4])
                stress = gr.Checkbox(label="Проверять ударения", value=start_values[5])
                refine = gr.Checkbox(label="Уточнять спикеров по голосу", value=start_values[6])
                f5_fast = gr.Checkbox(label="ESpeech: быстрый режим (7 шагов)", value=start_values[7])
            voice_model = gr.Checkbox(label="Скачать дообученную модель голосов перед обработкой (115 МБ, один раз)",
                                      info="Точнее определяет, кто говорит. Без неё берётся обычная модель.",
                                      value=True, visible=voice_model_missing())
            with gr.Accordion("Пробный отрывок (быстро проверить настройки)", open=False):
                preview = gr.Checkbox(label="Только отрывок (в папку _проба)", value=False)
                with gr.Row():
                    preview_from = gr.Number(label="С минуты", value=10)
                    preview_minutes = gr.Number(label="Минут", value=2)
            with gr.Row():
                start = gr.Button("Начать", variant="primary")
                resume = gr.Button("▶ Продолжить", interactive=False)
                stop = gr.Button("Остановить", variant="stop")
            message = gr.Markdown()
            status = gr.Markdown("Нет запущенной обработки.")
            progress = gr.Dataframe(headers=["Серия"], label="Ход обработки по этапам", interactive=False, wrap=True)
            checkpoints = gr.Markdown(label="Контрольные точки")
            log = gr.Textbox(label="Ход работы", lines=12, max_lines=12, autoscroll=True, interactive=False)
            results = gr.Dataframe(headers=["Серия", "Движок", "Файл"], label="Готовые файлы", interactive=False,
                                   wrap=True)
            with gr.Accordion("Подробный лог", open=False):
                raw_log = gr.Textbox(label="Лог", lines=12, max_lines=12, autoscroll=True, interactive=False)

            def rescan(f: str) -> tuple[Any, str]:
                if not f:
                    return gr.update(), NO_SEASON
                names = list_videos(f)
                return gr.update(choices=seasons.episode_choices(names), value=[]), (f"Найдено серий: {len(names)}." if names
                                                            else f"⚠ В папке {f} нет видео или она недоступна.")

            def pick_all(f: str) -> tuple[Any, str]:
                names = list_videos(f) if f else []
                return gr.update(value=names), (f"Выбрано серий: {len(names)}." if names else NO_SEASON)

            show.click(rescan, folder, [episodes, message])
            select_all.click(pick_all, folder, [episodes, message])
            select_none.click(lambda: (gr.update(value=[]), "Выбор снят."), None, [episodes, message])
            preset.change(preset_form, [preset, out_root], [engines, mode, subtitles, duck_voices, stress, refine, f5_fast])
            start.click(start_with_preset, [folder, episodes, out_root, engines, mode, model, subtitles, duck_voices, stress,
                                            refine, preview, preview_from, preview_minutes, f5_fast, transport, voice_model, preset],
                        message)
            stop.click(stop_job, None, message)
            resume.click(resume_job, None, message)
            gr.Timer(2.0).tick(refresh, None, [status, progress, log, results, checkpoints, raw_log, resume])
            gr.Timer(2.0).tick(strip_view, None, [strip, strip_stop], show_progress="hidden")
            strip_stop.click(stop_job, None, strip)
            gr.Timer(2.0).tick(season_table, [folder, out_root], season_status, show_progress="hidden")

        with gr.Tab("Голоса сезона"):
            voices_tab = voices.build(sys.modules[__name__], folder, out_root, voice_model)

        with gr.Tab("Кто это?"):
            who_tab = who.build(folder, out_root)

        with gr.Tab("Проверка и правка реплик"):
            note_boxes["edits"] = gr.Markdown(seasons.NO_SEASON_NOTE)
            gr.Markdown("Здесь можно поправить русские реплики серии: текст, говорящего, пол голоса или "
                        "«не озвучивать». Правки копятся в `corrections.json` серии; кнопка «Переозвучить "
                        "изменённое» заново озвучит только изменённые реплики (остальные берутся готовыми), "
                        "серия пересоберётся за минуты. Ударение: `+` перед ударной гласной (`зам+ок`). "
                        "Подробно: вкладка «Справка».")
            with gr.Row():
                edit_episode = gr.Dropdown(label="Серия", choices=[], scale=4)
                edit_refresh = gr.Button("Обновить", scale=1)
            with gr.Row():
                edit_show = gr.Radio(SHOW, value=SHOW[1], label="Показать", scale=3)
                edit_query = gr.Textbox(label="Поиск по тексту или имени", scale=2)
            edit_table = gr.Dataframe(
                headers=["№", "Время", "Говорящий", "Пол", "Текст", "Длина", "Правка", "Подозрение"],
                label="Реплики (нажмите на строку, чтобы открыть её)", interactive=False, wrap=True, max_height=420)
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    line_no = gr.Number(label="Реплика №", precision=0, interactive=False)
                    line_text = gr.Markdown()
                    with gr.Row():
                        original_audio = gr.Audio(label="Оригинал", interactive=False)
                        dub_audio = gr.Audio(label="Озвучка сейчас", interactive=False)
                with gr.Column(scale=3):
                    line_hint = gr.Markdown()
                    edit_text = gr.Textbox(label="Русский текст", lines=2)
                    edit_fit = gr.Markdown()
                    with gr.Row():
                        edit_speaker = gr.Dropdown(label="Говорящий", choices=[], scale=3)
                        edit_gender = gr.Radio(list(GENDER_RU), label="Пол голоса", scale=2)
                    sample_audio = gr.Audio(label="Голос выбранного говорящего", interactive=False)
                    edit_skip = gr.Checkbox(label="Не озвучивать эту реплику (останется оригинал)")
                    with gr.Row():
                        save_button_line = gr.Button("Сохранить правку", variant="primary")
                        undo_button_line = gr.Button("Убрать правку")
            with gr.Accordion("Правка в Subtitle Edit (или другом редакторе субтитров)", open=False):
                gr.Markdown("1. Скачайте реплики. **ASS** - для Subtitle Edit: говорящий в колонке «Актёр», пол - "
                            "стиль «Мужской» / «Женский», стиль «Не озвучивать» убирает реплику из озвучки. "
                            "**SRT** - для любого редактора: только текст и имя в начале `[Имя] текст`.\n"
                            "2. Поправьте реплики. Время и колонку «Эффект» (номер реплики) не меняйте, новые "
                            "реплики не добавляйте: они сверяются с серией по номеру и времени.\n"
                            "3. Загрузите исправленный файл, нажмите «Показать изменения», проверьте список и "
                            "нажмите «Записать правки», затем «Переозвучить изменённое».")
                with gr.Row():
                    export_kind = gr.Radio(["ASS (Subtitle Edit: говорящий и пол)", "SRT (текст и [Имя])"],
                                           value="ASS (Subtitle Edit: говорящий и пол)", label="Формат", scale=3)
                    export_scope = gr.Radio(["Вся серия", "Только для проверки"], value="Вся серия", label="Что",
                                            scale=2)
                    export_button = gr.Button("Скачать реплики", scale=1)
                export_out = gr.File(label="Файл для правки", interactive=False)
                import_file = gr.File(label="Исправленный файл (.ass или .srt)", file_types=[".ass", ".srt"],
                                      type="filepath")
                with gr.Row():
                    import_preview = gr.Button("Показать изменения")
                    import_button = gr.Button("Записать правки", variant="primary")
                import_table = gr.Dataframe(headers=["№", "Время", "Что изменится", "Длина"], interactive=False,
                                            wrap=True)
            with gr.Row():
                review_engines = gr.CheckboxGroup(label="Переозвучить движками (пусто: уже готовые)",
                                                  choices=list(ENGINES), value=[], scale=3)
                revoice_button = gr.Button("Переозвучить изменённое", variant="primary", scale=1)
            edit_message = gr.Markdown()

            def load_episodes(root: str, episode: str | None, show: str, query: str) -> tuple[Any, Any, Any]:
                episodes = list_episodes(root)
                episode = episode if episode in episodes else (episodes[0] if episodes else None)
                return (gr.update(choices=episodes, value=episode), gr.update(value=edit_rows(root, episode, show, query)),
                        gr.update(choices=speaker_choices(root, episode), value=None))

            def refresh_episodes(root: str, episode: str | None, show: str, query: str) -> tuple[Any, ...]:
                """The "Обновить" button: the lists again, and a line saying what was found."""
                found = load_episodes(root, episode, show, query)
                count = len(found[0]["choices"])
                return (*found, f"Серий с переводом: {count}." if count else
                        "Переведённых серий в папке результатов пока нет: сначала запустите обработку.")

            def pick_line(root: str, episode: str, show: str, query: str, engines: list[str],
                          evt: gr.SelectData) -> tuple[Any, ...]:
                rows = edit_rows(root, episode, show, query)
                line = rows[evt.index[0]][0] if rows and evt.index[0] < len(rows) else None
                if line is None:
                    return (None, "", None, None, "", "", "", None, None, False)
                try:
                    original, dub, info = line_audio(root, episode, line, [ENGINES[e] for e in engines])
                    text, fit_note, choice, gender, skip, hint = line_form(root, episode, line)
                except Exception as error:  # noqa: BLE001 - shown to the person, details in the log
                    log.exception("opening line %s", line)
                    return line, f"❌ Не получилось открыть реплику {line}: {error}", None, None, "", "", "", None, None, False
                return line, info, original, dub, hint, text, fit_note, choice, gender, skip

            def saved(root: str, episode: str, show: str, query: str, message: str) -> tuple[str, Any]:
                return message, gr.update(value=edit_rows(root, episode, show, query))

            listing = [out_root, edit_episode, edit_show, edit_query]
            edit_refresh.click(refresh_episodes, listing, [edit_episode, edit_table, edit_speaker, edit_message])
            edit_episode.change(load_episodes, listing, [edit_episode, edit_table, edit_speaker])
            edit_show.change(lambda r, e, sh, q: gr.update(value=edit_rows(r, e, sh, q)), listing, edit_table)
            edit_query.submit(lambda r, e, sh, q: gr.update(value=edit_rows(r, e, sh, q)), listing, edit_table)
            edit_table.select(pick_line, [*listing, review_engines],
                              [line_no, line_text, original_audio, dub_audio, line_hint, edit_text, edit_fit,
                               edit_speaker, edit_gender, edit_skip])
            edit_text.change(fit_text, [out_root, edit_episode, line_no, edit_text], edit_fit)
            edit_speaker.change(speaker_sample, [out_root, edit_episode, edit_speaker], sample_audio)
            save_button_line.click(
                lambda r, e, sh, q, n, t, who, g, sk: saved(r, e, sh, q, save_line(r, e, n, t, who, g, sk)),
                [*listing, line_no, edit_text, edit_speaker, edit_gender, edit_skip], [edit_message, edit_table])
            undo_button_line.click(lambda r, e, sh, q, n: saved(r, e, sh, q, undo_line(r, e, n)),
                                   [*listing, line_no], [edit_message, edit_table])
            export_button.click(export_subtitles, [out_root, edit_episode, export_kind, export_scope],
                                [export_out, edit_message])
            import_preview.click(import_rows, [out_root, edit_episode, import_file], [import_table, edit_message])
            import_button.click(lambda r, e, sh, q, f: saved(r, e, sh, q, import_apply(r, e, f)),
                                [*listing, import_file], [edit_message, edit_table])
            revoice_button.click(revoice, [out_root, edit_episode, review_engines], edit_message)

        with gr.Tab("Звук и размер видео"):
            note_boxes["sound"] = gr.Markdown(seasons.NO_SEASON_NOTE)
            mix_start = mix_form_defaults()
            gr.Markdown("### Громкость голосов\nСоберите короткий отрывок (45 секунд) с вашими настройками и "
                        "сравните с тем, что сейчас в серии. Остальную серию программа не трогает. Серии берутся из папки результатов выбранного сезона.")
            with gr.Row():
                mix_episode = gr.Dropdown(label="Серия (с готовой озвучкой)", choices=[], scale=3)
                mix_refresh = gr.Button("Обновить список", scale=1)
                mix_engine = gr.Radio(label="Озвучка", choices=list(ENGINES), value=list(ENGINES)[-1], scale=3)
            with gr.Row():
                mix_minute = gr.Number(label="С минуты", value=10)
                mix_spot = gr.Button("Найти самое громкое место")
            with gr.Row():
                mix_gain = gr.Slider(-3, 12, step=1, value=mix_start[0], label="Голоса громче, дБ")
                mix_rest = gr.Slider(0, 15, step=1, value=mix_start[1], label="Музыка и шум под голосом тише, дБ")
            with gr.Row():
                mix_duck = gr.Slider(6, 24, step=1, value=mix_start[2], label="Голоса оригинала под озвучкой тише, дБ")
                mix_lufs = gr.Slider(-23, -12, step=1, value=mix_start[3], label="Общая громкость, LUFS")
            mix_listen = gr.Button("Послушать", variant="primary")
            with gr.Row():
                mix_new = gr.Audio(label="С этими настройками", type="filepath")
                mix_now = gr.Audio(label="Как сейчас в серии", type="filepath")
            mix_message = gr.Markdown()
            gr.Markdown("### Сжатие видео\nРазрешение остаётся исходным; видео кодирует видеокарта.")
            with gr.Row():
                vid_compress = gr.Checkbox(label="Сжимать видео при сборке серии", value=mix_start[4])
                vid_gb = gr.Slider(0.8, 6, step=0.1, value=mix_start[5], label="Размер серии, ГБ")
                vid_codec = gr.Radio(label="Кодек", choices=list(settings.VIDEO_CODECS), value=mix_start[6])
            with gr.Row():
                mix_save = gr.Button("Сохранить настройки")
                mix_redo = gr.Button("Пересвести эту серию")
            mix_saved = gr.Markdown()
            gr.Markdown("«Пересвести» собирает серию заново с сохранёнными настройками; озвучка не повторяется. "
                        "Ход виден во вкладке «Обработка».")
            gr.Markdown("### Сжать готовое видео\nИсходный файл не меняется: рядом появится файл `.small.mkv` "
                        "(или `.sample.mkv` для пробного куска: 60 секунд с 10-й минуты).")
            with gr.Row():
                vid_path = gr.Dropdown(label="Готовое видео (можно вписать путь)", choices=[], allow_custom_value=True,
                                       scale=4)
                vid_refresh = gr.Button("Обновить список", scale=1)
            with gr.Row():
                vid_go = gr.Button("Сжать", variant="primary")
                vid_sample = gr.Button("Пробный кусок 60 с")
            vid_message = gr.Markdown()
            mix_inputs = [mix_gain, mix_rest, mix_duck, mix_lufs, vid_compress, vid_gb, vid_codec]
            mix_refresh.click(mix_episodes_listed, out_root, [mix_episode, mix_message])
            mix_spot.click(lambda: wait_state("Ищу…", "Ищу самое громкое место серии — до 30 секунд…"), None,
                           [mix_spot, mix_message], queue=False).then(
                find_spot_done, [out_root, mix_episode, mix_engine], [mix_spot, mix_minute, mix_message])
            mix_listen.click(lambda: wait_state("Собираю…", "Собираю отрывок в 45 секунд — около 20 секунд…"), None,
                             [mix_listen, mix_message], queue=False).then(
                listen_mix_done, [out_root, mix_episode, mix_engine, mix_minute, mix_gain, mix_rest, mix_duck, mix_lufs],
                [mix_listen, mix_new, mix_now, mix_message])
            mix_redo.click(remix, [out_root, mix_episode], mix_saved)
            vid_refresh.click(videos_listed, out_root, [vid_path, vid_message])
            vid_go.click(lambda p, g, c, r: start_compress(p, False, g, c, r), [vid_path, vid_gb, vid_codec, out_root],
                         vid_message)
            vid_sample.click(lambda p, g, c, r: start_compress(p, True, g, c, r), [vid_path, vid_gb, vid_codec, out_root],
                             vid_message)

        with gr.Tab("Настройки"):
            note_boxes["settings"] = gr.Markdown(seasons.NO_SEASON_NOTE)
            with gr.Row(equal_height=False):
                global_box = gr.Textbox(label="Общие настройки (файл)", value=str(season_settings.global_path()),
                                        info="По умолчанию stt_config.local.json в папке программы; можно выбрать другой файл "
                                             "(если его нет, он создаётся как копия нынешних общих настроек)", scale=6)
                pick_global = gr.Button("📁 Выбрать…", scale=1, min_width=140)
                use_global = gr.Button("Использовать этот файл", scale=1, min_width=200)
            gr.Markdown("Каждый сезон сначала берёт общие настройки. Настройки сезона хранятся в папке его результатов "
                        "и действуют только для него.")
            global_pick_message = gr.Markdown()
            global_browser = folder_pick.build_browser("Выбор файла общих настроек")
            effect = gr.Markdown(effect_text(None))
            target = gr.Radio(label="Сохранять изменения в", choices=target_choices(None), value=TARGET_GLOBAL)
            gr.Markdown("Показаны значения, которые действуют для выбранного сезона. Сохраняются только отличия от "
                        "`stt_config.json` (прежняя версия файла: `.bak`); в настройки сезона записываются все значения.")
            by_path: dict[tuple[str, ...], Any] = {}
            values = dict(zip((s.path for s in settings.SETTINGS), settings.form_values(base_config())))
            for group in dict.fromkeys(s.group for s in settings.SETTINGS):
                gr.Markdown(f"### {group}")
                _settings_group(group, values, by_path)
            fields = [by_path[s.path] for s in settings.SETTINGS]  # save / reset take the form in SETTINGS order
            with gr.Row():
                save_button = gr.Button("Сохранить", variant="primary")
                reset_button = gr.Button("Сбросить к стандартным")
                remove_button = gr.Button("Убрать настройки сезона (вернуть общие)", visible=False)
            settings_message = gr.Markdown()
            gr.Markdown("### Кэш озвучки\nКаждая озвученная реплика хранится в `output/tts_cache`: повторный "
                        "запуск берёт её оттуда. Реплики, которые не нужны ни одной серии, можно удалить.")
            with gr.Row():
                cache_check = gr.Button("Сколько освободится")
                cache_clean = gr.Button("Очистить кэш озвучки", variant="stop")
            cache_message = gr.Markdown()
            cache_check.click(lambda: wait_state("Считаю…", "Считаю размер кэша — до минуты…"), None,
                              [cache_check, cache_message], queue=False).then(
                lambda: cache_done(False, "Сколько освободится"), None, [cache_check, cache_message])
            cache_clean.click(lambda: wait_state("Очищаю…", "Удаляю ненужное из кэша — до минуты…"), None,
                              [cache_clean, cache_message], queue=False).then(
                lambda: cache_done(True, "Очистить кэш озвучки"), None, [cache_clean, cache_message])
            form = [engines, mode, model, subtitles, duck_voices, stress, refine, f5_fast]

        with gr.Tab("Справка"):
            gr.Markdown(HELP.read_text(encoding="utf-8") if HELP.exists() else "Справка не найдена.")
        # Everything that depends on the season's settings is shown again after a save, a reset and a change of season
        note_boxes["voices"] = voices_tab.note
        view_outputs = [*fields, *form, *mix_inputs, effect, target, remove_button, season_line, global_box,
                        *[note_boxes[tab] for tab in seasons.NOTE_TABS]]
        view_inputs = [folder, out_root, target]
        save_event = mix_save.click(save_mix_settings, [out_root, *mix_inputs], mix_saved)
        save_event.then(settings_view, view_inputs, view_outputs)
        save_button.click(save_settings, [*view_inputs, *fields], settings_message).then(
            settings_view, view_inputs, view_outputs)
        reset_button.click(reset_settings, view_inputs, settings_message).then(settings_view, view_inputs, view_outputs)
        remove_button.click(remove_season_settings, [folder, out_root], settings_message).then(
            lambda f, r: settings_view(f, r, None), [folder, out_root], view_outputs)
        # the global settings file: typed (Enter), the button, or picked with the dialog / in-page browser
        use_global.click(apply_global, global_box, settings_message).then(settings_view, view_inputs, view_outputs)
        global_box.submit(apply_global, global_box, settings_message).then(settings_view, view_inputs, view_outputs)
        for event in folder_pick.wire_browser(global_browser, {"global": global_box},
                                              {"global": (pick_global, folder_pick.FILE, "Файл общих настроек")},
                                              global_pick_message):
            event.then(apply_global, global_box, settings_message).then(settings_view, view_inputs, view_outputs)
        # the season bar: the picker drives every tab
        ask_outputs = [ask_panel, ask_text, ask_copy, pending]
        chosen = season_pick.change(select_season, [season_pick, out_root], [folder, out_root, season_line, episodes,
                                                                             season_message, season_status, *ask_outputs],
                                    show_progress="hidden")
        chosen.then(lambda f, r: settings_view(f, r, None), [folder, out_root], view_outputs)
        chosen.then(voices_tab.reload, voices_tab.inputs, voices_tab.outputs)
        chosen.then(who_tab.reload, who_tab.inputs, who_tab.outputs)
        chosen.then(load_episodes, [out_root, edit_episode, edit_show, edit_query], [edit_episode, edit_table, edit_speaker])
        chosen.then(mix_episodes, out_root, mix_episode)
        chosen.then(finished_videos, out_root, vid_path)
        for button, copy in ((ask_copy, True), (ask_global, False)):
            button.click(lambda pair, copy=copy: answer_settings_question(pair, copy), pending,
                         [ask_panel, season_message]).then(
                lambda f, r: settings_view(f, r, None), [folder, out_root], view_outputs)
        new_season.click(lambda: (gr.update(visible=True), "",
                                  "Укажите папку с оригинальными сериями, например /media/…/Survivor/Season_49."),
                         None, [season_form, season_message, source_note])
        season_cancel.click(lambda: (gr.update(visible=False), "Сезон не добавлен."), None, [season_form, season_message])
        new_source.change(source_changed, new_source, [source_note, new_results], show_progress="hidden")
        new_results.change(results_changed, [new_source, new_results], results_note, show_progress="hidden")
        season_save.click(save_season, [new_source, new_results], [season_message, season_pick, season_form])
        folder_pick.wire_browser(season_browser, {"source": new_source, "results": new_results},
                                 {"source": (pick_source, folder_pick.FOLDER, "Папка с оригинальными сериями"),
                                  "results": (pick_results, folder_pick.FOLDER, "Папка результатов")}, pick_message)
        app.load(season_picker, None, season_pick)
        app.load(settings_view, view_inputs, view_outputs)
    return app


def _prepare_voice_samples() -> None:
    """The samples of all Vosk voices, made once in the background so "▶ Послушать" plays at once."""
    try:
        made = voice_preview.prepare_vosk_voices()
        if made:
            log.info("Vosk voice samples made: %d", made)
    except Exception:  # noqa: BLE001 - only the samples are missing; a click still makes one
        log.exception("Vosk voice samples")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Browser UI for the voice-over pipeline")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    restore_last_job()  # the job of the previous start of the page: running (shown and stoppable) or interrupted
    threading.Thread(target=_prepare_voice_samples, daemon=True).start()
    # clips and videos may live outside the repo (an absolute output folder, the media drive)
    build().launch(server_name=args.host, server_port=args.port, allowed_paths=[str(REPO_ROOT / "output"), "/media"])


if __name__ == "__main__":
    main()
