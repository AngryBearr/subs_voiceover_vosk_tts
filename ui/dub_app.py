"""Assemble the dubbing UI and own its single background-task runner.

Run with ``subs_env/bin/python -m ui.dub_app [--port 7861]``. Tab builders live in
``ui.tabs``; the original helper names remain available here for existing callers.
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
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore

from stt import progress as stt_progress
from stt.refine_speakers import HUB_WEIGHTS, model_available
from stt.run import SUBSCRIPTION_DOWN, apply_preset, finished_folder
from utils.llm_route import PAID_MARK, STOP_MARK, stop_message_in, stop_message_ru
from ui import app_forms, folder_pick, job_view, season_bar, season_settings, seasons, settings, voice_preview, voices, who
from ui.app_forms import PREPARE_HELP, PREPARE_TITLE
from ui.app_jobs import DEFAULT_TITLE, RESUMABLE, TASK_MODULES, Job, _gone_state, _is_task_process, job_alive
from ui.app_paths import ENGINES, HELP, MODES, MODELS, NO_SEASON, REPO_ROOT, TRANSPORTS, VIDEO_EXT
from ui.app_paths import _episode, _root, _season, _season_root
from ui.app_status import (
    STAGE_RU, SUBSCRIPTION_WARNING, _episodes, current_run_log, failure_reason, last_line, llm_note,
    log_tail, progress_rows, progress_view, readable_log, result_file, results_rows, status_text, strip_text,
    subscription_failed,
)
from ui.season_bar import _season_name, answer_settings_question, results_changed, season_table, source_changed
from ui.tabs import edits, help as help_tab, process, settings as settings_tab, sound
from ui.tabs.edits import (
    FIT_MARK, GENDER_RU, SHOW, _clock, _corrections, _review, edit_rows, export_subtitles, fit_text,
    import_apply, import_rows, line_audio, line_form, list_episodes, list_reviewed, review_note,
    save_line, speaker_choices, speaker_sample, undo_line,
)
from ui.tabs.settings import (
    FIELDS_PER_ROW, TARGET_GLOBAL, TARGET_SEASON, VOICE_ENGINE, _listen_outputs, _listen_started,
    _setting_field, _settings_group, apply_global, effect_text, listen_voice, remove_season_settings,
    reset_settings, season_line_text, target_choices,
)
from ui.tabs.sound import (
    MIX_SECONDS, MIX_SETTINGS, episode_media, find_spot, finished_videos, mix_episodes, mix_episodes_listed,
    videos_listed, voiced_engines,
)
from ui.widgets import ready_button, wait_state

LAST_JOB = REPO_ROOT / "output" / "ui_last_job.json"  # the last job, so it can be found again after a restart
RECOUNT_TITLE = "Уточнение голосов сезона"
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


def adopt_external_job(path: Path | None = None) -> bool:
    """Pick up a running job another program started (``ui.run_setup launch`` writes ``ui_last_job.json``) when this
    page has no running job of its own, so the page shows it and does not start a second one. One small file read."""
    global JOB
    path = path or LAST_JOB
    if JOB is not None and JOB.state == "идёт" and job_alive(JOB):
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or data.get("state") != "идёт" or not data.get("pid"):
        return False
    if JOB is not None and JOB.pid == data.get("pid") and JOB.started == data.get("started"):
        return False
    probe = Job(files=[], out_root=Path(), engines=[], config_path=Path(), log_path=Path(), pid=data["pid"],
                boot=str(data.get("boot") or ""))
    if not job_alive(probe):
        return False
    return restore_last_job(path) is not None


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
    return app_forms.job_config(mode, model, subtitles, duck_voices, stress, refine, f5_fast, transport, preset,
                                base=base_config() if base is None else base)


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


def _busy() -> str | None:
    """Why nothing new can start: the last job is running (also one found again after a restart)."""
    adopt_external_job()
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
              download_voice_model: bool = False, title: str | None = None, detail: str | None = None) -> str:
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
        return started_text(job.title, detail or f"{seasons.episodes_word(len(names))}, озвучка: {', '.join(engines)}")


def voice_model_missing() -> bool:
    """The trained voice model of the config can be downloaded and is not on this computer yet."""
    model = str((base_config().get("season_voices") or {}).get("model") or "")
    return model in HUB_WEIGHTS and not model_available(model)


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
        config_path, note = back_to_subscription(old.config_path, stamp)
        job = Job(files=list(old.files), out_root=old.out_root, engines=list(old.engines), config_path=config_path,
                  log_path=old.log_path.with_name(f"job_{stamp}.log"), extra=list(old.extra), title=old.title)
        _launch(job)
        return (started_text(old.title, "тот же запуск: те же серии, движки и настройки; готовые этапы пропускаются, "
                                        "озвученные реплики и ответы языковой модели берутся из сохранённого") + note)


def back_to_subscription(config_path: Path, stamp: str) -> tuple[Path, str]:
    """The settings file for resuming a job. Paying through OpenRouter is allowed for one launch only: a job launched
    with "API OpenRouter" continues through the subscription (a copy of its settings with the subscription's routing
    from the global settings) when the global settings have one. Returns the file and a note for the user."""
    try:
        saved = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return config_path, ""
    llm = saved.get("llm") if isinstance(saved, dict) else None
    base_llm = base_config().get("llm") or {}
    if not isinstance(llm, dict) or llm.get("opencode") or not base_llm.get("opencode"):
        return config_path, ""
    llm.update(opencode=base_llm["opencode"], router=base_llm.get("router", "openrouter"),
               fallback=base_llm.get("fallback", False))
    resumed = config_path.with_name(f"config_{stamp}.json")
    resumed.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumed, (" Оплата через OpenRouter была разрешена только прошлому запуску: продолжение идёт через "
                     "подписку. Платный запуск нужно выбрать заново.")


def strip_view() -> tuple[str, Any]:
    """(text, "Остановить" button) of the status strip; the timer of the page calls this."""
    job = JOB
    return strip_text(job), gr.update(visible=bool(job and job.state == "идёт" and job_alive(job)))


def refresh() -> tuple[str, Any, str, Any, str, str, Any]:
    """Everything the timer updates: status, progress table, readable log, results, checkpoints, raw log, "Продолжить"."""
    adopt_external_job()
    job = JOB
    headers, rows = progress_view(job)
    checkpoints = job_view.checkpoints(_episodes(job), job.engines, job.state == "идёт", job.started) if job else ""
    can_resume = bool(job and job.is_stt_run and job.state in RESUMABLE and not job_alive(job))
    return (status_text(job), gr.update(value=rows, headers=headers), readable_log(job), gr.update(value=results_rows(job)),
            checkpoints, log_tail(job), gr.update(interactive=can_resume))


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
    return app_forms.default_model(base_config() if config is None else config)


def preset_choices() -> list[tuple[str, str]]:
    return app_forms.preset_choices(base_config())


def preset_form(preset: str | None, out_root: str = "") -> list[Any]:
    """What the "Обработка" form shows for a set of settings (on top of the settings in effect for the season):
    engines, mode, subtitles, duck voices, stress, refine, fast ESpeech."""
    values = form_defaults(apply_preset(season_base(out_root), preset))
    return [values[0], values[1], *values[3:]]


def start_with_preset(*values: Any) -> str:
    """``start_job`` from the form; the last values are the voice model tick and the chosen set of settings."""
    return start_job(*values[:-2], download_voice_model=values[-2], preset=values[-1])


def prepare_season(*values: Any) -> str:
    """``start_job`` for ALL videos of the season folder with ``--prepare`` (every episode up to the speakers stage),
    from the same form values as ``start_with_preset``; the trial piece is ignored, and with no engine ticked the first
    one is used (engines do not matter before translation)."""
    (folder, _ticked, out_root, engines, mode, model, subtitles, duck_voices, stress, refine, _preview, preview_from,
     preview_minutes, f5_fast, transport, voice_model, preset) = values
    names = list_videos(folder) if folder else []
    if folder and out_root and not names:
        return f"В папке {folder} нет видео или она недоступна."
    return start_job(folder, names, out_root, engines or list(ENGINES)[:1], mode, model, subtitles, duck_voices, stress,
                     refine, False, preview_from, preview_minutes, f5_fast, transport, stt_args=["--prepare"],
                     download_voice_model=voice_model, preset=preset, title=PREPARE_TITLE,
                     detail=f"{seasons.episodes_word(len(names))}: распознавание, без перевода")


def form_defaults(config: dict[str, Any] | None = None) -> list[Any]:
    return app_forms.form_defaults(base_config() if config is None else config)


def save_settings(folder: str, out_root: str, target: str, *values: Any) -> str:
    return settings_tab.save_settings(folder, out_root, target, *values, base_config=base_config)


def settings_view(folder: str, out_root: str, target: str | None = None) -> list[Any]:
    """Values for settings fields, processing form, mix sliders and season-dependent labels."""
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
    return settings_tab.tts_cache(delete, lock=LOCK, current_job=lambda: JOB)


def mix_form_defaults(out_root: str = "") -> list[Any]:
    return sound.mix_form_defaults(season_base(out_root))


def listen_mix(out_root: str, episode: str | None, engine: str, minute: float, gain: float, rest: float,
               duck_db: float, lufs: float) -> tuple[Any, Any, str]:
    return sound.listen_mix(out_root, episode, engine, minute, gain, rest, duck_db, lufs, season_base=season_base)


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


def season_picker(selected: seasons.Season | None = None) -> Any:
    return season_bar.season_picker(selected, base_config=base_config)


def select_season(source: str | None, previous: str = "") -> tuple[Any, ...]:
    return season_bar.select_season(source, previous, base_config=base_config)


def save_season(source: str, results: str) -> tuple[str, Any, Any]:
    return season_bar.save_season(source, results, base_config=base_config, season_picker=season_picker)


class _VoiceActions:
    """Explicit, late-bound dependencies for the existing season-voices builder."""

    ENGINES = ENGINES
    NO_SEASON = NO_SEASON

    @property
    def JOB(self) -> Job | None:
        return JOB

    def _busy(self) -> str | None:
        return _busy()

    def start_task(self, title: str, cmd: list[str], root: Path, result: str = "") -> str:
        return start_task(title, cmd, root, result)

    def season_base(self, out_root: str) -> dict[str, Any]:
        return season_base(out_root)

    def form_defaults(self, config: dict[str, Any] | None = None) -> list[Any]:
        return form_defaults(config)

    def start_job(self, *args: Any, **kwargs: Any) -> str:
        return start_job(*args, **kwargs)

    def recount_season_voices(self, folder: str, out_root: str, download: bool = False) -> str:
        return recount_season_voices(folder, out_root, download)


def build() -> gr.Blocks:
    personal = base_config().get("ui", {})
    start_values = form_defaults()
    with gr.Blocks(title="Озвучка сериалов") as app:
        gr.Markdown("# Озвучка сериалов\nАнглийское видео → русская закадровая озвучка.")
        bar = season_bar.build(strip_text(JOB), bool(JOB and JOB.state == "идёт"), save_season)
        folder, out_root = bar.folder, bar.out_root
        processing = process.build(
            folder, out_root, bar.strip, bar.stop, personal=personal, start_values=start_values,
            presets=preset_choices(), model_missing=voice_model_missing(), preset_form=preset_form,
            start_with_preset=start_with_preset, prepare_season=prepare_season, stop_job=stop_job,
            resume_job=resume_job, refresh=refresh, strip_view=strip_view, season_table=season_table)
        with gr.Tab("Голоса сезона"):
            voices_tab = voices.build(_VoiceActions(), folder, out_root, processing.voice_model)
        with gr.Tab("Кто это?"):
            who_tab = who.build(folder, out_root)
        edit_tab = edits.build(out_root, revoice, processing.log)
        sound_tab = sound.build(out_root, mix_form_defaults(), find_spot_done=find_spot_done,
                                listen_mix_done=listen_mix_done, remix=remix, start_compress=start_compress)
        config_tab = settings_tab.build(base_config(), cache_done)
        help_tab.build(HELP)
        note_boxes = {"process": processing.note, "voices": voices_tab.note, "edits": edit_tab.note,
                      "sound": sound_tab.note, "settings": config_tab.note}
        view_outputs = [*config_tab.fields, *processing.form, *sound_tab.inputs, config_tab.effect,
                        config_tab.target, config_tab.remove, bar.line, config_tab.global_box,
                        *[note_boxes[tab] for tab in seasons.NOTE_TABS]]
        view_inputs = [folder, out_root, config_tab.target]
        sound_tab.save.click(save_mix_settings, [out_root, *sound_tab.inputs], sound_tab.saved).then(
            settings_view, view_inputs, view_outputs)
        config_tab.save.click(save_settings, [*view_inputs, *config_tab.fields], config_tab.message).then(
            settings_view, view_inputs, view_outputs)
        config_tab.reset.click(reset_settings, view_inputs, config_tab.message).then(
            settings_view, view_inputs, view_outputs)
        config_tab.remove.click(remove_season_settings, [folder, out_root], config_tab.message).then(
            lambda f, r: settings_view(f, r, None), [folder, out_root], view_outputs)
        config_tab.use_global.click(apply_global, config_tab.global_box, config_tab.message).then(
            settings_view, view_inputs, view_outputs)
        config_tab.global_box.submit(apply_global, config_tab.global_box, config_tab.message).then(
            settings_view, view_inputs, view_outputs)
        for event in folder_pick.wire_browser(
                config_tab.browser, {"global": config_tab.global_box},
                {"global": (config_tab.pick_global, folder_pick.FILE, "Файл общих настроек")}, config_tab.pick_message):
            event.then(apply_global, config_tab.global_box, config_tab.message).then(
                settings_view, view_inputs, view_outputs)
        ask_outputs = [bar.ask_panel, bar.ask_text, bar.ask_copy, bar.pending]
        chosen = bar.picker.change(
            select_season, [bar.picker, out_root], [folder, out_root, bar.line, processing.episodes,
                                                  bar.message, processing.season_status, *ask_outputs],
            show_progress="hidden")
        chosen.then(lambda f, r: settings_view(f, r, None), [folder, out_root], view_outputs)
        chosen.then(voices_tab.reload, voices_tab.inputs, voices_tab.outputs)
        chosen.then(who_tab.reload, who_tab.inputs, who_tab.outputs)
        chosen.then(edit_tab.reload, edit_tab.inputs, edit_tab.outputs)
        chosen.then(mix_episodes, out_root, sound_tab.episode)
        chosen.then(finished_videos, out_root, sound_tab.video)
        for button, copy in ((bar.ask_copy, True), (bar.ask_global, False)):
            button.click(lambda pair, copy=copy: answer_settings_question(pair, copy), bar.pending,
                         [bar.ask_panel, bar.message]).then(
                lambda f, r: settings_view(f, r, None), [folder, out_root], view_outputs)
        bar.wire()
        app.load(season_picker, None, bar.picker)
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
