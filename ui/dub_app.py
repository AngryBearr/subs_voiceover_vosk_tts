"""Browser UI for the English video -> Russian voice-over pipeline (``stt.run``).

    subs_env/bin/python -m ui.dub_app [--port 7861]

Tab "Обработка": pick episodes in a folder, engines and options, start, watch the stages of every
episode, find the finished mkv files. A job runs ``stt.run`` in a background process (one job at
a time, engines one after the other); its config and log are kept in ``<out-root>/.ui/``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore

from stt.run import STAGES

REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXT = {".mkv", ".mp4", ".avi", ".mov", ".m4v", ".webm"}
ENGINES = {"Vosk": "vosk", "TeraTTS": "teratts"}
MODES = {"Точный: Cohere + Whisper + LLM": "accurate", "Быстрый: только Whisper": "fast"}
MODELS = ["openai/gpt-6.1-sol", "openai/gpt-6-luna"]
ENGINE_STAGES = {"dub", "voiceover", "check", "mix"}
STAGE_RU = {"transcribe": "распознавание", "glossary": "словарь", "speakers": "спикеры", "translate": "перевод",
            "dub": "сценарий", "stress": "ударения", "voiceover": "озвучка", "check": "проверка речи",
            "mix": "сведение"}


@dataclass
class Job:
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


JOB: Job | None = None
LOCK = threading.Lock()


def list_videos(folder: str) -> list[str]:
    path = Path(folder).expanduser()
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXT)


def job_config(mode: str, model: str, subtitles: bool, duck_voices: bool, stress: bool, refine: bool
               ) -> dict[str, Any]:
    """stt_config.json with the options of the form."""
    config = json.loads((REPO_ROOT / "stt_config.json").read_text(encoding="utf-8"))
    config["mode"] = MODES.get(mode, mode)
    config.setdefault("llm", {})["model"] = model
    config.setdefault("mix", {}).update(subtitles=subtitles, duck="voices" if duck_voices else "all")
    config.setdefault("voiceover", {})["stress_check"] = stress
    config["speaker_refine"] = {**(config.get("speaker_refine") or {}), "model": "resnet293" if refine else "none"}
    return config


def _run(job: Job) -> None:
    """The job's thread: stt.run once per engine, logging into one file."""
    with job.log_path.open("ab") as log_file:
        for engine in job.engines:
            job.engine = engine
            cmd = [sys.executable, "-m", "stt.run", *map(str, job.files), "--out-root", str(job.out_root),
                   "--config", str(job.config_path), "--engine", engine, *job.extra]
            log_file.write(f"\n=== {time.strftime('%H:%M:%S')} {engine}: {' '.join(cmd)}\n".encode())
            log_file.flush()
            job.process = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            code = job.process.wait()
            if job.state == "остановлено":
                break
            if code != 0:
                job.state = f"ошибка ({engine}, код {code}): смотрите лог"
                break
        else:
            job.state = "готово"
    job.finished = time.time()


def start_job(folder: str, names: list[str], out_root: str, engines: list[str], mode: str, model: str,
              subtitles: bool, duck_voices: bool, stress: bool, refine: bool, preview: bool,
              preview_from: float, preview_minutes: float) -> str:
    global JOB
    with LOCK:
        if JOB and JOB.state == "идёт":
            return "Уже идёт обработка: дождитесь её или остановите."
        if not names:
            return "Выберите хотя бы одну серию."
        if not engines:
            return "Выберите хотя бы один движок озвучки."
        root = Path(out_root).expanduser()
        if not root.is_absolute():
            root = REPO_ROOT / root
        extra: list[str] = []
        if preview:  # a short piece in its own folder, so it never mixes with full episodes
            root = root / "_проба"
            extra = ["--start", str(preview_from * 60), "--duration", str(preview_minutes * 60)]
        ui_dir = root / ".ui"
        ui_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        config_path = ui_dir / f"config_{stamp}.json"
        config_path.write_text(json.dumps(job_config(mode, model, subtitles, duck_voices, stress, refine),
                                          ensure_ascii=False, indent=2), encoding="utf-8")
        JOB = Job(files=[Path(folder).expanduser() / n for n in names], out_root=root,
                  engines=[ENGINES[e] for e in engines], config_path=config_path,
                  log_path=ui_dir / f"job_{stamp}.log", extra=extra)
        threading.Thread(target=_run, args=(JOB,), daemon=True).start()
        return f"Запущено: {len(names)} сер., движки: {', '.join(engines)}. Лог: {JOB.log_path}"


def stop_job() -> str:
    with LOCK:
        if not JOB or JOB.state != "идёт" or JOB.process is None:
            return "Нечего останавливать."
        JOB.state = "остановлено"
        try:
            os.killpg(JOB.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        return "Остановлено. Готовые этапы сохранены: следующий запуск продолжит с места остановки."


def _current_stage(job: Job, stem: str, log_tail: str) -> str | None:
    found = re.findall(rf"\[{re.escape(stem)}\] (\w+) \.\.\.", log_tail)
    return found[-1] if found else None


def progress_rows(job: Job | None) -> list[list[str]]:
    if job is None:
        return []
    tail = job.log_path.read_text(encoding="utf-8", errors="replace")[-200_000:] if job.log_path.exists() else ""
    rows = []
    for media in job.files:
        stages_dir = job.out_root / media.stem / ".stages"
        row = [media.stem]
        current = _current_stage(job, media.stem, tail) if job.state == "идёт" else None
        for engine in job.engines:
            done = [s for s in STAGES if (stages_dir / (f"{s}.{engine}" if s in ENGINE_STAGES else s)).exists()]
            if "mix" in done and result_file(job, media, engine):
                cell = "✅ готово"
            elif current and engine == job.engine:
                cell = f"⏳ {len(done)}/{len(STAGES)}: {STAGE_RU.get(current, current)}"
            else:
                cell = f"{len(done)}/{len(STAGES)}"
            row.append(cell)
        rows.append(row)
    return rows


def result_file(job: Job, media: Path, engine: str) -> Path | None:
    """The finished video (or, for a test piece, the mixed audio) of an episode and engine."""
    for suffix in ("mkv", "wav"):
        path = job.out_root / media.stem / f"{media.stem}.ru.{engine}.{suffix}"
        if path.exists():
            return path
    return None


def results_rows(job: Job | None) -> list[list[str]]:
    if job is None:
        return []
    return [[media.stem, engine, str(path)] for media in job.files for engine in job.engines
            if (path := result_file(job, media, engine))]


def log_tail(job: Job | None, lines: int = 40) -> str:
    if job is None or not job.log_path.exists():
        return ""
    text = job.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    keep = [line for line in text if " INFO " in line or "ERROR" in line or "Traceback" in line or line.startswith("===")
            or "Error" in line]
    return "\n".join((keep or text)[-lines:])


def refresh() -> tuple[str, Any, str, Any]:
    job = JOB
    if job is None:
        status = "Нет запущенной обработки."
    else:
        minutes = ((job.finished or time.time()) - job.started) / 60
        status = f"**{job.state}** · {minutes:.0f} мин · движок: {job.engine or '-'}"
    engines = job.engines if job else []
    progress = gr.update(value=progress_rows(job), headers=["Серия", *engines] if engines else ["Серия"])
    return status, progress, log_tail(job), gr.update(value=results_rows(job))


def build() -> gr.Blocks:
    with gr.Blocks(title="Озвучка сериалов") as app:
        gr.Markdown("# Озвучка сериалов\nАнглийское видео → русская закадровая озвучка.")
        with gr.Tab("Обработка"):
            with gr.Row():
                folder = gr.Textbox(label="Папка с сериями", value="/media/angrybear/large_storage/Survivor",
                                    scale=4)
                show = gr.Button("Показать серии", scale=1)
            episodes = gr.CheckboxGroup(label="Серии", choices=[])
            with gr.Row():
                select_all = gr.Button("Выбрать все", size="sm")
                select_none = gr.Button("Снять выбор", size="sm")
            out_root = gr.Textbox(label="Куда сохранять", value="output/stt")
            with gr.Row():
                engines = gr.CheckboxGroup(label="Озвучка", choices=list(ENGINES), value=["Vosk"])
                mode = gr.Radio(label="Распознавание", choices=list(MODES), value=list(MODES)[0])
                model = gr.Dropdown(label="Языковая модель (через подписку)", choices=MODELS, value=MODELS[0])
            with gr.Row():
                subtitles = gr.Checkbox(label="Вшить субтитры", value=True)
                duck_voices = gr.Checkbox(label="Приглушать только голоса (музыку оставить)", value=True)
                stress = gr.Checkbox(label="Проверять ударения", value=True)
                refine = gr.Checkbox(label="Уточнять спикеров по голосу", value=True)
            with gr.Accordion("Пробный отрывок (быстро проверить настройки)", open=False):
                preview = gr.Checkbox(label="Только отрывок (в папку _проба)", value=False)
                with gr.Row():
                    preview_from = gr.Number(label="С минуты", value=10)
                    preview_minutes = gr.Number(label="Минут", value=2)
            with gr.Row():
                start = gr.Button("Начать", variant="primary")
                stop = gr.Button("Остановить", variant="stop")
            message = gr.Markdown()
            status = gr.Markdown("Нет запущенной обработки.")
            progress = gr.Dataframe(headers=["Серия"], label="Ход обработки (этапов из 9)", interactive=False)
            results = gr.Dataframe(headers=["Серия", "Движок", "Файл"], label="Готовые файлы", interactive=False)
            log = gr.Textbox(label="Лог", lines=12, max_lines=12, autoscroll=True)

            show.click(lambda f: gr.update(choices=list_videos(f), value=[]), folder, episodes)
            select_all.click(lambda f: gr.update(value=list_videos(f)), folder, episodes)
            select_none.click(lambda: gr.update(value=[]), None, episodes)
            start.click(start_job, [folder, episodes, out_root, engines, mode, model, subtitles, duck_voices, stress,
                                    refine, preview, preview_from, preview_minutes], message)
            stop.click(stop_job, None, message)
            gr.Timer(2.0).tick(refresh, None, [status, progress, log, results])
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Browser UI for the voice-over pipeline")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    build().launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
