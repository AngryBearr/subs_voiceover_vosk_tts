"""Browser UI for the English video -> Russian voice-over pipeline (``stt.run``).

    subs_env/bin/python -m ui.dub_app [--port 7861]

Tab "Обработка": pick episodes in a folder, engines and options, start, watch the stages of every
episode, find the finished mkv files. A job runs ``stt.run`` in a background process (one job at
a time, engines one after the other); its config and log are kept in ``<out-root>/.ui/``. The
folders default to ``paths`` of the config (work, season files, finished videos).

Tab "Банк голосов": the season voice bank (``stt.voice_bank``): people, episodes, speech; cut
the voice-cloning references, listen, ask for another clip, re-check names across the season.

Tab "Проверка реплик": the lines ``stt.review`` found suspicious (gender / speaker), the original
and the voice-over of each, one-click fixes into ``corrections.json`` and re-voicing the episode.

Tab "Настройки": the defaults people change (folders, recognition, models, voices, checks, mix),
saved into ``stt_config.local.json`` (``ui.settings``); the "Обработка" form starts from them.
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

from stt.run import STAGES, SUBSCRIPTION_DOWN, finished_folder, load_config
from ui import settings

REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXT = {".mkv", ".mp4", ".avi", ".mov", ".m4v", ".webm"}
ENGINES, MODES, MODELS = settings.ENGINES, settings.MODES, settings.MODELS
TRANSPORTS = {"Подписка (opencode)": "opencode", "API OpenRouter": "openrouter"}
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


def base_config() -> dict[str, Any]:
    """stt_config.json with the personal stt_config.local.json on top."""
    return load_config(REPO_ROOT / "stt_config.json")


def job_config(mode: str, model: str, subtitles: bool, duck_voices: bool, stress: bool, refine: bool,
               f5_fast: bool = False, transport: str | None = None) -> dict[str, Any]:
    """stt_config.json (+ local settings) with the options of the form."""
    config = base_config()
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
              preview_from: float, preview_minutes: float, f5_fast: bool = False, transport: str | None = None) -> str:
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
        config = job_config(mode, model, subtitles, duck_voices, stress, refine, f5_fast, transport)
        if preview:  # a short piece in its own folder, so it never mixes with full episodes
            root = root / "_проба"
            extra = ["--start", str(preview_from * 60), "--duration", str(preview_minutes * 60)]
            config["paths"] = {**(config.get("paths") or {}), "season": None, "finished": None}
        ui_dir = root / ".ui"
        ui_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        config_path = ui_dir / f"config_{stamp}.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
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
    """The finished video with the engine's voice-over (or, without a video, its mixed audio)."""
    work = job.out_root / media.stem
    config = json.loads(job.config_path.read_text(encoding="utf-8")) if job.config_path.exists() else {}
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
    return [[media.stem, engine, str(path)] for media in job.files for engine in job.engines
            if (path := result_file(job, media, engine))]


def log_tail(job: Job | None, lines: int = 40) -> str:
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


def refresh() -> tuple[str, Any, str, Any]:
    job = JOB
    if job is None:
        status = "Нет запущенной обработки."
    else:
        minutes = ((job.finished or time.time()) - job.started) / 60
        status = f"**{job.state}** · {minutes:.0f} мин · движок: {job.engine or '-'}"
        if subscription_failed(job):
            status += "\n\n" + SUBSCRIPTION_WARNING
    engines = job.engines if job else []
    progress = gr.update(value=progress_rows(job), headers=["Серия", *engines] if engines else ["Серия"])
    return status, progress, log_tail(job), gr.update(value=results_rows(job))


# ---------------------------------------------------------------- voice bank tab

def _root(out_root: str) -> Path:
    root = Path(out_root).expanduser()
    return root if root.is_absolute() else REPO_ROOT / root


def _season_root(out_root: str) -> Path:
    """Folder of the season files (voice banks, voice clips): ``paths.season``, else the work folder."""
    season = (base_config().get("paths") or {}).get("season")
    return _root(season) if season else _root(out_root)


def list_banks(out_root: str) -> list[str]:
    return sorted(p.name for p in _season_root(out_root).glob("*.voices.json"))


def refs_dir(out_root: str, bank: str) -> Path:
    return _season_root(out_root) / bank.replace(".voices.json", ".references")


def _bank_cmd(out_root: str, bank: str, *args: str) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "stt.voice_bank", str(_season_root(out_root) / bank), *args]
    done = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    lines = [line for line in (done.stdout + done.stderr).splitlines()
             if not any(k in line.lower() for k in ("warn", "tf32", ">>>", "issues/"))]
    return done.returncode, "\n".join(lines[-40:])


def people_rows(out_root: str, bank: str) -> list[list[Any]]:
    if not bank:
        return []
    from stt.voice_bank import VoiceBank

    vb = VoiceBank.load(_season_root(out_root) / bank)
    index_path = refs_dir(out_root, bank) / "references.json"
    refs = {r["name"]: r for r in json.loads(index_path.read_text(encoding="utf-8"))} if index_path.exists() else {}
    rows = []
    for person in vb.people():
        ref = refs.get(person["name"])
        rows.append([person["name"], person.get("label_ru") or "", person.get("gender") or "",
                     len(person["episodes"]), round(person["speech_s"] / 60, 1),
                     f"{ref['score']:.2f}" if ref else "нет"])
    return rows


def person_view(out_root: str, bank: str, name: str) -> tuple[Any, str]:
    index_path = refs_dir(out_root, bank) / "references.json"
    if not name or not index_path.exists():
        return None, "Образца пока нет: нажмите «Собрать образцы»."
    ref = next((r for r in json.loads(index_path.read_text(encoding="utf-8")) if r["name"] == name), None)
    if ref is None:
        return None, "Для этого человека чистый образец не нашёлся."
    episode = re.search(r"S\d+E\d+", ref["episode"])
    info = (f"**{ref['name']}** · {episode.group(0) if episode else ref['episode']} "
            f"{ref['start']:.1f}–{ref['end']:.1f} с · похожесть {ref['score']:.2f} · "
            f"музыка {ref['music_db']:+.0f} дБ до очистки\n\n> {ref['text']}")
    return str(refs_dir(out_root, bank) / ref["wav"]), info


# ---------------------------------------------------------------- review tab

def list_reviewed(out_root: str) -> list[str]:
    return sorted(p.parent.name for p in _root(out_root).glob("*/review.json"))


def _episode(out_root: str, episode: str) -> Path:
    return _root(out_root) / episode


def review_rows(out_root: str, episode: str) -> list[list[Any]]:
    if not episode:
        return []
    folder = _episode(out_root, episode)
    items = json.loads((folder / "review.json").read_text(encoding="utf-8"))
    path = folder / "corrections.json"
    fixed = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    rows = []
    for item in items:
        what = item.get("heard") and f"слышно {item['heard']}" or ""
        if item.get("sounds_like"):
            what = (what + "; " if what else "") + f"похоже на {item['sounds_like']}"
        fix = fixed.get(str(item["line"]))
        rows.append([item["line"], item["time"][:8], item["speaker"], item["voice"], what, item["text_ru"],
                     json.dumps(fix, ensure_ascii=False) if fix else ""])
    return rows


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
    for engine in engines or ["vosk", "teratts"]:
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


def add_correction(out_root: str, episode: str, line: int | None, fix: dict[str, Any]) -> str:
    if not episode or line is None:
        return "Выберите серию и реплику."
    path = _episode(out_root, episode) / "corrections.json"
    corrections = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    corrections[str(int(line))] = {**corrections.get(str(int(line)), {}), **fix}
    path.write_text(json.dumps(corrections, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return f"Реплика {int(line)}: {json.dumps(fix, ensure_ascii=False)} → {path.name} " \
           f"({len(corrections)} правок). Нажмите «Переозвучить серию», когда закончите."


def speaker_choices(out_root: str, episode: str) -> list[str]:
    if not episode:
        return []
    speakers = json.loads((_episode(out_root, episode) / "speakers.json").read_text(encoding="utf-8"))
    return [f"{s['id']}: {s.get('name') or s.get('label_ru') or '?'} ({s.get('gender')})"
            for s in speakers if s.get("cues")]


def revoice(out_root: str, episode: str, engines: list[str]) -> str:
    if not episode:
        return "Выберите серию."
    folder = _episode(out_root, episode)
    media = Path(json.loads((folder / "source.json").read_text(encoding="utf-8"))["media"])
    done = [e for e in ENGINES if (folder / f"voiceover.{ENGINES[e]}").exists()]
    return start_job(str(media.parent), [media.name], str(_root(out_root)), engines or done or ["Vosk"],
                     list(MODES)[0], default_model(), True, True, True, True, False, 0, 0)


def default_model() -> str:
    model = base_config().get("llm", {}).get("model")
    return next((label for label, value in MODELS.items() if value == model), list(MODELS)[0])


def form_defaults() -> list[Any]:
    """Starting values of the "Обработка" form from the settings: engines, mode, model, out root and the
    subtitles / duck voices / stress / refine / fast ESpeech boxes."""
    config = base_config()
    chosen = (config.get("ui") or {}).get("engines") or ["vosk"]
    f5 = ((config.get("voiceover") or {}).get("engines") or {}).get("f5") or {}
    return [[label for label, value in ENGINES.items() if value in chosen],
            next((label for label, value in MODES.items() if value == config.get("mode")), list(MODES)[0]),
            default_model(), (config.get("paths") or {}).get("work") or "output/stt",
            bool((config.get("mix") or {}).get("subtitles", True)), (config.get("mix") or {}).get("duck") == "voices",
            bool((config.get("voiceover") or {}).get("stress_check", True)),
            (config.get("speaker_refine") or {}).get("model", "resnet293") != "none", int(f5.get("nfe", 32)) == 7]


def save_settings(*values: Any) -> list[Any]:
    try:
        own, warnings = settings.save(list(values))
    except settings.SettingsError as exc:
        return ["Не сохранено:\n\n" + "\n".join(f"- {p}" for p in exc.problems), *form_defaults()]
    text = f"Сохранено (своих значений: {own}). Действует со следующего запуска."
    return [text + "".join(f"\n- {w}" for w in warnings), *form_defaults()]


def reset_settings() -> list[Any]:
    settings.reset()
    return ["Настройки сброшены к стандартным (личные ключи вроде подписки остались).",
            *settings.form_values(base_config()), *form_defaults()]


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


def _setting_field(setting: settings.Setting, value: Any) -> Any:
    if setting.kind == "bool":
        return gr.Checkbox(label=setting.label, value=value, info=setting.info or None)
    if setting.kind == "choice":
        return gr.Dropdown(label=setting.label, choices=list(setting.choices), value=value, info=setting.info or None)
    if setting.kind == "engines":
        return gr.CheckboxGroup(label=setting.label, choices=list(setting.choices), value=value)
    if setting.kind in ("number", "integer"):
        return gr.Number(label=setting.label, value=value, precision=0 if setting.kind == "integer" else None,
                         info=setting.info or None)
    if setting.kind == "voice":
        voices = settings.TERATTS_VOICES + sorted(settings.teratts_styles() - set(settings.TERATTS_VOICES))
        return gr.Dropdown(label=setting.label, choices=voices, value=value, allow_custom_value=True,
                           info=setting.info or None)
    return gr.Textbox(label=setting.label, value=value, info=setting.info or None)


def build() -> gr.Blocks:
    personal = base_config().get("ui", {})
    start_values = form_defaults()
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
            out_root = gr.Textbox(label="Рабочая папка (готовые видео: вкладка «Настройки»)", value=start_values[3])
            with gr.Row():
                engines = gr.CheckboxGroup(label="Озвучка", choices=list(ENGINES), value=start_values[0])
                mode = gr.Radio(label="Распознавание", choices=list(MODES), value=start_values[1])
                with gr.Column():
                    model = gr.Dropdown(label="Языковая модель", choices=list(MODELS), value=start_values[2])
                    transport = gr.Radio(label="Запросы к модели", choices=list(TRANSPORTS),
                                         value=list(TRANSPORTS)[0], visible=bool(personal.get("llm_transport_choice")))
            with gr.Row():
                subtitles = gr.Checkbox(label="Вшить субтитры", value=start_values[4])
                duck_voices = gr.Checkbox(label="Приглушать только голоса (музыку оставить)", value=start_values[5])
                stress = gr.Checkbox(label="Проверять ударения", value=start_values[6])
                refine = gr.Checkbox(label="Уточнять спикеров по голосу", value=start_values[7])
                f5_fast = gr.Checkbox(label="ESpeech: быстрый режим (7 шагов)", value=start_values[8])
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
                                    refine, preview, preview_from, preview_minutes, f5_fast, transport], message)
            stop.click(stop_job, None, message)
            gr.Timer(2.0).tick(refresh, None, [status, progress, log, results])

        with gr.Tab("Банк голосов"):
            gr.Markdown("Банк собирается сам при обработке серий: голоса названных героев сезона. "
                        "Здесь можно вырезать образцы голосов для будущей озвучки голосами героев, "
                        "послушать их и заменить неудачные.")
            with gr.Row():
                bank_root = gr.Textbox(label="Папка результатов", value="output/stt", scale=3)
                bank = gr.Dropdown(label="Сезон", choices=[], scale=3)
                bank_refresh = gr.Button("Обновить", scale=1)
            people = gr.Dataframe(headers=["Имя", "По-русски", "Пол", "Серий", "Речи, мин", "Образец"],
                                  label="Люди сезона", interactive=False)
            with gr.Row():
                export_all = gr.Button("Собрать образцы (все)", variant="primary")
                recheck = gr.Button("Перепроверить имена по сезону")
                write_fixes = gr.Button("Записать исправления имён")
            bank_message = gr.Textbox(label="Результат", lines=6)
            person = gr.Dropdown(label="Человек", choices=[])
            with gr.Row():
                reference = gr.Audio(label="Образец голоса", type="filepath", interactive=False)
                reference_info = gr.Markdown()
            another = gr.Button("Другой образец для этого человека")

            def load_bank(root: str, name: str | None) -> tuple[Any, Any, Any]:
                banks = list_banks(root)
                name = name if name in banks else (banks[0] if banks else None)
                rows = people_rows(root, name) if name else []
                return (gr.update(choices=banks, value=name), gr.update(value=rows),
                        gr.update(choices=[r[0] for r in rows], value=None))

            def run_export(root: str, name: str) -> tuple[str, Any]:
                if not name:
                    return "Выберите сезон.", gr.update()
                code, out = _bank_cmd(root, name, "--export", str(refs_dir(root, name)))
                return (out if code == 0 else f"Ошибка (код {code}):\n{out}"), gr.update(value=people_rows(root, name))

            def run_another(root: str, name: str, who: str) -> tuple[Any, str, Any]:
                if not (name and who):
                    return None, "Выберите сезон и человека.", gr.update()
                code, out = _bank_cmd(root, name, "--export", str(refs_dir(root, name)), "--person", who, "--reject")
                audio, info = person_view(root, name, who)
                return audio, info if code == 0 else f"Ошибка (код {code}):\n{out}", gr.update(
                    value=people_rows(root, name))

            bank_refresh.click(load_bank, [bank_root, bank], [bank, people, person])
            bank.change(load_bank, [bank_root, bank], [bank, people, person])
            export_all.click(run_export, [bank_root, bank], [bank_message, people])
            recheck.click(lambda root, name: _bank_cmd(root, name, "--recheck")[1] if name else "Выберите сезон.",
                          [bank_root, bank], bank_message)
            write_fixes.click(lambda root, name: _bank_cmd(root, name, "--recheck", "--write-overrides")[1]
                              if name else "Выберите сезон.", [bank_root, bank], bank_message)
            person.change(lambda root, name, who: person_view(root, name, who), [bank_root, bank, person],
                          [reference, reference_info])
            another.click(run_another, [bank_root, bank, person], [reference, reference_info, people])

        with gr.Tab("Проверка реплик"):
            gr.Markdown("Реплики, где голос мог быть выбран неверно: по звуку реплики слышен другой пол или "
                        "другой спикер. Послушайте оригинал и озвучку, при необходимости поправьте и "
                        "переозвучьте серию: пересчитаются только сценарий и озвучка.")
            with gr.Row():
                review_root = gr.Textbox(label="Папка результатов", value="output/stt", scale=3)
                review_episode = gr.Dropdown(label="Серия", choices=[], scale=4)
                review_refresh = gr.Button("Обновить", scale=1)
            review_table = gr.Dataframe(
                headers=["№", "Время", "Спикер", "Голос", "Что слышно", "Текст", "Правка"],
                label="Реплики для проверки (нажмите на строку)", interactive=False)
            line_no = gr.Number(label="Реплика №", precision=0)
            line_text = gr.Markdown()
            with gr.Row():
                original_audio = gr.Audio(label="Оригинал", interactive=False)
                dub_audio = gr.Audio(label="Озвучка", interactive=False)
            with gr.Row():
                fix_female = gr.Button("Женский голос")
                fix_male = gr.Button("Мужской голос")
                fix_skip = gr.Button("Не озвучивать")
            with gr.Row():
                fix_speaker = gr.Dropdown(label="Другой спикер", choices=[], scale=3)
                apply_speaker = gr.Button("Назначить спикера", scale=1)
            with gr.Row():
                fix_text = gr.Textbox(label="Исправленный текст", scale=3)
                apply_text = gr.Button("Заменить текст", scale=1)
            review_engines = gr.CheckboxGroup(label="Переозвучить движками", choices=list(ENGINES), value=[])
            revoice_button = gr.Button("Переозвучить серию", variant="primary")
            review_message = gr.Markdown()

            def load_review(root: str, episode: str | None) -> tuple[Any, Any, Any]:
                episodes = list_reviewed(root)
                episode = episode if episode in episodes else (episodes[0] if episodes else None)
                return (gr.update(choices=episodes, value=episode), gr.update(value=review_rows(root, episode)),
                        gr.update(choices=speaker_choices(root, episode), value=None))

            def pick_line(root: str, episode: str, engines: list[str], evt: gr.SelectData) -> tuple[Any, ...]:
                rows = review_rows(root, episode)
                line = rows[evt.index[0]][0] if rows else None
                if line is None:
                    return None, "", None, None
                original, dub, text = line_audio(root, episode, line, [ENGINES[e] for e in engines])
                return line, text, original, dub

            def fixed(root: str, episode: str, line: Any, fix: dict[str, Any]) -> tuple[str, Any]:
                message = add_correction(root, episode, line, fix)
                return message, gr.update(value=review_rows(root, episode))

            review_refresh.click(load_review, [review_root, review_episode], [review_episode, review_table, fix_speaker])
            review_episode.change(load_review, [review_root, review_episode],
                                  [review_episode, review_table, fix_speaker])
            review_table.select(pick_line, [review_root, review_episode, review_engines],
                                [line_no, line_text, original_audio, dub_audio])
            fix_female.click(lambda r, e, n: fixed(r, e, n, {"gender": "female"}), [review_root, review_episode, line_no],
                             [review_message, review_table])
            fix_male.click(lambda r, e, n: fixed(r, e, n, {"gender": "male"}), [review_root, review_episode, line_no],
                           [review_message, review_table])
            fix_skip.click(lambda r, e, n: fixed(r, e, n, {"skip": True}), [review_root, review_episode, line_no],
                           [review_message, review_table])
            apply_speaker.click(lambda r, e, n, who: fixed(r, e, n, {"speaker": who.split(":")[0]}) if who
                                else ("Выберите спикера.", gr.update()),
                                [review_root, review_episode, line_no, fix_speaker], [review_message, review_table])
            apply_text.click(lambda r, e, n, t: fixed(r, e, n, {"text": t}) if t.strip()
                             else ("Введите текст.", gr.update()),
                             [review_root, review_episode, line_no, fix_text], [review_message, review_table])
            revoice_button.click(revoice, [review_root, review_episode, review_engines], review_message)

        with gr.Tab("Настройки"):
            gr.Markdown("Значения по умолчанию для новых запусков. Сохраняются в `stt_config.local.json` "
                        "(только отличия от `stt_config.json`; прежняя версия файла: `.bak`).")
            fields = []
            values = settings.form_values(base_config())
            for group in dict.fromkeys(s.group for s in settings.SETTINGS):
                gr.Markdown(f"### {group}")
                with gr.Row(equal_height=False):
                    for setting, value in zip(settings.SETTINGS, values):
                        if setting.group == group:
                            fields.append(_setting_field(setting, value))
            with gr.Row():
                save_button = gr.Button("Сохранить", variant="primary")
                reset_button = gr.Button("Сбросить к стандартным")
            settings_message = gr.Markdown()
            gr.Markdown("### Кэш озвучки\nКаждая озвученная реплика хранится в `output/tts_cache`: повторный "
                        "запуск берёт её оттуда. Реплики, которые не нужны ни одной серии, можно удалить.")
            with gr.Row():
                cache_check = gr.Button("Сколько освободится")
                cache_clean = gr.Button("Очистить кэш озвучки", variant="stop")
            cache_message = gr.Markdown()
            cache_check.click(lambda: tts_cache(False), None, cache_message)
            cache_clean.click(lambda: tts_cache(True), None, cache_message)
            form = [engines, mode, model, out_root, subtitles, duck_voices, stress, refine, f5_fast]
            save_button.click(save_settings, fields, [settings_message, *form])
            reset_button.click(reset_settings, None, [settings_message, *fields, *form])
        app.load(lambda: [*settings.form_values(base_config()), *form_defaults()], None, [*fields, *form])
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Browser UI for the voice-over pipeline")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    # clips and videos may live outside the repo (an absolute output folder, the media drive)
    build().launch(server_name=args.host, server_port=args.port, allowed_paths=[str(REPO_ROOT / "output"), "/media"])


if __name__ == "__main__":
    main()
