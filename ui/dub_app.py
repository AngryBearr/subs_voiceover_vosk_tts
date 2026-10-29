"""Browser UI for the English video -> Russian voice-over pipeline (``stt.run``).

    subs_env/bin/python -m ui.dub_app [--port 7861]

Tab "Обработка": pick episodes in a folder, engines and options, start, watch the stages of every
episode, find the finished mkv files. A job runs ``stt.run`` in a background process (one job at
a time, engines one after the other); its config and log are kept in ``<out-root>/.ui/``. The
folders default to ``paths`` of the config (work, season files, finished videos).

Tab "Банк голосов": the season voice bank (``stt.voice_bank``): people, episodes, speech; cut
the voice-cloning references, listen, ask for another clip, re-check names across the season.

Tab "Проверка и правка реплик": every Russian line of an episode (filters: the lines ``stt.review``
found suspicious, edited ones, ones too long for their time), the original and the voice-over of
each, edits of text / speaker (with a voice sample) / gender / "not voiced" into ``corrections.json``
(``stt.subtitle_edit``) with a length check, the Subtitle Edit round trip (ASS / SRT out and back
in), and re-voicing only the changed lines.

Tab "Справка": what the program does and how to work with it (``ui/help.md``).

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
HELP = Path(__file__).with_name("help.md")  # the "Справка" tab
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
    """The job's thread: one stt.run for all engines (Vosk next to the GPU engines, ``stt.run``)."""
    with job.log_path.open("ab") as log_file:
        engines = ",".join(job.engines)
        job.engine = engines
        cmd = [sys.executable, "-m", "stt.run", *map(str, job.files), "--out-root", str(job.out_root),
               "--config", str(job.config_path), "--engine", engines, *job.extra]
        log_file.write(f"\n=== {time.strftime('%H:%M:%S')} {engines}: {' '.join(cmd)}\n".encode())
        log_file.flush()
        job.process = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        code = job.process.wait()
        if job.state != "остановлено":
            job.state = "готово" if code == 0 else f"ошибка (код {code}): смотрите лог"
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


def _current_stage(job: Job, stem: str, log_tail: str, engine: str) -> str | None:
    """The stage the log last started for this episode: a shared one, or this engine's."""
    found = [stage for stage, who in re.findall(rf"\[{re.escape(stem)}\] (\w+)(?: \((\w+)\))? \.\.\.", log_tail)
             if who in ("", engine)]
    return found[-1] if found else None


def progress_rows(job: Job | None) -> list[list[str]]:
    if job is None:
        return []
    tail = job.log_path.read_text(encoding="utf-8", errors="replace")[-200_000:] if job.log_path.exists() else ""
    rows = []
    for media in job.files:
        stages_dir = job.out_root / media.stem / ".stages"
        row = [media.stem]
        for engine in job.engines:
            current = _current_stage(job, media.stem, tail, engine) if job.state == "идёт" else None
            done = [s for s in STAGES if (stages_dir / (f"{s}.{engine}" if s in ENGINE_STAGES else s)).exists()]
            if "mix" in done and result_file(job, media, engine):
                cell = "✅ готово"
            elif current:  # a shared stage counts for every engine of the episode
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

        with gr.Tab("Проверка и правка реплик"):
            gr.Markdown("Здесь можно поправить русские реплики серии: текст, говорящего, пол голоса или "
                        "«не озвучивать». Правки копятся в `corrections.json` серии; кнопка «Переозвучить "
                        "изменённое» заново озвучит только изменённые реплики (остальные берутся готовыми), "
                        "серия пересоберётся за минуты. Ударение: `+` перед ударной гласной (`зам+ок`). "
                        "Подробно: вкладка «Справка».")
            with gr.Row():
                edit_root = gr.Textbox(label="Папка результатов", value="output/stt", scale=3)
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

            def pick_line(root: str, episode: str, show: str, query: str, engines: list[str],
                          evt: gr.SelectData) -> tuple[Any, ...]:
                rows = edit_rows(root, episode, show, query)
                line = rows[evt.index[0]][0] if rows and evt.index[0] < len(rows) else None
                if line is None:
                    return (None, "", None, None, "", "", "", None, None, False)
                original, dub, info = line_audio(root, episode, line, [ENGINES[e] for e in engines])
                text, fit_note, choice, gender, skip, hint = line_form(root, episode, line)
                return line, info, original, dub, hint, text, fit_note, choice, gender, skip

            def saved(root: str, episode: str, show: str, query: str, message: str) -> tuple[str, Any]:
                return message, gr.update(value=edit_rows(root, episode, show, query))

            listing = [edit_root, edit_episode, edit_show, edit_query]
            edit_refresh.click(load_episodes, listing, [edit_episode, edit_table, edit_speaker])
            edit_episode.change(load_episodes, listing, [edit_episode, edit_table, edit_speaker])
            edit_show.change(lambda r, e, sh, q: gr.update(value=edit_rows(r, e, sh, q)), listing, edit_table)
            edit_query.submit(lambda r, e, sh, q: gr.update(value=edit_rows(r, e, sh, q)), listing, edit_table)
            edit_table.select(pick_line, [*listing, review_engines],
                              [line_no, line_text, original_audio, dub_audio, line_hint, edit_text, edit_fit,
                               edit_speaker, edit_gender, edit_skip])
            edit_text.change(fit_text, [edit_root, edit_episode, line_no, edit_text], edit_fit)
            edit_speaker.change(speaker_sample, [edit_root, edit_episode, edit_speaker], sample_audio)
            save_button_line.click(
                lambda r, e, sh, q, n, t, who, g, sk: saved(r, e, sh, q, save_line(r, e, n, t, who, g, sk)),
                [*listing, line_no, edit_text, edit_speaker, edit_gender, edit_skip], [edit_message, edit_table])
            undo_button_line.click(lambda r, e, sh, q, n: saved(r, e, sh, q, undo_line(r, e, n)),
                                   [*listing, line_no], [edit_message, edit_table])
            export_button.click(export_subtitles, [edit_root, edit_episode, export_kind, export_scope],
                                [export_out, edit_message])
            import_preview.click(import_rows, [edit_root, edit_episode, import_file], [import_table, edit_message])
            import_button.click(lambda r, e, sh, q, f: saved(r, e, sh, q, import_apply(r, e, f)),
                                [*listing, import_file], [edit_message, edit_table])
            revoice_button.click(revoice, [edit_root, edit_episode, review_engines], edit_message)

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

        with gr.Tab("Справка"):
            gr.Markdown(HELP.read_text(encoding="utf-8") if HELP.exists() else "Справка не найдена.")
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
