"""Seasons of the UI: a season is the folder with the original episodes and the results folder.

The results folder (suggested next to the source as ``STT_<season>``) holds everything: the work folders of
the episodes, the season files (voice bank, glossary, ...), the finished ``*.ru.mkv`` at its top level, the
``.ui/`` job files and the ``_проба/`` previews. Seasons are remembered in the UI state file ``ui.local.json``
(``ui.state``). ``season_config`` turns the config into the one of a job, so ``stt.run`` needs no change.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stt import progress
from stt.run import STAGES
from ui import job_view, settings
from ui import state as ui_state

REPO_ROOT = settings.REPO_ROOT
VIDEO_EXT = {".mkv", ".mp4", ".avi", ".mov", ".m4v", ".webm"}
RESULTS_PREFIX = "STT_"
MEDIA_GLOB = "/media/*/*/Survivor"  # /media/<user>/<drive>/Survivor
ENGINE_NAMES = {"vosk", "teratts", "f5"}  # the middle of ``<stem>.ru.<engine>.mkv``


@dataclass(frozen=True)
class Season:
    source: Path  # the original episodes
    results: Path  # work folders, season files, finished videos

    @property
    def name(self) -> str:
        return self.source.name


@dataclass(frozen=True)
class EpisodeStatus:
    video: str  # file name in the source folder
    stem: str
    state: str  # words for the table
    finished: Path | None = None  # the ``.ru.mkv`` when ready


def plural(n: int, one: str, few: str, many: str) -> str:
    """"1 серия", "2 серии", "5 серий"."""
    tail = n % 100
    if 11 <= tail <= 14:
        word = many
    else:
        word = one if n % 10 == 1 else few if 2 <= n % 10 <= 4 else many
    return f"{n} {word}"


def episodes_word(n: int) -> str:
    return plural(n, "серия", "серии", "серий")


def absolute(text: str | Path) -> Path:
    """A folder typed by a person: ``~`` expanded, relative to the repository when not absolute."""
    path = Path(str(text).strip().strip("'\"")).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def video_names(folder: Path) -> list[str]:
    """Video files directly in ``folder`` (sorted); empty when it is not a folder."""
    if not folder.is_dir():
        return []
    try:
        return sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXT)
    except OSError:
        return []


_CODE = re.compile(r"S(\d{1,2})E(\d{1,3})", re.IGNORECASE)
_TECH = re.compile(r"^(\d{3,4}p|web|webrip|web-dl|amzn|hdtv|bluray|x26[45]|h\.?26[45]|ddp?\d|aac|rawr)$", re.IGNORECASE)


def episode_label(name: str) -> str:
    """A short readable name of an episode file: "S48E01 · The Get to Know You Game" (the file name when there is no
    SxxEyy code)."""
    stem = Path(name).stem
    code = _CODE.search(stem)
    if not code:
        return stem
    title = ""
    bracket = re.search(r"\(([^()]+)\)\s*$", stem)
    if bracket:
        title = bracket.group(1).strip()
    else:
        words = []
        for word in re.split(r"[._ ]+", stem[code.end():]):
            if not word:
                continue
            if _TECH.match(word) or ("-" in word and _TECH.match(word.split("-")[0])):
                break
            words.append(word)
        title = " ".join(words)
    short = f"S{int(code.group(1)):02d}E{int(code.group(2)):02d}"
    return f"{short} · {title}" if title else short


def episode_choices(names: list[str]) -> list[tuple[str, str]]:
    """(label, file name) pairs for an episode list: short labels, the file name stays the value."""
    return [(episode_label(name), name) for name in names]


def suggest_results(source: str | Path) -> Path:
    """``<source parent>/STT_<source folder name>``."""
    source = absolute(source)
    return source.parent / f"{RESULTS_PREFIX}{source.name}"


# ---------------------------------------------------------------- remembered seasons

def _entries(state: dict[str, Any]) -> list[Season]:
    found = []
    for item in state.get("seasons") or []:
        if isinstance(item, dict) and item.get("source") and item.get("results"):
            found.append(Season(absolute(item["source"]), absolute(item["results"])))
    return found


def remembered(state_path: Path | None = None) -> list[Season]:
    """The seasons saved in the UI state file, in the order they were added."""
    return _entries(ui_state.read(state_path))


def current_source(state_path: Path | None = None) -> str | None:
    value = ui_state.read(state_path).get("current_season")
    return str(value) if value else None


def discover(config: dict[str, Any], media_glob: str | None = None) -> list[Season]:
    """Pairs ``<dir>/X`` + ``<dir>/STT_X`` where X has videos, one level deep, in the parent of ``paths.finished``
    of the config and in the ``/media/*/*/Survivor`` folders."""
    parents: list[Path] = []
    finished = (config.get("paths") or {}).get("finished")
    if finished:
        parents.append(absolute(finished).parent)
    parents += sorted(Path("/").glob((media_glob or MEDIA_GLOB).lstrip("/")))
    found: dict[Path, Season] = {}
    for parent in parents:
        try:
            children = sorted(p for p in parent.iterdir() if p.is_dir())
        except OSError:
            continue
        for results in children:
            if not results.name.startswith(RESULTS_PREFIX):
                continue
            source = parent / results.name[len(RESULTS_PREFIX):]
            if source.is_dir() and video_names(source) and source not in found:
                found[source] = Season(source, results)
    return list(found.values())


def available(config: dict[str, Any], state_path: Path | None = None) -> list[Season]:
    """The seasons of the picker: the remembered ones, or (nothing remembered yet) the ones found next to the
    folders of the config."""
    return remembered(state_path) or discover(config)


def pick_current(seasons: list[Season], state_path: Path | None = None) -> Season | None:
    """The saved current season if it is still in the list, else the first one."""
    wanted = current_source(state_path)
    return next((s for s in seasons if wanted and str(s.source) == wanted), seasons[0] if seasons else None)


def remember(season: Season, config: dict[str, Any], state_path: Path | None = None) -> None:
    """Make ``season`` the current one and remember it. Nothing remembered yet: the seasons the picker offered
    are remembered too. Every other key of the state file stays."""
    known = remembered(state_path) or discover(config)
    known = [s for s in known if s.source != season.source] + [season]
    ui_state.update({"seasons": [{"source": str(s.source), "results": str(s.results)} for s in known],
                     "current_season": str(season.source)}, state_path)


def choose(source: str, config: dict[str, Any], state_path: Path | None = None) -> Season | None:
    """Make a season of the picker the current one (by its source folder) and save the choice."""
    season = next((s for s in available(config, state_path) if str(s.source) == source), None)
    if season is not None and (not remembered(state_path) or current_source(state_path) != source):
        remember(season, config, state_path)
    return season


# ---------------------------------------------------------------- the form of a new season

def describe_source(text: str) -> tuple[bool, str]:
    """(usable, message) for the folder with the original episodes."""
    if not text.strip():
        return False, "Впишите путь к папке с оригинальными сериями."
    folder = absolute(text)
    if not folder.exists():
        return False, f"❌ Папки {folder} нет (диск не подключён?)."
    if not folder.is_dir():
        return False, f"❌ {folder} - это файл, а нужна папка."
    names = video_names(folder)
    if not names:
        return False, f"❌ В папке {folder} нет видео ({', '.join(sorted(VIDEO_EXT))})."
    return True, f"✅ Найдено видео: {len(names)}."


def _creatable(folder: Path) -> str | None:
    """Why ``folder`` cannot be made (None: it can, or it exists)."""
    if folder.exists():
        return None if folder.is_dir() else f"{folder} - это файл, а нужна папка."
    parent = next((p for p in folder.parents if p.exists()), None)
    if parent is None or not parent.is_dir():
        return f"Нет такого места для папки: {folder}."
    return None if os.access(parent, os.W_OK) else f"Нет прав создать папку в {parent}."


def describe_results(source_text: str, results_text: str) -> tuple[bool, str]:
    """(usable, message) for the results folder: whether it exists and how many episodes already have results."""
    if not results_text.strip():
        return False, "Впишите путь к папке результатов."
    results = absolute(results_text)
    if problem := _creatable(results):
        return False, f"❌ {problem}"
    if not results.exists():
        return True, "Папки пока нет: она будет создана при сохранении."
    rows = episode_rows(Season(absolute(source_text), results)) if source_text.strip() else []
    done = sum(1 for r in rows if r.state != STATE_NEW)
    ready = sum(1 for r in rows if r.finished)
    if not rows:
        return True, "Папка уже есть."
    return True, (f"Папка уже есть: у {done} из {len(rows)} серий там есть результаты"
                  f"{f', готовых видео: {ready}' if ready else ''}.")


def validate(source_text: str, results_text: str) -> tuple[Season | None, str]:
    """The season to save, or a message saying what is wrong."""
    ok, message = describe_source(source_text)
    if not ok:
        return None, message
    ok, message = describe_results(source_text, results_text)
    if not ok:
        return None, message
    source, results = absolute(source_text), absolute(results_text)
    if source == results:
        return None, "❌ Папка результатов должна отличаться от папки с оригиналами."
    return Season(source, results), ""


# ---------------------------------------------------------------- episodes of a season

STATE_NEW = "не начата"
STATE_PREPARED = "подготовлена (ждёт перевода)"
PREPARED_STAGES = {"transcribe", "glossary", "speakers"}  # what ``stt.run --prepare`` leaves
STAGE_TOTAL = len(STAGES)


def _ready_video(results: Path, stem: str, top: set[str]) -> Path | None:
    """The finished video of an episode at the top of the results folder (or, the older layout, in its work folder)."""
    if f"{stem}.ru.mkv" in top:
        return results / f"{stem}.ru.mkv"
    prefix = f"{stem}.ru."
    for name in sorted(top):
        if name.startswith(prefix) and name.endswith(".mkv") and name[len(prefix):-4] in ENGINE_NAMES:
            return results / name
    older = results / stem / f"{stem}.ru.mkv"
    return older if older.is_file() else None


def _work_state(folder: Path) -> str:
    """"3 из 9 этапов", "⏳ идёт: ...", "⏸ прервано: ...", "❌ ошибка: ..." out of the stage marks and progress records."""
    marks = folder / ".stages"
    try:
        names = [p.name for p in marks.iterdir()] if marks.is_dir() else []
    except OSError:
        names = []
    if not folder.is_dir():
        return STATE_NEW
    engines = sorted({n.split(".", 1)[1] for n in names if "." in n and n.split(".", 1)[1] in ENGINE_NAMES})
    states = job_view.stage_states(folder, engines)
    for state, mark, word in (("running", "⏳", "идёт"), ("interrupted", "⏸", "прервано"), ("failed", "❌", "ошибка")):
        for (stage, _), (found, _) in states.items():
            if found == state:
                return f"{mark} {word}: {job_view.stage_title(stage)}"
    done = {stage for stage in STAGES if any(n == stage or n.startswith(stage + ".") for n in names)}
    if done == PREPARED_STAGES and not any(n.startswith("translate") for n in names):
        return STATE_PREPARED
    if done:
        return f"{len(done)} из {STAGE_TOTAL} этапов"
    return STATE_NEW if not names and not progress.read_all(folder) else "начата"


def episode_rows(season: Season) -> list[EpisodeStatus]:
    """The videos of the source folder with the state read from the results folder (folder listings and stat only)."""
    try:
        top = {p.name for p in season.results.iterdir()} if season.results.is_dir() else set()
    except OSError:
        top = set()
    rows = []
    for video in video_names(season.source):
        stem = Path(video).stem
        finished = _ready_video(season.results, stem, top)
        if finished is not None:
            rows.append(EpisodeStatus(video, stem, "✅ готово", finished))
        else:
            rows.append(EpisodeStatus(video, stem, _work_state(season.results / stem) if stem in top else STATE_NEW))
    return rows


def table(season: Season | None) -> list[list[str]]:
    """Rows of the table "Серия | Состояние | Готовое видео"."""
    if season is None:
        return []
    return [[job_view.short_name(r.stem), r.state, r.finished.name if r.finished else ""] for r in episode_rows(season)]


def summary(season: Season | None) -> str:
    """One line under the picker: where the results go."""
    if season is None:
        return "Сезон не выбран: нажмите «＋ Новый сезон» и укажите папку с оригинальными сериями."
    state = "" if season.results.is_dir() else " (папки пока нет, она создастся при запуске)"
    return f"Результаты сезона **{season.name}**: `{season.results}`{state}"


NOTE_TABS = ("process", "voices", "edits", "sound", "settings")
NO_SEASON_NOTE = "Сначала выберите сезон вверху страницы: тогда здесь будет написано, куда сохраняются файлы."


def tab_notes(season: Season | None, global_file: Path | str = "") -> dict[str, str]:
    """One line for the top of each tab saying what it writes where (file names as ``stt.run`` makes them: the
    work folder ``<results>/<episode>/``, the season files ``<results>/<Season>.*``, the video ``<stem>.ru.mkv``)."""
    if season is None:
        return {tab: NO_SEASON_NOTE for tab in NOTE_TABS}
    r, name = season.results, season.name
    own = r / "season_settings.json"
    return {
        "process": (f"📂 Всё по сезону пишется в `{r}`: папка каждой серии, готовые видео `*.ru.mkv`, "
                    f"проба — в `_проба`."),
        "voices": (f"📂 Образцы голосов героев (для ESpeech): `{r}/{name}.references/` · "
                   f"банк голосов: `{name}.voices.json` рядом."),
        "edits": f"📂 Правки реплик сохраняются в папке серии: `{r}/<серия>/corrections.json`.",
        "sound": (f"📂 Готовые видео — в `{r}`; сжатая копия и пробный кусок появляются рядом с выбранным видео."),
        "settings": (f"Общие настройки: `{global_file}` · настройки сезона {name}: `{own}`" if global_file else
                     f"Настройки сезона {name}: `{own}`"),
    }


def label(season: Season) -> str:
    """"Season_48 · 7 серий · /media/.../Season_48"."""
    if not season.source.is_dir():
        return f"{season.name} · папка недоступна · {season.source}"
    return f"{season.name} · {episodes_word(len(video_names(season.source)))} · {season.source}"


def choices(seasons: list[Season]) -> list[tuple[str, str]]:
    return [(label(s), str(s.source)) for s in seasons]


# ---------------------------------------------------------------- the config of a job

def season_config(base: dict[str, Any], season: Season) -> dict[str, Any]:
    """The config for a job in ``season``: work folder, season files and finished videos all in the results folder
    (``finished_folder`` "." puts the video at its top level)."""
    config = copy.deepcopy(base)
    results = str(season.results)
    config["paths"] = {**(config.get("paths") or {}), "work": results, "season": results, "finished": results,
                       "finished_folder": "."}
    return config
