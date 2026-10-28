"""Settings page of the UI: the options of ``stt_config.json`` people change, kept in ``stt_config.local.json``.

Only values that differ from ``stt_config.json`` are written, so the rest follows future defaults;
other personal keys of the local file (LLM through a subscription, folders) are never touched.
"Reset" removes the page's own keys from the local file.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = REPO_ROOT / "stt_config.json"
LOCAL = REPO_ROOT / "stt_config.local.json"
TERATTS_VOICES = ["ru_f1+ru_f2", "ru_f1", "ru_f2", "ru_m5", "ru_m1"]  # offered first; any style of the model works
VOSK_SPEAKERS = 57  # vosk-model-tts-ru-0.10-multi


@dataclass(frozen=True)
class Setting:
    path: tuple[str, ...]
    label: str
    kind: str  # bool | choice | number | integer | text | folder | voice | engines
    group: str
    choices: dict[str, Any] = field(default_factory=dict)  # label -> value (choice, engines)
    info: str = ""
    nullable: bool = False  # an empty text means null


ENGINES = {"Vosk": "vosk", "TeraTTS": "teratts", "ESpeech (голоса героев)": "f5"}
MODES = {"Точный: Cohere + Whisper + LLM": "accurate", "Быстрый: только Whisper": "fast"}
MODELS = {"luna: дешевле, качество хорошее": "openai/gpt-6-luna", "Sol: лучше перевод, дороже": "openai/gpt-6.1-sol"}


class SettingsError(ValueError):
    """Values the page cannot save; ``problems`` are messages for the page."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems

SETTINGS: list[Setting] = [
    Setting(("paths", "work"), "Рабочая папка", "folder", "Папки", info="Промежуточные файлы серий"),
    Setting(("paths", "season"), "Папка файлов сезона", "folder", "Папки", nullable=True,
            info="Словарь, состав, банк голосов и образцы голосов; пусто: рабочая папка"),
    Setting(("paths", "finished"), "Папка готовых видео", "folder", "Папки", nullable=True,
            info="Видео кладутся в подпапку STT_<сезон>; пусто: в папку серии"),
    Setting(("ui", "engines"), "Движки, отмеченные при запуске", "engines", "Запуск", ENGINES),
    Setting(("mode",), "Распознавание", "choice", "Распознавание", MODES),
    Setting(("diarizer",), "Кто когда говорит", "choice", "Распознавание",
            {"DiariZen v2 (точнее; окружение diarizen_env)": "diarizen-v2",
             "pyannote community-1": "pyannote-community-1"}),
    Setting(("speaker_refine", "model"), "Уточнять спикеров по голосу", "choice", "Распознавание",
            {"Да (ResNet293)": "resnet293", "Нет": "none"}),
    Setting(("line_gender",), "Пол неопознанных спикеров по голосу каждой реплики", "bool", "Распознавание"),
    Setting(("ocr", "enabled"), "Читать надписи на экране (имена в титрах)", "bool", "Распознавание"),
    Setting(("cast", "enabled"), "Искать состав участников в интернете", "bool", "Распознавание"),
    Setting(("voice_bank", "enabled"), "Банк голосов сезона (узнавать героев между сериями)", "bool",
            "Распознавание"),
    Setting(("llm", "model"), "Языковая модель (перевод, имена)", "choice", "Языковые модели", MODELS),
    Setting(("llm", "stages", "stress_screen"), "Модель первого отсева ударений", "choice", "Языковые модели",
            {"luna": "openai/gpt-6-luna", "Sol": "openai/gpt-6.1-sol"}),
    Setting(("voiceover", "stress_check"), "Проверять ударения (три языковые модели)", "bool", "Озвучка"),
    Setting(("voiceover", "listen_stress"), "Слушать ударения в дублях (имена, омографы) и переозвучивать неверные",
            "bool", "Озвучка"),
    Setting(("voiceover", "speech_check"), "Проверять речь озвучки распознаванием", "bool", "Озвучка"),
    Setting(("voiceover", "engines", "vosk", "voices", "female"), "Vosk: женский голос", "integer", "Озвучка",
            info=f"Номер голоса 0-{VOSK_SPEAKERS - 1}"),
    Setting(("voiceover", "engines", "vosk", "voices", "male"), "Vosk: мужской голос", "integer", "Озвучка",
            info=f"Номер голоса 0-{VOSK_SPEAKERS - 1}"),
    Setting(("voiceover", "engines", "teratts", "voices", "female"), "TeraTTS: женский голос", "voice", "Озвучка",
            info="Голос или смесь двух: ru_f1+ru_f2"),
    Setting(("voiceover", "engines", "teratts", "voices", "male"), "TeraTTS: мужской голос", "voice", "Озвучка"),
    Setting(("voiceover", "engines", "f5", "fallback"), "ESpeech: голос людей без образца", "choice", "Озвучка",
            {"Голоса TeraTTS (выше)": "teratts", "Клон голоса Vosk": "vosk"}),
    Setting(("voiceover", "engines", "f5", "nfe"), "ESpeech: качество", "choice", "Озвучка",
            {"32 шага (лучше)": 32, "7 шагов (в 2 раза быстрее, на слух почти так же)": 7}),
    Setting(("mix", "subtitles"), "Вшивать русские субтитры", "bool", "Сведение"),
    Setting(("mix", "duck"), "Что приглушать под озвучкой", "choice", "Сведение",
            {"Только голоса (музыка остаётся)": "voices", "Весь звук": "all"}),
    Setting(("mix", "duck_db"), "Насколько приглушать, дБ", "number", "Сведение"),
    Setting(("mix", "tracks"), "Готовые видео", "choice", "Сведение",
            {"Одно видео со всеми озвучками": "combined", "Отдельное видео на каждый движок": "separate"}),
    Setting(("mix", "keep_wav"), "Хранить несжатый звук (~0,5 ГБ на серию)", "bool", "Сведение"),
    Setting(("crowd", "enabled"), "Слой толпы: выкрики несколькими голосами (опыт)", "bool", "Сведение"),
    Setting(("review",), "Список подозрительных реплик для проверки", "bool", "Сведение"),
]

FALLBACKS: dict[tuple[str, ...], Any] = {("line_gender",): True, ("review",): True, ("ui", "engines"): ["vosk"],
                                         ("voiceover", "listen_stress"): True}


def teratts_styles() -> set[str]:
    """Voices of the downloaded TeraTTS model (its ``styles`` folders); the usual ones if not downloaded."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    found = {p.name for p in hub.glob("models--TeraSpace--TeraTTSv2/snapshots/*/styles/*") if p.is_dir()}
    return found or {part for voice in TERATTS_VOICES for part in voice.split("+")}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def get(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return FALLBACKS.get(path)
        node = node[key]
    return node


def put(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = config
    for key in path[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[path[-1]] = value


def drop(config: dict[str, Any], path: tuple[str, ...]) -> None:
    """Remove a key and the dicts it leaves empty."""
    trail, node = [], config
    for key in path[:-1]:
        if not isinstance(node.get(key), dict):
            return
        trail.append((node, key))
        node = node[key]
    node.pop(path[-1], None)
    for parent, key in reversed(trail):
        if parent[key]:
            break
        del parent[key]


def to_form(setting: Setting, value: Any) -> Any:
    """A config value as the form shows it."""
    if setting.kind == "choice":
        return next((label for label, v in setting.choices.items() if v == value), None)
    if setting.kind == "engines":
        return [label for label, v in setting.choices.items() if v in (value or [])]
    if setting.kind in ("text", "folder", "voice"):
        return "" if value is None else str(value)
    return value


def from_form(setting: Setting, value: Any) -> Any:
    """A form value as the config stores it; ``ValueError`` with a message for the page."""
    if setting.kind == "choice":
        if value not in setting.choices:
            raise ValueError(f"«{setting.label}»: выберите вариант из списка")
        return setting.choices[value]
    if setting.kind == "engines":
        return [setting.choices[label] for label in value or [] if label in setting.choices]
    if setting.kind == "bool":
        return bool(value)
    if setting.kind in ("number", "integer"):
        if value is None or value == "":
            raise ValueError(f"«{setting.label}»: введите число")
        number = float(value)
        if setting.kind == "number":
            return number
        if number != int(number):
            raise ValueError(f"«{setting.label}»: нужно целое число")
        if setting.path[-2:-1] == ("voices",) and not 0 <= number < VOSK_SPEAKERS:
            raise ValueError(f"«{setting.label}»: номер голоса Vosk от 0 до {VOSK_SPEAKERS - 1}")
        return int(number)
    text = str(value or "").strip()
    if setting.kind == "voice":
        known = teratts_styles()
        parts = text.split("+") if text else []
        if not parts or len(parts) > 2 or any(part not in known for part in parts):
            raise ValueError(f"«{setting.label}»: голос TeraTTS ({', '.join(sorted(known))}) или смесь двух через +")
    if not text and setting.nullable:
        return None
    return text


def form_values(config: dict[str, Any]) -> list[Any]:
    return [to_form(s, get(config, s.path)) for s in SETTINGS]


def save(values: list[Any], defaults_path: Path = DEFAULTS, local_path: Path = LOCAL) -> tuple[int, list[str]]:
    """Write the form into the local file: (values that differ from the defaults, warnings).

    Raises ``SettingsError`` (nothing written) when a value is not valid."""
    defaults, local = read_json(defaults_path), read_json(local_path)
    parsed, problems = [], []
    for setting, value in zip(SETTINGS, values):
        try:
            parsed.append((setting, from_form(setting, value)))
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        raise SettingsError(problems)
    own, warnings = 0, []
    for setting, value in parsed:
        if value == get(defaults, setting.path):
            drop(local, setting.path)
        else:
            put(local, setting.path, value)
            own += 1
        if setting.kind == "folder" and value and not _folder(str(value)).is_dir():
            warnings.append(f"«{setting.label}»: папки {value} сейчас нет (диск не подключён?), сохранено как есть")
    _write(local_path, local)
    return own, warnings


def reset(local_path: Path = LOCAL) -> None:
    """Back to ``stt_config.json``: the page's keys leave the local file, other personal keys stay."""
    local = read_json(local_path)
    for setting in SETTINGS:
        drop(local, setting.path)
    if local_path.is_file():
        _write(local_path, local)


def _folder(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _write(local_path: Path, local: dict[str, Any]) -> None:
    """The local file, the previous version kept as ``.bak`` (personal settings are not in git)."""
    if local_path.is_file():
        shutil.copyfile(local_path, local_path.with_name(local_path.name + ".bak"))
    local_path.write_text(json.dumps(local, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
