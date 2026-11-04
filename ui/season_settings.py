"""Where the settings come from: the layers of the config the UI shows and runs with.

1. ``stt_config.json`` - the standard values (in git);
2. the global settings file - by default ``stt_config.local.json`` (personal keys live there too); the person may
   choose another file on the "Настройки" tab, the choice is kept in the UI state file (``ui.state``);
3. the settings of the season ``<results>/season_settings.json`` when that file exists: a snapshot of the values of
   the "Настройки" tab (the keys of ``ui.settings.SETTINGS`` only, nested like the config);
4. the options of the "Обработка" form, then the season's folders (``ui.seasons.season_config``) - made by the caller.

A season takes the global settings until it gets a file of its own. ``stt.run`` on the command line keeps reading
``stt_config.local.json``; the jobs of the UI get a full config file anyway.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from stt.run import deep_merge
from ui import settings
from ui import state as ui_state
from utils.atomic_io import write_text_atomic

FILE_NAME = "season_settings.json"
GLOBAL_LABEL = "общие настройки"
_MISSING = object()


def _absolute(text: str | Path) -> Path:
    path = Path(str(text).strip().strip("'\"")).expanduser()
    return path if path.is_absolute() else settings.REPO_ROOT / path


def _write(path: Path, data: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


# ---------------------------------------------------------------- the global file

def global_path(state_path: Path | None = None) -> Path:
    """The global settings file: the chosen one, else ``stt_config.local.json``."""
    chosen = ui_state.read(state_path).get("global_settings")
    return _absolute(chosen) if chosen else settings.LOCAL


def global_config(defaults_path: Path | None = None, state_path: Path | None = None) -> dict[str, Any]:
    """``stt_config.json`` with the global settings file on top."""
    try:
        own = settings.read_json(global_path(state_path))
    except ValueError:
        own = {}
    return deep_merge(settings.read_json(defaults_path or settings.DEFAULTS), own)


def choose_global(text: str, state_path: Path | None = None) -> tuple[str, str]:
    """Use the file ``text`` as the global settings. Returns (``ok`` | ``same`` | ``error``, message).

    A file that does not exist is created as a copy of the current global settings."""
    if not text.strip():
        return "error", "Впишите путь к файлу общих настроек."
    path = _absolute(text)
    current = global_path(state_path)
    if path == current:
        return "same", f"Общие настройки уже в `{path}`."
    if path.is_dir():
        return "error", f"❌ {path} - это папка, а нужен файл (например, {path / 'settings.json'})."
    if path.suffix.lower() != ".json":
        return "error", "❌ Файл настроек должен называться с расширением .json."
    if path.exists():
        try:
            if not isinstance(settings.read_json(path), dict):
                raise ValueError("not an object")
        except (OSError, ValueError):
            return "error", f"❌ Файл {path} не читается как файл настроек (JSON)."
        made = "Взят существующий файл."
    else:
        try:
            _write(path, settings.read_json(current) if current.is_file() else {})
        except (OSError, ValueError) as error:
            return "error", f"❌ Не получилось создать {path}: {error}"
        made = "Файла не было: он создан как копия прежних общих настроек."
    ui_state.update({"global_settings": None if path == settings.LOCAL else str(path)}, state_path)
    return "ok", f"Общие настройки теперь в `{path}`. {made}"


# ---------------------------------------------------------------- the file of a season

def season_file(results: Path) -> Path:
    return results / FILE_NAME


def has_own(results: Path | None) -> bool:
    return results is not None and season_file(results).is_file()


def _lookup(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        node = node[key]
    return node


def snapshot(config: dict[str, Any]) -> dict[str, Any]:
    """The values of the settings of the tab in ``config`` (with the page's fallbacks), nested like the config."""
    out: dict[str, Any] = {}
    for setting in settings.SETTINGS:
        value = settings.get(config, setting.path)
        if value is not None or setting.nullable:
            settings.put(out, setting.path, copy.deepcopy(value))
    return out


def read_own(results: Path | None) -> dict[str, Any]:
    """The settings of the season, only the keys of the tab; empty without a file (or a file that cannot be read)."""
    if not has_own(results):
        return {}
    assert results is not None
    try:
        raw = settings.read_json(season_file(results))
    except (OSError, ValueError):
        return {}
    out: dict[str, Any] = {}
    for setting in settings.SETTINGS:
        value = _lookup(raw, setting.path)
        if value is not _MISSING:
            settings.put(out, setting.path, value)
    return out


def effective(base: dict[str, Any], results: Path | None) -> dict[str, Any]:
    """A copy of ``base`` (standard + global) with the settings of the season on top, when it has them."""
    return deep_merge(copy.deepcopy(base), read_own(results))


def which(results: Path | None) -> str:
    return "season" if has_own(results) else "global"


def settings_differ(one: dict[str, Any], other: dict[str, Any]) -> bool:
    return any(settings.get(one, s.path) != settings.get(other, s.path) for s in settings.SETTINGS)


def save_own(values: list[Any], results: Path, base: dict[str, Any],
             only: list[settings.Setting] | None = None) -> Path:
    """Write the form into the season's file (``only``: just those settings, the rest stays). Without a file yet
    the rest is taken from ``base``. Raises ``settings.SettingsError`` (nothing written) for bad values."""
    parsed, problems = [], []
    for setting, value in zip(only or settings.SETTINGS, values):
        try:
            parsed.append((setting, settings.from_form(setting, value)))
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        raise settings.SettingsError(problems)
    data = read_own(results) if has_own(results) else snapshot(base)
    for setting, value in parsed:
        settings.put(data, setting.path, value)
    _write(season_file(results), data)
    return season_file(results)


def reset_own(results: Path, defaults: dict[str, Any]) -> Path:
    """The season's file with the standard values."""
    _write(season_file(results), snapshot(defaults))
    return season_file(results)


def remove_own(results: Path | None) -> bool:
    """Delete the season's file (the season goes back to the global settings); False when there was none."""
    if not has_own(results):
        return False
    assert results is not None
    season_file(results).unlink()
    return True


def copy_own(source: Path, target: Path) -> Path:
    """Copy the settings of one season to the results folder of another (made if needed)."""
    _write(season_file(target), read_own(source))
    return season_file(target)


def should_ask(previous: Path | None, new: Path | None, base: dict[str, Any]) -> bool:
    """Ask "which settings to take" when the new season has no settings, the previous one had some, and they
    differ from the global ones."""
    if previous is None or new is None or previous == new or has_own(new) or not has_own(previous):
        return False
    return settings_differ(effective(base, previous), base)


def in_effect(results: Path | None) -> str:
    """Words for the summary: "общие настройки" or "настройки сезона"."""
    return "настройки сезона" if has_own(results) else GLOBAL_LABEL
