"""The state of the UI itself: remembered seasons, the current season, the chosen global settings file.

Kept in ``ui.local.json`` in the repository root (``*.local.json`` is ignored by git)::

    {"seasons": [{"source": "...", "results": "..."}], "current_season": "<source>", "global_settings": "<file>"}

Before this file existed the seasons were kept in ``stt_config.local.json`` under ``ui.seasons`` and
``ui.current_season``: while ``ui.local.json`` knows no seasons they are read from there (that file is never
changed by this module). Every write is atomic and keeps unknown keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ui import settings
from utils.atomic_io import write_text_atomic

PATH = settings.REPO_ROOT / "ui.local.json"
LEGACY_KEYS = ("seasons", "current_season")


def read(path: Path | None = None) -> dict[str, Any]:
    """The state; without seasons of its own, the ones the older versions kept in ``stt_config.local.json``."""
    try:
        state = settings.read_json(path or PATH)
    except (OSError, ValueError):
        state = {}
    if not isinstance(state, dict):
        state = {}
    if not state.get("seasons"):
        try:
            legacy = settings.read_json(settings.LOCAL).get("ui") or {}
        except (OSError, ValueError, AttributeError):
            legacy = {}
        if isinstance(legacy, dict) and legacy.get("seasons"):
            for key in LEGACY_KEYS:
                if legacy.get(key):
                    state[key] = legacy[key]
    return state


def update(changes: dict[str, Any], path: Path | None = None) -> None:
    """Set keys of the state (a ``None`` value removes the key) and write the file."""
    target = path or PATH
    state = read(target)
    for key, value in changes.items():
        if value is None:
            state.pop(key, None)
        else:
            state[key] = value
    write_text_atomic(target, json.dumps(state, ensure_ascii=False, indent=2) + "\n")
