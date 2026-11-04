"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _last_job_in_tmp(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that starts a UI job must never write the real ``output/ui_last_job.json``."""
    from ui import dub_app

    monkeypatch.setattr(dub_app, "LAST_JOB", tmp_path_factory.mktemp("ui") / "ui_last_job.json")


@pytest.fixture(autouse=True)
def _seasons_in_tmp(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """The UI remembers seasons in the local config and looks for them on the media drives: never the real ones."""
    from ui import seasons, settings

    monkeypatch.setattr(seasons, "MEDIA_GLOB", "/nonexistent-media/*/*/Survivor")
    monkeypatch.setattr(settings, "LOCAL", tmp_path_factory.mktemp("local") / "stt_config.local.json")


@pytest.fixture(autouse=True)
def _ui_state_in_tmp(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """The UI state file (``ui.local.json``) of the tests is a temporary one, never the real one."""
    from ui import state

    monkeypatch.setattr(state, "PATH", tmp_path_factory.mktemp("state") / "ui.local.json")
