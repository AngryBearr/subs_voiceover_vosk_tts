"""Choosing a folder: the desktop dialog (zenity, mocked) and the helpers of the in-page browser."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from ui import folder_pick


def test_desktop_detection() -> None:
    assert not folder_pick.has_desktop({})
    assert folder_pick.has_desktop({"DISPLAY": ":0"}) and folder_pick.has_desktop({"WAYLAND_DISPLAY": "wayland-0"})


def test_dialog_needs_a_desktop_and_zenity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(folder_pick.shutil, "which", lambda name: "/usr/bin/zenity")
    assert folder_pick.dialog_available({"DISPLAY": ":0"}) and not folder_pick.dialog_available({})
    monkeypatch.setattr(folder_pick.shutil, "which", lambda name: None)
    assert not folder_pick.dialog_available({"DISPLAY": ":0"})


def test_start_dir_prefers_the_field_then_media_then_home(tmp_path: Path) -> None:
    media, home = tmp_path / "media", tmp_path / "home"
    (media / "bob").mkdir(parents=True)
    home.mkdir()
    folder = tmp_path / "data" / "Season_1"
    folder.mkdir(parents=True)
    kwargs: dict[str, Any] = {"media_root": media, "home": home, "user": "bob"}
    assert folder_pick.start_dir(str(folder), **kwargs) == folder
    assert folder_pick.start_dir(str(folder / "STT_new"), **kwargs) == folder  # not made yet: the folder above
    assert folder_pick.start_dir(str(folder / "x.json"), **kwargs) == folder
    assert folder_pick.start_dir("", **kwargs) == media / "bob"
    assert folder_pick.start_dir("relative/path", **kwargs) == media / "bob"
    assert folder_pick.start_dir("", media_root=media, home=home, user="nobody") == home


def test_zenity_commands(tmp_path: Path) -> None:
    folder = folder_pick.zenity_command(folder_pick.FOLDER, tmp_path, "Папка")
    assert folder == ["zenity", "--file-selection", "--directory", "--title=Папка", f"--filename={tmp_path}/"]
    file = folder_pick.zenity_command(folder_pick.FILE, tmp_path, "Файл", "g.json")
    assert file[:4] == ["zenity", "--file-selection", "--save", "--confirm-overwrite"]
    assert f"--filename={tmp_path}/g.json" in file and "--file-filter=JSON | *.json" in file


def _fake_zenity(monkeypatch: pytest.MonkeyPatch, code: int, out: str = "", err: str = "") -> list[list[str]]:
    seen: list[list[str]] = []

    def run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, code, out, err)

    monkeypatch.setattr(folder_pick.shutil, "which", lambda name: "/usr/bin/zenity")
    monkeypatch.setattr(folder_pick.subprocess, "run", run)
    return seen


def test_dialog_picked_cancelled_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = {"DISPLAY": ":0"}
    seen = _fake_zenity(monkeypatch, 0, f"{tmp_path}/Season_1\n")
    pick = folder_pick.run_dialog(folder_pick.FOLDER, str(tmp_path), "T", env)
    assert pick == folder_pick.Pick(f"{tmp_path}/Season_1", "picked") and seen[0][-1] == f"--filename={tmp_path}/"
    _fake_zenity(monkeypatch, 1)
    assert folder_pick.run_dialog(folder_pick.FOLDER, "", "T", env).status == "cancelled"
    _fake_zenity(monkeypatch, 5, err="cannot open display")
    failed = folder_pick.run_dialog(folder_pick.FOLDER, "", "T", env)
    assert failed.status == "unavailable" and "display" in failed.error
    seen = _fake_zenity(monkeypatch, 0)
    assert folder_pick.run_dialog(folder_pick.FOLDER, "", "T", {}).status == "unavailable" and not seen  # no desktop


def test_click_without_a_desktop_opens_the_browser(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    (tmp_path / "A").mkdir()
    out = folder_pick.open_picker(folder_pick.FOLDER, "source", str(tmp_path), "T")
    field, message, panel, path, target, kind, line, rows, name, choose = out
    assert panel["visible"] is True and path == str(tmp_path) and target == "source" and rows == [["A"]]
    assert str(tmp_path) in line


def test_click_with_a_desktop_fills_the_field_or_keeps_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DISPLAY", ":0")
    _fake_zenity(monkeypatch, 0, f"{tmp_path}\n")
    out = folder_pick.open_picker(folder_pick.FOLDER, "results", "", "T")
    assert out[0]["value"] == str(tmp_path) and out[2]["visible"] is False
    _fake_zenity(monkeypatch, 1)
    out = folder_pick.open_picker(folder_pick.FOLDER, "results", "", "T")
    assert "value" not in out[0] and "отменён" in out[1] and out[2]["visible"] is False


def test_browser_helpers(tmp_path: Path) -> None:
    for name in ("b", "A", ".hidden"):
        (tmp_path / name).mkdir()
    (tmp_path / "file.txt").write_text("")
    assert folder_pick.list_subfolders(tmp_path) == (["A", "b"], "")
    assert folder_pick.list_subfolders(tmp_path / "missing")[0] == []
    assert "нельзя прочитать" in folder_pick.list_subfolders(tmp_path / "missing")[1]
    path, line, rows = folder_pick.navigate(str(tmp_path), folder_pick.FOLDER, "A")
    assert path == str(tmp_path / "A") and rows == []
    assert folder_pick.navigate(path, folder_pick.FOLDER, "up")[0] == str(tmp_path)
    assert folder_pick.navigate(str(tmp_path), folder_pick.FOLDER, "nothing")[0] == str(tmp_path)  # no such folder
    (tmp_path / "e1.mkv").write_text("")
    assert "видео здесь: 1" in folder_pick.browser_view(tmp_path)[0]


def test_shortcuts_and_chosen_value(tmp_path: Path) -> None:
    (tmp_path / "media" / "bob").mkdir(parents=True)
    found = folder_pick.shortcuts(tmp_path / "media", tmp_path, "bob")
    assert found == [(str(tmp_path / "media" / "bob"), tmp_path / "media" / "bob"), ("Домашняя папка", tmp_path)]
    assert folder_pick.shortcuts(tmp_path / "media", tmp_path, "alice") == [("Домашняя папка", tmp_path)]
    assert folder_pick.chosen_value(tmp_path, folder_pick.FOLDER, "") == str(tmp_path)
    assert folder_pick.chosen_value(tmp_path, folder_pick.FILE, "g.json") == str(tmp_path / "g.json")
    assert folder_pick.chosen_value(tmp_path, folder_pick.FILE, " ") == str(tmp_path / "stt_config.local.json")
