"""Pinned OpenCode: which binary is used, and when ``current`` may be switched."""
from __future__ import annotations

from pathlib import Path

import pytest

from utils import opencode_pin as pin


def _fake_binary(path: Path, version: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho 'opencode v{version}'\n")
    path.chmod(0o755)
    return path


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SUBS_SHARED", str(tmp_path / "shared"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(tmp_path / "emptybin"))
    monkeypatch.delenv(pin.BIN_ENV, raising=False)
    pin._LOGGED.clear()
    return tmp_path


def test_resolution_order(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert pin.resolve_binary() == (None, "missing")
    home_bin = _fake_binary(home / "home" / ".opencode" / "bin" / "opencode", "1.0.0")
    assert pin.resolve_binary() == (home_bin, "home")
    path_bin = _fake_binary(home / "pathbin" / "opencode", "2.0.0")
    monkeypatch.setenv("PATH", str(path_bin.parent))
    assert pin.resolve_binary() == (path_bin, "PATH")
    _fake_binary(pin.tools_dir() / "2.1.0" / "opencode", "2.1.0")
    pinned = pin.pinned_binary()
    pin.switch_current("2.1.0")
    assert pin.resolve_binary() == (pinned, "pinned")
    env_bin = _fake_binary(home / "envbin" / "opencode", "3.0.0")
    monkeypatch.setenv(pin.BIN_ENV, str(env_bin))
    assert pin.resolve_binary() == (env_bin, "env")
    monkeypatch.setenv(pin.BIN_ENV, str(home / "missing"))  # a broken override falls through
    assert pin.resolve_binary() == (pinned, "pinned")


def test_version_and_description(home: Path) -> None:
    binary = _fake_binary(home / "x" / "opencode", "2.4.0")
    assert pin.binary_version(binary) == "2.4.0"
    _fake_binary(pin.tools_dir() / "2.4.0" / "opencode", "2.4.0")
    assert pin.describe()[1] is False  # nothing pinned yet
    pin.switch_current("2.4.0")
    text, pinned = pin.describe()
    assert pinned and "v2.4.0" in text


def test_pin_switches_current_only_after_a_passing_check(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old = _fake_binary(pin.tools_dir() / "2.3.0" / "opencode", "2.3.0")
    pin.switch_current("2.3.0")
    new = _fake_binary(home / "src" / "opencode", "2.4.0")
    checked: list[Path] = []
    monkeypatch.setattr(pin, "check_binary", lambda binary, models: checked.append(binary) or None)
    assert pin.pin(new, ("m",)) == 0
    assert checked == [pin.tools_dir() / "2.4.0" / "opencode"]  # the copy is checked, not the original
    assert pin.current_version() == "2.4.0"
    assert pin.installed_versions() == ["2.3.0", "2.4.0"]
    assert old.exists()


def test_pin_keeps_the_old_current_when_the_check_fails(home: Path, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    _fake_binary(pin.tools_dir() / "2.3.0" / "opencode", "2.3.0")
    pin.switch_current("2.3.0")
    new = _fake_binary(home / "src" / "opencode", "2.4.0")
    monkeypatch.setattr(pin, "check_binary", lambda binary, models: "m: HTTP 404")
    assert pin.pin(new, ("m",)) == 1
    assert pin.current_version() == "2.3.0"
    assert "HTTP 404" in capsys.readouterr().out


def test_dry_run_copies_nothing(home: Path) -> None:
    new = _fake_binary(home / "src" / "opencode", "2.4.0")
    assert pin.pin(new, ("m",), dry_run=True) == 0
    assert not pin.tools_dir().exists()


def test_note_binary_warns_when_not_pinned(home: Path, caplog: pytest.LogCaptureFixture) -> None:
    binary = _fake_binary(home / "x" / "opencode", "2.4.0")
    with caplog.at_level("INFO"):
        pin.note_binary(binary, "PATH")
        pin.note_binary(binary, "PATH")  # once per process
    assert sum("OpenCode binary" in r.message for r in caplog.records) == 1
    assert any("not pinned" in r.message for r in caplog.records)
