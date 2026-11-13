"""Startup of the clean OpenCode server: a healthy server may still be reading its config."""
from __future__ import annotations

import pytest

import utils.opencode_clean as oc


class _Process:
    def poll(self) -> None:
        return None


def _server(monkeypatch: pytest.MonkeyPatch, failures: int) -> tuple[oc.CleanOpenCodeServer, list[int]]:
    calls = [0]

    def verify(self: oc.CleanOpenCodeServer) -> None:
        calls[0] += 1
        if calls[0] <= failures:
            raise RuntimeError("foreign active agent enabled")

    def make_config(self: oc.CleanOpenCodeServer) -> None:
        self._neutral_dir = oc.Path(".")
        self._temp_root = oc.Path(".")

    monkeypatch.setattr(oc, "_validate_oauth", lambda env: None)
    monkeypatch.setattr(oc.CleanOpenCodeServer, "_make_config", make_config)
    monkeypatch.setattr(oc.CleanOpenCodeServer, "_healthy", lambda self: True)
    monkeypatch.setattr(oc.CleanOpenCodeServer, "_verify_metadata", verify)
    monkeypatch.setattr(oc.CleanOpenCodeServer, "stop", lambda self: None)
    monkeypatch.setattr(oc.subprocess, "Popen", lambda *a, **k: _Process())
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)
    return oc.CleanOpenCodeServer(required_models={}, timeout=60.0), calls


def test_start_waits_for_the_config(monkeypatch: pytest.MonkeyPatch) -> None:
    server, calls = _server(monkeypatch, failures=3)
    server.start()
    assert calls[0] == 4


def test_start_gives_up_on_a_config_that_stays_foreign(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]

    def monotonic() -> float:
        clock[0] += 1.0
        return clock[0]

    server, calls = _server(monkeypatch, failures=10**6)
    monkeypatch.setattr(oc.time, "monotonic", monotonic)
    with pytest.raises(RuntimeError, match="foreign active agent"):
        server.start()
    assert calls[0] <= oc.CONFIG_SETTLE_S + 2  # the settle window, not the whole startup timeout
