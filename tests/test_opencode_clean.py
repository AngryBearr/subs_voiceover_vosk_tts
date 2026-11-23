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
    monkeypatch.setattr(oc.opencode_pin, "resolve_binary", lambda: (oc.Path("/fake/opencode"), "pinned"))
    monkeypatch.setattr(oc.opencode_pin, "note_binary", lambda *a: None)
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


def test_warm_up_polls_until_the_first_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    server = oc.CleanOpenCodeServer(required_models={})
    server.api_version = 2
    calls = [0]

    def generate(self: oc.CleanOpenCodeServer, prompt: str, model: str, variant: object = None,
                 timeout: float = 0) -> str:
        calls[0] += 1
        if calls[0] <= 2:
            raise oc.OpenCodeError("HTTP 404: ModelNotFoundError", transient=True)
        return "OK"

    sleeps: list[float] = []
    monkeypatch.setattr(oc.CleanOpenCodeServer, "generate", generate)
    monkeypatch.setattr(oc.time, "sleep", sleeps.append)
    assert server.warm_up("opencode:openai/gpt-6-luna") is True
    assert calls[0] == 3 and sleeps == [oc.WARMUP_POLL_S] * 2
    assert server.warm_up("openai/gpt-6.1-sol") is True  # same provider: not asked again
    assert calls[0] == 3


def test_warm_up_gives_up_but_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    server = oc.CleanOpenCodeServer(required_models={})
    server.api_version = 2
    clock = [0.0]

    def fail(*args: object, **kwargs: object) -> str:
        raise oc.OpenCodeError("HTTP 502", transient=True)

    monkeypatch.setattr(oc.CleanOpenCodeServer, "generate", fail)
    monkeypatch.setattr(oc.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(oc.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    assert server.warm_up("openai/gpt-6-luna", timeout=10.0) is False
    assert 8.0 <= clock[0] <= 10.0


def test_server_command_uses_the_resolved_binary_and_disables_autoupdate(monkeypatch: pytest.MonkeyPatch) -> None:
    server, _ = _server(monkeypatch, failures=0)
    seen: list[list[str]] = []
    monkeypatch.setattr(oc.subprocess, "Popen", lambda cmd, **k: seen.append(cmd) or _Process())
    server.start()
    assert seen[0][0] == "/fake/opencode"
    assert server._environment()["OPENCODE_DISABLE_AUTOUPDATE"] == "1"
