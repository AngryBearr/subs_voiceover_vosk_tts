"""Offline child-process checks for the terminal wrapper; no model import or calls."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from utils import shortening_terminal as terminal


def test_passthrough_and_soft_limit(monkeypatch: object, tmp_path: Path) -> None:
    captured: list[tuple[list[str], float]] = []

    def fake_run(argv: list[str], _directory: Path, *, chunk: float) -> int:
        captured.append((argv, chunk))
        return 4

    monkeypatch.setattr(terminal, "_run", fake_run)  # type: ignore[attr-defined]
    base = ["input.srt", "--output-dir", str(tmp_path / "job"), "--profile", "profile.json", "--budget-usd", "10"]
    assert terminal.main(base) == 4
    assert captured[-1][0][-2:] == ["--run-time-limit", "600"]
    assert captured[-1][1] == 600
    assert terminal.main([*base, "--resume", "--retry-unknown", "abc"]) == 4
    assert "--retry-unknown" in captured[-1][0]
    assert "--run-time-limit" not in captured[-1][0]  # frozen resume contract may require None


def test_journal_partial_supersession_and_rotation(tmp_path: Path) -> None:
    journal = tmp_path / "requests.jsonl"
    old, new = "a" * 32, "b" * 32
    start = lambda key: json.dumps({"request_id": key, "state": "start", "stage": "review",
                                    "model": {"model": "sol"}, "started_at": "2026-09-30T06:00:00+00:00"}) + "\n"
    journal.write_text(start(old) + start(new)[:-1], encoding="utf-8")
    view = terminal.JournalView(tmp_path)
    assert view.refresh() == []
    assert view.active() == [old]
    with journal.open("a") as stream:
        stream.write("\n" + json.dumps({"request_id": new, "state": "complete", "started_at": "2026-09-30T06:00:00+00:00", "elapsed_sec": 2}) + "\n")
    (tmp_path / "resume_events.jsonl").write_text(json.dumps({"event": "retry_unknown", "request_id": old,
                                                           "replacement_request_id": new}) + "\n")
    assert view.refresh() == []
    assert view.active() == []
    assert "last_durable_complete=bbbbbbbb@2026-09-30T06:00:02+00:00" in view.snapshot()
    journal.write_text(start("c" * 32))
    assert any("rotated/truncated" in warning for warning in view.refresh())
    assert view.refresh() == []
    assert view.active() == ["c" * 32]


def test_child_streams_exit_log_and_stall(tmp_path: Path, capsys: object) -> None:
    code = "import sys,time;print('OUT', flush=True);print('ERR', file=sys.stderr, flush=True);time.sleep(.16);sys.exit(7)"
    result = terminal._run([], tmp_path, chunk=5, heartbeat=.03, stall=.08,
                           command=[sys.executable, "-u", "-c", code])
    text = capsys.readouterr().out  # type: ignore[attr-defined]
    assert result == 7
    assert "child/stdout: OUT" in text and "child/stderr: ERR" in text
    assert "нет подтверждённого прогресса" in text  # heartbeats did not reset stall
    assert "child_exit=7" in text
    logs = list((tmp_path / "logs").glob("terminal-*.log"))
    assert len(logs) == 1 and "child_exit=7" in logs[0].read_text()


def test_first_signal_forwarded_once(tmp_path: Path, capsys: object) -> None:
    ready = tmp_path / "ready"
    received = tmp_path / "received"
    code = ("import signal,time,pathlib,sys; "
            "ready=pathlib.Path(sys.argv[1]); received=pathlib.Path(sys.argv[2]); "
            "signal.signal(signal.SIGINT, lambda *_: (received.write_text('one'), sys.exit(0))); "
            "ready.touch(); "
            "\nwhile True: time.sleep(.01)")

    def send_once() -> None:
        for _ in range(200):
            if ready.exists():
                os.kill(os.getpid(), signal.SIGINT)
                return
            time.sleep(.01)

    sender = threading.Thread(target=send_once)
    sender.start()
    try:
        assert terminal._run([], tmp_path, chunk=5, heartbeat=.05,
                             command=[sys.executable, "-u", "-c", code, str(ready), str(received)]) == 0
    finally:
        sender.join(timeout=3)
    assert received.read_text() == "one"
    assert "forwarded_to_child count=1" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_deadline_is_soft_stop_and_help_is_offline(tmp_path: Path, capsys: object) -> None:
    with pytest.raises(SystemExit) as help_exit:
        terminal.main(["--help"])
    assert help_exit.value.code == 0
    seen = tmp_path / "signal"
    code = ("import signal,time,pathlib,sys; "
            "signal.signal(signal.SIGINT, lambda *_: (pathlib.Path(sys.argv[1]).write_text('once'), sys.exit(0))); "
            "\nwhile True: time.sleep(.01)")
    assert terminal._run([], tmp_path, chunk=.08, heartbeat=.02,
                         command=[sys.executable, "-u", "-c", code, str(seen)]) == 0
    assert seen.read_text() == "once"
    assert "chunk deadline: first SIGINT forwarded" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_child_exit_during_signal_still_reports_exit(tmp_path: Path, capsys: object,
                                                      monkeypatch: object) -> None:
    def exiting_child(self: subprocess.Popen[bytes], _signum: int) -> None:
        self.wait(timeout=2)
        raise ProcessLookupError

    monkeypatch.setattr(subprocess.Popen, "send_signal", exiting_child)  # type: ignore[attr-defined]
    assert terminal._run([], tmp_path, chunk=.005, heartbeat=.01,
                         command=[sys.executable, "-u", "-c", "import time;time.sleep(.03)"]) == 0
    text = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "signal not forwarded: child exited" in text
    assert "END child_exit=0" in text
