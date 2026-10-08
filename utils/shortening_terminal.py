"""Run one bounded shorten_file invocation in a visible terminal; no automatic resume."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
from typing import Any


def _stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class JournalView:
    """Read complete appended JSONL rows only; never modify the run journal."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.positions: dict[str, tuple[int, int, bytes]] = {}
        self.starts: dict[str, dict[str, Any]] = {}
        self.completed: set[str] = set()
        self.superseded: set[str] = set()
        self.latest: dict[str, Any] | None = None
        self.changed = False

    def refresh(self) -> list[str]:
        warnings: list[str] = []
        self.changed = False
        for name in ("requests.jsonl", "resume_events.jsonl"):
            path = self.directory / name
            try:
                with path.open("rb") as stream:
                    stat = os.fstat(stream.fileno())
                    inode, offset, pending = self.positions.get(name, (stat.st_ino, 0, b""))
                    if inode != stat.st_ino or stat.st_size < offset:
                        warnings.append(f"WARNING: {name} rotated/truncated; journal view rebuilding")
                        self.positions.clear()
                        self.starts.clear()
                        self.completed.clear()
                        self.superseded.clear()
                        self.latest = None
                        return warnings
                    stream.seek(offset)
                    data = stream.read()
                    lines = (pending + data).split(b"\n")
                    self.positions[name] = (stat.st_ino, stream.tell(), lines.pop())
                    for raw in lines:
                        try:
                            row = json.loads(raw)
                        except (ValueError, UnicodeDecodeError):
                            warnings.append(f"WARNING: invalid complete row in {name}; inspect journal")
                            continue
                        if name == "resume_events.jsonl":
                            if row.get("event") == "retry_unknown":
                                self.superseded.add(row["request_id"])
                        elif row.get("state") == "start":
                            self.starts[row["request_id"]] = {
                                "stage": row.get("stage", "unknown"),
                                "model": row.get("model", {}).get("model", "unknown"),
                                "started_at": row.get("started_at"),
                            }
                        elif row.get("state") == "complete":
                            self.completed.add(row["request_id"])
                            try:
                                when = datetime.fromisoformat(row["started_at"]) + timedelta(seconds=row["elapsed_sec"])
                                completed_at = when.isoformat(timespec="seconds")
                            except (KeyError, TypeError, ValueError):
                                completed_at = "time-unknown"
                            self.latest = {"id": row["request_id"], "time": completed_at}
                        self.changed = True
            except FileNotFoundError:
                continue
            except (OSError, TypeError, KeyError) as exc:
                warnings.append(f"WARNING: cannot monitor {name}: {type(exc).__name__}")
        return warnings

    def active(self) -> list[str]:
        return [key for key in self.starts if key not in self.completed and key not in self.superseded]

    def snapshot(self) -> str:
        active = self.active()
        if active:
            key = active[-1]
            entry = self.starts[key]
            try:
                seconds = max(0, int((datetime.now(timezone.utc) - datetime.fromisoformat(entry["started_at"])).total_seconds()))
            except (TypeError, ValueError):
                seconds = 0
            current = f"stage={entry['stage']} model={entry['model']} request={key[:8]} request_elapsed={seconds}s"
        else:
            current = "stage=unknown/replay-or-local-checkpoint request=none"
        last = f"{self.latest['id'][:8]}@{self.latest['time']}" if self.latest else "none"
        return f"{current} last_durable_complete={last} unresolved={','.join(active) or 'none'}"


def _checkpoint(directory: Path, prior: str) -> str:
    path = directory / "report.json"
    try:
        with path.open(encoding="utf-8") as stream:
            report = json.load(stream)
        return f"checkpoint accepted={report.get('summary', {}).get('accepted_count', '?')} unresolved={report.get('outcome_summary', {}).get('unresolved_count', '?')}"
    except (OSError, ValueError, TypeError):
        return prior


def _run(child_args: list[str], directory: Path, *, chunk: float = 600, heartbeat: float = 15,
         stall: float = 360, command: list[str] | None = None) -> int:
    log_dir = directory / "logs" if directory.is_dir() else directory.parent / "logs"
    log_path = log_dir / f"terminal-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{time.time_ns()}.log"
    log: Any = None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_path.open("x", encoding="utf-8", buffering=1)
    except OSError as exc:
        print(f"WARNING: log unavailable ({type(exc).__name__}); console only", file=sys.stderr, flush=True)

    def emit(message: str) -> None:
        nonlocal log
        line = f"[{_stamp()}] {message}"
        print(line, flush=True)
        if log is not None:
            try:
                log.write(line + "\n")
            except OSError as exc:
                print(f"WARNING: log write failed ({type(exc).__name__}); console continues", file=sys.stderr, flush=True)
                try:
                    log.close()
                except OSError:
                    pass
                log = None

    process: subprocess.Popen[bytes] | None = None
    previous: dict[signal.Signals, Any] = {}
    received: list[int] = []
    forwarded = 0
    handled_signals = 0
    view = JournalView(directory)
    checkpoint = "checkpoint unavailable"
    start = last_progress = time.monotonic()
    next_heartbeat = start
    warned = False
    buffers: dict[str, bytes] = {"stdout": b"", "stderr": b""}

    def on_signal(signum: int, _frame: Any) -> None:
        received.append(signum)

    def send_once(sig: signal.Signals) -> bool:
        assert process is not None
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            emit("signal not forwarded: child exited before delivery")
            return False
        return True

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, on_signal)
        process = subprocess.Popen(command or [sys.executable, "-u", "-m", "utils.shorten_file", *child_args],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, bufsize=0)
        emit(f"START child_pid={process.pid} chunk={chunk:g}s log={log_path if log else 'unavailable'}")
        with selectors.DefaultSelector() as selector:
            for label, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
                assert pipe is not None
                selector.register(pipe, selectors.EVENT_READ, label)
            while selector.get_map() or process.poll() is None:
                now = time.monotonic()
                if handled_signals < len(received) and forwarded < 2 and process.poll() is None:
                    sig = received[handled_signals]
                    handled_signals += 1
                    if send_once(signal.Signals(sig)):
                        forwarded += 1
                        emit(f"signal={signal.Signals(sig).name} forwarded_to_child count={forwarded}; waiting for checkpoint")
                        if forwarded == 2:
                            emit("WARNING: cancellation during a request may leave unknown billing; inspect journal")
                elif now - start >= chunk and forwarded == 0 and process.poll() is None:
                    if send_once(signal.SIGINT):
                        forwarded = 1
                        emit("chunk deadline: first SIGINT forwarded; graceful stop before next request (current request/checkpoint may continue)")
                if now >= next_heartbeat:
                    try:
                        warnings = view.refresh()
                        for warning in warnings:
                            emit(warning)
                        if view.changed:
                            last_progress = now
                            warned = False
                        checkpoint = _checkpoint(directory, checkpoint)
                    except Exception as exc:
                        emit(f"WARNING: monitor error ({type(exc).__name__}); child continues")
                    emit(f"heartbeat elapsed={int(now-start)}s child_pid={process.pid} alive={process.poll() is None} {view.snapshot()} {checkpoint}")
                    if now - last_progress >= stall and not warned and process.poll() is None:
                        emit(f"WARNING: нет подтверждённого прогресса {int(now-last_progress)}с; процесс жив, причина неизвестна (сеть, replay или локальная обработка)")
                        warned = True
                    next_heartbeat = now + heartbeat
                for key, _ in selector.select(timeout=min(0.2, max(0, next_heartbeat - time.monotonic()))):
                    label = key.data
                    data = os.read(key.fd, 4096)
                    if not data:
                        selector.unregister(key.fileobj)
                        if buffers[label]:
                            emit(f"child/{label}: {buffers[label].decode('utf-8', 'replace')}")
                        continue
                    last_progress = time.monotonic()
                    warned = False
                    lines = (buffers[label] + data).split(b"\n")
                    buffers[label] = lines.pop()
                    for line in lines:
                        emit(f"child/{label}: {line.decode('utf-8', 'replace')}")
        exit_code = process.wait()
        try:
            for warning in view.refresh():
                emit(warning)
            checkpoint = _checkpoint(directory, checkpoint)
        except Exception as exc:
            emit(f"WARNING: final monitor error ({type(exc).__name__})")
        emit(f"END child_exit={exit_code} log={log_path if log else 'unavailable'} {view.snapshot()} {checkpoint}")
        return exit_code
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if log is not None:
            try:
                log.close()
            except OSError as exc:
                print(f"WARNING: log close failed ({type(exc).__name__}); check disk log", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One terminal chunk of unchanged utils.shorten_file; all other flags pass through")
    parser.add_argument("--chunk-seconds", type=float, default=600, help="soft stop request after this time (default 600)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args, rest = parser.parse_known_args(argv)
    if not 0 < args.chunk_seconds < float("inf"):
        parser.error("--chunk-seconds must be finite and positive")
    child = [*rest, "--output-dir", str(args.output_dir)]
    if ("--resume" not in child and not any(value == "--model" or value.startswith("--model=") for value in child)
            and not any(value == "--run-time-limit" or value.startswith("--run-time-limit=") for value in child)):
        child += ["--run-time-limit", str(args.chunk_seconds)]
    return _run(child, args.output_dir, chunk=args.chunk_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
