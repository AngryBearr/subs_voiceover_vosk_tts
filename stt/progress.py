"""Progress of the stages of ``stt.run`` per episode, written for the UI: ``<episode>/.progress/<stage>[.<engine>].json``.

``stt.run`` wraps every stage in ``track(folder, stage, engine)``; code inside the stage reports how far it is with
``tick(done, total, unit)``. The file is replaced atomically, so a reader never sees half a file, and records the
process id: a ``running`` file whose process is gone means the run was interrupted there.

    {"stage": "voiceover", "engine": "f5", "state": "running", "done": 120, "total": 640, "unit": "lines",
     "started": 1793650000.0, "updated": 1793650300.0, "finished": null, "pid": 12345, "boot": "<boot id>"}

``state`` is ``running``, ``done`` or ``failed``; ``done``/``total``/``unit`` are null while a stage has no count.
``RESUME_NOTES`` says, per stage, what a stop in the middle of it keeps and what is redone.

Stages that ``stt.run`` starts as separate programs (stress, voiceover, check) inherit the tracker through the
environment variable ``ENV_VAR``: their ``tick`` calls update the same file, which keeps the pid of the main process.
``partial_dir`` is the per-episode folder where an unfinished stage keeps what it has already paid for.
"""

from __future__ import annotations

import contextvars
import json
import os
import shutil
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.atomic_io import write_text_atomic

DIR = ".progress"
PARTIAL_DIR = ".partial"  # <episode>/.partial/<kind>/<stage>[.<engine>]/: kept only while the stage is unfinished
ENV_VAR = "STT_PROGRESS"  # JSON of the running tracker, for programs started by a stage
MIN_INTERVAL_S = 2.0  # ticks closer than this are not written (the last one of a stage always is)
UNITS = ("lines", "parts", "requests", "steps", "clips")

# What a stop in the middle of a stage keeps (shown by the UI next to the running stage)
RESUME_NOTES: dict[str, str] = {
    "transcribe": "законченные шаги (голоса, распознавание речи и т.д.) сохранены, незаконченный шаг начнётся заново",
    "glossary": "полученный ответ языковой модели сохранён и повторно не оплачивается; запрос без ответа повторится",
    "speakers": "полученный ответ языковой модели сохранён и повторно не оплачивается; запрос без ответа повторится",
    "translate": "полученные ответы языковой модели сохранены и повторно не оплачиваются; запросы без ответа повторятся",
    "dub": "этап начнётся заново (он быстрый)",
    "stress": "полученные ответы языковых моделей сохранены и повторно не оплачиваются; запросы без ответа повторятся",
    "voiceover": "готовые реплики сохранены, озвучка продолжится с незаконченной",
    "check": "проверка начнётся заново (несколько минут)",
    "mix": "сведение начнётся заново (несколько минут)",
}


def boot_id() -> str:
    """Changes with every start of the computer: a process id of an earlier boot means nothing now."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def progress_path(folder: Path, stage: str, engine: str | None = None) -> Path:
    return folder / DIR / (f"{stage}.{engine}.json" if engine else f"{stage}.json")


def _write(path: Path, data: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(data, ensure_ascii=False))


@dataclass
class Tracker:
    """The progress file of one running stage."""

    folder: Path
    stage: str
    engine: str | None = None
    started: float = field(default_factory=time.time)
    done: int | None = None
    total: int | None = None
    unit: str | None = None
    owner_pid: int | None = None  # the main process, when this tracker lives in a program it started
    _written: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)  # ticks come from several threads

    @property
    def path(self) -> Path:
        return progress_path(self.folder, self.stage, self.engine)

    def record(self, state: str, finished: float | None = None) -> dict[str, Any]:
        return {"stage": self.stage, "engine": self.engine, "state": state, "done": self.done, "total": self.total,
                "unit": self.unit, "started": self.started, "updated": time.time(), "finished": finished,
                "pid": self.owner_pid or os.getpid(), "boot": boot_id()}

    def write(self, state: str = "running", finished: float | None = None) -> None:
        _write(self.path, self.record(state, finished))
        self._written = time.time()

    def partial_dir(self, kind: str) -> Path:
        """Where this stage keeps unfinished work of ``kind`` ("llm"); see ``clear_partial``."""
        name = f"{self.stage}.{self.engine}" if self.engine else self.stage
        return self.folder / PARTIAL_DIR / kind / name

    def environ(self) -> dict[str, str]:
        """Environment for a program started by this stage: its ``tick`` calls update this stage's file."""
        return {ENV_VAR: json.dumps({"folder": str(self.folder), "stage": self.stage, "engine": self.engine,
                                     "started": self.started, "pid": self.owner_pid or os.getpid()})}

    def adopt_counts(self) -> None:
        """Take over the counts a started program wrote to the file (before the final write of the stage)."""
        try:
            record = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(record, dict) and self.done is None and record.get("done") is not None:
            self.done, self.total, self.unit = record.get("done"), record.get("total"), record.get("unit")

    def update(self, done: int, total: int | None = None, unit: str | None = None, *, force: bool = False) -> None:
        with self._lock:
            self.done = done
            if total is not None:
                self.total = total
            if unit is not None:
                self.unit = unit
            last = self.total is not None and done >= self.total
            if force or last or time.time() - self._written >= MIN_INTERVAL_S:
                self.write()


_CURRENT: contextvars.ContextVar[Tracker | None] = contextvars.ContextVar("stt_progress", default=None)


@contextmanager
def track(folder: Path, stage: str, engine: str | None = None) -> Iterator[Tracker]:
    """Mark ``stage`` running for the code inside; ``done`` when it returns, ``failed`` when it raises."""
    tracker = Tracker(Path(folder), stage, engine)
    tracker.write()
    token = _CURRENT.set(tracker)
    try:
        yield tracker
    except BaseException:
        tracker.adopt_counts()
        tracker.write("failed", finished=time.time())
        raise
    else:
        tracker.adopt_counts()
        tracker.write("done", finished=time.time())
    finally:
        _CURRENT.reset(token)


_ENV_TRACKERS: dict[str, Tracker | None] = {}  # parsed ``ENV_VAR`` values (one tracker per value, so the throttle works)


def _from_environment() -> Tracker | None:
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return None
    if raw not in _ENV_TRACKERS:
        tracker = None
        try:
            data = json.loads(raw)
            tracker = Tracker(Path(data["folder"]), str(data["stage"]), data.get("engine"),
                              started=float(data.get("started") or time.time()),
                              owner_pid=int(data["pid"]) if data.get("pid") else None)
        except (ValueError, KeyError, TypeError, AttributeError):
            pass
        _ENV_TRACKERS[raw] = tracker
    return _ENV_TRACKERS[raw]


def current() -> Tracker | None:
    """The tracker of the stage running in this thread (``contextvars``: copy the context into worker threads),
    else the one a parent ``stt.run`` stage handed down through the environment."""
    return _CURRENT.get() or _from_environment()


def partial_dir(kind: str) -> Path | None:
    """Folder for unfinished work of the current stage (None outside ``track``)."""
    tracker = current()
    return tracker.partial_dir(kind) if tracker is not None else None


def clear_partial(folder: Path, stage: str, engine: str | None = None) -> None:
    """Drop what a stage kept for a restart: it finished, so the next run must not reuse it."""
    name = f"{stage}.{engine}" if engine else stage
    root = Path(folder) / PARTIAL_DIR
    if root.is_dir():
        for kind in root.iterdir():
            shutil.rmtree(kind / name, ignore_errors=True)
            try:
                kind.rmdir()
            except OSError:
                pass
        try:
            root.rmdir()
        except OSError:
            pass


def tick(done: int, total: int | None = None, unit: str | None = None) -> None:
    """Report progress of the current stage; nothing happens outside ``track``."""
    tracker = current()
    if tracker is not None:
        tracker.update(done, total, unit)


def read_all(folder: Path) -> list[dict[str, Any]]:
    """Every progress record of an episode folder, oldest start first; unreadable files are skipped."""
    records = []
    for path in sorted((Path(folder) / DIR).glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and "stage" in record:
            records.append(record)
    return sorted(records, key=lambda r: float(r.get("started") or 0))


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_alive(record: dict[str, Any]) -> bool:
    """The process that wrote ``record`` (with ``pid`` and ``boot``) still runs."""
    return record.get("boot", boot_id()) == boot_id() and pid_alive(record.get("pid"))


def interrupted(record: dict[str, Any]) -> bool:
    """A stage recorded as running whose process is gone (stopped, crashed, computer restarted)."""
    return record.get("state") == "running" and not process_alive(record)
