"""Background job records and process checks, independent of the page state."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from stt import progress as stt_progress

DEFAULT_TITLE = "Обработка серий"
TASK_MODULES = (b"stt.run", b"stt.voice_bank", b"stt.video_compress")  # programs the runner starts
RESUMABLE = ("прервано", "остановлено")


@dataclass
class Job:
    """A background task of the page (one at a time): ``stt.run`` for ``files`` (``cmd`` is None), or any other
    command (``cmd``: the voice clips, the video compression). Every task has a Russian ``title`` for the status strip."""

    files: list[Path]
    out_root: Path
    engines: list[str]
    config_path: Path
    log_path: Path
    extra: list[str] = field(default_factory=list)
    process: subprocess.Popen[bytes] | None = None
    engine: str | None = None
    state: str = "идёт"
    started: float = field(default_factory=time.time)
    finished: float | None = None
    pid: int | None = None  # of the program: it keeps running when this page is restarted
    boot: str = ""
    title: str = DEFAULT_TITLE
    cmd: list[str] | None = None  # None: ``stt.run`` for files / engines / extra
    result: str = ""  # a file the task makes (shown after "Готово")

    @property
    def is_stt_run(self) -> bool:
        return self.cmd is None


def _is_task_process(pid: int) -> bool:
    """The process ``pid`` is one of the programs of the runner (a process id can be reused by something else after
    a restart)."""
    try:
        return any(module in TASK_MODULES for module in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0"))
    except OSError:
        return False


def job_alive(job: Job | None) -> bool:
    """The program of the job still runs (started by this page, or found again after a restart)."""
    if job is None:
        return False
    if job.process is not None:
        return job.process.poll() is None
    return bool(job.pid) and stt_progress.process_alive({"pid": job.pid, "boot": job.boot}) and _is_task_process(job.pid)


def _gone_state(job: Job) -> str:
    """State of a job whose process is gone and whose exit code nobody saw: all mixes made -> done, else interrupted."""
    if not job.is_stt_run:
        return "прервано"
    complete = all((job.out_root / m.stem / ".stages" / f"mix.{engine}").exists()
                   for m in job.files for engine in job.engines)
    return "готово" if complete and job.files else "прервано"
