"""Run STT workers as subprocesses inside repo-local virtual environments."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.atomic_io import write_text_atomic

REPO_ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerSpec:
    """A worker's environment and entry point; isolated workers run their script with -I."""

    env: str
    module: str
    label: str = ""
    isolated: bool = False


ENV_WORKERS: dict[str, WorkerSpec] = {
    "f5": WorkerSpec("f5_env", "voiceover.f5_worker", "F5 worker"),
    "xvoice": WorkerSpec("xvoice_env", "voiceover.xvoice_worker", "X-Voice worker", isolated=True),
    "diarizen": WorkerSpec("diarizen_env", "stt.worker"),
    "lava": WorkerSpec("lava_env", "stt.widen", "LavaSR worker"),
}


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2))


def env_python(env: str) -> Path:
    """Interpreter of a repo-local venv (Linux or Windows layout)."""
    for candidate in (REPO_ROOT / env / "bin" / "python", REPO_ROOT / env / "Scripts" / "python.exe"):
        if candidate.exists():
            return candidate
    target = "main" if env == "subs_env" else env.removesuffix("_env")
    raise FileNotFoundError(f"Virtual environment {env!r} not found in {REPO_ROOT}; "
                            f"link it with `bash scripts/link_shared.sh` or run `bash scripts/setup_envs.sh {target}`")


def worker_python(spec: WorkerSpec, python: Path | None = None) -> Path:
    """Find the worker interpreter, respecting an explicit interpreter override."""
    if python is None:
        return env_python(spec.env)
    if not python.is_file():
        raise FileNotFoundError(f"{python} missing: create {spec.env} with "
                                f"`bash scripts/setup_envs.sh {spec.env.removesuffix('_env')}`")
    return python


def worker_command(spec: WorkerSpec, args: list[str], python: Path | None = None) -> list[str]:
    """Build the module command, or the isolated script command required by X-Voice."""
    command = [str(worker_python(spec, python))]
    if spec.isolated:
        command += ["-I", str(REPO_ROOT / (spec.module.replace(".", "/") + ".py"))]
    else:
        command += ["-m", spec.module]
    return [*command, *args]


def run_worker(spec: WorkerSpec, args: list[str], log_path: Path | None = None, *,
               python: Path | None = None, cwd: Path | None = None, env: dict[str, str] | None = None,
               capture_output: bool = False, combine_output: bool = False, check: bool = True,
               on_poll: Callable[[], None] | None = None, poll_s: float = 2.0) -> subprocess.CompletedProcess[str]:
    """Run a worker, keeping stderr in a log or capturing output; optionally poll live progress."""
    if capture_output and on_poll is not None:
        raise ValueError("capture_output cannot be combined with on_poll")
    if capture_output and log_path is not None:
        raise ValueError("capture_output cannot be combined with log_path")
    if combine_output and log_path is None:
        raise ValueError("combine_output requires log_path")
    command = worker_command(spec, args, python)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    # A file, not a pipe: a verbose model must not block while the parent polls progress.
    worker_log = log_path.open("w", encoding="utf-8") if log_path is not None else None
    try:
        kwargs: dict[str, Any] = {"cwd": REPO_ROOT if cwd is None else cwd, "env": env, "text": True}
        if worker_log is not None:
            kwargs.update(stderr=subprocess.STDOUT if combine_output else worker_log,
                          stdout=worker_log if combine_output else None)
        elif capture_output:
            kwargs["capture_output"] = True
        if on_poll is None:
            done = subprocess.run(command, check=False, **kwargs)
        else:
            process = subprocess.Popen(command, **kwargs)
            try:
                while True:
                    try:
                        process.wait(timeout=poll_s)
                        break
                    except subprocess.TimeoutExpired:
                        on_poll()
            except BaseException:  # Ctrl+C or a failed callback must not leave the worker running.
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
            done = subprocess.CompletedProcess(command, process.returncode)
    finally:
        if worker_log is not None:
            worker_log.close()
    if check and done.returncode:
        tail = (log_path.read_text(encoding="utf-8", errors="replace") if log_path is not None
                else done.stderr or "")[-2000:]
        location = f", see {log_path}" if log_path is not None else ""
        raise RuntimeError(f"{spec.label or spec.module} failed (exit {done.returncode}){location}: {tail}")
    return done


def run_module(env: str, module: str, args: list[str], log_path: Path) -> tuple[int, str]:
    """Run `python -m module args` in `env`; output goes to `log_path`.

    Returns (exit code, last log lines) so callers can report failures compactly.
    """
    environ = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    proc = run_worker(WorkerSpec(env, module), args, log_path, env=environ, combine_output=True, check=False)
    tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-5:]
    return proc.returncode, " | ".join(tail)
