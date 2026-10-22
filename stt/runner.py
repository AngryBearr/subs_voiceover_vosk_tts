"""Run STT workers as subprocesses inside repo-local virtual environments."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger(__name__)


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def env_python(env: str) -> Path:
    """Interpreter of a repo-local venv (Linux or Windows layout)."""
    for candidate in (REPO_ROOT / env / "bin" / "python", REPO_ROOT / env / "Scripts" / "python.exe"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Virtual environment {env!r} not found in {REPO_ROOT}")


def run_module(env: str, module: str, args: list[str], log_path: Path) -> tuple[int, str]:
    """Run `python -m module args` in `env`; output goes to `log_path`.

    Returns (exit code, last log lines) so callers can report failures compactly.
    """
    python = env_python(env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environ = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            [str(python), "-m", module, *args],
            cwd=REPO_ROOT,
            env=environ,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-5:]
    return proc.returncode, " | ".join(tail)
