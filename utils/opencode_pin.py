"""Which OpenCode binary the pipeline runs, and a pinned copy of it that an OpenCode update cannot change.

The user's interactive ``opencode`` updates itself at any time and a new version may change the HTTP API or the model
list. The pipeline therefore prefers its own copy in the shared folder, ``<shared>/tools/opencode/current/opencode``
(``current`` is a symlink to ``<version>``). Upgrading is explicit and checked: ``python -m utils.opencode_pin`` copies
the user's current binary, starts a private server with the copy, sends one tiny request per model, and switches
``current`` only when all of them were answered.

    python -m utils.opencode_pin --status
    python -m utils.opencode_pin [--from PATH] [--models m1,m2] [--dry-run]

Binary resolution order: env ``STT_OPENCODE_BIN``, the pinned copy, ``opencode`` on PATH, ``~/.opencode/bin/opencode``.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

BIN_ENV = "STT_OPENCODE_BIN"
DEFAULT_CHECK_MODELS = ("openai/gpt-6.1-sol", "openai/gpt-6-luna")
_VERSION_RE = re.compile(r"\d+\.\d+\.\d+[\w.+-]*")
_LOGGED: set[str] = set()


def shared_root() -> Path:
    """The shared folder (``$SUBS_SHARED`` or ``~/subs_voiceover_shared``)."""
    return Path(os.environ.get("SUBS_SHARED") or Path.home() / "subs_voiceover_shared")


def tools_dir() -> Path:
    return shared_root() / "tools" / "opencode"


def pinned_binary() -> Path:
    return tools_dir() / "current" / "opencode"


def user_binary() -> Optional[Path]:
    """The user's own ``opencode``: on PATH, else ``~/.opencode/bin/opencode``."""
    found = shutil.which("opencode")
    if found:
        return Path(found)
    home = Path.home() / ".opencode" / "bin" / "opencode"
    return home if home.is_file() and os.access(home, os.X_OK) else None


def _usable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def resolve_binary() -> tuple[Optional[Path], str]:
    """``(path, source)`` of the binary the pipeline runs; source is ``env``, ``pinned``, ``PATH`` or ``home``."""
    env = os.environ.get(BIN_ENV)
    if env:
        if _usable(Path(env)):
            return Path(env), "env"
        log.warning("%s=%s is not an executable file, ignored", BIN_ENV, env)
    pinned = pinned_binary()
    if _usable(pinned):
        return pinned, "pinned"
    found = shutil.which("opencode")
    if found:
        return Path(found), "PATH"
    home = Path.home() / ".opencode" / "bin" / "opencode"
    if _usable(home):
        return home, "home"
    return None, "missing"


def binary_version(path: Path) -> str:
    """The version printed by ``<path> --version`` (``unknown`` when it cannot be read)."""
    try:
        out = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=20, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    match = _VERSION_RE.search(out)
    return match.group(0) if match else "unknown"


def describe() -> tuple[str, bool]:
    """``(text, pinned)``: which binary and version the pipeline will use, and whether it is a pinned copy."""
    path, source = resolve_binary()
    if path is None:
        return "OpenCode binary not found", False
    return f"{path} (v{binary_version(path)}, {source})", source in {"pinned", "env"}


def note_binary(path: Path, source: str) -> None:
    """Log the binary once per process; warn when it is not a pinned copy."""
    if str(path) in _LOGGED:
        return
    _LOGGED.add(str(path))
    log.info("OpenCode binary: %s (v%s, %s)", path, binary_version(path), source)
    if source not in {"pinned", "env"}:
        log.warning("OpenCode is not pinned: the pipeline follows whatever version is installed. Pin one with "
                    "`python -m utils.opencode_pin`")


def installed_versions() -> list[str]:
    root = tools_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.is_symlink() and _usable(p / "opencode"))


def current_version() -> Optional[str]:
    link = tools_dir() / "current"
    return Path(os.readlink(link)).name if link.is_symlink() else None


def switch_current(version: str) -> None:
    """Point ``current`` at ``version`` atomically (a symlink is renamed over the old one)."""
    root = tools_dir()
    tmp = root / f".current.{os.getpid()}"
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(version)
    os.replace(tmp, root / "current")


def check_binary(binary: Path, models: tuple[str, ...]) -> Optional[str]:
    """Real check of ``binary``: private server, warm-up, one tiny request per model. ``None`` when all passed."""
    from utils.opencode_clean import CleanOpenCodeServer, OpenCodeError

    try:
        with CleanOpenCodeServer(required_models={}, timeout=60.0, binary=str(binary)) as server:
            for model in models:
                server.require_models({model: set()})
                server.warm_up(model)
                answer = server.generate("Reply with OK", model, None, timeout=120)
                if not answer.strip():
                    return f"{model}: empty answer"
    except Exception as exc:  # noqa: BLE001 - any failure means "do not switch"
        kind = "request failed" if isinstance(exc, OpenCodeError) else type(exc).__name__
        return f"{kind}: {exc}"
    return None


def pin(source: Optional[Path], models: tuple[str, ...], dry_run: bool = False) -> int:
    """Copy ``source`` into the shared folder, check it, switch ``current``. Returns the exit code."""
    source = source or user_binary()
    if source is None or not _usable(source):
        print("No OpenCode binary to pin: give --from PATH, or install opencode.")
        return 2
    source = source.resolve()
    version = binary_version(source)
    if version == "unknown":
        print(f"Cannot read the version of {source}; not pinning.")
        return 2
    target = tools_dir() / version / "opencode"
    print(f"Binary {source}: version {version}; will be kept as {target}")
    if dry_run:
        print(f"Dry run: would copy it, check {', '.join(models)} and switch `current` to {version}.")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_bytes() != source.read_bytes():
        tmp = target.with_name(f".opencode.{os.getpid()}")
        shutil.copy2(source, tmp)
        os.replace(tmp, target)
    target.chmod(0o755)
    print(f"Copied. Checking with a private server: {', '.join(models)} (up to a few minutes) ...")
    problem = check_binary(target, models)
    old = current_version()
    if problem is not None:
        print(f"Check FAILED: {problem}")
        print(f"`current` stays at {old or 'nothing (the pipeline uses the installed opencode)'}. "
              f"The copy {version} stays in {target.parent} but is not used.")
        return 1
    switch_current(version)
    print(f"Check passed. `current` now points to {version} (was {old or 'nothing'}). The pipeline uses {pinned_binary()}.")
    return 0


def status() -> int:
    path, source = resolve_binary()
    print(f"shared folder:     {shared_root()}")
    print(f"pipeline uses:     {f'{path} (v{binary_version(path)}, {source})' if path else 'nothing found'}")
    print(f"pinned `current`:  {current_version() or 'none'}")
    print(f"pinned versions:   {', '.join(installed_versions()) or 'none'}")
    user = user_binary()
    print(f"user's opencode:   {f'{user} (v{binary_version(user)})' if user else 'not found'}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pin the OpenCode binary the pipeline uses (checked before switching).")
    parser.add_argument("--from", dest="source", type=Path, help="binary to pin (default: the user's opencode)")
    parser.add_argument("--models", default=",".join(DEFAULT_CHECK_MODELS), help="models the check must get answers from")
    parser.add_argument("--dry-run", action="store_true", help="only show what would be done")
    parser.add_argument("--status", action="store_true", help="show the pinned and installed versions")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.status:
        return status()
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    return pin(args.source, models, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
