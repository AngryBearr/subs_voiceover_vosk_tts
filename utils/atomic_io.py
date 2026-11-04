"""Files that are never seen half written: a temporary file in the same folder, flushed to disk, then renamed.

After a crash or a power loss the target is either the old file or the new one, never a truncated one
(a plain ``write_text`` can leave an empty file behind, which "exists means done" caches then trust).
Standard library only: it is also imported by the F5 worker running in its own virtual environment.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path


def write_text_atomic(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Replace ``path`` with ``text`` in one step (the folder is created if missing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")  # two threads may write one file
    try:
        with tmp.open("w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def fsync_file(path: str | Path) -> None:
    """Flush a finished file to disk (a record written after it then never points at a file that is not there)."""
    try:
        with Path(path).open("rb") as handle:
            os.fsync(handle.fileno())
    except OSError:
        pass
