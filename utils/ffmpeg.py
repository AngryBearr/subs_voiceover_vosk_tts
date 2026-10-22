"""Locate an ffmpeg binary: PATH, the running virtualenv, or the one bundled with imageio-ffmpeg."""

from __future__ import annotations

import shutil
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """Path of an ffmpeg executable; raises ``FileNotFoundError`` when there is none."""
    for candidate in (shutil.which("ffmpeg"), str(Path(sys.executable).parent / "ffmpeg")):
        if candidate and Path(candidate).is_file():
            return candidate
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, RuntimeError) as error:
        raise FileNotFoundError("ffmpeg not found: install it or `uv pip install imageio-ffmpeg`") from error
