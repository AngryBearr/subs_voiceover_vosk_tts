"""
High-quality audio resampling utilities.

Uses SciPy's polyphase resampler when available.
"""
from __future__ import annotations

from typing import Optional
import math
import numpy as np


class ResampleUnavailableError(RuntimeError):
    """Raised when SciPy-based resampling is requested but SciPy is not installed."""


def resample_int16_scipy(audio_i16: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """
    Resample mono int16 PCM from `from_sr` to `to_sr` using SciPy's resample_poly.

    Args:
        audio_i16: int16 numpy array (mono) at from_sr.
        from_sr: source sample rate (Hz)
        to_sr: target sample rate (Hz)

    Returns:
        int16 numpy array resampled to `to_sr`.

    Raises:
        ResampleUnavailableError: if SciPy is not installed.
    """
    if from_sr == to_sr:
        return audio_i16

    try:
        from scipy.signal import resample_poly  # type: ignore
    except Exception as e:
        raise ResampleUnavailableError(
            "SciPy is required for resampling but is not installed."
        ) from e

    # Convert to float [-1, 1]
    x = audio_i16.astype(np.float32) / 32767.0

    # Use rational up/down derived from gcd for stability/perf
    g = math.gcd(from_sr, to_sr)
    up = to_sr // g
    down = from_sr // g

    y = resample_poly(x, up=up, down=down)

    # Back to int16 with clipping
    y = np.clip(y, -1.0, 1.0)
    return (y * 32767.0).astype(np.int16)
