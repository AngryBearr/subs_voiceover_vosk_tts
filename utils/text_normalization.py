"""Compatibility wrapper for RUNorm + RUAccent subtitle normalization.

The pipeline spec refers to `utils.text_normalization`.
The implementation lives in `utils.text_normalizer`.

This module provides a stable import path + a small helper to run the full
normalization pipeline on an in-memory list.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ruaccent import RUAccent  # type: ignore
from runorm import RUNorm  # type: ignore

from utils.text_normalizer import (  # noqa: F401
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEPARATOR,
    _load_runorm,
    _load_ruaccent,
    normalize_and_accent_subtitles,
)


def normalize_subtitles_items(
    items: List[Dict[str, Any]],
    *,
    normalizer: Optional[RUNorm] = None,
    accentizer: Optional[RUAccent] = None,
    norm_device: str = "cpu",
    accent_device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    separator: str = DEFAULT_SEPARATOR,
) -> List[Dict[str, Any]]:
    """RUNorm + RUAccent normalization for subtitle items.

    Preserves the existing subtitle schema and keeps `text` as `list[str]`.

    Args:
        items: subtitle entries.
        normalizer/accentizer: optional preloaded models (recommended for batch).
        norm_device/accent_device: used only if models are not provided.
    """

    if normalizer is None:
        normalizer = _load_runorm(device=norm_device)
    if accentizer is None:
        accentizer = _load_ruaccent(device=accent_device)

    return normalize_and_accent_subtitles(
        items,
        normalizer,
        accentizer,
        batch_size=int(batch_size),
        separator=str(separator),
    )
