"""Compatibility wrapper for subtitle text analysis.

Project historically uses `utils.analyze_text` for subtitle timing/text-length analysis.
The pipeline spec refers to `utils.text_analysis`, so this module provides a stable
import path and small convenience helpers.

The analysis attaches an `analysis` dict to every subtitle item.
"""

from __future__ import annotations

from typing import Any, Dict, List

from utils.analyze_text import (  # noqa: F401
    DEFAULT_AVG_CHARS_PER_SEC,
    DEFAULT_MAX_MISMATCH_RATIO,
    analyze_subtitles,
    join_text_lines,
)


def analyze_subtitles_items(
    items: List[Dict[str, Any]],
    *,
    avg_chars_per_sec: float = DEFAULT_AVG_CHARS_PER_SEC,
    max_mismatch_ratio: float = DEFAULT_MAX_MISMATCH_RATIO,
) -> Dict[str, Any]:
    """Analyze subtitles in-memory and return the full analysis result.

    Returns a dict:
        {
            "items": items_with_analysis,
            "checked_count": int,
            "critical_items": list,
        }

    Notes:
        - Mutates `items` in-place (same as `utils.analyze_text.analyze_subtitles`).
        - Each item gets `item["analysis"]`.
    """

    return analyze_subtitles(
        items,
        avg_chars_per_sec=float(avg_chars_per_sec),
        max_mismatch_ratio=float(max_mismatch_ratio),
    )
