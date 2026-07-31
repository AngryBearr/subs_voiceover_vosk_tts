"""Immutable domain primitives for subtitle shortening timing and budgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class TimingWindow:
    """Stable base and effective timing derived from one subtitle item."""

    base_duration_sec: float
    effective_duration_sec: float
    is_checked: bool
    has_source_timing: bool

    @classmethod
    def from_item(cls, item: Mapping[str, Any]) -> "TimingWindow":
        """Derive timing without mutating the source item."""
        analysis = item.get("analysis", {})
        base = analysis.get("available_duration_sec")
        effective = analysis.get("effective_duration_sec")
        estimated = analysis.get("estimated_sec")

        if not isinstance(base, (int, float)) or base <= 0:
            duration = analysis.get("duration_sec")
            ratio = analysis.get("mismatch_ratio")
            if isinstance(duration, (int, float)) and duration > 0:
                base = float(duration)
            elif isinstance(estimated, (int, float)) and isinstance(ratio, (int, float)) and ratio > 0:
                base = float(estimated) / float(ratio)
            else:
                base = max(0.0, (item.get("end", 0) - item.get("start", 0)) / 1000.0)

        if not isinstance(effective, (int, float)) or effective <= 0:
            extended = analysis.get("extended_duration_sec")
            extended_ratio = analysis.get("extended_mismatch_ratio")
            if isinstance(extended, (int, float)) and extended > 0:
                effective = float(extended)
            elif isinstance(estimated, (int, float)) and isinstance(extended_ratio, (int, float)) and extended_ratio > 0:
                effective = float(estimated) / float(extended_ratio)
            else:
                effective = base

        start = item.get("start", 0)
        end = item.get("end", 0)
        return cls(
            base_duration_sec=float(base),
            effective_duration_sec=float(effective),
            is_checked=bool(analysis.get("is_checked")),
            has_source_timing=(end - start) > 0,
        )

    def as_analysis_fields(self) -> dict[str, float]:
        """Return the legacy analysis fields represented by this window."""
        return {
            "available_duration_sec": self.base_duration_sec,
            "effective_duration_sec": self.effective_duration_sec,
        }


class BudgetEstimator(Protocol):
    """Estimate a generation character budget from a timing window."""

    def max_chars(self, window: TimingWindow, target_ratio: float) -> int:
        """Return the maximum generation character count."""


@dataclass(frozen=True)
class CharacterRateBudgetEstimator:
    """Approximate max_chars generation hint based on a character rate."""

    avg_chars_per_sec: float

    def max_chars(self, window: TimingWindow, target_ratio: float) -> int:
        """Calculate the stable character budget using the legacy formula."""
        return max(1, int(window.effective_duration_sec * self.avg_chars_per_sec * target_ratio))
