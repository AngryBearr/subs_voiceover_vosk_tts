"""Timing records of ``voiceover.f5_worker`` (no torch: testable with fakes).

``LineTimer`` splits the seconds of one line into phases (sampling, with the reference loaded inside it, the cached take
read, the speech check, stress listening, the WAV write) and counts the attempts actually executed;
``summarize`` turns the per-line records into totals, means per line and an attempts histogram.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

PHASES = ("sampling", "take_read", "check", "stress", "write")


class LineTimer:
    """Seconds per phase of one line, the attempts executed and the speech re-checks of a cut take."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._start = clock()
        self.seconds: dict[str, float] = dict.fromkeys(PHASES, 0.0)
        self.attempts = 0
        self.cut_rechecks = 0
        self.stress_listened = False

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        began = self._clock()
        try:
            yield
        finally:
            self.seconds[name] += self._clock() - began

    def record(self) -> dict[str, Any]:
        total = self._clock() - self._start
        return {"attempts_run": self.attempts, "cut_rechecks": self.cut_rechecks,
                "stress_listened": self.stress_listened,
                "seconds": {**{name: round(value, 4) for name, value in self.seconds.items()},
                            "total": round(total, 4)}}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals and mean per line of every phase, attempts histogram, lines heard for stress; ``records`` are
    the ``LineTimer.record()`` dicts of the finished lines."""
    names = (*PHASES, "total")
    totals = {name: sum(float(r["seconds"].get(name, 0.0)) for r in records) for name in names}
    lines = len(records)
    other = totals["total"] - sum(totals[name] for name in PHASES)
    return {"lines": lines,
            "seconds_total": {**{name: round(totals[name], 2) for name in names}, "other": round(other, 2)},
            "seconds_mean_per_line": {**{name: round(totals[name] / lines, 3) if lines else 0.0 for name in names},
                                      "other": round(other / lines, 3) if lines else 0.0},
            "attempts_histogram": {str(n): c for n, c in sorted(Counter(int(r["attempts_run"]) for r in records).items())},
            "attempts_total": sum(int(r["attempts_run"]) for r in records),
            "lines_with_stress_listening": sum(bool(r["stress_listened"]) for r in records),
            "cut_rechecks": sum(int(r["cut_rechecks"]) for r in records)}


def format_summary(summary: dict[str, Any]) -> str:
    """One log line: mean seconds per line by phase, attempts histogram."""
    mean = summary["seconds_mean_per_line"]
    phases = ", ".join(f"{name} {mean[name]:.2f}" for name in (*PHASES, "other", "total"))
    return (f"f5 timing over {summary['lines']} lines, mean s per line: {phases}; attempts {summary['attempts_histogram']}"
            f" ({summary['attempts_total']} in all); stress listening in {summary['lines_with_stress_listening']} lines")
