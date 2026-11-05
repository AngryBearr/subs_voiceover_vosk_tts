"""F5 worker instrumentation: timer and summary (fakes, no models)."""

from __future__ import annotations

from typing import Any

from voiceover.f5_stats import LineTimer, format_summary, summarize


def test_line_timer_splits_phases_and_counts() -> None:
    now = [0.0]
    timer = LineTimer(clock=lambda: now[0])
    for seconds in (0.5, 0.25):  # two attempts
        timer.attempts += 1
        with timer.phase("sampling"):
            now[0] += seconds
    with timer.phase("check"):
        now[0] += 0.1
    now[0] += 0.05  # unaccounted
    record = timer.record()
    assert record["attempts_run"] == 2
    assert record["seconds"]["sampling"] == 0.75 and record["seconds"]["check"] == 0.1
    assert record["seconds"]["stress"] == 0.0 and record["seconds"]["total"] == 0.9


def test_summary_totals_means_and_histogram() -> None:
    def record(attempts: int, sampling: float, stress: bool) -> dict[str, Any]:
        return {"attempts_run": attempts, "cut_rechecks": 1, "stress_listened": stress,
                "seconds": {"sampling": sampling, "check": 0.2, "total": sampling + 0.3}}

    summary = summarize([record(1, 1.0, False), record(3, 3.0, True), record(1, 1.0, True)])
    assert summary["lines"] == 3 and summary["attempts_histogram"] == {"1": 2, "3": 1}
    assert summary["attempts_total"] == 5 and summary["lines_with_stress_listening"] == 2
    assert summary["seconds_total"]["sampling"] == 5.0 and summary["seconds_total"]["other"] == 0.3
    assert summary["seconds_mean_per_line"]["check"] == 0.2 and summary["cut_rechecks"] == 3
    assert "3 lines" in format_summary(summary)
    assert summarize([])["lines"] == 0
