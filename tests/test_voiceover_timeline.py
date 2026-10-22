from __future__ import annotations

import pytest

from voiceover.timeline import Line, TimelinePolicy, schedule

POLICY = TimelinePolicy()


def _by_id(placements):  # type: ignore[no-untyped-def]
    return {p.cue_id: p for p in placements}


def test_line_before_silence_is_untouched_even_if_it_overflows_its_subtitle() -> None:
    placements = schedule([Line("a", 1000, 4.0), Line("b", 10000, 1.0)], POLICY)
    result = _by_id(placements)
    assert result["a"].status == "fits"
    assert result["a"].removed_sec == 0
    assert result["a"].tempo_factor == 1.0
    assert result["b"].status == "fits"


def test_small_overlap_is_solved_by_shift_and_tempo_before_text() -> None:
    # a: 2.6 s of speech, b starts 2.0 s later -> 0.75 s short including the gap.
    placements = schedule([Line("a", 1000, 2.6), Line("b", 3000, 0.8)], POLICY)
    result = _by_id(placements)
    assert result["a"].removed_sec == 0
    assert result["b"].removed_sec == 0
    assert result["a"].end_ms + POLICY.gap_ms <= result["b"].start_ms
    assert result["a"].start_ms >= 1000 - POLICY.lead_ms
    assert result["b"].start_ms <= 3000 + POLICY.lag_ms
    assert result["a"].tempo_factor <= POLICY.comfort_factor + 1e-9


def test_large_overflow_requires_text_and_reports_target() -> None:
    placements = schedule([Line("a", 1000, 6.0), Line("b", 3000, 3.0), Line("c", 6000, 3.0)], POLICY)
    result = _by_id(placements)
    needs = [p for p in placements if p.status == "needs_text"]
    assert needs, "some line must lose text"
    for p in placements:
        assert p.collision_ms == 0
    for p in needs:
        assert p.target_base_sec == pytest.approx(p.spoken_sec * p.tempo_factor)
        assert p.spoken_sec >= POLICY.min_speech_sec - 1e-6
    # The chain must respect gaps in time order.
    ordered = sorted(placements, key=lambda p: p.start_ms)
    for left, right in zip(ordered, ordered[1:]):
        assert left.end_ms + POLICY.gap_ms <= right.start_ms + 1


def test_ceiling_is_used_only_when_comfort_is_not_enough() -> None:
    policy = TimelinePolicy(weight_text=1000.0)
    # a cannot start earlier than 0; b may start at most at 3500; a must end by 3350:
    # factor 4200/3350 = 1.254 (> comfort 1.2, < ceiling 1.28).
    placements = schedule([Line("a", 0, 4.2), Line("b", 3000, 1.0)], policy)
    result = _by_id(placements)
    assert result["a"].removed_sec == 0
    assert result["a"].tempo_factor > policy.comfort_factor
    assert result["a"].tempo_factor <= policy.ceiling_factor + 1e-9
    assert result["a"].status == "over_comfort"


def test_collision_is_reported_when_nothing_else_is_possible() -> None:
    policy = TimelinePolicy(min_speech_sec=2.0)
    placements = schedule([Line("a", 0, 2.0), Line("b", 100, 2.0)], policy)
    result = _by_id(placements)
    assert result["b"].collision_ms > 0
    assert result["b"].status == "collision"


def test_output_order_matches_input_order_and_validates() -> None:
    placements = schedule([Line("later", 5000, 1.0), Line("earlier", 1000, 1.0)], POLICY)
    assert [p.cue_id for p in placements] == ["later", "earlier"]
    with pytest.raises(ValueError):
        schedule([Line("x", 0, 0.0)], POLICY)
    with pytest.raises(ValueError):
        TimelinePolicy(base_rate=2.0, tempo_comfort_total=1.5)


def test_tiny_collision_is_not_cheaper_than_a_small_shift() -> None:
    # Without a fixed penalty a 1 ms overlap would cost less than any shift or removal.
    placements = schedule([Line("a", 0, 2.0), Line("b", 2100, 1.0)], POLICY)
    assert all(p.collision_ms == 0 for p in placements)
