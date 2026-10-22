from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.phase4_fixtures import TARGET, add_manual_row, build_world
from voiceover.overrides import Overrides
from voiceover.resolve import Resolution, resolve


def _run(world: Any, overrides: Overrides | None = None) -> Resolution:
    return resolve(world.units, world.rows0, world.rows1, world.verdicts, world.ledger, world.placements0,
                   world.policy, world.oracle, world.cues, world.fit_slack_sec, overrides)


def _choice(resolution: Resolution, uid: str = "unit-0") -> dict[str, Any]:
    return next(c for c in resolution.choices if c["unit_id"] == uid)


def _no_collisions(resolution: Resolution) -> None:
    assert all(p["collision_ms"] == 0 for p in resolution.placements)


def test_schedule0_target_is_what_the_fixture_assumes(tmp_path: Path) -> None:
    world = build_world(tmp_path, {})
    assert abs(world.placements0[0]["target_base_sec"] - TARGET) < 1e-3


def test_pass_one_picks_first_measured_fit_not_the_estimate_choice(tmp_path: Path) -> None:
    world = build_world(tmp_path, {1: 4.6, 2: 4.2, 3: 3.0, 4: 2.0, 5: 1.0}, estimate_choice=3)
    resolution = _run(world)
    choice = _choice(resolution)
    assert choice["chosen_rung"] == 2 and choice["status"] == "resolved" and choice["pass"] == 1
    assert choice["chosen_text"] == world.rows1[1]["text"] and choice["chosen_speech_sec"] == 4.2
    assert [t["fits"] for t in choice["rungs_tried"]] == [False, True, True, True, True]
    assert len(resolution.passes) == 1  # nothing residual: pass 2 is skipped
    placement = next(p for p in resolution.placements if p["unit_id"] == "unit-0")
    assert placement["text_changed"] and placement["chosen_rung"] == 2 and placement["removed_sec"] <= 0.05
    _no_collisions(resolution)
    assert [c["unit_id"] for c in resolution.choices] == ["unit-0"]  # only ledger units get a choice
    assert not next(p for p in resolution.placements if p["unit_id"] == "unit-1")["text_changed"]


def test_no_fit_picks_the_shortest_usable_rung_as_unresolved(tmp_path: Path) -> None:
    resolution = _run(build_world(tmp_path, {1: 5.0, 2: 4.9, 3: 4.5, 4: 4.4, 5: 4.8}, status="unresolved"))
    choice = _choice(resolution)
    assert choice["status"] == "unresolved" and choice["chosen_rung"] == 4 and choice["pass"] == 1
    _no_collisions(resolution)


def test_verify_failed_rungs_are_skipped(tmp_path: Path) -> None:
    resolution = _run(build_world(tmp_path, {1: 4.6, 2: 4.2, 3: 3.0}, ok={2: False}))
    choice = _choice(resolution)
    assert choice["chosen_rung"] == 3 and choice["status"] == "resolved"
    assert [t["verify_ok"] for t in choice["rungs_tried"] if t["speech_sec"] is not None] == [True, False, True]


def test_rejected_reasons(tmp_path: Path) -> None:
    entailment = _choice(_run(build_world(tmp_path, {1: 4.0, 2: 3.0}, ok={1: False, 2: False})))
    assert entailment["status"] == "rejected" and entailment["reject_reason"] == "entailment"
    assert entailment["chosen_rung"] is None and entailment["chosen_text"] == entailment["original_text"]
    synthesis = _choice(_run(build_world(tmp_path, {1: 4.0, 2: 3.0}, errors={1, 2})))
    assert synthesis["reject_reason"] == "synthesis"
    every_check = {n: ["not_shorter"] for n in range(1, 6)}
    none = _choice(_run(build_world(tmp_path, {}, checks=every_check)))
    assert none["status"] == "rejected" and none["reject_reason"] == "no_candidates"


def test_timing_limited_stays_timing_limited(tmp_path: Path) -> None:
    world = build_world(tmp_path, {}, status="timing_limited")
    world.ledger[0]["rungs"] = []
    choice = _choice(_run(world))
    assert choice["status"] == "timing_limited" and choice["chosen_rung"] is None


def test_pass_two_steps_down_only_residual_units(tmp_path: Path) -> None:
    # Slack lets rung 1 (4.5 s) pass at the pass-1 target, but it leaves a real deficit; pass 2 steps to rung 2.
    world = build_world(tmp_path, {1: 4.5, 2: 4.2, 3: 3.0}, slack=0.3)
    resolution = _run(world)
    assert [p["pass"] for p in resolution.passes] == [1, 2]
    assert resolution.passes[0]["needs_text"] == 1 and resolution.passes[1]["needs_text"] == 0
    choice = _choice(resolution)
    assert choice["chosen_rung"] == 2 and choice["pass"] == 2 and choice["status"] == "resolved"
    assert choice["target_base_sec_pass1"] == world.placements0[0]["target_base_sec"]
    _no_collisions(resolution)


def test_lossy_rule(tmp_path: Path) -> None:
    rung_five = _choice(_run(build_world(tmp_path, {5: 1.0})))
    assert rung_five["chosen_rung"] == 5 and rung_five["lossy"]
    clean = _choice(_run(build_world(tmp_path, {2: 4.2})))
    assert clean["chosen_rung"] == 2 and not clean["lossy"]
    world = build_world(tmp_path, {2: 4.2})
    world.ledger[0]["rungs"][1]["dropped"] = ["u2"]  # drops a core fragment
    assert _choice(_run(world))["lossy"]


def test_choices_carry_reviewed_false_by_default(tmp_path: Path) -> None:
    assert _choice(_run(build_world(tmp_path, {2: 4.2})))["reviewed"] is False


def test_manual_text_replaces_the_choice_and_is_placed_with_its_measured_duration(tmp_path: Path) -> None:
    world = build_world(tmp_path, {1: 4.6, 2: 4.2, 3: 3.0})
    add_manual_row(world, tmp_path, "Но она думает иначе.", 3.4)
    resolution = _run(world, Overrides(texts={"unit-0": "Но она думает иначе."}))
    choice = _choice(resolution)
    assert choice["status"] == "manual" and choice["chosen_rung"] == "manual" and choice["reviewed"] is False
    assert choice["chosen_text"] == "Но она думает иначе." and choice["chosen_speech_sec"] == 3.4
    assert not choice["lossy"] and choice["reject_reason"] is None and choice["rungs_tried"] == []
    placement = next(p for p in resolution.placements if p["unit_id"] == "unit-0")
    assert placement["text_changed"] and placement["chosen_rung"] == "manual"
    assert placement["natural_sec"] * placement["tempo_factor"] == pytest.approx(3.4) and placement["removed_sec"] <= 0.05
    assert len(resolution.passes) == 1
    _no_collisions(resolution)


def test_manual_text_that_does_not_fit_is_never_stepped_in_pass_two(tmp_path: Path) -> None:
    world = build_world(tmp_path, {1: 4.6, 2: 4.2, 3: 3.0})
    add_manual_row(world, tmp_path, "Очень длинный ручной текст.", 5.5)
    resolution = _run(world, Overrides(texts={"unit-0": "Очень длинный ручной текст."}))
    choice = _choice(resolution)
    assert choice["chosen_rung"] == "manual" and choice["pass"] == 1
    assert [p["pass"] for p in resolution.passes] == [1]
    placement = next(p for p in resolution.placements if p["unit_id"] == "unit-0")
    assert placement["removed_sec"] > 0.05
    _no_collisions(resolution)


def test_manual_text_for_a_unit_outside_the_ledger(tmp_path: Path) -> None:
    world = build_world(tmp_path, {2: 4.2})
    add_manual_row(world, tmp_path, "Конечно.", 0.9, unit_id="unit-1")
    resolution = _run(world, Overrides(texts={"unit-1": "Конечно."}))
    choice = _choice(resolution, "unit-1")
    assert choice["status"] == "manual" and choice["original_text"] == "Да, конечно." and choice["chosen_text"] == "Конечно."
    assert next(p for p in resolution.placements if p["unit_id"] == "unit-1")["text_changed"]


def test_manual_text_without_a_good_measurement_is_an_error(tmp_path: Path) -> None:
    world = build_world(tmp_path, {2: 4.2})
    with pytest.raises(ValueError, match="unit-0"):
        _run(world, Overrides(texts={"unit-0": "Нет клипа."}))


def test_accept_keeps_the_choice_and_marks_it_reviewed(tmp_path: Path) -> None:
    world = build_world(tmp_path, {1: 5.0, 2: 4.9, 3: 4.5, 4: 4.4, 5: 4.8}, status="unresolved")
    plain = _choice(_run(world))
    accepted = _choice(_run(world, Overrides(accepted=frozenset({"unit-0"}))))
    assert accepted["reviewed"] is True
    assert {k: v for k, v in accepted.items() if k != "reviewed"} == {k: v for k, v in plain.items() if k != "reviewed"}
