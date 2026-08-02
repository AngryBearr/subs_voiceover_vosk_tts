"""Offline tests for semantic-unit construction."""

from __future__ import annotations

import copy
import math

import pytest

from utils.semantic_units import build_semantic_units


def cue(index: int, text: str, start: object = 0, end: object = 100) -> dict[str, object]:
    return {"index": index, "text": text, "start": start, "end": end}


def build(texts: list[str], *, starts: list[object] | None = None, ends: list[object] | None = None, **kwargs: object):
    starts = starts or [index * 100 for index in range(len(texts))]
    ends = ends or [start + 80 if isinstance(start, (int, float)) else None for start in starts]
    original = [cue(index, text, starts[index], ends[index]) for index, text in enumerate(texts)]
    candidate = copy.deepcopy(original)
    candidate[-1]["text"] = "CHANGED"
    return build_semantic_units(original, candidate, **kwargs)


def test_pair_and_triple_continuation_and_max3() -> None:
    original = [cue(0, "Он сказал,", 0, 80), cue(1, "что это", 100, 180), cue(2, "важно", 200, 280)]
    candidate = copy.deepcopy(original)
    candidate[0]["text"] = "Он сказал, изменено"
    units = build_semantic_units(original, candidate)
    assert units[0].cue_indices == (0, 1, 2)


def test_terminal_comma_ellipsis_and_connector_rules() -> None:
    assert build(["Он ушел.", "потом вернулся"])[0].cue_indices == (1,)
    assert build(["Он сказал,", "Это важно"])[0].cue_indices == (0, 1)
    assert build(["Он сказал...", "потом ушел"])[0].cue_indices == (0, 1)
    assert build(["Он сказал...", "Потом ушел"])[0].cue_indices == (1,)
    assert build(["Он ушел", "и вернулся"])[0].cue_indices == (0, 1)
    assert build(["Он ушел", "Из дома"])[0].cue_indices == (1,)


def test_gap_and_negative_overlap() -> None:
    assert build(["Он,", "Ушел"], starts=[0, 370], ends=[100, 450])[0].cue_indices == (0, 1)
    assert build(["Он", "Ушел"], starts=[0, 410], ends=[100, 450])[0].cue_indices == (1,)
    assert build(["Он", "ушел"], starts=[0, 50], ends=[100, 130])[0].cue_indices == (0, 1)


def test_max_cues_one_does_not_include_neighbors() -> None:
    assert build(["Он сказал,", "что это"], max_cues=1)[0].cue_indices == (1,)


def test_zero_gap_allows_touching_but_not_one_millisecond_gap() -> None:
    assert build(["Он сказал,", "что это"], starts=[0, 100], ends=[100, 200], max_gap_sec=0)[0].cue_indices == (0, 1)
    assert build(["Он сказал,", "что это"], starts=[0, 101], ends=[100, 201], max_gap_sec=0)[0].cue_indices == (1,)


def test_missing_invalid_timing_blocks() -> None:
    assert build(["Он", "ушел"], starts=[0, None], ends=[100, 200])[0].cue_indices == (1,)
    assert build(["Он", "ушел"], starts=[0, math.inf], ends=[100, 200])[0].cue_indices == (1,)


def test_context_source_allows_whitespace_only_boundary_difference() -> None:
    original = [cue(0, "source text")]
    candidate = copy.deepcopy(original)
    candidate[0]["text"] = "changed"
    context = [cue(0, "  source text \n")]
    assert build_semantic_units(original, candidate, context_items=context)[0].cue_indices == (0,)


def test_insertion_order_context_target_subset_and_evidence() -> None:
    original = [cue(2, "начало", 0, 100), cue(1, "продолжение", 100, 200), cue(3, "конец", 200, 300)]
    candidate = copy.deepcopy(original)
    candidate[1]["text"] = "изменено"
    assert build_semantic_units(original, candidate)[0].cue_indices == (2, 1, 3)
    candidate = copy.deepcopy(original)
    candidate[0]["text"] = "новое"
    candidate[2]["text"] = "новый конец"
    assert build_semantic_units(original, candidate, target_indices=[2]).__len__() == 1
    context = [cue(99, "сосед", -100, 0), *original]
    candidate = copy.deepcopy(original)
    candidate[1]["text"] = "изменено"
    assert build_semantic_units(original, candidate, context_items=context, max_cues=4)[0].cue_indices == (99, 2, 1, 3)


def test_changed_evidence_non_target_and_no_overlap() -> None:
    original = [cue(index, text, index * 100, index * 100 + 80) for index, text in enumerate(["начало,", "середина", "конец"])]
    candidate = copy.deepcopy(original)
    candidate[0]["text"] = "changed"
    candidate[1]["text"] = "also changed"
    units = build_semantic_units(original, candidate, target_indices=[0])
    assert units[0].changed_indices == (0,)
    assert 1 in units[0].cue_indices
    assert 1 not in units[0].changed_indices
    assert len({index for unit in units for index in unit.cue_indices}) == sum(len(unit.cue_indices) for unit in units)


def test_two_explicit_targets_are_both_changed_indices() -> None:
    original = [cue(0, "начало,", 0, 80), cue(1, "середина", 100, 180), cue(2, "конец", 200, 280)]
    candidate = copy.deepcopy(original)
    candidate[0]["text"] = "новое начало"
    candidate[2]["text"] = "новый конец"
    units = build_semantic_units(original, candidate, target_indices=[0, 2])
    assert units[0].changed_indices == (0, 2)


def test_stable_ids_duplicate_texts_and_inputs_unchanged() -> None:
    original = [cue(0, "Ёж", 0, 100), cue(1, "ёж", 100, 200)]
    candidate = copy.deepcopy(original)
    candidate[0]["text"] = "изменено"
    original_snapshot = copy.deepcopy(original)
    candidate_snapshot = copy.deepcopy(candidate)
    unit = build_semantic_units(original, candidate)[0]
    assert original == original_snapshot
    assert candidate == candidate_snapshot
    candidate[0]["text"] = "другое"
    assert unit.unit_id == build_semantic_units(original, candidate)[0].unit_id
    assert original == original_snapshot and candidate != candidate_snapshot
    other = [cue(5, "Ёж", 0, 100)]
    other_candidate = copy.deepcopy(other)
    other_candidate[0]["text"] = "изменено"
    assert unit.unit_id != build_semantic_units(other, other_candidate)[0].unit_id


@pytest.mark.parametrize("bad", [
    ([cue(0, "a"), cue(0, "b")], [cue(0, "a")]),
    ([cue(0, "a")], [cue(1, "a")]),
])
def test_duplicate_missing_mismatched_indices(bad: tuple[list[dict[str, object]], list[dict[str, object]]]) -> None:
    with pytest.raises(ValueError):
        build_semantic_units(*bad)


def test_invalid_target_and_config() -> None:
    original = [cue(0, "a")]
    candidate = [cue(0, "b")]
    with pytest.raises(ValueError):
        build_semantic_units(original, candidate, target_indices=[True])
    with pytest.raises(ValueError):
        build_semantic_units(original, candidate, max_cues=0)
    with pytest.raises(ValueError):
        build_semantic_units(original, candidate, max_gap_sec=math.inf)
