from __future__ import annotations

from typing import Any

import pytest

from voiceover.text import syllables
from voiceover.units import (MergePolicy, build_units, continuation_reason, is_dialogue, join_texts, member_timing,
                             scene_ids)


def make_cues(spec: list[tuple[int, int, str]]) -> list[dict[str, Any]]:
    return [{"cue_id": f"cue-{i}", "position": i, "index": i + 1, "start": s, "end": e, "text": t}
            for i, (s, e, t) in enumerate(spec)]


@pytest.mark.parametrize("prev,nxt,reason", [
    ("Ну...", "и пошёл.", "ellipsis"),
    ("Ну...", "...и пошёл.", "ellipsis"),
    ("Ну…", "…и пошёл.", "ellipsis"),
    ("Ну...", "И пошёл.", None),
    ("Он пришёл,", "увидел", "comma"),
    ("Он пришёл;", "Увидел", "comma"),
    ("Он пришёл —", "увидел", "dash"),
    ("Он пришёл-", "увидел", "dash"),
    ("Он пришёл и", "увидел", "dangling"),
    ("Он пришёл и", "Увидел", None),
    ("Он пришёл.", "и увидел", None),
    ("Он пришёл?»", "и увидел", None),
    ("Он пришёл,»", "«и увидел", "comma"),
    ("Он пришёл (", "и увидел", None),
])
def test_continuation_reason(prev: str, nxt: str, reason: str | None) -> None:
    assert continuation_reason(prev, nxt) == reason


def test_is_dialogue() -> None:
    assert is_dialogue("- Привет")
    assert is_dialogue("— Привет")
    assert is_dialogue("Нет. - Да.")
    assert not is_dialogue("Он пришёл")
    assert not is_dialogue("Кто-то пришёл")


def test_join_texts() -> None:
    assert join_texts(["a  b", "c\n d"]) == "a b c d"
    assert join_texts(["a ", " b"]) == "a b"


def test_join_collapses_ellipsis_seam() -> None:
    assert join_texts(["Ну...", "...и пошёл"]) == "Ну...и пошёл"
    assert join_texts(["Ну…", "…и пошёл"]) == "Ну...и пошёл"


def test_three_cue_chain() -> None:
    cues = make_cues([(0, 1000, "Он пришёл,"), (1100, 2000, "увидел и"), (2100, 3000, "победил."),
                      (3100, 4000, "Конец.")])
    units = build_units(cues, MergePolicy())
    assert len(units) == 2
    first = units[0]
    assert first.unit_id == "unit-0" and first.cue_ids == ("cue-0", "cue-1", "cue-2")
    assert (first.start, first.end, first.index) == (0, 3000, 1)
    assert first.reasons == ("comma", "dangling") and first.merged
    assert first.text == "Он пришёл, увидел и победил."
    assert units[1].unit_id == "unit-3" and not units[1].merged and units[1].reasons == ()


def test_sentence_end_does_not_merge() -> None:
    cues = make_cues([(0, 1000, "Он пришёл."), (1100, 2000, "и ушёл")])
    assert len(build_units(cues, MergePolicy())) == 2


def test_dialogue_never_merges() -> None:
    as_next = make_cues([(0, 1000, "Он сказал,"), (1100, 2000, "- да")])
    assert len(build_units(as_next, MergePolicy())) == 2
    as_prev = make_cues([(0, 1000, "- Он сказал,"), (1100, 2000, "что да")])
    assert len(build_units(as_prev, MergePolicy())) == 2
    inline = make_cues([(0, 1000, "Нет. - Да,"), (1100, 2000, "может")])
    assert len(build_units(inline, MergePolicy())) == 2


def test_gap_blocks_merge_and_overlap_allowed() -> None:
    far = make_cues([(0, 1000, "Он сказал,"), (1301, 2000, "что да")])
    assert len(build_units(far, MergePolicy())) == 2
    edge = make_cues([(0, 1000, "Он сказал,"), (1300, 2000, "что да")])
    assert len(build_units(edge, MergePolicy())) == 1
    overlap = make_cues([(0, 1000, "Он сказал,"), (900, 2000, "что да")])
    assert len(build_units(overlap, MergePolicy())) == 1


def test_max_cues_cap() -> None:
    cues = make_cues([(i * 1000, i * 1000 + 900, "слово,") for i in range(5)])
    sizes = [len(u.cue_ids) for u in build_units(cues, MergePolicy(max_cues=3))]
    assert sizes == [3, 2]


def test_max_span_cap() -> None:
    cues = make_cues([(0, 3000, "слово,"), (3100, 6000, "слово,"), (6100, 9000, "слово")])
    units = build_units(cues, MergePolicy(max_span_ms=8000))
    assert [len(u.cue_ids) for u in units] == [2, 1]


def test_scene_boundary_blocks_merge_and_ids_increment() -> None:
    cues = make_cues([(0, 1000, "Он сказал,"), (3000, 4000, "что да"), (4100, 5000, "и всё")])
    assert scene_ids(cues, 1500) == ["scene-0", "scene-1", "scene-1"]
    # max_gap is large enough to allow the merge, the scene gap must still block it.
    units = build_units(cues, MergePolicy(max_gap_ms=5000, scene_gap_ms=1500))
    assert [u.scene_id for u in units] == ["scene-0", "scene-1"]
    assert [len(u.cue_ids) for u in units] == [1, 2]


def test_disabled_gives_one_unit_per_cue() -> None:
    cues = make_cues([(0, 1000, "Он сказал,"), (1100, 2000, "что да")])
    units = build_units(cues, MergePolicy(enabled=False))
    assert [u.cue_ids for u in units] == [("cue-0",), ("cue-1",)]


def test_policy_validation() -> None:
    assert MergePolicy.from_dict({"max_cues": 2}).max_cues == 2
    with pytest.raises(ValueError):
        MergePolicy.from_dict({"max_cues": 0})
    with pytest.raises(ValueError):
        MergePolicy.from_dict({"max_gap_ms": -1})


def test_member_timing_is_syllable_proportional() -> None:
    cues = make_cues([(1000, 2000, "ааа"), (2000, 3000, "ааааааа")])  # 3 and 7 syllables
    rows = member_timing(cues, 1100, 10.0)
    assert syllables("ааа") == 3
    assert rows[0] == {"cue_id": "cue-0", "start": 1000, "est_start_ms": 1100, "drift_ms": 100}
    assert rows[1]["est_start_ms"] == 1100 + 3000
    assert rows[1]["drift_ms"] == 4100 - 2000
    even = member_timing(make_cues([(0, 1, "xx"), (10, 11, "yy")]), 0, 2.0)
    assert [r["est_start_ms"] for r in even] == [0, 1000]
