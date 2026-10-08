"""Local timing constraints, exact text provenance and physical silence checks."""

from __future__ import annotations

import json
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from utils.alignment_pilot import Cue, allocate, assemble, boundary_reasons, load_cues, select_groups


def cue(position: int, start: int, end: int, text: str = "Exact text.") -> Cue:
    return Cue(f"cue-{position}", position, position + 1, start, end, text, text,
               (text,), "short_timing", end - start)


def test_borrowing_preserves_text_order_caps_and_conservation() -> None:
    cues = [cue(0, 1000, 1200, "Да."), cue(1, 1300, 3000, "Neighbour text.")]
    before = list(cues)
    result = allocate(cues, [500, 1000])
    assert result["status"] == "feasible_inspect_only"
    placements = result["proposal"]["placements"]
    assert [row["text"] for row in placements] == [c.text for c in cues]
    assert [row["cue_id"] for row in placements] == [c.cue_id for c in cues]
    assert placements[0]["new_start_ms"] == 1000
    assert placements[1]["new_start_ms"] == 1700
    assert placements[1]["start_shift_ms"] == 400
    assert result["proposal"]["actual_pauses_ms"] == [200]
    assert result["conservation_ms"] == result["envelope_ms"] == 2000
    assert result["unused_edge_silence_ms"] == 300
    assert cues == before


def test_longer_pause_retained_then_explicitly_adjustable() -> None:
    cues = [cue(0, 0, 300), cue(1, 700, 1800)]
    retained = allocate(cues, [400, 600])
    assert retained["proposal"]["pause_mode"] == "source_gaps_retained"
    assert retained["proposal"]["actual_pauses_ms"] == [400]
    compressed = allocate(cues, [700, 800])
    assert len(compressed["attempts"]) == 2
    assert compressed["proposal"]["pause_mode"] == "adjustable_gaps_to_minimum"
    assert compressed["proposal"]["actual_pauses_ms"] == [200]
    assert compressed["conservation_ms"] == 1800


def test_shift_cap_fails_even_when_total_duration_fits() -> None:
    result = allocate([cue(0, 0, 200), cue(1, 300, 4000)], [700, 500])
    assert result["status"] == "unresolved"
    assert result["proposal"] is None
    assert result["reasons"] == ["cue-1:start_shift_cap"]
    assert result["attempts"][-1]["end_overrun_ms"] == 0


def test_exact_shift_and_gap_boundaries_are_allowed() -> None:
    result = allocate([cue(0, 0, 200), cue(1, 400, 2000)], [700, 500])
    assert result["proposal"]["placements"][1]["start_shift_ms"] == 500
    assert result["proposal"]["placements"][1]["new_end_ms"] == 1400
    assert not boundary_reasons([cue(0, 0, 200), cue(1, 700, 1800)], None, None)


def test_no_external_gap_borrowing_and_no_false_fit() -> None:
    result = allocate([cue(0, 0, 300), cue(1, 500, 1000)], [500, 501], next_start=9000)
    assert result["proposal"] is None
    assert "cue-1:out_of_envelope" in result["reasons"]
    assert result["envelope_end_ms"] == 1000
    assert result["attempts"][-1]["end_overrun_ms"] == 201
    with pytest.raises(ValueError):
        allocate([], [])
    with pytest.raises(ValueError):
        allocate([cue(0, 0, 300)], [0])


@pytest.mark.parametrize("variant,reason", [
    ("gap", "large_gap_boundary"), ("overlap", "source_overlap"),
    ("speaker", "speaker_boundary"), ("unknown_speaker", "speaker_boundary"),
    ("dialogue", "explicit_dialogue_marker"), ("invalid", "invalid_source_timing"),
    ("not_adjacent", "not_adjacent"),
])
def test_boundaries(variant: str, reason: str) -> None:
    left, right = cue(0, 0, 300), cue(1, 500, 1300)
    if variant == "gap":
        right = replace(right, start=801)
    elif variant == "overlap":
        right = replace(right, start=299)
    elif variant == "speaker":
        left, right = replace(left, speaker="Alice"), replace(right, speaker="Bob")
    elif variant == "unknown_speaker":
        left = replace(left, speaker="Alice")
    elif variant == "dialogue":
        left = replace(left, dialogue_marker=True)
    elif variant == "invalid":
        left = replace(left, end=0)
    elif variant == "not_adjacent":
        right = replace(right, position=2)
    result = allocate([left, right], [200, 200])
    assert reason in result["reasons"]
    assert result["proposal"] is None and not result["attempts"]


def test_external_overlaps_not_automatically_fixed() -> None:
    cues = [cue(1, 1000, 1400), cue(2, 1600, 2200)]
    assert boundary_reasons(cues, 1001, None) == ["overlap_with_previous_external_cue"]
    assert boundary_reasons(cues, None, 2199) == ["overlap_with_next_external_cue"]
    assert not boundary_reasons(cues, 1000, 2200)


def test_selection_is_deterministic_disjoint_bounded_and_keeps_invalid() -> None:
    cues = [cue(i, i * 2000, i * 2000 + 300 + i) for i in range(40)]
    groups = select_groups(cues)
    assert groups == select_groups(cues)
    assert len(groups) == 6
    selected = [c.position for g in groups for c in g["cues"]]
    assert len(selected) == len(set(selected)) == 15
    assert [len(g["cues"]) for g in groups] == [2, 3, 2, 3, 2, 3]
    assert all("large_gap_boundary" in boundary_reasons(g["cues"], g["previous_end_ms"],
                                                       g["next_start_ms"]) for g in groups)


def test_triple_band_prefers_saved_accepted_neighbor_not_predicted_fit() -> None:
    cues = [cue(i, i * 1000, i * 1000 + 300 + i) for i in range(40)]
    cues[12] = replace(cues[12], status="accepted", text="An accepted neighbour.")
    groups = select_groups(cues)
    triple = groups[1]
    assert any(c.status == "accepted" for c in triple["cues"])
    assert triple["anchor_id"] == "cue-11"


def test_physical_frames_rounding_silence_and_complete_speech(tmp_path: Path) -> None:
    rate, count = 22050, 2206
    paths = [tmp_path / f"{i}.wav" for i in range(2)]
    for path in paths:
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(b"\x01\x02" * count)
    result = allocate([cue(0, 1, 102), cue(1, 302, 703)], [101, 101])
    output = tmp_path / "assembled.wav"
    evidence = assemble(paths, result["proposal"]["placements"], output, 1, 703)
    assert evidence["frames"] == 702 * rate // 1000
    assert evidence["speech_frames"] == count * 2
    assert evidence["frame_conservation"]
    assert evidence["speech_bytes_unchanged"] and evidence["inserted_silence_verified_zero"]
    assert evidence["inter_cue_zero_frames"][0] >= 4410
    bad = [dict(row) for row in result["proposal"]["placements"]]
    bad[1]["new_start_ms"] = 1
    with pytest.raises(ValueError, match="overlaps"):
        assemble(paths, bad, tmp_path / "bad.wav", 1, 703)


def test_saved_final_provenance_unresolved_original_and_dialogue(tmp_path: Path) -> None:
    original = [{"index": 1, "start": 0, "end": 300, "text": ["Original."]},
                {"index": 2, "start": 500, "end": 1000, "text": ["- Да."]}]
    current = [dict(original[0], text=["Accepted final."]), original[1]]
    report = {"records": [{"position": 0, "status": "accepted", "final_text": "Accepted final.",
                            "budget": {"available_ms": 300}},
                           {"position": 1, "status": "short_timing", "budget": {"available_ms": 500}}]}
    for filename, payload in (("report.json", report), ("review_state.json", {"source_items": original}),
                              ("subtitles.json", current)):
        (tmp_path / filename).write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_cues(tmp_path)
    assert loaded[0].text == "Accepted final." and loaded[0].original_text == "Original."
    assert loaded[1].text == "- Да." and loaded[1].dialogue_marker
    current[1] = dict(current[1], text=["Unapproved replacement."])
    (tmp_path / "subtitles.json").write_text(json.dumps(current), encoding="utf-8")
    with pytest.raises(ValueError, match="text mismatch"):
        load_cues(tmp_path)
