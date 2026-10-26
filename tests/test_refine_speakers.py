from __future__ import annotations

from typing import Any

import numpy as np

from stt.refine_speakers import _overlapping, refine_turns

A, B = np.array([1.0, 0.0]), np.array([0.0, 1.0])


def _turn(start: float, end: float, speaker: str, voice: np.ndarray) -> dict[str, Any]:
    return {"start": start, "end": end, "speaker": speaker, "_voice": voice}


def test_a_turn_that_sounds_like_another_speaker_moves() -> None:
    turns = [_turn(0, 3, "S1", A), _turn(3, 6, "S2", B), _turn(6, 7, "S1", B),  # misassigned
             _turn(7, 10, "S1", A), _turn(10, 13, "S2", B), _turn(13, 13.3, "S1", B)]  # too short to judge
    refined, stats = refine_turns(turns, lambda t: t["_voice"], margin=0.1)
    assert [t["speaker"] for t in refined] == ["S1", "S2", "S2", "S1", "S2", "S1"]
    assert stats["moved"] == 1 and turns[2]["speaker"] == "S1"  # input untouched


def test_overlapping_turns_are_not_judged() -> None:
    turns = [_turn(0, 4, "S1", A), _turn(3, 5, "S2", A), _turn(6, 9, "S2", B)]
    assert _overlapping(turns) == [True, True, False]
    refined, stats = refine_turns(turns, lambda t: t["_voice"], margin=0.1)
    assert [t["speaker"] for t in refined] == ["S1", "S2", "S2"] and stats["moved"] == 0


def test_near_identical_speakers_merge() -> None:
    turns = [_turn(0, 3, "S1", A), _turn(3, 6, "S2", A), _turn(6, 9, "S3", B), _turn(9, 12, "S1", A)]
    refined, stats = refine_turns(turns, lambda t: t["_voice"], margin=0.1, merge=0.95)
    assert [t["speaker"] for t in refined] == ["S1", "S1", "S3", "S1"] and stats["merged"] == 1


def test_refine_diarization_keeps_embeddings_and_reports(monkeypatch: Any) -> None:
    import stt.refine_speakers as rs

    voices = {0.0: A, 3.0: B, 6.0: B, 7.0: A}
    monkeypatch.setattr(rs, "make_embedder", lambda name, device="cuda": None)
    monkeypatch.setattr(rs, "turn_embedder", lambda audio, embed: lambda t: voices[t["start"]])
    diar = {"turns": [{"start": 0.0, "end": 3.0, "speaker": "S1"}, {"start": 3.0, "end": 6.0, "speaker": "S2"},
                      {"start": 6.0, "end": 7.0, "speaker": "S1"}, {"start": 7.0, "end": 10.0, "speaker": "S1"}],
            "embeddings": {"S1": [1.0]}}
    diar["exclusive"] = [dict(t) for t in diar["turns"]]
    out = rs.refine_diarization(np.zeros(10), diar, "resnet293", 0.1)
    assert [t["speaker"] for t in out["exclusive"]] == ["S1", "S2", "S2", "S1"]
    assert out["embeddings"] == {"S1": [1.0]} and out["refined"]["moved_exclusive"] == 1
