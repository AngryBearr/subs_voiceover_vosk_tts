from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stt.dub import plan
from stt.review import apply_corrections, suspects, write_review


def _rec(index: int, start: float, end: float, speaker: str, gender: str, text: str = "Привет") -> dict[str, Any]:
    return {"index": index, "start": int(start * 1000), "end": int(end * 1000), "speaker": speaker,
            "gender": gender, "speaker_name": speaker, "text": [text], "text_en": ["Hi"]}


SPEAKERS = [{"id": "S01", "name": "Jeff Probst", "label_ru": "Джефф", "gender": "male", "embedding": [1.0, 0.0, 0.0]},
            {"id": "S02", "name": "Julie McGee", "label_ru": "Джули", "gender": "female",
             "embedding": [0.0, 1.0, 0.0]}]


def test_suspects_by_gender_and_by_voice(tmp_path: Path) -> None:
    rate = 100
    audio = np.zeros(10 * rate, dtype=np.float32)
    audio[2 * rate:4 * rate] = 1.0  # line 2 sounds female
    audio[5 * rate:7 * rate] = 0.5  # line 3 sounds like S02
    records = [_rec(1, 0, 1.5, "S01", "male"), _rec(2, 2, 4, "S01", "male"), _rec(3, 5, 7, "S01", "male"),
               _rec(4, 8, 9.5, "S01", "male")]

    def gender_of(segment: np.ndarray) -> tuple[float, float]:
        return (0.95, 0.05) if segment.mean() > 0.9 else (0.05, 0.95)

    def embed(segment: np.ndarray, rate: int) -> np.ndarray:
        if abs(segment.mean() - 0.5) < 0.01:
            return np.array([0.0, 1.0, 0.0])  # sounds like S02
        if segment.mean() > 0.9:
            return np.array([0.0, 0.0, 1.0])  # like nobody known: the gender model decides
        return np.array([1.0, 0.0, 0.0])  # clearly its own speaker

    items = suspects(records, SPEAKERS, audio, rate, gender_of, embed, skip={4})
    assert [(i["line"], i["fix"]) for i in items] == [(2, {"gender": "female"}), (3, {"speaker": "Julie McGee"})]
    write_review(tmp_path, items)
    assert json.loads((tmp_path / "review.json").read_text())[0]["time"] == "00:00:02,000"
    assert "#3 S01: sounds like Julie McGee" in (tmp_path / "review.srt").read_text()


def test_corrections_change_gender_speaker_text_and_skip() -> None:
    records = [_rec(i, i, i + 1, "S01", "male") for i in range(1, 5)]
    corrections = {"1": {"gender": "female"}, "2": {"speaker": "Julie"}, "3": {"text": "Новый текст"},
                   "4": {"skip": True}}
    records, skip = apply_corrections(records, corrections, SPEAKERS)
    assert records[0]["gender"] == "female"
    assert (records[1]["speaker"], records[1]["gender"], records[1]["speaker_name"]) == ("S02", "female", "Джули")
    assert records[2]["text"] == ["Новый текст"] and skip == {4}
    voiced, skipped = plan(records, {"male": 56, "female": 49, "unknown": 56}, skip=skip)
    assert [v["voice"] for v in voiced] == [49, 49, 56] and skipped[0]["reason"] == "skipped in corrections.json"
    with pytest.raises(ValueError):
        apply_corrections(records, {"1": {"speaker": "Nobody Here"}}, SPEAKERS)


def test_no_gender_flag_when_the_line_sounds_like_its_speaker() -> None:
    rate = 100
    audio = np.ones(3 * rate, dtype=np.float32)
    records = [_rec(1, 0, 2, "S01", "male")]
    items = suspects(records, SPEAKERS, audio, rate, lambda seg: (0.99, 0.01),
                     lambda seg, r: np.array([1.0, 0.0, 0.0]))
    assert items == []  # the host's shout "heard female" while it clearly is his voice
