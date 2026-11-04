"""Line gender by voice for speakers of known gender: override, corrections priority, voices, review list."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from stt.dub import plan
from stt.line_gender import host_speakers, override, voice_prints
from stt.review import apply_corrections, suspects, write_review

SPEAKERS = [{"id": "S01", "name": "Jeff Probst", "label_ru": "Джефф", "gender": "male", "role": "host",
             "embedding": [1.0, 0.0, 0.0]},
            {"id": "S02", "name": "Chrissy Sarnowsky", "label_ru": "Крисси Сарновски", "gender": "female",
             "embedding": [0.0, 1.0, 0.0]},
            {"id": "S03", "gender": "unknown"}]
VOICES = {"female": 49, "male": 56, "unknown": 56}


def _cue(index: int, speaker: str, gender: str, **extra: Any) -> dict[str, Any]:
    return {"index": index, "start": index * 2000, "end": index * 2000 + 1500, "duration": 1500, "speaker": speaker,
            "gender": gender, "speaker_name": speaker, "text": ["Привет"], "text_en": ["Hi"], **extra}


def test_override_only_when_sure_of_the_other_gender() -> None:
    cues = [_cue(1, "S02", "female"), _cue(2, "S02", "female"), _cue(3, "S02", "female"), _cue(4, "S02", "female")]
    changes = override(cues, SPEAKERS, {1: 0.05, 2: 0.2, 3: 0.97, 4: 0.1}, 0.9)
    assert [c["index"] for c in changes] == [1, 4]  # exactly 0.1 counts: the threshold is inclusive
    assert cues[0]["gender"] == "male" and cues[0]["gender_by_voice"] == {"p_female": 0.05, "speaker_gender": "female"}
    assert cues[1]["gender"] == "female" and "gender_by_voice" not in cues[1]
    assert cues[2]["gender"] == "female"  # sure of the speaker's own gender: nothing to do


def test_threshold_comes_from_the_caller() -> None:
    assert override([_cue(1, "S02", "female")], SPEAKERS, {1: 0.2}, 0.9) == []
    assert len(override([_cue(1, "S02", "female")], SPEAKERS, {1: 0.2}, 0.75)) == 1


def test_unknown_speakers_host_ocr_and_missing_scores_are_left_alone() -> None:
    cues = [_cue(1, "S03", "unknown"), _cue(2, "S01", "male"), _cue(3, "S02", "female", ocr=True),
            _cue(4, "S02", "female"), _cue(5, "S01", "male")]
    hosts = host_speakers(SPEAKERS)
    assert hosts == {"S01"}
    changes = override(cues, SPEAKERS, {1: 0.99, 2: 0.99, 3: 0.01, 5: 0.99}, 0.9, hosts)
    assert changes == [] and [c["gender"] for c in cues] == ["unknown", "male", "female", "female", "male"]


def test_rerun_decides_afresh() -> None:
    cues = [_cue(1, "S02", "female")]
    override(cues, SPEAKERS, {1: 0.02}, 0.9)
    assert cues[0]["gender"] == "male"
    assert override(cues, SPEAKERS, {1: 0.5}, 0.9) == []  # no longer sure: back to the speaker's gender
    assert cues[0]["gender"] == "female" and "gender_by_voice" not in cues[0]


def test_voice_prints_use_the_pipeline_crop_and_buckets() -> None:
    import torch

    seen: list[tuple[int, ...]] = []

    class Fake(torch.nn.Module):
        def forward(self, x: Any) -> Any:
            seen.append(tuple(x.shape))
            return torch.ones(x.shape[0], 256)

    clips = [np.zeros(int(s * 16000), dtype=np.float32) for s in (0.2, 1.3, 1.4, 9.0)]
    out = voice_prints(Fake(), clips, device="cpu")
    assert out.shape == (4, 256) and abs(float(np.linalg.norm(out[0])) - 1) < 1e-5
    # 0.2 s -> the 0.5 s minimum; 1.3 and 1.4 s -> 1.25 s bucket; 9 s -> 6 s crop
    assert sorted(seen) == [(1, 1, 8000), (1, 1, 96000), (2, 1, 20000)]


def test_dub_voice_follows_the_changed_gender_in_every_voice_map() -> None:
    cues = [_cue(1, "S02", "male", gender_by_voice={"p_female": 0.03, "speaker_gender": "female"}),
            _cue(2, "S02", "female")]
    for maps, own in (({"S02": "/clips/chrissy.wav"}, "/clips/chrissy.wav"),  # F5 clones
                      ({"S02": "teratts:ru_f1"}, "teratts:ru_f1"), ({"S01": "teratts:ru_m5"}, None)):
        voiced, _ = plan([dict(c) for c in cues], VOICES, speaker_voices=maps)
        assert voiced[0]["voice"] == 56  # a man's shout: the male voice, not Chrissy's
        assert voiced[1]["voice"] == (own or 49)
    teratts = {"female": "ru_f1", "male": "ru_m5", "unknown": "ru_m5"}
    voiced, _ = plan([dict(cues[0])], teratts, speaker_voices={"S02": "/clips/chrissy.wav"})
    assert voiced[0]["voice"] == "ru_m5"


def test_corrections_gender_wins_over_gender_by_voice_and_follows_the_voice() -> None:
    by_voice = {"p_female": 0.03, "speaker_gender": "female"}
    records = [_cue(1, "S02", "male", gender_by_voice=dict(by_voice)), _cue(2, "S02", "female"),
               _cue(3, "S02", "male", gender_by_voice=dict(by_voice))]
    records, _ = apply_corrections(records, {"1": {"gender": "female"}, "2": {"gender": "male"},
                                             "3": {"speaker": "Jeff"}}, SPEAKERS)
    assert [r["gender"] for r in records] == ["female", "male", "male"]
    assert "gender_by_voice" not in records[0] and "gender_by_voice" not in records[2]
    voiced, _ = plan(records, VOICES, speaker_voices={"S01": "/jeff.wav", "S02": "/chrissy.wav"})
    # 1: back to the speaker's gender and voice; 2: a person made her male: the male voice; 3: Jeff's own voice
    assert [v["voice"] for v in voiced] == ["/chrissy.wav", 56, "/jeff.wav"]


def test_review_lists_every_changed_line_with_an_undoing_fix(tmp_path: Path) -> None:
    rate = 100
    audio = np.zeros(10 * rate, dtype=np.float32)
    records = [_cue(1, "S02", "male", gender_by_voice={"p_female": 0.07, "speaker_gender": "female"}),
               _cue(2, "S02", "male", gender_by_voice={"p_female": 0.04, "speaker_gender": "female"}),
               _cue(3, "S02", "female")]

    def own_voice(segment: np.ndarray, rate: int) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0])  # clearly its own speaker: no other signal

    items = suspects(records, SPEAKERS, audio, rate, lambda seg: (0.5, 0.5), own_voice, fixed_gender={"2"})
    assert [i["line"] for i in items] == [1]  # line 2 already has a gender from a person
    item = items[0]
    assert item["fix"] == {"gender": "female"}
    assert item["note"] == ("Род изменён по голосу: мужской (уверенность 0,93); "
                            "герой — Крисси Сарновски (женский)")
    write_review(tmp_path, items)
    assert "Род изменён по голосу" in (tmp_path / "review.srt").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))[0]["note"] == item["note"]
    from ui.dub_app import review_note

    assert item["note"] in review_note(item)
