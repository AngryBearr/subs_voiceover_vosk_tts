"""Diarization data model and the exclusive-turn derivation (no GPU, no pyannote)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from stt.diarize import Diarization, Turn, create_diarizer, exclusive_turns, non_default_options

STEP = 0.02


def _case() -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros((11, 3))
    active = np.zeros((11, 3), dtype=bool)
    active[0:4, 0] = True
    scores[0:4, 0] = 0.9
    active[4:7, 0:2] = True
    scores[4:6] = [0.4, 0.8, 0.0]
    scores[6] = [0.7, 0.3, 0.0]
    active[9:11, 2] = True
    scores[9, 1] = 0.99  # inactive cluster must not win
    return scores, active


def test_exclusive_turns_synthetic() -> None:
    scores, active = _case()
    turns = exclusive_turns(scores, active, 0.0, STEP, STEP)
    assert turns == [
        Turn(0.01, 0.09, "0"),  # times are frame middles, as in pyannote's Binarize
        Turn(0.09, 0.13, "1"),
        Turn(0.13, 0.15, "0"),
        Turn(0.19, 0.21, "2"),
    ]
    for a, b in zip(turns, turns[1:]):
        assert a.end <= b.start
    for t in turns:
        first, stop = round((t.start - STEP / 2) / STEP), round((t.end - STEP / 2) / STEP)
        assert active[first:stop, int(t.speaker)].all()


def test_exclusive_turns_ties_and_empty() -> None:
    active = np.array([[False, True], [True, True]])
    turns = exclusive_turns(np.zeros((2, 2)), active, 1.0, STEP, 0.05)
    assert turns == [Turn(1.025, 1.045, "1")]  # the one-frame run at the end has no length and is dropped
    assert exclusive_turns(np.zeros((0, 2)), np.zeros((0, 2), dtype=bool), 0.0, STEP, STEP) == []


def test_non_default_options() -> None:
    assert non_default_options(None) == {}
    assert non_default_options({"median_filter": True, "exclusive": False}) == {}
    assert non_default_options({"median_filter": False, "exclusive": True}) == {
        "median_filter": False,
        "exclusive": True,
    }


def test_diarization_options_round_trip() -> None:
    legacy = Diarization.from_dict({"turns": [{"start": 0.0, "end": 1.0, "speaker": "0"}]})
    assert legacy.options == {}
    diar = Diarization(turns=[Turn(0.0, 1.0, "0")], options={"exclusive": True})
    again = Diarization.from_dict(diar.to_dict())
    assert again == diar
    assert again.to_dict()["options"] == {"exclusive": True}


def test_ahc_threshold_option() -> None:
    assert non_default_options({"ahc_threshold": 0.5}) == {}
    assert non_default_options({"median_filter": True, "ahc_threshold": 0.4}) == {"ahc_threshold": 0.4}
    default = create_diarizer("diarizen-v2", device="cpu")
    assert default.options["ahc_threshold"] == 0.5
    tuned = create_diarizer("diarizen-v2", device="cpu", options={"ahc_threshold": 0.4})
    assert tuned.options["ahc_threshold"] == 0.4 and tuned.options["exclusive"] is False


def test_one_segmentation_pass_gives_hard_and_soft() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("pyannote.core")
    from pyannote.core import SlidingWindow, SlidingWindowFeature

    from stt.diarize import _segment_hard_and_soft

    logits = torch.log_softmax(torch.tensor([[[3.0, 1.0], [0.0, 2.0]]]), dim=-1)

    class Conversion(torch.nn.Module):  # stands for Powerset: class 1 means speaker 1 speaks
        def forward(self, output: Any, soft: bool = False) -> Any:
            return output.exp()[..., 1:] if soft else (output.argmax(-1, keepdim=True) == 1).float()

    class Inference:
        device = "cpu"
        conversion: Any = Conversion()

    class Pipeline:
        _segmentation = Inference()
        calls = 0

        def get_segmentations(self, file: dict[str, Any], soft: bool = False) -> SlidingWindowFeature:
            self.calls += 1
            output = self._segmentation.conversion(logits, soft=soft)
            return SlidingWindowFeature(output.numpy(), SlidingWindow(start=0.0, duration=8.0, step=0.8))

    pipeline = Pipeline()
    conversion = Inference.conversion
    hard, soft = _segment_hard_and_soft(pipeline, {})
    assert pipeline.calls == 1 and pipeline._segmentation.conversion is conversion
    np.testing.assert_array_equal(hard.data, conversion(logits, soft=False).numpy())
    np.testing.assert_allclose(soft.data, conversion(logits, soft=True).numpy())
    assert hard.sliding_window.step == soft.sliding_window.step == 0.8
