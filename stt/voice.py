"""Per-speaker voice attributes (gender, age) from clean, non-overlapping speech.

Model: audeering/wav2vec2-large-robust-24-ft-age-gender (CC-BY-NC-SA-4.0): age (0..1 ~ 0..100
years) and probabilities for female / male / child. TeraSpace/TeraAnaly (distilled from it)
agreed on every speaker of a test episode and was dropped.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from stt.audio import SAMPLE_RATE
from stt.diarize import Diarization, Turn

log = logging.getLogger(__name__)

AGE_GENDER_MODEL = "audeering/wav2vec2-large-robust-24-ft-age-gender"


@dataclass
class VoiceProfile:
    speaker: str
    speech_s: float
    clean_s: float
    windows: int
    gender: str  # "female" | "male" | "unknown"
    gender_conf: float | None
    p_female: float | None
    p_male: float | None
    p_child: float | None
    age: float | None
    agreement: float | None  # share of windows voting for the chosen gender

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _subtract(interval: tuple[float, float], others: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Parts of `interval` not covered by any of `others` (sorted by start)."""
    pieces = [interval]
    for o_start, o_end in others:
        nxt = []
        for p_start, p_end in pieces:
            if o_end <= p_start or o_start >= p_end:
                nxt.append((p_start, p_end))
                continue
            if o_start > p_start:
                nxt.append((p_start, o_start))
            if o_end < p_end:
                nxt.append((o_end, p_end))
        pieces = nxt
    return pieces


def clean_windows(
    turns: list[Turn],
    speaker: str,
    *,
    min_s: float = 1.0,
    max_s: float = 6.0,
    budget_s: float = 45.0,
    edge_s: float = 0.15,
) -> list[tuple[float, float]]:
    """Single-speaker windows for `speaker`, spread over the recording, up to `budget_s` total."""
    own = [(t.start, t.end) for t in turns if t.speaker == speaker]
    others = sorted((t.start, t.end) for t in turns if t.speaker != speaker)
    windows: list[tuple[float, float]] = []
    for interval in own:
        for start, end in _subtract(interval, [o for o in others if o[1] > interval[0] and o[0] < interval[1]]):
            start, end = start + edge_s, end - edge_s  # keep away from turn boundaries
            while end - start >= min_s:
                windows.append((start, min(end, start + max_s)))
                start += max_s
    if sum(e - s for s, e in windows) <= budget_s:
        return windows
    # Evenly sample across the timeline instead of taking only the beginning.
    windows.sort(key=lambda w: w[0])
    picked: list[tuple[float, float]] = []
    total = 0.0
    step = max(1, len(windows) // max(1, int(budget_s / max_s)))
    for w in windows[::step]:
        if total >= budget_s:
            break
        picked.append(w)
        total += w[1] - w[0]
    return picked


class AgeGenderEstimator:
    """wav2vec2 age/gender model; one forward pass per window."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    def load(self) -> None:
        import torch
        from torch import nn
        from transformers import Wav2Vec2FeatureExtractor
        from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

        class ModelHead(nn.Module):
            def __init__(self, config: Any, num_labels: int) -> None:
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, num_labels)

            def forward(self, x: Any) -> Any:
                return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

        class AgeGenderModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config: Any) -> None:
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = ModelHead(config, 1)
                self.gender = ModelHead(config, 3)
                self.post_init()

            def forward(self, input_values: Any) -> tuple[Any, Any]:
                hidden = self.wav2vec2(input_values)[0].mean(dim=1)
                return self.age(hidden), torch.softmax(self.gender(hidden), dim=1)

        self._torch = torch
        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(AGE_GENDER_MODEL)
        self._model = AgeGenderModel.from_pretrained(AGE_GENDER_MODEL).to(self.device).eval()

    def predict(self, audio: np.ndarray) -> tuple[float, np.ndarray]:
        """Return (age in years, [p_female, p_male, p_child]) for one window."""
        torch = self._torch
        values = self._extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")["input_values"]
        with torch.inference_mode():
            age, probs = self._model(values.to(self.device))
        return float(age[0, 0]) * 100.0, probs[0].float().cpu().numpy()


def profile_speakers(
    audio: np.ndarray,
    diarization: Diarization,
    estimator: AgeGenderEstimator,
    *,
    threshold: float = 0.7,
    min_clean_s: float = 2.0,
) -> list[VoiceProfile]:
    """Estimate gender/age per speaker; abstain ("unknown") on low confidence or little clean speech."""
    profiles: list[VoiceProfile] = []
    for speaker in diarization.speakers:
        speech_s = sum(t.end - t.start for t in diarization.turns if t.speaker == speaker)
        windows = clean_windows(diarization.turns, speaker)
        clean_s = sum(e - s for s, e in windows)
        if clean_s < min_clean_s:
            profiles.append(VoiceProfile(speaker, round(speech_s, 2), round(clean_s, 2), len(windows),
                                         "unknown", None, None, None, None, None, None))
            continue
        weights, ages, probs = [], [], []
        for start, end in windows:
            age, p = estimator.predict(audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)])
            weights.append(end - start)
            ages.append(age)
            probs.append(p)
        w = np.asarray(weights) / sum(weights)
        p_mean = (np.asarray(probs) * w[:, None]).sum(axis=0)
        adult = p_mean[:2] / max(p_mean[:2].sum(), 1e-9)
        conf = float(adult.max())
        label = ("female", "male")[int(adult.argmax())]
        votes = [("female", "male")[int(np.argmax(p[:2]))] for p in probs]
        agreement = sum(v == label for v in votes) / len(votes)
        gender = label if conf >= threshold else "unknown"
        profiles.append(
            VoiceProfile(
                speaker=speaker,
                speech_s=round(speech_s, 2),
                clean_s=round(clean_s, 2),
                windows=len(windows),
                gender=gender,
                gender_conf=round(conf, 3),
                p_female=round(float(p_mean[0]), 3),
                p_male=round(float(p_mean[1]), 3),
                p_child=round(float(p_mean[2]), 3),
                age=round(float(np.dot(ages, w)), 1),
                agreement=round(agreement, 3),
            )
        )
    return profiles
