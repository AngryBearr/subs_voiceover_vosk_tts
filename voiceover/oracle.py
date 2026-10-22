"""Duration oracle: a per-run linear model from text measures to speech seconds.

The oracle is fitted on the current run's own first synthesis pass, so it
describes exactly the engine, voice, rate and post-processing in use.  It
makes no claim about other engines or other settings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import nnls

from voiceover.text import FEATURE_NAMES, features

MIN_SAMPLES = 30


@dataclass(frozen=True)
class Oracle:
    """Fitted duration predictor bound to one engine identity."""

    engine_identity: str
    coefficients: dict[str, float]
    sample_count: int
    mae_sec: float
    margin_sec: float
    residual_std_sec: float
    syllables_per_sec: float

    def predict(self, text: str) -> float:
        """Return the expected speech seconds without any safety margin."""
        measured = features(text)
        return max(0.0, sum(self.coefficients[name] * measured[name] for name in FEATURE_NAMES))

    def predict_safe(self, text: str) -> float:
        """Return the expected seconds plus the fitted margin (p90 underestimate)."""
        return self.predict(text) + self.margin_sec

    def syllable_capacity(self, target_sec: float, pause_count: int = 1) -> int:
        """Return how many syllables fit into ``target_sec`` with the margin applied."""
        per_syllable = self.coefficients["syllables"]
        if per_syllable <= 0:
            raise ValueError("oracle has no positive syllable coefficient")
        fixed = self.coefficients["intercept"] + self.coefficients["pauses"] * pause_count + self.margin_sec
        return max(0, int(math.floor((target_sec - fixed) / per_syllable)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_identity": self.engine_identity,
            "features": list(FEATURE_NAMES),
            "coefficients": dict(self.coefficients),
            "sample_count": self.sample_count,
            "mae_sec": self.mae_sec,
            "margin_sec": self.margin_sec,
            "residual_std_sec": self.residual_std_sec,
            "syllables_per_sec": self.syllables_per_sec,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Oracle":
        if tuple(data.get("features", ())) != FEATURE_NAMES:
            raise ValueError("oracle feature set does not match this code version")
        return cls(
            engine_identity=str(data["engine_identity"]),
            coefficients={name: float(data["coefficients"][name]) for name in FEATURE_NAMES},
            sample_count=int(data["sample_count"]),
            mae_sec=float(data["mae_sec"]),
            margin_sec=float(data["margin_sec"]),
            residual_std_sec=float(data["residual_std_sec"]),
            syllables_per_sec=float(data["syllables_per_sec"]),
        )


def fit_oracle(samples: list[tuple[str, float]], engine_identity: str) -> Oracle:
    """Fit non-negative coefficients on (text, measured seconds) pairs."""
    usable = [(text, seconds) for text, seconds in samples
              if isinstance(text, str) and text.strip() and math.isfinite(seconds) and seconds > 0]
    if len(usable) < MIN_SAMPLES:
        raise ValueError(f"oracle needs at least {MIN_SAMPLES} measured cues, got {len(usable)}")
    matrix = np.array([[features(text)[name] for name in FEATURE_NAMES] for text, _ in usable], dtype=float)
    target = np.array([seconds for _, seconds in usable], dtype=float)
    coefficients, _ = nnls(matrix, target)
    predicted = matrix @ coefficients
    residuals = target - predicted
    underestimates = np.clip(residuals, 0.0, None)
    total_syllables = float(matrix[:, FEATURE_NAMES.index("syllables")].sum())
    return Oracle(
        engine_identity=engine_identity,
        coefficients={name: float(value) for name, value in zip(FEATURE_NAMES, coefficients)},
        sample_count=len(usable),
        mae_sec=float(np.mean(np.abs(residuals))),
        margin_sec=float(np.quantile(underestimates, 0.9)),
        residual_std_sec=float(np.std(residuals)),
        syllables_per_sec=total_syllables / float(target.sum()) if target.sum() > 0 else 0.0,
    )
