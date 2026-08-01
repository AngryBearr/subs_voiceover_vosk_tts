"""Immutable domain primitives for subtitle shortening timing and budgets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol

from utils.analyze_text import join_text_lines


POST_SILENCE_MEASUREMENT_LABEL = (
    "edge_audio_decoded_with_libsndfile_pcm16_wav_emulation_measured_after_production_silence_reduction"
)


def count_ascii_punctuation(text: str) -> int:
    """Count each ASCII sentence punctuation character in text."""
    return sum(text.count(character) for character in ".,!?:;")


@dataclass(frozen=True)
class TimingWindow:
    """Stable base and effective timing derived from one subtitle item."""

    base_duration_sec: float
    effective_duration_sec: float
    is_checked: bool
    has_source_timing: bool

    @classmethod
    def from_item(cls, item: Mapping[str, Any]) -> "TimingWindow":
        """Derive timing without mutating the source item."""
        analysis = item.get("analysis", {})
        base = analysis.get("available_duration_sec")
        effective = analysis.get("effective_duration_sec")
        estimated = analysis.get("estimated_sec")

        if not isinstance(base, (int, float)) or base <= 0:
            duration = analysis.get("duration_sec")
            ratio = analysis.get("mismatch_ratio")
            if isinstance(duration, (int, float)) and duration > 0:
                base = float(duration)
            elif isinstance(estimated, (int, float)) and isinstance(ratio, (int, float)) and ratio > 0:
                base = float(estimated) / float(ratio)
            else:
                base = max(0.0, (item.get("end", 0) - item.get("start", 0)) / 1000.0)

        if not isinstance(effective, (int, float)) or effective <= 0:
            extended = analysis.get("extended_duration_sec")
            extended_ratio = analysis.get("extended_mismatch_ratio")
            if isinstance(extended, (int, float)) and extended > 0:
                effective = float(extended)
            elif isinstance(estimated, (int, float)) and isinstance(extended_ratio, (int, float)) and extended_ratio > 0:
                effective = float(estimated) / float(extended_ratio)
            else:
                effective = base

        start = item.get("start", 0)
        end = item.get("end", 0)
        return cls(
            base_duration_sec=float(base),
            effective_duration_sec=float(effective),
            is_checked=bool(analysis.get("is_checked")),
            has_source_timing=(end - start) > 0,
        )

    def as_analysis_fields(self) -> dict[str, float]:
        """Return the legacy analysis fields represented by this window."""
        return {
            "available_duration_sec": self.base_duration_sec,
            "effective_duration_sec": self.effective_duration_sec,
        }


class BudgetEstimator(Protocol):
    """Estimate a generation character budget from a timing window."""

    def max_chars(self, window: TimingWindow, target_ratio: float) -> int:
        """Return the maximum generation character count."""


@dataclass(frozen=True)
class CharacterRateBudgetEstimator:
    """Approximate max_chars generation hint based on a character rate."""

    avg_chars_per_sec: float

    def max_chars(self, window: TimingWindow, target_ratio: float) -> int:
        """Calculate the stable character budget using the legacy formula."""
        return max(1, int(window.effective_duration_sec * self.avg_chars_per_sec * target_ratio))


@dataclass(frozen=True)
class CalibratedDurationModel:
    """A physically validated post-silence duration regression model."""

    intercept_sec: float
    seconds_per_budget_char: float
    seconds_per_punctuation: float
    measurement: str
    voice: str
    rate: str
    sample_count: int
    episode_count: int
    validation_method: str
    validation_mae_sec: float
    validation_rmse_sec: float
    validation_r_squared: float

    def __post_init__(self) -> None:
        coefficient_values = (
            ("intercept_sec", self.intercept_sec, lambda value: value >= 0),
            ("seconds_per_budget_char", self.seconds_per_budget_char, lambda value: value > 0),
            ("seconds_per_punctuation", self.seconds_per_punctuation, lambda value: value >= 0),
        )
        for name, value, physical_check in coefficient_values:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not physical_check(value)):
                raise ValueError(f"{name} must be finite and physically valid")
        if not isinstance(self.measurement, str) or self.measurement != POST_SILENCE_MEASUREMENT_LABEL:
            raise ValueError("measurement must be the accepted post-silence measurement")
        if not isinstance(self.voice, str) or not self.voice or not isinstance(self.rate, str) or not self.rate:
            raise ValueError("voice and rate must be nonempty")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 30:
            raise ValueError("sample_count must be at least 30")
        if isinstance(self.episode_count, bool) or not isinstance(self.episode_count, int) or self.episode_count < 2:
            raise ValueError("episode_count must be at least 2")
        if not isinstance(self.validation_method, str) or self.validation_method != "leave_one_episode_out":
            raise ValueError("validation_method must be leave_one_episode_out")
        for name, value in (("validation_mae_sec", self.validation_mae_sec),
                            ("validation_rmse_sec", self.validation_rmse_sec)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (isinstance(self.validation_r_squared, bool)
                or not isinstance(self.validation_r_squared, (int, float))
                or not math.isfinite(self.validation_r_squared)
                or not 0 <= self.validation_r_squared <= 1):
            raise ValueError("validation_r_squared must be finite and between 0 and 1")

    def predict_seconds(self, text: Any) -> float:
        joined = join_text_lines(text)
        return (self.intercept_sec + self.seconds_per_budget_char * len(joined)
                + self.seconds_per_punctuation * count_ascii_punctuation(joined))


@dataclass(frozen=True)
class DurationSelectionPolicy:
    """Selection policy using a calibrated duration model."""

    model: CalibratedDurationModel
    fit_ratio: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.fit_ratio, bool) or not math.isfinite(self.fit_ratio) or self.fit_ratio <= 0:
            raise ValueError("fit_ratio must be finite and positive")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def load_calibrated_duration_profile(path: Path) -> CalibratedDurationModel:
    """Load and strictly validate the calibrated profile schema."""
    try:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid calibrated duration profile: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("profile must be a JSON object")
    required = {"schema_version", "profile_type", "measurement", "voice", "rate",
                "coefficients", "training", "validation"}
    if set(raw) != required:
        raise ValueError("profile has missing or extra fields")
    if isinstance(raw["schema_version"], bool) or not isinstance(raw["schema_version"], int) \
            or raw["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if raw["profile_type"] != "calibrated_post_silence_duration":
        raise ValueError("profile_type is unsupported")
    if raw["measurement"] != POST_SILENCE_MEASUREMENT_LABEL:
        raise ValueError("profile measurement is not the accepted post-silence measurement")
    if not isinstance(raw["voice"], str) or not raw["voice"] or not isinstance(raw["rate"], str) or not raw["rate"]:
        raise ValueError("voice and rate must be nonempty strings")
    coefficients = raw["coefficients"]
    training = raw["training"]
    validation = raw["validation"]
    if not isinstance(coefficients, dict) or set(coefficients) != {
            "intercept_sec", "seconds_per_budget_char", "seconds_per_punctuation"}:
        raise ValueError("coefficients must contain exactly the model keys")
    if not isinstance(training, dict) or set(training) != {"sample_count", "episode_count"}:
        raise ValueError("training must contain exactly sample_count and episode_count")
    if not isinstance(validation, dict) or set(validation) != {"method", "mae_sec", "rmse_sec", "r_squared"}:
        raise ValueError("validation must contain exactly method, mae_sec, rmse_sec, and r_squared")
    sample_count = training["sample_count"]
    episode_count = training["episode_count"]
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 30:
        raise ValueError("training.sample_count must be an integer at least 30")
    if isinstance(episode_count, bool) or not isinstance(episode_count, int) or episode_count < 2:
        raise ValueError("training.episode_count must be an integer at least 2")
    if validation["method"] != "leave_one_episode_out":
        raise ValueError("validation.method must be leave_one_episode_out")
    return CalibratedDurationModel(
        intercept_sec=_number(coefficients["intercept_sec"], "intercept_sec"),
        seconds_per_budget_char=_number(coefficients["seconds_per_budget_char"], "seconds_per_budget_char"),
        seconds_per_punctuation=_number(coefficients["seconds_per_punctuation"], "seconds_per_punctuation"),
        measurement=raw["measurement"], voice=raw["voice"], rate=raw["rate"],
        sample_count=sample_count, episode_count=episode_count,
        validation_method=validation["method"],
        validation_mae_sec=_number(validation["mae_sec"], "validation.mae_sec"),
        validation_rmse_sec=_number(validation["rmse_sec"], "validation.rmse_sec"),
        validation_r_squared=_number(validation["r_squared"], "validation.r_squared"),
    )
