from __future__ import annotations

import random

import pytest

from voiceover.oracle import Oracle, fit_oracle
from voiceover.text import features, join_lines, pauses, syllables, words


def test_syllables_count_vowels_ignoring_stress() -> None:
    assert syllables("Приве+т, ми+р!") == 3
    assert syllables("ЁЖ") == 1
    assert syllables("...") == 0


def test_words_and_pauses() -> None:
    assert words("Им дали 2 минуты, честно...") == 5
    assert pauses("Да, нет... Что?") == 3
    assert pauses("Он сказал - уходи") == 1


def test_join_lines_handles_lists() -> None:
    assert join_lines(["Привет", "мир"]) == "Привет мир"
    assert join_lines(None) == ""


def test_features_keys() -> None:
    assert set(features("абв")) == {"intercept", "syllables", "pauses"}


def _synthetic_samples(intercept: float, per_syllable: float, per_pause: float, n: int = 60) -> list[tuple[str, float]]:
    rng = random.Random(1)
    samples = []
    for _ in range(n):
        syllable_count = rng.randint(2, 40)
        pause_count = rng.randint(0, 3)
        text = "ла " * syllable_count + ", " * pause_count
        seconds = intercept + per_syllable * syllable_count + per_pause * pause_count + rng.uniform(0, 0.1)
        samples.append((text, seconds))
    return samples


def test_fit_oracle_recovers_coefficients_and_roundtrips() -> None:
    oracle = fit_oracle(_synthetic_samples(0.5, 0.2, 0.15), "engine-x")
    assert oracle.coefficients["intercept"] == pytest.approx(0.55, abs=0.08)
    assert oracle.coefficients["syllables"] == pytest.approx(0.2, abs=0.01)
    assert oracle.coefficients["pauses"] == pytest.approx(0.15, abs=0.05)
    assert 0 <= oracle.margin_sec <= 0.12
    restored = Oracle.from_dict(oracle.to_dict())
    assert restored == oracle
    assert restored.predict("ла " * 10) == pytest.approx(oracle.predict("ла " * 10))
    assert restored.syllable_capacity(2.0) == int((2.0 - oracle.coefficients["intercept"] - oracle.coefficients["pauses"]
                                                   - oracle.margin_sec) / oracle.coefficients["syllables"])


def test_fit_oracle_requires_enough_samples() -> None:
    with pytest.raises(ValueError):
        fit_oracle(_synthetic_samples(0.5, 0.2, 0.1, n=5), "engine-x")
