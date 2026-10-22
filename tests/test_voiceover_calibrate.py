from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from voiceover import calibrate as cal
from voiceover import pipeline
from voiceover.calibrate import Calibration, RateSample, calibrate, derive_timeline_policy, solve_rate
from voiceover.text import syllables


def sample(rate: float, speed: float) -> RateSample:
    return RateSample(rate=rate, syllables=100, speech_sec=100 / speed, syl_per_sec=speed, failed=0)


def make_calibration(rate: float = 1.1) -> Calibration:
    return Calibration(calibration_version=1, engine={"voice": 49}, target_syl_per_sec=5.5,
                       samples=[sample(rate, 5.5)], rate=rate, measured_syl_per_sec=5.5, within_tolerance=True)


def test_corpus_shape() -> None:
    sentences = cal.CALIBRATION_SENTENCES
    assert len(sentences) == 20 and len(set(sentences)) == 20
    counts = [syllables(s) for s in sentences]
    assert min(counts) <= 7 and max(counts) >= 30
    assert not any(ch.isdigit() or ("a" <= ch.lower() <= "z") for s in sentences for ch in s)


def test_solve_rate_on_linear_samples() -> None:
    samples = [sample(r, 4.0 * r + 1.0) for r in (0.8, 1.1, 1.4)]
    assert solve_rate(samples, 5.5) == pytest.approx(1.125, abs=1e-3)


def test_solve_rate_refuses_bad_input() -> None:
    with pytest.raises(ValueError):
        solve_rate([sample(1.0, 5.0)], 5.5)
    with pytest.raises(ValueError):
        solve_rate([sample(0.8, 5.0), sample(1.4, 5.0)], 5.5)
    with pytest.raises(ValueError):
        solve_rate([sample(0.8, 6.0), sample(1.4, 5.0)], 5.5)


def patch_measure(monkeypatch: pytest.MonkeyPatch, law: Any) -> list[float]:
    calls: list[float] = []

    def fake(engine_config: dict[str, Any], rate: float, cache_dir: Path, workers: int) -> RateSample:
        calls.append(rate)
        return sample(rate, law(rate))

    monkeypatch.setattr(cal, "measure_rate", fake)
    monkeypatch.setattr(cal, "engine_identity", lambda config: {"voice": config["voice"]})
    return calls


def test_calibrate_linear_law_needs_no_refinement(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_measure(monkeypatch, lambda r: 4.0 * r + 1.0)
    result = calibrate({"voice": 49}, 5.5, tmp_path, 1)
    assert len(calls) == 4 and result.rate == pytest.approx(1.125, abs=1e-3)
    assert result.within_tolerance and result.measured_syl_per_sec == pytest.approx(5.5, rel=0.01)
    assert result.engine == {"voice": 49}
    assert [s.rate for s in result.samples[:3]] == [0.8, 1.1, 1.4]
    assert set(result.to_dict()) == {"calibration_version", "engine", "target_syl_per_sec", "samples", "rate",
                                     "measured_syl_per_sec", "within_tolerance"}


def test_calibrate_refines_when_first_solve_is_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Curved law: the straight-line fit over the probes misses the target by more than 2 %.
    calls = patch_measure(monkeypatch, lambda r: 3.0 * r * r + 1.0)
    result = calibrate({"voice": 49}, 5.5, tmp_path, 1)
    assert len(calls) > 4
    assert result.within_tolerance
    assert 3.0 * result.rate ** 2 + 1.0 == pytest.approx(5.5, rel=0.02)


def test_calibrate_gives_up_after_max_refinements(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = patch_measure(monkeypatch, lambda r: 3.0 * r * r + 1.0)
    result = calibrate({"voice": 49}, 5.5, tmp_path, 1, tolerance=1e-9, max_refinements=1)
    assert len(calls) == 3 + 1 + 1 and not result.within_tolerance


def test_derive_timeline_policy_numbers() -> None:
    config = {"base_syl_per_sec": 5.5, "comfort_syl_per_sec": 6.3, "ceiling_syl_per_sec": 6.7, "lead_ms": 300}
    policy = derive_timeline_policy(config, make_calibration(1.1))
    assert policy.base_rate == 1.1
    assert policy.tempo_comfort_total == pytest.approx(1.26, abs=1e-4)
    assert policy.tempo_ceiling_total == pytest.approx(1.34, abs=1e-4)
    assert policy.lead_ms == 300


@pytest.mark.parametrize("base,comfort,ceiling", [(6.3, 6.3, 6.7), (5.5, 6.7, 6.3), (0.0, 6.3, 6.7), (-1.0, 6.3, 6.7)])
def test_derive_timeline_policy_validates(base: float, comfort: float, ceiling: float) -> None:
    config = {"base_syl_per_sec": base, "comfort_syl_per_sec": comfort, "ceiling_syl_per_sec": ceiling}
    with pytest.raises(ValueError):
        derive_timeline_policy(config, make_calibration())


def test_stage_calibrate_writes_then_skips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runs: list[int] = []

    def fake(engine_config: dict[str, Any], target: float, cache_dir: Path, workers: int, **kwargs: Any) -> Calibration:
        runs.append(1)
        return make_calibration(1.2)

    monkeypatch.setattr(pipeline, "calibrate", fake)
    monkeypatch.setattr(pipeline, "engine_identity", lambda config: {"voice": config["voice"]})
    policy = {"base_syl_per_sec": 5.5, "comfort_syl_per_sec": 6.3, "ceiling_syl_per_sec": 6.7}
    first = pipeline.stage_calibrate({"voice": 49}, policy, tmp_path / "cache", 1, tmp_path)
    assert first is not None and first.rate == 1.2 and (tmp_path / "calibration.json").exists()
    second = pipeline.stage_calibrate({"voice": 49}, policy, tmp_path / "cache", 1, tmp_path)
    assert second == first and len(runs) == 1
    pipeline.stage_calibrate({"voice": 56}, policy, tmp_path / "cache", 1, tmp_path)
    assert len(runs) == 2


def test_stage_calibrate_is_none_for_legacy_policy(tmp_path: Path) -> None:
    legacy = {"base_rate": 1.4, "tempo_comfort_total": 1.6, "tempo_ceiling_total": 1.7}
    assert pipeline.stage_calibrate({"voice": 8}, legacy, tmp_path / "cache", 1, tmp_path) is None
    assert not (tmp_path / "calibration.json").exists()
