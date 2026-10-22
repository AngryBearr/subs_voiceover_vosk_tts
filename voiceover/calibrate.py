"""Engine-independent tempo: find the engine rate that yields a requested syllables-per-second speed.

The tempo policy is written in syllables per second.  Engines express speed in their own
voice-specific units, so a fixed built-in corpus is synthesised at a few probe rates, a straight
line ``syl_per_sec = a * rate + b`` is fitted, and the rate for the target speed is solved from it.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from voiceover.engine import build_engine
from voiceover.text import plain, syllables
from voiceover.timeline import TimelinePolicy

log = logging.getLogger("voiceover")

# Bump when CALIBRATION_SENTENCES change: it invalidates stored calibrations.
CALIBRATION_VERSION = 1

CALIBRATION_SENTENCES: tuple[str, ...] = (
    "Ну что, пошли?",
    "Я не буду голосовать против тебя.",
    "Сегодня на острове очень жарко.",
    "Нам надо держаться вместе до конца.",
    "Мы проиграли испытание, потому что не доверяли друг другу.",
    "Вечером на совете племени кого-то отправят домой.",
    "Я целый день ничего не ел и мечтаю только о горячем супе.",
    "Если мы объединимся с ними, у нас будет большинство голосов.",
    "Дождь начался ночью, и весь наш лагерь промок насквозь.",
    "Это было самое трудное испытание, которое я когда-либо проходил.",
    "Пожалуйста, не рассказывай никому о нашем разговоре.",
    "Рис почти закончился, а рыба совсем не клюёт.",
    "Я думаю, что они обязательно выберут именно тебя.",
    "Давай сначала найдём дрова, а потом поговорим про союзы.",
    "Мне кажется, он нас обманывает уже несколько дней подряд.",
    "Когда я увидел костёр на берегу, я наконец-то почувствовал себя дома.",
    "Победители получат награду, а проигравшие вернутся в лагерь голодными.",
    "Я не хочу уходить, но если придётся, то уйду с высоко поднятой головой.",
    "Вчера мы весь день строили шалаш, а ночью ветер сломал половину крыши, и нам пришлось начинать всё заново.",
    "Честно говоря, я не ожидал, что наш союз продержится так долго, но теперь мы должны довести дело до конца.",
)


@dataclass(frozen=True)
class RateSample:
    rate: float
    syllables: int
    speech_sec: float
    syl_per_sec: float
    failed: int


@dataclass(frozen=True)
class Calibration:
    calibration_version: int
    engine: dict[str, Any]
    target_syl_per_sec: float
    samples: list[RateSample]
    rate: float
    measured_syl_per_sec: float
    within_tolerance: bool

    def to_dict(self) -> dict[str, Any]:
        return {"calibration_version": self.calibration_version, "engine": self.engine,
                "target_syl_per_sec": self.target_syl_per_sec, "samples": [asdict(s) for s in self.samples],
                "rate": self.rate, "measured_syl_per_sec": self.measured_syl_per_sec,
                "within_tolerance": self.within_tolerance}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Calibration":
        return cls(calibration_version=int(data["calibration_version"]), engine=dict(data["engine"]),
                   target_syl_per_sec=float(data["target_syl_per_sec"]),
                   samples=[RateSample(**s) for s in data["samples"]], rate=float(data["rate"]),
                   measured_syl_per_sec=float(data["measured_syl_per_sec"]),
                   within_tolerance=bool(data["within_tolerance"]))


def engine_identity(engine_config: dict[str, Any]) -> dict[str, Any]:
    """What fixes the voice's speed curve: the synthesis identity without the rate, plus the device."""
    engine: Any = build_engine(engine_config)
    identity = {key: value for key, value in engine.synth_identity().items() if key != "base_rate"}
    identity["device"] = str(engine_config.get("device", "cpu"))
    return identity


def measure_rate(engine_config: dict[str, Any], rate: float, cache_dir: Path, workers: int) -> RateSample:
    """Synthesise the calibration corpus at ``rate`` and measure syllables per second of trimmed speech."""
    engine = build_engine({**engine_config, "base_rate": rate})
    items = [{"cue_id": f"cal-{i}", "index": i + 1, "text": sentence}
             for i, sentence in enumerate(CALIBRATION_SENTENCES)]
    prepared = engine.prepare(items)
    results = engine.synthesize_many(prepared, cache_dir, workers)
    texts = {row.cue_id: row.tts_text for row in prepared}
    total_syllables, total_sec, failed = 0, 0.0, 0
    for result in results:
        if result.error or result.speech_sec <= 0:
            failed += 1
            continue
        total_syllables += syllables(plain(texts[result.cue_id]))
        total_sec += float(result.speech_sec)
    speed = total_syllables / total_sec if total_sec > 0 else 0.0
    return RateSample(rate=rate, syllables=total_syllables, speech_sec=total_sec, syl_per_sec=speed, failed=failed)


def solve_rate(samples: list[RateSample], target_syl_per_sec: float) -> float:
    """Least-squares line ``syl_per_sec = a * rate + b``; return the rate that gives the target."""
    if len(samples) < 2:
        raise ValueError("need at least 2 samples to solve the rate")
    rates = np.array([s.rate for s in samples], dtype=float)
    speeds = np.array([s.syl_per_sec for s in samples], dtype=float)
    slope, intercept = np.polyfit(rates, speeds, 1)
    if not slope > 1e-9:
        raise ValueError(f"speed does not grow with rate (slope {slope:.4f}); cannot calibrate")
    return round(float((target_syl_per_sec - intercept) / slope), 4)


def calibrate(engine_config: dict[str, Any], target_syl_per_sec: float, cache_dir: Path, workers: int,
              probe_rates: tuple[float, ...] = (0.8, 1.1, 1.4), tolerance: float = 0.02,
              max_refinements: int = 2) -> Calibration:
    """Probe, solve, verify at the solved rate, and re-solve (at most ``max_refinements`` times) if off."""
    samples = [measure_rate(engine_config, rate, cache_dir, workers) for rate in probe_rates]
    rate = solve_rate(samples, target_syl_per_sec)
    resolves = 0
    while True:
        sample = measure_rate(engine_config, rate, cache_dir, workers)
        samples.append(sample)
        within = abs(sample.syl_per_sec / target_syl_per_sec - 1.0) <= tolerance
        if within or resolves >= max_refinements:
            break
        resolves += 1
        rate = solve_rate(samples, target_syl_per_sec)
    return Calibration(calibration_version=CALIBRATION_VERSION, engine=engine_identity(engine_config),
                       target_syl_per_sec=target_syl_per_sec, samples=samples, rate=rate,
                       measured_syl_per_sec=sample.syl_per_sec, within_tolerance=within)


SYL_KEYS = ("base_syl_per_sec", "comfort_syl_per_sec", "ceiling_syl_per_sec")


def derive_timeline_policy(policy_config: dict[str, Any], calibration: Calibration) -> TimelinePolicy:
    """Turn the syllables-per-second tempo limits into engine-unit totals at the calibrated rate."""
    base = float(policy_config["base_syl_per_sec"])
    comfort = float(policy_config["comfort_syl_per_sec"])
    ceiling = float(policy_config["ceiling_syl_per_sec"])
    if not 0 < base < comfort <= ceiling:
        raise ValueError("require 0 < base_syl_per_sec < comfort_syl_per_sec <= ceiling_syl_per_sec")
    rest = {k: v for k, v in policy_config.items() if k not in SYL_KEYS}
    return TimelinePolicy.from_dict({**rest, "base_rate": calibration.rate,
                                     "tempo_comfort_total": round(calibration.rate * comfort / base, 4),
                                     "tempo_ceiling_total": round(calibration.rate * ceiling / base, 4)})
