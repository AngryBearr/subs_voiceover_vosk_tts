"""Elastic timeline solver.

Every spoken line may start a little before or after its subtitle (preferably
``target_lag_ms`` after it, voice-over style; starting early costs ``weight_early`` extra), may be
sped up within a comfort limit (and up to a hard ceiling at a higher price),
and, as a last resort, may lose speech seconds (text shortening).  Lines may
not collide: a line must end ``gap_ms`` before the next line starts.

A collision (overlap with the previous line) carries a fixed penalty plus a
per-second price, so it is chosen only when no shift, tempo or text removal
can avoid it.

The solver is an exact dynamic programme over cues in time order.  The state
after cue *i* is an upper bound on its end time on a fixed millisecond grid.
Costs are additive, so the result is the global minimum for the chosen
weights, not a greedy per-cue decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

EPS = 1e-9


@dataclass(frozen=True)
class TimelinePolicy:
    """Placement limits and the relative price of each kind of adjustment."""

    lead_ms: int = 360
    lag_ms: int = 500
    gap_ms: int = 150
    base_rate: float = 1.25
    tempo_comfort_total: float = 1.5
    tempo_ceiling_total: float = 1.6
    min_speech_sec: float = 0.6
    grid_ms: int = 10
    tempo_step: float = 0.02
    weight_shift: float = 1.0
    weight_tempo: float = 20.0
    weight_tempo_over: float = 200.0
    weight_text: float = 100.0
    weight_collision: float = 1000.0
    weight_collision_fixed: float = 50.0
    target_lag_ms: int = 0  # preferred start after the subtitle start (voice-over lags the original voice)
    weight_early: float = 0.0  # extra price per second of starting before the subtitle start

    def __post_init__(self) -> None:
        for name in ("lead_ms", "lag_ms", "gap_ms", "grid_ms", "target_lag_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.grid_ms == 0:
            raise ValueError("grid_ms must be positive")
        if not 0 < self.base_rate <= self.tempo_comfort_total <= self.tempo_ceiling_total:
            raise ValueError("require 0 < base_rate <= tempo_comfort_total <= tempo_ceiling_total")
        if self.min_speech_sec < 0 or self.tempo_step <= 0:
            raise ValueError("min_speech_sec must be >= 0 and tempo_step > 0")
        if self.target_lag_ms > self.lag_ms or self.weight_early < 0:
            raise ValueError("require target_lag_ms <= lag_ms and weight_early >= 0")

    @property
    def comfort_factor(self) -> float:
        return self.tempo_comfort_total / self.base_rate

    @property
    def ceiling_factor(self) -> float:
        return self.tempo_ceiling_total / self.base_rate

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimelinePolicy":
        known = {field: data[field] for field in cls.__dataclass_fields__ if field in data}
        return cls(**known)


@dataclass(frozen=True)
class Line:
    """One spoken line: where the subtitle starts and how long its speech is."""

    cue_id: str
    start_ms: int
    speech_sec: float


@dataclass(frozen=True)
class Placement:
    """The solver's decision for one line."""

    cue_id: str
    cue_start_ms: int
    start_ms: int
    shift_ms: int
    tempo_factor: float
    total_speed: float
    natural_sec: float
    spoken_sec: float
    removed_sec: float
    target_base_sec: float
    end_ms: int
    collision_ms: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tempo_factors(policy: TimelinePolicy) -> np.ndarray:
    factors = list(np.round(np.arange(1.0, policy.ceiling_factor + EPS, policy.tempo_step), 6))
    if not factors or factors[-1] < policy.ceiling_factor - EPS:
        factors.append(round(policy.ceiling_factor, 6))
    if abs(factors[-1] - policy.comfort_factor) > EPS and all(abs(f - policy.comfort_factor) > EPS for f in factors):
        factors.append(round(policy.comfort_factor, 6))
    return np.array(sorted(set(factors)), dtype=float)


def _tempo_cost(factors: np.ndarray, policy: TimelinePolicy) -> np.ndarray:
    within = np.minimum(factors - 1.0, max(0.0, policy.comfort_factor - 1.0))
    over = np.maximum(0.0, factors - policy.comfort_factor)
    return policy.weight_tempo * within + policy.weight_tempo_over * over


def _status(placement_shift: int, factor: float, removed_sec: float, collision_ms: int,
            policy: TimelinePolicy) -> str:
    if collision_ms > 0:
        return "collision"
    if removed_sec > 0.005:
        return "needs_text"
    if factor > policy.comfort_factor + 1e-6:
        return "over_comfort"
    if factor > 1.0 + 1e-6:
        return "accelerated"
    if placement_shift != 0:
        return "shifted"
    return "fits"


def schedule(lines: list[Line], policy: TimelinePolicy) -> list[Placement]:
    """Place all lines; returns placements in the input order of ``lines``."""
    if not lines:
        return []
    for line in lines:
        if not isinstance(line.start_ms, int) or isinstance(line.start_ms, bool) or line.start_ms < 0:
            raise ValueError(f"{line.cue_id}: start_ms must be a non-negative integer")
        if not line.speech_sec > 0:
            raise ValueError(f"{line.cue_id}: speech_sec must be positive")

    order = sorted(range(len(lines)), key=lambda i: (lines[i].start_ms, i))
    grid = policy.grid_ms
    factors = _tempo_factors(policy)
    tempo_cost = _tempo_cost(factors, policy)
    min_speech_ms = policy.min_speech_sec * 1000.0

    # Per-cue tables kept for backtracking.
    starts_tables: list[np.ndarray] = []
    end_grids: list[np.ndarray] = []
    arg_combo: list[np.ndarray] = []
    arg_prev_end: list[np.ndarray] = []
    prev_end_grid: np.ndarray | None = None
    prev_cost: np.ndarray | None = None

    for position in order:
        line = lines[position]
        starts = np.arange(line.start_ms - policy.lead_ms, line.start_ms + policy.lag_ms + 1, grid, dtype=np.int64)
        starts = starts[starts >= 0] if np.any(starts >= 0) else np.array([line.start_ms], dtype=np.int64)
        n_starts, n_factors = len(starts), len(factors)

        if prev_cost is None or prev_end_grid is None:
            prev_best = np.zeros(n_starts)
            prev_arg = np.full(n_starts, -1, dtype=np.int64)
        else:
            prefix_min = np.minimum.accumulate(prev_cost)
            prefix_arg = np.zeros(len(prev_cost), dtype=np.int64)
            best = 0
            for index in range(len(prev_cost)):
                if prev_cost[index] < prev_cost[best]:
                    best = index
                prefix_arg[index] = best
            penalised = prev_cost + policy.weight_collision * prev_end_grid / 1000.0
            suffix_min = np.minimum.accumulate(penalised[::-1])[::-1]
            suffix_arg = np.zeros(len(penalised), dtype=np.int64)
            best = len(penalised) - 1
            for index in range(len(penalised) - 1, -1, -1):
                if penalised[index] < penalised[best]:
                    best = index
                suffix_arg[index] = best
            boundary = starts - policy.gap_ms
            split = np.searchsorted(prev_end_grid, boundary, side="right")
            no_collision = np.where(split > 0, prefix_min[np.maximum(split - 1, 0)], np.inf)
            collision = np.where(split < len(penalised),
                                 suffix_min[np.minimum(split, len(penalised) - 1)]
                                 + policy.weight_collision * (policy.gap_ms - starts) / 1000.0
                                 + policy.weight_collision_fixed, np.inf)
            use_collision = collision < no_collision
            prev_best = np.where(use_collision, collision, no_collision)
            prev_arg = np.where(use_collision, suffix_arg[np.minimum(split, len(penalised) - 1)],
                                prefix_arg[np.maximum(split - 1, 0)])

        natural_ms = line.speech_sec * 1000.0 / factors                     # (n_factors,)
        natural_end = starts[:, None] + natural_ms[None, :]                   # (n_starts, n_factors)
        removable = np.maximum(0.0, natural_ms - min_speech_ms)[None, :]      # (1, n_factors)
        earliest_end = natural_end - removable
        shift_cost = (policy.weight_shift * np.abs(starts - line.start_ms - policy.target_lag_ms)
                      + policy.weight_early * np.maximum(0, line.start_ms - starts)) / 1000.0
        combo_cost = (prev_best + shift_cost)[:, None] + tempo_cost[None, :]

        end_lo = int(np.floor(earliest_end.min() / grid) * grid)
        end_hi = int(np.ceil(natural_end.max() / grid) * grid)
        end_grid = np.arange(end_lo, end_hi + 1, grid, dtype=np.int64)

        flat_cost = combo_cost.reshape(-1)
        flat_natural = natural_end.reshape(-1)
        flat_earliest = earliest_end.reshape(-1)
        removed = np.maximum(0.0, flat_natural[:, None] - end_grid[None, :]) / 1000.0
        total = flat_cost[:, None] + policy.weight_text * removed
        total = np.where(end_grid[None, :] + EPS >= flat_earliest[:, None], total, np.inf)
        best_combo = np.argmin(total, axis=0)
        cost = total[best_combo, np.arange(len(end_grid))]

        starts_tables.append(starts)
        end_grids.append(end_grid)
        arg_combo.append(best_combo)
        arg_prev_end.append(prev_arg)
        prev_end_grid, prev_cost = end_grid, cost

    assert prev_cost is not None
    # Backtrack from the cheapest final state.
    chosen_end_index = int(np.argmin(prev_cost))
    decisions: list[tuple[int, int, float, float]] = []  # (start, end_bound, factor, natural_ms)
    for table_index in range(len(order) - 1, -1, -1):
        combo = int(arg_combo[table_index][chosen_end_index])
        start_index, factor_index = divmod(combo, len(factors))
        start = int(starts_tables[table_index][start_index])
        factor = float(factors[factor_index])
        natural = lines[order[table_index]].speech_sec * 1000.0 / factor
        decisions.append((start, int(end_grids[table_index][chosen_end_index]), factor, natural))
        chosen_end_index = int(arg_prev_end[table_index][start_index])
    decisions.reverse()

    placements: list[Placement | None] = [None] * len(lines)
    previous_actual_end: int | None = None
    for table_index, position in enumerate(order):
        line = lines[position]
        start, end_bound, factor, natural = decisions[table_index]
        actual_end = min(end_bound, start + natural)
        removed_sec = max(0.0, (start + natural - actual_end) / 1000.0)
        spoken_sec = max(0.0, (actual_end - start) / 1000.0)
        collision_ms = 0
        if previous_actual_end is not None:
            collision_ms = max(0, int(round(previous_actual_end + policy.gap_ms - start)))
        shift = start - line.start_ms
        placements[position] = Placement(
            cue_id=line.cue_id,
            cue_start_ms=line.start_ms,
            start_ms=start,
            shift_ms=shift,
            tempo_factor=factor,
            total_speed=round(factor * policy.base_rate, 4),
            natural_sec=natural / 1000.0,
            spoken_sec=spoken_sec,
            removed_sec=removed_sec,
            target_base_sec=spoken_sec * factor,
            end_ms=int(round(actual_end)),
            collision_ms=collision_ms,
            status=_status(shift - policy.target_lag_ms, factor, removed_sec, collision_ms, policy),
        )
        previous_actual_end = int(round(actual_end))
    return [placement for placement in placements if placement is not None]
