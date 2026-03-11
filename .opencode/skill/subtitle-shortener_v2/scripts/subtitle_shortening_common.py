from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


PathLike = str | Path

DEFAULT_MAX_ITEMS_PER_CHUNK = 75
DEFAULT_CONTEXT_HALO = 2
DEFAULT_TARGET_BRIDGE_GAP = 1
DEFAULT_STAGE_COUNT = 3
DEFAULT_MAX_MISMATCH_RATIO = 1.5


@dataclass(frozen=True)
class ProblemCluster:
    """Cluster of nearby target items with context positions."""

    target_positions: tuple[int, ...]
    item_positions: tuple[int, ...]


def load_json(path: PathLike) -> Any:
    """Load UTF-8 JSON with BOM tolerance."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_json(path: PathLike, data: Any) -> None:
    """Save JSON with stable formatting."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def coerce_text_lines(text_field: Any) -> list[str]:
    """Normalize subtitle text to a list of non-empty lines."""

    if text_field is None:
        return []
    if isinstance(text_field, list):
        return [str(part).strip() for part in text_field if str(part).strip()]

    text_value = str(text_field)
    if not text_value.strip():
        return []

    split_lines = [line.strip() for line in text_value.splitlines() if line.strip()]
    if split_lines:
        return split_lines
    return [text_value.strip()]


def join_text_lines(text_field: Any, separator: str = " ") -> str:
    """Join subtitle text lines into one string."""

    return separator.join(coerce_text_lines(text_field)).strip()


def effective_ratio(item: Mapping[str, Any]) -> float | None:
    """Return the best available ratio from item analysis."""

    analysis = item.get("analysis")
    if not isinstance(analysis, Mapping):
        return None

    extended_ratio = coerce_float(analysis.get("extended_mismatch_ratio"))
    if extended_ratio is not None:
        return extended_ratio
    return coerce_float(analysis.get("mismatch_ratio"))


def calculate_staged_target_ratio(
    current_ratio: float | None,
    max_ratio: float,
    stage_index: int,
    stage_count: int,
) -> float | None:
    """Move the requested target ratio toward the final threshold."""

    if current_ratio is None:
        return None
    if stage_count <= 0:
        raise ValueError("stage_count must be > 0")
    if stage_index <= 0:
        raise ValueError("stage_index must be > 0")

    capped_stage = min(stage_index, stage_count)
    start_ratio = max(current_ratio, max_ratio)
    progress = capped_stage / stage_count
    return start_ratio - ((start_ratio - max_ratio) * progress)


def discover_problem_clusters(
    items: Sequence[Mapping[str, Any]],
    *,
    context_halo: int = DEFAULT_CONTEXT_HALO,
    target_bridge_gap: int = DEFAULT_TARGET_BRIDGE_GAP,
    max_ratio: float | None = None,
) -> list[ProblemCluster]:
    """Find nearby over-threshold items and expand them with context."""

    if context_halo < 0:
        raise ValueError("context_halo must be >= 0")
    if target_bridge_gap < 0:
        raise ValueError("target_bridge_gap must be >= 0")

    target_positions = [
        position
        for position, item in enumerate(items)
        if is_target_item(item, max_ratio=max_ratio)
    ]
    if not target_positions:
        return []

    clusters: list[ProblemCluster] = []
    group_start = 0
    for idx in range(1, len(target_positions) + 1):
        at_end = idx == len(target_positions)
        if not at_end and target_positions[idx] <= target_positions[idx - 1] + target_bridge_gap + 1:
            continue

        group_targets = tuple(target_positions[group_start:idx])
        halo_start = max(0, group_targets[0] - context_halo)
        halo_end = min(len(items) - 1, group_targets[-1] + context_halo)
        clusters.append(
            ProblemCluster(
                target_positions=group_targets,
                item_positions=tuple(range(halo_start, halo_end + 1)),
            )
        )
        group_start = idx
    return clusters


def pack_clusters_by_item_count(
    clusters: Sequence[ProblemCluster],
    *,
    max_items: int,
) -> list[list[ProblemCluster]]:
    """Greedily pack clusters into chunks by unique item count."""

    if max_items <= 0:
        raise ValueError("max_items must be > 0")

    packed: list[list[ProblemCluster]] = []
    current_chunk: list[ProblemCluster] = []
    current_positions: set[int] = set()

    for cluster in clusters:
        cluster_positions = set(cluster.item_positions)
        if not current_chunk:
            current_chunk = [cluster]
            current_positions = cluster_positions
            continue

        merged_positions = current_positions | cluster_positions
        if len(merged_positions) <= max_items:
            current_chunk.append(cluster)
            current_positions = merged_positions
            continue

        packed.append(current_chunk)
        current_chunk = [cluster]
        current_positions = cluster_positions

    if current_chunk:
        packed.append(current_chunk)
    return packed


def build_soft_joint_groups(clusters: Sequence[ProblemCluster]) -> list[dict[str, Any]]:
    """Build concise hints for multi-target local rewrites."""

    groups: list[dict[str, Any]] = []
    for cluster in clusters:
        if len(cluster.target_positions) < 2:
            continue
        groups.append(
            {
                "target_positions": list(cluster.target_positions),
                "item_positions": list(cluster.item_positions),
                "hint": "These target items likely form one thought. Prefer a soft joint rewrite before splitting back.",
            }
        )
    return groups


def make_temp_workdir_name(prefix: str, temp_suffix: str = "_temp") -> str:
    """Build a timestamped temporary workdir name."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{temp_suffix}"


def item_index(item: Mapping[str, Any], position: int) -> int:
    """Return stable source index for an item."""

    index_value = coerce_int(item.get("index"))
    if index_value is not None:
        return index_value
    return position + 1


def coerce_int(value: Any) -> int | None:
    """Coerce a value to int when possible."""

    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_float(value: Any) -> float | None:
    """Coerce a value to float when possible."""

    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def is_target_item(item: Mapping[str, Any], *, max_ratio: float | None) -> bool:
    """Select targets by the active threshold, not stale flags."""

    ratio = effective_ratio(item)
    if ratio is not None and max_ratio is not None:
        return ratio > max_ratio

    analysis = item.get("analysis")
    return isinstance(analysis, Mapping) and bool(analysis.get("is_critical"))
