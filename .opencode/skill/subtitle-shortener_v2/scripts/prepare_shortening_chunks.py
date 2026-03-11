from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .subtitle_shortening_common import (
        DEFAULT_CONTEXT_HALO,
        DEFAULT_MAX_ITEMS_PER_CHUNK,
        DEFAULT_MAX_MISMATCH_RATIO,
        DEFAULT_STAGE_COUNT,
        DEFAULT_TARGET_BRIDGE_GAP,
        ProblemCluster,
        build_soft_joint_groups,
        calculate_staged_target_ratio,
        coerce_int,
        coerce_text_lines,
        discover_problem_clusters,
        effective_ratio,
        item_index,
        load_json,
        make_temp_workdir_name,
        pack_clusters_by_item_count,
        save_json,
    )
except ImportError:
    from subtitle_shortening_common import (
        DEFAULT_CONTEXT_HALO,
        DEFAULT_MAX_ITEMS_PER_CHUNK,
        DEFAULT_MAX_MISMATCH_RATIO,
        DEFAULT_STAGE_COUNT,
        DEFAULT_TARGET_BRIDGE_GAP,
        ProblemCluster,
        build_soft_joint_groups,
        calculate_staged_target_ratio,
        coerce_int,
        coerce_text_lines,
        discover_problem_clusters,
        effective_ratio,
        item_index,
        load_json,
        make_temp_workdir_name,
        pack_clusters_by_item_count,
        save_json,
    )


def main(argv: list[str] | None = None) -> int:
    """Prepare chunk files for one iterative shortening pass."""

    parser = argparse.ArgumentParser(
        description="Prepare iterative subtitle-shortening chunks from analyzed JSON."
    )
    parser.add_argument("input", help="Input JSON produced by analyze_text.py")
    parser.add_argument(
        "--workdir",
        help="Optional workflow directory. If omitted, a temp directory is created next to the input.",
    )
    parser.add_argument("--iteration", type=int, default=1, help="1-based iteration index")
    parser.add_argument(
        "--stage-count",
        type=int,
        default=DEFAULT_STAGE_COUNT,
        help=f"Planned shortening stages (default: {DEFAULT_STAGE_COUNT}).",
    )
    parser.add_argument(
        "--max-mismatch-ratio",
        type=float,
        default=DEFAULT_MAX_MISMATCH_RATIO,
        help=f"Active target threshold (default: {DEFAULT_MAX_MISMATCH_RATIO}).",
    )
    parser.add_argument(
        "--context-halo",
        type=int,
        default=DEFAULT_CONTEXT_HALO,
        help=f"Neighbor items included around targets (default: {DEFAULT_CONTEXT_HALO}).",
    )
    parser.add_argument(
        "--target-bridge-gap",
        type=int,
        default=DEFAULT_TARGET_BRIDGE_GAP,
        help=f"Allowed non-target bridge size inside one cluster (default: {DEFAULT_TARGET_BRIDGE_GAP}).",
    )
    parser.add_argument(
        "--max-items-per-chunk",
        type=int,
        default=DEFAULT_MAX_ITEMS_PER_CHUNK,
        help=f"Maximum unique source items per chunk (default: {DEFAULT_MAX_ITEMS_PER_CHUNK}).",
    )
    parser.add_argument(
        "--temp-suffix",
        default="_temp",
        help="Suffix used when auto-generating a workdir name.",
    )
    args = parser.parse_args(argv)

    _validate_args(args)

    input_path = Path(args.input).resolve()
    items = _require_item_list(load_json(input_path))
    workdir = _resolve_workdir(input_path, args.workdir, args.temp_suffix)
    iteration_dir = workdir / f"iter_{args.iteration:03d}"
    chunks_dir = iteration_dir / "chunks"
    _prepare_iteration_dir(iteration_dir, chunks_dir)

    clusters = discover_problem_clusters(
        items,
        context_halo=args.context_halo,
        target_bridge_gap=args.target_bridge_gap,
        max_ratio=args.max_mismatch_ratio,
    )
    packed_clusters = pack_clusters_by_item_count(clusters, max_items=args.max_items_per_chunk)

    seen_target_indices: set[int] = set()
    manifest_chunks: list[dict[str, Any]] = []
    total_target_items = 0
    total_context_items = 0

    for chunk_id, cluster_group in enumerate(packed_clusters, start=1):
        chunk_payload = _build_chunk_payload(
            items=items,
            clusters=cluster_group,
            seen_target_indices=seen_target_indices,
            iteration=args.iteration,
            stage_count=args.stage_count,
            max_ratio=args.max_mismatch_ratio,
            chunk_id=chunk_id,
        )
        if chunk_payload is None:
            continue

        chunk_path = chunks_dir / f"chunk_{chunk_id:03d}.json"
        save_json(chunk_path, chunk_payload)

        target_count = int(chunk_payload["target_count"])
        context_count = int(chunk_payload["context_count"])
        total_target_items += target_count
        total_context_items += context_count
        manifest_chunks.append(
            {
                "chunk_id": chunk_id,
                "path": str(chunk_path.relative_to(iteration_dir)),
                "item_count": len(chunk_payload["items"]),
                "target_count": target_count,
                "context_count": context_count,
                "indices": chunk_payload["indices"],
                "target_indices": chunk_payload["target_indices"],
                "soft_joint_group_count": len(chunk_payload["soft_joint_groups"]),
            }
        )

    manifest = {
        "iteration": args.iteration,
        "stage_count": args.stage_count,
        "source": str(input_path),
        "workdir": str(workdir),
        "max_mismatch_ratio": args.max_mismatch_ratio,
        "context_halo": args.context_halo,
        "target_bridge_gap": args.target_bridge_gap,
        "max_items_per_chunk": args.max_items_per_chunk,
        "cluster_count": len(clusters),
        "chunk_count": len(manifest_chunks),
        "target_count": total_target_items,
        "context_count": total_context_items,
        "processed_indices": sorted(seen_target_indices),
        "processed_target_indices": sorted(seen_target_indices),
        "clusters": [asdict(cluster) for cluster in clusters],
        "chunks": manifest_chunks,
    }
    save_json(iteration_dir / "manifest.json", manifest)

    print(
        "[prepare] "
        f"iteration={args.iteration} "
        f"targets={total_target_items} "
        f"context={total_context_items} "
        f"clusters={len(clusters)} "
        f"chunks={len(manifest_chunks)} "
        f"workdir={workdir}"
    )
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.iteration <= 0:
        raise ValueError("iteration must be > 0")
    if args.stage_count <= 0:
        raise ValueError("stage_count must be > 0")
    if args.context_halo < 0:
        raise ValueError("context_halo must be >= 0")
    if args.target_bridge_gap < 0:
        raise ValueError("target_bridge_gap must be >= 0")
    if args.max_items_per_chunk <= 0:
        raise ValueError("max_items_per_chunk must be > 0")


def _require_item_list(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of subtitle items")

    items: list[dict[str, Any]] = []
    for value in data:
        if not isinstance(value, Mapping):
            raise ValueError("Each subtitle item must be an object")
        items.append(dict(value))
    return items


def _resolve_workdir(input_path: Path, workdir_value: str | None, temp_suffix: str) -> Path:
    if workdir_value:
        return Path(workdir_value).resolve()
    auto_name = make_temp_workdir_name(f"{input_path.stem}_shortening", temp_suffix=temp_suffix)
    return (input_path.parent / auto_name).resolve()


def _prepare_iteration_dir(iteration_dir: Path, chunks_dir: Path) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = iteration_dir / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    for existing_chunk in chunks_dir.glob("chunk_*.json"):
        existing_chunk.unlink()


def _build_chunk_payload(
    *,
    items: list[dict[str, Any]],
    clusters: list[ProblemCluster],
    seen_target_indices: set[int],
    iteration: int,
    stage_count: int,
    max_ratio: float,
    chunk_id: int,
) -> dict[str, Any] | None:
    target_positions = {position for cluster in clusters for position in cluster.target_positions}
    chunk_positions = sorted({position for cluster in clusters for position in cluster.item_positions})
    soft_joint_groups = build_soft_joint_groups(clusters)

    chunk_items: list[dict[str, Any]] = []
    indices: list[int] = []
    target_indices: list[int] = []
    target_count = 0
    context_count = 0

    for position in chunk_positions:
        item = items[position]
        source_index = item_index(item, position)
        role = "target" if position in target_positions else "context"
        if role == "target" and source_index in seen_target_indices:
            continue

        chunk_items.append(
            _build_chunk_item(
                item=item,
                position=position,
                source_index=source_index,
                role=role,
                iteration=iteration,
                stage_count=stage_count,
                max_ratio=max_ratio,
            )
        )
        indices.append(source_index)

        if role == "target":
            seen_target_indices.add(source_index)
            target_indices.append(source_index)
            target_count += 1
        else:
            context_count += 1

    if not chunk_items or target_count == 0:
        return None

    return {
        "schema": "subtitle-shortening-chunk/v1",
        "iteration": iteration,
        "chunk_id": chunk_id,
        "target_count": target_count,
        "context_count": context_count,
        "indices": indices,
        "target_indices": target_indices,
        "soft_joint_groups": soft_joint_groups,
        "items": chunk_items,
    }


def _build_chunk_item(
    *,
    item: Mapping[str, Any],
    position: int,
    source_index: int,
    role: str,
    iteration: int,
    stage_count: int,
    max_ratio: float,
) -> dict[str, Any]:
    prepared: dict[str, Any] = {
        "index": source_index,
        "position": position,
        "role": role,
        "text": coerce_text_lines(item.get("text")),
    }

    start = coerce_int(item.get("start"))
    end = coerce_int(item.get("end"))
    duration = coerce_int(item.get("duration"))
    if duration is None and start is not None and end is not None:
        duration = end - start

    if start is not None:
        prepared["start"] = start
    if end is not None:
        prepared["end"] = end
    if duration is not None:
        prepared["duration"] = duration

    ratio = effective_ratio(item)
    if ratio is not None:
        prepared["effective_ratio"] = round(ratio, 4)

    if role == "target":
        target_ratio = calculate_staged_target_ratio(
            ratio,
            max_ratio=max_ratio,
            stage_index=iteration,
            stage_count=stage_count,
        )
        if target_ratio is not None:
            prepared["target_ratio"] = round(target_ratio, 4)
        prepared["edit_rule"] = "Rewrite this item only if needed to reach target_ratio."
    else:
        prepared["edit_rule"] = "Context only. Do not edit in result files."

    return prepared


if __name__ == "__main__":
    raise SystemExit(main())
