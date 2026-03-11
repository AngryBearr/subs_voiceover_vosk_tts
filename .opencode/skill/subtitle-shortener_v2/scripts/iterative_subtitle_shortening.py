from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .analyze_text import DEFAULT_AVG_CHARS_PER_SEC, analyze_subtitles
    from .merge_shortening_results import (
        load_edit_entries,
        load_source_items,
        merge_edits_with_report,
    )
    from .prepare_shortening_chunks import main as prepare_shortening_main
    from .subtitle_shortening_common import (
        DEFAULT_CONTEXT_HALO,
        DEFAULT_MAX_ITEMS_PER_CHUNK,
        DEFAULT_MAX_MISMATCH_RATIO,
        DEFAULT_STAGE_COUNT,
        DEFAULT_TARGET_BRIDGE_GAP,
        coerce_int,
        effective_ratio,
        item_index,
        load_json,
        make_temp_workdir_name,
        save_json,
    )
except ImportError:
    from analyze_text import DEFAULT_AVG_CHARS_PER_SEC, analyze_subtitles
    from merge_shortening_results import (
        load_edit_entries,
        load_source_items,
        merge_edits_with_report,
    )
    from prepare_shortening_chunks import main as prepare_shortening_main
    from subtitle_shortening_common import (
        DEFAULT_CONTEXT_HALO,
        DEFAULT_MAX_ITEMS_PER_CHUNK,
        DEFAULT_MAX_MISMATCH_RATIO,
        DEFAULT_STAGE_COUNT,
        DEFAULT_TARGET_BRIDGE_GAP,
        coerce_int,
        effective_ratio,
        item_index,
        load_json,
        make_temp_workdir_name,
        save_json,
    )


RESULTS_GLOB = "*.result.json"


@dataclass(frozen=True)
class CollectedEdits:
    """Collected result payloads for one iteration."""

    edit_files: list[Path]
    edit_count: int


def main(argv: list[str] | None = None) -> int:
    """Run the self-contained iterative subtitle shortening workflow."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)

    input_path = Path(args.input).resolve()
    final_json_path = _resolve_final_json_path(input_path, args.output)
    final_report_path = _resolve_final_report_path(final_json_path, args.final_report)
    workdir = _resolve_workdir(input_path, args.workdir, args.temp_suffix)
    _validate_final_paths(
        workdir=workdir,
        final_json_path=final_json_path,
        final_report_path=final_report_path,
        cleanup_temp=args.cleanup_temp,
    )
    workdir.mkdir(parents=True, exist_ok=True)

    initial_json_path = workdir / "initial_analyzed.json"
    if not initial_json_path.exists():
        analyzed_items = _reanalyze_items(
            load_source_items(input_path),
            avg_chars_per_sec=args.avg_chars_per_sec,
            max_mismatch_ratio=args.max_mismatch_ratio,
        )
        save_json(initial_json_path, analyzed_items)

    current_json_path = _find_resume_point(workdir, initial_json_path)
    iteration = _next_iteration_number(workdir)

    while iteration <= args.max_iterations:
        iteration_dir = workdir / f"iter_{iteration:03d}"
        manifest_path = iteration_dir / "manifest.json"
        if not manifest_path.exists():
            _prepare_iteration(
                current_json_path=current_json_path,
                workdir=workdir,
                iteration=iteration,
                stage_count=args.stage_count,
                max_mismatch_ratio=args.max_mismatch_ratio,
                context_halo=args.context_halo,
                target_bridge_gap=args.target_bridge_gap,
                max_items_per_chunk=args.max_items_per_chunk,
            )

        manifest = _load_manifest(manifest_path)
        _ensure_iteration_instructions(iteration_dir=iteration_dir, manifest=manifest)

        if int(manifest.get("chunk_count", 0)) == 0:
            _finalize_success(
                input_path=input_path,
                workdir=workdir,
                current_json_path=current_json_path,
                final_json_path=final_json_path,
                final_report_path=final_report_path,
                max_mismatch_ratio=args.max_mismatch_ratio,
                cleanup_temp=args.cleanup_temp,
            )
            print(
                "[done] "
                f"iterations={iteration - 1} "
                f"final_json={final_json_path} "
                f"final_report={final_report_path}"
            )
            return 0

        collected = _collect_edit_results(iteration_dir / "results")
        if collected.edit_count == 0:
            print(
                "[awaiting-edits] "
                f"iteration={iteration} "
                f"chunks={manifest.get('chunk_count', 0)} "
                f"results_dir={iteration_dir / 'results'} "
                f"instructions={iteration_dir / 'instructions.txt'}"
            )
            return 0

        current_items = load_source_items(current_json_path)
        _validate_result_targets(iteration_dir=iteration_dir, edit_files=collected.edit_files)
        merged_items, merge_report = merge_edits_with_report(current_items, collected.edit_files)
        if not merge_report["applied"]:
            print(
                "[no-effective-edits] "
                f"iteration={iteration} "
                f"results_dir={iteration_dir / 'results'}"
            )
            return 0

        merged_path = iteration_dir / "merged.json"
        save_json(merged_path, merged_items)

        reanalyzed_items = _reanalyze_items(
            merged_items,
            avg_chars_per_sec=args.avg_chars_per_sec,
            max_mismatch_ratio=args.max_mismatch_ratio,
        )
        reanalyzed_path = iteration_dir / "reanalyzed.json"
        save_json(reanalyzed_path, reanalyzed_items)

        report = _build_iteration_report(
            iteration=iteration,
            source_json_path=current_json_path,
            output_json_path=reanalyzed_path,
            current_items=current_items,
            updated_items=reanalyzed_items,
            manifest=manifest,
            merge_report=merge_report,
        )
        save_json(iteration_dir / "report.json", report)

        current_json_path = reanalyzed_path
        if int(report["remaining_critical_count"]) == 0:
            _finalize_success(
                input_path=input_path,
                workdir=workdir,
                current_json_path=current_json_path,
                final_json_path=final_json_path,
                final_report_path=final_report_path,
                max_mismatch_ratio=args.max_mismatch_ratio,
                cleanup_temp=args.cleanup_temp,
            )
            print(
                "[done] "
                f"iterations={iteration} "
                f"final_json={final_json_path} "
                f"final_report={final_report_path}"
            )
            return 0

        iteration += 1

    print(
        "[stopped] "
        f"reason=max_iterations_reached "
        f"max_iterations={args.max_iterations} "
        f"workdir={workdir}"
    )
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the iterative subtitle shortening workflow without project dependencies."
    )
    parser.add_argument("input", help="Input subtitles JSON. Analysis is refreshed inside the workflow.")
    parser.add_argument(
        "--workdir",
        help="Optional workflow directory. If omitted, a timestamped temp folder is created next to the input.",
    )
    parser.add_argument(
        "--output",
        help="Final shortened JSON path. Defaults next to the input file.",
    )
    parser.add_argument(
        "--final-report",
        help="Final workflow report path. Defaults next to the final JSON.",
    )
    parser.add_argument(
        "--avg-chars-per-sec",
        type=float,
        default=DEFAULT_AVG_CHARS_PER_SEC,
        help=f"Characters per second used for analysis (default: {DEFAULT_AVG_CHARS_PER_SEC}).",
    )
    parser.add_argument(
        "--max-mismatch-ratio",
        type=float,
        default=DEFAULT_MAX_MISMATCH_RATIO,
        help=f"Final allowed ratio after reanalysis (default: {DEFAULT_MAX_MISMATCH_RATIO}).",
    )
    parser.add_argument(
        "--stage-count",
        type=int,
        default=DEFAULT_STAGE_COUNT,
        help=f"Planned shortening stages for target_ratio hints (default: {DEFAULT_STAGE_COUNT}).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_STAGE_COUNT,
        help="Hard limit for completed merge and reanalysis iterations.",
    )
    parser.add_argument(
        "--context-halo",
        type=int,
        default=DEFAULT_CONTEXT_HALO,
        help=f"Neighboring items kept around each target cluster (default: {DEFAULT_CONTEXT_HALO}).",
    )
    parser.add_argument(
        "--target-bridge-gap",
        type=int,
        default=DEFAULT_TARGET_BRIDGE_GAP,
        help=f"Maximum count of non-target items inside one cluster bridge (default: {DEFAULT_TARGET_BRIDGE_GAP}).",
    )
    parser.add_argument(
        "--max-items-per-chunk",
        type=int,
        default=DEFAULT_MAX_ITEMS_PER_CHUNK,
        help=f"Maximum source items per chunk (default: {DEFAULT_MAX_ITEMS_PER_CHUNK}).",
    )
    parser.add_argument(
        "--temp-suffix",
        default="_temp",
        help="Suffix used when auto-generating the workdir name.",
    )
    parser.add_argument(
        "--cleanup-temp",
        action="store_true",
        help="After success, remove the temp workdir and keep only final files outside it.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.stage_count <= 0:
        raise ValueError("stage_count must be > 0")
    if args.max_iterations <= 0:
        raise ValueError("max_iterations must be > 0")
    if args.context_halo < 0:
        raise ValueError("context_halo must be >= 0")
    if args.target_bridge_gap < 0:
        raise ValueError("target_bridge_gap must be >= 0")
    if args.max_items_per_chunk <= 0:
        raise ValueError("max_items_per_chunk must be > 0")


def _resolve_workdir(input_path: Path, workdir_value: str | None, temp_suffix: str) -> Path:
    if workdir_value:
        return Path(workdir_value).resolve()
    auto_name = make_temp_workdir_name(f"{input_path.stem}_shortening", temp_suffix=temp_suffix)
    return (input_path.parent / auto_name).resolve()


def _resolve_final_json_path(input_path: Path, output_value: str | None) -> Path:
    if output_value:
        return Path(output_value).resolve()
    return (input_path.parent / f"{input_path.stem}_shortened.json").resolve()


def _resolve_final_report_path(final_json_path: Path, report_value: str | None) -> Path:
    if report_value:
        return Path(report_value).resolve()
    return final_json_path.with_name(f"{final_json_path.stem}_report.json")


def _validate_final_paths(
    *,
    workdir: Path,
    final_json_path: Path,
    final_report_path: Path,
    cleanup_temp: bool,
) -> None:
    if not cleanup_temp:
        return
    if _is_within_directory(final_json_path, workdir):
        raise ValueError("final output path must be outside workdir when --cleanup-temp is enabled")
    if _is_within_directory(final_report_path, workdir):
        raise ValueError("final report path must be outside workdir when --cleanup-temp is enabled")


def _find_resume_point(workdir: Path, initial_json_path: Path) -> Path:
    current_json_path = initial_json_path
    for report_path in sorted(workdir.glob("iter_*/report.json")):
        reanalyzed_path = report_path.parent / "reanalyzed.json"
        if reanalyzed_path.exists():
            current_json_path = reanalyzed_path
    return current_json_path


def _next_iteration_number(workdir: Path) -> int:
    iteration = 1
    while (workdir / f"iter_{iteration:03d}" / "report.json").exists():
        iteration += 1
    return iteration


def _prepare_iteration(
    *,
    current_json_path: Path,
    workdir: Path,
    iteration: int,
    stage_count: int,
    max_mismatch_ratio: float,
    context_halo: int,
    target_bridge_gap: int,
    max_items_per_chunk: int,
) -> None:
    exit_code = prepare_shortening_main(
        [
            str(current_json_path),
            "--workdir",
            str(workdir),
            "--iteration",
            str(iteration),
            "--stage-count",
            str(stage_count),
            "--max-mismatch-ratio",
            str(max_mismatch_ratio),
            "--context-halo",
            str(context_halo),
            "--target-bridge-gap",
            str(target_bridge_gap),
            "--max-items-per-chunk",
            str(max_items_per_chunk),
        ]
    )
    if exit_code != 0:
        raise RuntimeError(f"prepare_shortening_chunks failed with exit code {exit_code}")


def _load_manifest(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected manifest object in {path}")
    return dict(data)


def _ensure_iteration_instructions(*, iteration_dir: Path, manifest: Mapping[str, Any]) -> None:
    results_dir = iteration_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"Iteration: {manifest.get('iteration', '?')}",
        f"Threshold: {manifest.get('max_mismatch_ratio', '?')}",
        f"Chunks: {manifest.get('chunk_count', 0)}",
        f"Targets: {manifest.get('target_count', 0)}",
        "",
        "Rules:",
        "- Edit only items with role=target.",
        "- Keep order, indices, timing, and all non-text fields unchanged.",
        "- Shorten semantically and preserve local continuity.",
        "- Leave already compliant context untouched.",
        "- Prefer the smallest edit that reaches target_ratio.",
        "- Use soft joint rewrite when a chunk marks a multi-target local thought.",
        "- Omit unchanged targets from result files.",
        "",
        f"Write one or more result files into: {results_dir}",
        f"Allowed pattern: {RESULTS_GLOB}",
        "",
        "Strict result format:",
        "{",
        '  "chunk_id": 1,',
        '  "edits": [',
        '    {"index": 123, "text": ["Shorter line 1", "Shorter line 2"]}',
        "  ]",
        "}",
    ]
    (iteration_dir / "instructions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_edit_results(results_dir: Path) -> CollectedEdits:
    if not results_dir.exists():
        return CollectedEdits(edit_files=[], edit_count=0)

    edit_files = sorted(results_dir.glob(RESULTS_GLOB))
    edit_count = 0
    for result_path in edit_files:
        edit_count += len(load_edit_entries(result_path))
    return CollectedEdits(edit_files=edit_files, edit_count=edit_count)


def _validate_result_targets(*, iteration_dir: Path, edit_files: Sequence[Path]) -> None:
    target_index_map = _load_target_index_map(iteration_dir)
    seen_indices: set[int] = set()

    for edit_file in edit_files:
        payload = load_json(edit_file)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Edit JSON must be an object: {edit_file}")

        chunk_id = coerce_int(payload.get("chunk_id"))
        edits = load_edit_entries(edit_file)
        for edit in edits:
            source_index = edit["index"]
            if source_index not in target_index_map:
                raise ValueError(f"Edit index {source_index} is not a target in {edit_file}")
            expected_chunk_id = target_index_map[source_index]
            if chunk_id is not None and chunk_id != expected_chunk_id:
                raise ValueError(
                    f"Edit index {source_index} belongs to chunk {expected_chunk_id}, not {chunk_id}"
                )
            if source_index in seen_indices:
                raise ValueError(f"Duplicate edit for index {source_index} in {iteration_dir / 'results'}")
            seen_indices.add(source_index)


def _load_target_index_map(iteration_dir: Path) -> dict[int, int]:
    target_map: dict[int, int] = {}
    for chunk_path in sorted((iteration_dir / "chunks").glob("chunk_*.json")):
        payload = load_json(chunk_path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Expected chunk object in {chunk_path}")

        chunk_id = coerce_int(payload.get("chunk_id"))
        if chunk_id is None:
            raise ValueError(f"Missing chunk_id in {chunk_path}")

        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError(f"Expected items list in {chunk_path}")

        for position, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(f"Invalid chunk item at position {position} in {chunk_path}")
            if item.get("role") != "target":
                continue
            source_index = coerce_int(item.get("index"))
            if source_index is None:
                raise ValueError(f"Missing target index in {chunk_path}")
            if source_index in target_map:
                raise ValueError(f"Duplicate target index {source_index} in {iteration_dir}")
            target_map[source_index] = chunk_id
    return target_map


def _reanalyze_items(
    items: Sequence[Mapping[str, Any]],
    *,
    avg_chars_per_sec: float,
    max_mismatch_ratio: float,
) -> list[dict[str, Any]]:
    copied_items = [dict(item) for item in items]
    result = analyze_subtitles(
        copied_items,
        avg_chars_per_sec=avg_chars_per_sec,
        max_mismatch_ratio=max_mismatch_ratio,
    )
    analyzed_items = result.get("items")
    if not isinstance(analyzed_items, list):
        raise ValueError("analyze_subtitles returned invalid items")

    normalized: list[dict[str, Any]] = []
    for item in analyzed_items:
        if not isinstance(item, Mapping):
            raise ValueError("analyze_subtitles returned invalid item")
        normalized.append(dict(item))
    return normalized


def _build_iteration_report(
    *,
    iteration: int,
    source_json_path: Path,
    output_json_path: Path,
    current_items: Sequence[Mapping[str, Any]],
    updated_items: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    merge_report: Mapping[str, Any],
) -> dict[str, Any]:
    before_targets = _critical_index_set(
        current_items,
        max_mismatch_ratio=float(manifest.get("max_mismatch_ratio", DEFAULT_MAX_MISMATCH_RATIO)),
    )
    after_targets = _critical_index_set(
        updated_items,
        max_mismatch_ratio=float(manifest.get("max_mismatch_ratio", DEFAULT_MAX_MISMATCH_RATIO)),
    )
    applied_indices = sorted(entry["index"] for entry in _report_entries(merge_report.get("applied")))
    changed_set = set(applied_indices)

    fixed_indices = sorted(before_targets - after_targets)
    remaining_indices = sorted(after_targets)
    still_critical_indices = sorted(after_targets & changed_set)
    newly_critical_indices = sorted(after_targets - before_targets)

    return {
        "iteration": iteration,
        "source_json": str(source_json_path),
        "output_json": str(output_json_path),
        "chunk_count": int(manifest.get("chunk_count", 0)),
        "target_count": int(manifest.get("target_count", 0)),
        "context_count": int(manifest.get("context_count", 0)),
        "requested_edit_count": int(merge_report.get("total_edits", 0)),
        "applied_edit_count": len(applied_indices),
        "applied_indices": applied_indices,
        "critical_before_count": len(before_targets),
        "critical_after_count": len(after_targets),
        "fixed_count": len(fixed_indices),
        "fixed_indices": fixed_indices,
        "remaining_critical_count": len(remaining_indices),
        "remaining_critical_indices": remaining_indices,
        "still_critical_after_edit_count": len(still_critical_indices),
        "still_critical_after_edit_indices": still_critical_indices,
        "newly_critical_count": len(newly_critical_indices),
        "newly_critical_indices": newly_critical_indices,
        "merge_report": dict(merge_report),
        "max_mismatch_ratio": float(manifest.get("max_mismatch_ratio", DEFAULT_MAX_MISMATCH_RATIO)),
    }


def _report_entries(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, Mapping)]


def _critical_index_set(
    items: Sequence[Mapping[str, Any]],
    *,
    max_mismatch_ratio: float,
) -> set[int]:
    critical_indices: set[int] = set()
    for position, item in enumerate(items):
        if _is_critical_item(item, max_mismatch_ratio=max_mismatch_ratio):
            critical_indices.add(item_index(item, position))
    return critical_indices


def _is_critical_item(item: Mapping[str, Any], *, max_mismatch_ratio: float) -> bool:
    ratio = effective_ratio(item)
    if ratio is not None:
        return ratio > max_mismatch_ratio
    analysis = item.get("analysis")
    return isinstance(analysis, Mapping) and bool(analysis.get("is_critical"))


def _finalize_success(
    *,
    input_path: Path,
    workdir: Path,
    current_json_path: Path,
    final_json_path: Path,
    final_report_path: Path,
    max_mismatch_ratio: float,
    cleanup_temp: bool,
) -> None:
    final_items = load_source_items(current_json_path)
    save_json(final_json_path, final_items)

    history = _load_iteration_history(workdir)
    final_report = {
        "status": "completed",
        "input_json": str(input_path),
        "workdir": str(workdir),
        "final_json": str(final_json_path),
        "final_report": str(final_report_path),
        "remaining_critical_count": len(
            _critical_index_set(final_items, max_mismatch_ratio=max_mismatch_ratio)
        ),
        "max_mismatch_ratio": max_mismatch_ratio,
        "iterations_completed": len(history),
        "history": history,
    }
    save_json(final_report_path, final_report)

    local_final_report_path = workdir / "final_report.json"
    if not cleanup_temp:
        save_json(local_final_report_path, final_report)

    if cleanup_temp:
        shutil.rmtree(workdir, ignore_errors=False)


def _load_iteration_history(workdir: Path) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for report_path in sorted(workdir.glob("iter_*/report.json")):
        payload = load_json(report_path)
        if isinstance(payload, Mapping):
            history.append(dict(payload))
    return history


def _is_within_directory(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
