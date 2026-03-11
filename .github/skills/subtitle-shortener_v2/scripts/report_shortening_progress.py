from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from subtitle_shortening_common import effective_ratio, item_index, load_json


def main(argv: list[str] | None = None) -> int:
    """Print concise progress information for workflow artifacts."""

    parser = argparse.ArgumentParser(
        description="Print concise human-readable subtitle-shortening progress."
    )
    parser.add_argument("path", help="Path to analyzed JSON, workflow dir, manifest, or report JSON.")
    args = parser.parse_args(argv)

    target_path = Path(args.path).resolve()
    for line in build_progress_lines(target_path):
        print(line)
    return 0


def build_progress_lines(target_path: Path) -> list[str]:
    """Build human-readable progress lines for one artifact path."""

    if target_path.is_dir():
        return _build_dir_lines(target_path)

    payload = load_json(target_path)
    if isinstance(payload, list):
        return _build_analyzed_json_lines(target_path, payload)
    if isinstance(payload, Mapping):
        return _build_mapping_lines(target_path, payload)
    raise ValueError(f"Unsupported JSON structure in {target_path}")


def _build_dir_lines(workdir: Path) -> list[str]:
    final_report_path = workdir / "final_report.json"
    if final_report_path.exists():
        payload = load_json(final_report_path)
        if isinstance(payload, Mapping):
            return _build_final_report_lines(final_report_path, payload)

    report_paths = sorted(workdir.glob("iter_*/report.json"))
    manifest_paths = sorted(workdir.glob("iter_*/manifest.json"))
    if report_paths:
        payload = load_json(report_paths[-1])
        if isinstance(payload, Mapping):
            return _build_iteration_report_lines(report_paths[-1], payload, workdir=workdir)

    if manifest_paths:
        payload = load_json(manifest_paths[-1])
        if isinstance(payload, Mapping):
            return _build_manifest_lines(manifest_paths[-1], payload, workdir=workdir)

    initial_json_path = workdir / "initial_analyzed.json"
    if initial_json_path.exists():
        payload = load_json(initial_json_path)
        if isinstance(payload, list):
            return _build_analyzed_json_lines(initial_json_path, payload)

    raise ValueError(f"No recognizable workflow artifacts found in {workdir}")


def _build_mapping_lines(target_path: Path, payload: Mapping[str, Any]) -> list[str]:
    if "history" in payload and "status" in payload and "final_json" in payload:
        return _build_final_report_lines(target_path, payload)
    if "remaining_critical_count" in payload and "applied_edit_count" in payload:
        return _build_iteration_report_lines(target_path, payload, workdir=None)
    if "chunks" in payload and "chunk_count" in payload:
        return _build_manifest_lines(target_path, payload, workdir=None)
    return [f"File: {target_path}", "Unsupported report object."]


def _build_analyzed_json_lines(target_path: Path, items: Sequence[Any]) -> list[str]:
    validated_items = _normalize_item_sequence(items, source=target_path)
    critical_items = _critical_entries(validated_items)
    checked_count = sum(1 for item in validated_items if _is_checked(item))

    lines = [
        f"Analyzed JSON: {target_path}",
        f"Items: {len(validated_items)}",
        f"Checked: {checked_count}",
        f"Critical: {len(critical_items)}",
    ]
    if critical_items:
        preview = ", ".join(
            f"#{entry['index']} {entry['ratio']:.2f}x" for entry in critical_items[:5]
        )
        lines.append(f"Worst: {preview}")
    else:
        lines.append("Worst: none")
    return lines


def _build_manifest_lines(
    target_path: Path,
    payload: Mapping[str, Any],
    *,
    workdir: Path | None,
) -> list[str]:
    lines = [
        f"Manifest: {target_path}",
        f"Iteration: {payload.get('iteration', '?')}",
        f"Chunks: {payload.get('chunk_count', 0)}",
        f"Targets: {payload.get('target_count', 0)}",
        f"Context: {payload.get('context_count', 0)}",
        f"Indices: {_format_indices(_coerce_index_list(payload.get('processed_indices')))}",
    ]
    if workdir is not None:
        lines.append(f"Result files: {len(sorted((target_path.parent / 'results').glob('*.result.json')))}")
    return lines


def _build_iteration_report_lines(
    target_path: Path,
    payload: Mapping[str, Any],
    *,
    workdir: Path | None,
) -> list[str]:
    lines = [
        f"Iteration report: {target_path}",
        f"Iteration: {payload.get('iteration', '?')}",
        f"Applied edits: {payload.get('applied_edit_count', 0)} / {payload.get('requested_edit_count', 0)}",
        f"Critical: {payload.get('critical_before_count', 0)} -> {payload.get('critical_after_count', 0)}",
        f"Fixed: {payload.get('fixed_count', 0)}",
        f"Still critical after edit: {payload.get('still_critical_after_edit_count', 0)}",
        f"Newly critical: {payload.get('newly_critical_count', 0)}",
        f"Remaining indices: {_format_indices(_coerce_index_list(payload.get('remaining_critical_indices')))}",
    ]
    if workdir is not None:
        lines.append(f"Workdir: {workdir}")
    return lines


def _build_final_report_lines(target_path: Path, payload: Mapping[str, Any]) -> list[str]:
    raw_history = payload.get("history")
    history: list[Mapping[str, Any]] = []
    if isinstance(raw_history, list):
        history = [entry for entry in raw_history if isinstance(entry, Mapping)]
    last_iteration = history[-1] if history and isinstance(history[-1], Mapping) else None
    iteration_count = payload.get("iterations_completed")
    if not isinstance(iteration_count, int):
        iteration_count = len(history)
    lines = [
        f"Final report: {target_path}",
        f"Status: {payload.get('status', 'unknown')}",
        f"Iterations completed: {iteration_count}",
        f"Remaining critical: {payload.get('remaining_critical_count', '?')}",
        f"Final JSON: {payload.get('final_json', '?')}",
    ]
    if last_iteration is not None:
        lines.append(
            "Last iteration: "
            f"#{last_iteration.get('iteration', '?')} "
            f"edits={last_iteration.get('applied_edit_count', 0)} "
            f"critical={last_iteration.get('critical_before_count', 0)}"
            f"->{last_iteration.get('critical_after_count', 0)}"
        )
    return lines


def _normalize_item_sequence(items: Sequence[Any], *, source: Path) -> list[Mapping[str, Any]]:
    normalized: list[Mapping[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError(f"Each subtitle item must be an object in {source}")
        normalized.append(item)
    return normalized


def _critical_entries(items: Sequence[Mapping[str, Any]]) -> list[dict[str, float | int]]:
    entries: list[dict[str, float | int]] = []
    for position, item in enumerate(items):
        ratio = effective_ratio(item)
        if ratio is None or not _is_critical(item):
            continue
        entries.append({"index": item_index(item, position), "ratio": ratio})
    entries.sort(key=lambda entry: float(entry["ratio"]), reverse=True)
    return entries


def _is_checked(item: Mapping[str, Any]) -> bool:
    analysis = item.get("analysis")
    return isinstance(analysis, Mapping) and bool(analysis.get("is_checked"))


def _is_critical(item: Mapping[str, Any]) -> bool:
    analysis = item.get("analysis")
    return isinstance(analysis, Mapping) and bool(analysis.get("is_critical"))


def _coerce_index_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    indices: list[int] = []
    for item in value:
        if isinstance(item, int):
            indices.append(item)
        elif isinstance(item, str) and item.isdigit():
            indices.append(int(item))
    return indices


def _format_indices(indices: Sequence[int], *, limit: int = 10) -> str:
    if not indices:
        return "none"
    preview = ", ".join(str(index) for index in indices[:limit])
    if len(indices) > limit:
        preview += f", ... (+{len(indices) - limit})"
    return preview


if __name__ == "__main__":
    raise SystemExit(main())
