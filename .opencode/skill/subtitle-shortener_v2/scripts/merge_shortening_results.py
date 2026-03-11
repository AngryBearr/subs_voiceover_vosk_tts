from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, TypedDict

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .subtitle_shortening_common import coerce_text_lines, load_json, save_json
except ImportError:
    from subtitle_shortening_common import coerce_text_lines, load_json, save_json


class EditEntry(TypedDict):
    index: int
    text: list[str]


class ReportEntry(TypedDict):
    index: int
    old_text: list[str] | None
    new_text: list[str]
    reason: str
    edit_source: str


def main(argv: list[str] | None = None) -> int:
    """Merge validated shortening edits into analyzed subtitle JSON."""

    parser = argparse.ArgumentParser(
        description="Merge validated shortening edits into analyzed subtitle JSON."
    )
    parser.add_argument("source", help="Source analyzed subtitle JSON")
    parser.add_argument("edits", help="Edit JSON file or directory with edit JSON files")
    parser.add_argument("output", help="Output path for merged subtitle JSON")
    parser.add_argument("report", help="Output path for merge report JSON")
    args = parser.parse_args(argv)

    items = load_source_items(Path(args.source))
    edit_files = resolve_edit_files(Path(args.edits))
    merged_items, report = merge_edits_with_report(items, edit_files)

    save_json(args.output, merged_items)
    save_json(
        args.report,
        {
            "source": str(Path(args.source).resolve()),
            "edits_source": [str(path) for path in edit_files],
            "total_source_items": len(items),
            "total_edits": report["total_edits"],
            "applied": report["applied"],
            "skipped": report["skipped"],
            "unknown": report["unknown"],
        },
    )

    print(
        "[merge] "
        f"files={len(edit_files)} "
        f"edits={report['total_edits']} "
        f"applied={len(report['applied'])} "
        f"skipped={len(report['skipped'])} "
        f"unknown={len(report['unknown'])}"
    )
    return 0


def load_source_items(path: Path) -> list[dict[str, Any]]:
    """Load source subtitle items."""

    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("Source JSON must be a list of subtitle objects")

    items: list[dict[str, Any]] = []
    for index, value in enumerate(data):
        if not isinstance(value, Mapping):
            raise ValueError(f"Source item at position {index} must be an object")
        items.append(dict(value))
    return items


def resolve_edit_files(path: Path) -> list[Path]:
    """Resolve one edit file or all edit files in a directory."""

    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError("Edit file must have .json extension")
        return [path]

    if path.is_dir():
        files = sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() == ".json"
        )
        if not files:
            raise ValueError("Edit directory does not contain JSON files")
        return files

    raise FileNotFoundError(f"Edit path not found: {path}")


def load_edit_entries(path: Path) -> list[EditEntry]:
    """Load one strict result payload."""

    data = load_json(path)
    if not isinstance(data, Mapping):
        raise ValueError(f"Edit JSON must be an object: {path}")

    keys = set(data.keys())
    if keys != {"chunk_id", "edits"} and keys != {"edits"}:
        raise ValueError(f"Edit JSON must contain only 'edits' and optional 'chunk_id': {path}")

    raw_edits = data.get("edits")
    if not isinstance(raw_edits, list):
        raise ValueError(f"'edits' must be a list: {path}")

    edits: list[EditEntry] = []
    for position, value in enumerate(raw_edits):
        if not isinstance(value, Mapping):
            raise ValueError(f"Edit entry at position {position} must be an object: {path}")
        if set(value.keys()) != {"index", "text"}:
            raise ValueError(
                f"Edit entry at position {position} must contain only 'index' and 'text': {path}"
            )

        edit_index = _require_int(value.get("index"), f"Invalid edit index at position {position}: {path}")
        text_lines = coerce_text_lines(value.get("text"))
        edits.append({"index": edit_index, "text": text_lines})
    return edits


def merge_edits_with_report(
    items: list[dict[str, Any]],
    edit_files: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge result payloads and build a report."""

    index_map = _build_index_map(items)
    merged_items = [dict(item) for item in items]
    merged_index_map = _build_index_map(merged_items)

    applied: list[ReportEntry] = []
    skipped: list[ReportEntry] = []
    unknown: list[ReportEntry] = []
    seen_edit_indices: set[int] = set()
    total_edits = 0

    for edit_file in edit_files:
        for edit in load_edit_entries(edit_file):
            total_edits += 1
            source_index = edit["index"]
            new_text = list(edit["text"])
            source_item = index_map.get(source_index)
            old_text = _extract_text(source_item) if source_item is not None else None
            edit_source = str(edit_file)

            if source_index in seen_edit_indices:
                skipped.append(
                    _make_report_entry(
                        index=source_index,
                        old_text=old_text,
                        new_text=new_text,
                        reason="duplicate_edit_index",
                        edit_source=edit_source,
                    )
                )
                continue

            seen_edit_indices.add(source_index)

            if source_item is None:
                unknown.append(
                    _make_report_entry(
                        index=source_index,
                        old_text=None,
                        new_text=new_text,
                        reason="index_not_found",
                        edit_source=edit_source,
                    )
                )
                continue

            if old_text == new_text:
                skipped.append(
                    _make_report_entry(
                        index=source_index,
                        old_text=old_text,
                        new_text=new_text,
                        reason="unchanged_text",
                        edit_source=edit_source,
                    )
                )
                continue

            merged_index_map[source_index]["text"] = list(new_text)
            applied.append(
                _make_report_entry(
                    index=source_index,
                    old_text=old_text,
                    new_text=new_text,
                    reason="applied",
                    edit_source=edit_source,
                )
            )

    report = {
        "total_edits": total_edits,
        "applied": applied,
        "skipped": skipped,
        "unknown": unknown,
    }
    return merged_items, report


def _build_index_map(items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    index_map: dict[int, dict[str, Any]] = {}
    for position, item in enumerate(items):
        source_index = _require_int(
            item.get("index"),
            f"Source item at position {position} has invalid index",
        )
        if source_index in index_map:
            raise ValueError(f"Duplicate source index: {source_index}")
        index_map[source_index] = item
    return index_map


def _require_int(value: Any, message: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(message)
    return value


def _extract_text(item: Mapping[str, Any] | None) -> list[str] | None:
    if item is None:
        return None
    return coerce_text_lines(item.get("text"))


def _make_report_entry(
    *,
    index: int,
    old_text: list[str] | None,
    new_text: list[str],
    reason: str,
    edit_source: str,
) -> ReportEntry:
    return {
        "index": index,
        "old_text": None if old_text is None else list(old_text),
        "new_text": list(new_text),
        "reason": reason,
        "edit_source": edit_source,
    }


if __name__ == "__main__":
    raise SystemExit(main())
