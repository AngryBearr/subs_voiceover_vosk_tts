"""Apply explicitly approved human review edits into a new export bundle."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from filelock import FileLock, Timeout

from utils.shortening_plan import CuePlan, PlannerConfig
from utils.shortening_review import ReviewError, build_review_state, make_review_queue, validate_edit_queue
from utils.shortening_review_io import load_review_bundle, write_derived_bundle


def _load_json(path: Path, label: str) -> Any:
    try:
        raw = path.read_bytes().decode("utf-8-sig")
        def reject_constant(value: str) -> Any:
            raise ValueError(f"non-finite JSON value {value} is not allowed")
        def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON object key {key!r}")
                result[key] = value
            return result
        return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=reject_duplicate)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReviewError(str(exc)) from exc


def _planner(state: dict[str, Any]) -> PlannerConfig:
    try:
        return PlannerConfig(**copy.deepcopy(state["planner_config"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ReviewError(f"planner configuration is invalid: {exc}") from exc


def _plans(state: dict[str, Any]) -> list[CuePlan]:
    try:
        return [CuePlan(**copy.deepcopy(value)) for value in state["plans"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ReviewError(f"review plans are invalid: {exc}") from exc


def _descendant(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _apply_review_locked(run_dir: Path, review_path: Path, output_dir: Path, source_path: Path | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise ReviewError(f"run directory does not exist: {run_dir}")
    if output_dir.exists():
        raise ReviewError(f"output directory already exists: {output_dir}")
    if output_dir == run_dir or _descendant(output_dir, run_dir):
        raise ReviewError("output directory must be outside the input run directory")
    if not review_path.exists() or not review_path.is_file():
        raise ReviewError(f"review file does not exist: {review_path}")
    bundle = load_review_bundle(run_dir, source_path=source_path)
    state = bundle["state"]
    edited = _load_json(review_path, "edited review queue")
    approved = validate_edit_queue(state, edited)
    current = copy.deepcopy(state["current_items"])
    changed_ids: list[str] = []
    expected_by_id = {entry["cue_id"]: entry for entry in make_review_queue(state)["entries"]}
    for cue_id, approval in approved.items():
        position = expected_by_id[cue_id]["position"]
        item = current[position]
        new_text = approval["text"]
        old_display = expected_by_id[cue_id]["current_text"]
        # A keep approval for a multi-fragment list preserves its exact shape.
        # A replacement of a list is deliberately represented as one fragment.
        if new_text == old_display:
            continue
        if isinstance(item.get("text"), list):
            item["text"] = [new_text]
        else:
            item["text"] = new_text
        changed_ids.append(cue_id)
    plans = _plans(state)
    new_state = build_review_state(
        source_items=copy.deepcopy(state["source_items"]),
        current_items=current,
        plans=plans,
        records=copy.deepcopy(state["records"]),
        planner_config=_planner(state),
        metadata=copy.deepcopy(state["metadata"]),
        selected_ids=copy.deepcopy(state["selected_ids"]),
        previous_state=state,
        human_approvals=approved,
    )
    decisions = [
        {"cue_id": cue_id, "decision": "replace" if cue_id in changed_ids else "keep", "approved": True}
        for cue_id in approved
    ]
    provenance = {
        "parent_snapshot_id": state["snapshot_id"],
        "decisions": decisions,
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "explicit_approved": list(approved),
        "changed_cue_ids": changed_ids,
    }
    report = write_derived_bundle(output_dir, new_state, provenance, kind="application")
    return {"report": report, "approved_count": len(approved), "text_changed_count": len(changed_ids), "unresolved_count": new_state["summary"]["unresolved_count"]}


def apply_review(run_dir: Path, review_path: Path, output_dir: Path, source_path: Path | None = None) -> dict[str, Any]:
    """Apply a queue while holding the source run lock for the whole operation."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise ReviewError(f"run directory does not exist: {run_dir}")
    lock = FileLock(str(run_dir / ".run.lock"), timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise ReviewError("run directory is locked by another process") from exc
    try:
        return _apply_review_locked(run_dir, review_path, output_dir, source_path)
    finally:
        lock.release()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply explicitly approved subtitle review edits into a new bundle")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--review", type=Path, required=True, help="edited review.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        run_dir = args.run_dir.resolve()
        result = apply_review(run_dir, args.review, args.output_dir, args.source)
        print(json.dumps({key: result[key] for key in ("approved_count", "text_changed_count", "unresolved_count")}, sort_keys=True))
        return 0 if result["report"].get("export_error") is None else 1
    except Exception as exc:
        lines = str(exc).splitlines()
        detail = lines[0][:240] if lines else "no error message"
        print(f"apply_review failed: {type(exc).__name__}: {detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
