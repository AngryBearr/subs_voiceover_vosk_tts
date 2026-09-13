"""Reanalyze an existing review bundle with an explicitly supplied subtitle export."""

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
from utils.shortening_review import ReviewError, build_review_state
from utils.shortening_review_io import load_review_bundle, write_derived_bundle


def _read(path: Path, label: str) -> Any:
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


def _plans(state: dict[str, Any]) -> list[CuePlan]:
    try:
        return [CuePlan(**copy.deepcopy(value)) for value in state["plans"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ReviewError(f"review plans are invalid: {exc}") from exc


def _planner(state: dict[str, Any]) -> PlannerConfig:
    try:
        return PlannerConfig(**copy.deepcopy(state["planner_config"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ReviewError(f"planner configuration is invalid: {exc}") from exc


def _validate_current(value: Any, old: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ReviewError("current JSON must be an array of objects")
    if len(value) != len(old):
        raise ReviewError("current JSON cue count differs from the review snapshot")
    for position, (before, after) in enumerate(zip(old, value)):
        if before.get("index") != after.get("index"):
            raise ReviewError(f"current JSON cue index differs at position {position}")
    return copy.deepcopy(value)


def _reanalyze_locked(run_dir: Path, output_dir: Path, current_json: Path | None = None, source_path: Path | None = None, force_recheck: bool = False) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise ReviewError(f"run directory does not exist: {run_dir}")
    if output_dir.exists():
        raise ReviewError(f"output directory already exists: {output_dir}")
    try:
        output_dir.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise ReviewError("output directory must be outside the input run directory")
    bundle = load_review_bundle(run_dir, source_path=source_path)
    state = bundle["state"]
    current = copy.deepcopy(state["current_items"])
    if current_json is not None:
        current = _validate_current(_read(current_json, "current JSON"), current)
    changed_ids = [
        f"cue-{position}" for position, (before, after) in enumerate(zip(state["current_items"], current))
        if before != after
    ]
    new_state = build_review_state(
        source_items=copy.deepcopy(state["source_items"]),
        current_items=current,
        plans=_plans(state),
        records=copy.deepcopy(state["records"]),
        planner_config=_planner(state),
        metadata=copy.deepcopy(state["metadata"]),
        selected_ids=copy.deepcopy(state["selected_ids"]),
        previous_state=state,
        force_recheck=force_recheck,
    )
    provenance = {
        "parent_snapshot_id": state["snapshot_id"],
        "changed_cue_ids": changed_ids,
        "explicit_approved": [],
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "force_recheck": force_recheck,
        "current_json": str(current_json) if current_json is not None else None,
    }
    report = write_derived_bundle(output_dir, new_state, provenance, kind="reanalysis")
    return {"report": report, "changed_count": len(changed_ids), "unresolved_count": new_state["summary"]["unresolved_count"]}


def reanalyze(run_dir: Path, output_dir: Path, current_json: Path | None = None, source_path: Path | None = None, force_recheck: bool = False) -> dict[str, Any]:
    """Reanalyze a bundle while holding the source run lock for the whole operation."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise ReviewError(f"run directory does not exist: {run_dir}")
    lock = FileLock(str(run_dir / ".run.lock"), timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise ReviewError("run directory is locked by another process") from exc
    try:
        return _reanalyze_locked(run_dir, output_dir, current_json, source_path, force_recheck)
    finally:
        lock.release()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reanalyze a subtitle review snapshot offline")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--current-json", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--force-recheck", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        run_dir = args.run_dir.resolve()
        result = reanalyze(run_dir, args.output_dir, args.current_json, args.source, args.force_recheck)
        print(json.dumps({key: result[key] for key in ("changed_count", "unresolved_count")}, sort_keys=True))
        return 0 if result["report"].get("export_error") is None else 1
    except Exception as exc:
        lines = str(exc).splitlines()
        detail = lines[0][:240] if lines else "no error message"
        print(f"review_subtitles failed: {type(exc).__name__}: {detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
