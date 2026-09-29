"""Inspect a saved primary shortening run without resuming or modifying it."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterator

from utils.shortening_budget import BudgetPolicy
from utils.shortening_resume import ResumeError, _read_json, canonical_hash, inspect_journal, load_baseline, load_manifest_contract
from utils.shortening_runner import _module_hashes


_FILES = ("run_manifest.json", "resume_baseline.json", "requests.jsonl", "calls.jsonl",
          "resume_events.jsonl", "report.json", "resume_status.json")
_MODELS = {"generator": ("meta/muse-spark-1.3-contributor", "low"),
           "arbiter": ("openai/gpt-5.6-sol", "medium")}
_NEW_MODELS = {**_MODELS, "arbiter": ("openai/gpt-6-sol", "high")}
_FAILURE_CODES = {"interrupted_response", "transport_error", "timeout", "unknown_request"}
_REQUEST_ID = re.compile(r"[0-9a-f]{32}\Z")
_RECEIPT_ID = re.compile(r"gen-[A-Za-z0-9_-]{1,100}\Z")


class InvalidRun(ValueError):
    """The run cannot be classified from trustworthy saved artifacts."""


def _signature(directory: Path) -> tuple[tuple[str, int, int, int, str] | tuple[str, None], ...]:
    result: list[tuple[str, int, int, int, str] | tuple[str, None]] = []
    for name in (*_FILES, ".run.lock"):
        path = directory / name
        try:
            if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
                raise InvalidRun("invalid_artifacts")
            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW), "rb") as stream:
                file_stat = os.fstat(stream.fileno())
                if not stat.S_ISREG(file_stat.st_mode):
                    raise InvalidRun("invalid_artifacts")
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            result.append((name, file_stat.st_ino, file_stat.st_size, file_stat.st_mtime_ns, digest))
        except FileNotFoundError:
            result.append((name, None))
        except OSError as exc:
            raise InvalidRun("invalid_artifacts") from exc
    return tuple(result)


@contextmanager
def _stable_read(directory: Path) -> Iterator[tuple[tuple[str, int, int, int, str] | tuple[str, None], ...]]:
    if not directory.is_dir() or directory.is_symlink():
        raise InvalidRun("invalid_artifacts")
    descriptor: int | None = None
    try:
        try:
            lock_path = directory / ".run.lock"
            if not stat.S_ISREG(lock_path.stat(follow_symlinks=False).st_mode):
                raise InvalidRun("invalid_artifacts")
            descriptor = os.open(lock_path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise InvalidRun("invalid_artifacts")
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise InvalidRun("invalid_artifacts") from exc
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise InvalidRun("concurrent_run") from exc
        before = _signature(directory)
        yield before
        if _signature(directory) != before:
            raise InvalidRun("concurrent_change")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _identifier(value: Any, *, receipt: bool = False) -> str | None:
    pattern = _RECEIPT_ID if receipt else _REQUEST_ID
    return value if isinstance(value, str) and pattern.fullmatch(value) else None


def inspect_run(directory: Path) -> dict[str, Any]:
    """Return only bounded metadata; this is not a resume readiness check."""
    with _stable_read(directory):
        try:
            contract = load_manifest_contract(directory)
            source = contract.get("source")
            if (not isinstance(source, dict) or not isinstance(source.get("items"), list)
                    or source.get("sha256") != canonical_hash(source["items"])):
                raise InvalidRun("invalid_manifest")
            models = contract.get("models")
            frozen = contract.get("budget_policy")
            if not isinstance(models, dict) or set(models) != set(_MODELS):
                raise InvalidRun("unsupported_run")
            snapshot_models = frozen.get("snapshot", {}).get("models", {}) if isinstance(frozen, dict) and isinstance(frozen.get("snapshot"), dict) else {}
            expected_models = _NEW_MODELS if isinstance(snapshot_models, dict) and "openai/gpt-6-sol" in snapshot_models else _MODELS
            for role, (model, variant) in expected_models.items():
                if models[role] != {"provider": "openrouter", "model": model, "variant": variant}:
                    raise InvalidRun("unsupported_run")
            if (not isinstance(frozen, dict) or not isinstance(frozen.get("snapshot"), dict)
                    or frozen.get("pricing_snapshot_id") != frozen["snapshot"].get("id")):
                raise InvalidRun("invalid_manifest")
            policy = BudgetPolicy.from_snapshot(frozen["limit_usd"], frozen["snapshot"],
                                                max_requests=frozen["max_requests"],
                                                max_unknown_calls=frozen.get("max_unknown_calls", 0))
            hashes = contract.get("module_sha256")
            current_hashes = _module_hashes(primary=True)
            if (not isinstance(hashes, dict) or set(hashes) != set(current_hashes)
                    or any(not isinstance(value, str) for value in hashes.values())):
                raise InvalidRun("invalid_manifest")
            module_differences = sorted(name for name, value in hashes.items() if value != current_hashes[name])
            baseline = load_baseline(directory)
            if baseline is None:
                raise InvalidRun("missing_baseline")
            targets = contract.get("target_ids")
            planner = contract.get("planner")
            if (not isinstance(targets, list) or not isinstance(planner, dict)
                    or baseline.get("source_hash") != source["sha256"]
                    or baseline.get("selected_ids") != targets
                    or baseline.get("length_basis") != planner.get("length_basis")
                    or not isinstance(baseline.get("source_ids"), list)
                    or len(baseline["source_ids"]) != len(source["items"])
                    or not isinstance(baseline.get("plans"), list)
                    or len(baseline["plans"]) != len(source["items"])
                    or not isinstance(baseline.get("records"), list)
                    or len(baseline["records"]) != len(targets)):
                raise InvalidRun("invalid_baseline")
            report = _read_json(directory / "report.json", "report.json")
            status_path = directory / "resume_status.json"
            saved_status = _read_json(status_path, "resume_status.json") if status_path.exists() else None
            if not isinstance(report, dict) or (saved_status is not None and not isinstance(saved_status, dict)):
                raise InvalidRun("invalid_report")
            saved_state = report.get("run_state")
            if (saved_state not in {"completed", "incomplete", "running"}
                    or (saved_status is not None and "run_state" not in saved_status
                        and saved_status.get("status") != "blocked_unknown")
                    or (saved_status is not None and "run_state" in saved_status
                        and saved_status["run_state"] != saved_state)
                    or report.get("model") != models
                    or not isinstance(report.get("source_manifest"), dict)
                    or report["source_manifest"].get("sha256") != source["sha256"]
                    or report.get("selected_ids") != targets
                    or not isinstance(report.get("records"), list)
                    or len(report["records"]) != len(targets)
                    or not isinstance(report.get("summary"), dict)
                    or not isinstance(report.get("budget"), dict)):
                raise InvalidRun("invalid_report")
            if not (directory / "requests.jsonl").exists():
                counts_in_report = report["summary"].get("request_counts")
                if (any((directory / name).exists() for name in ("calls.jsonl", "resume_events.jsonl"))
                        or not isinstance(counts_in_report, dict)
                        or set(counts_in_report) != {"generation", "review", "repair", "recheck"}
                        or any(type(value) is not int or value != 0 for value in counts_in_report.values())
                        or report["budget"].get("physical_calls") != 0
                        or report["budget"].get("known_spent_usd") != "0"
                        or report["budget"].get("unknown_reserved_usd") != "0"
                        or report["summary"].get("request_failures")):
                    raise InvalidRun("missing_journal")
            journal = inspect_journal(directory, policy)
            budget = policy.summary(journal.calls, journal.failures)
            requests: list[dict[str, Any]] = []
            for request in journal.requests:
                request_id = _identifier(request["request_id"])
                role = "generator" if request["stage"] in {"generation", "repair"} else "arbiter"
                if request_id is None or request["model"] != models[role]:
                    raise InvalidRun("invalid_journal")
                if (request["completed"] and not request["receipt_ids"]
                        and not any(failure.get("request_id") == request_id
                                    and failure.get("accounting_incomplete") is True
                                    for failure in journal.failures)):
                    raise InvalidRun("invalid_journal")
                requests.append({"request_id": request_id, "stage": request["stage"],
                                 "model": models[role], "completed": request["completed"],
                                 "superseded": request["superseded"],
                                 "receipt_ids": [_identifier(value, receipt=True) for value in request["receipt_ids"]]})
            by_id = {request["request_id"]: request for request in requests}
            failures = [{"request_id": _identifier(failure.get("request_id")),
                         "stage": failure["stage"],
                         "model": by_id[failure["request_id"]]["model"],
                         "receipt_ids": by_id[failure["request_id"]]["receipt_ids"],
                         "reason": failure.get("error") if failure.get("error") in _FAILURE_CODES else "other_error",
                         "cost_unknown": failure["accounting_incomplete"]}
                        for failure in journal.failures]
            if any(row["request_id"] is None for row in failures):
                raise InvalidRun("invalid_journal")
            counts: dict[str, int] = {}
            for request in requests:
                stage = request["stage"]
                counts[stage] = counts.get(stage, 0) + 1
            saved_budget = report["budget"]
            stale = (report["summary"].get("request_counts") != {stage: counts.get(stage, 0) for stage in
                      ("generation", "review", "repair", "recheck")}
                    or any(saved_budget.get(key) != budget[key] for key in
                            ("known_spent_usd", "unknown_reserved_usd", "physical_calls")))
            reasons: list[str] = []
            if journal.unknown_requests:
                reasons.append("unknown_request")
            if budget["status"] != "ok":
                reasons.append(budget["status"])
            if failures:
                reasons.append("request_failure")
            if stale:
                reasons.append("saved_report_stale")
            if saved_status is None and saved_state == "completed":
                reasons.append("missing_resume_status")
            elif saved_status is not None and saved_status.get("status") == "blocked_unknown":
                reasons.append("saved_status_blocked_unknown")
            if module_differences:
                reasons.append("module_hash_changed")
            if saved_state != "completed":
                reasons.append("run_incomplete")
            result = {
                "status": "blocked" if reasons else "completed_snapshot",
                "saved_run_state": saved_state,
                "reason_codes": reasons,
                "next_action": ("preserve_and_reconcile_unknown_request" if "unknown_request" in reasons else
                                "preserve_and_reconcile_unknown_cost" if budget["unknown_reserved_usd"] != "0" else
                                "review_saved_failures_and_resume_contract" if reasons else "no_action_required"),
                "budget": budget,
                "module_hash_differences": module_differences,
                "requests": requests,
                "unknown_requests": [request for request in requests if any(
                    row["request_id"] == request["request_id"] for row in journal.unknown_requests)],
                "failures": failures,
                "limitations": "Snapshot only; source, profile, live pricing and paid resume eligibility are not checked.",
            }
            return result
        except InvalidRun:
            raise
        except (ResumeError, KeyError, TypeError, ValueError, OSError) as exc:
            raise InvalidRun("invalid_artifacts") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = inspect_run(args.run_dir)
    except InvalidRun as exc:
        result = {"status": "invalid", "reason_codes": [str(exc)],
                  "next_action": "preserve_run_and_inspect_manually"}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "completed_snapshot" else 2


if __name__ == "__main__":
    raise SystemExit(main())
