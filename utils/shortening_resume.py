"""Fail-closed persistence primitives for shortening-run resume."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid

from utils.llm_types import CallRecord, DetailedResponse, call_record_from_dict
from utils.shortening_costs import summarize_costs


RESUME_SCHEMA_VERSION = 1
RUNNER_ALGORITHM_VERSION = "shortening-resume-v1"


class ResumeError(ValueError):
    """The durable resume state cannot be trusted."""


class UnknownRequestsError(ResumeError):
    """A resume contains physical attempts which require an explicit decision."""

    def __init__(self, request_ids: list[str]) -> None:
        self.request_ids = list(request_ids)
        super().__init__("unknown requests require explicit retry decisions: " + ", ".join(request_ids))


def canonical_hash(value: Any) -> str:
    """Hash the one canonical JSON representation used by the persistence files."""
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResumeError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResumeError(f"value is not JSON serializable: {exc}") from exc


def _pretty_json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResumeError(f"value is not JSON serializable: {exc}") from exc


def _fsync_directory(directory: Path) -> None:
    # Directory handles are not fsync-able on Windows.
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON durably, replacing the destination only after fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_pretty_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, value: Any) -> None:
    """Append one durable JSONL row without changing existing history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path, label: str) -> Any:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON value {value} is not allowed")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream, parse_constant=reject_constant, object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ResumeError(f"invalid {label}; preserve the directory for manual inspection: {exc}") from exc


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON value {value} is not allowed")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ResumeError(f"truncated final row in {label}; preserve the directory")
        with path.open("r", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    raise ResumeError(f"blank line in {label} at line {number}; preserve the directory")
                try:
                    value = json.loads(
                        line, parse_constant=reject_constant, object_pairs_hook=reject_duplicate_keys,
                    )
                except (UnicodeError, ValueError) as exc:
                    raise ResumeError(f"malformed {label} at line {number}; preserve the directory") from exc
                if not isinstance(value, dict):
                    raise ResumeError(f"non-object {label} row at line {number}; preserve the directory")
                rows.append(value)
    except (OSError, UnicodeError) as exc:
        raise ResumeError(f"cannot read {label}; preserve the directory: {exc}") from exc
    return rows


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResumeError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResumeError(f"{label} must be a non-empty string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ResumeError(f"{label} must be boolean")
    return value


def _require_elapsed(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise ResumeError(f"{label} must be a finite nonnegative number")
    return float(value)


def _parse_started_at(value: Any, label: str) -> datetime:
    text = _require_string(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResumeError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ResumeError(f"{label} must include a timezone")
    return parsed


def _require_schema_version(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != RESUME_SCHEMA_VERSION:
        raise ResumeError(f"{label} has an unsupported schema version")
    return value


def ensure_manifest(output_dir: Path, contract: dict[str, Any], *, resume: bool) -> None:
    """Create or validate the immutable run contract before any other writes."""
    output_dir = Path(output_dir)
    path = output_dir / "run_manifest.json"
    if resume and (output_dir / ".migration-incomplete").exists():
        raise ResumeError("migration is incomplete; preserve the directory")
    if not resume:
        if path.exists():
            raise ResumeError("run_manifest.json already exists; use resume and preserve the run directory")
        payload = {
            "schema_version": RESUME_SCHEMA_VERSION,
            "runner_algorithm_version": RUNNER_ALGORITHM_VERSION,
            "contract": contract,
            "contract_hash": canonical_hash(contract),
        }
        atomic_json(path, payload)
        return
    if not path.exists():
        raise ResumeError("resume manifest is missing; preserve the directory instead of starting a new run")
    stored = _require_dict(_read_json(path, "run_manifest.json"), "run_manifest.json")
    _require_schema_version(stored.get("schema_version"), "run manifest")
    if stored.get("runner_algorithm_version") != RUNNER_ALGORITHM_VERSION:
        raise ResumeError("run manifest algorithm version is incompatible; preserve the directory")
    if "contract" not in stored or "contract_hash" not in stored:
        raise ResumeError("run manifest is incomplete; preserve the directory")
    actual_hash = canonical_hash(stored["contract"])
    if stored["contract_hash"] != actual_hash:
        raise ResumeError("run manifest contract hash is corrupt; preserve the directory")
    if actual_hash != canonical_hash(contract) or stored["contract"] != contract:
        raise ResumeError("run manifest contract canonical mismatch; preserve the directory")


def save_baseline(output_dir: Path, payload: dict[str, Any]) -> None:
    """Persist the effective plans once; a valid baseline is immutable."""
    path = Path(output_dir) / "resume_baseline.json"
    if path.exists():
        existing = load_baseline(Path(output_dir))
        if existing != payload:
            raise ResumeError("valid resume baseline differs; preserve the directory")
        return
    atomic_json(path, {
        "schema_version": RESUME_SCHEMA_VERSION,
        "runner_algorithm_version": RUNNER_ALGORITHM_VERSION,
        "payload": payload,
        "checksum": canonical_hash(payload),
    })


def load_baseline(output_dir: Path) -> dict[str, Any] | None:
    path = Path(output_dir) / "resume_baseline.json"
    if not path.exists():
        return None
    value = _require_dict(_read_json(path, "resume_baseline.json"), "resume_baseline.json")
    if (isinstance(value.get("schema_version"), bool)
            or value.get("schema_version") != RESUME_SCHEMA_VERSION
            or value.get("runner_algorithm_version") != RUNNER_ALGORITHM_VERSION):
        raise ResumeError("unsupported resume baseline; preserve the directory")
    payload = value.get("payload")
    if not isinstance(payload, dict) or value.get("checksum") != canonical_hash(payload):
        raise ResumeError("resume baseline checksum or payload is corrupt; preserve the directory")
    return payload


_DESCRIPTOR_KEYS = ("stage", "model", "target_ids", "system_prompt", "user_prompt", "output_schema", "allow_tool")
_START_KEYS = ("request_id", "stage", "model", "target_ids", "system_prompt", "user_prompt", "output_schema", "allow_tool", "started_at", "elapsed_sec")


def _validate_reservation(value: Any, label: str = "request reservation") -> None:
    if not isinstance(value, dict) or set(value) != {"usd", "pricing_snapshot_id", "input_tokens", "output_tokens", "model"}:
        raise ResumeError(f"{label} is invalid")
    if not isinstance(value.get("usd"), str):
        raise ResumeError(f"{label} usd is invalid")
    try:
        from decimal import Decimal, InvalidOperation
        amount = Decimal(value["usd"])
    except (InvalidOperation, TypeError, ValueError):
        raise ResumeError(f"{label} usd is invalid") from None
    if not amount.is_finite() or amount <= 0:
        raise ResumeError(f"{label} usd is invalid")
    snapshot_id = value["pricing_snapshot_id"]
    if (not isinstance(snapshot_id, str) or len(snapshot_id) != 64
            or any(char not in "0123456789abcdef" for char in snapshot_id)):
        raise ResumeError(f"{label} pricing_snapshot_id is invalid")
    for key in ("input_tokens", "output_tokens"):
        token_count = value[key]
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
            raise ResumeError(f"{label} {key} is invalid")
    model = value["model"]
    if (not isinstance(model, dict) or set(model) != {"provider", "model", "variant"}
            or model.get("provider") != "openrouter"
            or any(not isinstance(model.get(key), str) or not model[key] for key in ("model", "variant"))):
        raise ResumeError(f"{label} model is invalid")


def _descriptor_from_start(row: dict[str, Any]) -> dict[str, Any]:
    descriptor = {key: row[key] for key in _DESCRIPTOR_KEYS}
    _validate_descriptor(descriptor)
    return descriptor


def _validate_descriptor(descriptor: dict[str, Any]) -> None:
    if (not isinstance(descriptor["stage"], str)
            or descriptor["stage"] not in {"generation", "review", "repair", "recheck"}):
        raise ResumeError("request descriptor stage/model is invalid")
    model = descriptor["model"]
    if (not isinstance(model, dict) or set(model) != {"provider", "model", "variant"}
            or any(not isinstance(model[key], str) or not model[key] for key in ("provider", "model", "variant"))):
        raise ResumeError("request descriptor stage/model is invalid")
    targets = descriptor["target_ids"]
    if (not isinstance(targets, list)
            or any(not isinstance(item, str) or not item for item in targets)
            or (isinstance(targets, list) and len(set(targets)) != len(targets))):
        raise ResumeError("request target_ids are invalid")
    if not isinstance(descriptor["system_prompt"], str) or not isinstance(descriptor["user_prompt"], str):
        raise ResumeError("request prompts are invalid")
    if not isinstance(descriptor["output_schema"], dict):
        raise ResumeError("request output_schema is invalid")
    _require_bool(descriptor["allow_tool"], "request allow_tool")


def _validate_start(row: dict[str, Any]) -> None:
    for key in _START_KEYS:
        if key not in row:
            raise ResumeError(f"request start missing {key}")
    _require_string(row["request_id"], "request_id")
    _descriptor_from_start(row)
    _require_string(row["started_at"], "started_at")
    _require_elapsed(row["elapsed_sec"], "elapsed_sec")
    if row.get("state") != "start":
        raise ResumeError("request start has invalid state")
    if "reservation" in row:
        _validate_reservation(row["reservation"])


def _validate_failure(failure: dict[str, Any], start: dict[str, Any]) -> None:
    required = {"request_id", "stage", "model", "target_ids", "started_at", "error", "accounting_incomplete"}
    if not required.issubset(failure):
        raise ResumeError("completion failure is missing metadata")
    if failure["request_id"] != start["request_id"]:
        raise ResumeError("completion failure request_id differs from start")
    if failure["stage"] != start["stage"] or not _model_equal(failure["model"], start["model"]):
        raise ResumeError("completion failure metadata differs from start")
    try:
        targets_match = canonical_hash(failure["target_ids"]) == canonical_hash(start["target_ids"])
    except ResumeError:
        targets_match = False
    if not targets_match or failure["started_at"] != start["started_at"]:
        raise ResumeError("completion failure metadata differs from start")
    _require_string(failure["error"], "completion failure error")
    _require_bool(failure["accounting_incomplete"], "completion failure accounting_incomplete")
    if "elapsed_sec" in failure:
        _require_elapsed(failure["elapsed_sec"], "completion failure elapsed_sec")
    if "physical_call_count" in failure:
        count = failure["physical_call_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ResumeError("completion failure physical_call_count must be nonnegative integer")


def _model_equal(left: Any, right: Any) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    try:
        return canonical_hash(left) == canonical_hash(right)
    except ResumeError:
        return False


@dataclass(frozen=True)
class JournalStep:
    request_id: str
    response: DetailedResponse | None
    replayed: bool


@dataclass(frozen=True)
class JournalInspection:
    """Read-only journal data, with no replay cursor or request authorization API."""

    calls: tuple[CallRecord, ...]
    failures: tuple[dict[str, Any], ...]
    requests: tuple[dict[str, Any], ...]
    unknown_requests: tuple[dict[str, Any], ...]


class RunJournal:
    """Read the old journal and expose an ordered, non-writing replay cursor."""

    def __init__(self, output_dir: Path, *, resume: bool, retry_unknown_ids: tuple[str, ...] = (),
                 retry_budget_policy: Any = None) -> None:
        self._initialize(output_dir)
        self._logical_unknown_only = retry_budget_policy is not None
        if resume:
            self._load()
        else:
            if any((self.output_dir / name).exists() for name in ("requests.jsonl", "calls.jsonl", "resume_events.jsonl")):
                raise ResumeError("journal files exist on a fresh run; preserve the directory and use resume")
        self._load_report_snapshot()
        unresolved = self._current_unresolved()
        requested = list(retry_unknown_ids)
        if len(set(requested)) != len(requested):
            raise ResumeError("retry_unknown_ids contains duplicates")
        if any(not isinstance(item, str) or not item for item in requested):
            raise ResumeError("retry_unknown_ids contains an invalid request ID")
        if not set(requested).issubset(set(unresolved)):
            raise ResumeError("retry_unknown_ids contains an already-completed or stale request ID")
        if unresolved and set(unresolved) != set(requested):
            self._write_blocked(unresolved, requested)
            raise UnknownRequestsError(unresolved)
        if requested:
            if retry_budget_policy is not None and any(old_id in self._replacements for old_id in requested):
                raise ResumeError("an unknown request already has its one permitted replacement")
            if retry_budget_policy is not None:
                self.validate_budget_reservations(retry_budget_policy)
                calls_by_stage, _, failures = self.accounting()
                all_calls = [call for group in calls_by_stage.values() for call in group]
                for old_id in requested:
                    retry_budget_policy.admit(self._starts[old_id]["model"], all_calls, failures)
        for old_id in requested:
            replacement = uuid.uuid4().hex
            while replacement in self._starts or replacement in self._events.values():
                replacement = uuid.uuid4().hex
            self._events[old_id] = replacement
            self._reserved[replacement] = dict(self._starts[old_id])
            self._replacements.add(replacement)
            append_jsonl(self.output_dir / "resume_events.jsonl", {
                "event": "retry_unknown",
                "request_id": old_id,
                "replacement_request_id": replacement,
                "decided_at": datetime.now(timezone.utc).isoformat(),
            })
        self._rebuild_roots()
        self._validate_root_order()

    def _initialize(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self._starts: dict[str, dict[str, Any]] = {}
        self._completes: dict[str, dict[str, Any]] = {}
        self._calls: dict[str, list[tuple[int, CallRecord]]] = {}
        self._events: dict[str, str] = {}
        self._reserved: dict[str, dict[str, Any]] = {}
        self._replacements: set[str] = set()
        self._roots: list[str] = []
        self._cursor = 0
        self._logical_unknown_only = False
        self._accepted_snapshot: dict[str, tuple[str, Any]] = {}

    @classmethod
    def readonly_snapshot(cls, output_dir: Path, *, logical_unknown_only: bool = False) -> "RunJournal":
        """Refresh saved starts and receipts after cancellation without retrying them."""
        journal = object.__new__(cls)
        journal._initialize(output_dir)
        journal._logical_unknown_only = logical_unknown_only
        journal._load()
        journal._load_report_snapshot()
        journal._validate_root_order()
        journal._cursor = len(journal._roots)
        return journal

    def _load(self) -> None:
        starts_rows = _read_jsonl(self.output_dir / "requests.jsonl", "requests.jsonl")
        complete_rows: list[dict[str, Any]] = []
        for row in starts_rows:
            state = row.get("state")
            if state == "start":
                _validate_start(row)
                request_id = row["request_id"]
                if request_id in self._starts:
                    raise ResumeError("duplicate request start; preserve the directory")
                self._starts[request_id] = row
            elif state == "complete":
                request_id = _require_string(row.get("request_id"), "complete request_id")
                if request_id not in self._starts:
                    raise ResumeError("completion precedes its request start; preserve the directory")
                complete_rows.append(row)
            else:
                raise ResumeError("invalid request journal state; preserve the directory")
        for row in complete_rows:
            request_id = _require_string(row.get("request_id"), "complete request_id")
            if row.get("state") != "complete":
                raise ResumeError("request completion has invalid state")
            if request_id in self._completes:
                raise ResumeError("duplicate request completion; preserve the directory")
            if request_id not in self._starts:
                raise ResumeError("completion has no request start; preserve the directory")
            start = self._starts[request_id]
            for key in ("stage", "model", "target_ids"):
                if key == "model":
                    same = _model_equal(row.get(key), start.get(key))
                else:
                    try:
                        same = canonical_hash(row.get(key)) == canonical_hash(start.get(key))
                    except ResumeError:
                        same = False
                if not same:
                    raise ResumeError("completion metadata differs from start; preserve the directory")
            if ("text" not in row or "error" not in row or "call_count" not in row or "failure" not in row
                    or "started_at" not in row or "elapsed_sec" not in row):
                raise ResumeError("request completion is missing accounting fields")
            if row["text"] is not None and not isinstance(row["text"], str):
                raise ResumeError("completion text must be string or null")
            if row["error"] is not None and not isinstance(row["error"], str):
                raise ResumeError("completion error must be string or null")
            if row["started_at"] != start["started_at"]:
                raise ResumeError("completion started_at differs from start")
            _require_elapsed(row["elapsed_sec"], "completion elapsed_sec")
            if isinstance(row["call_count"], bool) or not isinstance(row["call_count"], int) or row["call_count"] < 0:
                raise ResumeError("completion call_count must be nonnegative integer")
            if row["failure"] is not None and not isinstance(row["failure"], dict):
                raise ResumeError("completion failure must be object or null")
            failed = row["text"] is None or row["error"] is not None
            if failed != (row["failure"] is not None):
                raise ResumeError("completion failure does not match response outcome")
            if row["failure"] is not None:
                _validate_failure(row["failure"], start)
            self._completes[request_id] = row
        self._load_calls()
        events = _read_jsonl(self.output_dir / "resume_events.jsonl", "resume_events.jsonl")
        for row in events:
            if set(row) != {"event", "request_id", "replacement_request_id", "decided_at"} or row.get("event") != "retry_unknown":
                raise ResumeError("invalid resume event row; preserve the directory")
            old = _require_string(row.get("request_id"), "resume event request_id")
            replacement = _require_string(row.get("replacement_request_id"), "resume event replacement_request_id")
            if old not in self._starts or old in self._events or replacement in self._events.values() or replacement == old:
                raise ResumeError("invalid or duplicate resume event chain; preserve the directory")
            _require_string(row.get("decided_at"), "resume event decided_at")
            try:
                datetime.fromisoformat(row["decided_at"].replace("Z", "+00:00"))
            except ValueError as exc:
                raise ResumeError("resume event decided_at is invalid") from exc
            if old in self._completes:
                raise ResumeError("completed request cannot be retried")
            self._events[old] = replacement
            self._replacements.add(replacement)
            if replacement in self._starts:
                if _parse_started_at(self._starts[replacement]["started_at"], "replacement started_at") <= _parse_started_at(
                        self._starts[old]["started_at"], "original started_at"):
                    raise ResumeError("replacement request must start after the original request")
                if canonical_hash(_descriptor_from_start(self._starts[replacement])) != canonical_hash(_descriptor_from_start(self._starts[old])):
                    raise ResumeError("replacement descriptor differs from original")
            else:
                self._reserved[replacement] = dict(self._starts[old])
        self._validate_chains()
        self._validate_root_order()

    def _load_calls(self) -> None:
        rows = _read_jsonl(self.output_dir / "calls.jsonl", "calls.jsonl")
        seen: dict[str, set[int]] = {}
        for row in rows:
            required = {"request_id", "stage", "target_ids", "model", "provider", "record", "cost_estimate", "call_index"}
            if not required.issubset(row):
                raise ResumeError("call receipt is missing fields; preserve the directory")
            request_id = _require_string(row.get("request_id"), "call request_id")
            if request_id not in self._starts:
                raise ResumeError("call receipt has no request start; preserve the directory")
            start = self._starts[request_id]
            try:
                same_stage = canonical_hash(row["stage"]) == canonical_hash(start["stage"])
                same_targets = canonical_hash(row["target_ids"]) == canonical_hash(start["target_ids"])
            except ResumeError:
                same_stage = same_targets = False
            if not same_stage or not same_targets or not _model_equal(row["model"], start["model"]):
                raise ResumeError("call receipt metadata differs from request start")
            index = row["call_index"]
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ResumeError("call_index must be a nonnegative integer")
            if index in seen.setdefault(request_id, set()):
                raise ResumeError("duplicate call_index for request")
            seen[request_id].add(index)
            if not isinstance(row["cost_estimate"], dict):
                raise ResumeError("call cost_estimate must be an object")
            try:
                record = call_record_from_dict(row["record"])
            except (TypeError, ValueError, KeyError) as exc:
                raise ResumeError(f"invalid typed call receipt: {exc}") from exc
            if record.request_id is not None and record.request_id != request_id:
                raise ResumeError("typed call request_id differs from journal request_id")
            if not isinstance(row["provider"], str) or row["provider"] != record.provider:
                raise ResumeError("call provider metadata differs from typed receipt")
            self._calls.setdefault(request_id, []).append((index, record))
        for request_id, values in self._calls.items():
            values.sort(key=lambda item: item[0])
            if [index for index, _ in values] != list(range(len(values))):
                raise ResumeError("call indexes are not contiguous")
        for request_id, complete in self._completes.items():
            if complete["call_count"] != len(self._calls.get(request_id, [])):
                raise ResumeError("completion call_count does not match call receipts")

    def _validate_chains(self) -> None:
        for old, replacement in self._events.items():
            if replacement in self._starts and canonical_hash(_descriptor_from_start(self._starts[replacement])) != canonical_hash(_descriptor_from_start(self._starts[old])):
                raise ResumeError("replacement descriptor differs from original")
        for start_id in self._starts:
            visited: set[str] = set()
            current = start_id
            while current in self._events:
                if current in visited:
                    raise ResumeError("cyclic resume event chain")
                visited.add(current)
                current = self._events[current]
        self._rebuild_roots()

    def _validate_root_order(self) -> None:
        for position, root in enumerate(self._roots):
            terminal = self._terminal(root)
            unresolved = (terminal in self._reserved
                          or (terminal in self._starts and terminal not in self._completes))
            if unresolved and position != len(self._roots) - 1:
                raise ResumeError("a newer logical request bypasses an unresolved request")

    def _rebuild_roots(self) -> None:
        self._roots = [request_id for request_id in self._starts if request_id not in self._replacements]

    def _terminal(self, root: str) -> str:
        current = root
        visited: set[str] = set()
        while current in self._events:
            if current in visited:
                raise ResumeError("cyclic resume event chain")
            visited.add(current)
            current = self._events[current]
        return current

    def _current_unresolved(self) -> list[str]:
        result: list[str] = []
        for root in self._roots:
            terminal = self._terminal(root)
            if terminal in self._starts and terminal not in self._completes:
                result.append(terminal)
        return result

    def _load_report_snapshot(self) -> None:
        path = self.output_dir / "report.json"
        if not path.exists():
            return
        value = _require_dict(_read_json(path, "report.json"), "report.json")
        if "schema_version" in value:
            _require_schema_version(value["schema_version"], "existing report")
        records = value.get("records")
        if not isinstance(records, list):
            raise ResumeError("existing report records are invalid; preserve the directory")
        for record in records:
            if not isinstance(record, dict):
                raise ResumeError("existing report record is invalid; preserve the directory")
            status = record.get("status")
            if status in {"accepted", "actual_fits"}:
                cue_id = _require_string(record.get("cue_id"), "report cue_id")
                if "final_text" not in record or not isinstance(record["final_text"], str):
                    raise ResumeError("existing accepted report record is invalid")
                if cue_id in self._accepted_snapshot:
                    raise ResumeError("duplicate accepted report cue_id")
                self._accepted_snapshot[cue_id] = (status, record["final_text"])

    def _write_blocked(self, unresolved: list[str], requested: list[str]) -> None:
        calls_by_stage, counts, failures = self.accounting()
        summaries = {stage: summarize_costs(calls_by_stage.get(stage, [])) for stage in counts}
        descriptors = [{"request_id": request_id, **_descriptor_from_start(self._starts[request_id])} for request_id in unresolved]
        atomic_json(self.output_dir / "resume_status.json", {
            "schema_version": RESUME_SCHEMA_VERSION,
            "status": "blocked_unknown",
            "unresolved_requests": descriptors,
            "requested_retry_ids": requested,
            "known_physical_call_summaries": summaries,
            "failures": failures,
            "accounting_complete": False,
            "explanation": "Retry decisions are required because an unknown attempt may have been billed; preserve the directory.",
        })

    @property
    def unknown_requests(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        request_ids = (self._current_unresolved() if self._logical_unknown_only else
                       [request_id for request_id in self._starts if request_id not in self._completes])
        for request_id in request_ids:
            result.append({"request_id": request_id, **_descriptor_from_start(self._starts[request_id])})
        return result

    @property
    def unresolved_request_ids(self) -> list[str]:
        return self._current_unresolved()

    def accounting(self) -> tuple[dict[str, list[CallRecord]], dict[str, int], list[dict[str, Any]]]:
        calls_by_stage: dict[str, list[CallRecord]] = {}
        counts: dict[str, int] = {}
        failures: list[dict[str, Any]] = []
        for request_id, start in self._starts.items():
            stage = start["stage"]
            counts[stage] = counts.get(stage, 0) + 1
            for _, record in self._calls.get(request_id, []):
                calls_by_stage.setdefault(stage, []).append(record)
            complete = self._completes.get(request_id)
            if complete is not None and complete["failure"] is not None:
                failures.append(dict(complete["failure"]))
            elif complete is None:
                failure = {"request_id": request_id, "stage": stage, "target_ids": list(start["target_ids"]),
                           "model": start["model"], "error": "unknown_request", "started_at": start["started_at"],
                           "elapsed_sec": start["elapsed_sec"], "physical_call_count": len(self._calls.get(request_id, [])),
                           "accounting_incomplete": True}
                failures.append(failure)
        return calls_by_stage, counts, failures

    def next_request(self, descriptor: dict[str, Any]) -> JournalStep:
        if set(descriptor) != set(_DESCRIPTOR_KEYS):
            raise ResumeError("request descriptor keys do not match the journal contract")
        _validate_descriptor(descriptor)
        canonical_hash(descriptor)
        if self._cursor >= len(self._roots):
            return JournalStep(uuid.uuid4().hex, None, False)
        root = self._roots[self._cursor]
        terminal = self._terminal(root)
        expected = _descriptor_from_start(self._starts[root])
        if canonical_hash(descriptor) != canonical_hash(expected):
            raise ResumeError("replay request descriptor mismatch; preserve the directory")
        self._cursor += 1
        if terminal in self._starts:
            terminal_descriptor = _descriptor_from_start(self._starts[terminal])
            if canonical_hash(descriptor) != canonical_hash(terminal_descriptor):
                raise ResumeError("replacement request descriptor mismatch")
            if terminal not in self._completes:
                raise ResumeError("unresolved request was not explicitly authorized")
            calls = [record for _, record in self._calls.get(terminal, [])]
            complete = self._completes[terminal]
            return JournalStep(terminal, DetailedResponse(complete["text"], calls, complete["error"]), True)
        if terminal not in self._reserved:
            raise ResumeError("journal terminal request is missing")
        return JournalStep(terminal, None, False)

    def assert_consumed(self) -> None:
        if self._cursor != len(self._roots):
            raise ResumeError("durable replay requests remain unconsumed; preserve the directory")

    def checkpoint_allowed(self, records: list[dict[str, Any]]) -> bool:
        if self._cursor != len(self._roots):
            return False
        current: dict[str, tuple[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict):
                raise ResumeError("checkpoint report record is invalid")
            status = record.get("status")
            if status in {"accepted", "actual_fits"}:
                cue_id = _require_string(record.get("cue_id"), "checkpoint cue_id")
                if not isinstance(record.get("final_text"), str):
                    raise ResumeError("checkpoint final_text is invalid")
                if cue_id in current:
                    raise ResumeError("duplicate checkpoint cue_id")
                current[cue_id] = (status, record["final_text"])
        for cue_id, expected in self._accepted_snapshot.items():
            if current.get(cue_id) != expected:
                raise ResumeError("checkpoint would overwrite an accepted report record")
        return True

    def before_live(self, records: list[dict[str, Any]]) -> None:
        self.assert_consumed()
        if not self.checkpoint_allowed(records):
            raise ResumeError("checkpoint is not yet allowed")

    def validate_budget_reservations(self, policy: Any) -> None:
        """Validate every saved physical request against the frozen policy."""
        for request_id, start in self._starts.items():
            reservation = start.get("reservation")
            if reservation is None:
                raise ResumeError(f"request {request_id} is missing its budget reservation")
            try:
                expected = policy.reservation(start["model"])
            except Exception as exc:
                raise ResumeError(f"request {request_id} has an invalid budget model") from exc
            if canonical_hash(reservation) != canonical_hash(expected):
                raise ResumeError(f"request {request_id} budget reservation differs from frozen policy")


def inspect_journal(output_dir: Path, policy: Any) -> JournalInspection:
    """Validate saved journal data without constructing an actionable replay cursor."""
    journal = object.__new__(RunJournal)
    journal._initialize(output_dir)
    journal._load()
    journal._load_report_snapshot()
    journal._validate_root_order()
    journal.validate_budget_reservations(policy)
    by_stage, _, failures = journal.accounting()
    requests = tuple({
        "request_id": request_id,
        "stage": start["stage"],
        "model": start["model"],
        "completed": request_id in journal._completes,
        "superseded": request_id in journal._events,
        "receipt_ids": [call.generation_id for _, call in journal._calls.get(request_id, [])],
    } for request_id, start in journal._starts.items())
    contract = load_manifest_contract(output_dir)
    unresolved = (journal._current_unresolved() if contract.get("run", {}).get("scheduling") == "windowed" else
                  [request_id for request_id in journal._starts if request_id not in journal._completes])
    unknown = tuple(request for request in requests if request["request_id"] in unresolved)
    return JournalInspection(
        calls=tuple(call for stage_calls in by_stage.values() for call in stage_calls),
        failures=tuple(failures), requests=requests, unknown_requests=unknown,
    )


def load_manifest_contract(output_dir: Path) -> dict[str, Any]:
    """Read and validate an existing immutable run contract."""
    path = Path(output_dir) / "run_manifest.json"
    if not path.exists():
        raise ResumeError("resume manifest is missing; preserve the directory instead of starting a new run")
    stored = _require_dict(_read_json(path, "run_manifest.json"), "run_manifest.json")
    _require_schema_version(stored.get("schema_version"), "run manifest")
    if stored.get("runner_algorithm_version") != RUNNER_ALGORITHM_VERSION:
        raise ResumeError("run manifest algorithm version is incompatible; preserve the directory")
    if "contract" not in stored or "contract_hash" not in stored:
        raise ResumeError("run manifest is incomplete; preserve the directory")
    contract = stored["contract"]
    if stored["contract_hash"] != canonical_hash(contract):
        raise ResumeError("run manifest contract hash is corrupt; preserve the directory")
    return _require_dict(contract, "run manifest contract")


def validate_inherited_evidence(output_dir: Path, migration: Any) -> None:
    """Bind ancestral bytes before retry intent or any live request; suffixes may grow."""
    if not isinstance(migration, dict) or migration.get("version") != "sol6-generation-v1":
        raise ResumeError("unsupported inherited journal contract")
    hashes = migration.get("source_files_sha256")
    sizes = migration.get("inherited_file_sizes")
    prefix = {"requests.jsonl", "calls.jsonl"}
    immutable = {"resume_baseline.json", "preflight_audio/audio_report.json"}
    if (not isinstance(hashes, dict) or not isinstance(sizes, dict)
            or set(sizes) != prefix | immutable or not set(sizes).issubset(hashes)):
        raise ResumeError("inherited evidence binding is incomplete")
    for name, size in sizes.items():
        digest = hashes[name]
        if type(size) is not int or size < 1 or not isinstance(digest, str) or len(digest) != 64:
            raise ResumeError("invalid inherited evidence length or digest")
        path = output_dir / name
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size < size or (
                    name in immutable and path.stat().st_size != size):
                raise ResumeError(f"inherited evidence missing or changed: {name}")
            with path.open("rb") as stream:
                actual = hashlib.sha256(stream.read(size)).hexdigest()
        except OSError as exc:
            raise ResumeError(f"cannot read inherited evidence: {name}") from exc
        if actual != digest:
            raise ResumeError(f"inherited evidence hash mismatch: {name}")
