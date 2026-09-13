from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from utils.llm_types import CallRecord, DetailedResponse
from utils.shortening_resume import (
    RESUME_SCHEMA_VERSION,
    JournalStep,
    ResumeError,
    RunJournal,
    UnknownRequestsError,
    append_jsonl,
    canonical_hash,
    ensure_manifest,
    load_baseline,
    save_baseline,
)


def _descriptor() -> dict[str, object]:
    return {
        "stage": "generation",
        "model": {"provider": "fixture", "model": "m", "variant": "v"},
        "target_ids": ["cue-1"],
        "system_prompt": "system",
        "user_prompt": "user",
        "output_schema": {"type": "object"},
        "allow_tool": False,
    }


def _start(request_id: str = "old") -> dict[str, object]:
    return {"request_id": request_id, **_descriptor(), "state": "start", "started_at": "2026-01-01T00:00:00Z", "elapsed_sec": 0.0}


def _record() -> CallRecord:
    return CallRecord("fixture", "m", "v", "2026-01-01T00:00:00Z", 0.1, None, "error")


def _complete(request_id: str = "old", *, call_count: int = 0) -> dict[str, object]:
    return {"request_id": request_id, "stage": "generation", "model": _descriptor()["model"],
            "target_ids": ["cue-1"], "state": "complete", "text": "answer", "error": None,
            "started_at": "2026-01-01T00:00:00Z", "elapsed_sec": 0.2,
            "call_count": call_count, "failure": None}


def _receipt(request_id: str = "old", *, index: int = 0, model: dict[str, str] | None = None) -> dict[str, object]:
    return {"request_id": request_id, "stage": "generation", "target_ids": ["cue-1"],
            "model": model or _descriptor()["model"], "provider": "fixture", "call_index": index,
            "record": asdict(_record()), "cost_estimate": {}}


def test_manifest_hash_mismatch_is_fail_closed(tmp_path: Path) -> None:
    contract = {"input": "frozen"}
    ensure_manifest(tmp_path, contract, resume=False)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["contract"]["input"] = "changed"
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ResumeError):
        ensure_manifest(tmp_path, contract, resume=True)


def test_replay_is_ordered_and_does_not_append_logs(tmp_path: Path) -> None:
    append_jsonl(tmp_path / "requests.jsonl", _start())
    append_jsonl(tmp_path / "requests.jsonl", _complete())
    journal = RunJournal(tmp_path, resume=True)
    before = (tmp_path / "requests.jsonl").read_bytes()
    step = journal.next_request(_descriptor())
    assert step == JournalStep("old", DetailedResponse("answer", [], None), True)
    assert (tmp_path / "requests.jsonl").read_bytes() == before


def test_typed_accounting_and_unknown_retry_decision(tmp_path: Path) -> None:
    append_jsonl(tmp_path / "requests.jsonl", _start())
    append_jsonl(tmp_path / "calls.jsonl", _receipt())
    with pytest.raises(UnknownRequestsError) as blocked:
        RunJournal(tmp_path, resume=True)
    assert blocked.value.request_ids == ["old"]
    assert json.loads((tmp_path / "resume_status.json").read_text(encoding="utf-8"))["status"] == "blocked_unknown"
    journal = RunJournal(tmp_path, resume=True, retry_unknown_ids=("old",))
    step = journal.next_request(_descriptor())
    assert not step.replayed and step.response is None and step.request_id != "old"
    assert journal.accounting()[1] == {"generation": 1}


def test_canonical_hash_is_deterministic() -> None:
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    assert RESUME_SCHEMA_VERSION == 1


@pytest.mark.parametrize("kind,raw", [
    ("malformed", b"{\"state\": \"start\"}\nnot-json\n"),
    ("truncated", b"{\"state\":"),
    ("blank", b"{\"state\": \"start\"}\n\n"),
])
def test_jsonl_malformed_truncated_or_blank_is_rejected(tmp_path: Path, kind: str, raw: bytes) -> None:
    (tmp_path / "requests.jsonl").write_bytes(raw)
    with pytest.raises(ResumeError):
        RunJournal(tmp_path, resume=True)


@pytest.mark.parametrize("raw", [b"1\n", b"null\n", b"[1]\n"])
def test_jsonl_nonobject_rows_are_rejected(tmp_path: Path, raw: bytes) -> None:
    (tmp_path / "requests.jsonl").write_bytes(raw)
    with pytest.raises(ResumeError):
        RunJournal(tmp_path, resume=True)


@pytest.mark.parametrize("raw", [b"{\"state\": \"start\", \"n\": NaN}\n", b"{\"state\": \"start\", \"n\": Infinity}\n",
                                 b"{\"state\": \"start\", \"n\": 1, \"n\": 2}\n"])
def test_jsonl_nonfinite_values_and_duplicate_keys_are_rejected(tmp_path: Path, raw: bytes) -> None:
    (tmp_path / "requests.jsonl").write_bytes(raw)
    with pytest.raises(ResumeError):
        RunJournal(tmp_path, resume=True)


def test_invalid_utf8_jsonl_is_wrapped_in_resume_error(tmp_path: Path) -> None:
    (tmp_path / "requests.jsonl").write_bytes(b"\xff\n")
    with pytest.raises(ResumeError):
        RunJournal(tmp_path, resume=True)


@pytest.mark.parametrize("state", ["duplicate_start", "duplicate_complete", "duplicate_call_index", "missing_calls"])
def test_duplicate_and_missing_history_rows_are_rejected(tmp_path: Path, state: str) -> None:
    append_jsonl(tmp_path / "requests.jsonl", _start())
    if state == "duplicate_start":
        append_jsonl(tmp_path / "requests.jsonl", _start())
    elif state == "duplicate_complete":
        append_jsonl(tmp_path / "requests.jsonl", _complete())
        append_jsonl(tmp_path / "requests.jsonl", _complete())
    elif state == "duplicate_call_index":
        append_jsonl(tmp_path / "calls.jsonl", _receipt(index=0))
        append_jsonl(tmp_path / "calls.jsonl", _receipt(index=0))
    else:
        append_jsonl(tmp_path / "requests.jsonl", _complete(call_count=1))
    with pytest.raises(ResumeError):
        RunJournal(tmp_path, resume=True)


@pytest.mark.parametrize("field", ["stage", "model", "target_ids"])
def test_inconsistent_completion_metadata_is_rejected(tmp_path: Path, field: str) -> None:
    start = _start()
    complete = _complete()
    if field == "stage":
        complete[field] = "review"
    elif field == "model":
        complete[field] = {"provider": "fixture", "model": "other", "variant": "v"}
    else:
        complete[field] = ["cue-2"]
    append_jsonl(tmp_path / "requests.jsonl", start)
    append_jsonl(tmp_path / "requests.jsonl", complete)
    with pytest.raises(ResumeError):
        RunJournal(tmp_path, resume=True)


@pytest.mark.parametrize("field", ["stage", "model", "target_ids"])
def test_inconsistent_call_receipt_metadata_is_rejected(tmp_path: Path, field: str) -> None:
    append_jsonl(tmp_path / "requests.jsonl", _start())
    receipt = _receipt()
    if field == "stage":
        receipt[field] = "review"
    elif field == "model":
        receipt[field] = {"provider": "fixture", "model": "other", "variant": "v"}
    else:
        receipt[field] = ["cue-2"]
    append_jsonl(tmp_path / "calls.jsonl", receipt)
    with pytest.raises(ResumeError):
        RunJournal(tmp_path, resume=True)


def test_unknown_request_with_partial_calls_is_explicitly_retried(tmp_path: Path) -> None:
    append_jsonl(tmp_path / "requests.jsonl", _start())
    append_jsonl(tmp_path / "calls.jsonl", _receipt())
    with pytest.raises(UnknownRequestsError):
        RunJournal(tmp_path, resume=True)
    journal = RunJournal(tmp_path, resume=True, retry_unknown_ids=("old",))
    step = journal.next_request(_descriptor())
    assert step.replayed is False and step.response is None
    assert journal.accounting()[1] == {"generation": 1}
    assert len(journal.accounting()[0]["generation"]) == 1


def test_retry_chain_requires_new_id_for_second_unknown_attempt(tmp_path: Path) -> None:
    append_jsonl(tmp_path / "requests.jsonl", _start())
    first = RunJournal(tmp_path, resume=True, retry_unknown_ids=("old",))
    replacement = first.next_request(_descriptor()).request_id
    replacement_start = _start(replacement)
    replacement_start["started_at"] = "2026-01-01T00:00:01Z"
    append_jsonl(tmp_path / "requests.jsonl", replacement_start)
    with pytest.raises(UnknownRequestsError) as blocked:
        RunJournal(tmp_path, resume=True)
    assert blocked.value.request_ids == [replacement]
    second = RunJournal(tmp_path, resume=True, retry_unknown_ids=(replacement,))
    second_step = second.next_request(_descriptor())
    assert second_step.request_id not in {"old", replacement}
    assert second_step.response is None


def test_retry_decision_rejects_stale_original_id(tmp_path: Path) -> None:
    append_jsonl(tmp_path / "requests.jsonl", _start())
    journal = RunJournal(tmp_path, resume=True, retry_unknown_ids=("old",))
    journal.next_request(_descriptor())
    with pytest.raises(ResumeError):
        RunJournal(tmp_path, resume=True, retry_unknown_ids=("old",))


@pytest.mark.parametrize("retry_ids", [("missing",), ("old", "old")])
def test_invalid_or_duplicate_retry_decision_is_rejected(tmp_path: Path, retry_ids: tuple[str, ...]) -> None:
    append_jsonl(tmp_path / "requests.jsonl", _start())
    with pytest.raises(ResumeError):
        RunJournal(tmp_path, resume=True, retry_unknown_ids=retry_ids)


def test_completed_request_cannot_be_retried(tmp_path: Path) -> None:
    append_jsonl(tmp_path / "requests.jsonl", _start())
    append_jsonl(tmp_path / "requests.jsonl", _complete())
    with pytest.raises(ResumeError):
        RunJournal(tmp_path, resume=True, retry_unknown_ids=("old",))


@pytest.mark.parametrize("descriptor_change", ["stage", "model", "target_ids"])
def test_live_descriptor_mismatch_is_rejected(tmp_path: Path, descriptor_change: str) -> None:
    append_jsonl(tmp_path / "requests.jsonl", _start())
    append_jsonl(tmp_path / "requests.jsonl", _complete())
    journal = RunJournal(tmp_path, resume=True)
    descriptor = _descriptor()
    if descriptor_change == "stage":
        descriptor[descriptor_change] = "review"
    elif descriptor_change == "model":
        descriptor[descriptor_change] = {"provider": "fixture", "model": "m2", "variant": "v"}
    else:
        descriptor[descriptor_change] = ["cue-2"]
    with pytest.raises(ResumeError):
        journal.next_request(descriptor)


def test_unused_call_receipt_is_rejected(tmp_path: Path) -> None:
    append_jsonl(tmp_path / "requests.jsonl", _start())
    receipt = _receipt("not-started")
    append_jsonl(tmp_path / "calls.jsonl", receipt)
    with pytest.raises(ResumeError):
        RunJournal(tmp_path, resume=True)


@pytest.mark.parametrize("field,value", [
    ("stage", "unknown"),
    ("model", {"provider": "fixture", "model": "m"}),
    ("target_ids", ["cue-1", "cue-1"]),
    ("target_ids", [""]),
])
def test_descriptor_contract_rejects_invalid_values(tmp_path: Path, field: str, value: object) -> None:
    descriptor = _descriptor()
    descriptor[field] = value
    append_jsonl(tmp_path / "requests.jsonl", {"request_id": "old", **descriptor, "state": "start",
                                                "started_at": "2026-01-01T00:00:00Z", "elapsed_sec": 0.0})
    with pytest.raises(ResumeError):
        RunJournal(tmp_path, resume=True)


def test_observed_call_model_alias_is_allowed_when_wrapper_matches(tmp_path: Path) -> None:
    append_jsonl(tmp_path / "requests.jsonl", _start())
    receipt = _receipt()
    receipt["record"] = asdict(CallRecord("fixture", "provider-alias", "backend", "2026-01-01T00:00:00Z", 0.1, None, "error"))
    append_jsonl(tmp_path / "calls.jsonl", receipt)
    append_jsonl(tmp_path / "requests.jsonl", _complete(call_count=1))
    journal = RunJournal(tmp_path, resume=True)
    assert journal.next_request(_descriptor()).response is not None


@pytest.mark.parametrize("outcome", ["failure_empty", "failure_wrong_id", "failure_wrong_time", "failure_bad_flag", "failure_bad_elapsed"])
def test_corrupt_failure_metadata_is_rejected(tmp_path: Path, outcome: str) -> None:
    start = _start()
    complete = _complete()
    complete["text"] = None
    complete["error"] = "transport"
    failure: dict[str, object] = {"request_id": "old", "stage": "generation", "model": _descriptor()["model"],
                                  "target_ids": ["cue-1"], "started_at": start["started_at"], "error": "transport",
                                  "accounting_incomplete": True, "elapsed_sec": 0.2}
    if outcome == "failure_empty":
        failure = {}
    elif outcome == "failure_wrong_id":
        failure["request_id"] = "other"
    elif outcome == "failure_wrong_time":
        failure["started_at"] = "2026-01-01T00:01:00Z"
    elif outcome == "failure_bad_flag":
        failure["accounting_incomplete"] = 1
    else:
        failure["elapsed_sec"] = float("inf")
    complete["failure"] = failure
    append_jsonl(tmp_path / "requests.jsonl", start)
    if outcome == "failure_bad_elapsed":
        with (tmp_path / "requests.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(complete) + "\n")
    else:
        append_jsonl(tmp_path / "requests.jsonl", complete)
    with pytest.raises(ResumeError):
        RunJournal(tmp_path, resume=True)


def test_success_and_failure_completion_contracts(tmp_path: Path) -> None:
    append_jsonl(tmp_path / "requests.jsonl", _start("success"))
    success = _complete("success")
    append_jsonl(tmp_path / "requests.jsonl", success)
    journal = RunJournal(tmp_path, resume=True)
    assert journal.next_request(_descriptor()).replayed

    failure_start = _start("failure")
    failure = _complete("failure")
    failure["text"] = None
    failure["error"] = "transport"
    failure["failure"] = {"request_id": "failure", "stage": "generation", "model": _descriptor()["model"],
                           "target_ids": ["cue-1"], "started_at": failure_start["started_at"], "error": "transport",
                           "accounting_incomplete": False, "elapsed_sec": 0.2}
    append_jsonl(tmp_path / "requests.jsonl", failure_start)
    append_jsonl(tmp_path / "requests.jsonl", failure)
    failure_journal = RunJournal(tmp_path, resume=True)
    assert failure_journal.next_request(_descriptor()).response is not None


def test_retry_replacement_must_be_after_original_and_not_old_completed_root(tmp_path: Path) -> None:
    append_jsonl(tmp_path / "requests.jsonl", _start("old"))
    append_jsonl(tmp_path / "requests.jsonl", _start("replacement"))
    replacement = _start("replacement")
    replacement["started_at"] = "2025-12-31T23:59:59Z"
    (tmp_path / "requests.jsonl").write_text(
        "\n".join(json.dumps(row) for row in [_start("old"), replacement]) + "\n", encoding="utf-8",
    )
    append_jsonl(tmp_path / "resume_events.jsonl", {"event": "retry_unknown", "request_id": "old",
        "replacement_request_id": "replacement", "decided_at": "2026-01-01T00:00:02Z"})
    with pytest.raises(ResumeError):
        RunJournal(tmp_path, resume=True)


def test_unresolved_root_cannot_be_bypassed_by_newer_root(tmp_path: Path) -> None:
    append_jsonl(tmp_path / "requests.jsonl", _start("old"))
    newer = _start("newer")
    newer["started_at"] = "2026-01-01T00:00:01Z"
    append_jsonl(tmp_path / "requests.jsonl", newer)
    with pytest.raises(ResumeError):
        RunJournal(tmp_path, resume=True)


def test_completed_retry_can_be_followed_by_a_later_root(tmp_path: Path) -> None:
    replacement = _start("replacement")
    replacement["started_at"] = "2026-01-01T00:00:01Z"
    later = _start("later")
    later["started_at"] = "2026-01-01T00:00:02Z"
    replacement_complete = _complete("replacement")
    replacement_complete["started_at"] = replacement["started_at"]
    later_complete = _complete("later")
    later_complete["started_at"] = later["started_at"]
    append_jsonl(tmp_path / "requests.jsonl", _start("old"))
    append_jsonl(tmp_path / "requests.jsonl", replacement)
    append_jsonl(tmp_path / "requests.jsonl", replacement_complete)
    append_jsonl(tmp_path / "requests.jsonl", later)
    append_jsonl(tmp_path / "requests.jsonl", later_complete)
    append_jsonl(tmp_path / "resume_events.jsonl", {"event": "retry_unknown", "request_id": "old",
        "replacement_request_id": "replacement", "decided_at": "2026-01-01T00:00:03Z"})
    journal = RunJournal(tmp_path, resume=True)
    assert journal.next_request(_descriptor()).request_id == "replacement"
    assert journal.next_request(_descriptor()).request_id == "later"


def test_accounting_counts_old_and_replacement_physical_calls_once(tmp_path: Path) -> None:
    append_jsonl(tmp_path / "requests.jsonl", _start("old"))
    append_jsonl(tmp_path / "calls.jsonl", _receipt("old"))
    with pytest.raises(UnknownRequestsError):
        RunJournal(tmp_path, resume=True)
    journal = RunJournal(tmp_path, resume=True, retry_unknown_ids=("old",))
    replacement = journal.next_request(_descriptor()).request_id
    replacement_start = _start(replacement)
    replacement_start["started_at"] = "2026-01-01T00:00:01Z"
    append_jsonl(tmp_path / "requests.jsonl", replacement_start)
    append_jsonl(tmp_path / "calls.jsonl", _receipt(replacement))
    with pytest.raises(UnknownRequestsError):
        RunJournal(tmp_path, resume=True)
    resumed = RunJournal(tmp_path, resume=True, retry_unknown_ids=(replacement,))
    calls_by_stage, counts, failures = resumed.accounting()
    assert counts == {"generation": 2}
    assert len(calls_by_stage["generation"]) == 2
    assert len(failures) == 2 and all(item["accounting_incomplete"] for item in failures)


def test_checkpoint_is_suppressed_before_frontier_and_preserves_accepted_subset(tmp_path: Path) -> None:
    append_jsonl(tmp_path / "requests.jsonl", _start())
    append_jsonl(tmp_path / "requests.jsonl", _complete())
    atomic_report = {"schema_version": RESUME_SCHEMA_VERSION,
                     "records": [{"cue_id": "cue-1", "status": "accepted", "final_text": "kept"}]}
    (tmp_path / "report.json").write_text(json.dumps(atomic_report), encoding="utf-8")
    journal = RunJournal(tmp_path, resume=True)
    assert journal.checkpoint_allowed([]) is False
    journal.next_request(_descriptor())
    assert journal.checkpoint_allowed([
        {"cue_id": "cue-1", "status": "accepted", "final_text": "kept"},
        {"cue_id": "cue-2", "status": "accepted", "final_text": "new"},
    ]) is True
    with pytest.raises(ResumeError):
        journal.checkpoint_allowed([{ "cue_id": "cue-1", "status": "accepted", "final_text": "changed"}])


def test_report_schema_version_rejects_boolean(tmp_path: Path) -> None:
    append_jsonl(tmp_path / "requests.jsonl", _start())
    append_jsonl(tmp_path / "requests.jsonl", _complete())
    (tmp_path / "report.json").write_text(json.dumps({"schema_version": True, "records": []}), encoding="utf-8")
    with pytest.raises(ResumeError):
        RunJournal(tmp_path, resume=True)


@pytest.mark.parametrize("kind", ["missing_manifest", "legacy_manifest", "corrupt_manifest", "manifest_nonfinite",
                                  "manifest_duplicate", "missing_baseline", "legacy_baseline", "corrupt_baseline",
                                  "baseline_nonfinite", "baseline_duplicate"])
def test_manifest_and_baseline_compatibility_is_strict(tmp_path: Path, kind: str) -> None:
    contract = {"input": "frozen"}
    if kind != "missing_manifest":
        ensure_manifest(tmp_path, contract, resume=False)
    if kind == "legacy_manifest":
        manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
        manifest["schema_version"] = True
        (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    elif kind == "corrupt_manifest":
        (tmp_path / "run_manifest.json").write_text("{\"schema_version\":", encoding="utf-8")
    elif kind == "manifest_nonfinite":
        (tmp_path / "run_manifest.json").write_text("{\"schema_version\":1,\"value\":NaN}", encoding="utf-8")
    elif kind == "manifest_duplicate":
        (tmp_path / "run_manifest.json").write_text("{\"schema_version\":1,\"schema_version\":1}", encoding="utf-8")
    if kind in {"legacy_baseline", "corrupt_baseline", "baseline_nonfinite", "baseline_duplicate"}:
        save_baseline(tmp_path, {"plans": []})
        baseline_path = tmp_path / "resume_baseline.json"
        if kind == "legacy_baseline":
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline["schema_version"] = True
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        elif kind == "corrupt_baseline":
            baseline_path.write_text("{\"schema_version\":", encoding="utf-8")
        elif kind == "baseline_nonfinite":
            baseline_path.write_text("{\"schema_version\":1,\"payload\":NaN}", encoding="utf-8")
        elif kind == "baseline_duplicate":
            baseline_path.write_text("{\"schema_version\":1,\"schema_version\":1}", encoding="utf-8")
    if kind == "missing_baseline":
        assert load_baseline(tmp_path) is None
    elif kind in {"legacy_baseline", "corrupt_baseline", "baseline_nonfinite", "baseline_duplicate"}:
        with pytest.raises(ResumeError):
            load_baseline(tmp_path)
    else:
        with pytest.raises(ResumeError):
            ensure_manifest(tmp_path, contract, resume=True)
