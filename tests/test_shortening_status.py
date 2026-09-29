"""Read-only primary-run diagnostics with fake, local-only receipts."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
from typing import Any

import pytest

from utils.llm_types import CallRecord, DetailedResponse
from utils.opencode_transport import OpenCodePromptUsage
from utils.shortening_budget import BudgetPolicy
from utils.shortening_plan import PlannerConfig, plan_subtitles
from utils.shortening_resume import UnknownRequestsError, canonical_hash
from utils.shortening_runner import ModelSpec, RunConfig, _module_hashes, run_trial
from utils.shortening_status import main


_MODELS = {"generator": {"provider": "openrouter", "model": "meta/muse-spark-1.3-contributor", "variant": "low"},
           "arbiter": {"provider": "openrouter", "model": "openai/gpt-5.6-sol", "variant": "medium"}}
_ID = "a" * 32
_RECEIPT = "gen-" + "b" * 31


def _write(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def _append(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data) + "\n")


def _policy() -> BudgetPolicy:
    entries: dict[str, Any] = {}
    for role, model in _MODELS.items():
        name = model["model"]
        entries[name] = {
            "model": name, "canonical_model": name, "effort": model["variant"],
            "provider_slug": "meta" if role == "generator" else "openai",
            "provider_name": "Meta" if role == "generator" else "OpenAI",
            "context_length": 20000, "max_input_tokens": 1000, "max_output_tokens": 16384,
            "max_input_rate": "0.000001", "max_output_rate": "0.000001",
            "model_metadata": {}, "endpoint_metadata": {},
        }
    snapshot: dict[str, Any] = {"schema": "openrouter_production.v1", "id": "",
                                "captured_at": "2026-09-29T00:00:00Z", "models": entries}
    snapshot["id"] = canonical_hash({key: value for key, value in snapshot.items() if key != "id"})
    return BudgetPolicy.from_snapshot("10", snapshot)


def _fixture(root: Path, *, outstanding: bool = False, receipt: bool = True,
             failure: bool = False, changed_hash: bool = False) -> None:
    root.mkdir()
    policy = _policy()
    snapshot = policy.snapshot
    hashes = _module_hashes(primary=True)
    if changed_hash:
        hashes["shortening_resume.py"] = "0" * 64
    items = [{"index": 1, "text": "SECRET SUBTITLE"}]
    source: dict[str, Any] = {"items": items, "sha256": canonical_hash(items)}
    contract = {"source": source, "models": _MODELS, "module_sha256": hashes,
                "target_ids": ["cue-1"], "planner": {"length_basis": "normalized_tts_text"},
                "budget_policy": {"limit_usd": "10", "max_requests": 1000,
                                  "pricing_snapshot_id": snapshot["id"], "snapshot": snapshot}}
    _write(root / "run_manifest.json", {"schema_version": 1, "runner_algorithm_version": "shortening-resume-v1",
                                        "contract": contract, "contract_hash": canonical_hash(contract)})
    baseline = {"source_hash": source["sha256"], "selected_ids": ["cue-1"],
                "source_ids": ["cue-1"], "length_basis": "normalized_tts_text",
                "plans": [{}], "records": [{"cue_id": "cue-1", "status": "pending"}]}
    _write(root / "resume_baseline.json", {"schema_version": 1, "runner_algorithm_version": "shortening-resume-v1",
                                           "payload": baseline, "checksum": canonical_hash(baseline)})
    start = {"state": "start", "request_id": _ID, "stage": "generation", "model": _MODELS["generator"],
             "target_ids": ["cue-1"], "system_prompt": "SECRET PROMPT", "user_prompt": "SECRET SUBTITLE",
             "output_schema": {}, "allow_tool": False, "started_at": "2026-09-29T00:00:00Z", "elapsed_sec": 0,
             "reservation": policy.reservation(_MODELS["generator"])}
    _append(root / "requests.jsonl", start)
    calls: list[CallRecord] = []
    if receipt:
        record = CallRecord(provider="openrouter", model=_MODELS["generator"]["model"], variant="low",
                            requested_model=_MODELS["generator"]["model"], requested_variant="low",
                            started_at=start["started_at"], elapsed_sec=1, status="ok", generation_id=_RECEIPT,
                            usage=OpenCodePromptUsage(cost=None if outstanding or failure else 0.2))
        calls.append(record)
        _append(root / "calls.jsonl", {"request_id": _ID, "call_index": 0, "stage": "generation",
                                       "target_ids": ["cue-1"], "model": _MODELS["generator"],
                                       "provider": "openrouter", "record": asdict(record), "cost_estimate": {}})
    failed = {"request_id": _ID, "stage": "generation", "model": _MODELS["generator"],
              "target_ids": ["cue-1"], "started_at": start["started_at"], "error": "SECRET RAW ERROR",
              "accounting_incomplete": True, "physical_call_count": int(receipt)}
    if not outstanding:
        _append(root / "requests.jsonl", {"state": "complete", "request_id": _ID,
                                            "stage": "generation", "model": _MODELS["generator"],
                                            "target_ids": ["cue-1"], "started_at": start["started_at"],
                                            "elapsed_sec": 1, "call_count": len(calls),
                                            "text": None if failure else "SECRET RESPONSE", "error": failed["error"] if failure else None,
                                            "failure": failed if failure else None})
    failures = ([failed] if failure else [{**failed, "error": "unknown_request"}] if outstanding else [])
    budget = policy.summary(calls, failures)
    run_state = "incomplete" if outstanding or failure else "completed"
    _write(root / "report.json", {"schema_version": 1, "run_state": run_state, "records": baseline["records"],
                                  "selected_ids": ["cue-1"],
                                  "source_manifest": {"sha256": source["sha256"]}, "model": _MODELS,
                                  "summary": {"request_counts": {"generation": 1, "review": 0, "repair": 0, "recheck": 0}},
                                  "budget": budget})
    _write(root / "resume_status.json", {"run_state": run_state})
    (root / ".run.lock").touch()


def _run(root: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any], str]:
    code = main(["--run-dir", str(root)])
    raw = capsys.readouterr().out
    return code, json.loads(raw), raw


def test_completed_is_read_only_and_redacted(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "run"
    _fixture(root)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()}
    monkeypatch.setattr("socket.socket.connect", lambda *args: pytest.fail("network attempted"))
    code, status, raw = _run(root, capsys)
    assert code == 0 and status["status"] == "completed_snapshot"
    assert status["budget"]["known_spent_usd"] == "0.2"
    assert status["requests"][0]["receipt_ids"] == [_RECEIPT]
    assert "SECRET" not in raw
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()}


def _real_run(root: Path, *, cancel: bool = False, zero_targets: bool = False) -> None:
    items = [{"index": 1, "start": 0, "end": 1000, "text": "SECRET subtitle needs shortening"}]
    planner = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(items, planner)
    generator = ModelSpec(**_MODELS["generator"])
    arbiter = ModelSpec(**_MODELS["arbiter"])

    async def backend(model: ModelSpec, _system: str, user: str, schema: dict[str, Any], _tool: bool) -> DetailedResponse:
        if cancel:
            raise asyncio.CancelledError
        ids = [target["cue_id"] for target in json.loads(user.split("Input JSON:\n", 1)[1])["targets"]]
        rows = ([{"cue_id": cue_id, "verdict": "accept", "issues": [], "advice": ""} for cue_id in ids]
                if "verdict" in json.dumps(schema)
                else [{"cue_id": cue_id, "text": "short", "status": "ok"} for cue_id in ids])
        record = CallRecord(provider="openrouter", model=model.model, variant=model.variant,
                            requested_model=model.model, requested_variant=model.variant,
                            started_at="2026-09-29T00:00:00Z", elapsed_sec=1, status="ok",
                            generation_id=_RECEIPT, usage=OpenCodePromptUsage(cost=0.001))
        return DetailedResponse(json.dumps({"results": rows}), [record])

    options = dict(items=items, plans=plans, target_ids=[] if zero_targets else [plans[0].cue_id],
                   planner_config=planner, run_config=RunConfig(max_repairs=0),
                   generator=generator, arbiter=arbiter, output_dir=root, request_fn=backend,
                   budget_policy=_policy())
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run_trial(**options))
    else:
        assert asyncio.run(run_trial(**options))["run_state"] == "completed"


def test_real_runner_completed_snapshot(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "real-completed"
    _real_run(root)
    report = json.loads((root / "report.json").read_text())
    assert report["records"][0]["status"] == "accepted"
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir() if p.is_file()}
    code, status, raw = _run(root, capsys)
    assert code == 0 and status["status"] == "completed_snapshot"
    assert len(status["requests"]) == 2 and status["budget"]["known_spent_usd"] == "0.002"
    assert "SECRET" not in raw
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir() if p.is_file()}


def test_real_runner_cancellation_and_blocked_sidecar(tmp_path: Path,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "real-cancelled"
    _real_run(root, cancel=True)
    assert json.loads((root / "report.json").read_text())["run_state"] == "incomplete"
    assert json.loads((root / "resume_status.json").read_text())["status"] == "incomplete"
    code, status, raw = _run(root, capsys)
    assert code == 2 and status["saved_run_state"] == "incomplete"
    assert status["reason_codes"][:2] == ["unknown_request", "blocked_unknown_cost"]
    assert status["unknown_requests"] and status["budget"]["unknown_reserved_usd"] != "0"
    assert "SECRET" not in raw

    items = [{"index": 1, "start": 0, "end": 1000, "text": "SECRET subtitle needs shortening"}]
    planner = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(items, planner)

    async def forbidden(*_args: Any) -> DetailedResponse:
        pytest.fail("backend called")

    with pytest.raises(UnknownRequestsError):
        asyncio.run(run_trial(items=items, plans=plans, target_ids=[plans[0].cue_id],
                              planner_config=planner, run_config=RunConfig(max_repairs=0),
                              generator=ModelSpec(**_MODELS["generator"]),
                              arbiter=ModelSpec(**_MODELS["arbiter"]), output_dir=root,
                              request_fn=forbidden, budget_policy=_policy(), resume=True))
    sidecar = json.loads((root / "resume_status.json").read_text())
    assert sidecar["status"] == "blocked_unknown" and "run_state" not in sidecar
    code, status, _ = _run(root, capsys)
    assert code == 2 and "saved_status_blocked_unknown" in status["reason_codes"]
    assert status["unknown_requests"]


def test_real_runner_zero_targets_without_journals(tmp_path: Path,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "zero-targets"
    _real_run(root, zero_targets=True)
    assert not (root / "requests.jsonl").exists() and not (root / "calls.jsonl").exists()
    code, status, _ = _run(root, capsys)
    assert code == 0 and status["status"] == "completed_snapshot"
    assert status["budget"]["known_spent_usd"] == "0" and status["requests"] == []


@pytest.mark.parametrize("receipt", [False, True])
def test_outstanding_request_is_not_retried_or_ignored(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                       receipt: bool) -> None:
    root = tmp_path / "run"
    _fixture(root, outstanding=True, receipt=receipt)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    code, status, raw = _run(root, capsys)
    assert code == 2 and "unknown_request" in status["reason_codes"]
    assert status["budget"]["unknown_reserved_usd"] != "0"
    assert status["unknown_requests"][0]["receipt_ids"] == ([_RECEIPT] if receipt else [])
    assert status["failures"][0]["reason"] == "unknown_request"
    assert "SECRET" not in raw
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


def test_completed_failure_has_unknown_cost_not_unknown_request(tmp_path: Path,
                                                                capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "run"
    _fixture(root, failure=True)
    code, status, raw = _run(root, capsys)
    assert code == 2 and status["unknown_requests"] == []
    assert status["budget"]["unknown_reserved_usd"] != "0"
    assert status["failures"][0]["reason"] == "other_error"
    assert status["failures"][0]["receipt_ids"] == [_RECEIPT]
    assert "SECRET" not in raw


def test_incomplete_snapshot_with_known_cost_is_not_completed(tmp_path: Path,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "run"
    _fixture(root)
    report = json.loads((root / "report.json").read_text())
    report["run_state"] = "incomplete"
    _write(root / "report.json", report)
    _write(root / "resume_status.json", {"run_state": "incomplete"})
    code, status, _ = _run(root, capsys)
    assert code == 2 and status["reason_codes"] == ["run_incomplete"]
    assert status["budget"]["known_spent_usd"] == "0.2"


def test_completed_failure_without_receipt_reserves_unknown_cost(tmp_path: Path,
                                                                 capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "run"
    _fixture(root, failure=True, receipt=False)
    code, status, _ = _run(root, capsys)
    assert code == 2 and not status["unknown_requests"]
    assert status["budget"]["unknown_reserved_usd"] != "0"


@pytest.mark.parametrize("missing", ["run_manifest.json", "resume_baseline.json", "requests.jsonl", "report.json"])
def test_missing_artifact_is_invalid(tmp_path: Path, capsys: pytest.CaptureFixture[str], missing: str) -> None:
    root = tmp_path / "run"
    _fixture(root)
    (root / missing).unlink()
    code, status, _ = _run(root, capsys)
    assert code == 2 and status["status"] == "invalid" and "budget" not in status


def test_missing_journal_with_saved_cost_evidence_is_invalid(tmp_path: Path,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "run"
    _fixture(root)
    (root / "requests.jsonl").unlink()
    (root / "calls.jsonl").unlink()
    report = json.loads((root / "report.json").read_text())
    report["summary"]["request_counts"] = dict.fromkeys(report["summary"]["request_counts"], 0)
    report["budget"]["physical_calls"] = 0
    _write(root / "report.json", report)
    code, status, _ = _run(root, capsys)
    assert code == 2 and status["reason_codes"] == ["missing_journal"]


def test_corrupt_and_inconsistent_artifacts_are_invalid(tmp_path: Path,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "run"
    _fixture(root)
    (root / "calls.jsonl").write_text("not json\n", encoding="utf-8")
    assert _run(root, capsys)[1]["status"] == "invalid"
    _fixture(other := tmp_path / "other")
    report = json.loads((other / "report.json").read_text())
    report["source_manifest"]["sha256"] = "bad"
    _write(other / "report.json", report)
    assert _run(other, capsys)[1]["status"] == "invalid"
    _fixture(third := tmp_path / "third")
    _write(third / "resume_status.json", {"status": "processed"})
    assert _run(third, capsys)[1]["status"] == "invalid"


def test_changed_module_hash_blocks_completed_snapshot(tmp_path: Path,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "run"
    _fixture(root, changed_hash=True)
    code, status, _ = _run(root, capsys)
    assert code == 2 and status["module_hash_differences"] == ["shortening_resume.py"]


def test_unsupported_legacy_run_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "run"
    _fixture(root)
    manifest = json.loads((root / "run_manifest.json").read_text())
    manifest["contract"]["models"]["generator"]["provider"] = "opencode"
    manifest["contract_hash"] = canonical_hash(manifest["contract"])
    _write(root / "run_manifest.json", manifest)
    code, status, _ = _run(root, capsys)
    assert code == 2 and status["reason_codes"] == ["unsupported_run"]


@pytest.mark.parametrize("snapshot", [[], None, "SECRET VALUE", {"id": []}])
def test_malformed_nested_snapshot_is_sanitized(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                snapshot: Any) -> None:
    root = tmp_path / "run"
    _fixture(root)
    manifest = json.loads((root / "run_manifest.json").read_text())
    manifest["contract"]["budget_policy"]["snapshot"] = snapshot
    manifest["contract_hash"] = canonical_hash(manifest["contract"])
    _write(root / "run_manifest.json", manifest)
    code, status, raw = _run(root, capsys)
    assert code == 2 and status["status"] == "invalid"
    assert "SECRET" not in raw


def test_fifo_is_rejected_without_opening_it(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "run"
    _fixture(root)
    (root / "calls.jsonl").unlink()
    os.mkfifo(root / "calls.jsonl")
    code, status, _ = _run(root, capsys)
    assert code == 2 and status["reason_codes"] == ["invalid_artifacts"]


def test_active_lock_refuses_inspection_without_writes(tmp_path: Path,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "run"
    _fixture(root)
    with (root / ".run.lock").open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code, status, _ = _run(root, capsys)
    assert code == 2 and status["reason_codes"] == ["concurrent_run"]


def test_changed_files_refuse_snapshot(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from utils import shortening_status

    root = tmp_path / "run"
    _fixture(root)
    original = shortening_status._signature
    count = 0

    def changing_signature(directory: Path) -> tuple[Any, ...]:
        nonlocal count
        count += 1
        signature = original(directory)
        return signature if count == 1 else (*signature, ("changed", None))

    monkeypatch.setattr(shortening_status, "_signature", changing_signature)
    code, status, _ = _run(root, capsys)
    assert code == 2 and status["reason_codes"] == ["concurrent_change"]
