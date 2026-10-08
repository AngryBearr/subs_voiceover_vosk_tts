"""No external assets, credentials, audio, or HTML publication in migration tests."""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from utils import shortening_migrate as migration
from utils.llm_types import CallRecord, DetailedResponse
from utils.opencode_transport import OpenCodePromptUsage
from utils.shortening_budget import BudgetPolicy, BudgetStop
from utils.shortening_plan import CuePlan, PlannerConfig, plan_subtitles
from utils.shortening_resume import ResumeError, UnknownRequestsError, canonical_hash, inspect_journal, load_manifest_contract
from utils.shortening_runner import ModelSpec, RunConfig, run_trial


MUSE = ModelSpec("openrouter", "meta/muse-spark-1.3-contributor", "low")
SOL = ModelSpec("openrouter", "openai/gpt-6-sol", "high")


def _policy() -> BudgetPolicy:
    models: dict[str, Any] = {}
    for model, provider, in_rate, out_rate in ((MUSE, "Meta", "0.0000001", "0.0000002"),
                                                (SOL, "OpenAI", "0.000005", "0.000015")):
        models[model.model] = {
            "model": model.model, "canonical_model": model.model + "-20260922", "effort": model.variant,
            "provider_slug": model.model.split("/")[0], "provider_name": provider,
            "context_length": 1_050_000, "max_input_tokens": 922_000, "max_output_tokens": 16_384,
            "max_input_rate": in_rate, "max_output_rate": out_rate, "request_fee": "0",
            "request_fee_source": "not_listed_assumed_zero", "model_metadata": {}, "endpoint_metadata": {},
        }
    snapshot: dict[str, Any] = {"schema": "openrouter_production.v1", "id": "",
                                "captured_at": "2026-09-29T00:00:00Z", "models": models}
    snapshot["id"] = canonical_hash({k: v for k, v in snapshot.items() if k != "id"})
    return BudgetPolicy.from_snapshot("10", snapshot, max_requests=1000, max_unknown_calls=3)


def _response(spec: ModelSpec, user: str, *, number: int) -> DetailedResponse:
    targets = json.loads(user.split("Input JSON:\n", 1)[1])["targets"]
    review = spec == SOL
    result = ({"results": [{"cue_id": row["cue_id"], "verdict": "accept", "issues": [], "advice": ""}
                            for row in targets]} if review else
              {"results": [{"cue_id": row["cue_id"], "status": "ok", "text": "short"}
                            for row in targets]})
    call = CallRecord("openrouter", spec.model, spec.variant, "2026-09-29T00:00:00Z", 0.01,
                      OpenCodePromptUsage(input_tokens=12, output_tokens=3, cost=0.001), "success",
                      requested_model=spec.model, requested_variant=spec.variant,
                      generation_id=f"synthetic-{number}", provider_name="OpenAI" if review else "Meta", is_byok=False)
    return DetailedResponse(json.dumps(result), [call])


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, str]:
    """Generate a tiny stage-major journal; patch only this episode's fixed identity/shape."""
    items = [{"index": i + 1, "start": i * 3000, "end": i * 3000 + 1500,
              "text": "alpha beta gamma delta epsilon"} for i in range(36)]
    planner = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(items, planner)
    selected = [plan.cue_id for plan in plans]
    old = tmp_path / "source"
    destination = tmp_path / "new"
    policy = _policy()
    input_hash = "fixture-input-sha256"
    metadata: dict[str, Any] = {
        "input": {"path": str(tmp_path / "fixture.srt"), "absolute_path": str(tmp_path / "fixture.srt"),
                  "sha256": input_hash, "format": "srt"},
        "calibration": {"path": str(tmp_path / "fixture-profile.json")}, "model_metadata": {},
        "model_path": str(tmp_path / "fixture-model"), "voice": 8, "rate": 1.0,
        "roles": {"generator": asdict(MUSE), "arbiter": asdict(SOL)},
        "openrouter": {"pricing_snapshot": policy.snapshot},
        "budget": {"limit_usd": "10", "max_requests": 1000,
                   "reservation_basis": "published_context_and_max_rates", "max_unknown_calls": 3,
                   "retry_policy": "one queued recovery per failed cue/stage; no automatic replay of completed successes"},
        "config": {"planner": asdict(planner), "run": {**asdict(RunConfig()), "timeout": 300.0, "max_tokens": None}},
    }
    attempts = 0

    async def interrupted(spec: ModelSpec, _system: str, user: str, _schema: Any, _tools: bool) -> DetailedResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 3:
            raise asyncio.CancelledError()
        return _response(spec, user, number=attempts)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_trial(
            items=items, plans=plans, target_ids=selected, planner_config=planner,
            run_config=RunConfig(), generator=MUSE, arbiter=SOL, output_dir=old,
            request_fn=interrupted, original_timing_verifier=lambda *_: {},
            report_metadata=metadata, budget_policy=policy,
        ))
    (old / "preflight_audio").mkdir()
    (old / "preflight_audio" / "audio_report.json").write_text('{"records": []}\n', encoding="utf-8")
    # Historical source predates the explicit scheduling key.
    manifest_path = old / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["contract"]["run"].pop("scheduling")
    manifest["contract"]["report_metadata"]["config"]["run"].pop("scheduling")
    manifest["contract_hash"] = canonical_hash(manifest["contract"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report_path = old / "report.json"
    report = json.loads(report_path.read_text())
    report["config"]["run"].pop("scheduling")
    report["metadata"]["config"]["run"].pop("scheduling")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    metadata["config"]["run"].pop("scheduling")
    unknown = json.loads((old / "requests.jsonl").read_text().splitlines()[-1])["request_id"]
    monkeypatch.setattr(migration, "SOURCE_ITEMS_HASH", canonical_hash(items))
    monkeypatch.setattr(migration, "SOURCE_BYTES_HASH", input_hash)
    monkeypatch.setattr(migration, "SOURCE_REPORT_HASH", migration._sha(old / "report.json"))
    monkeypatch.setattr(migration, "SOURCE_STARTS", 3)
    monkeypatch.setattr(migration, "SOURCE_PROPOSALS", 12)
    monkeypatch.setattr(migration, "SOURCE_AUDIO_ROWS", 0)
    monkeypatch.setattr(migration, "SOURCE_SELECTED", 36)
    monkeypatch.setattr(migration, "SOURCE_ITEMS", 36)
    monkeypatch.setattr(migration, "SOURCE_ACTUAL_FITS", 0)
    monkeypatch.setattr(migration, "SOURCE_UNKNOWN_ID", unknown)
    frozen = inspect_journal(old, policy)
    summary = policy.summary(frozen.calls, frozen.failures)
    monkeypatch.setattr(migration, "SOURCE_KNOWN_USD", summary["known_spent_usd"])
    monkeypatch.setattr(migration, "SOURCE_RESERVED_USD", summary["unknown_reserved_usd"])
    monkeypatch.setattr(migration, "SOURCE_UNKNOWN_COUNT", summary["unknown_call_count"])

    def prepared(_args: Any) -> SimpleNamespace:
        return SimpleNamespace(input_bytes_sha256=input_hash, items=items, metadata=metadata,
                               planner_config=planner, plans=plans, target_ids=selected)

    monkeypatch.setattr(migration, "prepare_run", prepared)
    monkeypatch.setattr(migration, "publish_html_report", lambda _p, *, confirmed_real_run: Path("unpublished"))
    return old, destination, unknown


def _migrate(source: tuple[Path, Path, str]) -> Path:
    old, destination, _ = source
    before = {name: migration._sha(old / name) for name in migration.EVIDENCE}
    migration.migrate(old, destination)
    assert before == {name: migration._sha(old / name) for name in migration.EVIDENCE}
    return destination


def _resume(destination: Path, *, unknown: str | None = None, request: Any = None,
            stop: Any = None, limit: str = "10") -> dict[str, Any]:
    contract = load_manifest_contract(destination)
    frozen = contract["budget_policy"]
    policy = BudgetPolicy.from_snapshot(limit, frozen["snapshot"], max_requests=frozen["max_requests"],
                                         max_unknown_calls=frozen["max_unknown_calls"])

    async def no_call(*_: Any) -> DetailedResponse:
        raise AssertionError("unexpected transport call")

    def no_audio(*_: Any) -> Any:
        raise AssertionError("original audio preflight must not repeat")

    return asyncio.run(run_trial(
        items=copy.deepcopy(contract["source"]["items"]),
        plans=[CuePlan(**row) for row in contract["initial_plans"]], target_ids=contract["target_ids"],
        planner_config=PlannerConfig(**contract["planner"]), run_config=RunConfig(**contract["run"]),
        generator=MUSE, arbiter=SOL, output_dir=destination, request_fn=request or no_call,
        original_timing_verifier=no_audio, report_metadata=contract["report_metadata"],
        resume=True, retry_unknown_ids=(unknown,) if unknown else (), budget_policy=policy,
        stop_requested=stop,
    ))


def test_migration_integrity_and_source_rejections(source: tuple[Path, Path, str]) -> None:
    old, destination, _unknown = source
    _migrate(source)
    inherited = load_manifest_contract(destination)["report_metadata"]["migration"]
    assert inherited["inherited_generation_starts"] == 3
    assert inherited["inherited_proposals"] == 12
    assert all((old / name).read_bytes() == (destination / name).read_bytes()
               for name in ("resume_baseline.json", "requests.jsonl", "calls.jsonl", "preflight_audio/audio_report.json"))
    with pytest.raises(UnknownRequestsError):
        _resume(destination)
    with pytest.raises(FileExistsError):
        migration.migrate(old, destination)
    for name, change in (
        ("report.json", lambda data: data["records"][-1].update(candidate="fabricated")),
        ("report.json", lambda data: data["summary"].update(accepted_count=1)),
        ("run_manifest.json", lambda data: data["contract"]["report_metadata"]["budget"].update(limit_usd="20")),
    ):
        path = old / name
        before = path.read_bytes()
        data = json.loads(before)
        change(data)
        if name == "run_manifest.json":
            data["contract_hash"] = canonical_hash(data["contract"])
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ResumeError):
            migration.migrate(old, old.parent / "rejected")
        assert not (old.parent / "rejected").exists()
        path.write_bytes(before)


@pytest.mark.parametrize("damage", ("unknown_start", "completed_pair", "receipt", "baseline", "preflight"))
def test_inherited_bytes_fail_before_retry_or_network(source: tuple[Path, Path, str], damage: str) -> None:
    destination = _migrate(source)
    unknown = source[2]
    requests = destination / "requests.jsonl"
    calls = destination / "calls.jsonl"
    if damage == "unknown_start":
        requests.write_bytes(b"".join(requests.read_bytes().splitlines(keepends=True)[:-1]))
    elif damage == "completed_pair":
        rows = requests.read_bytes().splitlines(keepends=True)
        requests.write_bytes(b"".join(rows[:-3]))
        calls.write_bytes(b"".join(calls.read_bytes().splitlines(keepends=True)[:-1]))
    elif damage == "receipt":
        raw = calls.read_bytes()
        calls.write_bytes(raw.replace(b"synthetic-1", b"synthetic-x", 1))
    elif damage == "preflight":
        path = destination / "preflight_audio" / "audio_report.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        path = destination / "resume_baseline.json"
        path.write_bytes(path.read_bytes() + b" ")
    before = {name: migration._sha(destination / name) for name in ("report.json", "requests.jsonl", "calls.jsonl")}
    with pytest.raises(ResumeError, match="inherited evidence"):
        _resume(destination, unknown=unknown)
    assert not (destination / "resume_events.jsonl").exists()
    assert before == {name: migration._sha(destination / name) for name in before}


def test_windows_stop_and_resume_without_duplicate_calls(source: tuple[Path, Path, str]) -> None:
    destination = _migrate(source)
    unknown = source[2]
    calls: list[str] = []

    async def fake(spec: ModelSpec, _system: str, user: str, _schema: Any, _tools: bool) -> DetailedResponse:
        calls.append("review" if spec == SOL else "generation")
        return _response(spec, user, number=len(calls) + 10)

    with pytest.raises(ResumeError, match="canonical mismatch"):
        _resume(destination, unknown=unknown, limit="0.01")
    assert not (destination / "resume_events.jsonl").exists()
    frozen = load_manifest_contract(destination)["budget_policy"]
    tiny = BudgetPolicy.from_snapshot("0.01", frozen["snapshot"], max_unknown_calls=1000)
    inspected = inspect_journal(destination, tiny)
    with pytest.raises(BudgetStop):
        tiny.admit(MUSE, inspected.calls, inspected.failures)
    report = _resume(destination, unknown=unknown, request=fake, stop=lambda: len(calls) >= 4)
    assert calls == ["generation", "review", "generation", "review"]
    assert report["run_state"] == "incomplete" and report["summary"]["accepted_count"] >= 12
    assert report["budget"]["physical_calls"] == 3 + len(calls)
    assert report["budget"]["unknown_reserved_usd"] == migration.SOURCE_RESERVED_USD
    assert len((destination / "resume_events.jsonl").read_text().splitlines()) == 1
    with pytest.raises(ResumeError):
        _resume(destination, unknown=unknown, request=fake)
    old_count = len(calls)
    replayed = _resume(destination, request=fake, stop=lambda: True)
    assert len(calls) == old_count
    assert replayed["summary"]["accepted_count"] == report["summary"]["accepted_count"]
    # A later pending window is generated exactly once; earlier reviews replay.
    finished = _resume(destination, request=fake, stop=lambda: len(calls) >= old_count + 3)
    assert calls[old_count:] == ["generation", "generation", "review"]
    assert finished["summary"]["accepted_count"] >= report["summary"]["accepted_count"]
    assert _resume(destination, request=fake, stop=lambda: True)["summary"]["accepted_count"] == finished["summary"]["accepted_count"]
    assert len(calls) == old_count + 3


def test_forced_cancel_after_accepted_windows_retains_checkpoint(source: tuple[Path, Path, str]) -> None:
    destination = _migrate(source)
    seen: list[str] = []
    cancelled = False

    async def fake(spec: ModelSpec, _system: str, user: str, _schema: Any, _tools: bool) -> DetailedResponse:
        nonlocal cancelled
        seen.append("review" if spec == SOL else "generation")
        if len(seen) == 5 and not cancelled:
            cancelled = True
            raise asyncio.CancelledError()
        return _response(spec, user, number=len(seen) + 30)

    first = _resume(destination, unknown=source[2], request=fake, stop=lambda: len(seen) >= 4)
    accepted = first["summary"]["accepted_count"]
    assert accepted >= 12
    with pytest.raises(asyncio.CancelledError):
        _resume(destination, request=fake)
    interrupted = json.loads((destination / "report.json").read_text())
    assert interrupted["summary"]["accepted_count"] == accepted
    new_unknown = json.loads((destination / "requests.jsonl").read_text().splitlines()[-1])["request_id"]
    assert new_unknown != source[2]
    with pytest.raises(UnknownRequestsError):
        _resume(destination, request=fake)
    assert len(seen) == 5
    continued = _resume(destination, unknown=new_unknown, request=fake, stop=lambda: len(seen) >= 6)
    assert continued["summary"]["accepted_count"] == accepted
    assert len(seen) == 6
    events = [json.loads(row) for row in (destination / "resume_events.jsonl").read_text().splitlines()]
    assert [row["request_id"] for row in events] == [source[2], new_unknown]
