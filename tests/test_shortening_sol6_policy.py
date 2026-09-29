"""Offline regression checks for the new paid primary policy."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from utils import openrouter_transport as transport
from utils.llm_types import CallRecord, DetailedResponse
from utils.opencode_transport import OpenCodePromptUsage
from utils.shortening_budget import BudgetPolicy, BudgetStop
from utils.shortening_plan import PlannerConfig, plan_subtitles
from utils.shortening_resume import ResumeError, canonical_hash
from utils.shortening_runner import ModelSpec, RunConfig, run_trial
from utils.shortening_status import inspect_run


MUSE = ModelSpec("openrouter", "meta/muse-spark-1.3-contributor", "low")
SOL = ModelSpec("openrouter", "openai/gpt-6-sol", "high")


def _policy(*, limit: str = "10", max_requests: int = 1000,
            max_unknown_calls: int = 3) -> BudgetPolicy:
    models: dict[str, Any] = {}
    for model, provider, input_rate, output_rate in (
        (MUSE, "Meta", "0.0000001", "0.0000002"),
        (SOL, "OpenAI", "0.000005", "0.000015"),
    ):
        models[model.model] = {
            "model": model.model, "canonical_model": model.model + "-20260922",
            "effort": model.variant, "provider_slug": model.model.split("/")[0],
            "provider_name": provider, "context_length": 1_050_000,
            "max_input_tokens": 922_000, "max_output_tokens": 16_384,
            "max_input_rate": input_rate, "max_output_rate": output_rate,
            "request_fee": "0", "request_fee_source": "not_listed_assumed_zero",
            "model_metadata": {}, "endpoint_metadata": {},
        }
    snapshot: dict[str, Any] = {"schema": "openrouter_production.v1", "id": "",
                                "captured_at": "2026-09-29T00:00:00Z", "models": models}
    snapshot["id"] = canonical_hash({key: value for key, value in snapshot.items() if key != "id"})
    return BudgetPolicy.from_snapshot(limit, snapshot, max_requests=max_requests, max_unknown_calls=max_unknown_calls)


def _call(model: ModelSpec, *, cost: float | None = None, status: str = "success") -> CallRecord:
    return CallRecord("openrouter", model.model, model.variant, "2026-09-29T00:00:00Z", 0.01,
                      OpenCodePromptUsage(input_tokens=12, output_tokens=3, cost=cost), status,
                      requested_model=model.model, requested_variant=model.variant,
                      generation_id="gen-test", provider_name="Meta" if model == MUSE else "OpenAI", is_byok=False)


def test_missing_cost_valid_response_is_not_repeated(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy()
    body = {"id": "gen-test", "model": MUSE.model, "provider_name": "Meta", "is_byok": False,
            "choices": [{"finish_reason": "stop", "message": {"content": '{"results":[]}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3}}
    posts = 0
    async def post(**_: Any) -> tuple[int, bytes]:
        nonlocal posts
        posts += 1
        return 200, json.dumps(body).encode()
    async def lookup(**_: Any) -> dict[str, Any]:
        return {}
    monkeypatch.setattr(transport, "_http_post_json", post)
    monkeypatch.setattr(transport, "_generation_metadata", lookup)
    result = asyncio.run(transport.request_json_detailed(
        api_key="dummy", model=MUSE.model, reasoning_effort=MUSE.variant, system_prompt="s", user_prompt="u",
        schema={"type": "object"}, pricing_snapshot=policy.snapshot,
    ))
    assert posts == 1 and result.text is not None and result.calls[0].usage.cost is None
    assert policy.summary(result.calls, [])["unknown_reserved_usd"] == "0.0000018"


def test_cap_counts_unknown_receipts_retries_and_unreceipted_failure() -> None:
    policy = _policy(limit="4.86", max_requests=4)
    missing = {"request_id": "r2", "model": asdict(MUSE), "physical_call_count": 0,
               "accounting_incomplete": True}
    calls = [_call(MUSE), _call(MUSE, cost=0.01)]
    summary = policy.summary(calls, [missing])
    assert summary["unknown_call_count"] == 2 and summary["physical_calls"] == 3
    assert float(summary["unknown_reserved_usd"]) > 0
    with pytest.raises(BudgetStop, match="blocked_by_budget"):
        policy.admit(SOL, calls, [missing])
    with pytest.raises(BudgetStop, match="blocked_request_limit"):
        _policy(max_requests=3).admit(MUSE, calls, [missing])
    with pytest.raises(BudgetStop, match="blocked_unknown_limit"):
        policy.admit(MUSE, calls + [_call(MUSE)], [missing])


def test_permanent_http_error_and_foreign_route_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy()
    async def denied(**_: Any) -> tuple[int, bytes]:
        return 401, b'{"error":{"message":"not authorized"}}'
    monkeypatch.setattr(transport, "_http_post_json", denied)
    result = asyncio.run(transport.request_json_detailed(
        api_key="dummy", model=SOL.model, reasoning_effort=SOL.variant,
        system_prompt="s", user_prompt="u", schema={"type": "object"}, pricing_snapshot=policy.snapshot,
    ))
    assert result.error == "http_permanent" and len(result.calls) == 1
    foreign = _call(SOL)
    foreign.provider_name = "Azure"
    with pytest.raises(BudgetStop, match="blocked_unknown_cost"):
        policy.admit(MUSE, [foreign], [])


def test_new_policy_third_unknown_not_a_global_stop() -> None:
    policy = _policy(max_requests=10, max_unknown_calls=10)
    calls = [_call(MUSE) for _ in range(3)]
    assert policy.summary(calls, [])["status"] == "ok"
    assert policy.admit(MUSE, calls, [])
    limited = _policy(limit="0.095477", max_requests=10, max_unknown_calls=10)
    with pytest.raises(BudgetStop, match="blocked_by_budget"):
        limited.admit(MUSE, calls, [])


def _window_fixture(root: Path) -> dict[str, Any]:
    items = [{"index": i, "start": i * 3000, "end": i * 3000 + 1000,
              "text": "alpha beta gamma delta"} for i in range(3)]
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    return dict(items=items, plans=plans, target_ids=[plan.cue_id for plan in plans if plan.status == "needs_shortening"],
                planner_config=planner, run_config=RunConfig(gen_batch_size=1, review_batch_size=1,
                                                             max_repairs=1, scheduling="windowed"),
                generator=MUSE, arbiter=SOL, output_dir=root, budget_policy=_policy(max_unknown_calls=1000))


def _success(spec: ModelSpec, user: str) -> DetailedResponse:
    ids = [row["cue_id"] for row in json.loads(user.split("Input JSON:\n", 1)[1])["targets"]]
    rows = ([{"cue_id": cue_id, "text": "short", "status": "ok"} for cue_id in ids] if spec == MUSE else
            [{"cue_id": cue_id, "verdict": "accept", "issues": [], "advice": ""} for cue_id in ids])
    return DetailedResponse(json.dumps({"results": rows}), [_call(spec, cost=0.00001)])


def _unknown_id(root: Path) -> str:
    starts: list[str] = []
    completes: set[str] = set()
    for line in (root / "requests.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["state"] == "start":
            starts.append(row["request_id"])
        else:
            completes.add(row["request_id"])
    return next(request_id for request_id in reversed(starts) if request_id not in completes)


def test_window_acceptance_precedes_next_generation_and_graceful_resume(tmp_path: Path) -> None:
    options = _window_fixture(tmp_path / "window")
    paid: list[str] = []
    stopped = False

    async def fake(spec: ModelSpec, _system: str, user: str, _schema: dict[str, Any], _tools: bool) -> DetailedResponse:
        nonlocal stopped
        ids = [row["cue_id"] for row in json.loads(user.split("Input JSON:\n", 1)[1])["targets"]]
        if ids[0] == options["target_ids"][1]:
            checkpoint = json.loads((options["output_dir"] / "report.json").read_text())
            assert checkpoint["records"][0]["status"] == "accepted"
        paid.append(spec.model)
        if spec == SOL and ids[0] == options["target_ids"][0]:
            stopped = True
        return _success(spec, user)

    first = asyncio.run(run_trial(**options, request_fn=fake, stop_requested=lambda: stopped))
    assert first["run_state"] == "incomplete" and first["budget_stop"]["reason"] == "graceful_stop"
    assert first["records"][0]["status"] == "accepted"
    assert len(paid) == 2 and not _unknown_id_if_any(options["output_dir"])
    stopped = False
    second = asyncio.run(run_trial(**options, request_fn=fake, resume=True, stop_requested=lambda: stopped))
    assert second["run_state"] == "completed"
    assert len(paid) == 6 and second["records"][0]["status"] == "accepted"
    assert second["resume"]["replayed_request_count"] == 2


def _unknown_id_if_any(root: Path) -> bool:
    rows = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]
    return bool({row["request_id"] for row in rows if row["state"] == "start"} -
                {row["request_id"] for row in rows if row["state"] == "complete"})


def test_forced_cancel_keeps_accepted_and_replacement_risk_once(tmp_path: Path) -> None:
    options = _window_fixture(tmp_path / "forced")
    failures = 0

    async def fake(spec: ModelSpec, _system: str, user: str, _schema: dict[str, Any], _tools: bool) -> DetailedResponse:
        nonlocal failures
        ids = [row["cue_id"] for row in json.loads(user.split("Input JSON:\n", 1)[1])["targets"]]
        if spec == MUSE and ids[0] == options["target_ids"][1] and failures < 2:
            failures += 1
            raise asyncio.CancelledError()
        return _success(spec, user)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_trial(**options, request_fn=fake))
    first = json.loads((options["output_dir"] / "report.json").read_text())
    assert first["run_state"] == "incomplete" and first["records"][0]["status"] == "accepted"
    assert first["budget"]["physical_calls"] == 3
    old_unknown = _unknown_id(options["output_dir"])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_trial(**options, request_fn=fake, resume=True, retry_unknown_ids=(old_unknown,)))
    second = json.loads((options["output_dir"] / "report.json").read_text())
    replacement = _unknown_id(options["output_dir"])
    assert replacement != old_unknown and second["records"][0]["status"] == "accepted"
    assert second["budget"]["physical_calls"] == 4
    assert Decimal(second["budget"]["unknown_reserved_usd"]) == 2 * Decimal(options["budget_policy"].reservation(MUSE)["usd"])
    events_before = (options["output_dir"] / "resume_events.jsonl").read_bytes()
    with pytest.raises(ResumeError, match="one permitted replacement"):
        asyncio.run(run_trial(**options, request_fn=fake, resume=True, retry_unknown_ids=(replacement,)))
    assert (options["output_dir"] / "resume_events.jsonl").read_bytes() == events_before


@pytest.mark.parametrize("failure_site", ["snapshot", "checkpoint"])
def test_cancel_snapshot_failure_preserves_original_cancel_and_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure_site: str,
) -> None:
    import utils.shortening_runner as runner

    options = _window_fixture(tmp_path / failure_site)
    root = options["output_dir"]
    cancelled = asyncio.CancelledError("original cancellation")
    stored: dict[str, bytes] = {}
    attempted = 0

    def fail_snapshot(*_args: Any, **_kwargs: Any) -> Any:
        raise ResumeError("SECRET_CANARY")

    def fail_checkpoint(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("SECRET_CANARY")

    async def fake(spec: ModelSpec, _system: str, user: str, _schema: dict[str, Any], _tools: bool) -> DetailedResponse:
        nonlocal attempted
        attempted += 1
        ids = [row["cue_id"] for row in json.loads(user.split("Input JSON:\n", 1)[1])["targets"]]
        if spec == MUSE and ids == [options["target_ids"][1]]:
            assert json.loads((root / "report.json").read_text())["records"][0]["status"] == "accepted"
            stored.update({name: (root / name).read_bytes() for name in
                           ("report.json", "review_state.json", "requests.jsonl", "calls.jsonl")})
            if failure_site == "snapshot":
                monkeypatch.setattr(runner.RunJournal, "readonly_snapshot", fail_snapshot)
            else:
                monkeypatch.setattr(runner, "_checkpoint", fail_checkpoint)
            raise cancelled
        return _success(spec, user)

    with pytest.raises(asyncio.CancelledError) as caught:
        asyncio.run(run_trial(**options, request_fn=fake))
    assert caught.value is cancelled
    assert attempted == 3
    assert stored and all((root / name).read_bytes() == contents for name, contents in stored.items())
    assert json.loads((root / "report.json").read_text())["records"][0]["status"] == "accepted"
    assert _unknown_id(root)
    warning = capsys.readouterr().err
    expected_type = "ResumeError" if failure_site == "snapshot" else "OSError"
    assert f"({expected_type})" in warning
    assert "SECRET_CANARY" not in warning


def test_explicit_replacement_keeps_old_reservation_and_success_replays_free(tmp_path: Path) -> None:
    options = _window_fixture(tmp_path / "replace")
    calls = 0
    async def fake(spec: ModelSpec, _system: str, user: str, _schema: dict[str, Any], _tools: bool) -> DetailedResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.CancelledError()
        return _success(spec, user)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_trial(**options, request_fn=fake))
    old = _unknown_id(options["output_dir"])
    report = asyncio.run(run_trial(**options, request_fn=fake, resume=True, retry_unknown_ids=(old,)))
    assert report["run_state"] == "completed" and report["records"][0]["status"] == "accepted"
    assert report["budget"]["known_spent_usd"] == "0.00006"
    assert Decimal(report["budget"]["unknown_reserved_usd"]) == Decimal(options["budget_policy"].reservation(MUSE)["usd"])
    assert not report["resume"]["unknown_requests"]
    async def forbidden(*_args: Any) -> DetailedResponse:
        pytest.fail("completed resume issued a paid request")
    replay = asyncio.run(run_trial(**options, request_fn=forbidden, resume=True))
    assert replay["budget"]["physical_calls"] == report["budget"]["physical_calls"]


def test_retry_admission_denied_before_writing_intent(tmp_path: Path) -> None:
    options = _window_fixture(tmp_path / "cap")
    options["budget_policy"] = _policy(limit="0.1", max_requests=10, max_unknown_calls=10)
    async def cancel(*_args: Any) -> DetailedResponse:
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_trial(**options, request_fn=cancel))
    unknown = _unknown_id(options["output_dir"])
    with pytest.raises(BudgetStop, match="blocked_by_budget"):
        asyncio.run(run_trial(**options, request_fn=cancel, resume=True, retry_unknown_ids=(unknown,)))
    assert not (options["output_dir"] / "resume_events.jsonl").exists()


def test_exhausted_failure_in_one_window_does_not_block_next(tmp_path: Path) -> None:
    options = _window_fixture(tmp_path / "neighbor")
    first = options["target_ids"][0]
    async def fake(spec: ModelSpec, _system: str, user: str, _schema: dict[str, Any], _tools: bool) -> DetailedResponse:
        ids = [row["cue_id"] for row in json.loads(user.split("Input JSON:\n", 1)[1])["targets"]]
        if spec == MUSE and ids == [first]:
            missing = _call(spec, status="timeout")
            missing.usage = None
            missing.generation_id = None
            missing.provider_name = None
            missing.is_byok = None
            return DetailedResponse(None, [missing], "timeout")
        return _success(spec, user)
    report = asyncio.run(run_trial(**options, request_fn=fake))
    assert report["run_state"] == "completed"
    assert report["records"][0]["status"] == "error"
    assert report["records"][0]["final_text"] == report["records"][0]["original"]
    assert all(row["status"] == "accepted" for row in report["records"][1:])
    assert report["summary"]["request_counts"]["repair"] == 1


@pytest.mark.parametrize("recover", [True, False])
def test_failed_batch_recovers_once_and_preserves_neighbors(tmp_path: Path, recover: bool) -> None:
    items = [{"index": i, "start": i * 3000, "end": i * 3000 + 1000,
              "text": "alpha beta gamma delta"} for i in range(2)]
    planner = PlannerConfig(cps=5, max_speed=1.0, gap_ms=200)
    plans = plan_subtitles(items, planner)
    ids = [plan.cue_id for plan in plans if plan.status == "needs_shortening"]
    policy = _policy()
    attempts = 0

    async def fake(spec: ModelSpec, system: str, user: str, schema: dict[str, Any], tools: bool) -> DetailedResponse:
        nonlocal attempts
        attempts += 1
        targets = [row["cue_id"] for row in json.loads(user.split("Input JSON:\n", 1)[1])["targets"]]
        if spec == MUSE and targets == [ids[0]] and (attempts == 1 or not recover):
            call = _call(spec, status="timeout")
            call.usage = None
            call.generation_id = None
            call.provider_name = None
            call.is_byok = None
            return DetailedResponse(None, [call], "timeout")
        if spec == MUSE:
            text = {"results": [{"cue_id": cue_id, "text": "short", "status": "ok"} for cue_id in targets]}
        else:
            text = {"results": [{"cue_id": cue_id, "verdict": "accept", "issues": [], "advice": ""} for cue_id in targets]}
        return DetailedResponse(json.dumps(text), [_call(spec, cost=0.00001)])

    kwargs = dict(items=items, plans=plans, target_ids=ids, planner_config=planner,
                  run_config=RunConfig(gen_batch_size=1, review_batch_size=1, max_repairs=1),
                  generator=MUSE, arbiter=SOL, output_dir=tmp_path / "trial", request_fn=fake,
                  budget_policy=policy)
    report = asyncio.run(run_trial(**kwargs))
    assert report["run_state"] == "completed"
    assert report["summary"]["accounting_complete"] is False
    assert attempts <= 5
    assert report["records"][1]["status"] == "accepted"
    assert report["records"][0]["status"] == ("accepted" if recover else "error")
    assert inspect_run(tmp_path / "trial")["saved_run_state"] == "completed"
    async def forbidden(*_: Any) -> DetailedResponse:
        pytest.fail("completed resume made a new POST")
    replay = asyncio.run(run_trial(**{**kwargs, "resume": True, "request_fn": forbidden}))
    assert replay["budget"]["physical_calls"] == report["budget"]["physical_calls"]
    assert replay["resume"]["replayed_request_count"] > 0
    if not recover:
        assert report["records"][0]["final_text"] == report["records"][0]["original"]
        assert (tmp_path / "trial" / "review.json").exists()
