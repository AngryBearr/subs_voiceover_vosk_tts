from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from utils.llm_types import CallRecord, DetailedResponse
from utils.opencode_transport import OpenCodePromptUsage
from utils.shortening_plan import PlannerConfig, plan_subtitles
from utils.shortening_resume import UnknownRequestsError
from utils.shortening_runner import ModelSpec, RunConfig, run_trial


def _run_inputs() -> tuple[list[dict[str, Any]], PlannerConfig, list[Any], list[str]]:
    items = [{"index": 1, "start": 0, "end": 1000, "text": "alpha beta gamma delta"}]
    planner = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(items, planner)
    return items, planner, plans, [plans[0].cue_id]


def _response(user: str, schema: dict[str, Any]) -> DetailedResponse:
    payload = json.loads(user.split("Input JSON:\n", 1)[1])
    ids = [row["cue_id"] for row in payload["targets"]]
    if "verdict" in json.dumps(schema):
        value = {"results": [{"cue_id": cue_id, "verdict": "accept", "issues": [], "advice": ""} for cue_id in ids]}
    else:
        value = {"results": [{"cue_id": cue_id, "text": "short", "status": "ok"} for cue_id in ids]}
    return DetailedResponse(json.dumps(value), [])


def test_cancelled_attempt_requires_explicit_retry_and_then_completes(tmp_path: Path) -> None:
    items, planner, plans, ids = _run_inputs()
    output = tmp_path / "run"

    async def cancelled(*_: Any) -> DetailedResponse:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_trial(
            items=items, plans=plans, target_ids=ids, planner_config=planner,
            run_config=RunConfig(max_repairs=0), generator=ModelSpec("fixture", "g", "v"),
            arbiter=ModelSpec("fixture", "a", "v"), output_dir=output, request_fn=cancelled,
        ))

    async def should_not_run(*_: Any) -> DetailedResponse:
        raise AssertionError("unknown request must block before backend")

    with pytest.raises(UnknownRequestsError):
        asyncio.run(run_trial(
            items=items, plans=plans, target_ids=ids, planner_config=planner,
            run_config=RunConfig(max_repairs=0), generator=ModelSpec("fixture", "g", "v"),
            arbiter=ModelSpec("fixture", "a", "v"), output_dir=output, request_fn=should_not_run,
            resume=True,
        ))
    request_id = json.loads((output / "requests.jsonl").read_text(encoding="utf-8").splitlines()[0])["request_id"]

    async def retry(*_: Any) -> DetailedResponse:
        return _response(_[2], _[3])

    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=RunConfig(max_repairs=0), generator=ModelSpec("fixture", "g", "v"),
        arbiter=ModelSpec("fixture", "a", "v"), output_dir=output, request_fn=retry,
        resume=True, retry_unknown_ids=(request_id,),
    ))
    assert report["run_state"] == "incomplete"
    assert report["summary"]["unknown_request_ids"] == [request_id]


def test_completed_resume_replays_without_backend(tmp_path: Path) -> None:
    items, planner, plans, ids = _run_inputs()
    output = tmp_path / "run"
    calls = 0

    async def live(*args: Any) -> DetailedResponse:
        nonlocal calls
        calls += 1
        return _response(args[2], args[3])

    first = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=RunConfig(max_repairs=0), generator=ModelSpec("fixture", "g", "v"),
        arbiter=ModelSpec("fixture", "a", "v"), output_dir=output, request_fn=live,
    ))
    assert first["run_state"] == "completed" and calls == 2

    async def forbidden(*_: Any) -> DetailedResponse:
        raise AssertionError("completed resume should only replay")

    replayed = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=RunConfig(max_repairs=0), generator=ModelSpec("fixture", "g", "v"),
        arbiter=ModelSpec("fixture", "a", "v"), output_dir=output, request_fn=forbidden,
        resume=True,
    ))
    assert replayed["summary"]["accepted_count"] == 1


def test_completed_resume_preserves_identical_repair_skip_without_backend(tmp_path: Path) -> None:
    items, planner, plans, ids = _run_inputs()
    output = tmp_path / "identical-repair"
    calls = 0
    review_calls = 0

    async def live(*args: Any) -> DetailedResponse:
        nonlocal calls, review_calls
        calls += 1
        payload = json.loads(args[2].split("Input JSON:\n", 1)[1])
        target_ids = [row["cue_id"] for row in payload["targets"]]
        if "verdict" in json.dumps(args[3]):
            verdict = "revise" if review_calls == 0 else "accept"
            review_calls += 1
            value = {"results": [{"cue_id": cue_id, "verdict": verdict, "issues": [], "advice": "retry"}
                                  for cue_id in target_ids]}
        else:
            value = {"results": [{"cue_id": cue_id, "text": "short", "status": "ok"}
                                  for cue_id in target_ids]}
        return DetailedResponse(json.dumps(value), [])

    first = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=RunConfig(max_repairs=1), generator=ModelSpec("fixture", "g", "v"),
        arbiter=ModelSpec("fixture", "a", "v"), output_dir=output, request_fn=live,
    ))
    assert first["run_state"] == "completed"
    assert calls == 3
    first_record = first["records"][0]
    assert first_record["status"] == "manual"
    assert first_record["final_text"] == first_record["original"]
    assert first_record["failure"] == "repair_unchanged"
    assert [event["stage"] for event in first_record["history"]] == [
        "generation", "review", "repair", "recheck",
    ]
    assert first_record["history"][-1]["status"] == "skipped_identical_proposal"
    assert first["summary"]["repairs"] == 1
    assert first["summary"]["request_counts"] == {
        "generation": 1, "review": 1, "repair": 1, "recheck": 0,
    }

    async def forbidden(*_: Any) -> DetailedResponse:
        raise AssertionError("completed resume should only replay")

    replayed = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=RunConfig(max_repairs=1), generator=ModelSpec("fixture", "g", "v"),
        arbiter=ModelSpec("fixture", "a", "v"), output_dir=output, request_fn=forbidden,
        resume=True,
    ))
    assert calls == 3
    assert replayed["records"] == first["records"]
    assert replayed["summary"]["manual_count"] == 1
    assert replayed["records"][0]["final_text"] == replayed["records"][0]["original"]
    assert sum(event.get("status") == "skipped_identical_proposal"
               for event in replayed["records"][0]["history"]) == 1


def test_completed_resume_restores_full_file_plans_and_noncontiguous_targets(tmp_path: Path) -> None:
    items = [
        {"index": 10, "start": 0, "end": 1000, "text": "fit"},
        {"index": 11, "start": 1200, "end": 2200, "text": 123},
        {"index": 12, "start": 2400, "end": 3400, "text": "alpha beta gamma delta"},
        {"index": 13, "start": 3600, "end": 4600, "text": "one two three four five"},
    ]
    planner = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(items, planner)
    ids = ["cue-2", "cue-3"]
    output = tmp_path / "full-file"

    async def live(*args: Any) -> DetailedResponse:
        return _response(args[2], args[3])

    first = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=RunConfig(max_repairs=0), generator=ModelSpec("fixture", "g", "v"),
        arbiter=ModelSpec("fixture", "a", "v"), output_dir=output, request_fn=live,
    ))
    assert first["run_state"] == "completed"
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))["contract"]
    assert set(manifest) == {
        "source", "initial_plans", "target_ids", "planner", "run", "models",
        "report_metadata", "has_original_timing_verifier", "prompt_version", "prompts",
        "runner_algorithm_version", "module_sha256",
    }
    baseline = json.loads((output / "resume_baseline.json").read_text(encoding="utf-8"))["payload"]
    assert baseline["source_ids"] == [plan.cue_id for plan in plans]
    assert len(baseline["plans"]) == len(plans)
    assert [record["cue_id"] for record in baseline["records"]] == ids

    async def forbidden(*_: Any) -> DetailedResponse:
        raise AssertionError("ordinary full-file resume should not call the backend")

    replayed = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=RunConfig(max_repairs=0), generator=ModelSpec("fixture", "g", "v"),
        arbiter=ModelSpec("fixture", "a", "v"), output_dir=output, request_fn=forbidden,
        resume=True,
    ))
    assert replayed["records"] == first["records"]
    assert replayed["resume"]["replayed_request_count"] == 2


@pytest.mark.parametrize("interrupt_stage", ["generation", "review", "repair", "recheck"])
def test_resume_replays_durable_complete_before_checkpoint_for_every_llm_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt_stage: str,
) -> None:
    items = [
        {"index": 1, "start": 0, "end": 2000, "text": "alpha beta gamma delta"},
        {"index": 2, "start": 2200, "end": 4200, "text": "delta epsilon zeta eta"},
    ]
    planner = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(items, planner)
    ids = ["cue-0", "cue-1"]
    config = RunConfig(gen_batch_size=6, review_batch_size=1, max_repairs=1)
    model = ModelSpec("fixture", "g", "v")

    def make_backend(events: list[str]) -> Any:
        review_number = 0

        async def request(_: ModelSpec, __: str, user: str, schema: dict[str, Any], ___: bool) -> DetailedResponse:
            nonlocal review_number
            payload = json.loads(user.split("Input JSON:\n", 1)[1])
            request_ids = [row["cue_id"] for row in payload["targets"]]
            is_review = "verdict" in json.dumps(schema)
            stage = "review" if is_review else "generation"
            events.append(stage)
            if is_review:
                cue_id = request_ids[0]
                verdict = "accept" if review_number == 0 or review_number >= 2 else "revise"
                review_number += 1
                value: dict[str, Any] = {
                    "results": [{"cue_id": cue_id, "verdict": verdict, "issues": [], "advice": ""}]
                }
            else:
                is_repair = any("prior_candidate" in row for row in payload["targets"])
                text = "new short" if is_repair else "short"
                value = {"results": [{"cue_id": cue_id, "text": text, "status": "ok"} for cue_id in request_ids]}
            usage = OpenCodePromptUsage(input_tokens=10, output_tokens=5, total_tokens=15, cost=0.01)
            call = CallRecord("fixture", "g", "v", "2026-01-01T00:00:00Z", 0.1, usage, "ok")
            return DetailedResponse(json.dumps(value), [call])

        return request

    uninterrupted_events: list[str] = []
    uninterrupted = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=config, generator=model, arbiter=model, output_dir=tmp_path / "uninterrupted",
        request_fn=make_backend(uninterrupted_events),
    ))
    uninterrupted_exports = {
        name: (tmp_path / "uninterrupted" / name).read_bytes()
        for name in ("subtitles.json", "subtitles.srt")
    }

    interrupted_events: list[str] = []
    original_log = __import__("utils.shortening_runner", fromlist=["_request_log"])._request_log
    interrupted = False

    class DurableCompleteInterrupt(BaseException):
        pass

    completed_by_stage: dict[str, int] = {}

    def interrupting_log(*args: Any, **kwargs: Any) -> None:
        nonlocal interrupted
        original_log(*args, **kwargs)
        stage = args[2] if len(args) > 2 else kwargs["stage"]
        state = args[9] if len(args) > 9 else kwargs["state"]
        if state == "complete":
            completed_by_stage[stage] = completed_by_stage.get(stage, 0) + 1
        target_completion = 2 if interrupt_stage == "review" else 1
        if (not interrupted and stage == interrupt_stage and state == "complete"
                and completed_by_stage[stage] == target_completion):
            interrupted = True
            raise DurableCompleteInterrupt

    import utils.shortening_runner as shortening_runner
    monkeypatch.setattr(shortening_runner, "_request_log", interrupting_log)
    output = tmp_path / "interrupted"
    interrupted_request = make_backend(interrupted_events)
    with pytest.raises(DurableCompleteInterrupt):
        asyncio.run(run_trial(
            items=items, plans=plans, target_ids=ids, planner_config=planner,
            run_config=config, generator=model, arbiter=model, output_dir=output,
            request_fn=interrupted_request,
        ))

    before_resume_exports = {
        name: (output / name).read_bytes() for name in ("subtitles.json", "subtitles.srt")
    }
    before_resume_report = (output / "report.json").read_bytes()
    from utils.shortening_resume import RunJournal
    original_checkpoint_allowed = RunJournal.checkpoint_allowed

    def checkpoint_hook(journal: RunJournal, records: list[dict[str, Any]]) -> bool:
        allowed = original_checkpoint_allowed(journal, records)
        if not allowed:
            assert {
                name: (output / name).read_bytes() for name in ("subtitles.json", "subtitles.srt")
            } == before_resume_exports
            assert (output / "report.json").read_bytes() == before_resume_report
        return allowed

    monkeypatch.setattr(RunJournal, "checkpoint_allowed", checkpoint_hook)
    resumed = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=config, generator=model, arbiter=model, output_dir=output,
        request_fn=interrupted_request, resume=True,
    ))
    assert interrupted
    assert resumed["records"] == uninterrupted["records"]
    assert resumed["cost"] == uninterrupted["cost"]
    assert resumed["summary"]["request_counts"] == uninterrupted["summary"]["request_counts"]
    assert len(interrupted_events) == len(uninterrupted_events) == 5
    assert {
        name: (output / name).read_bytes() for name in ("subtitles.json", "subtitles.srt")
    } == uninterrupted_exports


class AcceptanceInterrupt(BaseException):
    pass


def _acceptance_kwargs(output: Path) -> dict[str, Any]:
    items, planner, plans, ids = _run_inputs()
    return dict(items=items, plans=plans, target_ids=ids, planner_config=planner,
                run_config=RunConfig(max_repairs=0), generator=ModelSpec("fixture", "g", "v"),
                arbiter=ModelSpec("fixture", "a", "v"), output_dir=output,
                report_metadata={"input": {"sha256": "raw-input"},
                                 "calibration": {"profile_sha256": "frozen-profile"}})


async def _acceptance_backend(*args: Any) -> DetailedResponse:
    response = _response(args[2], args[3])
    response.calls = [CallRecord(
        "deepseek", "deepseek-flash", "off", "2026-01-01T00:00:00Z", 0.1,
        OpenCodePromptUsage(input_tokens=100, output_tokens=10, total_tokens=110,
                            cache_read_tokens=0), "ok")]
    return response


async def _must_not_send(*args: Any) -> DetailedResponse:
    pytest.fail("resume must not send a request")


@pytest.mark.parametrize("change", ["items", "raw_hash", "profile", "model", "batch", "prompt", "code"])
def test_acceptance_changed_contract_never_appends_or_sends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    import copy
    import utils.shortening_runner as runner
    from utils.shortening_resume import ResumeError
    kwargs = _acceptance_kwargs(tmp_path / "run")
    asyncio.run(run_trial(**kwargs, request_fn=_acceptance_backend))
    before = {p.name: p.read_bytes() for p in kwargs["output_dir"].iterdir() if p.is_file() and p.name != ".run.lock"}
    altered = copy.deepcopy(kwargs)
    if change == "items":
        altered["items"][0]["text"] += " changed"
    elif change == "raw_hash":
        altered["report_metadata"]["input"]["sha256"] = "different-bytes"
    elif change == "profile":
        altered["report_metadata"]["calibration"]["profile_sha256"] = "different-profile"
    elif change == "model":
        altered["generator"] = ModelSpec("fixture", "g", "different")
    elif change == "batch":
        altered["run_config"] = RunConfig(gen_batch_size=1, max_repairs=0)
    elif change == "prompt":
        monkeypatch.setattr(runner, "PROMPT_VERSION", "changed")
    else:
        original_module_hashes = runner._module_hashes
        def tampered_module_hashes(*, primary: bool = False) -> dict[str, str]:
            hashes = original_module_hashes(primary=primary)
            hashes["shortening_costs.py"] = "changed"
            return hashes
        monkeypatch.setattr(runner, "_module_hashes", tampered_module_hashes)
    with pytest.raises(ResumeError):
        asyncio.run(run_trial(**altered, request_fn=_must_not_send, resume=True))
    after = {p.name: p.read_bytes() for p in kwargs["output_dir"].iterdir() if p.is_file() and p.name != ".run.lock"}
    assert after == before


def test_acceptance_saved_usage_without_complete_stays_unknown_and_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import utils.shortening_runner as runner
    kwargs = _acceptance_kwargs(tmp_path / "run")
    original = runner._record_call
    def interrupt_after_call(*args: Any, **kw: Any) -> None:
        original(*args, **kw)
        raise AcceptanceInterrupt
    monkeypatch.setattr(runner, "_record_call", interrupt_after_call)
    with pytest.raises(AcceptanceInterrupt):
        asyncio.run(run_trial(**kwargs, request_fn=_acceptance_backend))
    monkeypatch.setattr(runner, "_record_call", original)
    output = kwargs["output_dir"]
    before = (output / "requests.jsonl").read_bytes()
    old_calls = (output / "calls.jsonl").read_bytes()
    old_id = json.loads(before.splitlines()[0])["request_id"]
    with pytest.raises(UnknownRequestsError) as error:
        asyncio.run(run_trial(**kwargs, request_fn=_must_not_send, resume=True))
    assert error.value.request_ids == [old_id]
    assert (output / "requests.jsonl").read_bytes() == before
    assert (output / "calls.jsonl").read_bytes() == old_calls
    resumed = asyncio.run(run_trial(**kwargs, request_fn=_acceptance_backend,
                                   resume=True, retry_unknown_ids=(old_id,)))
    assert resumed["summary"]["accepted_count"] == 1
    assert resumed["summary"]["request_counts"]["generation"] == 2
    assert resumed["summary"]["request_counts"]["review"] == 1
    assert resumed["cost"]["direct"]["input_tokens"] == 300
    assert resumed["summary"]["accounting_complete"] is False
    assert resumed["summary"]["unknown_request_ids"] == [old_id]
    assert resumed["cost"]["projection_500"]["applicable"] is False
    assert (output / "requests.jsonl").read_bytes().startswith(before)
    assert (output / "calls.jsonl").read_bytes().startswith(old_calls)
    receipt_bytes = (output / "requests.jsonl").read_bytes()
    repeated = asyncio.run(run_trial(**kwargs, request_fn=_must_not_send, resume=True))
    assert repeated["cost"] == resumed["cost"]
    assert repeated["records"] == resumed["records"]
    assert (output / "requests.jsonl").read_bytes() == receipt_bytes


def test_acceptance_second_unknown_requires_new_id(tmp_path: Path) -> None:
    from utils.shortening_resume import ResumeError
    kwargs = _acceptance_kwargs(tmp_path / "run")
    async def cancel(*args: Any) -> DetailedResponse:
        raise asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_trial(**kwargs, request_fn=cancel))
    log = kwargs["output_dir"] / "requests.jsonl"
    first_id = json.loads(log.read_text().splitlines()[0])["request_id"]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_trial(**kwargs, request_fn=cancel, resume=True, retry_unknown_ids=(first_id,)))
    starts = [json.loads(line) for line in log.read_text().splitlines() if json.loads(line)["state"] == "start"]
    second_id = starts[-1]["request_id"]
    assert second_id != first_id
    before = log.read_bytes()
    with pytest.raises(UnknownRequestsError) as error:
        asyncio.run(run_trial(**kwargs, request_fn=_must_not_send, resume=True))
    assert error.value.request_ids == [second_id]
    with pytest.raises(ResumeError):
        asyncio.run(run_trial(**kwargs, request_fn=_must_not_send, resume=True, retry_unknown_ids=(first_id,)))
    assert log.read_bytes() == before


def test_acceptance_process_lock_rejects_resume(tmp_path: Path) -> None:
    from filelock import FileLock
    from utils.shortening_resume import ResumeError
    kwargs = _acceptance_kwargs(tmp_path / "run")
    asyncio.run(run_trial(**kwargs, request_fn=_acceptance_backend))
    with FileLock(str(kwargs["output_dir"] / ".run.lock"), timeout=0):
        with pytest.raises(ResumeError, match="locked"):
            asyncio.run(run_trial(**kwargs, request_fn=_must_not_send, resume=True))
    report = asyncio.run(run_trial(**kwargs, request_fn=_must_not_send, resume=True))
    assert report["summary"]["accepted_count"] == 1


@pytest.mark.parametrize("failed", [False, True])
def test_acceptance_preflight_baseline_never_repeats(tmp_path: Path, failed: bool) -> None:
    kwargs = _acceptance_kwargs(tmp_path / "run")
    kwargs["items"][0]["text"] = "go now"
    kwargs["plans"] = plan_subtitles(kwargs["items"], kwargs["planner_config"])
    calls = 0
    def verifier(plans: list[Any], output: Path) -> dict[str, dict[str, Any]]:
        nonlocal calls
        calls += 1
        if failed:
            raise RuntimeError("local synthesis failed")
        return {p.cue_id: {"original": p.text, "available_ms": p.available_ms, "duration_sec": 0.2} for p in plans}
    kwargs["original_timing_verifier"] = verifier
    first = asyncio.run(run_trial(**kwargs, request_fn=_must_not_send))
    second = asyncio.run(run_trial(**kwargs, request_fn=_must_not_send, resume=True))
    assert calls == 1
    assert second["records"] == first["records"]
    assert second["records"][0]["status"] == ("timing_unverified" if failed else "actual_fits")


def test_acceptance_preflight_crash_preserves_partial_audio(tmp_path: Path) -> None:
    kwargs = _acceptance_kwargs(tmp_path / "run")
    kwargs["items"][0]["text"] = "go now"
    kwargs["plans"] = plan_subtitles(kwargs["items"], kwargs["planner_config"])
    paths: list[Path] = []
    def verifier(plans: list[Any], output: Path) -> dict[str, dict[str, Any]]:
        paths.append(output)
        if len(paths) == 1:
            (output / "preflight_audio").mkdir(parents=True)
            (output / "preflight_audio" / "partial.txt").write_text("preserve")
            raise AcceptanceInterrupt
        assert "preflight_attempts" in output.parts
        return {p.cue_id: {"original": p.text, "available_ms": p.available_ms, "duration_sec": 0.2} for p in plans}
    kwargs["original_timing_verifier"] = verifier
    with pytest.raises(AcceptanceInterrupt):
        asyncio.run(run_trial(**kwargs, request_fn=_must_not_send))
    report = asyncio.run(run_trial(**kwargs, request_fn=_must_not_send, resume=True))
    assert len(paths) == 2
    assert (paths[0] / "preflight_audio" / "partial.txt").read_text() == "preserve"
    assert report["records"][0]["status"] == "actual_fits"


def test_acceptance_zero_targets_preserves_exports_on_resume(tmp_path: Path) -> None:
    kwargs = _acceptance_kwargs(tmp_path / "run")
    kwargs["items"][0]["text"] = "fit"
    kwargs["plans"] = plan_subtitles(kwargs["items"], kwargs["planner_config"])
    kwargs["target_ids"] = []
    asyncio.run(run_trial(**kwargs, request_fn=_must_not_send))
    output = kwargs["output_dir"]
    saved = {name: (output / name).read_bytes() for name in ("subtitles.json", "subtitles.srt")}
    report = asyncio.run(run_trial(**kwargs, request_fn=_must_not_send, resume=True))
    assert report["records"] == []
    assert {name: (output / name).read_bytes() for name in saved} == saved
    assert not (output / "requests.jsonl").exists()


@pytest.mark.parametrize("damage", ["legacy", "missing_baseline", "corrupt_baseline"])
def test_acceptance_untrusted_state_never_sends(tmp_path: Path, damage: str) -> None:
    from utils.shortening_resume import ResumeError
    kwargs = _acceptance_kwargs(tmp_path / "run")
    asyncio.run(run_trial(**kwargs, request_fn=_acceptance_backend))
    output = kwargs["output_dir"]
    if damage == "legacy":
        (output / "run_manifest.json").unlink()
    elif damage == "missing_baseline":
        (output / "resume_baseline.json").unlink()
    else:
        (output / "resume_baseline.json").write_text("{}")
    before = (output / "requests.jsonl").read_bytes()
    with pytest.raises(ResumeError):
        asyncio.run(run_trial(**kwargs, request_fn=_must_not_send, resume=True))
    assert (output / "requests.jsonl").read_bytes() == before
