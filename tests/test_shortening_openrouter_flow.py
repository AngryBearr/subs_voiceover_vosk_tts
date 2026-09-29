"""Primary OpenRouter shortening flow tests using the real runner and backend."""

from __future__ import annotations

import asyncio
import copy
import json
import tempfile
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from utils import openrouter_transport as transport
from utils.shortening_budget import BudgetPolicy, BudgetStop
from utils.llm_types import CallRecord, DetailedResponse
from utils.opencode_transport import OpenCodePromptUsage
from utils.shortening_plan import PlannerConfig, plan_subtitles
from utils.shortening_runner import ModelSpec, RunConfig, _is_reconciled_interruption, _request, run_trial
from utils.shortening_resume import JournalStep, ResumeError, RunJournal, append_jsonl, canonical_hash
from utils.shorten_file import main
from utils.vosk_calibration import model_metadata, profile_identity


MODELS: dict[str, tuple[str, str, str, str, str]] = {
    "meta/muse-spark-1.3-contributor": ("low", "meta", "Meta", "0.0000001", "0.0000002"),
    "openai/gpt-6-sol": ("high", "openai", "OpenAI", "0.000005", "0.000015"),
}
MUSE = "meta/muse-spark-1.3-contributor"
SOL = "openai/gpt-6-sol"
CANONICAL_MODELS = {
    MUSE: "meta/muse-spark-1.3-contributor-20260902",
    SOL: "openai/gpt-6-sol-20260922",
}
MUSE_COST = 0.106496
SOL_COST = 0.25


def _catalog(model: str, *, input_rate: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    effort, tag, provider, prompt, completion = MODELS[model]
    if model == MUSE:
        context_length = 1_048_576
        model_pricing: dict[str, Any] = {
            "prompt": "0.0000001",
            "completion": "0.0000002",
            "web_search": "0.0025",
            "input_cache_read": "0.000000002",
        }
        endpoint_pricing = dict(model_pricing)
        endpoint_pricing["discount"] = 0
        max_prompt_tokens = None
        max_completion_tokens = 943_718
        supported_efforts = ["max", "xhigh", "high", "medium", "low", "minimal"]
        model_parameters = [
            "include_reasoning", "max_tokens", "reasoning", "reasoning_effort",
            "repetition_penalty", "response_format", "temperature", "tool_choice",
            "tools", "top_k", "top_p",
        ]
        endpoint_parameters = [
            "reasoning", "include_reasoning", "max_tokens", "repetition_penalty",
            "top_k", "temperature", "top_p", "tools", "tool_choice",
            "structured_outputs", "response_format", "reasoning_effort",
        ]
        reasoning = {"mandatory": True, "supported_efforts": supported_efforts, "default_effort": "medium"}
    else:
        context_length = 1_050_000
        model_pricing = {
            "prompt": "0.000002",
            "completion": "0.00001",
            "web_search": "0.01",
            "input_cache_read": "0.0000002",
            "input_cache_write": "0.0000025",
            "overrides": [{
                "min_prompt_tokens": 272_000,
                "prompt": "0.000004",
                "completion": "0.000015",
                "input_cache_read": "0.0000004",
                "input_cache_write": "0.000005",
            }],
        }
        endpoint_pricing = copy.deepcopy(model_pricing)
        endpoint_pricing["discount"] = 0.5
        max_prompt_tokens = 922_000
        max_completion_tokens = 128_000
        supported_efforts = ["max", "xhigh", "high", "medium", "low", "none"]
        model_parameters = [
            "include_reasoning", "max_completion_tokens", "max_tokens", "reasoning",
            "reasoning_effort", "response_format", "seed", "structured_outputs",
            "tool_choice", "tools",
        ]
        endpoint_parameters = [
            "reasoning", "include_reasoning", "seed", "max_tokens", "response_format",
            "structured_outputs", "tools", "tool_choice", "reasoning_effort",
        ]
        reasoning = {"mandatory": False, "default_enabled": True, "supported_efforts": supported_efforts, "default_effort": "medium"}
    if input_rate is not None:
        model_pricing["prompt"] = input_rate
    model_data: dict[str, Any] = {
        "id": model,
        "canonical_slug": CANONICAL_MODELS[model],
        "context_length": context_length,
        "supported_parameters": model_parameters,
        "reasoning": reasoning,
        "pricing": model_pricing,
    }
    endpoint = {
        "tag": tag,
        "provider_name": provider,
        "model_id": model,
        "max_prompt_tokens": max_prompt_tokens,
        "max_completion_tokens": max_completion_tokens,
        "pricing": endpoint_pricing,
        "supported_parameters": endpoint_parameters,
    }
    return {"data": model_data}, {"data": {"id": model, "endpoints": [endpoint]}}


def _snapshot(*, muse_input_rate: str | None = None) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model in MODELS:
        model_data, endpoint_data = _catalog(model, input_rate=muse_input_rate if model == MUSE else None)
        endpoint = endpoint_data["data"]["endpoints"][0]
        metadata = transport._make_catalog_entry(model, model_data["data"], endpoint)
        models[model] = {
            "model": model,
            "canonical_model": metadata["canonical_model"],
            "effort": MODELS[model][0],
            "provider_slug": MODELS[model][1],
            "provider_name": MODELS[model][2],
            "context_length": metadata["context_length"],
            "max_input_tokens": metadata["max_input_tokens"],
            "max_output_tokens": metadata["max_output_tokens"],
            "max_input_rate": metadata["max_input_rate"],
            "max_output_rate": metadata["max_output_rate"],
            "request_fee": "0",
            "request_fee_source": "not_listed_assumed_zero",
            "model_metadata": metadata["model_metadata"],
            "endpoint_metadata": metadata["endpoint_metadata"],
        }
    snapshot: dict[str, Any] = {
        "schema": transport.OPENROUTER_SNAPSHOT_SCHEMA,
        "id": "",
        "captured_at": "2026-09-20T00:00:00Z",
        "models": models,
    }
    snapshot["id"] = transport._snapshot_id(snapshot)
    return snapshot


def _write_fixture(tmp_path: Path, *, srt: bool = False, two_targets: bool = False) -> tuple[Path, Path, Path]:
    model_path = tmp_path / "vosk-model"
    model_path.mkdir()
    (model_path / "model.onnx").write_bytes(b"fixture-model")
    (model_path / "dictionary").write_text("fixture", encoding="utf-8")
    (model_path / "config.json").write_text(json.dumps({"num_speakers": 9}), encoding="utf-8")

    first = "This subtitle definitely needs shortening"
    second = "Another subtitle definitely needs shortening"
    nearby = "Nearby context fits"
    if srt:
        blocks = [
            f"1\n00:00:00,000 --> 00:00:02,000\n{first}",
            f"2\n00:00:02,200 --> 00:00:04,200\n{second if two_targets else nearby}",
            f"3\n00:00:04,400 --> 00:00:06,400\n{nearby}",
        ]
        source = tmp_path / "episode.srt"
        source.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    else:
        rows = [
            {"index": 101, "start": 0, "end": 2000, "text": first, "keep": {"source": "episode"}},
            {"index": 102, "start": 2200, "end": 4200, "text": second if two_targets else nearby, "keep": {"source": "episode"}},
            {"index": 103, "start": 4400, "end": 6400, "text": nearby, "keep": {"source": "episode"}},
        ]
        source = tmp_path / "episode.json"
        source.write_text(json.dumps(rows), encoding="utf-8")

    identity = profile_identity(model_metadata(model_path), 8, 1.5)
    profile = {
        "rate": 1.5,
        "status": "ok",
        "normalizer": "vosk_punctuation_v1_no_ruaccent",
        "input_character_basis": "normalized_tts_text",
        "chosen_model": "char_linear",
        "models": {"char_linear": {"name": "char_linear", "rate": 1.5, "coefficients": [0.0, 0.1]}},
        "char_linear_guard_sec": 0.0,
        "identity": identity,
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    return source, profile_path, model_path


class HTTPFixture:
    """HTTP-only transport fixture; catalog and response payloads stay native."""

    def __init__(self, *, mode: str = "success", current_input_rate: str | None = None,
                 muse_cost: float = MUSE_COST) -> None:
        self.mode = mode
        self.current_input_rate = current_input_rate
        self.muse_cost = muse_cost
        self.get_urls: list[str] = []
        self.post_bodies: list[dict[str, Any]] = []
        self.generation_gets = 0

    async def get(self, *, url: str, **_: Any) -> tuple[int, bytes]:
        self.get_urls.append(url)
        if "/generation?" in url:
            self.generation_gets += 1
            return 500, b'{"error":{"code":"not_found"}}'
        for model in MODELS:
            model_payload, endpoints_payload = _catalog(
                model,
                input_rate=self.current_input_rate if model == MUSE else None,
            )
            if f"/model/{model}" in url:
                return 200, json.dumps(model_payload).encode()
            if f"/models/{model}/endpoints" in url:
                return 200, json.dumps(endpoints_payload).encode()
        raise AssertionError(f"unexpected OpenRouter GET: {url}")

    async def post(self, **kwargs: Any) -> tuple[int, bytes]:
        from utils.shortening_html import note_model_request

        note_model_request()
        body = kwargs["body"]
        self.post_bodies.append(copy.deepcopy(body))
        model = body["model"]
        if self.mode == "unknown_cost":
            cost: Any = None
        elif self.mode in {"identity_error", "billed_malformed"}:
            cost = 0.25
        else:
            cost = self.muse_cost if model == MUSE else SOL_COST
        if model == MUSE:
            payload = {"results": [
                {"cue_id": "cue-0", "text": "Short first cue", "status": "ok"},
                {"cue_id": "cue-1", "text": "Another subtitle definitely needs shortening", "status": "cannot_shorten"},
            ]}
        else:
            payload = {"results": [
                {"cue_id": "cue-0", "verdict": "accept", "issues": [], "advice": ""},
                {"cue_id": "cue-1", "verdict": "impossible", "issues": ["manual review"], "advice": "Keep the original."},
            ]}
        response: dict[str, Any] = {
            "id": f"gen-{len(self.post_bodies)}",
            "model": CANONICAL_MODELS[model],
            "provider_name": MODELS[model][2],
            "is_byok": self.mode == "identity_error",
            "choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "length" if self.mode == "billed_malformed" else "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20, "cost": cost},
        }
        return 200, json.dumps(response).encode()


def _patch_live(monkeypatch: pytest.MonkeyPatch, fixture: HTTPFixture) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter-key")
    monkeypatch.setattr(transport, "_http_get_json", fixture.get)
    monkeypatch.setattr(transport, "_http_post_json", fixture.post)
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(transport.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        "utils.shorten_file.verify_original_timing",
        lambda plans, output_dir, **_: {
            plan.cue_id: {"cue_id": plan.cue_id, "original": plan.text, "available_ms": plan.available_ms, "duration_sec": 0.1}
            for plan in plans
        },
    )


def _primary_args(source: Path, profile: Path, model_path: Path, output: Path, *, budget: str = "10", extra: list[str] | None = None) -> list[str]:
    return [str(source), "--output-dir", str(output), "--profile", str(profile), "--model-path", str(model_path), "--budget-usd", budget, *(extra or [])]


def test_primary_default_cli_uses_native_roles_and_applies_srt_with_original_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    source, profile, model_path = _write_fixture(tmp_path, srt=True, two_targets=True)
    output = tmp_path / "run"
    http = HTTPFixture()
    _patch_live(monkeypatch, http)

    assert main(_primary_args(source, profile, model_path, output)) == 0
    stdout = capsys.readouterr().out
    summary = json.loads(stdout)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    calls = [json.loads(line) for line in (output / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    subtitles = (output / "subtitles.srt").read_text(encoding="utf-8")

    assert summary["run_state"] == "completed"
    assert report["model"]["generator"] == {"provider": "openrouter", "model": MUSE, "variant": "low"}
    assert report["model"]["arbiter"] == {"provider": "openrouter", "model": SOL, "variant": "high"}
    assert report["metadata"]["roles"] == {"generator": report["model"]["generator"], "arbiter": report["model"]["arbiter"]}
    assert report["config"]["run"]["length_tools"] is False
    assert report["config"]["run"]["gen_batch_size"] == 6
    assert report["config"]["run"]["review_batch_size"] == 12
    assert report["config"]["run"]["max_repairs"] == 1
    assert "experiment" not in json.dumps(report["metadata"], sort_keys=True).lower()
    assert "Short first cue" in subtitles
    assert "Another subtitle definitely needs shortening" in subtitles
    assert report["records"][0]["status"] == "accepted"
    assert report["records"][1]["final_text"] == report["records"][1]["original"]
    assert report["records"][1]["status"] == "manual"
    assert report["budget"]["known_spent_usd"] == "0.356496"
    assert report["budget"]["unknown_reserved_usd"] == "0"
    assert report["cost"]["direct"]["known_subtotal_usd"] == pytest.approx(0.356496)
    assert report["cost"]["stages"]["generation"]["known_subtotal_usd"] == pytest.approx(MUSE_COST)
    assert report["cost"]["stages"]["review"]["known_subtotal_usd"] == pytest.approx(SOL_COST)
    assert len(calls) == 2
    assert {row["record"]["generation_id"] for row in calls} == {"gen-1", "gen-2"}
    assert {row["record"]["pricing_snapshot_id"] for row in calls} == {report["budget"]["pricing_snapshot_id"]}
    assert {row["record"]["model"] for row in calls} == {CANONICAL_MODELS[MUSE], CANONICAL_MODELS[SOL]}
    assert {row["record"]["requested_model"] for row in calls} == {MUSE, SOL}
    assert all(row["record"]["provider_name"] in {"Meta", "OpenAI"} and row["record"]["is_byok"] is False for row in calls)
    assert all(body["max_tokens"] == 16_384 and body["stream"] is False for body in http.post_bodies)
    assert all(body["provider"] == {
            "only": [MODELS[body["model"]][1]],
            "require_parameters": True,
            "allow_fallbacks": False,
            "max_price": {
                "prompt": float(Decimal(MODELS[body["model"]][3]) * 1_000_000),
                "completion": float(Decimal(MODELS[body["model"]][4]) * 1_000_000),
            },
        } for body in http.post_bodies)
    assert all(body["reasoning"] == {"effort": MODELS[body["model"]][0]} for body in http.post_bodies)
    assert all("tools" not in body and "temperature" not in body and "models" not in body for body in http.post_bodies)
    assert len(http.get_urls) == 4 and http.generation_gets == 0
    assert (output / "shortening.html").is_file()
    assert "subs-shortening-html:v1" in (output / "shortening.html").read_text(encoding="utf-8")
    assert http.post_bodies[0]["model"] == MUSE and http.post_bodies[1]["model"] == SOL


def test_fixture_catalog_reservations_match_production_catalog_helper() -> None:
    policy = BudgetPolicy.from_snapshot("10", _snapshot())
    assert policy.reservation(ModelSpec("openrouter", MUSE, "low"))["usd"] == "0.106496"
    assert policy.reservation(ModelSpec("openrouter", SOL, "high"))["usd"] == "4.85576"


def test_primary_dry_run_has_no_http_key_tts_server_or_html(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    source, profile, model_path = _write_fixture(tmp_path)
    output = tmp_path / "dry"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(transport, "_http_get_json", lambda **_: pytest.fail("dry run performed GET"))
    monkeypatch.setattr(transport, "_http_post_json", lambda **_: pytest.fail("dry run performed POST"))
    monkeypatch.setattr("utils.shorten_file.LiveBackend", lambda *_a, **_k: pytest.fail("dry run constructed backend"))
    monkeypatch.setattr("utils.shorten_file.verify_original_timing", lambda *_a, **_k: pytest.fail("dry run ran Vosk"))

    assert main([str(source), "--output-dir", str(output), "--profile", str(profile), "--model-path", str(model_path), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["run_state"] == "dry_run"
    assert not (output / "shortening.html").exists()
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "[--model " not in help_text and "--variant" not in help_text and "--length-tools" not in help_text


def test_budget_reservation_blocks_sol_before_post_and_reports_partial_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, profile, model_path = _write_fixture(tmp_path)
    output = tmp_path / "run"
    http = HTTPFixture()
    _patch_live(monkeypatch, http)

    assert main(_primary_args(source, profile, model_path, output, budget="0.20")) == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert len(http.post_bodies) == 1 and http.post_bodies[0]["model"] == MUSE
    assert report["budget"]["next_reservation_usd"] == "4.85576"
    assert report["budget"]["known_spent_usd"] == "0.106496"
    assert report["budget"]["status"] == "blocked_by_budget"
    assert report["budget"]["reason"] == "blocked_by_budget"
    assert report["run_state"] == "incomplete"
    assert report["records"][0]["final_text"] == report["records"][0]["original"]
    assert "blocked_by_budget" in json.dumps(report)
    assert (output / "shortening.html").is_file()


def test_unknown_cost_generation_lookup_keeps_usable_response_and_does_not_regenerate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, profile, model_path = _write_fixture(tmp_path)
    output = tmp_path / "run"
    http = HTTPFixture(mode="unknown_cost")
    _patch_live(monkeypatch, http)

    assert main(_primary_args(source, profile, model_path, output)) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert [body["model"] for body in http.post_bodies] == [MUSE, SOL]
    assert http.generation_gets == 6
    assert float(report["budget"]["unknown_reserved_usd"]) > 0
    assert report["summary"]["accounting_complete"] is False
    assert report["run_state"] == "completed"


def test_reported_cost_over_reservation_is_recorded_and_stops_before_second_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, profile, model_path = _write_fixture(tmp_path)
    output = tmp_path / "run"
    http = HTTPFixture(muse_cost=0.2)
    _patch_live(monkeypatch, http)

    assert main(_primary_args(source, profile, model_path, output)) == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert len(http.post_bodies) == 1 and http.post_bodies[0]["model"] == MUSE
    assert report["budget"]["known_spent_usd"] == "0.2"
    assert report["budget"]["reason"] == "reported_cost_exceeds_reservation"
    assert report["budget"]["status"] == "reported_cost_exceeds_reservation"


@pytest.mark.parametrize("field", ["provider_name", "is_byok"])
def test_billed_native_identity_error_retains_cost_and_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    source, profile, model_path = _write_fixture(tmp_path)
    output = tmp_path / "run"
    http = HTTPFixture(mode="identity_error")
    _patch_live(monkeypatch, http)
    original_post = http.post

    async def mismatched_post(**kwargs: Any) -> tuple[int, bytes]:
        status, raw = await original_post(**kwargs)
        payload = json.loads(raw)
        payload[field] = "Wrong provider" if field == "provider_name" else True
        return status, json.dumps(payload).encode()

    monkeypatch.setattr(transport, "_http_post_json", mismatched_post)
    assert main(_primary_args(source, profile, model_path, output)) == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    calls = [json.loads(line) for line in (output / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(http.post_bodies) == 1
    assert report["budget"]["known_spent_usd"] == "0"
    assert float(report["budget"]["unknown_reserved_usd"]) > 0
    assert calls[0]["record"]["usage"]["cost"] == 0.25
    assert report["budget"]["reason"] == "openrouter_response_error"


def test_billed_malformed_response_retains_cost_and_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, profile, model_path = _write_fixture(tmp_path)
    output = tmp_path / "run"
    http = HTTPFixture(mode="billed_malformed")
    _patch_live(monkeypatch, http)

    assert main(_primary_args(source, profile, model_path, output)) == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    calls = [json.loads(line) for line in (output / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(http.post_bodies) == 1
    assert report["budget"]["known_spent_usd"] == "0.25"
    assert report["budget"]["unknown_reserved_usd"] == "0"
    assert calls[0]["record"]["usage"]["cost"] == 0.25
    assert report["budget"]["reason"] == "reported_cost_exceeds_reservation"


def test_physical_request_limit_blocks_at_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, profile, model_path = _write_fixture(tmp_path)
    output = tmp_path / "run"
    http = HTTPFixture()
    _patch_live(monkeypatch, http)

    assert main(_primary_args(source, profile, model_path, output, extra=["--max-requests", "1"])) == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert len(http.post_bodies) == 1
    assert report["budget"]["physical_calls"] == 1
    assert report["budget"]["reason"] == "blocked_request_limit"
    assert report["budget"]["status"] == "blocked_request_limit"


def test_resume_same_cap_replays_without_post_and_rejects_cap_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, profile, model_path = _write_fixture(tmp_path)
    output = tmp_path / "run"
    first = HTTPFixture()
    _patch_live(monkeypatch, first)
    assert main(_primary_args(source, profile, model_path, output, budget="0.20")) == 1
    first_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    first_snapshot = first_report["budget"]["pricing_snapshot_id"]

    resumed = HTTPFixture()
    _patch_live(monkeypatch, resumed)
    assert main(_primary_args(source, profile, model_path, output, budget="0.20", extra=["--resume"])) == 1
    second_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert resumed.post_bodies == []
    assert second_report["budget"]["known_spent_usd"] == first_report["budget"]["known_spent_usd"]
    assert second_report["budget"]["pricing_snapshot_id"] == first_snapshot

    assert main(_primary_args(source, profile, model_path, output, budget="10", extra=["--resume"])) == 1
    assert resumed.post_bodies == []


def test_completed_resume_replays_zero_posts_with_frozen_cost_and_pricing_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, profile, model_path = _write_fixture(tmp_path)
    output = tmp_path / "run"
    initial = HTTPFixture(muse_cost=0.05)
    _patch_live(monkeypatch, initial)
    assert main(_primary_args(source, profile, model_path, output)) == 0
    first_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    first_cost = first_report["budget"]["known_spent_usd"]
    first_snapshot = first_report["budget"]["pricing_snapshot_id"]

    resumed = HTTPFixture(muse_cost=0.99)
    _patch_live(monkeypatch, resumed)
    assert main(_primary_args(source, profile, model_path, output, extra=["--resume"])) == 0
    second_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert resumed.post_bodies == []
    assert len(resumed.get_urls) == 4
    assert second_report["budget"]["known_spent_usd"] == first_cost
    assert second_report["budget"]["pricing_snapshot_id"] == first_snapshot


def test_resume_pricing_increase_is_rejected_before_post(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, profile, model_path = _write_fixture(tmp_path)
    output = tmp_path / "run"
    initial = HTTPFixture()
    _patch_live(monkeypatch, initial)
    assert main(_primary_args(source, profile, model_path, output, budget="0.20")) == 1
    changed = HTTPFixture(current_input_rate="0.0000002")
    _patch_live(monkeypatch, changed)
    assert main(_primary_args(source, profile, model_path, output, budget="0.20", extra=["--resume"])) == 1
    assert changed.post_bodies == []


@pytest.mark.parametrize("tamper", ["source", "models", "module_hash", "reservation"])
def test_resume_contract_tampering_blocks_before_post(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str) -> None:
    source, profile, model_path = _write_fixture(tmp_path)
    output = tmp_path / "run"
    initial = HTTPFixture()
    _patch_live(monkeypatch, initial)
    assert main(_primary_args(source, profile, model_path, output, budget="0.20")) == 1
    if tamper == "source":
        source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif tamper == "models":
        manifest_path = output / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["contract"]["models"]["generator"]["variant"] = "medium"
        manifest["contract_hash"] = canonical_hash(manifest["contract"])
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif tamper == "module_hash":
        manifest_path = output / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["contract"]["module_sha256"]["shortening_runner.py"] = "0" * 64
        manifest["contract_hash"] = canonical_hash(manifest["contract"])
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        requests_path = output / "requests.jsonl"
        rows = [json.loads(line) for line in requests_path.read_text(encoding="utf-8").splitlines()]
        rows[0]["reservation"]["usd"] = "999"
        requests_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    resumed = HTTPFixture()
    _patch_live(monkeypatch, resumed)
    assert main(_primary_args(source, profile, model_path, output, budget="0.20", extra=["--resume"])) == 1
    assert resumed.post_bodies == []


def test_legacy_resume_record_without_reservation_remains_allowed() -> None:
    descriptor = {
        "stage": "generation", "model": {"provider": "fixture", "model": "m", "variant": "v"},
        "target_ids": ["cue-1"], "system_prompt": "system", "user_prompt": "user",
        "output_schema": {"type": "object"}, "allow_tool": False,
    }
    # Legacy journals are intentionally provider-agnostic and predate reservations.
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        start = {"request_id": "legacy", **descriptor, "state": "start", "started_at": "2026-01-01T00:00:00Z", "elapsed_sec": 0.0}
        complete = {"request_id": "legacy", **descriptor, "state": "complete", "text": "answer", "error": None,
                    "started_at": start["started_at"], "elapsed_sec": 0.1, "call_count": 0, "failure": None}
        append_jsonl(root / "requests.jsonl", start)
        append_jsonl(root / "requests.jsonl", complete)
        assert RunJournal(root, resume=True).next_request(descriptor).replayed


@pytest.mark.parametrize("reservation", [None, {"usd": "0"}])
def test_primary_missing_or_invalid_reservation_fails_closed_before_physical_call(
    tmp_path: Path, reservation: dict[str, Any] | None,
) -> None:
    snapshot = _snapshot()
    policy = BudgetPolicy.from_snapshot("10", snapshot)
    descriptor = {"stage": "generation", "model": {"provider": "openrouter", "model": MUSE, "variant": "low"},
                  "target_ids": ["cue-1"], "system_prompt": "system", "user_prompt": "user",
                  "output_schema": {"type": "object"}, "allow_tool": False}
    root = tmp_path / "journal"
    start = {"request_id": "primary", **descriptor, "state": "start", "started_at": "2026-01-01T00:00:00Z", "elapsed_sec": 0.0}
    if reservation is not None:
        start["reservation"] = reservation
    complete = {"request_id": "primary", **descriptor, "state": "complete", "text": "answer", "error": None,
                "started_at": start["started_at"], "elapsed_sec": 0.1, "call_count": 0, "failure": None}
    append_jsonl(root / "requests.jsonl", start)
    append_jsonl(root / "requests.jsonl", complete)
    if reservation is None:
        journal = RunJournal(root, resume=True)
        with pytest.raises(ResumeError):
            journal.validate_budget_reservations(policy)
    else:
        with pytest.raises(ResumeError):
            RunJournal(root, resume=True)


def test_openrouter_zero_reported_cost_is_valid_and_does_not_lookup_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = _snapshot()
    posts = 0
    gets = 0

    async def post(**_: Any) -> tuple[int, bytes]:
        nonlocal posts
        posts += 1
        return 200, json.dumps({
            "id": "zero", "model": SOL, "provider_name": "OpenAI", "is_byok": False,
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5, "cost": 0},
        }).encode()

    async def get(**_: Any) -> tuple[int, bytes]:
        nonlocal gets
        gets += 1
        raise AssertionError("zero cost must not request generation metadata")

    monkeypatch.setattr(transport, "_http_post_json", post)
    monkeypatch.setattr(transport, "_http_get_json", get)
    result = asyncio.run(transport.request_json_detailed(
        api_key="dummy", model=SOL, reasoning_effort="high", system_prompt="s", user_prompt="u",
        schema={"type": "object"}, pricing_snapshot=snapshot,
    ))
    assert result.error is None and result.calls[0].usage is not None
    assert result.calls[0].usage.cost == 0 and posts == 1 and gets == 0


def _reconciled_call(model: ModelSpec, *, cost: float | None = 0.05, **changes: Any) -> CallRecord:
    entry = _snapshot()["models"][model.model]
    usage = OpenCodePromptUsage(input_tokens=12, output_tokens=8, total_tokens=20, cost=cost)
    values: dict[str, Any] = {
        "provider": "openrouter", "model": entry["canonical_model"], "variant": model.variant,
        "started_at": "2026-09-20T00:00:00Z", "elapsed_sec": 0.1, "usage": usage,
        "status": "error", "error": "interrupted_response", "requested_model": model.model,
        "requested_variant": model.variant, "generation_id": "gen-interrupted",
        "provider_name": entry["provider_name"], "is_byok": False,
        "cost_source": "openrouter_generation",
    }
    values.update(changes)
    return CallRecord(**values)


def test_reconciled_interrupted_receipt_resumes_lost_batch_without_new_calls(tmp_path: Path) -> None:
    items = [
        {"index": index, "start": index * 2200, "end": index * 2200 + 2000,
         "text": "alpha beta gamma delta epsilon zeta"}
        for index in range(7)
    ]
    planner = PlannerConfig(cps=5, max_speed=1.0)
    plans = plan_subtitles(items, planner)
    ids = [plan.cue_id for plan in plans]
    assert len(ids) == 7
    generator = ModelSpec("openrouter", MUSE, "low")
    arbiter = ModelSpec("openrouter", SOL, "high")
    policy = BudgetPolicy.from_snapshot("10", _snapshot())
    output = tmp_path / "reconciled-resume"
    first_calls: list[list[str]] = []

    async def interrupted(spec: ModelSpec, _: str, user: str, __: dict[str, Any], ___: bool) -> DetailedResponse:
        target_ids = [row["cue_id"] for row in json.loads(user.split("Input JSON:\n", 1)[1])["targets"]]
        first_calls.append(target_ids)
        assert spec == generator
        return DetailedResponse(None, [_reconciled_call(generator)], "interrupted_response")

    first = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=RunConfig(gen_batch_size=6, review_batch_size=12, max_repairs=0),
        generator=generator, arbiter=arbiter, output_dir=output, request_fn=interrupted,
        budget_policy=policy,
    ))
    assert first["run_state"] == "incomplete"
    assert first_calls == [ids[:6]]
    assert first["budget"]["known_spent_usd"] == "0.05"

    resumed_calls: list[list[str]] = []

    async def continue_pending(spec: ModelSpec, _: str, user: str, schema: dict[str, Any], ___: bool) -> DetailedResponse:
        target_ids = [row["cue_id"] for row in json.loads(user.split("Input JSON:\n", 1)[1])["targets"]]
        resumed_calls.append(target_ids)
        assert target_ids == [ids[6]]
        if "verdict" in json.dumps(schema):
            body = {"results": [{"cue_id": ids[6], "verdict": "accept", "issues": [], "advice": ""}]}
            call = replace(_reconciled_call(arbiter, cost=0.05), error=None, status="success", cost_source="openrouter_usage")
        else:
            body = {"results": [{"cue_id": ids[6], "text": "short", "status": "ok"}]}
            call = replace(_reconciled_call(generator, cost=0.05), error=None, status="success", cost_source="openrouter_usage")
        return DetailedResponse(json.dumps(body), [call])

    replayed = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=RunConfig(gen_batch_size=6, review_batch_size=12, max_repairs=0),
        generator=generator, arbiter=arbiter, output_dir=output, request_fn=continue_pending,
        resume=True, budget_policy=policy,
    ))
    assert resumed_calls == [[ids[6]], [ids[6]]]
    assert replayed["run_state"] == "incomplete"
    assert replayed["resume"]["unknown_requests"] == []
    assert replayed["budget"]["known_spent_usd"] == "0.15"
    assert replayed["resume"]["replayed_request_count"] == 1
    assert all(record["final_text"] == record["original"] for record in replayed["records"][:6])
    assert all(record["status"] == "error" for record in replayed["records"][:6])
    assert all([event["stage"] for event in record["history"]] == ["generation"] for record in replayed["records"][:6])
    assert replayed["records"][6]["status"] == "accepted"

    requests = [json.loads(line) for line in (output / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    calls = [json.loads(line) for line in (output / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(requests) == 6 and len(calls) == 3
    assert sum(row["record"]["usage"]["cost"] for row in calls) == pytest.approx(0.15)

    before = len(resumed_calls)
    completed_known = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=ids, planner_config=planner,
        run_config=RunConfig(gen_batch_size=6, review_batch_size=12, max_repairs=0),
        generator=generator, arbiter=arbiter, output_dir=output, request_fn=continue_pending,
        resume=True, budget_policy=policy,
    ))
    assert len(resumed_calls) == before
    assert completed_known["resume"]["unknown_requests"] == []


@pytest.mark.parametrize("field", [
    "provider", "requested_model", "requested_variant", "variant", "model", "provider_name",
    "is_byok", "generation_id", "cost_source", "status", "error",
])
def test_reconciled_interruption_requires_every_receipt_proof_field(field: str) -> None:
    model = ModelSpec("openrouter", MUSE, "low")
    policy = BudgetPolicy.from_snapshot("10", _snapshot())
    valid = _reconciled_call(model)
    invalid_values: dict[str, Any] = {
        "provider": "foreign", "requested_model": "other/model", "requested_variant": "medium",
        "variant": "medium", "model": "other/model", "provider_name": "Wrong",
        "is_byok": True, "generation_id": None, "cost_source": "openrouter_usage",
        "status": "success", "error": "other_error",
    }
    invalid = replace(valid, **{field: invalid_values[field]})
    response = DetailedResponse(None, [invalid], "interrupted_response")
    assert not _is_reconciled_interruption(response, model, policy, [invalid], [])


@pytest.mark.parametrize("elsewhere", [
    _reconciled_call(ModelSpec("openrouter", MUSE, "low"), cost=None),
    _reconciled_call(ModelSpec("openrouter", MUSE, "low"), cost=0.25, provider="foreign"),
])
def test_reconciled_interruption_is_blocked_by_unknown_or_foreign_elsewhere(elsewhere: CallRecord) -> None:
    model = ModelSpec("openrouter", MUSE, "low")
    policy = BudgetPolicy.from_snapshot("10", _snapshot())
    valid = _reconciled_call(model)
    response = DetailedResponse(None, [valid], "interrupted_response")
    summary = policy.summary([valid, elsewhere], [])
    assert summary["status"] == "blocked_unknown_cost"
    assert not _is_reconciled_interruption(response, model, policy, [valid, elsewhere], [])


def test_replayed_noninterruption_error_still_stops_before_backend() -> None:
    model = ModelSpec("openrouter", MUSE, "low")
    policy = BudgetPolicy.from_snapshot("10", _snapshot())
    call = _reconciled_call(model, error="known_error")
    response = DetailedResponse(None, [call], "known_error")

    class ReplayedJournal:
        def next_request(self, _: dict[str, Any]) -> JournalStep:
            return JournalStep("historical", response, True)

    async def forbidden(*_: Any) -> DetailedResponse:
        raise AssertionError("replayed errors must not reach the backend")

    with pytest.raises(BudgetStop) as stopped:
        asyncio.run(_request(
            forbidden, model, "system", "user", {"type": "object"}, False,
            stage="generation", ids=["cue-1"], output_dir=Path("."),
            stage_calls={"generation": [call]}, request_counts={}, request_failures=[],
            journal=ReplayedJournal(), budget_policy=policy,
        ))
    assert stopped.value.reason == "openrouter_response_error"
