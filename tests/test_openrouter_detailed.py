from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from dataclasses import asdict
from typing import Any

import pytest

from utils import openrouter_transport as transport
from utils.llm_types import CallRecord, call_record_from_dict


MODELS = {
    "meta/muse-spark-1.3-contributor": ("low", "meta", "Meta", "0.0000001", "0.0000002"),
    "openai/gpt-6-sol": ("high", "openai", "OpenAI", "0.000005", "0.000015"),
}


def _catalog(model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    effort, tag, provider, prompt, completion = MODELS[model]
    metadata = {
        "id": model,
        "canonical_slug": model,
        "context_length": 200_000,
        "supported_parameters": ["max_tokens", "reasoning", "response_format", "structured_outputs"],
        "reasoning": {"efforts": [effort]},
        "pricing": {
            "prompt": prompt,
            "completion": completion,
            "overrides": [{
                "min_prompt_tokens": 272000,
                "prompt": prompt,
                "completion": completion,
                "input_cache_read": prompt,
                "input_cache_write": prompt,
            }],
        },
    }
    endpoint = {
        "tag": tag,
        "provider_name": provider,
        "model_id": model,
        "max_prompt_tokens": 100_000,
        "supported_parameters": ["max_tokens", "reasoning", "response_format", "structured_outputs"],
    }
    return {"data": metadata}, {"data": {"id": model, "endpoints": [endpoint]}}


def test_snapshot_fetches_exact_pins_and_hashes_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    responses = {model: _catalog(model) for model in MODELS}

    async def get(*, url: str, **_: Any) -> tuple[int, bytes]:
        calls.append(url)
        for model, (model_response, endpoints_response) in responses.items():
            if f"/model/{model}" in url:
                return 200, json.dumps(model_response).encode()
            if f"/models/{model}/endpoints" in url:
                return 200, json.dumps(endpoints_response).encode()
        raise AssertionError(url)

    monkeypatch.setattr(transport, "_http_get_json", get)
    snapshot = asyncio.run(transport.fetch_pricing_snapshot())
    assert snapshot["schema"] == "openrouter_production.v1"
    assert set(snapshot["models"]) == set(MODELS)
    assert snapshot["id"] == transport._snapshot_id(snapshot)
    assert len(calls) == 4
    assert all("/endpoints" in url or "/model/" in url for url in calls)


def test_rate_parser_collects_override_ceilings_and_ignores_thresholds() -> None:
    assert transport._rate_strings({
        "prompt": "0.000001", "completion": "0.000002",
        "overrides": [{
            "min_prompt_tokens": 272000,
            "prompt": "0.000004", "completion": ".000015",
            "input_cache_read": "0.000006", "input_cache_write": ".000005",
        }],
    }) == ("0.000006", ".000015", "0")


def test_detailed_payload_is_native_json_pinned_and_reports_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {model: _catalog(model) for model in MODELS}

    async def get(*, url: str, **_: Any) -> tuple[int, bytes]:
        for model, (model_response, endpoints_response) in responses.items():
            if f"/model/{model}" in url:
                return 200, json.dumps(model_response).encode()
            if f"/models/{model}/endpoints" in url:
                return 200, json.dumps(endpoints_response).encode()
        raise AssertionError(url)

    async def post(**kwargs: Any) -> tuple[int, bytes]:
        body = kwargs["body"]
        assert body["model"] == "openai/gpt-6-sol"
        assert body["reasoning"] == {"effort": "high"}
        assert body["max_tokens"] == 16_384
        assert body["stream"] is False
        assert body["provider"] == {
            "only": ["openai"], "require_parameters": True, "allow_fallbacks": False,
            "max_price": {"prompt": 5.0, "completion": 15.0},
        }
        assert "temperature" not in body and "tools" not in body and "models" not in body
        return 200, json.dumps({
            "id": "gen-1", "model": "openai/gpt-6-sol", "provider_name": "OpenAI", "is_byok": False,
            "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5, "cost": 0},
        }).encode()

    monkeypatch.setattr(transport, "_http_get_json", get)
    snapshot = asyncio.run(transport.fetch_pricing_snapshot())
    monkeypatch.setattr(transport, "_http_post_json", post)
    result = asyncio.run(transport.request_json_detailed(
        api_key="secret", model="openai/gpt-6-sol", reasoning_effort="high",
        system_prompt="s", user_prompt="u", schema={"type": "object"}, pricing_snapshot=snapshot,
    ))
    assert result.text == '{"ok":true}'
    assert result.calls[0].usage is not None and result.calls[0].usage.cost == 0
    assert result.calls[0].generation_id == "gen-1"
    assert result.calls[0].provider_name == "OpenAI"
    assert result.calls[0].cost_source == "openrouter_usage"


def test_missing_cost_uses_bounded_generation_get_without_post_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = {
        "schema": transport.OPENROUTER_SNAPSHOT_SCHEMA, "id": "x", "captured_at": "now", "models": {},
    }
    # Build a valid minimal snapshot without making public calls.
    entries: dict[str, Any] = {}
    for model, (effort, tag, provider, prompt, completion) in MODELS.items():
        entries[model] = {
            "model": model, "canonical_model": model, "effort": effort, "provider_slug": tag,
            "provider_name": provider, "context_length": 200_000, "max_input_tokens": 100_000,
            "max_output_tokens": 16_384, "max_input_rate": prompt, "max_output_rate": completion,
            "request_fee": "0", "request_fee_source": "not_listed_assumed_zero",
            "model_metadata": {}, "endpoint_metadata": {},
        }
    snapshot["models"] = entries
    snapshot["id"] = transport._snapshot_id(snapshot)
    posts = 0
    gets = 0
    get_timeouts: list[float] = []

    async def post(**_: Any) -> tuple[int, bytes]:
        nonlocal posts
        posts += 1
        return 200, json.dumps({
            "id": "gen-2", "model": "meta/muse-spark-1.3-contributor",
            "choices": [{"message": {"content": "{}"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }).encode()

    async def get(*, url: str, **_: Any) -> tuple[int, bytes]:
        nonlocal gets
        gets += 1
        get_timeouts.append(float(_["timeout"]))
        return 200, json.dumps({"data": {"total_cost": 0.25, "provider_name": "Meta", "is_byok": False}}).encode()

    async def sleep(_: float) -> None:
        return None

    monkeypatch.setattr(transport, "_http_post_json", post)
    monkeypatch.setattr(transport, "_http_get_json", get)
    monkeypatch.setattr(transport.asyncio, "sleep", sleep)
    result = asyncio.run(transport.request_json_detailed(
        api_key="secret", model="meta/muse-spark-1.3-contributor", reasoning_effort="low",
        system_prompt="s", user_prompt="u", schema={"type": "object"}, pricing_snapshot=snapshot,
    ))
    assert posts == 1 and gets == 1
    assert get_timeouts == [30.0]
    assert result.text is None and result.calls[0].usage is not None
    assert result.calls[0].usage.cost == 0.25
    assert result.calls[0].status == "malformed"


def _valid_snapshot() -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for model, (effort, tag, provider, prompt, completion) in MODELS.items():
        entries[model] = {
            "model": model, "canonical_model": model, "effort": effort, "provider_slug": tag,
            "provider_name": provider, "context_length": 200_000, "max_input_tokens": 100_000,
            "max_output_tokens": 16_384, "max_input_rate": prompt, "max_output_rate": completion,
            "request_fee": "0", "request_fee_source": "not_listed_assumed_zero",
            "model_metadata": {}, "endpoint_metadata": {},
        }
    snapshot: dict[str, Any] = {
        "schema": transport.OPENROUTER_SNAPSHOT_SCHEMA, "id": "", "captured_at": "now", "models": entries,
    }
    snapshot["id"] = transport._snapshot_id(snapshot)
    return snapshot


def test_reported_zero_cost_does_not_lookup_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    posts = 0
    gets = 0

    async def post(**_: Any) -> tuple[int, bytes]:
        nonlocal posts
        posts += 1
        return 200, json.dumps({
            "id": "gen-zero", "model": "openai/gpt-6-sol", "provider_name": "OpenAI", "is_byok": False,
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5, "cost": 0},
        }).encode()

    async def get(**_: Any) -> tuple[int, bytes]:
        nonlocal gets
        gets += 1
        raise AssertionError("reported zero must not trigger a generation lookup")

    monkeypatch.setattr(transport, "_http_post_json", post)
    monkeypatch.setattr(transport, "_http_get_json", get)
    result = asyncio.run(transport.request_json_detailed(
        api_key="secret", model="openai/gpt-6-sol", reasoning_effort="high",
        system_prompt="s", user_prompt="u", schema={"type": "object"}, pricing_snapshot=_valid_snapshot(),
    ))
    assert result.error is None and result.calls[0].usage is not None
    assert result.calls[0].usage.cost == 0 and posts == 1 and gets == 0


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("model", "other/model", "model_mismatch"),
        ("provider_name", "Other", "provider_identity_mismatch"),
        ("is_byok", True, "provider_identity_mismatch"),
    ],
)
def test_receipt_identity_is_pinned(monkeypatch: pytest.MonkeyPatch, field: str, value: Any, error: str) -> None:
    payload: dict[str, Any] = {
        "id": "gen-identity", "model": "openai/gpt-6-sol", "provider_name": "OpenAI", "is_byok": False,
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5, "cost": 0},
    }
    payload[field] = value

    async def post(**_: Any) -> tuple[int, bytes]:
        return 200, json.dumps(payload).encode()

    monkeypatch.setattr(transport, "_http_post_json", post)
    result = asyncio.run(transport.request_json_detailed(
        api_key="secret", model="openai/gpt-6-sol", reasoning_effort="high",
        system_prompt="s", user_prompt="u", schema={"type": "object"}, pricing_snapshot=_valid_snapshot(),
    ))
    assert result.text is None and result.error == error and len(result.calls) == 1


def test_http_401_keeps_usage_and_has_no_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    posts = 0

    async def post(**_: Any) -> tuple[int, bytes]:
        nonlocal posts
        posts += 1
        return 401, json.dumps({
            "id": "gen-401", "model": "openai/gpt-6-sol", "provider_name": "OpenAI", "is_byok": False,
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "cost": 0.1},
            "error": {"code": "unauthorized", "message": "nope"},
        }).encode()

    monkeypatch.setattr(transport, "_http_post_json", post)
    result = asyncio.run(transport.request_json_detailed(
        api_key="secret", model="openai/gpt-6-sol", reasoning_effort="high",
        system_prompt="s", user_prompt="u", schema={"type": "object"}, pricing_snapshot=_valid_snapshot(),
    ))
    assert result.error == "http_permanent" and posts == 1
    assert result.calls[0].usage is not None and result.calls[0].usage.cost == 0.1


def test_response_bound_fails_closed() -> None:
    class Content:
        async def iter_chunked(self, _: int) -> Any:
            yield b"x" * (transport._MAX_RESPONSE_BYTES + 1)

    with pytest.raises(ValueError, match="response_too_large"):
        asyncio.run(transport._read_bounded_response(SimpleNamespace(content=Content())))


def test_call_record_new_fields_roundtrip_and_legacy_defaults() -> None:
    record = CallRecord("openrouter", "openai/gpt-5.6-sol", "medium", "now", 0.1, None, "error",
                        generation_id="gen", provider_name="OpenAI", is_byok=False, cost_source="openrouter_generation")
    assert call_record_from_dict(asdict(record)) == record
    legacy = asdict(CallRecord("fixture", "model", None, "now", 0.0, None, "ok"))
    assert call_record_from_dict(legacy).generation_id is None
    legacy["is_byok"] = "false"
    with pytest.raises(TypeError):
        call_record_from_dict(legacy)
