"""Behavioral tests for the fail-closed DeepSeek transport."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

import utils.deepseek_transport as transport


def payload(**usage: Any) -> dict[str, Any]:
    return {"model": "deepseek-v4-flash", "choices": [{"finish_reason": "stop", "message": {"content": '{"results": []}'}}], "usage": usage}


def raw(value: Any, *, model: Any = "deepseek-v4-flash") -> bytes:
    data = payload(prompt_tokens=2, completion_tokens=3)
    data["model"] = model
    data.update(value if isinstance(value, dict) else {})
    return json.dumps(data).encode()


def call(monkeypatch: pytest.MonkeyPatch, body: bytes, status: int = 200, model: str = "deepseek-v4-flash", **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    seen: dict[str, Any] = {}

    async def post(**values: Any) -> tuple[int, bytes]:
        seen.update(values)
        return status, body

    monkeypatch.setattr(transport, "_http_post_json", post)
    result = asyncio.run(transport.request_json_object(api_key="secret", model=model, system_prompt="system", user_prompt="user", **kwargs))
    return result, seen


@pytest.mark.parametrize("model", [None, "", "a b", "a?b", "a\\b", "a\tb", "é"])
def test_model_validation_before_post(monkeypatch: pytest.MonkeyPatch, model: Any) -> None:
    async def fail(**_: Any) -> Any:
        pytest.fail("network seam called")
    monkeypatch.setattr(transport, "_http_post_json", fail)
    result = asyncio.run(transport.request_json_object(api_key="key", model=model, system_prompt="s", user_prompt="u"))
    assert result is None


@pytest.mark.parametrize("base_url", ["", "ftp://host", "https://", "https://user:pass@host", "https://host/path?x=1", "https://host/path#x", "https://host/\tbad"])
def test_base_url_validation_before_post(monkeypatch: pytest.MonkeyPatch, base_url: str) -> None:
    async def fail(**_: Any) -> Any:
        pytest.fail("network seam called")
    monkeypatch.setattr(transport, "_http_post_json", fail)
    assert asyncio.run(transport.request_json_object(api_key="key", model="m", system_prompt="s", user_prompt="u", base_url=base_url)) is None


@pytest.mark.parametrize("kwargs", [{"api_key": ""}, {"system_prompt": ""}, {"user_prompt": ""}, {"timeout": 0}, {"temperature": 3}, {"temperature": True}])
def test_other_validation_before_post(monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any]) -> None:
    async def fail(**_: Any) -> Any:
        pytest.fail("network seam called")
    monkeypatch.setattr(transport, "_http_post_json", fail)
    values: dict[str, Any] = {"api_key": "key", "model": "m", "system_prompt": "s", "user_prompt": "u"}
    values.update(kwargs)
    assert asyncio.run(transport.request_json_object(**values)) is None


def test_exact_request_and_custom_url(monkeypatch: pytest.MonkeyPatch) -> None:
    result, seen = call(monkeypatch, raw({}), base_url="https://example.test/api/", temperature=0.25, timeout=7)
    assert result is not None
    assert seen["url"] == "https://example.test/api/chat/completions"
    assert seen["headers"] == {"Authorization": "Bearer secret", "Content-Type": "application/json"}
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["thinking"] == {"type": "disabled"}
    assert seen["body"]["stream"] is False and seen["body"]["temperature"] == 0.25
    assert "provider" not in seen["body"] and "json_schema" not in json.dumps(seen["body"])


def test_bom_and_model_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    data = json.loads(raw({}))
    data.pop("model")
    result, _ = call(monkeypatch, b"\xef\xbb\xbf" + json.dumps(data).encode())
    assert result is not None and result.usage is not None and result.usage.model_id == "deepseek-v4-flash"


@pytest.mark.parametrize("model", ["", "bad model", 3])
def test_invalid_present_response_model_fails_closed(monkeypatch: pytest.MonkeyPatch, model: Any) -> None:
    result, _ = call(monkeypatch, raw({}, model=model))
    assert result is None


@pytest.mark.parametrize("body", [b"", b"data: {}", b"<html>", b"\x1f\x8bbad", b"\xff", b"not json"])
def test_malformed_bytes_fail_closed(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    result, _ = call(monkeypatch, body)
    assert result is None


@pytest.mark.parametrize("field,value", [("prompt_tokens", True), ("prompt_tokens", 1.2), ("prompt_tokens", -1), ("completion_tokens", None), ("completion_tokens_details", []), ("total_tokens", 99)])
def test_usage_errors_fail_closed(monkeypatch: pytest.MonkeyPatch, field: str, value: Any) -> None:
    usage = {"prompt_tokens": 2, "completion_tokens": 3}
    usage[field] = value
    result, _ = call(monkeypatch, raw({"usage": usage}))
    assert result is None


def test_usage_derivations_and_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = call(monkeypatch, raw({"usage": {"prompt_cache_hit_tokens": 2, "prompt_cache_miss_tokens": 4, "completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 1}}}))
    assert result is not None and result.usage is not None
    assert result.usage.input_tokens == 6 and result.usage.total_tokens == 9 and result.usage.reasoning_tokens == 1
    result, _ = call(monkeypatch, raw({"usage": {"prompt_tokens": 7, "prompt_cache_hit_tokens": 2, "prompt_cache_miss_tokens": 4, "completion_tokens": 3}}))
    assert result is None


@pytest.mark.parametrize("choice", [{}, {"finish_reason": "length"}, {"finish_reason": "stop", "error": "x"}, {"finish_reason": "stop", "message": {"refusal": "no"}}, {"finish_reason": "stop", "message": {"content": ""}}])
def test_choice_validation_fail_closed(monkeypatch: pytest.MonkeyPatch, choice: dict[str, Any]) -> None:
    data = payload(prompt_tokens=1, completion_tokens=1)
    data["choices"] = [choice]
    result, _ = call(monkeypatch, json.dumps(data).encode())
    assert result is None


def test_provider_error_redacts_diagnostics(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    body = json.dumps({"error": {"code": "bad", "message": "api_key=SECRET Bearer TOP sk-abc123"}}).encode()
    result, _ = call(monkeypatch, body, status=401)
    stderr = capsys.readouterr().err
    assert result is None and "SECRET" not in stderr and "TOP" not in stderr and "abc123" not in stderr


def test_bounded_chunks_and_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    class Content:
        def __init__(self, chunks: list[bytes]) -> None:
            self.chunks = chunks
            self.reads = 0
        async def iter_chunked(self, _: int) -> Any:
            for chunk in self.chunks:
                self.reads += 1
                yield chunk
    content = Content([b"a", b"b"])
    response = type("Response", (), {"content": content})()
    assert asyncio.run(transport._read_bounded_response(response)) == b"ab" and content.reads == 2
    huge = Content([b"x" * (transport._MAX_RESPONSE_BYTES + 1), b"never"])
    with pytest.raises(ValueError, match="response_too_large"):
        asyncio.run(transport._read_bounded_response(type("Response", (), {"content": huge})()))
    assert huge.reads == 1


def test_transport_exception_is_safe_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fail(**_: Any) -> tuple[int, bytes]:
        raise RuntimeError("api_key=SECRET Bearer TOP sk-abc123")

    monkeypatch.setattr(transport, "_http_post_json", fail)
    result = asyncio.run(transport.request_json_object(
        api_key="SECRET",
        model="deepseek-v4-flash",
        system_prompt="system",
        user_prompt="user",
    ))
    stderr = capsys.readouterr().err
    assert result is None
    assert "SECRET" not in stderr
    assert "TOP" not in stderr
    assert "abc123" not in stderr
    assert "request failed" in stderr
