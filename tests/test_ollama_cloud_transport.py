import asyncio
import json

import pytest

import utils.ollama_cloud_transport as transport


def _response(**extra):
    value = {"model": "actual", "done": True, "done_reason": "stop",
             "message": {"role": "assistant", "content": " {\"results\":[]} "}}
    value.update(extra)
    return json.dumps(value).encode()


def test_normalization_and_exact_body_headers(monkeypatch):
    seen = {}

    async def post(**kwargs):
        seen.update(kwargs)
        return 200, _response()

    monkeypatch.setattr(transport, "_http_post_json", post)
    result = asyncio.run(transport.request_strict_json(api_key="secret", model=" glm-5.2 ", system_prompt="s", user_prompt="u"))
    assert result is not None and result.text == '{"results":[]}'
    assert seen["body"] == {"model": "glm-5.2", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], "stream": False, "options": {"temperature": 0.0, "seed": 42}}
    assert "format" not in seen["body"] and "tools" not in seen["body"] and "think" not in seen["body"]
    assert seen["headers"] == {"Authorization": "Bearer secret", "Content-Type": "application/json", "Accept": "application/json"}


@pytest.mark.parametrize("value", ["", "a b", "a\u2003b", "a<>b", "a\\b", "a@b"])
def test_model_rejects_invalid_catalog_ids(value):
    with pytest.raises(ValueError):
        transport.normalize_ollama_cloud_model_id(value)


@pytest.mark.parametrize("kwargs", [{"api_key": ""}, {"model": "bad model"}, {"system_prompt": None}, {"timeout": 0}, {"temperature": True}, {"seed": True}, {"seed": -1}])
def test_validation_precedes_post(monkeypatch, kwargs):
    called = False

    async def post(**args):
        nonlocal called
        called = True
        return 200, b"{}"

    monkeypatch.setattr(transport, "_http_post_json", post)
    values = {"api_key": "key", "model": "glm-5.2", "system_prompt": "s", "user_prompt": "u"}
    values.update(kwargs)
    assert asyncio.run(transport.request_strict_json(**values)) is None
    assert not called


@pytest.mark.parametrize("raw, code", [(b"", "response_json_empty"), (b"data: x", "response_json_sse"), (b"<html>", "response_json_html"), (b"\x1f\x8b", "response_json_gzip"), (b"\xff", "response_json_invalid_utf8")])
def test_safe_decode_diagnostics(raw, code):
    with pytest.raises(ValueError, match=f"^{code}$"):
        transport._decode_json_bytes(raw)


def test_response_validation_and_usage(monkeypatch):
    async def post(**kwargs):
        return 200, _response(prompt_eval_count=3, eval_count=4, thinking="secret", images=[])

    monkeypatch.setattr(transport, "_http_post_json", post)
    result = asyncio.run(transport.request_strict_json(api_key="key", model="glm-5.2", system_prompt="s", user_prompt="u"))
    assert result is not None and result.usage is not None
    assert result.usage.provider_id == "ollama-cloud" and result.usage.total_tokens == 7
    assert result.usage.reasoning_tokens is None and result.usage.cost is None


def test_success_allows_null_error(monkeypatch):
    async def post(**kwargs):
        return 200, _response(error=None)

    monkeypatch.setattr(transport, "_http_post_json", post)
    result = asyncio.run(transport.request_strict_json(api_key="key", model="glm-5.2", system_prompt="s", user_prompt="u"))
    assert result is not None and result.text == '{"results":[]}'


@pytest.mark.parametrize(
    "payload, expected_type, expected_message",
    [
        ({"error": "temporarily unavailable"}, "provider_error", "temporarily unavailable"),
        ({"error": {"code": "rate_limited", "message": "try again later"}}, "rate_limited", "try again later"),
    ],
)
def test_non_2xx_provider_error_is_safe_and_preserves_status(capsys, monkeypatch, payload, expected_type, expected_message):
    async def post(**kwargs):
        return 429, json.dumps(payload).encode()

    monkeypatch.setattr(transport, "_http_post_json", post)
    assert asyncio.run(transport.request_strict_json(api_key="key", model="glm-5.2", system_prompt="s", user_prompt="u")) is None
    error = capsys.readouterr().err
    assert f"status=429 type={expected_type} message={expected_message}" in error
    assert "key" not in error


def test_non_2xx_provider_message_redacts_exact_api_key(capsys, monkeypatch):
    async def post(**kwargs):
        return 500, b'{"error":"secret-ollama-key"}'

    monkeypatch.setattr(transport, "_http_post_json", post)
    assert asyncio.run(transport.request_strict_json(api_key="secret-ollama-key", model="glm-5.2", system_prompt="s", user_prompt="u")) is None
    error = capsys.readouterr().err
    assert "secret-ollama-key" not in error
    assert "[redacted-credential]" in error


@pytest.mark.parametrize("raw", [b"not-json", b"<html>private body</html>"])
def test_non_2xx_invalid_body_uses_generic_http_error(capsys, monkeypatch, raw):
    async def post(**kwargs):
        return 503, raw

    monkeypatch.setattr(transport, "_http_post_json", post)
    assert asyncio.run(transport.request_strict_json(api_key="private-key", model="glm-5.2", system_prompt="s", user_prompt="u")) is None
    error = capsys.readouterr().err
    assert "status=503 type=protocol message=http_error" in error
    assert "private body" not in error and "private-key" not in error


@pytest.mark.parametrize("failure", [TimeoutError("private timeout"), RuntimeError("private client body")])
def test_transport_exceptions_use_safe_generic_logging(capsys, monkeypatch, failure):
    async def post(**kwargs):
        raise failure

    monkeypatch.setattr(transport, "_http_post_json", post)
    assert asyncio.run(transport.request_strict_json(api_key="private-key", model="glm-5.2", system_prompt="s", user_prompt="u")) is None
    error = capsys.readouterr().err
    assert "status=client type=client_error message=request failed" in error
    assert "private" not in error


def test_oversized_transport_response_is_bounded_and_safe(capsys, monkeypatch):
    async def post(**kwargs):
        raise ValueError("response_too_large")

    monkeypatch.setattr(transport, "_http_post_json", post)
    assert asyncio.run(transport.request_strict_json(api_key="private-key", model="glm-5.2", system_prompt="s", user_prompt="u")) is None
    error = capsys.readouterr().err
    assert "status=client type=protocol message=response_too_large" in error
    assert "private-key" not in error


@pytest.mark.parametrize("payload", [{"done": False}, {"done": True, "done_reason": "length"}, {"done": True, "done_reason": "stop", "message": {"role": "user", "content": "x"}}, {"done": True, "done_reason": "stop", "message": {"role": "assistant", "content": "x", "tool_calls": [{}]}}])
def test_fail_closed_response_shapes(monkeypatch, payload):
    async def post(**kwargs):
        return 200, _response(**payload)

    monkeypatch.setattr(transport, "_http_post_json", post)
    assert asyncio.run(transport.request_strict_json(api_key="key", model="glm-5.2", system_prompt="s", user_prompt="u")) is None


def test_bounded_chunks_and_redaction(capsys, monkeypatch):
    class Content:
        async def iter_chunked(self, size):
            yield b"a" * (transport._MAX_RESPONSE_BYTES - 1)
            yield b"xx"

    class Response:
        content = Content()

    with pytest.raises(ValueError, match="response_too_large"):
        asyncio.run(transport._read_bounded_response(Response()))

    async def post(**kwargs):
        return 400, b'{"error":{"message":"Bearer private"}}'

    monkeypatch.setattr(transport, "_http_post_json", post)
    asyncio.run(transport.request_strict_json(api_key="key", model="glm-5.2", system_prompt="private prompt", user_prompt="u"))
    error = capsys.readouterr().err
    assert "private" not in error and "Bearer [redacted]" not in error
