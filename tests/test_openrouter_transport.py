import asyncio
import json

import pytest
import aiohttp

import utils.openrouter_transport as transport


def test_model_normalization():
    assert transport.normalize_openrouter_model_id(" google/gemini:free ") == "google/gemini:free"
    for value in ("", "model", "/model", "provider/", "provider/\tmodel", "provider/\nmodel"):
        with pytest.raises(ValueError):
            transport.normalize_openrouter_model_id(value)


@pytest.mark.parametrize("raw, expected", [
    (b"", "response_json_empty"),
    (b" \t\r\n", "response_json_empty"),
    (b" \r\ndata: {\"x\": 1}", "response_json_sse"),
    (b"\xef\xbb\xbf<not-json>", "response_json_html"),
    (b"\x1f\x8b\x08\x00", "response_json_gzip"),
    (b"\xff{}", "response_json_invalid_utf8"),
    (b"not-json", "response_json_invalid"),
])
def test_json_bytes_diagnostics_are_classified_without_body_leakage(raw, expected):
    with pytest.raises(ValueError, match=f"^{expected}$"):
        transport._decode_json_bytes(raw)


def test_json_bytes_accepts_utf8_bom():
    assert transport._decode_json_bytes(b"\xef\xbb\xbf{\"ok\": true}") == {"ok": True}


def test_bounded_response_concatenates_multiple_chunks():
    class Content:
        async def iter_chunked(self, size):
            assert size == 64 * 1024
            for chunk in (b"first", b"-", b"second"):
                yield chunk

    class Response:
        content = Content()

    assert asyncio.run(transport._read_bounded_response(Response())) == b"first-second"


def test_bounded_response_raises_when_cap_is_exceeded_across_chunks():
    consumed = []

    class Content:
        async def iter_chunked(self, size):
            consumed.append(b"a" * (transport._MAX_RESPONSE_BYTES - 1))
            yield consumed[-1]
            consumed.append(b"bc")
            yield consumed[-1]
            consumed.append(b"should not be read")
            yield consumed[-1]

    class Response:
        content = Content()

    with pytest.raises(ValueError, match="^response_too_large$"):
        asyncio.run(transport._read_bounded_response(Response()))
    assert len(consumed) == 2


def test_bounded_response_handles_empty_chunks_and_eof():
    class Content:
        async def iter_chunked(self, size):
            yield b""
            yield b"body"
            yield b""

    class Response:
        content = Content()

    assert asyncio.run(transport._read_bounded_response(Response())) == b"body"


def test_non_2xx_json_diagnostic_preserves_status_and_hides_body(capsys, monkeypatch):
    secret = b"<html>private response body"

    async def post(**kwargs):
        return 502, secret

    monkeypatch.setattr(transport, "_http_post_json", post)
    assert asyncio.run(transport.request_json_schema(api_key="key", model="p/m", system_prompt="s", user_prompt="u", schema={})) is None
    error = capsys.readouterr().err
    assert "status=502" in error
    assert "response_json_html" in error
    assert "private response body" not in error


@pytest.mark.parametrize("raw, expected", [(b"", "response_json_empty"), (b"not-json", "response_json_invalid")])
def test_non_2xx_json_diagnostic_preserves_http_status(capsys, monkeypatch, raw, expected):
    async def post(**kwargs):
        return 418, raw

    monkeypatch.setattr(transport, "_http_post_json", post)
    assert asyncio.run(transport.request_json_schema(api_key="key", model="p/m", system_prompt="s", user_prompt="u", schema={})) is None
    error = capsys.readouterr().err
    assert f"status=418 type=protocol message={expected}" in error


def test_native_body_schema_is_deepcopied_and_headers(monkeypatch):
    seen = {}

    async def post(**kwargs):
        seen.update(kwargs)
        return 200, json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": " {\"ok\":true} "}}]}).encode()

    monkeypatch.setattr(transport, "_http_post_json", post)
    schema = {"properties": {"x": {"type": "string"}}}
    result = asyncio.run(transport.request_json_schema(api_key="sk-or-v1-secret", model="openai/model", system_prompt="s", user_prompt="u", schema=schema, http_referer="https://example", app_title="App"))
    assert result is not None and result.text == '{"ok":true}'
    body = seen["body"]
    assert body["model"] == "openai/model"
    assert body["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "subtitle_semantic_verification"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["provider"] == {"require_parameters": True} and body["stream"] is False
    assert body["temperature"] == 0.0
    assert seen["headers"]["Authorization"] == "Bearer sk-or-v1-secret"
    assert seen["headers"]["HTTP-Referer"] == "https://example"
    assert seen["headers"]["X-OpenRouter-Title"] == "App"
    body["response_format"]["json_schema"]["schema"]["x"] = 1
    assert "x" not in schema


def test_custom_temperature_is_forwarded_as_float(monkeypatch):
    seen = {}

    async def post(**kwargs):
        seen.update(kwargs)
        return 200, b'{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}]}'

    monkeypatch.setattr(transport, "_http_post_json", post)
    assert asyncio.run(transport.request_json_schema(
        api_key="key", model="p/m", system_prompt="s", user_prompt="u", schema={}, temperature=1
    )) is not None
    assert seen["body"]["temperature"] == 1.0


def test_usage_mapping_and_actual_model(monkeypatch):
    async def post(**kwargs):
        return 200, json.dumps({"model": "actual/model", "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "cost": 0.25, "prompt_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 5}, "completion_tokens_details": {"reasoning_tokens": 6}}}).encode()

    monkeypatch.setattr(transport, "_http_post_json", post)
    result = asyncio.run(transport.request_json_schema(api_key="key", model="p/m", system_prompt="s", user_prompt="u", schema={}))
    assert result is not None and result.usage is not None
    assert result.usage.model_id == "actual/model" and result.usage.provider_id == "openrouter"
    assert result.usage.input_tokens == 1 and result.usage.cache_write_tokens == 5 and result.usage.cost == 0.25


def test_empty_response_model_fails_closed(monkeypatch):
    async def post(**kwargs):
        return 200, json.dumps({"model": "", "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}], "usage": {}}).encode()

    monkeypatch.setattr(transport, "_http_post_json", post)
    assert asyncio.run(transport.request_json_schema(api_key="key", model="p/m", system_prompt="s", user_prompt="u", schema={})) is None


@pytest.mark.parametrize("payload", [
    {"error": {"code": "bad_request", "message": "Bearer sk-or-v1-secret"}},
    {"choices": []},
    {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "```json```"}}], "usage": {"prompt_tokens": True}},
])
def test_fail_closed_responses(monkeypatch, payload):
    async def post(**kwargs):
        return 200, json.dumps(payload).encode()

    monkeypatch.setattr(transport, "_http_post_json", post)
    assert asyncio.run(transport.request_json_schema(api_key="key", model="p/m", system_prompt="s", user_prompt="u", schema={})) is None


@pytest.mark.parametrize("raw, expected", [
    (b"", "response_json_empty"),
    (b"not-json", "response_json_invalid"),
    (json.dumps([]).encode(), "response_object"),
    (json.dumps({}).encode(), "choices"),
    (json.dumps({"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}).encode(), "finish_reason"),
    (json.dumps({"choices": [{"finish_reason": "stop", "error": {}, "message": {"content": "{}"}}]}).encode(), "choice_error"),
    (json.dumps({"choices": [{"finish_reason": "stop", "message": []}]}).encode(), "message"),
    (json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": []}}]}).encode(), "content"),
    (json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}], "usage": []}).encode(), "usage"),
    (json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}], "usage": {"prompt_tokens_details": []}}).encode(), "usage_prompt_details"),
    (json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}], "usage": {"completion_tokens_details": []}}).encode(), "usage_completion_details"),
    (json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}], "usage": {"prompt_tokens": True}}).encode(), "usage_input_tokens"),
])
def test_parser_value_errors_log_safe_code(capsys, monkeypatch, raw, expected):
    async def post(**kwargs):
        return 200, raw

    monkeypatch.setattr(transport, "_http_post_json", post)
    assert asyncio.run(transport.request_json_schema(api_key="key", model="p/m", system_prompt="s", user_prompt="u", schema={})) is None
    error = capsys.readouterr().err
    assert f"status=200 type=protocol message={expected}" in error


def test_arbitrary_value_error_logs_only_protocol_error(capsys, monkeypatch):
    secret = "sk-or-v1-private-secret"

    async def post(**kwargs):
        raise ValueError(f"Authorization: Bearer {secret}")

    monkeypatch.setattr(transport, "_http_post_json", post)
    assert asyncio.run(transport.request_json_schema(api_key="key", model="p/m", system_prompt="s", user_prompt="u", schema={})) is None
    error = capsys.readouterr().err
    assert "status=client type=protocol message=protocol_error" in error
    assert secret not in error
    assert "Authorization" not in error


def test_non_2xx_redacts_key(capsys, monkeypatch):
    async def post(**kwargs):
        return 400, json.dumps({"error": {"code": "invalid", "message": "Authorization: Bearer sk-or-v1-real-secret"}}).encode()

    monkeypatch.setattr(transport, "_http_post_json", post)
    assert asyncio.run(transport.request_json_schema(api_key="sk-or-v1-real-secret", model="p/m", system_prompt="s", user_prompt="u", schema={})) is None
    assert "sk-or-v1-real-secret" not in capsys.readouterr().err


@pytest.mark.parametrize("model", ["openai/gpt/a", "google/gemini:free", " provider/model:free "])
def test_model_variants_are_preserved(model):
    assert "/" in transport.normalize_openrouter_model_id(model)


@pytest.mark.parametrize("model", ["p\u00a0/m", "p/\u2003m", "p/\nmodel"])
def test_any_unicode_model_whitespace_is_rejected(model):
    with pytest.raises(ValueError):
        transport.normalize_openrouter_model_id(model)


@pytest.mark.parametrize("kwargs", [
    {"api_key": ""}, {"api_key": "  "}, {"model": "model"}, {"system_prompt": None},
    {"user_prompt": None}, {"schema": []}, {"timeout": 0}, {"timeout": float("nan")},
    {"temperature": -0.1}, {"temperature": 2.1}, {"temperature": float("nan")}, {"temperature": True},
])
def test_request_validation_precedes_post(monkeypatch, kwargs):
    called = False
    async def post(**args):
        nonlocal called
        called = True
        return 200, b"{}"
    monkeypatch.setattr(transport, "_http_post_json", post)
    values = {"api_key": "key", "model": "p/m", "system_prompt": "s", "user_prompt": "u", "schema": {}, "timeout": 180.0}
    values.update(kwargs)
    assert asyncio.run(transport.request_json_schema(**values)) is None
    assert called is False


@pytest.mark.parametrize("referer,title", [(" ", "App"), ("https://x\n", "App"), ("https://x", "\x00")])
def test_invalid_attribution_precedes_post(monkeypatch, referer, title):
    called = False
    async def post(**args):
        nonlocal called
        called = True
        return 200, b"{}"
    monkeypatch.setattr(transport, "_http_post_json", post)
    assert asyncio.run(transport.request_json_schema(api_key="key", model="p/m", system_prompt="s", user_prompt="u", schema={}, http_referer=referer, app_title=title)) is None
    assert called is False


@pytest.mark.parametrize("payload", [
    {"choices": [{"finish_reason": "stop", "message": {"content": None}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": []}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "   "}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "{}", "refusal": "no"}}]},
])
def test_message_content_fail_closed(monkeypatch, payload):
    async def post(**kwargs):
        return 200, json.dumps(payload).encode()
    monkeypatch.setattr(transport, "_http_post_json", post)
    assert asyncio.run(transport.request_json_schema(api_key="key", model="p/m", system_prompt="s", user_prompt="u", schema={})) is None


@pytest.mark.parametrize("finish", ["length", "error", "content_filter", "tool_calls"])
def test_finish_reasons_fail_closed(monkeypatch, finish):
    async def post(**kwargs):
        return 200, json.dumps({"choices": [{"finish_reason": finish, "message": {"content": "{}"}}]}).encode()
    monkeypatch.setattr(transport, "_http_post_json", post)
    assert asyncio.run(transport.request_json_schema(api_key="key", model="p/m", system_prompt="s", user_prompt="u", schema={})) is None


def test_choice_errors_and_shape_fail_closed(monkeypatch):
    payloads = [
        {"choices": [{"finish_reason": "stop", "error": {"code": "x"}, "message": {"content": "{}"}}]},
        {"choices": [{"finish_reason": "stop", "error": None, "message": {"content": "{}"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}, {"finish_reason": "stop", "message": {"content": "{}"}}]},
        {},
    ]
    for payload in payloads:
        async def post(**kwargs):
            return 200, json.dumps(payload).encode()
        monkeypatch.setattr(transport, "_http_post_json", post)
        assert asyncio.run(transport.request_json_schema(api_key="key", model="p/m", system_prompt="s", user_prompt="u", schema={})) is None


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": True}, {"prompt_tokens": 1.5}, {"prompt_tokens_details": []},
    {"completion_tokens_details": [], "completion_tokens": 1}, {"cost": "bad"},
    {"cost": float("inf")},
])
def test_malformed_usage_fails_closed(monkeypatch, usage):
    async def post(**kwargs):
        payload = {"model": "actual/m", "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}], "usage": usage}
        return 200, json.dumps(payload, allow_nan=False).encode()
    monkeypatch.setattr(transport, "_http_post_json", post)
    assert asyncio.run(transport.request_json_schema(api_key="key", model="p/m", system_prompt="s", user_prompt="u", schema={})) is None


def test_partial_and_missing_usage(monkeypatch):
    payloads = [
        {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}], "usage": {"prompt_tokens": 2}},
        {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]},
    ]
    for payload in payloads:
        async def post(**kwargs):
            return 200, json.dumps(payload).encode()
        monkeypatch.setattr(transport, "_http_post_json", post)
        result = asyncio.run(transport.request_json_schema(api_key="key", model="p/m", system_prompt="s", user_prompt="u", schema={}))
        assert result is not None


def test_oversized_and_timeout_fail_safe(capsys, monkeypatch):
    async def oversized(**kwargs):
        return 200, b"x" * (2 * 1024 * 1024 + 1)
    monkeypatch.setattr(transport, "_http_post_json", oversized)
    assert asyncio.run(transport.request_json_schema(api_key="key", model="p/m", system_prompt="secret prompt", user_prompt="secret user", schema={})) is None
    assert "secret prompt" not in capsys.readouterr().err
    async def timeout(**kwargs):
        raise TimeoutError("request_info secret")
    monkeypatch.setattr(transport, "_http_post_json", timeout)
    assert asyncio.run(transport.request_json_schema(api_key="key", model="p/m", system_prompt="s", user_prompt="u", schema={})) is None


def test_aiohttp_client_error_is_safe(capsys, monkeypatch):
    async def client_error(**kwargs):
        raise aiohttp.ClientError("request_info private")
    monkeypatch.setattr(transport, "_http_post_json", client_error)
    assert asyncio.run(transport.request_json_schema(api_key="key", model="p/m", system_prompt="s", user_prompt="u", schema={})) is None
    assert "request_info" not in capsys.readouterr().err


def test_arbitrary_bearer_is_redacted(capsys, monkeypatch):
    async def post(**kwargs):
        return 400, json.dumps({"error": {"code": "bad", "message": "Bearer arbitrary-secret"}}).encode()
    monkeypatch.setattr(transport, "_http_post_json", post)
    asyncio.run(transport.request_json_schema(api_key="key", model="p/m", system_prompt="s", user_prompt="u", schema={}))
    error = capsys.readouterr().err
    assert "arbitrary-secret" not in error and "Bearer [redacted]" in error
