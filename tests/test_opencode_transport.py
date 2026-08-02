import base64
import asyncio

import pytest

from utils import opencode_transport as transport
from utils import shorten_subtitles_opencode as legacy


def test_legacy_transport_exports_are_identical(monkeypatch):
    assert legacy.OpenCodeServer is transport.OpenCodeServer
    assert legacy._parse_model_string("openai/gpt-5.6-luna") == {"providerID": "openai", "modelID": "gpt-5.6-luna"}
    monkeypatch.setenv("OPENCODE_SERVER_USERNAME", "alice")
    monkeypatch.setenv("OPENCODE_SERVER_PASSWORD", "secret")
    assert transport._get_auth_header() == {"Authorization": "Basic " + base64.b64encode(b"alice:secret").decode()}


def test_server_environment_removes_server_credentials(monkeypatch):
    monkeypatch.setenv("OPENCODE_SERVER_PASSWORD", "secret")
    monkeypatch.setenv("OPENCODE_SERVER_USERNAME", "alice")
    env = transport._make_server_env()
    assert "OPENCODE_SERVER_PASSWORD" not in env
    assert "OPENCODE_SERVER_USERNAME" not in env


def test_extract_prompt_text_concatenates_text_parts():
    data = {"info": {"finish": "stop"}, "parts": [{"type": "text", "text": "one"}, {"type": "reasoning", "text": "hidden"}, {"type": "text", "text": " two "}]}
    assert transport._extract_prompt_text(data) == "one two"


@pytest.mark.parametrize(
    "data",
    [
        {"info": {"error": {"name": "ProviderError", "data": {"message": "quota exceeded"}}}, "parts": []},
        {"error": {"type": "ProviderError", "message": "quota exceeded"}, "info": {}, "parts": []},
    ],
)
def test_extract_prompt_text_provider_error_is_safe(data):
    with pytest.raises(RuntimeError, match=r"^OpenCode provider error: ProviderError: quota exceeded$"):
        transport._extract_prompt_text(data)


@pytest.mark.parametrize(
    "data",
    [{"parts": []}, {"info": {}, "parts": {}}, {"info": {}, "parts": ["not a part"]}],
)
def test_extract_prompt_text_rejects_malformed_response(data):
    with pytest.raises(RuntimeError, match="^OpenCode response malformed:"):
        transport._extract_prompt_text(data)


def test_extract_prompt_text_reports_no_text_without_raw_response():
    data = {"info": {"finish": "length", "secret": "do not leak"}, "parts": [{"type": "step"}, {"type": "reasoning"}]}
    with pytest.raises(RuntimeError, match=r"finish='length'.*part_types=\['reasoning', 'step'\]") as exc_info:
        transport._extract_prompt_text(data)
    assert "do not leak" not in str(exc_info.value)


def test_send_prompt_returns_none_and_logs_sanitized_provider_error(monkeypatch, capsys):
    async def fake_post(*args, **kwargs):
        return {"info": {"error": {"name": "ProviderError", "data": {"message": "quota exceeded"}}}, "parts": []}

    monkeypatch.setattr(transport, "_http_post", fake_post)
    result = asyncio.run(transport.send_prompt("http://unused", "session", "system", "user", "model"))
    assert result is None
    stderr = capsys.readouterr().err
    assert "OpenCode provider error: ProviderError: quota exceeded" in stderr
    assert "Authorization" not in stderr


def test_extract_prompt_result_full_usage_metadata():
    data = {
        "info": {
            "tokens": {"input": 10, "output": 20, "reasoning": 3, "total": 30, "cache": {"read": 4, "write": 5}},
            "cost": 0.125,
            "providerID": "provider",
            "modelID": "model",
            "finish": "stop",
        },
        "parts": [{"type": "text", "text": "answer"}],
    }
    result = transport._extract_prompt_result(data)
    assert result.text == "answer"
    assert result.usage == transport.OpenCodePromptUsage(10, 20, 3, 30, 4, 5, 0.125, "provider", "model", "stop")


def test_extract_prompt_result_provider_fields_without_usage_is_absent():
    result = transport._extract_prompt_result({"info": {"providerID": "secret-provider", "modelID": "secret-model", "finish": "stop"}, "parts": [{"type": "text", "text": "ok"}]})
    assert result.usage is None


@pytest.mark.parametrize(
    "info",
    [
        {"tokens": []},
        {"tokens": {"cache": []}},
        {"tokens": {"input": None}},
        {"tokens": {"input": True}},
        {"tokens": {"input": -1}},
        {"tokens": {"cache": {"read": False}}},
        {"tokens": {"cache": {"write": -1}}},
        {"cost": True},
        {"cost": -0.1},
        {"cost": float("nan")},
        {"cost": float("inf")},
    ],
)
def test_extract_prompt_result_rejects_malformed_usage(info):
    data = {"info": info, "parts": [{"type": "text", "text": "answer"}]}
    with pytest.raises(RuntimeError, match="^OpenCode response malformed:"):
        transport._extract_prompt_result(data)


def test_send_prompt_legacy_returns_text_from_result(monkeypatch):
    async def fake_result(*args, **kwargs):
        return transport.OpenCodePromptResult("legacy text", transport.OpenCodePromptUsage(output_tokens=2))

    monkeypatch.setattr(transport, "send_prompt_result", fake_result)
    assert asyncio.run(transport.send_prompt("http://unused", "session", "system", "user", "model")) == "legacy text"


def test_send_prompt_legacy_returns_none_when_result_errors(monkeypatch):
    async def fake_result(*args, **kwargs):
        return None

    monkeypatch.setattr(transport, "send_prompt_result", fake_result)
    assert asyncio.run(transport.send_prompt("http://unused", "session", "system", "user", "model")) is None


def test_send_prompt_result_extracts_mocked_http_response(monkeypatch):
    async def fake_post(*args, **kwargs):
        return {"info": {"tokens": {"input": 1}, "cost": 0.5}, "parts": [{"type": "text", "text": "result"}]}

    monkeypatch.setattr(transport, "_http_post", fake_post)
    result = asyncio.run(transport.send_prompt_result("http://unused", "session", "system", "user", "model"))
    assert result is not None and result.text == "result" and result.usage is not None
    assert result.usage.input_tokens == 1 and result.usage.cost == 0.5


def test_extract_prompt_result_structured_object_allows_empty_parts_and_unicode():
    result = transport._extract_prompt_result({"info": {"structured": {"text": "Привет"}}, "parts": []})
    assert result.text == '{"text":"Привет"}'


def test_extract_prompt_result_structured_list_is_canonical():
    result = transport._extract_prompt_result({"info": {"structured": [{"value": 1}, "x"]}, "parts": []})
    assert result.text == '[{"value":1},"x"]'


def test_extract_prompt_result_structured_preserves_usage():
    result = transport._extract_prompt_result({"info": {"structured": {"ok": True}, "tokens": {"output": 2}, "cost": 0.5}, "parts": []})
    assert result.text == '{"ok":true}'
    assert result.usage is not None and result.usage.output_tokens == 2 and result.usage.cost == 0.5


@pytest.mark.parametrize("structured", [None, "text", 1, True])
def test_extract_prompt_result_rejects_invalid_structured_value(structured):
    with pytest.raises(RuntimeError, match="structured"):
        transport._extract_prompt_result({"info": {"structured": structured}, "parts": []})


def test_structured_provider_error_wins_over_structured_payload():
    data = {
        "info": {"error": {"type": "ProviderError", "message": "quota"}, "structured": {"ok": True}, "tokens": {"output": 2}},
        "parts": [],
    }
    with pytest.raises(RuntimeError, match="ProviderError: quota"):
        transport._extract_prompt_result(data)


def test_send_prompt_result_schema_format_is_deep_copied(monkeypatch):
    seen = {}

    async def fake_post(base_url, path, body, **kwargs):
        seen["body"] = body
        return {"info": {"structured": {"results": []}}, "parts": []}

    monkeypatch.setattr(transport, "_http_post", fake_post)
    schema = {"type": "object", "nested": {"required": ["x"]}}
    result = asyncio.run(transport.send_prompt_result("http://unused", "session", "system", "user", "model", output_schema=schema))
    schema["nested"]["required"].append("mutated")
    assert result is not None and result.text == '{"results":[]}'
    assert seen["body"]["format"] == {"type": "json_schema", "schema": {"type": "object", "nested": {"required": ["x"]}}, "retryCount": 0}


def test_send_prompt_result_without_schema_has_no_format(monkeypatch):
    seen = {}

    async def fake_post(base_url, path, body, **kwargs):
        seen["body"] = body
        return {"info": {}, "parts": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr(transport, "_http_post", fake_post)
    result = asyncio.run(transport.send_prompt_result("http://unused", "session", "system", "user", "model"))
    assert result is not None and "format" not in seen["body"]
