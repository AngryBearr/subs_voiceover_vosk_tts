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
