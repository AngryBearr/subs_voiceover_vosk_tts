"""The subscription is retried before a request fails; a paid fallback covers one request only."""

from __future__ import annotations

import io
import urllib.error
from typing import Any

import pytest

import stt.llm
from utils.opencode_clean import CleanOpenCodeServer, OpenCodeError

NOT_FOUND = '{"_tag":"ModelNotFoundError","providerID":"openai","modelID":"gpt-6.1-sol"}'


class FakeServer:
    """Answers from a script: an exception instance is raised, a string is returned."""

    def __init__(self, script: list[object]) -> None:
        self.script, self.calls = list(script), 0

    def generate(self, *args: object, **kwargs: object) -> str:
        self.calls += 1
        item = self.script.pop(0) if self.script else '{"ok": 1}'
        if isinstance(item, Exception):
            raise item
        return str(item)


@pytest.fixture()
def env(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"sleeps": [], "posted": [], "server": FakeServer([])}
    monkeypatch.setattr(stt.llm.time, "sleep", state["sleeps"].append)
    monkeypatch.setattr(stt.llm, "shared_server", lambda model: state["server"])
    monkeypatch.setattr(stt.llm, "load_api_key", lambda *a, **k: "k")

    def post(self: Any, body: dict[str, Any], router: str = "openrouter") -> dict[str, Any]:
        state["posted"].append(router)
        return {"model": "m", "choices": [{"message": {"content": '{"paid": 1}'}}], "usage": {}}

    monkeypatch.setattr(stt.llm.OpenRouterClient, "_post", post)
    return state


def busy() -> OpenCodeError:
    return OpenCodeError("HTTP 404: " + NOT_FOUND, transient=True)


def test_model_not_found_404_is_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    server = CleanOpenCodeServer.__new__(CleanOpenCodeServer)
    server.api_version, server.base_url, server._auth_header = 2, "http://x", {}

    def refuse(code: int, body: str) -> Any:
        def urlopen(request: Any, timeout: float = 0) -> Any:
            raise urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(body.encode()))  # type: ignore[arg-type]
        return urlopen

    for code, body, transient, quota in [(404, NOT_FOUND, True, False), (404, "Model not found", True, False),
                                         (404, "no such route", False, False), (429, "slow", True, False),
                                         (502, "usage limit reached (5-hour)", False, True)]:
        monkeypatch.setattr("utils.opencode_clean.urllib.request.urlopen", refuse(code, body))
        with pytest.raises(OpenCodeError) as caught:
            server.generate("p", "openai/gpt-6.1-sol")
        assert (caught.value.transient, caught.value.quota) == (transient, quota), (code, body)


def test_busy_subscription_is_retried_on_the_subscription(env: dict[str, Any]) -> None:
    env["server"] = FakeServer([busy(), busy(), '{"ok": 1}', '{"ok": 2}'])
    client = stt.llm.OpenRouterClient("opencode+openrouter:openai/gpt-6.1-sol")
    assert client.chat_json("s", "u") == {"ok": 1}
    assert env["sleeps"] == [5.0, 15.0] and env["posted"] == []
    assert client.chat_json("s", "u") == {"ok": 2} and client.transport == "opencode"


def test_budget_exhausted_without_fallback_fails_unpaid(env: dict[str, Any]) -> None:
    env["server"] = FakeServer([busy()] * 50)
    client = stt.llm.OpenRouterClient("opencode:openai/gpt-6.1-sol")
    with pytest.raises(RuntimeError, match="subscription not answering"):
        client.chat_json("s", "u")
    assert sum(env["sleeps"]) == 600.0 and env["posted"] == []
    assert env["sleeps"][:6] == [5.0, 15.0, 30.0, 60.0, 120.0, 240.0]


def test_fallback_covers_one_request_only(env: dict[str, Any]) -> None:
    env["server"] = FakeServer([busy()] * 50)
    client = stt.llm.OpenRouterClient("opencode+openrouter:openai/gpt-6.1-sol", subscription_retry_s=20)
    assert client.chat_json("s", "u") == {"paid": 1}
    calls = env["server"].calls
    env["server"].script = ['{"ok": 3}']
    assert client.chat_json("s", "u") == {"ok": 3}  # the subscription is tried first again
    assert env["server"].calls == calls + 1 and env["posted"] == ["openrouter"]


def test_quota_error_is_not_retried(env: dict[str, Any]) -> None:
    env["server"] = FakeServer([OpenCodeError("HTTP 502: usage limit reached", quota=True)])
    client = stt.llm.OpenRouterClient("opencode:openai/gpt-6.1-sol")
    with pytest.raises(RuntimeError, match="usage limit"):
        client.chat_json("s", "u")
    assert env["sleeps"] == [] and env["server"].calls == 1


def test_model_list_refresh_is_retried_within_the_budget(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    import utils.opencode_clean as opencode

    refused = [True]

    def shared(model: str) -> FakeServer:
        if refused:
            refused.clear()
            raise opencode.ModelNotOffered("OpenCode does not offer openai/gpt-6.1-sol; available: openai/gpt-6-sol")
        return FakeServer(['{"ok": 1}'])

    monkeypatch.setattr(stt.llm, "shared_server", shared)
    client = stt.llm.OpenRouterClient("opencode:openai/gpt-6.1-sol")
    assert client.chat_json("s", "u") == {"ok": 1}
    assert env["sleeps"] == [5.0] and env["posted"] == []
