from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.fakes import FakeTransport, chat_response, make_client
from voiceover.llm import (LLMClient, LLMConfig, LLMError, LLMTransientError, load_api_key, parse_env_file,
                           strip_code_fence)


def test_second_identical_call_hits_the_cache(tmp_path: Path) -> None:
    transport = FakeTransport(lambda body: {"answer": 1})
    client = make_client(tmp_path, transport)
    first = client.complete_json("sys", "user", prompt_version="v1", tag="t")
    second = client.complete_json("sys", "user", prompt_version="v1", tag="t")
    assert transport.calls == 1
    assert (first.cached, second.cached) == (False, True)
    assert second.content == {"answer": 1} and second.model == "fake-model-1"
    assert second.request_sha == first.request_sha
    assert client.stats() == {"calls_made": 1, "cache_hits": 1, "tokens_in": 10, "tokens_out": 5}
    # a new client on the same cache directory makes no call either
    again = make_client(tmp_path, FakeTransport([]))
    assert again.complete_json("sys", "user", prompt_version="v1", tag="t").cached


def test_prompt_version_and_messages_change_the_key(tmp_path: Path) -> None:
    transport = FakeTransport(lambda body: {"a": 1})
    client = make_client(tmp_path, transport)
    client.complete_json("sys", "user", prompt_version="v1", tag="t")
    client.complete_json("sys", "user", prompt_version="v2", tag="t")
    client.complete_json("sys", "other", prompt_version="v1", tag="t")
    assert transport.calls == 3


def test_request_body_shape(tmp_path: Path) -> None:
    transport = FakeTransport(lambda body: {"a": 1})
    make_client(tmp_path, transport).complete_json("sys", "user", prompt_version="v1", tag="t")
    body = transport.bodies[0]
    assert body["model"] == "deepseek-flash" and body["response_format"] == {"type": "json_object"}
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}]
    assert body["temperature"] == 0.0 and body["max_tokens"] == 4096


def test_code_fenced_json_is_parsed(tmp_path: Path) -> None:
    transport = FakeTransport(['```json\n{"a": 2}\n```'])
    assert make_client(tmp_path, transport).complete_json("s", "u", prompt_version="v1", tag="t").content == {"a": 2}
    assert strip_code_fence("```\n{}\n```") == "{}"
    assert strip_code_fence('{"x": 1}') == '{"x": 1}'


def test_invalid_json_retries_once_then_fails(tmp_path: Path) -> None:
    transport = FakeTransport(["not json", "still not json " * 40])
    client = make_client(tmp_path, transport)
    with pytest.raises(LLMError) as error:
        client.complete_json("s", "u", prompt_version="v1", tag="t")
    assert transport.calls == 2
    assert transport.bodies[1]["messages"][-1]["content"] == "Ответ должен быть только корректным JSON без пояснений."
    assert len(transport.bodies[1]["messages"]) == 3
    assert "still not json" in str(error.value) and len(str(error.value)) < 500
    assert not list((tmp_path / "cache").rglob("*.json"))  # failures are never cached


def test_invalid_json_then_valid_is_cached_under_the_original_key(tmp_path: Path) -> None:
    transport = FakeTransport(["oops", '{"ok": true}'])
    client = make_client(tmp_path, transport)
    assert client.complete_json("s", "u", prompt_version="v1", tag="t").content == {"ok": True}
    assert client.complete_json("s", "u", prompt_version="v1", tag="t").cached
    assert transport.calls == 2


def test_transient_errors_are_retried_with_backoff(tmp_path: Path) -> None:
    attempts: list[int] = []
    delays: list[float] = []

    def flaky(body: dict[str, Any]) -> dict[str, Any]:
        attempts.append(1)
        if len(attempts) < 3:
            raise LLMTransientError("HTTP 503")
        return chat_response('{"a": 1}')

    client = LLMClient(LLMConfig(cache_dir=tmp_path / "c"), flaky, sleep=delays.append)
    assert client.complete_json("s", "u", prompt_version="v1", tag="t").content == {"a": 1}
    assert delays == [2.0, 4.0]


def test_transient_errors_give_up_after_max_retries(tmp_path: Path) -> None:
    def down(body: dict[str, Any]) -> dict[str, Any]:
        raise LLMTransientError("timeout")

    delays: list[float] = []
    client = LLMClient(LLMConfig(cache_dir=tmp_path / "c", max_retries=3), down, sleep=delays.append)
    with pytest.raises(LLMError):
        client.complete_json("s", "u", prompt_version="v1", tag="t")
    assert delays == [2.0, 4.0, 8.0]


def test_permanent_errors_are_not_retried(tmp_path: Path) -> None:
    calls: list[int] = []

    def denied(body: dict[str, Any]) -> dict[str, Any]:
        calls.append(1)
        raise LLMError("HTTP 401")

    client = LLMClient(LLMConfig(cache_dir=tmp_path / "c"), denied, sleep=lambda _: None)
    with pytest.raises(LLMError):
        client.complete_json("s", "u", prompt_version="v1", tag="t")
    assert len(calls) == 1


def test_env_file_fallback_for_the_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text('# comment\n\nOTHER=1\nMY_TEST_KEY="secret-value"\nexport QUOTED=\'x\'\n', encoding="utf-8")
    monkeypatch.delenv("MY_TEST_KEY", raising=False)
    assert parse_env_file(env) == {"OTHER": "1", "MY_TEST_KEY": "secret-value", "QUOTED": "x"}
    assert load_api_key("MY_TEST_KEY", env) == "secret-value"
    monkeypatch.setenv("MY_TEST_KEY", "from-environment")
    assert load_api_key("MY_TEST_KEY", env) == "from-environment"


def test_missing_key_error_names_the_variable_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MY_TEST_KEY", raising=False)
    with pytest.raises(LLMError, match="MY_TEST_KEY"):
        load_api_key("MY_TEST_KEY", tmp_path / "missing.env")


def test_default_transport_needs_the_key_only_when_a_call_is_made(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MY_TEST_KEY", raising=False)
    config = LLMConfig(api_key_env="MY_TEST_KEY", cache_dir=tmp_path / "c")
    client = LLMClient(config, env_file=tmp_path / "none.env")
    with pytest.raises(LLMError, match="MY_TEST_KEY"):
        client.complete_json("s", "u", prompt_version="v1", tag="t")


def test_config_from_dict_uses_defaults() -> None:
    config = LLMConfig.from_dict({"model": "m", "cache_dir": "x/y"})
    assert config.model == "m" and config.provider == "deepseek" and config.cache_dir == Path("x/y")
    assert config.max_retries == 3 and config.timeout_sec == 120.0


def test_truncated_answer_fails_without_a_pointless_retry(tmp_path: Path) -> None:
    calls: list[int] = []

    def cut_off(body: dict[str, Any]) -> dict[str, Any]:
        calls.append(1)
        response = chat_response("")
        response["choices"][0]["finish_reason"] = "length"
        return response

    client = make_client(tmp_path, cut_off)
    with pytest.raises(LLMError, match="max_tokens"):
        client.complete_json("s", "u", prompt_version="v1", tag="t")
    assert len(calls) == 1


def test_reasoning_setting_changes_request_body_and_cache_key(tmp_path) -> None:
    import pytest
    from voiceover.llm import LLMClient, LLMConfig
    from tests.fakes import chat_response
    bodies: list[dict] = []

    def transport(body: dict) -> dict:
        bodies.append(body)
        return chat_response('{"ok": 1}')

    for reasoning in ("default", "off", "low"):
        config = LLMConfig.from_dict({"cache_dir": str(tmp_path), "reasoning": reasoning})
        client = LLMClient(config, transport=transport)
        client.complete_json("s", "u", prompt_version="v", tag="t")
    assert "thinking" not in bodies[0]
    assert bodies[1]["thinking"] == {"type": "disabled"} and "reasoning_effort" not in bodies[1]
    assert bodies[2]["thinking"] == {"type": "enabled"} and bodies[2]["reasoning_effort"] == "low"
    assert len(bodies) == 3  # three different cache keys
    with pytest.raises(ValueError):
        LLMConfig.from_dict({"reasoning": "sometimes"})


def test_opencode_model_goes_through_the_opencode_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import utils.opencode_clean as opencode

    calls: list[tuple[str, str, str | None]] = []
    answers: list[Any] = [opencode.OpenCodeError("HTTP 429: slow down", transient=True), '{"ok": true}']

    class _Server:
        def generate(self, prompt: str, model: str, variant: str | None = None, timeout: float = 0) -> str:
            calls.append((prompt, model, variant))
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return str(answer)

    monkeypatch.setattr(opencode, "shared_server", lambda model: _Server())
    config = LLMConfig(provider="openrouter", model="opencode:openai/gpt-6-luna", api_key_env="NO_SUCH_KEY",
                       reasoning="high", cache_dir=tmp_path)
    client = LLMClient(config, sleep=lambda _: None)
    result = client.complete_json("sys", "user", prompt_version="v1", tag="t")
    assert result.content == {"ok": True} and result.model == "opencode:openai/gpt-6-luna"
    assert calls == [("sys\n\nuser", "openai/gpt-6-luna", "high")] * 2
    assert client.stats()["calls_made"] == 2


def test_opencode_refusal_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import utils.opencode_clean as opencode

    class _Server:
        def generate(self, *args: Any, **kwargs: Any) -> str:
            raise opencode.OpenCodeError("HTTP 400: unknown model")

    monkeypatch.setattr(opencode, "shared_server", lambda model: _Server())
    client = LLMClient(LLMConfig(model="opencode:openai/nope", cache_dir=tmp_path), sleep=lambda _: None)
    with pytest.raises(LLMError, match="unknown model"):
        client.complete_json("sys", "user", prompt_version="v1", tag="t")
    assert client.stats()["calls_made"] == 1


def test_opencode_model_list_of_v2_servers() -> None:
    from utils.opencode_clean import _find_model_variants

    payload = {"data": [{"id": "gpt-6-luna", "providerID": "openai", "variants": [{"id": "none"}, {"id": "high"}]}]}
    assert _find_model_variants(payload, "openai/gpt-6-luna") == {"none", "high"}
    assert _find_model_variants(payload, "openai/gpt-6.1-sol") is None


def test_route_hands_over_to_the_paid_router(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import utils.opencode_clean as opencode
    import voiceover.llm as vllm

    class _Down:
        def generate(self, *args: Any, **kwargs: Any) -> str:
            raise opencode.OpenCodeError("HTTP 503: busy", transient=True)

    routed: list[tuple[str, str]] = []

    def fake_openai(config: LLMConfig, api_key: str) -> Any:
        def call(body: dict[str, Any]) -> dict[str, Any]:
            routed.append((config.base_url, body["model"]))
            assert "thinking" not in body and body["reasoning_effort"] == "high"
            return {"model": body["model"], "choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
        return call

    monkeypatch.setattr(opencode, "shared_server", lambda model: _Down())
    monkeypatch.setattr(vllm, "openai_transport", fake_openai)
    monkeypatch.setattr(vllm, "load_api_key", lambda name, env_file=None: "key")
    client = LLMClient(LLMConfig(model="opencode+orcarouter:openai/gpt-6.1-sol", reasoning="high",
                                 cache_dir=tmp_path), sleep=lambda _: None)
    assert client.complete_json("sys", "user", prompt_version="v1", tag="t").content == {"a": 1}
    assert routed == [("https://api.orcarouter.ai/v1", "openai/gpt-6.1-sol")]
    assert client.stats()["calls_made"] == vllm.FAILOVER_AFTER  # two transient failures were retried first


def test_route_prefix_parsing() -> None:
    from utils.llm_route import split_route, with_route

    assert split_route("opencode+orcarouter:openai/gpt-6.1-sol") == (["opencode", "orcarouter"], "openai/gpt-6.1-sol")
    assert split_route("meta/llama-4:free") == ([], "meta/llama-4:free")
    assert with_route("anthropic/claude-sonnet-5.5", ["orcarouter"]) == "orcarouter:anthropic/claude-sonnet-5.5"
    assert with_route("openrouter:google/gemini-4-argon", ["orcarouter"]) == "openrouter:google/gemini-4-argon"
