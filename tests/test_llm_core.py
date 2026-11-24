"""The shared LLM core: keys from the environment and .env, JSON answers, router ids, cost log, subscription."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

import stt.llm
import voiceover.llm
from utils import llm_core, llm_route


def test_env_value_prefers_the_environment_then_the_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("# a comment\nexport CORE_KEY = 'from-file'\nOTHER=1\n", encoding="utf-8")
    monkeypatch.delenv("CORE_KEY", raising=False)
    assert llm_core.env_value("CORE_KEY", env) == "from-file"
    assert llm_core.env_value("MISSING_KEY", env) == ""
    monkeypatch.setenv("CORE_KEY", "  from-env  ")
    assert llm_core.env_value("CORE_KEY", env) == "from-env"


def test_both_clients_read_keys_the_same_way(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text('CORE_KEY="secret"\n', encoding="utf-8")
    monkeypatch.delenv("CORE_KEY", raising=False)
    assert stt.llm.load_api_key(env, "CORE_KEY") == voiceover.llm.load_api_key("CORE_KEY", env) == "secret"
    with pytest.raises(RuntimeError, match="NOPE_KEY"):
        stt.llm.load_api_key(env, "NOPE_KEY")
    with pytest.raises(voiceover.llm.LLMError, match="NOPE_KEY"):
        voiceover.llm.load_api_key("NOPE_KEY", env)
    assert voiceover.llm.parse_env_file is llm_core.parse_env_file


def test_json_answers() -> None:
    assert llm_core.extract_json('Here you go:\n```json\n{"a": [1]}\n```') == {"a": [1]}
    assert llm_core.extract_json('noise [1, 2]') == [1, 2]
    with pytest.raises(json.JSONDecodeError):
        llm_core.extract_json("no json at all")
    assert llm_core.strip_code_fence("```json\n{}\n```") == "{}"
    assert llm_core.parse_json_object('```\n{"x": 1}\n```') == {"x": 1}
    with pytest.raises(ValueError, match="not an object"):
        llm_core.parse_json_object("[1]")
    assert stt.llm._extract_json is llm_core.extract_json  # the old names stay


def test_routed_model_refuses_an_unknown_go_model() -> None:
    assert llm_core.routed_model("opencode-go/glm-5.3") == "z-ai/glm-5.3"
    assert llm_core.routed_model("openai/gpt-6-luna") == "openai/gpt-6-luna"
    with pytest.raises(RuntimeError, match="no OpenRouter id for opencode-go/nope"):
        llm_core.routed_model("opencode-go/nope")
    with pytest.raises(voiceover.llm.LLMError):
        llm_core.routed_model("opencode-go/nope", voiceover.llm.LLMError)


def test_post_chat_completions_sends_one_authorized_request(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class _Response(io.BytesIO):
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    def urlopen(request: Any, timeout: float = 0) -> _Response:
        seen.update(url=request.full_url, auth=request.get_header("Authorization"), timeout=timeout,
                    body=json.loads(request.data))
        return _Response(b'{"choices": []}')

    monkeypatch.setattr(llm_core.urllib.request, "urlopen", urlopen)
    answer = llm_core.post_chat_completions("https://router.test/v1", "k", {"model": "m"}, 12.0)
    assert answer == {"choices": []}
    assert seen == {"url": "https://router.test/v1/chat/completions", "auth": "Bearer k", "timeout": 12.0,
                    "body": {"model": "m"}}


def test_record_answer_cost_writes_the_cost_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "cost.jsonl"
    monkeypatch.setenv(llm_route.COST_LOG_ENV, str(log_path))
    assert llm_core.response_cost({"usage": {"cost": 0.5}}) == 0.5 and llm_core.response_cost({}) == 0.0
    llm_core.record_answer_cost("m", "openrouter", {"usage": {"cost": 0.25}})
    llm_core.record_answer_cost("m", "opencode", {"usage": {}})  # the subscription is free: no line
    rows = llm_route.paid_rows(log_path)
    assert [(r["model"], r["transport"], r["cost_usd"]) for r in rows] == [("m", "openrouter", 0.25)]


def test_subscription_chat_marks_the_stand_in_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_core, "generate_via_subscription",
                        lambda server, prompt, model, variant, timeout, retry: (prompt.upper(), "openai/gpt-6-luna"))
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    response = llm_core.subscription_chat(lambda m: None, messages, "openai/gpt-6.1-sol", None, 10.0, 20.0)
    assert response["model"] == "opencode:openai/gpt-6-luna" and llm_core.answered_by_fallback(response)
    assert response["choices"][0]["message"]["content"] == "S\n\nU" and response["usage"] == {}
    same = llm_core.subscription_chat(lambda m: None, messages, "openai/gpt-6-luna", None, 10.0, 20.0)
    assert not llm_core.answered_by_fallback(same)
