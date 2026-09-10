from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from utils import deepseek_transport as deepseek
from utils import opencode_transport as opencode


def _deepseek_response(*, content: str | None = "{}", usage: object = None, tool_calls: object = None,
                       reasoning_content: str | None = None) -> bytes:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    body: dict[str, object] = {"model": "deepseek-v4-flash", "choices": [{"message": message, "finish_reason": "stop"}]}
    if usage is not None:
        body["usage"] = usage
    return json.dumps(body).encode()


def test_deepseek_reasoning_validation_happens_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail(**_: object) -> tuple[int, bytes]:
        raise AssertionError("network called")

    monkeypatch.setattr(deepseek, "_http_post_json", fail)
    result = asyncio.run(deepseek.request_json_detailed(api_key="secret", model="deepseek-v4-flash",
                                                        system_prompt="s", user_prompt="u", reasoning_effort="medium"))
    assert result.error == "invalid_reasoning_effort"


@pytest.mark.parametrize("model", ["deepseek-flash", "deepseek-v4-flash"])
def test_deepseek_records_requested_and_returned_model(monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    async def post(**_: object) -> tuple[int, bytes]:
        return 200, _deepseek_response(usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})

    monkeypatch.setattr(deepseek, "_http_post_json", post)
    result = asyncio.run(deepseek.request_json_detailed(
        api_key="secret", model=model, system_prompt="s", user_prompt="u", reasoning_effort="low",
    ))
    assert result.calls[0].requested_model == model
    assert result.calls[0].model == "deepseek-v4-flash"
    assert result.calls[0].variant is None
    assert result.calls[0].requested_variant == "low"
    assert result.calls[0].request_max_tokens == 65536


def test_deepseek_response_is_read_incrementally_and_overcap_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    class Content:
        def __init__(self, chunks: list[bytes]) -> None:
            self.chunks = chunks

        async def iter_chunked(self, _: int) -> AsyncIterator[bytes]:
            for chunk in self.chunks:
                yield chunk

    class Response:
        def __init__(self, chunks: list[bytes]) -> None:
            self.content = Content(chunks)

    monkeypatch.setattr(deepseek, "_DETAILED_MAX_RESPONSE_BYTES", 5)
    complete = asyncio.run(deepseek._read_bounded_response(Response([b"12", b"345"])))
    assert complete == b"12345"
    with pytest.raises(ValueError, match="response_too_large"):
        asyncio.run(deepseek._read_bounded_response(Response([b"12", b"3456"])))

    attempts = 0

    async def overcap(**_: object) -> tuple[int, bytes]:
        nonlocal attempts
        attempts += 1
        raise ValueError("response_too_large")

    monkeypatch.setattr(deepseek, "_http_post_json", overcap)
    result = asyncio.run(deepseek.request_json_detailed(
        api_key="secret", model="deepseek-flash", system_prompt="s", user_prompt="u",
    ))
    assert attempts == 1
    assert len(result.calls) == 1
    assert result.calls[0].usage is None


def test_deepseek_counts_usage_on_malformed_content(monkeypatch: pytest.MonkeyPatch) -> None:
    async def post(**_: object) -> tuple[int, bytes]:
        return 200, _deepseek_response(content="", usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})

    monkeypatch.setattr(deepseek, "_http_post_json", post)
    result = asyncio.run(deepseek.request_json_detailed(api_key="secret", model="deepseek-v4-flash",
                                                        system_prompt="s", user_prompt="u"))
    assert result.text is None
    assert result.calls[0].usage is not None
    assert result.calls[0].usage.input_tokens == 3


@pytest.mark.parametrize("malformed", [
    {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
    {"choices": [{"message": {"content": "", "tool_calls": []}, "finish_reason": "stop"}],
     "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
])
def test_deepseek_malformed_first_turn_keeps_its_usage(monkeypatch: pytest.MonkeyPatch, malformed: dict[str, object]) -> None:
    async def post(**_: object) -> tuple[int, bytes]:
        return 200, json.dumps(malformed).encode()

    monkeypatch.setattr(deepseek, "_http_post_json", post)
    result = asyncio.run(deepseek.request_json_detailed(api_key="secret", model="deepseek-v4-flash",
                                                        system_prompt="s", user_prompt="u"))
    assert len(result.calls) == 1
    assert result.calls[0].status == "malformed"
    assert result.calls[0].usage is not None
    assert result.calls[0].usage.input_tokens == 3


def test_deepseek_tool_call_limit_and_finish_length_are_accounted(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = [{"id": str(index), "type": "function", "function": {"name": "x", "arguments": "{}"}} for index in range(33)]
    responses = [
        _deepseek_response(content=None, tool_calls=calls, usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}),
        json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}).encode(),
    ]

    async def post(**_: object) -> tuple[int, bytes]:
        return 200, responses.pop(0)

    monkeypatch.setattr(deepseek, "_http_post_json", post)
    limited = asyncio.run(deepseek.request_json_detailed(api_key="secret", model="deepseek-v4-flash",
                                                         system_prompt="s", user_prompt="u", allow_length_tool=True))
    assert limited.calls[0].tool_calls == 33
    assert limited.calls[0].usage is not None
    assert limited.calls[0].status == "malformed"

    monkeypatch.setattr(deepseek, "_http_post_json", post)
    length = asyncio.run(deepseek.request_json_detailed(api_key="secret", model="deepseek-v4-flash",
                                                        system_prompt="s", user_prompt="u"))
    assert length.calls[0].error == "finish_reason_length"
    assert length.calls[0].usage is not None


def test_deepseek_replays_thirty_length_tools_and_counts_both_usages(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_calls = [
        {
            "id": f"call-{index}",
            "type": "function",
            "function": {
                "name": "check_length",
                "arguments": json.dumps({"text": f"candidate-{index}", "max_chars": 20}),
            },
        }
        for index in range(30)
    ]
    responses = [
        _deepseek_response(
            content=None,
            tool_calls=tool_calls,
            reasoning_content="private reasoning",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 30,
                "total_tokens": 40,
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        ),
        _deepseek_response(
            usage={
                "prompt_tokens": 50,
                "completion_tokens": 4,
                "total_tokens": 54,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        ),
    ]
    seen: list[dict[str, object]] = []

    async def post(**kwargs: object) -> tuple[int, bytes]:
        seen.append(kwargs["body"])  # type: ignore[arg-type]
        return 200, responses.pop(0)

    monkeypatch.setattr(deepseek, "_http_post_json", post)
    result = asyncio.run(deepseek.request_json_detailed(
        api_key="secret", model="deepseek-v4-flash", system_prompt="s", user_prompt="u",
        reasoning_effort="low", allow_length_tool=True,
    ))

    assert result.text == "{}"
    assert len(seen) == 2
    assert seen[0]["tool_choice"] == "auto"
    assert seen[1]["tool_choice"] == "none"
    messages = seen[1]["messages"]
    assert isinstance(messages, list)
    assert messages[2]["reasoning_content"] == "private reasoning"  # type: ignore[index]
    assert len([message for message in messages if message.get("role") == "tool"]) == 30  # type: ignore[union-attr]
    assert [record.tool_calls for record in result.calls] == [30, 0]
    assert [(record.usage.input_tokens, record.usage.output_tokens, record.usage.reasoning_tokens)
            for record in result.calls if record.usage is not None] == [(10, 30, 7), (50, 4, 2)]


def test_deepseek_tool_call_limit_rejects_thirty_three_before_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_calls = [
        {
            "id": f"call-{index}",
            "type": "function",
            "function": {
                "name": "check_length",
                "arguments": json.dumps({"text": "candidate", "max_chars": 20}),
            },
        }
        for index in range(33)
    ]
    response = _deepseek_response(
        content=None,
        tool_calls=tool_calls,
        usage={"prompt_tokens": 11, "completion_tokens": 33, "total_tokens": 44},
    )
    executions = 0

    def fail_if_executed(*_: object, **__: object) -> dict[str, object]:
        nonlocal executions
        executions += 1
        raise AssertionError("tool executed before tool_call_limit")

    async def post(**_: object) -> tuple[int, bytes]:
        return 200, response

    monkeypatch.setattr(deepseek, "_http_post_json", post)
    monkeypatch.setattr(deepseek, "check_length", fail_if_executed)
    result = asyncio.run(deepseek.request_json_detailed(
        api_key="secret", model="deepseek-v4-flash", system_prompt="s", user_prompt="u",
        allow_length_tool=True,
    ))

    assert result.error == "malformed_response"
    assert executions == 0
    assert len(result.calls) == 1
    assert result.calls[0].tool_calls == 33
    assert result.calls[0].error == "tool_call_limit"
    assert result.calls[0].usage is not None
    assert result.calls[0].usage.input_tokens == 11


@pytest.mark.parametrize("effort", ["off", "low", "high", "max"])
def test_deepseek_reasoning_effort_payload_modes(monkeypatch: pytest.MonkeyPatch, effort: str) -> None:
    seen: list[dict[str, object]] = []

    async def post(**kwargs: object) -> tuple[int, bytes]:
        seen.append(kwargs["body"])  # type: ignore[arg-type]
        return 200, _deepseek_response(usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})

    monkeypatch.setattr(deepseek, "_http_post_json", post)
    result = asyncio.run(deepseek.request_json_detailed(api_key="secret", model="deepseek-v4-flash",
                                                        system_prompt="s", user_prompt="u", reasoning_effort=effort))
    assert result.text == "{}"
    assert seen[0]["thinking"] == {"type": "disabled" if effort == "off" else "enabled"}
    if effort == "off":
        assert "reasoning_effort" not in seen[0]
    else:
        assert seen[0]["reasoning_effort"] == effort


def test_deepseek_length_tool_replays_reasoning_and_limits_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, object]] = []
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "check_length", "arguments": '{"text":"😀","max_chars":1}'}}

    async def post(**kwargs: object) -> tuple[int, bytes]:
        seen.append(kwargs["body"])  # type: ignore[arg-type]
        if len(seen) == 1:
            return 200, _deepseek_response(content=None, tool_calls=[tool_call], usage={"prompt_tokens": 1, "completion_tokens": 2})
        return 200, _deepseek_response(usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6})

    monkeypatch.setattr(deepseek, "_http_post_json", post)
    result = asyncio.run(deepseek.request_json_detailed(api_key="secret", model="deepseek-v4-flash",
                                                        system_prompt="s", user_prompt="u", reasoning_effort="low",
                                                        allow_length_tool=True))
    assert result.text == "{}"
    assert len(result.calls) == 2
    assert seen[0]["tool_choice"] == "auto"
    assert seen[1]["tool_choice"] == "none"
    assert seen[1]["messages"][-1]["role"] == "tool"  # type: ignore[index]


def test_opencode_detailed_deduplicates_post_message(monkeypatch: pytest.MonkeyPatch) -> None:
    post = {"id": "assistant-1", "info": {"id": "assistant-1", "role": "assistant", "modelID": "gpt-5.6-luna", "providerID": "openai", "tokens": {"input": 2, "output": 1}}, "parts": [{"type": "text", "text": "ok"}]}

    async def fake_post(*_: object, **__: object) -> dict[str, object]:
        return post

    async def fake_get(*_: object, **__: object) -> list[dict[str, object]]:
        return [post, {"id": "tool-1", "info": {"id": "tool-1", "role": "assistant", "modelID": "gpt-5.6-luna", "providerID": "openai", "tokens": {"input": 3, "output": 2}}, "parts": [{"type": "tool", "tool": "x"}]}]

    monkeypatch.setattr(opencode, "_http_post", fake_post)
    monkeypatch.setattr(opencode, "_http_get", fake_get)
    result = asyncio.run(opencode.send_prompt_detailed("http://server", "s", "system", "user", "openai/gpt-5.6-luna", "medium"))
    assert result.text == "ok"
    assert len(result.calls) == 2
    assert sum(record.usage.input_tokens for record in result.calls if record.usage is not None) == 5


def test_opencode_session_reuse_scopes_to_new_assistant_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    old_user = {"id": "u0", "info": {"id": "u0", "role": "user"}, "parts": []}
    old = {"id": "a0", "info": {"id": "a0", "role": "assistant", "parentID": "u0", "modelID": "luna", "providerID": "openai", "variant": "medium", "tokens": {"input": 10}}, "parts": [{"type": "text", "text": "old"}]}
    new_user = {"id": "u1", "info": {"id": "u1", "role": "user"}, "parts": []}
    new = {"id": "a1", "info": {"id": "a1", "role": "assistant", "parentID": "u1", "modelID": "luna", "providerID": "openai", "variant": "medium", "tokens": {"input": 2}}, "parts": [{"type": "text", "text": "new"}]}
    history = [old_user, old]
    async def fake_post(*_: object, **__: object) -> dict[str, object]:
        return new
    async def fake_get(*_: object, **__: object) -> list[dict[str, object]]:
        return history
    monkeypatch.setattr(opencode, "_http_post", fake_post)
    monkeypatch.setattr(opencode, "_http_get", fake_get)
    result = asyncio.run(opencode.send_prompt_detailed("http://server", "s", "system", "user", "openai/luna", "medium"))
    assert result.text == "new"
    assert [record.usage.input_tokens for record in result.calls if record.usage] == [2]


@pytest.mark.parametrize("info", [
    {"modelID": "other", "providerID": "openai", "variant": "medium"},
    {"modelID": "luna", "providerID": "openai", "variant": "high"},
])
def test_opencode_returned_model_or_variant_mismatch_is_fail_closed(monkeypatch: pytest.MonkeyPatch, info: dict[str, str]) -> None:
    post = {"id": "a1", "info": {"id": "a1", "role": "assistant", **info}, "parts": [{"type": "text", "text": "ok"}]}
    async def fake_post(*_: object, **__: object) -> dict[str, object]:
        return post
    async def fake_get(*_: object, **__: object) -> list[dict[str, object]]:
        return []
    monkeypatch.setattr(opencode, "_http_post", fake_post)
    monkeypatch.setattr(opencode, "_http_get", fake_get)
    result = asyncio.run(opencode.send_prompt_detailed("http://server", "s", "system", "user", "openai/luna", "medium"))
    assert result.error == "model_mismatch"
    assert result.calls[0].status == "mismatch"


def test_opencode_absent_variant_is_unknown_not_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    post = {"id": "a1", "info": {"id": "a1", "role": "assistant", "modelID": "luna", "providerID": "openai"}, "parts": [{"type": "text", "text": "ok"}]}
    async def fake_post(*_: object, **__: object) -> dict[str, object]:
        return post
    async def fake_get(*_: object, **__: object) -> list[dict[str, object]]:
        return []
    monkeypatch.setattr(opencode, "_http_post", fake_post)
    monkeypatch.setattr(opencode, "_http_get", fake_get)
    result = asyncio.run(opencode.send_prompt_detailed("http://server", "s", "system", "user", "openai/luna", "medium"))
    assert result.calls[0].variant is None
    assert result.calls[0].requested_variant == "medium"


def test_opencode_post_timeout_aborts_before_usage_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    post_attempts = 0
    recovered = {"id": "a1", "info": {"id": "a1", "role": "assistant", "parentID": "u1", "modelID": "luna", "providerID": "openai", "tokens": {"input": 7}}, "parts": [{"type": "text", "text": "partial"}]}
    async def fake_post(*_: object, **kwargs: object) -> dict[str, object]:
        nonlocal post_attempts
        post_attempts += 1
        path = str(kwargs.get("path", ""))
        order.append(path)
        if post_attempts == 1:
            raise TimeoutError("timed out")
        return {}
    async def fake_get(*_: object, **__: object) -> list[dict[str, object]]:
        order.append("get")
        return [] if order.count("get") == 1 else [recovered]
    async def post_with_path(*args: object, **kwargs: object) -> dict[str, object]:
        return await fake_post(*args, path=args[1] if len(args) > 1 else "", **kwargs)
    monkeypatch.setattr(opencode, "_http_post", post_with_path)
    monkeypatch.setattr(opencode, "_http_get", fake_get)
    result = asyncio.run(opencode.send_prompt_detailed("http://server", "s", "system", "user", "openai/luna", "medium"))
    assert order[:3] == ["get", "/session/s/message", "/session/s/abort"]
    assert any(record.usage and record.usage.input_tokens == 7 for record in result.calls)


def test_opencode_detailed_defaults_to_plain_json_and_native_format_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, object]] = []
    post = {"id": "a1", "info": {"id": "a1", "role": "assistant", "modelID": "luna", "providerID": "openai", "variant": "medium"}, "parts": [{"type": "text", "text": "{}"}]}

    async def fake_post(*args: object, **__: object) -> dict[str, object]:
        seen.append(args[2])  # type: ignore[arg-type]
        return post

    async def fake_get(*_: object, **__: object) -> list[dict[str, object]]:
        return []

    monkeypatch.setattr(opencode, "_http_post", fake_post)
    monkeypatch.setattr(opencode, "_http_get", fake_get)
    schema = {"type": "object", "properties": {"result": {"type": "string"}}}
    asyncio.run(opencode.send_prompt_detailed("http://server", "s", "system", "user", "openai/luna", "medium", output_schema=schema))
    assert "format" not in seen[-1]
    asyncio.run(opencode.send_prompt_detailed("http://server", "s", "system", "user", "openai/luna", "medium", output_schema=schema, native_structured_output=True))
    assert seen[-1]["format"] == {"type": "json_schema", "schema": schema, "retryCount": 0}


def test_opencode_structured_output_error_keeps_usage_and_useful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    post = {
        "id": "a1",
        "info": {
            "id": "a1", "role": "assistant", "modelID": "luna", "providerID": "openai", "variant": "medium",
            "tokens": {"input": 3, "output": 2, "total": 5},
            "error": {"name": "StructuredOutputError", "data": {"message": "Model did not produce structured output"}},
        },
        "parts": [{"type": "step-start"}, {"type": "step-finish", "tokens": {"input": 3, "output": 2}}, {"type": "text", "text": "{}"}],
    }

    async def fake_post(*_: object, **__: object) -> dict[str, object]:
        return post

    async def fake_get(*_: object, **__: object) -> list[dict[str, object]]:
        return [post]

    monkeypatch.setattr(opencode, "_http_post", fake_post)
    monkeypatch.setattr(opencode, "_http_get", fake_get)
    result = asyncio.run(opencode.send_prompt_detailed("http://server", "s", "system", "user", "openai/luna", "medium"))
    assert result.text is None
    assert len(result.calls) == 1
    assert result.calls[0].usage is not None
    assert result.calls[0].usage.input_tokens == 3
    assert result.calls[0].error is not None
    assert "StructuredOutputError" in result.calls[0].error
    assert "Model did not produce structured output" in (result.error or "")


def test_opencode_does_not_choose_partial_text_before_terminal_error(monkeypatch: pytest.MonkeyPatch) -> None:
    partial = {"id": "a0", "info": {"id": "a0", "role": "assistant", "modelID": "luna", "providerID": "openai", "variant": "medium"}, "parts": [{"type": "text", "text": "partial"}]}
    terminal = {"id": "a1", "info": {"id": "a1", "role": "assistant", "modelID": "luna", "providerID": "openai", "variant": "medium", "error": {"name": "StructuredOutputError", "data": {"message": "terminal failure"}}}, "parts": [{"type": "step-start"}, {"type": "step-finish"}]}

    async def fake_post(*_: object, **__: object) -> dict[str, object]:
        return terminal

    async def fake_get(*_: object, **__: object) -> list[dict[str, object]]:
        return [partial, terminal]

    monkeypatch.setattr(opencode, "_http_post", fake_post)
    monkeypatch.setattr(opencode, "_http_get", fake_get)
    result = asyncio.run(opencode.send_prompt_detailed("http://server", "s", "system", "user", "openai/luna", "medium"))
    assert result.text is None
    assert "terminal failure" in (result.error or "")


def test_opencode_latest_terminal_wins_when_post_response_is_not_in_fetched_history(monkeypatch: pytest.MonkeyPatch) -> None:
    post_partial = {"id": "a0", "info": {"id": "a0", "role": "assistant", "modelID": "luna", "providerID": "openai", "variant": "medium"}, "parts": [{"type": "text", "text": "partial"}]}
    terminal = {"id": "a1", "info": {"id": "a1", "role": "assistant", "modelID": "luna", "providerID": "openai", "variant": "medium", "error": {"name": "ProviderError", "data": {"message": "terminal failure"}}}, "parts": [{"type": "step-finish"}]}

    async def fake_post(*_: object, **__: object) -> dict[str, object]:
        return post_partial

    get_count = 0

    async def fake_get(*_: object, **__: object) -> list[dict[str, object]]:
        nonlocal get_count
        get_count += 1
        return [] if get_count == 1 else [terminal]

    monkeypatch.setattr(opencode, "_http_post", fake_post)
    monkeypatch.setattr(opencode, "_http_get", fake_get)
    result = asyncio.run(opencode.send_prompt_detailed("http://server", "s", "system", "user", "openai/luna", "medium"))
    assert result.text is None
    assert "terminal failure" in (result.error or "")
