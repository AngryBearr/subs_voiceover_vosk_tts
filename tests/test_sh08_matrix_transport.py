from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from utils import sh08_matrix_transport as transport
from utils import sh08_matrix_models as models
from utils.llm_types import CallRecord, DetailedResponse
from utils.opencode_transport import OpenCodePromptUsage


def test_model_specs_are_locked() -> None:
    assert transport.MODEL_SPECS is models.MODEL_SPECS
    assert transport.MAX_TOKENS == models.MAX_TOKENS == 16384
    assert transport.OPENCODE_REQUIRED_VERSION == models.OPENCODE_REQUIRED_VERSION == "1.18.29"
    assert transport.MODEL_SPECS["deepseek"]["variant"] == "off"
    assert transport.MODEL_SPECS["gemini"]["endpoint_tag"] == "google-ai-studio/flex"
    assert transport.MODEL_SPECS["muse"]["reasoning_mandatory"] is True
    assert transport.MODEL_SPECS["luna"]["quota_units"] == 6
    assert transport.MODEL_SPECS["deepseek"]["max_tokens"] == 16384


def test_openrouter_request_is_pinned_and_cost_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, object]] = []

    async def post(*, headers, body):
        seen.append(body)
        return 200, json.dumps({
            "model": "google/gemini-3.8-flash-20260902",
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 2000, "cost": 0.004125},
        }).encode()

    backend = transport.MatrixBackend(openrouter_no_byok=True)
    backend._credentials = {"OPENROUTER_API_KEY": "fake"}
    monkeypatch.setattr(backend, "_post_openrouter", post)
    result = asyncio.run(backend("gemini", "system", "user", {"type": "object"}))
    body = seen[0]
    assert body["max_tokens"] == 16384
    assert body["provider"] == {"require_parameters": True, "allow_fallbacks": False, "only": ["google-ai-studio/flex"], "max_price": {"prompt": 0.375, "completion": 1.875}}
    assert "tools" not in body and "temperature" not in body
    assert result["cost_known"] is True
    assert result["cost_kind"] == "provider_reported"
    assert result["raw"]["usage"]["cost"] == 0.004125


def test_openrouter_unknown_usage_and_byok_are_not_free(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        {"model": "google/gemini-3.8-flash", "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}], "usage": {}},
        {"model": "google/gemini-3.8-flash", "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.1, "is_byok": True}},
    ]

    async def post(*, headers, body):
        return 200, json.dumps(responses.pop(0)).encode()

    backend = transport.MatrixBackend(openrouter_no_byok=True)
    backend._credentials = {"OPENROUTER_API_KEY": "fake"}
    monkeypatch.setattr(backend, "_post_openrouter", post)
    unknown = asyncio.run(backend("gemini", "s", "u", {}))
    byok = asyncio.run(backend("gemini", "s", "u", {}))
    assert unknown["cost_known"] is False and unknown["cost_usd"] is None
    assert byok["cost_known"] is False and byok["error"] == "openrouter_byok_attribution_unknown"


def test_openrouter_http_error_is_numeric_and_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    async def post(*, headers, body):
        return 402, json.dumps({"error": {"message": "Bearer secret-key"}}).encode()

    backend = transport.MatrixBackend(openrouter_no_byok=True)
    backend._credentials = {"OPENROUTER_API_KEY": "secret-key"}
    backend._secrets = ("secret-key",)
    monkeypatch.setattr(backend, "_post_openrouter", post)
    result = asyncio.run(backend("gemini", "s", "u", {}))
    assert result["error"].startswith("http_402:")
    assert "secret-key" not in json.dumps(result)


def test_deepseek_uses_fixed_budget_and_does_not_double_count_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, object]] = []

    async def request_json_detailed(**kwargs: object) -> DetailedResponse:
        seen.append(kwargs)
        return DetailedResponse(
            "{}", [CallRecord("deepseek", "deepseek-flash", None, "now", 0.1,
                               OpenCodePromptUsage(input_tokens=1000, output_tokens=2000, reasoning_tokens=1500), "success")],
        )

    import utils.deepseek_transport as deepseek
    monkeypatch.setattr(deepseek, "request_json_detailed", request_json_detailed)
    backend = transport.MatrixBackend()
    backend._credentials = {"DEEPSEEK_API_KEY": "fake"}
    result = asyncio.run(backend("deepseek", "s", "u", {}))
    assert seen[0]["reasoning_effort"] == "off"
    assert seen[0]["max_tokens"] == 16384
    assert seen[0]["allow_length_tool"] is False
    assert result["cost_usd"] == 0.0027
    assert result["usage"]["reasoning_tokens"] == 1500


def test_deepseek_usage_limit_error_keeps_observable_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    async def request_json_detailed(**kwargs: object) -> DetailedResponse:
        return DetailedResponse(
            "{}", [CallRecord("deepseek", "deepseek-flash", None, "now", 0.1,
                              OpenCodePromptUsage(input_tokens=100, output_tokens=16385, total_tokens=16485), "success")]
        )

    import utils.deepseek_transport as deepseek
    monkeypatch.setattr(deepseek, "request_json_detailed", request_json_detailed)
    backend = transport.MatrixBackend()
    backend._credentials = {"DEEPSEEK_API_KEY": "fake"}
    result = asyncio.run(backend("deepseek", "s", "u", {}))
    assert result["error"] == "usage_output_exceeds_max_tokens"
    assert result["cost_known"] is True and result["cost_usd"] is not None
    assert result["text"] == "{}"
    assert result["raw"]["detailed_response_snapshot"]["text"] == "{}"


def test_opencode_accepts_only_one_tool_free_step(monkeypatch: pytest.MonkeyPatch) -> None:
    import utils.opencode_transport as opencode

    class FakeServer:
        base_url = "http://fake"
        auth_header = {"Authorization": "fake"}

    calls: list[dict[str, object]] = []

    async def create(*args: object, **kwargs: object) -> str:
        return "session"

    async def send(*args: object, **kwargs: object) -> DetailedResponse:
        calls.append(kwargs)
        return DetailedResponse("{}", [CallRecord("openai", "gpt-5.6-luna", "medium", "now", 0.1,
                                                     OpenCodePromptUsage(input_tokens=1, output_tokens=1, total_tokens=2), "success")])

    async def delete(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(opencode, "create_session", create)
    monkeypatch.setattr(opencode, "send_prompt_detailed", send)
    monkeypatch.setattr(opencode, "delete_session", delete)
    monkeypatch.setattr(transport.MatrixBackend, "_check_opencode_version", staticmethod(lambda: None))
    backend = transport.MatrixBackend()
    backend._server = FakeServer()
    result = asyncio.run(backend("luna", "s", "u", {}))
    assert result["text"] == "{}"
    assert calls[0]["allow_length_tool"] is False
    assert calls[0]["native_structured_output"] is False
    assert result["cost_kind"] == "subscription_not_priced"


def test_opencode_usage_counts_visible_and_reasoning_tokens_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import utils.opencode_transport as opencode

    async def send(*args: object, **kwargs: object) -> DetailedResponse:
        return DetailedResponse(
            "successful text", [CallRecord("openai", "gpt-5.6-luna", "medium", "now", 0.1,
                                            OpenCodePromptUsage(input_tokens=2455, output_tokens=127,
                                                                reasoning_tokens=882, total_tokens=3464), "success")],
        )

    monkeypatch.setattr(opencode, "create_session", lambda *args, **kwargs: _async_value("session"))
    monkeypatch.setattr(opencode, "send_prompt_detailed", send)
    monkeypatch.setattr(opencode, "delete_session", lambda *args, **kwargs: _async_value(None))
    backend = transport.MatrixBackend()
    backend._server = type("FakeServer", (), {"base_url": "http://fake", "auth_header": {}})()

    result = asyncio.run(backend._call_opencode(transport.MODEL_SPECS["luna"], "s", "u", {}, 0.0))

    assert result["error"] is None
    assert result["text"] == "successful text"
    snapshot = result["raw"]["detailed_response_snapshot"]
    assert snapshot == {"text": "successful text", "error": None, "calls": result["calls"]}
    assert result["usage"] == {"input_tokens": 2455, "output_tokens": 127, "reasoning_tokens": 882, "total_tokens": 3464}


def test_opencode_combined_completion_cap_rejects_before_admission_can_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    import utils.opencode_transport as opencode

    async def send(*args: object, **kwargs: object) -> DetailedResponse:
        return DetailedResponse(
            "text survives", [CallRecord("openai", "gpt-5.6-luna", "medium", "now", 0.1,
                                          OpenCodePromptUsage(input_tokens=1, output_tokens=16384,
                                                              reasoning_tokens=1, total_tokens=16386), "success")],
        )

    monkeypatch.setattr(opencode, "create_session", lambda *args, **kwargs: _async_value("session"))
    monkeypatch.setattr(opencode, "send_prompt_detailed", send)
    monkeypatch.setattr(opencode, "delete_session", lambda *args, **kwargs: _async_value(None))
    backend = transport.MatrixBackend()
    backend._server = type("FakeServer", (), {"base_url": "http://fake", "auth_header": {}})()

    result = asyncio.run(backend._call_opencode(transport.MODEL_SPECS["luna"], "s", "u", {}, 0.0))

    assert result["error"] == "usage_output_exceeds_max_tokens"
    assert result["text"] == "text survives"
    assert result["cost_known"] is False


def test_deepseek_reasoning_must_fit_completion_but_text_is_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    async def request_json_detailed(**kwargs: object) -> DetailedResponse:
        return DetailedResponse(
            "deepseek text", [CallRecord("deepseek", "deepseek-flash", None, "now", 0.1,
                                          OpenCodePromptUsage(input_tokens=100, output_tokens=127,
                                                              reasoning_tokens=882, total_tokens=227), "success")],
            "usage_reasoning_exceeds_output",
        )

    import utils.deepseek_transport as deepseek
    monkeypatch.setattr(deepseek, "request_json_detailed", request_json_detailed)
    backend = transport.MatrixBackend()
    backend._credentials = {"DEEPSEEK_API_KEY": "fake"}

    result = asyncio.run(backend("deepseek", "s", "u", {}))

    assert result["error"] == "usage_reasoning_exceeds_output"
    assert result["text"] == "deepseek text"
    assert result["raw"]["detailed_response_snapshot"]["text"] == "deepseek text"


def test_openencode_config_isolated_and_version_refusal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(transport, "REPO_ROOT", tmp_path)
    original = {"OPENCODE_SERVER_PASSWORD": "keep", "DEEPSEEK_API_KEY": "secret", "MY_ORC_ROUTER_KEY": "secret2"}
    monkeypatch.setattr(transport.os, "environ", original.copy())
    server = object.__new__(transport.MatrixCleanOpenCodeServer)
    server._temp_root = tmp_path
    server._password = "fake-password"
    assert transport.MatrixCleanOpenCodeServer._environment(server)["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] == "16384"
    env = transport.MatrixCleanOpenCodeServer._environment(server)
    assert "DEEPSEEK_API_KEY" not in env and "MY_ORC_ROUTER_KEY" not in env
    assert original["OPENCODE_SERVER_PASSWORD"] == "keep"
    configured = transport.MatrixCleanOpenCodeServer()
    configured._make_config()
    assert configured._temp_root is not None
    with (configured._temp_root / "opencode" / "opencode.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    assert config["agent"]["subtitle"]["steps"] == 1
    assert config["permission"] == {"*": "deny"}
    assert config["tools"] == {}
    assert config["compaction"] == {"auto": False}
    configured.stop()

    class Version:
        returncode = 0
        stdout = "opencode 2.0.0"
        stderr = ""

    monkeypatch.setattr(transport.subprocess, "run", lambda *args, **kwargs: Version())
    backend = transport.MatrixBackend()
    result = asyncio.run(backend("luna", "s", "u", {}))
    assert result["physical_calls_observed"] is None
    assert result["error"] == "opencode_version_required:1.18.29"


def _openrouter_backend(monkeypatch: pytest.MonkeyPatch, payload: dict[str, object], status: int = 200) -> dict[str, object]:
    async def post(*, headers: dict[str, str], body: dict[str, object]) -> tuple[int, bytes]:
        return status, json.dumps(payload).encode()

    backend = transport.MatrixBackend(openrouter_no_byok=True)
    backend._credentials = {"OPENROUTER_API_KEY": "fake"}
    monkeypatch.setattr(backend, "_post_openrouter", post)
    return asyncio.run(backend("gemini", "s", "u", {}))


def _usage(cost: float | None = 0.001, *, output: int = 2) -> dict[str, object]:
    result: dict[str, object] = {"prompt_tokens": 3, "completion_tokens": output, "total_tokens": 3 + output}
    if cost is not None:
        result["cost"] = cost
    return result


def test_openrouter_raw_text_is_not_truncated_or_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    long_text = " {\n" + ("x" * 500) + "\n} "
    result = _openrouter_backend(monkeypatch, {
        "model": "google/gemini-3.8-flash", "provider_name": "google-ai-studio/flex",
        "choices": [{"finish_reason": "stop", "message": {"content": long_text}}],
        "usage": _usage(),
    })
    assert result["raw"]["choices"][0]["message"]["content"] == long_text
    assert result["text"] == long_text


def test_openrouter_http_error_with_usage_keeps_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _openrouter_backend(monkeypatch, {
        "model": "google/gemini-3.8-flash", "usage": _usage(0.004),
        "error": {"message": "provider rejected request"},
    }, status=402)
    assert result["error"].startswith("http_402:")
    assert result["calls"][0]["status"] == "error"
    assert result["cost_known"] is True and result["cost_usd"] == 0.004


def test_openrouter_mismatch_keeps_cost_and_actual_model(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _openrouter_backend(monkeypatch, {
        "model": "google/other-model", "provider_name": "google-ai-studio/flex",
        "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
        "usage": _usage(0.004),
    })
    assert result["error"] == "returned_model_mismatch"
    assert result["model"] == "google/gemini-3.8-flash"
    assert result["returned_model"] == "google/other-model"
    assert result["cost_known"] is True and result["cost_usd"] == 0.004


def test_openrouter_tools_are_blocked_even_when_finish_is_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _openrouter_backend(monkeypatch, {
        "model": "google/gemini-3.8-flash", "provider_name": "google-ai-studio/flex",
        "choices": [{"finish_reason": "stop", "message": {"content": "{}", "tool_calls": [{"id": "1"}]}}],
        "usage": _usage(0.004),
    })
    assert result["error"] == "unexpected_tool_calls"
    assert result["text"] == "{}"
    assert result["cost_known"] is True


def test_openrouter_error_retains_choice_text_as_data(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _openrouter_backend(monkeypatch, {
        "model": "google/gemini-3.8-flash", "provider_name": "google-ai-studio/flex",
        "choices": [{"finish_reason": "length", "message": {"content": "rejected text"}}],
        "usage": _usage(0.004),
        "error": {"message": "provider rejected request"},
    }, status=402)

    assert result["error"].startswith("http_402:")
    assert result["text"] == "rejected text"
    assert result["raw"]["choices"][0]["message"]["content"] == "rejected text"


def test_openrouter_invalid_usage_and_missing_cost_are_not_free(monkeypatch: pytest.MonkeyPatch) -> None:
    invalid = _openrouter_backend(monkeypatch, {
        "model": "google/gemini-3.8-flash", "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
        "usage": _usage(0.004, output=16385),
    })
    missing = _openrouter_backend(monkeypatch, {
        "model": "google/gemini-3.8-flash", "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
        "usage": _usage(None),
    })
    assert invalid["error"] == "usage_output_exceeds_max_tokens"
    assert invalid["cost_known"] is True and invalid["cost_usd"] == 0.004
    assert missing["error"] == "usage_cost_unknown"
    assert missing["cost_known"] is False and missing["cost_usd"] is None


def test_opencode_error_cannot_be_accounted_as_a_free_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import utils.opencode_transport as opencode

    async def send(*args: object, **kwargs: object) -> DetailedResponse:
        return DetailedResponse(
            "{}", [CallRecord("openai", "gpt-5.6-luna", "medium", "now", 0.1,
                              OpenCodePromptUsage(input_tokens=1, output_tokens=1, total_tokens=2), "success")],
            "provider_error",
        )

    monkeypatch.setattr(opencode, "create_session", lambda *args, **kwargs: _async_value("session"))
    monkeypatch.setattr(opencode, "send_prompt_detailed", send)
    monkeypatch.setattr(opencode, "delete_session", lambda *args, **kwargs: _async_value(None))
    backend = transport.MatrixBackend()
    backend._server = type("FakeServer", (), {"base_url": "http://fake", "auth_header": {}})()
    result = asyncio.run(backend._call_opencode(transport.MODEL_SPECS["luna"], "s", "u", {}, 0.0))
    assert result["error"] == "provider_error"
    assert result["cost_known"] is False and result["cost_usd"] is None
    assert result["observed_assistant_steps"] == 1


async def _async_value(value: object) -> object:
    return value
