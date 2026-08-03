"""CLI and integration coverage for the DeepSeek verifier."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

import utils.verify_subtitles_deepseek as verifier


def write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    original = tmp_path / "original.json"
    candidate = tmp_path / "candidate.json"
    original.write_text(json.dumps([{"index": 1, "text": ["original"]}]), encoding="utf-8")
    candidate.write_text(json.dumps([{"index": 1, "text": ["changed"]}]), encoding="utf-8")
    return original, candidate


def test_parser_defaults_and_custom_values() -> None:
    parser = verifier.build_parser()
    defaults = parser.parse_args(["o", "c"])
    assert defaults.model == "deepseek-v4-flash" and defaults.batch_size == 4 and defaults.concurrency == 1
    custom = parser.parse_args(["o", "c", "--base-url", "https://example.test/v1", "--model", "deepseek-x", "--temperature", "1.5", "--timeout", "9"])
    assert custom.base_url == "https://example.test/v1" and custom.model == "deepseek-x" and custom.temperature == 1.5 and custom.timeout == 9


def test_invalid_config_and_input_precede_key_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original, candidate = write_inputs(tmp_path)
    looked_up = False
    def key(_: str) -> str:
        nonlocal looked_up
        looked_up = True
        return "secret"
    monkeypatch.setattr(verifier, "load_api_key", key)
    assert verifier.main([str(original), str(candidate), "--base-url", "ftp://bad"]) == 2
    assert not looked_up
    assert verifier.main([str(tmp_path / "missing"), str(candidate)]) == 2
    assert not looked_up


def test_key_precedence_and_environment_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original, candidate = write_inputs(tmp_path)
    captured: list[str] = []
    async def request(**kwargs: Any) -> Any:
        captured.append(kwargs["api_key"])
        return None
    monkeypatch.setattr(verifier, "request_json_object", request)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment")
    assert verifier.main([str(original), str(candidate), "--api-key", "explicit", "--output-dir", str(tmp_path / "out")]) == 0
    assert captured and set(captured) == {"explicit"}
    captured.clear()
    assert verifier.main([str(original), str(candidate), "--output-dir", str(tmp_path / "out2")]) == 0
    assert captured and set(captured) == {"environment"}
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    monkeypatch.setattr(verifier, "load_api_key", lambda _: "")
    assert verifier.main([str(original), str(candidate)]) == 1


def test_callback_and_verify_items_forwarding_and_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original, candidate = write_inputs(tmp_path)
    captured: dict[str, Any] = {}
    async def request_json(**kwargs: Any) -> Any:
        captured["request"] = kwargs
        return None
    async def fake_verify(left: Any, right: Any, **kwargs: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        captured["verify"] = kwargs
        callback = kwargs["request_callable"]
        assert await callback('{"targets": []}', "deepseek-v4-flash") is None
        return right, {"backend": "deepseek", "accounting": "metered_api"}, {"backend": "deepseek", "accounting": "metered_api", "provider_cost_is_billing_authoritative": False}
    monkeypatch.setattr(verifier, "request_json_object", request_json)
    monkeypatch.setattr(verifier, "verify_items", fake_verify)
    output_dir = tmp_path / "output"
    assert verifier.main([str(original), str(candidate), "--api-key", "secret", "--base-url", "https://example.test/api", "--timeout", "8", "--temperature", "0.4", "--transport-retries", "2", "--schema-retries", "0", "--semantic-unit-max-cues", "2", "--semantic-unit-max-gap-sec", "0.2", "--output-dir", str(output_dir)]) == 0
    assert captured["request"] == {"api_key": "secret", "model": "deepseek-v4-flash", "system_prompt": verifier.UNIT_VERIFIER_SYSTEM_PROMPT, "user_prompt": '{"targets": []}', "base_url": "https://example.test/api", "timeout": 8.0, "temperature": 0.4}
    forwarded = captured["verify"]
    assert forwarded["semantic_units"] is True and forwarded["backend"] == "deepseek" and forwarded["accounting"] == "metered_api" and forwarded["cost_is_billing_authoritative"] is False
    assert forwarded["transport_retries"] == 2 and forwarded["schema_retries"] == 0 and forwarded["semantic_unit_max_cues"] == 2
    for suffix in (".json", ".report.json", ".usage.json"):
        path = output_dir / f"candidate_deepseek_verified{suffix}"
        assert path.exists() and path.read_bytes().endswith(b"\n") and b"secret" not in path.read_bytes()


def test_transport_failure_persists_original_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original, candidate = write_inputs(tmp_path)
    async def fake_verify(left: Any, right: Any, **kwargs: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        callback = kwargs["request_callable"]
        assert await callback("prompt", kwargs["model"]) is None
        return [{"index": 1, "text": ["original"]}], {"status": "completed_with_unresolved"}, {"token_usage_available": False}
    async def no_response(**_: Any) -> None:
        return None
    monkeypatch.setattr(verifier, "verify_items", fake_verify)
    monkeypatch.setattr(verifier, "request_json_object", no_response)
    out = tmp_path / "out"
    assert verifier.main([str(original), str(candidate), "--api-key", "secret", "--output-dir", str(out)]) == 0
    assert json.loads((out / "candidate_deepseek_verified.json").read_text()) == [{"index": 1, "text": ["original"]}]
