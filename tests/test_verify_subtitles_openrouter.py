import asyncio
import json
from pathlib import Path

import pytest

import utils.verify_subtitles_openrouter as cli
from utils.opencode_transport import OpenCodePromptResult


def _files(tmp_path: Path):
    original = tmp_path / "original.json"
    candidate = tmp_path / "candidate.name.json"
    original.write_text('[{"index": 1, "text": "old"}]', encoding="utf-8")
    candidate.write_text('[{"index": 1, "text": "new"}]', encoding="utf-8")
    return original, candidate


def test_parser_defaults_exact():
    args = cli.build_parser().parse_args(["original", "candidate"])
    assert args.model == "xiaomi/mimo-v2.5-pro"
    assert (args.batch_size, args.concurrency, args.context_window) == (4, 1, 3)
    assert (args.transport_retries, args.schema_retries) == (1, 1)
    assert (args.semantic_unit_max_cues, args.semantic_unit_max_gap_sec) == (3, 0.3)
    assert args.timeout == 180.0 and args.temperature == 0.0
    assert args.output_dir == "output/openrouter_semantic_verified"


@pytest.mark.parametrize("extra", [["--model", "model"], ["--timeout", "0"], ["--temperature", "-0.1"], ["--temperature", "2.1"], ["--temperature", "nan"], ["--batch-size", "0"], ["--context-window", "-1"]])
def test_invalid_config_precedes_key_and_request(tmp_path, monkeypatch, extra):
    original, candidate = _files(tmp_path)
    monkeypatch.setattr(cli, "load_api_key", lambda name: pytest.fail("key lookup"))
    monkeypatch.setattr(cli, "request_json_schema", lambda **kwargs: pytest.fail("request"))
    assert cli.main([str(original), str(candidate), *extra]) == 2


def test_boolean_temperature_is_invalid_config():
    args = cli.build_parser().parse_args(["original", "candidate"])
    args.temperature = True
    with pytest.raises(ValueError, match="^invalid_temperature$"):
        cli._validate_config(args)


def test_invalid_input_precedes_key(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_api_key", lambda name: pytest.fail("key lookup"))
    assert cli.main([str(tmp_path / "missing"), str(tmp_path / "candidate")]) == 2


def test_missing_key_returns_one_without_request(tmp_path, monkeypatch):
    original, candidate = _files(tmp_path)
    monkeypatch.setattr(cli, "load_api_key", lambda name: None)
    monkeypatch.setattr(cli, "request_json_schema", lambda **kwargs: pytest.fail("request"))
    assert cli.main([str(original), str(candidate)]) == 1


def test_explicit_key_precedes_loader(tmp_path, monkeypatch):
    original, candidate = _files(tmp_path)
    seen = {}
    monkeypatch.setattr(cli, "load_api_key", lambda name: pytest.fail("loader called"))
    async def request(**kwargs):
        seen["key"] = kwargs["api_key"]
        return OpenCodePromptResult('{"results":[]}', None)
    monkeypatch.setattr(cli, "request_json_schema", request)
    async def fake_verify(*args, **kwargs):
        await kwargs["request_callable"]("{}", kwargs["model"])
        return args[1], {"status": "completed"}, {"backend": "openrouter"}
    monkeypatch.setattr(cli, "verify_items", fake_verify)
    assert cli.main([str(original), str(candidate), "--api-key", "explicit", "--output-dir", str(tmp_path / "out")]) == 0
    assert seen["key"] == "explicit"


def test_environment_key_and_attribution_fallback(tmp_path, monkeypatch):
    original, candidate = _files(tmp_path)
    seen = {}
    monkeypatch.setattr(cli, "load_api_key", lambda name: "env-key")
    async def request(**kwargs):
        seen.update(kwargs)
        return OpenCodePromptResult('{"results":[]}', None)
    monkeypatch.setattr(cli, "request_json_schema", request)
    async def fake_verify(*args, **kwargs):
        await kwargs["request_callable"]("{}", kwargs["model"])
        return args[1], {}, {}
    monkeypatch.setattr(cli, "verify_items", fake_verify)
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "env-ref")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "env-title")
    assert cli.main([str(original), str(candidate), "--output-dir", str(tmp_path / "out")]) == 0
    assert seen["api_key"] == "env-key" and seen["http_referer"] == "env-ref" and seen["app_title"] == "env-title"


def test_cli_attribution_overrides_environment(tmp_path, monkeypatch):
    original, candidate = _files(tmp_path)
    seen = {}
    monkeypatch.setattr(cli, "load_api_key", lambda name: "key")
    async def request(**kwargs):
        seen.update(kwargs)
        return OpenCodePromptResult('{"results":[]}', None)
    monkeypatch.setattr(cli, "request_json_schema", request)
    async def fake_verify(*args, **kwargs):
        await kwargs["request_callable"]("{}", kwargs["model"])
        return args[1], {}, {}
    monkeypatch.setattr(cli, "verify_items", fake_verify)
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "env-ref")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "env-title")
    assert cli.main([str(original), str(candidate), "--http-referer", "cli-ref", "--app-title", "cli-title", "--output-dir", str(tmp_path / "out")]) == 0
    assert seen["http_referer"] == "cli-ref" and seen["app_title"] == "cli-title"


def test_request_callable_uses_exact_unit_prompt_and_fresh_schema(tmp_path, monkeypatch):
    original, candidate = _files(tmp_path)
    seen = []
    monkeypatch.setattr(cli, "load_api_key", lambda name: "key")
    async def request(**kwargs):
        seen.append(kwargs)
        return OpenCodePromptResult('{"results":[]}', None)
    monkeypatch.setattr(cli, "request_json_schema", request)
    captured = {}
    async def fake_verify(*args, **kwargs):
        captured.update(kwargs)
        await kwargs["request_callable"]("payload", kwargs["model"])
        await kwargs["request_callable"]("payload2", kwargs["model"])
        return args[1], {}, {}
    monkeypatch.setattr(cli, "verify_items", fake_verify)
    assert cli.main([str(original), str(candidate), "--model", "p/m", "--batch-size", "2", "--concurrency", "1", "--timeout", "12", "--output-dir", str(tmp_path / "out")]) == 0
    assert captured["semantic_units"] is True and captured["backend"] == "openrouter"
    assert captured["accounting"] == "metered_api" and captured["cost_is_billing_authoritative"] is True
    assert len(seen) == 2 and all(item["system_prompt"] == cli.UNIT_VERIFIER_SYSTEM_PROMPT for item in seen)
    assert all(item["temperature"] == 0.0 for item in seen)
    assert seen[0]["schema"] == cli.unit_verification_schema() and seen[0]["schema"] is not seen[1]["schema"]


def test_custom_temperature_is_forwarded(tmp_path, monkeypatch):
    original, candidate = _files(tmp_path)
    seen = {}
    monkeypatch.setattr(cli, "load_api_key", lambda name: "key")

    async def request(**kwargs):
        seen.update(kwargs)
        return OpenCodePromptResult('{"results":[]}', None)

    monkeypatch.setattr(cli, "request_json_schema", request)

    async def fake_verify(*args, **kwargs):
        await kwargs["request_callable"]("payload", kwargs["model"])
        return args[1], {}, {}

    monkeypatch.setattr(cli, "verify_items", fake_verify)
    assert cli.main([
        str(original), str(candidate), "--temperature", "1.25", "--output-dir", str(tmp_path / "out")
    ]) == 0
    assert seen["temperature"] == 1.25


def test_outputs_exact_names_newlines_and_no_secret(tmp_path, monkeypatch):
    original, candidate = _files(tmp_path)
    monkeypatch.setattr(cli, "load_api_key", lambda name: "secret")
    async def fake_verify(*args, **kwargs):
        return args[1], {"backend": "openrouter"}, {"marker": False}
    monkeypatch.setattr(cli, "verify_items", fake_verify)
    out = tmp_path / "out"
    assert cli.main([str(original), str(candidate), "--output-dir", str(out)]) == 0
    paths = sorted(path.name for path in out.iterdir())
    assert paths == ["candidate.name_openrouter_verified.json", "candidate.name_openrouter_verified.report.json", "candidate.name_openrouter_verified.usage.json"]
    for path in out.iterdir():
        assert path.read_bytes().endswith(b"\n") and b"secret" not in path.read_bytes()


def test_fail_closed_output_is_persisted(tmp_path, monkeypatch):
    original, candidate = _files(tmp_path)
    monkeypatch.setattr(cli, "load_api_key", lambda name: "key")
    output = [{"index": 1, "text": "old"}]
    async def fake_verify(*args, **kwargs):
        return output, {"status": "completed_with_unresolved"}, {"transport_failures": 1}
    monkeypatch.setattr(cli, "verify_items", fake_verify)
    out = tmp_path / "out"
    assert cli.main([str(original), str(candidate), "--output-dir", str(out)]) == 0
    assert json.loads((out / "candidate.name_openrouter_verified.json").read_text()) == output


def test_cli_does_not_use_server_lifecycle(tmp_path, monkeypatch):
    original, candidate = _files(tmp_path)
    monkeypatch.setattr(cli, "load_api_key", lambda name: "key")
    async def fake_verify(*args, **kwargs):
        return args[1], {}, {}
    monkeypatch.setattr(cli, "verify_items", fake_verify)
    assert cli.main([str(original), str(candidate), "--output-dir", str(tmp_path / "out")]) == 0
    assert not hasattr(cli, "OpenCodeServer")
