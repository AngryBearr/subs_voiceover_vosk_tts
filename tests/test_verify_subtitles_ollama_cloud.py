import json
from pathlib import Path

import utils.verify_subtitles_ollama_cloud as cli
from utils.opencode_transport import OpenCodePromptResult


def _files(tmp_path: Path):
    original = tmp_path / "original.json"
    candidate = tmp_path / "candidate.name.json"
    original.write_text('[{"index": 1, "text": "old"}]', encoding="utf-8")
    candidate.write_text('[{"index": 1, "text": "new"}]', encoding="utf-8")
    return original, candidate


def test_model_is_required_and_other_defaults_exact():
    args = cli.build_parser().parse_args(["original", "candidate", "--model", "glm-5.2"])
    assert args.model == "glm-5.2" and args.seed == 42 and args.timeout == 300.0
    assert args.output_dir == "output/ollama_cloud_semantic_verified"
    assert args.temperature == 0.0 and args.batch_size == 4 and args.concurrency == 1


def test_missing_model_is_argparse_error():
    try:
        cli.build_parser().parse_args(["original", "candidate"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("missing --model must be rejected")


def test_invalid_model_is_config_error(tmp_path):
    original, candidate = _files(tmp_path)
    assert cli.main([str(original), str(candidate), "--model", "bad model", "--api-key", "key"]) == 2


def test_key_precedence_and_request_contract(tmp_path, monkeypatch):
    original, candidate = _files(tmp_path)
    seen = {}
    monkeypatch.setattr(cli, "load_api_key", lambda name: (_ for _ in ()).throw(AssertionError(name)))

    async def request(**kwargs):
        seen.update(kwargs)
        return OpenCodePromptResult('{"results":[]}', None)

    monkeypatch.setattr(cli, "request_strict_json", request)

    async def verify(*args, **kwargs):
        seen["verify"] = {key: kwargs[key] for key in ("semantic_units", "backend", "accounting", "cost_is_billing_authoritative")}
        await kwargs["request_callable"]("payload", kwargs["model"])
        return args[1], {}, {}

    monkeypatch.setattr(cli, "verify_items", verify)
    assert cli.main([str(original), str(candidate), "--model", "glm-5.2", "--api-key", "explicit", "--output-dir", str(tmp_path / "out")]) == 0
    assert seen["api_key"] == "explicit" and seen["seed"] == 42
    assert seen["verify"] == {"semantic_units": True, "backend": "ollama_cloud", "accounting": "subscription", "cost_is_billing_authoritative": False}


def test_environment_key_and_outputs(tmp_path, monkeypatch):
    original, candidate = _files(tmp_path)
    monkeypatch.setattr(cli, "load_api_key", lambda name: "env-key")

    async def verify(*args, **kwargs):
        return args[1], {"backend": "ollama_cloud"}, {"accounting": "subscription"}

    monkeypatch.setattr(cli, "verify_items", verify)
    out = tmp_path / "out"
    assert cli.main([str(original), str(candidate), "--model", "glm-5.2", "--output-dir", str(out)]) == 0
    assert sorted(path.name for path in out.iterdir()) == ["candidate.name_ollama_cloud_verified.json", "candidate.name_ollama_cloud_verified.report.json", "candidate.name_ollama_cloud_verified.usage.json"]
    assert all(path.read_bytes().endswith(b"\n") for path in out.iterdir())


def test_invalid_input_precedes_key(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_api_key", lambda name: (_ for _ in ()).throw(AssertionError(name)))
    assert cli.main([str(tmp_path / "missing"), str(tmp_path / "candidate"), "--model", "glm-5.2"]) == 2
