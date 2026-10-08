from __future__ import annotations

import json

import pytest

from utils import opencode_clean


def test_auth_preflight_requires_oauth(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path  # type: ignore[assignment]
    data = root / "opencode"
    data.mkdir()
    (data / "auth.json").write_text(json.dumps({"openai": {"type": "api"}}), encoding="utf-8")
    monkeypatch.setenv("XDG_DATA_HOME", str(root))
    with pytest.raises(RuntimeError, match="OAuth"):
        opencode_clean._validate_oauth(dict(opencode_clean.os.environ))


def test_environment_does_not_inherit_server_or_api_configuration(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_CONFIG", "foreign")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    server = opencode_clean.CleanOpenCodeServer(timeout=0.01)
    server._temp_root = server._temp_root or opencode_clean.Path(str(tmp_path))
    env = server._environment()
    assert "OPENCODE_CONFIG" not in env
    assert "OPENAI_API_KEY" not in env
    assert env["OPENCODE_PURE"] == "1"


def test_startup_exception_cleans_temp_dirs(monkeypatch: pytest.MonkeyPatch) -> None:
    server = opencode_clean.CleanOpenCodeServer(timeout=0.01)
    monkeypatch.setattr(opencode_clean, "_validate_oauth", lambda _: None)
    monkeypatch.setattr(server, "_healthy", lambda: False)
    with pytest.raises(TimeoutError):
        server.start()
    assert server._process is None
    assert server._temp_root is None


def test_find_model_variants_does_not_cross_provider_boundaries() -> None:
    payload = {"all": [
        {"id": "foreign", "models": {"luna": {"variants": {"medium": {}}}}},
        {"id": "openai", "models": {"other": {"variants": {"medium": {}}}}},
    ]}
    assert opencode_clean._find_model_variants(payload, "openai/luna") is None


def test_subtitle_prompt_allows_only_explicit_counter_or_format_tools() -> None:
    config_path = opencode_clean.PRESET_ROOT / "opencode" / "opencode.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prompt = config["agent"]["subtitle"]["prompt"]
    assert "length-checking" in prompt
    assert "response-format" in prompt
    assert "Do not access files or delegate to agents" in prompt


def test_config_creation_failure_cleans_owned_temporary_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    server = opencode_clean.CleanOpenCodeServer(timeout=0.01)

    def fail_after_allocating() -> None:
        server._temp_directory = opencode_clean.tempfile.TemporaryDirectory(prefix="opencode-clean-test-")
        server._temp_root = opencode_clean.Path(server._temp_directory.name)
        raise RuntimeError("config creation failed")

    monkeypatch.setattr(opencode_clean, "_validate_oauth", lambda _: None)
    monkeypatch.setattr(server, "_make_config", fail_after_allocating)
    with pytest.raises(RuntimeError, match="config creation failed"):
        server.start()
    assert server._temp_directory is None
    assert server._temp_root is None
