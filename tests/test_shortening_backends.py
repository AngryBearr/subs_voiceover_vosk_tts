from __future__ import annotations

import asyncio
import builtins
import json
import os
import platform
import sys
from typing import Any

import pytest

from utils import deepseek_transport
from utils import opencode_transport
from utils.llm_types import DetailedResponse
from utils.shortening_backends import LiveBackend, transport_runtime_metadata
from utils.shortening_runner import ModelSpec


def test_live_backend_deepseek_uses_bound_limit_without_module_global(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    async def fake_request(**kwargs: Any) -> DetailedResponse:
        seen.update(kwargs)
        return DetailedResponse("{}", [])

    monkeypatch.setattr(deepseek_transport, "request_json_detailed", fake_request)
    server = type("Server", (), {"base_url": "http://server", "auth_header": None})()
    backend = LiveBackend(server, deepseek_api_key="secret")
    result = asyncio.run(backend(ModelSpec("deepseek", "deepseek-flash", "low"), "system", "user", {}, False))

    assert result.text == "{}"
    assert seen["max_tokens"] == 65_536
    import utils.shortening_backends as backend_module
    assert not hasattr(backend_module, "limit")


def test_transport_runtime_metadata_records_safe_environment_provenance(monkeypatch: Any) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/subs_env")
    monkeypatch.setenv("TRANSPORT_SECRET", "do-not-record")

    metadata = transport_runtime_metadata()

    assert metadata["python_executable"] == sys.executable
    assert metadata["python_version"] == platform.python_version()
    assert metadata["sys_prefix"] == sys.prefix
    assert metadata["active_virtual_env"] == "/tmp/subs_env"
    assert metadata["virtual_env"] == "/tmp/subs_env"
    assert metadata["aiohttp_version"]
    assert "do-not-record" not in json.dumps(metadata)
    assert metadata["virtual_env"] == os.environ["VIRTUAL_ENV"]


def test_missing_transport_dependency_fails_before_server_use(monkeypatch: Any) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "aiohttp":
            raise ModuleNotFoundError("No module named aiohttp", name="aiohttp")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    server = type("Server", (), {})()
    with pytest.raises(RuntimeError, match="^missing_transport_dependency:aiohttp$"):
        LiveBackend(server)


def test_opencode_module_not_found_error_keeps_only_safe_module_name(monkeypatch: Any) -> None:
    async def fail_create_session(*_: Any, **__: Any) -> str:
        raise ModuleNotFoundError("secret /path", name="aiohttp")

    monkeypatch.setattr(opencode_transport, "create_session", fail_create_session)
    backend = object.__new__(LiveBackend)
    backend.server = type("Server", (), {"base_url": "http://server", "auth_header": None})()
    backend.timeout = 1.0
    result = asyncio.run(backend._opencode(
        ModelSpec("opencode", "openai/gpt-5.6-luna", "medium"), "system", "user", {}, False
    ))

    assert result.error == "ModuleNotFoundError:aiohttp"
    assert "secret" not in (result.error or "")
