"""Isolated, metadata-only OpenCode OAuth runtime for subtitle experiments."""

from __future__ import annotations

import base64
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
PRESET_ROOT = REPO_ROOT / "config" / "opencode_clean"
DEFAULT_REQUIRED_MODELS: dict[str, set[str]] = {"openai/gpt-5.6-luna": {"medium"}}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _auth_path(env: dict[str, str]) -> Path:
    data_home = Path(env.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data_home / "opencode" / "auth.json"


def _validate_oauth(env: dict[str, str]) -> None:
    path = _auth_path(env)
    try:
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, ValueError) as exc:
        raise RuntimeError("OpenAI OAuth auth preflight failed") from exc
    entry = data.get("openai") if isinstance(data, dict) else None
    if not isinstance(entry, dict) or entry.get("type") != "oauth":
        raise RuntimeError("OpenAI OAuth auth preflight failed")


def _find_model_variants(payload: Any, wanted: str) -> Optional[set[str]]:
    if "/" not in wanted or not isinstance(payload, (dict, list)):
        return None
    provider, model = wanted.split("/", 1)
    providers = payload.get("all") if isinstance(payload, dict) else payload
    if not isinstance(providers, list):
        return None
    for item in providers:
        if not isinstance(item, dict) or item.get("id") != provider:
            continue
        models = item.get("models")
        if not isinstance(models, dict) or not isinstance(models.get(model), dict):
            return None
        variants = models[model].get("variants")
        if isinstance(variants, dict):
            return {str(key) for key in variants}
        return None
    return None


class CleanOpenCodeServer:
    """A disposable OpenCode server with independent config and neutral cwd."""

    def __init__(self, *, enable_length_tool: bool = False,
                 required_models: Optional[dict[str, set[str]]] = None,
                 timeout: float = 45.0) -> None:
        if not isinstance(enable_length_tool, bool):
            raise TypeError("enable_length_tool must be bool")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self.enable_length_tool = enable_length_tool
        selected_models = required_models if required_models is not None else DEFAULT_REQUIRED_MODELS
        if not isinstance(selected_models, dict) or any(
            not isinstance(model, str) or not model.strip() or not isinstance(variants, set)
            or any(not isinstance(variant, str) or not variant.strip() for variant in variants)
            for model, variants in selected_models.items()
        ):
            raise TypeError("required_models must map model ids to variant sets")
        self.required_models = {model: set(variants) for model, variants in selected_models.items()}
        self.timeout = float(timeout)
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._process: Optional[subprocess.Popen[Any]] = None
        self._temp_root: Optional[Path] = None
        self._neutral_dir: Optional[Path] = None
        self._temp_directory: Optional[tempfile.TemporaryDirectory[str]] = None
        self._neutral_directory: Optional[tempfile.TemporaryDirectory[str]] = None
        self._password = secrets.token_urlsafe(32)
        self._auth_header = {"Authorization": "Basic " + base64.b64encode(f"opencode:{self._password}".encode()).decode()}

    @property
    def auth_header(self) -> dict[str, str]:
        return dict(self._auth_header)

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("OPENCODE_") or key in {"OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"}:
                env.pop(key, None)
        assert self._temp_root is not None
        env.update({
            "XDG_CONFIG_HOME": str(self._temp_root),
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            "OPENCODE_DISABLE_CLAUDE_CODE": "1",
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
            "OPENCODE_PURE": "1",
            "OPENCODE_SERVER_PASSWORD": self._password,
            "OPENCODE_SERVER_USERNAME": "opencode",
        })
        env["VIRTUAL_ENV"] = str(REPO_ROOT / "subs_env")
        return env

    def _make_config(self) -> None:
        self._temp_directory = tempfile.TemporaryDirectory(prefix="opencode-clean-config-")
        self._temp_root = Path(self._temp_directory.name)
        source = PRESET_ROOT / "opencode"
        target = self._temp_root / "opencode"
        shutil.copytree(source, target)
        config_path = target / "opencode.json"
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        if self.enable_length_tool:
            config["mcp"] = {
                "length": {
                    "type": "local", "command": ["uv", "--directory", str(REPO_ROOT), "run", "--active", "--no-sync", "-m", "utils.length_mcp"],
                    "environment": {"VIRTUAL_ENV": str(REPO_ROOT / "subs_env")}, "enabled": True,
                }
            }
            config["permission"]["length_check_length"] = "allow"
        with config_path.open("w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)
        self._neutral_directory = tempfile.TemporaryDirectory(prefix="opencode-clean-cwd-")
        self._neutral_dir = Path(self._neutral_directory.name)

    def _request(self, path: str, timeout: float = 3.0) -> Any:
        import urllib.request
        request = urllib.request.Request(self.base_url + path, headers=self.auth_header)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())

    def _healthy(self) -> bool:
        try:
            return self._request("/global/health", 1.5).get("healthy") is True
        except (OSError, ValueError, AttributeError):
            return False

    def _verify_metadata(self) -> None:
        config = self._request("/config")
        if config.get("plugin") not in ([], None) or config.get("instructions") not in ([], None) or config.get("skills") not in ({"paths": [], "urls": []}, None):
            raise RuntimeError("clean config contains foreign extensions")
        allowed_mcp = {"length"} if self.enable_length_tool else set()
        mcp_config = config.get("mcp")
        if mcp_config not in ({}, None) and (not isinstance(mcp_config, dict) or set(mcp_config) - allowed_mcp):
            raise RuntimeError("clean config unexpectedly enables MCP")
        agent = self._request("/agent")
        agent_values: list[dict[str, Any]] = []
        if isinstance(agent, dict):
            for key, value in agent.items():
                if isinstance(value, dict):
                    item = dict(value)
                    item.setdefault("name", key)
                    agent_values.append(item)
        elif isinstance(agent, list):
            agent_values = [item for item in agent if isinstance(item, dict)]
        allowed_active = {"subtitle", "title", "summary", "compaction"}
        forbidden_active = {"general", "explore", "build", "plan"}
        for item in agent_values:
            name = item.get("name")
            if not isinstance(name, str) or item.get("disable", item.get("disabled", False)):
                continue
            if name in forbidden_active or name not in allowed_active:
                raise RuntimeError("foreign active agent enabled")
        providers = self._request("/provider")
        for model, variants in self.required_models.items():
            available = _find_model_variants(providers, model)
            if available is None or not variants.issubset(available):
                raise RuntimeError("required model metadata unavailable")
        if self.enable_length_tool:
            tools = self._request("/mcp")
            if not isinstance(tools, dict) or tools.get("length", {}).get("status") != "connected":
                raise RuntimeError("length MCP tool metadata unavailable")
            if not self._verify_length_tool_registry():
                raise RuntimeError("length MCP tool metadata unavailable")

    def _verify_length_tool_registry(self) -> bool:
        """List the configured server's real tools without invoking a model."""
        command = ["uv", "--directory", str(REPO_ROOT), "run", "--active", "--no-sync", "-m", "utils.length_mcp"]
        request = "\n".join(json.dumps(item) for item in (
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "clean-runtime-check", "version": "1"},
            }},
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )) + "\n"
        env = self._environment()
        try:
            result = subprocess.run(command, input=request, text=True, capture_output=True,
                                    timeout=10, env=env, cwd=str(REPO_ROOT), check=False)
            for line in result.stdout.splitlines():
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                tools = payload.get("result", {}).get("tools") if isinstance(payload, dict) else None
                if isinstance(tools, list) and any(isinstance(tool, dict) and tool.get("name") == "check_length" for tool in tools):
                    return True
        except (OSError, subprocess.TimeoutExpired):
            return False
        return False

    def start(self) -> None:
        if self._process is not None:
            return
        env = os.environ.copy()
        _validate_oauth(env)
        try:
            self._make_config()
            assert self._neutral_dir is not None
            command = ["opencode", "serve", "--hostname", "127.0.0.1", "--port", str(self.port)]
            self._process = subprocess.Popen(command, cwd=str(self._neutral_dir), env=self._environment(),
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError("opencode server exited during startup")
                if self._healthy():
                    self._verify_metadata()
                    return
                time.sleep(0.15)
            raise TimeoutError("opencode server startup timed out")
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._neutral_directory is not None:
            self._neutral_directory.cleanup()
        if self._temp_directory is not None:
            self._temp_directory.cleanup()
        self._neutral_directory = None
        self._temp_directory = None
        self._neutral_dir = None
        self._temp_root = None

    def __enter__(self) -> "CleanOpenCodeServer":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Check isolated OpenCode metadata without inference.")
    parser.add_argument("--check", action="store_true", help="start, verify metadata, and stop")
    parser.add_argument("--with-length-tool", action="store_true", help="include the narrowly scoped length MCP")
    args = parser.parse_args()
    if not args.check:
        parser.error("--check is required")
    with CleanOpenCodeServer(enable_length_tool=args.with_length_tool) as server:
        print(json.dumps({"status": "ok", "base_url": server.base_url, "length_tool": args.with_length_tool}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
