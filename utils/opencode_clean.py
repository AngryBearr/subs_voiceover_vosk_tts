"""Isolated OpenCode OAuth runtime: subtitle experiments and the pipeline's subscription LLM calls.

OpenCode keeps the login of a ChatGPT subscription (``opencode auth``, OpenAI OAuth). A private
``opencode serve`` with a clean preset config lets the pipeline send its prompts to the OpenAI
models of that subscription instead of paying OpenRouter. OpenCode 2.x has a stateless text
generation route (``generate``); 1.x servers are still checked for the old experiments.
"""

from __future__ import annotations

import atexit
import base64
import json
import logging
import os
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from utils import opencode_pin


log = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
PRESET_ROOT = REPO_ROOT / "config" / "opencode_clean"
DEFAULT_REQUIRED_MODELS: dict[str, set[str]] = {"openai/gpt-6-luna": {"medium"}}
PREFIX = "opencode:"  # a model id "opencode:openai/gpt-6-luna" is sent through the OpenCode subscription
# pipeline reasoning names -> OpenCode variants ("default" / None: the model's own default)
VARIANTS = {"off": "none", "minimal": "low", "max": "xhigh"}


class OpenCodeError(RuntimeError):
    """OpenCode refused or failed a request."""

    def __init__(self, message: str, transient: bool = False, quota: bool = False, status: int | None = None,
                 exhausted: bool = False, empty: bool = False, spent: float = 0.0) -> None:
        super().__init__(message)
        self.transient = transient
        self.quota = quota  # the subscription's usage limit is reached: retrying before its reset is pointless
        self.status = status  # the HTTP status of OpenCode's answer, when there was one
        self.exhausted = exhausted  # still failing after the whole retry budget (``generate_with_retry``)
        self.empty = empty  # the answer had no text (retried a couple of times, never "the subscription is down")
        self.spent = spent  # seconds of the retry budget used (pauses + requests) when ``exhausted``


def is_quota_error(detail: str) -> bool:
    """An OpenCode answer saying the plan's usage limit is reached (OpenCode Go: "usage limit reached (5-hour,
    resets in 4h 23m)"); unlike a short rate limit it lasts hours."""
    return "usage limit" in detail.lower()


def is_model_missing_answer(code: int, detail: str) -> bool:
    """HTTP 404 ``ModelNotFoundError`` / "Model not found": OpenCode 2.x answers it for a while when its model list
    refreshes, and the same model works again moments later, so it is not a permanent failure."""
    lowered = detail.lower()
    return code == 404 and ("modelnotfound" in lowered or "model not found" in lowered)


class ModelNotOffered(RuntimeError):
    """The login's model list does not contain the model (e.g. renamed by an OpenCode update)."""


def provider_models(payload: Any, provider: str) -> list[str]:
    """Ids ``provider/model`` the OpenCode model list (2.x ``/api/model`` or 1.x ``/provider``) offers for a provider."""
    found: list[str] = []
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        found = [f"{provider}/{item.get('id')}" for item in payload["data"]
                 if isinstance(item, dict) and item.get("providerID") == provider and item.get("id")]
    else:
        providers = payload.get("all") if isinstance(payload, dict) else payload
        for item in providers if isinstance(providers, list) else []:
            if isinstance(item, dict) and item.get("id") == provider and isinstance(item.get("models"), dict):
                found = [f"{provider}/{name}" for name in item["models"]]
    return sorted(found)


CONFIG_SETTLE_S = 10.0  # how long a healthy server may still be reading its config
WARMUP_TIMEOUT_S = 60.0  # how long a fresh server may need before its first generation works
WARMUP_POLL_S = 2.0
WARMUP_PROMPT = "Reply with OK"


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
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):  # 2.x: /api/model
        for item in payload["data"]:
            if isinstance(item, dict) and item.get("providerID") == provider and item.get("id") == model:
                return {str(v.get("id")) for v in item.get("variants") or [] if isinstance(v, dict)}
        return None
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
                 timeout: float = 45.0, binary: Optional[str] = None) -> None:
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
        self.binary = binary  # None: the binary chosen by ``opencode_pin.resolve_binary``
        self.warmed: set[str] = set()  # providers whose first generation already worked on this server
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._process: Optional[subprocess.Popen[Any]] = None
        self._temp_root: Optional[Path] = None
        self._neutral_dir: Optional[Path] = None
        self._temp_directory: Optional[tempfile.TemporaryDirectory[str]] = None
        self._neutral_directory: Optional[tempfile.TemporaryDirectory[str]] = None
        self._password = secrets.token_urlsafe(32)
        self.api_version = 1
        self.checked_models: set[str] = set()
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
            "OPENCODE_DISABLE_AUTOUPDATE": "1",  # the pipeline's OpenCode must not change under a running job
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
            if isinstance(self._request("/api/info", 1.5).get("version"), str):  # OpenCode 2.x
                self.api_version = 2
                return True
        except (OSError, ValueError, AttributeError):
            pass
        try:
            return self._request("/global/health", 1.5).get("healthy") is True
        except (OSError, ValueError, AttributeError):
            return False

    def _verify_metadata_v2(self) -> None:
        agents = self._request("/api/agent").get("data") or []
        active = {str(a.get("name", "")).lower() for a in agents if isinstance(a, dict) and not a.get("disable")}
        if active - {"subtitle", "title", "summary", "compaction"}:
            raise RuntimeError("foreign active agent enabled")
        plugins = self._request("/api/plugin").get("data") or []
        if any((p.get("source") or {}).get("type") != "builtin" for p in plugins if isinstance(p, dict)):
            raise RuntimeError("clean config contains foreign plugins")
        if self._request("/api/mcp").get("data") not in ([], None, {}) and not self.enable_length_tool:
            raise RuntimeError("clean config unexpectedly enables MCP")
        self.require_models(self.required_models)

    def require_models(self, models: dict[str, set[str]]) -> None:
        """Fail early when the login does not give these models (and variants)."""
        payload = self._request("/api/model" if self.api_version == 2 else "/provider")
        for model, variants in models.items():
            available = _find_model_variants(payload, model)
            if available is None or not variants.issubset(available):
                offered = provider_models(payload, model.split("/", 1)[0]) if "/" in model else []
                raise ModelNotOffered(
                    f"OpenCode does not offer {model} {sorted(variants)} with this login"
                    + (f"; available: {', '.join(offered)}" if offered else ""))

    def generate(self, prompt: str, model: str, variant: Optional[str] = None, timeout: float = 600.0) -> str:
        """One stateless generation (OpenCode 2.x): the model's text answer to ``prompt``."""
        if self.api_version != 2:
            raise RuntimeError("stateless generation needs OpenCode 2.x")
        provider, model_id = model.removeprefix(PREFIX).split("/", 1)
        ref: dict[str, str] = {"id": model_id, "providerID": provider}
        if variant:
            ref["variant"] = VARIANTS.get(variant, variant)
        request = urllib.request.Request(
            self.base_url + "/api/experimental/generate", method="POST",
            data=json.dumps({"prompt": prompt, "model": ref}).encode("utf-8"),
            headers={**self.auth_header, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            quota = is_quota_error(detail)
            transient = not quota and (exc.code == 429 or exc.code >= 500 or is_model_missing_answer(exc.code, detail))
            raise OpenCodeError(f"HTTP {exc.code}: {detail}", transient=transient, quota=quota,
                                status=exc.code) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OpenCodeError(f"{type(exc).__name__}: {exc}", transient=True) from exc
        text = (data.get("data") or {}).get("text") if isinstance(data, dict) else None
        if not isinstance(text, str):
            raise OpenCodeError("answer without text", transient=True, empty=True)
        return text

    def _verify_metadata(self) -> None:
        if self.api_version == 2:
            self._verify_metadata_v2()
            return
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

    def _binary_path(self) -> str:
        if self.binary:
            return self.binary
        path, source = opencode_pin.resolve_binary()
        if path is None:
            raise RuntimeError("opencode binary not found (install it, or pin one: python -m utils.opencode_pin)")
        opencode_pin.note_binary(path, source)
        return str(path)

    def warm_up(self, model: str, timeout: float = WARMUP_TIMEOUT_S, poll: float = WARMUP_POLL_S) -> bool:
        """Wait until a fresh server really answers for ``model``'s provider (once per provider and server).

        A freshly started OpenCode 2.x lists its models at once but answers the first generations with HTTP 404
        ``ModelNotFoundError`` or 502 "authentication" for up to ~13 s; no status endpoint flips at that moment (checked:
        ``/api/provider``, ``/api/model``, ``/api/credential``, ``/api/integration``), so a tiny generation is the probe.
        Returns False when it never answered: the server is handed out anyway and the callers' retries take over.
        """
        model = model.removeprefix(PREFIX)
        provider = model.split("/", 1)[0]
        if provider in self.warmed or self.api_version != 2:
            return True
        start = time.monotonic()
        while True:
            try:
                self.generate(WARMUP_PROMPT, model, None, timeout=60.0)
            except OpenCodeError as exc:
                if exc.quota:
                    log.warning("OpenCode warm-up for %s stopped: %s", provider, exc)
                    self.warmed.add(provider)
                    return False
                log.debug("OpenCode warming up (%s): %s", provider, exc)
                if time.monotonic() - start + poll > timeout:
                    log.warning("OpenCode did not answer for %s within %.0f s; carrying on", provider, timeout)
                    self.warmed.add(provider)
                    return False
                time.sleep(poll)
            else:
                log.info("OpenCode warming up (%s): ready after %.1f s", provider, time.monotonic() - start)
                self.warmed.add(provider)
                return True

    def start(self) -> None:
        if self._process is not None:
            return
        env = os.environ.copy()
        _validate_oauth(env)
        try:
            self._make_config()
            assert self._neutral_dir is not None
            command = [self._binary_path(), "serve", "--hostname", "127.0.0.1", "--port", str(self.port)]
            self._process = subprocess.Popen(command, cwd=str(self._neutral_dir), env=self._environment(),
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + self.timeout
            settle: Optional[float] = None
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError("opencode server exited during startup")
                if self._healthy():
                    # OpenCode 2.2 reports healthy before it has read the clean config: for a moment it lists
                    # its default agents (Build, General, ...); verify again for a few seconds before giving up.
                    settle = settle if settle is not None else time.monotonic() + CONFIG_SETTLE_S
                    try:
                        self._verify_metadata()
                        return
                    except RuntimeError:
                        if time.monotonic() >= min(settle, deadline):
                            raise
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


_SHARED: Optional[CleanOpenCodeServer] = None
_SHARED_LOCK = threading.Lock()


def shared_server(model: str) -> CleanOpenCodeServer:
    """The process-wide server (started on first use, stopped at exit) after checking ``model``."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            server = CleanOpenCodeServer(required_models={}, timeout=60.0)
            server.start()
            atexit.register(server.stop)
            _SHARED = server
        server = _SHARED
        if model not in server.checked_models:
            server.require_models({model.removeprefix(PREFIX): set()})
            server.checked_models.add(model)
        server.warm_up(model)
        return server


def flatten_messages(messages: list[dict[str, str]]) -> str:
    """Chat messages as one prompt: the stateless route has no separate system role."""
    return "\n\n".join(m["content"] for m in messages if m.get("content"))


SUBSCRIPTION_RETRY_S = 600.0  # how long a busy subscription is retried before the request fails (llm.subscription_retry_s)
SUBSCRIPTION_RETRY_ENV = "STT_LLM_SUBSCRIPTION_RETRY_S"  # set by stt.run from the config for the stage processes
SUBSCRIPTION_BACKOFF_S = (5.0, 15.0, 30.0, 60.0, 120.0, 240.0)  # pauses between tries; the last one repeats


def subscription_retry_budget() -> float:
    """Seconds of waiting a busy subscription gets: the env var set by ``stt.run``, else the default."""
    try:
        return float(os.environ.get(SUBSCRIPTION_RETRY_ENV, SUBSCRIPTION_RETRY_S))
    except ValueError:
        return SUBSCRIPTION_RETRY_S


EMPTY_ANSWER_RETRIES = 2  # an answer without text is retried this often, then fails that request only
EMPTY_ANSWER_PAUSE_S = 2.0


def generate_with_retry(get_server: Callable[[], Any], prompt: str, model: str, variant: Optional[str],
                        timeout: float, retry_s: float) -> str:
    """``generate`` on the subscription, retried while it is busy (shared by ``stt.llm`` and ``voiceover.llm``).

    Transient failures (429, 5xx, connection errors, the transient 404 ModelNotFound, a model list that does not
    show the model for a moment, a server that cannot start or log in yet) wait 5, 15, 30, ... s until ``retry_s``
    is used up, then raise an ``OpenCodeError`` with ``exhausted=True``. The budget counts the pauses (nominal) plus the
    measured duration of the failed requests; a request in flight is never cut short by it. A quota error or a
    refusal is raised at once. An answer without text is retried ``EMPTY_ANSWER_RETRIES`` times, then raised as a
    plain failure of that request (not ``exhausted``). ``get_server`` is called for every try (the model is only
    marked as checked once it was found).
    """
    spent = 0.0
    empty = 0
    for attempt in range(1_000):
        started = time.monotonic()
        try:
            try:
                server = get_server()
            except (OpenCodeError, ModelNotOffered):
                raise
            except RuntimeError as exc:  # the server did not start or the login failed: it may work in a moment
                raise OpenCodeError(f"OpenCode server not ready: {exc}", transient=True) from exc
            return str(server.generate(prompt, model, variant, timeout=timeout))
        except (OpenCodeError, ModelNotOffered) as exc:
            transient = isinstance(exc, ModelNotOffered) or (exc.transient and not exc.quota)
            if not transient:
                raise
            spent += max(0.0, time.monotonic() - started)
            if isinstance(exc, OpenCodeError) and exc.empty:
                empty += 1
                if empty > EMPTY_ANSWER_RETRIES:
                    raise OpenCodeError(f"{exc} (after {empty} tries)", empty=True) from exc
                pause = EMPTY_ANSWER_PAUSE_S
            else:
                remaining = retry_s - spent
                if remaining <= 0:
                    raise OpenCodeError(f"subscription not answering after {spent:.0f} s: {exc}", exhausted=True,
                                        spent=spent) from exc
                pause = min(SUBSCRIPTION_BACKOFF_S[min(attempt, len(SUBSCRIPTION_BACKOFF_S) - 1)], remaining)
            log.warning("subscription busy (%s), retry in %.0f s", exc, pause)
            time.sleep(pause)
            spent += pause
    raise OpenCodeError("subscription not answering", exhausted=True, spent=spent)


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
