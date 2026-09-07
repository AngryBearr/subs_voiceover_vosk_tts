"""Neutral OpenCode HTTP transport and local server lifecycle helpers."""

from __future__ import annotations

import base64
import copy
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class OpenCodePromptUsage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    cost: Optional[float] = None
    provider_id: Optional[str] = None
    model_id: Optional[str] = None
    finish: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"OpenCode usage {name} must be a nonnegative integer")
        if self.cost is not None and (isinstance(self.cost, bool) or not isinstance(self.cost, (int, float)) or not math.isfinite(float(self.cost)) or self.cost < 0):
            raise ValueError("OpenCode usage cost must be a nonnegative finite number")
        for name in ("provider_id", "model_id", "finish"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"OpenCode usage {name} must be a string")


@dataclass(frozen=True)
class OpenCodePromptResult:
    text: str
    usage: Optional[OpenCodePromptUsage]


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_server_env() -> Dict[str, str]:
    """Create a server environment without inherited server credentials."""
    env = os.environ.copy()
    env.pop("OPENCODE_SERVER_PASSWORD", None)
    env.pop("OPENCODE_SERVER_USERNAME", None)
    return env


class OpenCodeServer:
    """Manages an opencode serve subprocess."""

    def __init__(self, port: Optional[int] = None, hostname: str = "127.0.0.1") -> None:
        self.port = port or _find_free_port()
        self.hostname = hostname
        self.base_url = f"http://{self.hostname}:{self.port}"
        self._process: Optional[subprocess.Popen[Any]] = None

    def start(self, timeout: float = 30.0) -> None:
        cmd = ["opencode", "serve", "--port", str(self.port), "--hostname", self.hostname]
        print(f"Starting opencode server on {self.base_url} ...")
        self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_make_server_env())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                raise RuntimeError(f"opencode serve exited with code {self._process.returncode}. stderr: {stderr[:500]}")
            if self._is_healthy():
                print(f"Server ready (pid={self._process.pid})")
                return
            time.sleep(0.5)
        self.stop()
        raise TimeoutError(f"opencode server did not become healthy within {timeout}s")

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        self._process = None
        print("Server stopped")

    def _is_healthy(self) -> bool:
        import urllib.error
        import urllib.request
        try:
            req = urllib.request.Request(f"{self.base_url}/global/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return json.loads(resp.read()).get("healthy") is True
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return False

    def __enter__(self) -> "OpenCodeServer":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


def _get_auth_header() -> Optional[Dict[str, str]]:
    password = os.environ.get("OPENCODE_SERVER_PASSWORD")
    if not password:
        return None
    username = os.environ.get("OPENCODE_SERVER_USERNAME", "opencode")
    credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {credentials}"}


def _parse_model_string(model: str) -> Dict[str, str]:
    if "/" in model:
        provider_id, model_id = model.split("/", 1)
        return {"providerID": provider_id, "modelID": model_id}
    return {"providerID": "opencode-go", "modelID": model}


def _extract_prompt_text(data: Dict[str, Any]) -> str:
    """Extract final text from a classic OpenCode prompt response."""
    if not isinstance(data, dict):
        raise RuntimeError("OpenCode response malformed: expected object")

    info = data.get("info")
    if not isinstance(info, dict):
        raise RuntimeError("OpenCode response malformed: info must be an object")

    parts = data.get("parts")
    if not isinstance(parts, list):
        raise RuntimeError("OpenCode response malformed: parts must be a list")
    if any(not isinstance(part, dict) for part in parts):
        raise RuntimeError("OpenCode response malformed: parts must contain objects")

    error = info.get("error") if "error" in info else data.get("error")
    if "error" in info or "error" in data:
        error_type = ""
        message = ""
        if isinstance(error, dict):
            raw_type = error.get("type")
            raw_name = error.get("name")
            if isinstance(raw_type, str) and raw_type.strip():
                error_type = raw_type.strip()[:100]
            elif isinstance(raw_name, str) and raw_name.strip():
                error_type = raw_name.strip()[:100]

            raw_message = error.get("message")
            raw_data = error.get("data")
            if isinstance(raw_message, str):
                message = raw_message
            elif isinstance(raw_data, dict) and isinstance(raw_data.get("message"), str):
                message = raw_data["message"]
            elif isinstance(raw_data, str):
                message = raw_data
        elif isinstance(error, str):
            message = error

        error_type = error_type or "unknown"
        message = message.strip()[:500] or "unspecified provider error"
        raise RuntimeError(f"OpenCode provider error: {error_type}: {message}")

    if "structured" in info:
        structured = info["structured"]
        if not isinstance(structured, (dict, list)):
            raise RuntimeError("OpenCode response malformed: structured must be an object or list")
        return json.dumps(structured, ensure_ascii=False, separators=(",", ":"))

    text_parts = [
        part["text"]
        for part in parts
        if part.get("type") == "text"
        and isinstance(part.get("text"), str)
        and part["text"].strip()
    ]
    if text_parts:
        return "".join(text_parts).strip()

    part_types = sorted({part["type"] for part in parts if isinstance(part.get("type"), str)})
    finish = info.get("finish")
    safe_finish = finish.strip()[:100] if isinstance(finish, str) and finish.strip() else "unknown"
    raise RuntimeError(
        f"OpenCode response contained no text: finish={safe_finish!r}, part_types={part_types!r}"
    )


def _optional_nonnegative_int(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"OpenCode response malformed: usage {name} must be a nonnegative integer")
    return value


def _optional_nonnegative_float(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise RuntimeError(f"OpenCode response malformed: usage {name} must be a nonnegative finite number")
    return float(value)


def _optional_string(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError(f"OpenCode response malformed: usage {name} must be a string")
    return value


def _extract_prompt_result(data: Dict[str, Any]) -> OpenCodePromptResult:
    text = _extract_prompt_text(data)
    info = data["info"]
    tokens = info.get("tokens")
    usage_keys = {"tokens", "cost"}
    if not any(key in info for key in usage_keys):
        return OpenCodePromptResult(text=text, usage=None)
    if "tokens" in info and not isinstance(tokens, dict):
        raise RuntimeError("OpenCode response malformed: usage tokens must be an object")
    tokens = tokens or {}
    cache = tokens.get("cache")
    if "cache" in tokens and not isinstance(cache, dict):
        raise RuntimeError("OpenCode response malformed: usage cache must be an object")
    for key in ("input", "output", "reasoning", "total"):
        if key in tokens and tokens[key] is None:
            raise RuntimeError(f"OpenCode response malformed: usage {key} must not be null")
    if isinstance(cache, dict):
        for key in ("read", "write"):
            if key in cache and cache[key] is None:
                raise RuntimeError(f"OpenCode response malformed: usage cache.{key} must not be null")
    for key in ("cost", "providerID", "modelID", "finish"):
        if key in info and info[key] is None:
            raise RuntimeError(f"OpenCode response malformed: usage {key} must not be null")
    usage = OpenCodePromptUsage(
        input_tokens=_optional_nonnegative_int(tokens.get("input"), "input_tokens"),
        output_tokens=_optional_nonnegative_int(tokens.get("output"), "output_tokens"),
        reasoning_tokens=_optional_nonnegative_int(tokens.get("reasoning"), "reasoning_tokens"),
        total_tokens=_optional_nonnegative_int(tokens.get("total"), "total_tokens"),
        cache_read_tokens=_optional_nonnegative_int(cache.get("read") if cache is not None else None, "cache_read_tokens"),
        cache_write_tokens=_optional_nonnegative_int(cache.get("write") if cache is not None else None, "cache_write_tokens"),
        cost=_optional_nonnegative_float(info.get("cost"), "cost"),
        provider_id=_optional_string(info.get("providerID"), "provider_id"),
        model_id=_optional_string(info.get("modelID"), "model_id"),
        finish=_optional_string(info.get("finish"), "finish"),
    )
    return OpenCodePromptResult(text=text, usage=usage)


async def _http_post(base_url: str, path: str, body: Dict[str, Any], timeout: float = 120.0, auth_header: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{base_url}{path}", json=body, headers=auth_header or {}, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status} from {path}: {(await resp.text())[:500]}")
            return await resp.json()


async def _http_get(base_url: str, path: str, timeout: float = 10.0, auth_header: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}{path}", headers=auth_header or {}, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status} from {path}: {(await resp.text())[:500]}")
            return await resp.json()


async def _http_delete(base_url: str, path: str, timeout: float = 10.0, auth_header: Optional[Dict[str, str]] = None) -> None:
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(f"{base_url}{path}", headers=auth_header or {}, timeout=aiohttp.ClientTimeout(total=timeout)):
                pass
    except Exception:
        pass


async def create_session(base_url: str, title: str = "session", auth_header: Optional[Dict[str, str]] = None) -> str:
    return (await _http_post(base_url, "/session", {"title": title}, auth_header=auth_header))["id"]


async def send_prompt_result(base_url: str, session_id: str, system_prompt: str, user_prompt: str, model: str, reasoning_effort: Optional[str] = None, auth_header: Optional[Dict[str, str]] = None, *, output_schema: Optional[Dict[str, Any]] = None) -> Optional[OpenCodePromptResult]:
    body: Dict[str, Any] = {"parts": [{"type": "text", "text": user_prompt}], "model": _parse_model_string(model), "system": system_prompt}
    if output_schema is not None:
        body["format"] = {"type": "json_schema", "schema": copy.deepcopy(output_schema), "retryCount": 0}
    try:
        data = await _http_post(base_url, f"/session/{session_id}/message", body, timeout=180.0, auth_header=auth_header)
        return _extract_prompt_result(data)
    except Exception as exc:
        print(f"  HTTP error: {exc}", file=sys.stderr)
        return None


async def send_prompt(base_url: str, session_id: str, system_prompt: str, user_prompt: str, model: str, reasoning_effort: Optional[str] = None, auth_header: Optional[Dict[str, str]] = None) -> Optional[str]:
    result = await send_prompt_result(base_url, session_id, system_prompt, user_prompt, model, reasoning_effort, auth_header)
    return result.text if result is not None else None


async def delete_session(base_url: str, session_id: str, auth_header: Optional[Dict[str, str]] = None) -> None:
    await _http_delete(base_url, f"/session/{session_id}", auth_header=auth_header)
