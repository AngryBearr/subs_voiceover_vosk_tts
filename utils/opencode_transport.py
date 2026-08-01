"""Neutral OpenCode HTTP transport and local server lifecycle helpers."""

from __future__ import annotations

import base64
import json
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Optional


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


async def create_session(base_url: str, title: str = "shorten", auth_header: Optional[Dict[str, str]] = None) -> str:
    return (await _http_post(base_url, "/session", {"title": title}, auth_header=auth_header))["id"]


async def send_prompt(base_url: str, session_id: str, system_prompt: str, user_prompt: str, model: str, reasoning_effort: Optional[str] = None, auth_header: Optional[Dict[str, str]] = None) -> Optional[str]:
    body: Dict[str, Any] = {"parts": [{"type": "text", "text": user_prompt}], "model": _parse_model_string(model), "system": system_prompt}
    try:
        data = await _http_post(base_url, f"/session/{session_id}/message", body, timeout=180.0, auth_header=auth_header)
        return _extract_prompt_text(data)
    except Exception as exc:
        print(f"  HTTP error: {exc}", file=sys.stderr)
        return None


async def delete_session(base_url: str, session_id: str, auth_header: Optional[Dict[str, str]] = None) -> None:
    await _http_delete(base_url, f"/session/{session_id}", auth_header=auth_header)
