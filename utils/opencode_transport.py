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
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, Optional

from utils.llm_types import CallRecord, DetailedResponse


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


def _provider_error(data: Dict[str, Any]) -> Optional[str]:
    """Return a useful, redacted provider error without exposing raw metadata."""
    info = data.get("info")
    info = info if isinstance(info, dict) else {}
    if "error" not in info and "error" not in data:
        return None
    error = info.get("error", data.get("error"))
    labels: list[str] = []
    message = ""
    if isinstance(error, dict):
        for key in ("name", "type"):
            value = error.get(key)
            if isinstance(value, str) and value.strip() and value.strip() not in labels:
                labels.append(value.strip()[:100])
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
    label = " / ".join(labels) or "unknown"
    detail = f"{label}: {message.strip()[:500] or 'unspecified provider error'}"
    return _redact_transport_error(detail)


def _extract_prompt_text(data: Dict[str, Any]) -> str:
    """Extract final text from a classic OpenCode prompt response."""
    if not isinstance(data, dict):
        raise RuntimeError("OpenCode response malformed: expected object")

    info = data.get("info")
    if not isinstance(info, dict):
        raise RuntimeError("OpenCode response malformed: info must be an object")

    provider_error = _provider_error(data)
    if provider_error is not None:
        raise RuntimeError(provider_error)

    structured_value = info.get("structured", info.get("structured_output", data.get("structured_output")))
    if structured_value is not None:
        structured = structured_value
        if not isinstance(structured, (dict, list)):
            raise RuntimeError("OpenCode response malformed: structured must be an object or list")
        return json.dumps(structured, ensure_ascii=False, separators=(",", ":"))

    parts = data.get("parts")
    if not isinstance(parts, list):
        raise RuntimeError("OpenCode response malformed: parts must be a list")
    if any(not isinstance(part, dict) for part in parts):
        raise RuntimeError("OpenCode response malformed: parts must contain objects")

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
        session_id = path[len("/session/"):-len("/message")] if path.startswith("/session/") and path.endswith("/message") else ""
        if session_id and "/" not in session_id:
            from utils.shortening_html import note_model_request
            note_model_request()
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
    if reasoning_effort is not None and (not isinstance(reasoning_effort, str) or not reasoning_effort.strip()):
        return None
    body: Dict[str, Any] = {"parts": [{"type": "text", "text": user_prompt}], "model": _parse_model_string(model), "system": system_prompt}
    if reasoning_effort is not None:
        body["variant"] = reasoning_effort
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _redact_transport_error(value: Any) -> str:
    import re
    message = value if isinstance(value, str) else "request failed"
    message = " ".join(message.split())
    message = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", message, flags=re.IGNORECASE)
    message = re.sub(r"(?:authorization|api[_ -]?key)\s*[:=]\s*[^\s,;]+", "[redacted-credential]", message, flags=re.IGNORECASE)
    return message[:300] or "request failed"


def _usage_from_info(info: Any) -> Optional[OpenCodePromptUsage]:
    if not isinstance(info, dict):
        return None
    tokens = info.get("tokens")
    if tokens is not None and not isinstance(tokens, dict):
        return None
    cache = tokens.get("cache") if isinstance(tokens, dict) else None
    if cache is not None and not isinstance(cache, dict):
        cache = None

    def integer(value: Any) -> Optional[int]:
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    usage_present = isinstance(tokens, dict) or "cost" in info
    if not usage_present:
        return None
    parsed_values = [
        integer(tokens.get("input") if isinstance(tokens, dict) else None),
        integer(tokens.get("output") if isinstance(tokens, dict) else None),
        integer(tokens.get("reasoning") if isinstance(tokens, dict) else None),
        integer(tokens.get("total") if isinstance(tokens, dict) else None),
        integer(cache.get("read") if isinstance(cache, dict) else None),
        integer(cache.get("write") if isinstance(cache, dict) else None),
        info.get("cost") if isinstance(info.get("cost"), (int, float)) and not isinstance(info.get("cost"), bool) and math.isfinite(float(info["cost"])) and info["cost"] >= 0 else None,
    ]
    if not any(value is not None for value in parsed_values):
        return None
    return OpenCodePromptUsage(
        input_tokens=integer(tokens.get("input") if isinstance(tokens, dict) else None),
        output_tokens=integer(tokens.get("output") if isinstance(tokens, dict) else None),
        reasoning_tokens=integer(tokens.get("reasoning") if isinstance(tokens, dict) else None),
        total_tokens=integer(tokens.get("total") if isinstance(tokens, dict) else None),
        cache_read_tokens=integer(cache.get("read") if isinstance(cache, dict) else None),
        cache_write_tokens=integer(cache.get("write") if isinstance(cache, dict) else None),
        cost=info.get("cost") if isinstance(info.get("cost"), (int, float)) and not isinstance(info.get("cost"), bool) and math.isfinite(float(info["cost"])) and info["cost"] >= 0 else None,
        provider_id=info.get("providerID") if isinstance(info.get("providerID"), str) else None,
        model_id=info.get("modelID") if isinstance(info.get("modelID"), str) else None,
        finish=info.get("finish") if isinstance(info.get("finish"), str) else None,
    )


def _usage_from_entry(entry: Dict[str, Any]) -> Optional[OpenCodePromptUsage]:
    """Prefer the assistant summary; otherwise recover one step-finish usage."""
    info_usage = _usage_from_info(entry.get("info"))
    if info_usage is not None:
        return info_usage
    parts = entry.get("parts")
    if not isinstance(parts, list):
        return None
    for part in reversed(parts):
        if not isinstance(part, dict) or part.get("type") != "step-finish" or not isinstance(part.get("tokens"), dict):
            continue
        tokens = part["tokens"]

        def integer(value: Any) -> Optional[int]:
            return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

        input_tokens = integer(tokens.get("input"))
        output_tokens = integer(tokens.get("output"))
        reasoning_tokens = integer(tokens.get("reasoning"))
        total_tokens = integer(tokens.get("total"))
        if input_tokens is None and output_tokens is None and reasoning_tokens is None and total_tokens is None:
            continue
        return OpenCodePromptUsage(input_tokens=input_tokens, output_tokens=output_tokens,
                                   reasoning_tokens=reasoning_tokens, total_tokens=total_tokens,
                                   provider_id=entry["info"].get("providerID") if isinstance(entry.get("info"), dict) else None,
                                   model_id=entry["info"].get("modelID") if isinstance(entry.get("info"), dict) else None,
                                   finish="step-finish")
    return None


def _assistant_entries(messages: Any) -> list[Dict[str, Any]]:
    if isinstance(messages, dict):
        messages = messages.get("messages", messages.get("data", []))
    if not isinstance(messages, list):
        return []
    return [item for item in messages if isinstance(item, dict) and isinstance(item.get("info"), dict)
            and item["info"].get("role", "assistant") == "assistant"]


def _entry_identity(entry: Dict[str, Any]) -> str:
    info = entry.get("info")
    if not isinstance(info, dict):
        info = {}
    identity = entry.get("id") or info.get("id")
    return str(identity) if identity is not None else json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str)


def _entry_has_error(entry: Dict[str, Any]) -> bool:
    info = entry.get("info")
    return (isinstance(info, dict) and "error" in info) or "error" in entry


def _entry_is_terminal(entry: Dict[str, Any]) -> bool:
    """Identify terminal response entries while ignoring tool-only history items."""
    if _entry_has_error(entry):
        return True
    info = entry.get("info")
    if isinstance(info, dict):
        if any(key in info for key in ("structured", "structured_output", "finish")):
            return True
    parts = entry.get("parts")
    if not isinstance(parts, list):
        return False
    return any(
        isinstance(part, dict)
        and (
            (part.get("type") == "text" and isinstance(part.get("text"), str) and bool(part["text"].strip()))
            or part.get("type") == "step-finish"
        )
        for part in parts
    )


def _final_entry(entries: list[Dict[str, Any]], post_data: Any, fetched: Any = None) -> Optional[Dict[str, Any]]:
    """Select one terminal response and never fall back to an earlier text part."""
    terminal = [entry for entry in entries if _entry_is_terminal(entry)]
    if not terminal:
        return None
    post_entries = _assistant_entries([post_data]) if isinstance(post_data, dict) else []
    post_ids = {_entry_identity(entry) for entry in post_entries}
    fetched_entries = _assistant_entries(fetched)
    fetched_ids = {_entry_identity(entry) for entry in fetched_entries}
    post_ids &= fetched_ids
    exact_post = next((entry for entry in terminal if _entry_identity(entry) in post_ids), None)
    if exact_post is not None:
        exact_position = terminal.index(exact_post)
        if exact_position == len(terminal) - 1:
            return exact_post
    fetched_terminal = [entry for entry in fetched_entries if _entry_is_terminal(entry) and _entry_identity(entry) in {_entry_identity(item) for item in terminal}]
    return fetched_terminal[-1] if fetched_terminal else terminal[-1]


def _user_entries(messages: Any) -> list[Dict[str, Any]]:
    if isinstance(messages, dict):
        messages = messages.get("messages", messages.get("data", []))
    if not isinstance(messages, list):
        return []
    return [item for item in messages if isinstance(item, dict) and isinstance(item.get("info"), dict)
            and item["info"].get("role") == "user"]


def _current_user_entries(
    fetched: Any,
    post_data: Any,
    baseline_user_ids: set[str],
    current_parent_ids: set[str],
) -> list[Dict[str, Any]]:
    candidates = _user_entries(fetched)
    if isinstance(post_data, dict):
        candidates += _user_entries([post_data])
    selected: dict[str, Dict[str, Any]] = {}
    for entry in candidates:
        identity = _entry_identity(entry)
        if identity not in baseline_user_ids and (not current_parent_ids or identity in current_parent_ids):
            selected.setdefault(identity, entry)
    return list(selected.values())


def _current_assistant_entries(
    fetched: Any,
    post_data: Any,
    baseline_assistant_ids: set[str],
    baseline_user_ids: set[str],
) -> list[Dict[str, Any]]:
    """Keep only assistant steps belonging to the message submitted by this call."""
    fetched_assistants = _assistant_entries(fetched)
    post_assistants = _assistant_entries([post_data]) if isinstance(post_data, dict) else []
    current_parent_ids: set[str] = set()
    for entry in post_assistants:
        parent = entry.get("info", {}).get("parentID")
        if parent is not None:
            current_parent_ids.add(str(parent))
    for entry in _user_entries(fetched):
        identity = _entry_identity(entry)
        if identity not in baseline_user_ids:
            current_parent_ids.add(identity)
    post_ids = {_entry_identity(entry) for entry in post_assistants}
    selected: dict[str, Dict[str, Any]] = {}
    for entry in fetched_assistants + post_assistants:
        identity = _entry_identity(entry)
        info = entry.get("info", {})
        parent = info.get("parentID")
        belongs_to_prompt = (
            identity in post_ids
            or identity not in baseline_assistant_ids
            and (not current_parent_ids or (parent is not None and str(parent) in current_parent_ids))
        )
        if belongs_to_prompt:
            selected.setdefault(identity, entry)
    # Older OpenCode responses omitted parentID.  If the POST response is also
    # present in the fetched history, retain following assistant steps from
    # that response group rather than treating them as an older invocation.
    if not current_parent_ids and post_ids & baseline_assistant_ids:
        identities = [_entry_identity(entry) for entry in fetched_assistants]
        first = next((identities.index(identity) for identity in post_ids if identity in identities), None)
        if first is not None:
            for entry in fetched_assistants[first:]:
                selected.setdefault(_entry_identity(entry), entry)
    return list(selected.values())


async def send_prompt_detailed(
    base_url: str, session_id: str, system_prompt: str, user_prompt: str, model: str,
    reasoning_effort: str, auth_header: Optional[Dict[str, str]] = None, *,
    output_schema: Optional[Dict[str, Any]] = None, allow_length_tool: bool = False,
    timeout: float = 300.0, native_structured_output: bool = False,
) -> DetailedResponse:
    """Send one isolated subtitle-agent message and account for every assistant step."""
    started = _utc_now()
    clock = time.monotonic()
    calls: list[CallRecord] = []
    post_started = False
    post_succeeded = False
    baseline_assistant_ids: set[str] = set()
    baseline_user_ids: set[str] = set()
    try:
        if not isinstance(reasoning_effort, str) or not reasoning_effort.strip():
            raise ValueError("invalid_reasoning_effort")
        if not isinstance(model, str) or not model.strip() or "/" not in model:
            raise ValueError("model_required")
        if not isinstance(session_id, str) or not session_id.strip() or not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("session_required")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or timeout <= 0:
            raise ValueError("invalid_timeout")
        if output_schema is not None and not isinstance(output_schema, dict):
            raise ValueError("invalid_output_schema")
        if not isinstance(native_structured_output, bool):
            raise ValueError("invalid_native_structured_output")
        requested_provider, requested_model = _parse_model_string(model).values()
        try:
            baseline = await _http_get(base_url, f"/session/{session_id}/message", timeout=float(timeout), auth_header=auth_header)
        except Exception:
            baseline = []
        baseline_assistant_ids = {_entry_identity(entry) for entry in _assistant_entries(baseline)}
        baseline_user_ids = {_entry_identity(entry) for entry in _user_entries(baseline)}
        effective_system = system_prompt
        if output_schema is not None and not native_structured_output:
            compact_schema = json.dumps(output_schema, ensure_ascii=False, separators=(",", ":"))
            if compact_schema not in system_prompt and compact_schema not in user_prompt:
                effective_system = f"{system_prompt}\n\nReturn JSON matching this schema: {compact_schema}"
        body: Dict[str, Any] = {
            "parts": [{"type": "text", "text": user_prompt}], "model": _parse_model_string(model),
            "variant": reasoning_effort, "agent": "subtitle", "system": effective_system,
            "tools": {"length_check_length": bool(allow_length_tool)},
        }
        if native_structured_output and output_schema is not None:
            body["format"] = {"type": "json_schema", "schema": copy.deepcopy(output_schema), "retryCount": 0}
        post_started = True
        post_data = await _http_post(base_url, f"/session/{session_id}/message", body, timeout=float(timeout), auth_header=auth_header)
        post_succeeded = True
        try:
            fetched = await _http_get(base_url, f"/session/{session_id}/message", timeout=float(timeout), auth_header=auth_header)
        except Exception:
            fetched = []
        entries = _current_assistant_entries(fetched, post_data, baseline_assistant_ids, baseline_user_ids)
        if not entries and isinstance(post_data, dict):
            entries = _assistant_entries([post_data])
        unique: dict[str, Dict[str, Any]] = {}
        for entry in entries:
            unique.setdefault(_entry_identity(entry), entry)
        current_parent_ids = {
            str(entry["info"]["parentID"])
            for entry in entries
            if entry["info"].get("parentID") is not None
        }
        current_users = _current_user_entries(fetched, post_data, baseline_user_ids, current_parent_ids)
        user_metadata_mismatch = any(
            (isinstance(entry.get("info", {}).get("modelID"), str)
             and entry["info"]["modelID"] != requested_model)
            or (isinstance(entry.get("info", {}).get("providerID"), str)
                and entry["info"]["providerID"] != requested_provider)
            or (isinstance(entry.get("info", {}).get("variant"), str)
                and entry["info"]["variant"] != reasoning_effort)
            for entry in current_users
        )
        for entry in unique.values():
            info = entry["info"]
            actual_model = info.get("modelID") if isinstance(info.get("modelID"), str) else None
            actual_provider = info.get("providerID") if isinstance(info.get("providerID"), str) else None
            actual_variant = info.get("variant") if isinstance(info.get("variant"), str) else None
            mismatch = ((actual_model is not None and actual_model != requested_model)
                        or (actual_provider is not None and actual_provider != requested_provider)
                        or (actual_variant is not None and actual_variant != reasoning_effort)
                        or user_metadata_mismatch)
            has_error = "error" in info or "error" in entry
            provider_error = _provider_error(entry) if has_error else None
            error = provider_error or ("model_mismatch" if mismatch else None)
            status = "error" if has_error else ("mismatch" if mismatch else "success")
            parts = entry.get("parts")
            tool_count = sum(1 for part in parts if isinstance(part, dict) and part.get("type") in {"tool", "tool-call"}) if isinstance(parts, list) else 0
            calls.append(CallRecord(actual_provider or "unknown", actual_model or "unknown", actual_variant,
                                    started, time.monotonic() - clock, _usage_from_entry(entry), status,
                                    tool_calls=tool_count, error=error, requested_model=model,
                                    requested_variant=reasoning_effort))
        final_entry = _final_entry(list(unique.values()), post_data, fetched)
        text: Optional[str] = None
        final_error: Optional[str] = None
        if final_entry is not None:
            final_error = _provider_error(final_entry)
            if final_error is None:
                try:
                    text = _extract_prompt_text(final_entry)
                except Exception as exc:
                    final_error = _redact_transport_error(str(exc))
        if text is None and final_error is None:
            final_error = "response contained no final text"
        call_error = next((record.error for record in calls if record.status in {"error", "mismatch"} and record.error), None)
        response_error = final_error or call_error
        return DetailedResponse(text, calls, response_error)
    except Exception as exc:
        if post_started and not post_succeeded:
            # Abort before fetching usage for every interrupted or failed POST.
            try:
                await _http_post(base_url, f"/session/{session_id}/abort", {}, timeout=10.0, auth_header=auth_header)
            except Exception:
                pass
            try:
                fetched = await _http_get(base_url, f"/session/{session_id}/message", timeout=10.0, auth_header=auth_header)
                recovered = _current_assistant_entries(fetched, None, baseline_assistant_ids, baseline_user_ids)
                requested_provider, requested_model = _parse_model_string(model).values()
                for entry in recovered:
                    info = entry["info"]
                    actual_model = info.get("modelID") if isinstance(info.get("modelID"), str) else None
                    actual_provider = info.get("providerID") if isinstance(info.get("providerID"), str) else None
                    actual_variant = info.get("variant") if isinstance(info.get("variant"), str) else None
                    mismatch = ((actual_model is not None and actual_model != requested_model)
                                or (actual_provider is not None and actual_provider != requested_provider)
                                or (actual_variant is not None and actual_variant != reasoning_effort))
                    entry_error = _provider_error(entry)
                    calls.append(CallRecord(
                        actual_provider or "unknown", actual_model or "unknown", actual_variant,
                        started, time.monotonic() - clock, _usage_from_entry(entry),
                        "error" if entry_error else ("mismatch" if mismatch else "recovered"),
                        error=entry_error or ("model_mismatch" if mismatch else None), requested_model=model,
                        requested_variant=reasoning_effort,
                    ))
            except Exception:
                pass
        response_error = _redact_transport_error(str(exc))
        if not calls:
            calls.append(CallRecord("opencode", model if isinstance(model, str) else "unknown", reasoning_effort if isinstance(reasoning_effort, str) else None,
                                    started, time.monotonic() - clock, None, "error", error=response_error))
        return DetailedResponse(None, calls, response_error)


async def delete_session(base_url: str, session_id: str, auth_header: Optional[Dict[str, str]] = None) -> None:
    await _http_delete(base_url, f"/session/{session_id}", auth_header=auth_header)
