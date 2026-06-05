"""Subtitle shortening using OpenCode SDK via HTTP server.

Starts an opencode server, sends prompts via HTTP API, and parses responses.

Usage:
  uv run -m utils.shorten_subtitles_opencode input.json --model deepseek-v4-flash
  uv run -m utils.shorten_subtitles_opencode input.json --model opencode-go/deepseek-v4-flash --concurrency 3
  uv run -m utils.shorten_subtitles_opencode input.json --server-url http://localhost:4096
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.shorten_helpers import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    ShortenFunc,
    ShortenResult,
    add_common_args,
    build_context,
    build_user_prompt,
    load_json,
    parse_shortened_response,
    run_iterative_shortening,
    validate_shortened_text,
)
from utils.analyze_text import join_text_lines


# ---------------------------------------------------------------------------
# OpenCode Server management
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_server_env() -> Dict[str, str]:
    """Create a clean environment for the opencode server.

    Removes OPENCODE_SERVER_PASSWORD so the server doesn't require auth.
    """
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
        self._process: Optional[subprocess.Popen] = None

    def start(self, timeout: float = 30.0) -> None:
        """Start the opencode server and wait until it's ready."""
        cmd = [
            "opencode", "serve",
            "--port", str(self.port),
            "--hostname", self.hostname,
        ]
        print(f"Starting opencode server on {self.base_url} ...")

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_make_server_env(),
        )

        # Wait for server to become healthy
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                raise RuntimeError(
                    f"opencode serve exited with code {self._process.returncode}. "
                    f"stderr: {stderr[:500]}"
                )
            if self._is_healthy():
                print(f"Server ready (pid={self._process.pid})")
                return
            time.sleep(0.5)

        self.stop()
        raise TimeoutError(f"opencode server did not become healthy within {timeout}s")

    def stop(self) -> None:
        """Stop the opencode server."""
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
        """Check if the server is healthy via HTTP."""
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(f"{self.base_url}/global/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                return data.get("healthy") is True
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return False

    def __enter__(self) -> "OpenCodeServer":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _get_auth_header() -> Optional[Dict[str, str]]:
    """Build Basic auth header from environment variables.

    Returns None if no password is set.
    """
    password = os.environ.get("OPENCODE_SERVER_PASSWORD")
    if not password:
        return None
    username = os.environ.get("OPENCODE_SERVER_USERNAME", "opencode")
    credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {credentials}"}


# ---------------------------------------------------------------------------
# HTTP API helpers
# ---------------------------------------------------------------------------


def _parse_model_string(model: str) -> Dict[str, str]:
    """Parse 'provider/model' or just 'model' into providerID/modelID.

    If only model name is given (no slash), tries to infer provider:
    - Models starting with 'deepseek' -> 'deepseek'
    - Otherwise -> 'opencode-go'
    """
    if "/" in model:
        provider_id, model_id = model.split("/", 1)
        return {"providerID": provider_id, "modelID": model_id}

    # Infer provider from model name
    if model.startswith("deepseek"):
        return {"providerID": "opencode-go", "modelID": model}
    return {"providerID": "opencode-go", "modelID": model}


async def _http_post(
    base_url: str,
    path: str,
    body: Dict[str, Any],
    timeout: float = 120.0,
    auth_header: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Make an async HTTP POST request."""
    import aiohttp

    url = f"{base_url}{path}"
    headers = auth_header or {}
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=body, headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status} from {path}: {text[:500]}")
            return await resp.json()


async def _http_get(
    base_url: str,
    path: str,
    timeout: float = 10.0,
    auth_header: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Make an async HTTP GET request."""
    import aiohttp

    url = f"{base_url}{path}"
    headers = auth_header or {}
    async with aiohttp.ClientSession() as session:
        async with session.get(
            url, headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status} from {path}: {text[:500]}")
            return await resp.json()


async def _http_delete(
    base_url: str,
    path: str,
    timeout: float = 10.0,
    auth_header: Optional[Dict[str, str]] = None,
) -> None:
    """Make an async HTTP DELETE request."""
    import aiohttp

    url = f"{base_url}{path}"
    headers = auth_header or {}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                url, headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                pass  # Ignore errors on cleanup
    except Exception:
        pass


# ---------------------------------------------------------------------------
# OpenCode API calls
# ---------------------------------------------------------------------------


async def create_session(
    base_url: str,
    title: str = "shorten",
    auth_header: Optional[Dict[str, str]] = None,
) -> str:
    """Create a new opencode session and return its ID."""
    data = await _http_post(base_url, "/session", {"title": title}, auth_header=auth_header)
    return data["id"]


async def send_prompt(
    base_url: str,
    session_id: str,
    system_prompt: str,
    user_prompt: str,
    model: str,
    reasoning_effort: Optional[str] = None,
    auth_header: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Send a prompt to an opencode session and return the text response.

    Returns None if no text response was received.
    """
    model_spec = _parse_model_string(model)

    body: Dict[str, Any] = {
        "parts": [{"type": "text", "text": user_prompt}],
        "model": model_spec,
        "system": system_prompt,
    }

    try:
        data = await _http_post(
            base_url,
            f"/session/{session_id}/message",
            body,
            timeout=180.0,
            auth_header=auth_header,
        )
    except Exception as e:
        print(f"  HTTP error: {e}", file=sys.stderr)
        return None

    # Extract text from response parts
    parts = data.get("parts", [])
    text_parts = [p for p in parts if p.get("type") == "text"]
    if not text_parts:
        return None

    # Concatenate all text parts
    return "".join(p.get("text", "") for p in text_parts).strip()


async def delete_session(
    base_url: str,
    session_id: str,
    auth_header: Optional[Dict[str, str]] = None,
) -> None:
    """Delete an opencode session."""
    await _http_delete(base_url, f"/session/{session_id}", auth_header=auth_header)


# ---------------------------------------------------------------------------
# Shorten implementation
# ---------------------------------------------------------------------------


async def _shorten_one(
    base_url: str,
    items: List[Dict[str, Any]],
    target_idx: int,
    position: int,
    total: int,
    model: str,
    context_window: int,
    min_words: int,
    system_prompt: str,
    reasoning_effort: Optional[str] = None,
    auth_header: Optional[Dict[str, str]] = None,
) -> ShortenResult:
    """Shorten a single subtitle via opencode."""
    item = items[target_idx]
    original_text = join_text_lines(item.get("text", ""))
    print(f"  [{position}/{total}] Subtitle #{item.get('index', target_idx+1)}: {original_text[:50]}...")

    context = build_context(items, target_idx, context_window)
    user_prompt = build_user_prompt(context, min_words)

    # Create a dedicated session for this subtitle
    session_id = await create_session(base_url, title=f"shorten-{target_idx}", auth_header=auth_header)
    try:
        response = await send_prompt(
            base_url, session_id, system_prompt, user_prompt,
            model, reasoning_effort, auth_header=auth_header,
        )
        shortened = parse_shortened_response(response)

        if shortened and validate_shortened_text(original_text, shortened, min_words):
            print(f"    -> Shortened: {shortened[:50]}...")
            return ShortenResult(
                index=target_idx,
                original_text=original_text,
                shortened_text=shortened,
                success=True,
            )
        else:
            print(f"    -> Failed, keeping original")
            return ShortenResult(
                index=target_idx,
                original_text=original_text,
                shortened_text=original_text,
                success=False,
                error="Failed to get valid shortened text",
            )
    finally:
        await delete_session(base_url, session_id, auth_header=auth_header)


async def shorten_subtitles_opencode(
    items: List[Dict[str, Any]],
    target_indices: List[int],
    model: str,
    context_window: int,
    min_words: int,
    base_url: str,
    concurrency: int,
    reasoning_effort: Optional[str] = None,
    auth_header: Optional[Dict[str, str]] = None,
) -> List[ShortenResult]:
    """Shorten subtitles in parallel via opencode HTTP API."""
    system_prompt = SYSTEM_PROMPT.format(min_words=min_words)
    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(target_idx: int, position: int) -> ShortenResult:
        async with semaphore:
            return await _shorten_one(
                base_url, items, target_idx, position, len(target_indices),
                model, context_window, min_words, system_prompt,
                reasoning_effort, auth_header=auth_header,
            )

    tasks = [process_one(idx, i + 1) for i, idx in enumerate(target_indices)]
    results = await asyncio.gather(*tasks)
    return list(results)


# ---------------------------------------------------------------------------
# ShortenFunc adapter
# ---------------------------------------------------------------------------


def make_opencode_shorten_func(
    base_url: str,
    concurrency: int,
    auth_header: Optional[Dict[str, str]] = None,
) -> ShortenFunc:
    """Create a ShortenFunc adapter for opencode HTTP API."""

    def _shorten(
        items: List[Dict[str, Any]],
        target_indices: List[int],
        model: str,
        context_window: int,
        min_words: int,
        reasoning_effort: Optional[str] = None,
    ) -> List[ShortenResult]:
        return asyncio.run(shorten_subtitles_opencode(
            items, target_indices, model, context_window, min_words,
            base_url, concurrency, reasoning_effort, auth_header=auth_header,
        ))

    return _shorten


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Iteratively shorten subtitles using OpenCode HTTP API."
    )
    add_common_args(parser)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max parallel API calls (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for opencode server (default: random free port)",
    )
    parser.add_argument(
        "--hostname",
        default="127.0.0.1",
        help="Hostname for opencode server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="Use existing opencode server URL instead of starting a new one",
    )

    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input
    print(f"Loading: {input_path}")
    items = load_json(input_path)
    print(f"Total subtitles: {len(items)}")

    stem = input_path.stem

    if args.server_url:
        # Use existing server — need auth header since server may have password
        base_url = args.server_url.rstrip("/")
        auth_header = _get_auth_header()
        print(f"Using existing server: {base_url}")
        shorten_func = make_opencode_shorten_func(base_url, args.concurrency, auth_header)
        run_iterative_shortening(
            items=items,
            shorten_func=shorten_func,
            model=args.model,
            threshold=args.threshold,
            context_window=args.context_window,
            max_iterations=args.max_iterations,
            min_words=args.min_words,
            output_dir=output_dir,
            stem=stem,
            reasoning_effort=args.reasoning_effort,
        )
    else:
        # Start our own server — no auth needed (clean env)
        server = OpenCodeServer(port=args.port, hostname=args.hostname)
        try:
            server.start()
            shorten_func = make_opencode_shorten_func(server.base_url, args.concurrency)
            run_iterative_shortening(
                items=items,
                shorten_func=shorten_func,
                model=args.model,
                threshold=args.threshold,
                context_window=args.context_window,
                max_iterations=args.max_iterations,
                min_words=args.min_words,
                output_dir=output_dir,
                stem=stem,
                reasoning_effort=args.reasoning_effort,
            )
        finally:
            server.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
