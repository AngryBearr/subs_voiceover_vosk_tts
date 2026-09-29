from __future__ import annotations

import asyncio
import json
import inspect
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from utils import deepseek_transport, opencode_transport, sh08_matrix, sh08_matrix_transport
from utils import shorten_file, shorten_subtitles, shortening_experiment, shortening_html, shortening_replay


class _Content:
    def __init__(self, body: bytes) -> None:
        self.body = body

    async def iter_chunked(self, _: int) -> AsyncIterator[bytes]:
        yield self.body


class _Response:
    def __init__(self, body: bytes) -> None:
        self.status = 200
        self.content = _Content(body)

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def read(self) -> bytes:
        return self.content.body

    async def text(self) -> str:
        return self.content.body.decode()

    async def json(self) -> dict[str, Any]:
        return json.loads(self.content.body)


class _Session:
    def __init__(self, body: bytes, calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]]) -> None:
        self.body = body
        self.calls = calls

    async def __aenter__(self) -> "_Session":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def post(self, *args: Any, **kwargs: Any) -> _Response:
        self.calls.append(("post", args, kwargs))
        return _Response(self.body)


def _aiohttp(monkeypatch: pytest.MonkeyPatch, body: bytes = b"{}") -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setitem(sys.modules, "aiohttp", SimpleNamespace(
        ClientSession=lambda: _Session(body, calls),
        ClientTimeout=lambda **kwargs: kwargs,
    ))
    return calls


def _trial_artifact(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps({
        "status": "partial",
        "records": [{"cue_id": "cue-1", "original": "original", "status": "pending"}],
    }), encoding="utf-8")


def test_live_scope_publishes_on_real_request_and_not_outside(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    _trial_artifact(output_dir)

    shortening_html.note_model_request()
    assert not (output_dir / "shortening.html").exists()
    with shortening_html.live_html_report(output_dir):
        pass
    assert not (output_dir / "shortening.html").exists()

    with shortening_html.live_html_report(output_dir):
        shortening_html.note_model_request()
    report = output_dir / "shortening.html"
    assert report.is_file()
    assert "<!doctype html>" in report.read_text(encoding="utf-8").lower()


def test_live_scope_publishes_partial_report_without_masking_error(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    _trial_artifact(output_dir)
    with pytest.raises(RuntimeError, match="original failure"):
        with shortening_html.live_html_report(output_dir):
            shortening_html.note_model_request()
            raise RuntimeError("original failure")
    assert (output_dir / "shortening.html").is_file()


def test_transport_hooks_only_mark_actual_inference_posts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _aiohttp(monkeypatch)
    observed: list[None] = []
    monkeypatch.setattr(shortening_html, "note_model_request", lambda: observed.append(None))

    async def exercise() -> None:
        await deepseek_transport._http_post_json(url="https://example.test", headers={}, body={}, timeout=1)
        await opencode_transport._http_post("http://server", "/session", {}, timeout=1)
        await opencode_transport._http_post("http://server", "/session/id/abort", {}, timeout=1)
        await opencode_transport._http_post("http://server", "/session/id/message", {}, timeout=1)
        await opencode_transport._http_post("http://server", "/session/id/message/extra", {}, timeout=1)
        backend = sh08_matrix_transport.MatrixBackend(openrouter_no_byok=True)
        backend._credentials = {"OPENROUTER_API_KEY": "test"}
        await backend._post_openrouter(headers={}, body={})

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    with shortening_html.live_html_report(output_dir):
        asyncio.run(exercise())

    assert len(observed) == 3
    assert len(calls) == 6


@pytest.mark.parametrize(
    ("module", "function"),
    [
        (sh08_matrix, "_cli"),
        (shorten_file, "execute_run"),
        (shorten_subtitles, "main"),
        (shortening_experiment, "main"),
        (shortening_replay, "main"),
    ],
)
def test_live_html_scope_is_attached_to_each_live_cli_root(module: Any, function: str) -> None:
    assert "live_html_report" in inspect.getsource(getattr(module, function))
