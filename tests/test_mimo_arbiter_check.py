"""Offline checks for the opt-in, single-candidate MiMo experiment."""

from __future__ import annotations

import json
import socket
import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from utils import mimo_arbiter_check as check
from utils.mimo_arbiter_check import IDS, MODEL, _digest, _ordered_context, prepare, report, run
from utils.shortening_prompts import ARBITER_SYSTEM_PROMPT, build_review_prompt


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """Tiny saved-run shape: no ignored output or network required."""
    folder = tmp_path / "source"
    folder.mkdir()
    outcomes = []
    records = []
    for index, cue_id in enumerate(IDS):
        accepted = index < 4
        original = f"Original cue {index}: keep the meaning."
        candidate = "Не сдам" if cue_id == "cue-531" else f"Short {index}"
        judgment = {"cue_id": cue_id, "verdict": "accept" if accepted else "revise",
                    "issues": [] if accepted else ["Meaning changed"], "advice": ""}
        outcomes.append({"cue_id": cue_id, "outcome": "accepted_correction" if accepted else "needs_attention",
                         "max_chars": 20, "length_basis": "normalized_tts_text",
                         "original_context_before": [{"cue_id": "before", "text": "Prior context"}],
                         "original_context_after": [{"cue_id": "after", "text": "Next context"}]})
        records.append({"cue_id": cue_id, "original": original, "candidate": candidate,
                        "final_text": candidate if accepted else original,
                        "history": [{"stage": "generation", "candidate": candidate,
                                     "local": {"actual_chars": len(candidate), "valid": True}},
                                    {"stage": "review", "candidate": candidate, "judgment": judgment}]})
    (folder / "outcomes.json").write_text(json.dumps({"cues": outcomes}))
    (folder / "report.json").write_text(json.dumps({"records": records}))
    return folder


def test_freeze_saved_pairs_without_leaking_sol_into_prompt(tmp_path: Path, source: Path) -> None:
    prepare(source=source, dest=tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["cases"]) == 8
    assert [row["sol"]["verdict"] for row in manifest["cases"]] == ["accept"] * 4 + ["revise"] * 4
    assert manifest["cases"][4]["target"]["candidate"] == "Не сдам"
    for case in manifest["cases"]:
        assert case["target"]["candidate"] != case["historical_final_text"] or case["sol"]["verdict"] == "accept"
        prompt = build_review_prompt(_ordered_context(case["context"]), [case["target"]])
        assert _digest({"system": ARBITER_SYSTEM_PROMPT, "prompt": prompt}) == case["input_sha256"]
        assert "last_model_review" not in prompt and "judgment" not in prompt
    with pytest.raises(FileExistsError):
        prepare(source=source, dest=tmp_path)


def test_fake_success_is_offline_and_does_not_generate_published_artifact(tmp_path: Path, source: Path) -> None:
    prepare(source=source, dest=tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    calls: list[str] = []

    def fake(url: str, body: dict[str, Any] | None, key: str, timeout: float) -> dict[str, Any]:
        assert key == "test-only" and timeout <= 120
        calls.append(url)
        if body is None:
            return {"data": [{"id": MODEL}]}
        assert body["model"] == MODEL and body["max_tokens"] == 4096
        cue_id = manifest["cases"][len(calls) - 2]["cue_id"]
        return {"model": MODEL, "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"results": [
            {"cue_id": cue_id, "verdict": "accept", "issues": [], "advice": ""}]})}}],
            "usage": {"prompt_tokens": 33, "completion_tokens": 12}}

    rows = run(manifest, tmp_path, "https://example.invalid/v1", MODEL, "test-only", fake)
    assert len(rows) == 8 and all(row["status"] == "ok" for row in rows)
    assert len(calls) == 9 and all(row["cost_usd"] is None for row in rows)
    assert rows[0]["started_at"] and rows[0]["elapsed_sec"] >= 0
    assert rows[0]["mode"] == "single_candidate" and rows[0]["returned_model"] == MODEL
    assert "Historical accepted final text" in (tmp_path / "report.html").read_text()
    with pytest.raises(FileExistsError):
        run(manifest, tmp_path, "https://example.invalid/v1", MODEL, "test-only", fake)
    assert len(calls) == 9


@pytest.mark.parametrize("failure", ["invalid", "unauthorized", "timeout", "length", "wrong_model"])
def test_error_stops_without_retry_or_fallback(tmp_path: Path, source: Path, failure: str) -> None:
    manifest = prepare(source=source, dest=tmp_path)
    calls: list[dict[str, Any] | None] = []

    def fake(url: str, body: dict[str, Any] | None, key: str, timeout: float) -> dict[str, Any]:
        calls.append(body)
        if body is None:
            return {"data": [{"id": MODEL}]}
        if failure == "unauthorized":
            raise urllib.error.HTTPError(url, 401, "unauthorized", {}, None)
        if failure == "timeout":
            raise socket.timeout()
        return {"model": "other" if failure == "wrong_model" else MODEL,
                "choices": [{"finish_reason": "length" if failure == "length" else "stop",
                             "message": {"content": "not JSON"}}], "usage": {"prompt_tokens": 5}}

    rows = run(manifest, tmp_path, "https://example.invalid/v1", MODEL, "test-only", fake)
    assert len(calls) == 2 and len(rows) == 1 and rows[0]["status"] == "error"
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 2
    assert "no valid verdict" in (tmp_path / "report.html").read_text()
    if failure in {"invalid", "length", "wrong_model"}:
        assert rows[0]["returned_model"] and rows[0]["usage"]["prompt_tokens"] == 5


def test_discovery_fails_before_post_and_report_escapes(tmp_path: Path, source: Path) -> None:
    manifest = prepare(source=source, dest=tmp_path)
    def missing(url: str, body: dict[str, Any] | None, key: str, timeout: float) -> dict[str, Any]:
        assert body is None
        return {"data": [{"id": "wrong-model"}]}
    with pytest.raises(ValueError, match="not uniquely present"):
        run(manifest, tmp_path, "https://example.invalid/v1", MODEL, "test-only", missing)
    assert not (tmp_path / "calls.jsonl").exists()
    manifest["cases"][0]["target"]["original"] = "<script>alert(1)</script>"
    rendered = report(manifest, [])
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered


def test_discovery_unauthorized_never_starts_a_paid_attempt(tmp_path: Path, source: Path) -> None:
    manifest = prepare(source=source, dest=tmp_path)
    calls = 0

    def unauthorized(url: str, body: dict[str, Any] | None, key: str, timeout: float) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        assert body is None
        raise urllib.error.HTTPError(url, 401, "unauthorized", {}, None)

    with pytest.raises(urllib.error.HTTPError) as error:
        run(manifest, tmp_path, "https://example.invalid/v1", MODEL, "test-only", unauthorized)
    assert error.value.code == 401 and calls == 1
    assert not (tmp_path / "calls.jsonl").exists()


def test_tampered_last_case_stops_before_discovery(tmp_path: Path, source: Path) -> None:
    manifest = prepare(source=source, dest=tmp_path)
    manifest["cases"][-1]["target"]["candidate"] = "changed"
    def unreachable(url: str, body: dict[str, Any] | None, key: str, timeout: float) -> dict[str, Any]:
        raise AssertionError("transport must not be used")
    with pytest.raises(ValueError, match="frozen input changed"):
        run(manifest, tmp_path, "https://example.invalid/v1", MODEL, "test-only", unreachable)
    assert not (tmp_path / "calls.jsonl").exists()


def test_exclusive_ledger_prevents_second_runner(tmp_path: Path, source: Path) -> None:
    manifest = prepare(source=source, dest=tmp_path)
    def race(url: str, body: dict[str, Any] | None, key: str, timeout: float) -> dict[str, Any]:
        assert body is None
        (tmp_path / "calls.jsonl").touch(exist_ok=False)
        return {"data": [{"id": MODEL}]}
    with pytest.raises(FileExistsError):
        run(manifest, tmp_path, "https://example.invalid/v1", MODEL, "test-only", race)
    assert not (tmp_path / "report.html").exists()


def test_cli_counts_valid_responses_separately_from_accepts(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = prepare(source=source, dest=tmp_path)
    monkeypatch.setattr(check, "DEST", tmp_path)
    monkeypatch.setenv("OMNIROUTE_API_KEY", "test-only")
    monkeypatch.setattr(sys, "argv", ["mimo_arbiter_check", "--run"])
    monkeypatch.setattr(check, "run", lambda *args, **kwargs: [
        {"status": "ok", "judgment": {"verdict": "accept" if index < 4 else "revise"}}
        for index in range(len(manifest["cases"]))])
    assert check.main() == 0
    assert "valid responses: 8; accept verdicts: 4" in capsys.readouterr().out
    monkeypatch.setattr(check, "run", lambda *args, **kwargs: [{"status": "error"}])
    assert check.main() == 1


def _first_saved_error(tmp_path: Path, manifest: dict[str, Any]) -> None:
    case = manifest["cases"][0]
    start = {"event": "start", "cue_id": IDS[0], "status": "outcome unknown (started)",
             "input_sha256": case["input_sha256"], "model": MODEL, "base_url": "https://example.invalid/v1",
             "max_tokens": 4096}
    response = {"model": check.BARE_MODEL,
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"results": [
                    {"cue_id": IDS[0], "verdict": "accept", "issues": [], "advice": ""}]})}}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 20}}
    complete = {**start, "event": "complete", "status": "error", "error": "ValueError",
                "response": response, "returned_model": check.BARE_MODEL, "usage": response["usage"],
                "cost_usd": None, "elapsed_sec": 12.5, "started_at": "2026-10-01T00:00:00+00:00"}
    (tmp_path / "calls.jsonl").write_text(json.dumps(start) + "\n" + json.dumps(complete) + "\n")


def test_resume_validates_saved_reply_and_sends_only_seven(tmp_path: Path, source: Path) -> None:
    prepare(source=source, dest=tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    _first_saved_error(tmp_path, manifest)
    calls: list[str] = []

    def fake(url: str, body: dict[str, Any] | None, key: str, timeout: float) -> dict[str, Any]:
        calls.append("GET" if body is None else body["messages"][1]["content"])
        if body is None:
            return {"data": [{"id": MODEL, "owned_by": "opencode-go", "root": check.BARE_MODEL,
                              "name": check.BARE_MODEL}]}
        cue_id = IDS[len(calls) - 1]
        return {"model": check.BARE_MODEL, "choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"results": [{"cue_id": cue_id, "verdict": "accept", "issues": [], "advice": ""}]})}}]}

    rows = run(manifest, tmp_path, "https://example.invalid/v1", MODEL, "test-only", fake, resume=True)
    assert len(rows) == 8 and all(row["status"] == "ok" for row in rows)
    assert len(calls) == 8 and calls[0] == "GET"
    events = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events[:3]] == ["start", "complete", "validation"]
    assert events[1]["status"] == "error" and events[2]["response_sha256"] == _digest(events[1]["response"])
    assert len([e for e in events if e["event"] == "start"]) == 8
    assert (tmp_path / "identity_verification.json").exists()
    assert "Offline validation" in (tmp_path / "report.html").read_text()
    with pytest.raises(FileExistsError):
        run(manifest, tmp_path, "https://example.invalid/v1", MODEL, "test-only", fake, resume=True)
    assert len(calls) == 8


def test_unknown_start_and_unverified_alias_block_without_post(tmp_path: Path, source: Path) -> None:
    prepare(source=source, dest=tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    _first_saved_error(tmp_path, manifest)
    ledger = tmp_path / "calls.jsonl"
    original = ledger.read_text()
    ledger.write_text(original.splitlines()[0] + "\n")
    def never(url: str, body: dict[str, Any] | None, key: str, timeout: float) -> dict[str, Any]:
        raise AssertionError("no transport permitted")
    with pytest.raises(ValueError, match="unresolved request start"):
        run(manifest, tmp_path, "https://example.invalid/v1", MODEL, "test-only", never, resume=True)
    ledger.write_text(original)
    def wrong_owner(url: str, body: dict[str, Any] | None, key: str, timeout: float) -> dict[str, Any]:
        assert body is None
        return {"data": [{"id": MODEL, "owned_by": "other", "root": check.BARE_MODEL,
                          "name": check.BARE_MODEL}]}
    with pytest.raises(ValueError, match="verified bare-model mismatch"):
        run(manifest, tmp_path, "https://example.invalid/v1", MODEL, "test-only", wrong_owner, resume=True)
    assert len(ledger.read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="does not match"):
        check._content({"model": "different"}, MODEL, alias_verified=True)


def test_identity_proof_readback_is_stable_without_post(tmp_path: Path) -> None:
    catalog = {"data": [{"id": MODEL, "owned_by": "opencode-go", "root": check.BARE_MODEL,
                         "name": check.BARE_MODEL}]}
    assert check._identity(catalog, tmp_path, MODEL)
    proof = (tmp_path / "identity_verification.json")
    original = proof.read_bytes()
    assert check._identity(catalog, tmp_path, MODEL)
    assert proof.read_bytes() == original
