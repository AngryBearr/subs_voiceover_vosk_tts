from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from utils import jev_replay as jev


def _questions() -> dict[str, object]:
    return {
        "best": {"type": "choice", "criteria": {"c1": None, "none_acceptable": None}},
        "safety_c1": {"type": "choice", "criteria": {"accept": None, "reject": None, "insufficient_context": None}},
    }


def _answer(options: list[str], choice: str) -> dict[str, object]:
    value = 1 / len(options)
    probabilities = {option: value for option in options}
    probabilities[choice] = 0.7
    remainder = 0.3 / (len(options) - 1)
    for option in options:
        if option != choice:
            probabilities[option] = remainder
    return {"type": "choice", "choice": choice, "probabilities": probabilities, "confidence": 0.7}


def test_parser_is_strict_and_model_pinned() -> None:
    response = {
        "model": jev.MODEL,
        "answers": {"best": _answer(["c1", "none_acceptable"], "c1"),
                    "safety_c1": _answer(["accept", "reject", "insufficient_context"], "accept")},
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }
    assert jev.parse_jev_response(response, _questions())["best"]["choice"] == "c1"
    bad = json.loads(json.dumps(response))
    bad["answers"]["best"]["probabilities"]["c1"] = 0.1
    with pytest.raises(jev.JevReplayError):
        jev.parse_jev_response(bad, _questions())


def test_parser_allows_choice_metadata() -> None:
    response = {
        "model": jev.MODEL,
        "answers": {"best": {**_answer(["c1", "none_acceptable"], "c1"), "trace_id": "saved"},
                    "safety_c1": _answer(["accept", "reject", "insufficient_context"], "accept")},
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }
    assert jev.parse_jev_response(response, _questions())["best"]["choice"] == "c1"
    bad = json.loads(json.dumps(response))
    bad["model"] = "jev-latest"
    with pytest.raises(jev.JevReplayError):
        jev.parse_jev_response(bad, _questions())


def test_decision_requires_joint_safety_and_fit() -> None:
    case = {"candidates": [{"id": "c1", "text": "коротко", "fits": True}]}
    answers = {"best": {"choice": "c1"}, "safety_c1": {"choice": "accept"}}
    assert jev.decide(case, answers)["accepted"] is True
    answers["safety_c1"] = {"choice": "reject"}
    assert jev.decide(case, answers)["accepted"] is False
    case["candidates"][0]["fits"] = False
    answers["safety_c1"] = {"choice": "accept"}
    assert jev.decide(case, answers)["accepted"] is False


def test_admission_rejects_boundary_and_unknown_resume() -> None:
    ledger = {"requests": {str(index): {"status": "complete", "attempts": 1,
                                         "estimated_cost_usd": "0"} for index in range(47)}}
    jev._admission(ledger)
    ledger["requests"]["unknown"] = {"status": "unknown", "attempts": 1,
                                      "reservation_usd": str(jev._reservation())}
    with pytest.raises(jev.JevReplayError):
        jev._admission(ledger)
    ledger["requests"].pop("unknown")
    ledger["requests"]["last"] = {"status": "complete", "attempts": 1,
                                   "estimated_cost_usd": "0"}
    with pytest.raises(jev.JevReplayError):
        jev._admission(ledger)


def test_key_redaction_never_returns_secret() -> None:
    secret = "TYPESAFE-test-secret"
    assert secret not in jev._redact(f"Bearer {secret}", (secret,))


def test_payload_orders_reverse_with_same_candidate_ids() -> None:
    case = {
        "case_id": "source:cue-1", "original": "исходный текст", "context": [
            {"cue_id": "cue-0", "text": "слева", "start": 1, "end": 2},
            {"cue_id": "cue-1", "text": "исходный текст", "start": 3, "end": 4},
            {"cue_id": "cue-2", "text": "справа", "start": 5, "end": 6},
        ], "target_position": 1, "max_chars": 20, "length_basis": "display_text",
        "candidates": [{"id": "c1", "text": "один", "actual_chars": 4, "fits": True},
                       {"id": "c2", "text": "два", "actual_chars": 3, "fits": True}],
    }
    first = jev._payload(case, 1)
    second = jev._payload(case, 2)
    ids1 = [item["candidate_id"] for item in first["state"]["candidate_windows"]]
    ids2 = [item["candidate_id"] for item in second["state"]["candidate_windows"]]
    assert ids1 == list(reversed(ids2))
    assert first["model"] == second["model"] == jev.MODEL
    assert first["state"]["original_window"][1]["text"] == "исходный текст"
    assert second["state"]["candidate_windows"][0]["window"][1]["marked_target"] is True


def test_prepare_real_saved_inputs_is_offline(tmp_path: Path) -> None:
    low = Path("output/sh08_muse_reasoning_20260915_low")
    high = Path("output/sh08_muse_reasoning_20260915_high")
    if not (low / "manifest.json").is_file() or not (high / "manifest.json").is_file():
        pytest.skip("saved SH08 Muse inputs are not present")
    prepared = jev.prepare(tmp_path / "jev", low_dir=low, high_dir=high)
    assert prepared["schema"] == jev.SCHEMA
    assert prepared["status"] == "prepared"
    assert len(prepared["cases"]) == 24
    assert len(prepared["requests"]) == 48
    assert not (tmp_path / "jev" / "shortening.html").exists()
    first = prepared["cases"][0]
    assert first["target_position"] == 3
    assert first["candidates"][0]["provenance"]
    for candidate in first["candidates"]:
        window = next(item["window"] for item in prepared["requests"][0]["payload"]["state"]["candidate_windows"]
                      if item["candidate_id"] == candidate["id"])
        assert sum(a["text"] != b["text"] for a, b in zip(first["context"], window)) == 1


def test_report_replay_does_not_need_key(tmp_path: Path) -> None:
    low = Path("output/sh08_muse_reasoning_20260915_low")
    high = Path("output/sh08_muse_reasoning_20260915_high")
    if not (low / "manifest.json").is_file() or not (high / "manifest.json").is_file():
        pytest.skip("saved SH08 Muse inputs are not present")
    out = tmp_path / "jev"
    jev.prepare(out, low_dir=low, high_dir=high)
    report = jev.run(out, max_new_requests=0)
    assert report["backend_consumed"] is False
    assert report["summary"]["pending"] == 48


def _descriptor(order: int) -> dict[str, object]:
    return {"case_id": "case-1", "order": order, "body_hash": f"hash-{order}",
            "payload": {"questions": _questions()}}


def _prepared_fixture(tmp_path: Path, *, status: str = "prepared", count: int = 1) -> Path:
    output = tmp_path / "jev"
    output.mkdir()
    descriptors = [_descriptor(order) for order in range(1, count + 1)]
    case = {"case_id": "case-1", "source": "source-1", "start": 1, "end": 2,
            "original": "исходный текст", "context": [], "target_position": 0,
            "max_chars": 20, "length_basis": "display_text",
            "candidates": [{"id": "c1", "text": "коротко", "fits": True}]}
    manifest = {"schema": jev.SCHEMA, "version": jev.VERSION, "run_id": "run-1", "status": status,
                "model": jev.MODEL, "mode": "test", "cases": [case], "requests": descriptors,
                "requests_sha256": jev.canonical_hash(descriptors), "notices": []}
    manifest["checksum"] = jev._manifest_checksum(manifest)
    jev.atomic_json(output / "manifest.json", manifest)
    ledger = {"schema": jev.SCHEMA, "run_id": "run-1", "manifest_checksum": manifest["checksum"],
              "requests": {jev._request_key(descriptor): {"key": jev._request_key(descriptor),
                  "descriptor": descriptor, "status": "pending", "attempts": 0,
                  "reservation_usd": str(jev._reservation()), "response": None, "error": None}
                            for descriptor in descriptors}}
    jev._save_ledger(output, ledger)
    return output


def _valid_response(input_tokens: int = 100) -> dict[str, object]:
    return {"model": jev.MODEL,
            "answers": {"best": _answer(["c1", "none_acceptable"], "c1"),
                        "safety_c1": _answer(["accept", "reject", "insufficient_context"], "accept")},
            "usage": {"input_tokens": input_tokens, "output_tokens": 20}}


def test_blocked_manifest_never_calls_key_or_post(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = _prepared_fixture(tmp_path, status="blocked")
    monkeypatch.setattr(jev, "_key", lambda: pytest.fail("key must not be loaded"))
    monkeypatch.setattr(jev, "_post", lambda *_args: pytest.fail("post must not run"))
    with pytest.raises(jev.JevReplayError, match="not prepared"):
        asyncio.run(jev._run_async(output, 1))


def test_timeout_keeps_current_receipt_and_reserves_only_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = _prepared_fixture(tmp_path, count=3)
    calls: list[int] = []

    async def fake_post(descriptor: dict[str, object], key: str) -> tuple[dict[str, object], dict[str, object]]:
        calls.append(int(descriptor["order"]))
        if len(calls) == 1:
            raw = _valid_response()
            return raw, {"status_code": 200, "body": "first", "request_id": "first", "json": raw}
        raise jev._ProviderError("provider timeout", {"status_code": 504, "body": "second",
                                                       "request_id": "second"}, unknown=True)

    monkeypatch.setattr(jev, "_key", lambda: "test-key")
    monkeypatch.setattr(jev, "_post", fake_post)
    report = asyncio.run(jev._run_async(output, 2))
    ledger = json.loads((output / "ledger.json").read_text())
    rows = list(ledger["requests"].values())
    first = next(row for row in rows if row["descriptor"]["order"] == 1)
    second = next(row for row in rows if row["descriptor"]["order"] == 2)
    assert calls == [1, 2]
    assert first["response"]["request_id"] == "first"
    assert second["response"]["request_id"] == "second"
    assert second["response"] != first["response"]
    assert second["status"] == "unknown"
    assert report["summary"]["attempted_requests"] == 2
    assert report["summary"]["completed_requests"] == 1
    assert report["summary"]["pending"] == 1
    assert report["summary"]["unknown_reserved_usd"] == str(jev._reservation())


def test_invalid_answers_still_record_valid_usage_cost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = _prepared_fixture(tmp_path)
    raw = _valid_response(123)
    raw["answers"]["best"]["probabilities"]["c1"] = 0.1  # type: ignore[index]

    async def fake_post(_descriptor: dict[str, object], _key: str) -> tuple[dict[str, object], dict[str, object]]:
        return raw, {"status_code": 200, "body": "invalid answers", "request_id": "invalid", "json": raw}

    monkeypatch.setattr(jev, "_key", lambda: "test-key")
    monkeypatch.setattr(jev, "_post", fake_post)
    report = asyncio.run(jev._run_async(output, 1))
    expected = str(jev.Decimal(123) * jev.RATE / jev.Decimal(1_000_000))
    assert report["summary"]["errors"] == 1
    assert report["summary"]["completed_requests"] == 0
    assert report["summary"]["cost_usd_estimated"] == expected
    assert report["summary"]["unknown_reserved_usd"] == "0"


def test_refresh_report_rejects_stored_answer_mismatch(tmp_path: Path) -> None:
    output = _prepared_fixture(tmp_path)
    descriptor = _descriptor(1)
    raw = _valid_response()
    key = jev._request_key(descriptor)
    ledger = json.loads((output / "ledger.json").read_text())
    row = ledger["requests"][key]
    row.update({"status": "complete", "attempts": 1, "response": {"status_code": 200, "json": raw},
                "answers": {"best": _answer(["c1", "none_acceptable"], "none_acceptable"),
                            "safety_c1": raw["answers"]["safety_c1"]},
                "usage": raw["usage"], "estimated_cost_usd": str(jev._reservation())})
    jev._save_ledger(output, ledger)
    report = jev.refresh_report(output)
    assert report["summary"]["accepted"] == 0
    assert report["records"][0]["status"] == "error"
