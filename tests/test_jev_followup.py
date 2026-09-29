from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from utils import jev_followup as followup


SOURCE = Path("output/jev_replay_20260919")


def test_protocol_constants_are_frozen() -> None:
    assert followup.SCHEMA == "jev_followup.v1"
    assert followup.JEV_MODEL == "jev-1.13.0"
    assert followup.DEEPSEEK_MODEL == "deepseek-flash"
    assert followup.DEEPSEEK_MODE == "off"
    assert followup.JEV_RATE == followup.Decimal("0.042")
    assert followup.JEV_CAP == followup.Decimal("0.15")
    assert followup.JEV_MAX_REQUESTS == 35
    assert followup.DEEPSEEK_INPUT_RATE == followup.Decimal("0.30")
    assert followup.DEEPSEEK_OUTPUT_RATE == followup.Decimal("1.20")
    assert followup.DEEPSEEK_CAP == followup.Decimal("0.75")
    assert followup.DEEPSEEK_MAX_REQUESTS == 23
    assert followup.JEV_RESERVATION == followup.Decimal("0.002752512")
    assert followup.DEEPSEEK_RESERVATION == followup.Decimal("0.3342336")


def test_prepare_actual_saved_source_if_present(tmp_path: Path) -> None:
    if not (SOURCE / "manifest.json").is_file():
        pytest.skip("saved Jev replay source is not present")
    report = followup.prepare(SOURCE, tmp_path / "followup")
    assert report["schema"] == followup.SCHEMA
    assert report["summary"]["cases"] == 6
    assert report["summary"]["records"] == 58
    assert report["backend_consumed"] is False
    manifest = json.loads((tmp_path / "followup" / "manifest.json").read_text(encoding="utf-8"))
    assert {case["case_id"] for case in manifest["cases"]} == set(followup.EXPECTED_CASES)
    assert len(manifest["requests"]) == 58
    assert manifest["limits"] == {
        "jev_cap_usd": "0.15", "jev_max_requests": 35,
        "jev_reservation_usd": "0.002752512", "deepseek_cap_usd": "0.75",
        "deepseek_max_requests": 23, "deepseek_reservation_usd": "0.3342336",
    }
    repeats = [item for item in manifest["requests"] if item["request_kind"] == "repeat"]
    isolated = [item for item in manifest["requests"] if item["request_kind"] == "isolated_safety"]
    reviews = [item for item in manifest["requests"] if item["request_kind"] == "independent_review"]
    assert len(repeats) == 12 and len(isolated) == 23 and len(reviews) == 23
    for request in repeats:
        old = next(item for item in json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))["requests"]
                   if item["case_id"] == request["case_id"] and item["order"] == request["order"])
        assert request["payload"] == old["payload"]
        assert request["old_payload_hash"] == followup.canonical_hash(old["payload"])
    for request in isolated:
        assert list(request["payload"]["questions"]) == [f"safety_{request['candidate_id']}"]
        assert len(request["payload"]["state"]["candidate_windows"]) == 1
        assert list(request["payload"]["state"]["precomputed_length_fit"]) == [request["candidate_id"]]
    for request in reviews:
        target_id = f"{request['case_id']}::{request['candidate_id']}"
        target = [row for row in request["context"] if row.get("is_target") is True]
        assert len(target) == 1 and target[0]["cue_id"] == target_id
        assert target_id in request["user_prompt"]
        assert "generator_model" not in request["user_prompt"]
    assert not (tmp_path / "followup" / "shortening.html").exists()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        followup.prepare(SOURCE, output)


def _fresh_pending_row(descriptor: dict[str, Any]) -> dict[str, Any]:
    key = followup._request_key(descriptor)
    provider = "jev" if descriptor["request_kind"] in {"repeat", "isolated_safety"} else "deepseek"
    return {
        "key": key, "descriptor": deepcopy(descriptor), "provider": provider, "status": "pending",
        "attempts": 0, "usage": None, "cost_usd": None, "cost_kind": None, "response": None,
        "receipt_id": None, "answers": {}, "review": None, "error": None,
    }


def _descriptor_match_key(descriptor: Mapping[str, Any]) -> tuple[str, str, Any, Any, str]:
    return (
        str(descriptor.get("request_kind")), str(descriptor.get("case_id")),
        descriptor.get("order"), descriptor.get("candidate_id"),
        str(descriptor.get("descriptor_hash")),
    )


def _test_localcopy(tmp_path: Path, *, second_invalid: bool = False) -> Path:
    saved = Path("output/jev_followup_20260919")
    if not all((path / name).is_file() for path in (saved, SOURCE) for name in ("manifest.json", "ledger.json")):
        pytest.skip("saved partial Jev follow-up is not present")
    output = tmp_path / "followup"
    raw_paths = [saved / name for name in ("manifest.json", "ledger.json")] + [
        SOURCE / name for name in ("manifest.json", "ledger.json")
    ]
    raw_hashes = {path: followup._sha_file(path) for path in raw_paths}
    source_manifest = json.loads((saved / "manifest.json").read_text(encoding="utf-8"))
    source_ledger = json.loads((saved / "ledger.json").read_text(encoding="utf-8"))
    followup.prepare(SOURCE, output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    old_descriptors = source_manifest["requests"]
    current_descriptors = manifest["requests"]
    assert len(old_descriptors) == len(current_descriptors) == 58
    old_by_identity = {_descriptor_match_key(item): item for item in old_descriptors}
    assert len(old_by_identity) == 58
    ledger = {
        "schema": manifest["schema"], "run_id": manifest["run_id"],
        "manifest_checksum": manifest["checksum"], "campaign_status": "prepared",
        "requests": {},
    }
    old_rows = source_ledger["requests"]
    for descriptor in current_descriptors:
        identity = _descriptor_match_key(descriptor)
        old_descriptor = old_by_identity.get(identity)
        assert old_descriptor is not None
        assert followup.canonical_hash(old_descriptor) == followup.canonical_hash(descriptor)
        old_key = followup._request_key(old_descriptor)
        current_key = followup._request_key(descriptor)
        assert old_key in old_rows and old_key == current_key
        assert old_rows[old_key]["descriptor"] == old_descriptor

    for index, descriptor in enumerate(current_descriptors):
        key = followup._request_key(descriptor)
        if index < 10 or index == 10:
            old_key = followup._request_key(old_by_identity[_descriptor_match_key(descriptor)])
            historical = deepcopy(old_rows[old_key])
            historical["key"] = key
            historical["descriptor"] = deepcopy(descriptor)
            ledger["requests"][key] = historical
        else:
            ledger["requests"][key] = _fresh_pending_row(descriptor)

    if second_invalid:
        historical_hashes = source_manifest["code_hashes"]
        current_hashes = manifest["code_hashes"]
        assert historical_hashes["jev_followup.py"] != current_hashes["jev_followup.py"]
        manifest["code_hashes"] = dict(current_hashes)
        manifest["code_hashes"]["jev_followup.py"] = historical_hashes["jev_followup.py"]
        manifest["checksum"] = followup._manifest_checksum(manifest)
        ledger["manifest_checksum"] = manifest["checksum"]
        followup.atomic_json(output / "manifest.json", manifest)
    followup._save_ledger(output, ledger)

    if second_invalid:
        first_invalid = followup._request_key(current_descriptors[10])
        followup.authorize_continuation(output, first_invalid)
        ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
        second_key = followup._request_key(current_descriptors[11])
        old_key = followup._request_key(old_by_identity[_descriptor_match_key(current_descriptors[11])])
        historical = deepcopy(old_rows[old_key])
        historical["key"] = second_key
        historical["descriptor"] = deepcopy(current_descriptors[11])
        ledger["requests"][second_key] = historical
        followup._save_ledger(output, ledger)

    assert {path: followup._sha_file(path) for path in raw_paths} == raw_hashes
    return output


def test_actual_saved_partial_continuation_is_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = _test_localcopy(tmp_path, second_invalid=True)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    repeat_keys = [followup._request_key(item) for item in manifest["requests"] if item["request_kind"] == "repeat"]
    invalid = [key for key in repeat_keys if ledger["requests"][key].get("status") == "error"]
    assert repeat_keys.index(invalid[0]) == 10
    assert repeat_keys.index(invalid[1]) == 11
    next_key = next(key for key in ledger["requests"] if ledger["requests"][key]["status"] == "pending")
    assert all(ledger["requests"][key]["attempts"] == 1 for key in invalid)
    assert ledger["requests"][next_key]["status"] == "pending"
    continuation_bytes = (output / "continuation.json").read_bytes()
    followup.authorize_invalid_answer_policy(output)
    assert (output / "continuation.json").read_bytes() == continuation_bytes
    assert (output / followup.INVALID_ANSWER_POLICY_FILE).stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        followup.authorize_invalid_answer_policy(output)
    monkeypatch.setattr(followup.jev, "_post", lambda *_args: pytest.fail("acknowledged request must not be retried"))
    monkeypatch.setattr(followup, "MatrixBackend", lambda: pytest.fail("resume must not make a network call"))
    report = asyncio.run(followup._run_async(output, 0))
    assert report["records"][10]["status"] == "error"
    assert report["records"][10]["not_retried"] is True
    resumed_ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    assert all(resumed_ledger["requests"][key]["attempts"] == 1 for key in invalid)
    assert resumed_ledger["requests"][next_key]["status"] == "pending"


@pytest.mark.parametrize("mutation", ["status_code", "known_usage_cost"])
def test_invalid_answer_policy_rejects_tampered_row_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    output = _test_localcopy(tmp_path, second_invalid=True)
    ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    invalid = next(
        key for key, row in ledger["requests"].items()
        if row.get("provider") == "jev" and row.get("status") == "error"
    )
    row = ledger["requests"][invalid]
    if mutation == "status_code":
        row["response"]["status_code"] = 401
    else:
        row["usage"]["input_tokens"] += 1
        row["cost_usd"] = "0.000249228"
    followup._save_ledger(output, ledger)

    network_calls = 0

    async def no_post(*_args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("authorization must not make a network call")

    monkeypatch.setattr(followup.jev, "_post", no_post)
    monkeypatch.setattr(followup, "MatrixBackend", lambda: pytest.fail("authorization must not create a backend"))
    with pytest.raises(followup.FollowupError):
        followup.authorize_invalid_answer_policy(output)
    assert network_calls == 0
    assert not (output / followup.INVALID_ANSWER_POLICY_FILE).exists()


def _jev_policy_receipt(descriptor: dict[str, Any], *, invalid: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    answers: dict[str, Any] = {}
    for question_id, question in descriptor["payload"]["questions"].items():
        options = list(question["criteria"])
        assert len(options) > 1
        choice = options[0]
        remainder = (0.9 if invalid else 0.3) / (len(options) - 1)
        probabilities = {option: remainder for option in options}
        probabilities[choice] = 0.1 if invalid else 0.7
        answers[question_id] = {"type": "choice", "choice": choice,
                                "probabilities": probabilities, "confidence": 0.7}
    raw = {"model": followup.JEV_MODEL, "answers": answers,
           "usage": {"input_tokens": 100, "output_tokens": 20}}
    return raw, {"status_code": 200, "request_id": "policy-test", "json": raw}


def _mixed_invalid_jev_receipt(descriptor: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    answers: dict[str, Any] = {}
    for index, (question_id, question) in enumerate(descriptor["payload"]["questions"].items()):
        options = list(question["criteria"])
        assert len(options) > 1
        if index == 0:
            probabilities = {option: 0.9 / (len(options) - 1) for option in options}
            probabilities[options[0]] = 0.1
        else:
            probabilities = {option: 0.1 for option in options}
        answers[question_id] = {"type": "choice", "choice": options[0],
                                "probabilities": probabilities, "confidence": 0.7}
    raw = {"model": followup.JEV_MODEL, "answers": answers,
           "usage": {"input_tokens": 100, "output_tokens": 20}}
    return raw, {"status_code": 200, "request_id": "mixed-invalid-test", "json": raw}


def test_policy_allows_only_known_invalid_parser_error_to_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _test_localcopy(tmp_path, second_invalid=True)
    followup.authorize_invalid_answer_policy(output)
    ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    pending = [item for item in manifest["requests"] if ledger["requests"][followup._request_key(item)]["status"] == "pending"]
    assert pending[0]["request_kind"] == "isolated_safety"
    calls: list[str] = []

    async def fake_post(descriptor: dict[str, Any], _key: str) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append(followup._request_key(descriptor))
        return _jev_policy_receipt(descriptor, invalid=len(calls) == 1)

    monkeypatch.setattr(followup.jev, "_key", lambda: "offline-test-key")
    monkeypatch.setattr(followup.jev, "_post", fake_post)
    monkeypatch.setattr(followup, "MatrixBackend", lambda: _FakeMatrix())
    report = asyncio.run(followup._run_async(output, 2))
    resumed = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    first = resumed["requests"][calls[0]]
    assert first["status"] == "error"
    assert first["error"] == followup.INVALID_JEV_ERROR
    assert first["cost_kind"] == "estimated_usage"
    assert len(calls) == 2
    assert report["summary"]["errors"] == 3
    assert report["summary"]["attempted_requests"] == 14
    assert report["status"] == "partial"


def test_policy_does_not_quarantine_text_only_when_another_question_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _test_localcopy(tmp_path, second_invalid=True)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    descriptor = next(
        item for item in manifest["requests"]
        if item["request_kind"] == "repeat"
        and ledger["requests"][followup._request_key(item)]["status"] == "complete"
        and len(item["payload"]["questions"]) > 1
    )
    key = followup._request_key(descriptor)
    ledger["requests"][key].update({"status": "pending", "attempts": 0, "usage": None,
                                     "cost_usd": None, "cost_kind": None, "response": None,
                                     "receipt_id": None, "answers": {}, "error": None})
    followup._save_ledger(output, ledger)
    followup.authorize_invalid_answer_policy(output)
    calls: list[str] = []

    async def fake_post(request: dict[str, Any], _key: str) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append(followup._request_key(request))
        return _mixed_invalid_jev_receipt(request)

    monkeypatch.setattr(followup.jev, "_key", lambda: "offline-test-key")
    monkeypatch.setattr(followup.jev, "_post", fake_post)
    monkeypatch.setattr(followup, "MatrixBackend", lambda: _FakeMatrix())
    with pytest.raises(followup.FollowupError, match="unauthorized error row"):
        asyncio.run(followup._run_async(output, 1))
    resumed = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    assert calls == [key]
    assert resumed["requests"][key]["status"] == "error"
    assert resumed["requests"][key]["error"] == followup.INVALID_JEV_ERROR
    next_pending = next(request_id for request_id, row in resumed["requests"].items()
                        if row["status"] == "pending")
    assert next_pending != key
    policy = json.loads((output / followup.INVALID_ANSWER_POLICY_FILE).read_text(encoding="utf-8"))
    assert key not in {item["request_id"] for item in policy["currently_quarantined"]}
    assert len(policy["currently_quarantined"]) == 2


def test_continuation_only_stops_on_new_invalid_answer_without_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _test_localcopy(tmp_path, second_invalid=True)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    acknowledged = json.loads((output / "continuation.json").read_text(encoding="utf-8"))["acknowledged_request_ids"][0]["request_id"]
    unacknowledged = next(request_id for request_id, row in ledger["requests"].items()
                          if row["status"] == "error" and request_id != acknowledged)
    ledger["requests"][unacknowledged]["status"] = "complete"
    followup._save_ledger(output, ledger)
    descriptor = next(item for item in manifest["requests"]
                      if ledger["requests"][followup._request_key(item)]["status"] == "pending")
    key = followup._request_key(descriptor)
    calls: list[str] = []

    async def fake_post(request: dict[str, Any], _key: str) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append(followup._request_key(request))
        return _jev_policy_receipt(request, invalid=True)

    monkeypatch.setattr(followup.jev, "_key", lambda: "offline-test-key")
    monkeypatch.setattr(followup.jev, "_post", fake_post)
    monkeypatch.setattr(followup, "MatrixBackend", lambda: _FakeMatrix())
    continuation = json.loads((output / "continuation.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(followup, "_code_hashes", lambda: continuation["new_code_hashes"])
    report = asyncio.run(followup._run_async(output, 1))
    resumed = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    assert calls == [key]
    assert resumed["requests"][key]["status"] == "error"
    assert resumed["requests"][key]["error"] == followup.INVALID_JEV_ERROR
    assert next(row for request_id, row in resumed["requests"].items()
                if request_id != key and row["status"] == "pending")
    assert not (output / followup.INVALID_ANSWER_POLICY_FILE).exists()
    new_record = next(record for item, record in zip(manifest["requests"], report["records"])
                      if followup._request_key(item) == key)
    assert new_record.get("manual_acknowledged") is not True
    assert report["summary"]["acknowledged_invalid_responses"] == 1


def test_terminal_policy_run_retains_acknowledged_invalid_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _test_localcopy(tmp_path, second_invalid=True)
    followup.authorize_invalid_answer_policy(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    pending_deepseek = [item for item in manifest["requests"]
                        if ledger["requests"][followup._request_key(item)]["status"] == "pending"
                        and ledger["requests"][followup._request_key(item)]["provider"] == "deepseek"]
    matrix_calls = 0

    class _SequenceMatrix:
        async def __aenter__(self) -> "_SequenceMatrix":
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        def preflight(self, _model: str) -> dict[str, Any]:
            return {"ok": True}

        async def __call__(self, *_args: Any) -> dict[str, Any]:
            nonlocal matrix_calls
            descriptor = pending_deepseek[matrix_calls]
            matrix_calls += 1
            text = json.dumps({"results": [{"cue_id": descriptor["target"]["cue_id"],
                                               "verdict": "accept", "issues": [], "advice": ""}]})
            return _valid_deepseek_receipt(text=text)

    async def fake_post(descriptor: dict[str, Any], _key: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return _jev_policy_receipt(descriptor, invalid=False)

    monkeypatch.setattr(followup.jev, "_key", lambda: "offline-test-key")
    monkeypatch.setattr(followup.jev, "_post", fake_post)
    monkeypatch.setattr(followup, "MatrixBackend", _SequenceMatrix)
    report = asyncio.run(followup._run_async(output, 46))
    resumed = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    errors = [row for row in resumed["requests"].values() if row["status"] == "error"]
    assert matrix_calls == len(pending_deepseek) == 23
    assert all(row["status"] != "pending" for row in resumed["requests"].values())
    assert len(errors) == 2
    assert report["summary"]["pending"] == 0
    assert report["summary"]["errors"] == 2
    assert report["summary"]["clean_complete"] is False
    assert report["campaign_status"] == "completed_with_invalid_response"
    assert report["summary"]["completed_with_invalid_response"] is True


def test_deepseek_review_parser_rejects_foreign_ids() -> None:
    text = json.dumps({"results": [{"cue_id": "foreign", "verdict": "accept", "issues": [], "advice": ""}]})
    with pytest.raises(ValueError):
        followup.parse_review(text, ["case::c1"])


def test_admission_accounts_unknown_reservation() -> None:
    ledger = {"requests": {str(index): {"provider": "deepseek", "status": "unknown", "attempts": 1}
                              for index in range(14)}}
    with pytest.raises(followup.FollowupError):
        followup._admit(ledger, "deepseek")


def _review_descriptor() -> dict[str, Any]:
    case = {
        "case_id": "source:cue-1", "source": "source", "start": 1, "end": 2,
        "original": "original", "context": [], "target_position": 1, "max_chars": 20,
        "length_basis": "display_text", "candidates": [{"id": "c1", "text": "short", "actual_chars": 5, "fits": True}],
    }
    questions = {"safety_c1": {"type": "choice", "criteria": {"accept": None, "reject": None}}}
    state = {
        "original_window": [
            {"cue_id": "cue-0", "text": "left", "start": 0, "end": 1},
            {"cue_id": "cue-1", "text": "original", "start": 1, "end": 2, "marked_target": True},
            {"cue_id": "cue-2", "text": "right", "start": 2, "end": 3},
        ],
        "target_position": 1, "original_target": "original", "max_chars": 20, "length_basis": "display_text",
        "candidate_windows": [{"candidate_id": "c1", "window": [
            {"cue_id": "cue-0", "text": "left", "start": 0, "end": 1},
            {"cue_id": "cue-1", "text": "short", "start": 1, "end": 2, "marked_target": True},
            {"cue_id": "cue-2", "text": "right", "start": 2, "end": 3},
        ]}],
        "precomputed_length_fit": {"c1": {"fits": True}},
    }
    old = {"payload": {"state": state, "questions": questions, "model": followup.JEV_MODEL}}
    return {"case": case, "old": old, "candidate": case["candidates"][0], "questions": questions}


def test_isolated_question_is_exact_source_dict() -> None:
    fixture = _review_descriptor()
    isolated = followup._isolated_descriptor(fixture["case"], fixture["old"], fixture["candidate"])
    assert isolated["payload"]["questions"] == fixture["questions"]


def test_independent_prompt_is_blinded_and_marks_only_target() -> None:
    fixture = _review_descriptor()
    descriptor = followup._review_descriptor(fixture["case"], fixture["old"], fixture["candidate"])
    lowered = descriptor["user_prompt"].lower()
    for forbidden in ("muse", "historic_sol", "source_run", "provenance"):
        assert forbidden not in lowered
    marked = [row for row in descriptor["context"] if row.get("is_target") is True]
    assert len(marked) == 1
    assert marked[0]["cue_id"] == "source:cue-1::c1"
    assert descriptor["target"]["cue_id"] == "source:cue-1::c1"


def _run_fixture(tmp_path: Path, *, provider: str = "deepseek", status: str = "pending", attempts: int = 0,
                 cost_usd: str | None = None, usage: dict[str, int] | None = None) -> tuple[Path, str, dict[str, Any]]:
    output = tmp_path / "followup"
    output.mkdir(mode=0o700)
    if provider == "deepseek":
        descriptor: dict[str, Any] = {
            "request_kind": "independent_review", "case_id": "case-1", "order": None, "candidate_id": "c1",
            "system_prompt": "system", "user_prompt": "user", "schema": followup.review_schema(),
            "target": {"cue_id": "case-1::c1"}, "old_payload_hash": "old",
        }
    else:
        descriptor = {"request_kind": "repeat", "case_id": "case-1", "order": 1, "candidate_id": None,
                      "payload": {"questions": {}}, "body_hash": "body", "old_payload_hash": "old"}
    descriptor["record"] = {"case_id": "case-1", "source": "source", "start": 1, "end": 2,
                             "original": "original", "context_original_window": [], "target_position": 0,
                             "max_chars": 20, "length_basis": "display_text", "candidates": [], "status": "pending",
                             "answers": {}, "review": None, "receipt_id": None}
    manifest: dict[str, Any] = {"schema": followup.SCHEMA, "version": followup.VERSION, "run_id": "run-1",
                                "status": "prepared", "model": followup.JEV_MODEL, "requests": [descriptor],
                                "cases": [{"case_id": "case-1", "candidates": []}],
                                "requests_sha256": followup.canonical_hash([descriptor])}
    manifest["checksum"] = followup._manifest_checksum(manifest)
    followup.atomic_json(output / "manifest.json", manifest)
    key = followup._request_key(descriptor)
    row = {"key": key, "descriptor": descriptor, "provider": provider, "status": status, "attempts": attempts,
           "usage": usage, "cost_usd": cost_usd, "response": None, "receipt_id": None, "answers": {}, "review": None, "error": None}
    ledger = {"schema": followup.SCHEMA, "run_id": "run-1", "manifest_checksum": manifest["checksum"],
              "campaign_status": "prepared", "requests": {key: row}}
    followup._save_ledger(output, ledger)
    return output, key, descriptor


class _FakeMatrix:
    def __init__(self, receipt: dict[str, Any] | None = None) -> None:
        self.receipt = receipt or {}
        self.preflight_calls = 0
        self.calls = 0

    async def __aenter__(self) -> "_FakeMatrix":
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def preflight(self, _model: str) -> dict[str, Any]:
        self.preflight_calls += 1
        return {"ok": True}

    async def __call__(self, *_args: Any) -> dict[str, Any]:
        self.calls += 1
        return self.receipt


def _patch_prepared_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(followup, "_validate_manifest", lambda *_args: None)


def _valid_deepseek_receipt(input_tokens: int = 100, output_tokens: int = 20, text: str | None = None) -> dict[str, Any]:
    expected = (followup.Decimal(input_tokens) * followup.DEEPSEEK_INPUT_RATE
                + followup.Decimal(output_tokens) * followup.DEEPSEEK_OUTPUT_RATE) / followup.Decimal(1_000_000)
    return {"text": text or json.dumps({"results": [{"cue_id": "case-1::c1", "verdict": "accept", "issues": [], "advice": ""}]}),
            "raw": {"request_id": "deepseek-request"}, "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "cost_usd": float(expected), "cost_known": True, "error": None}


@pytest.mark.parametrize("status,provider", [("started", "jev"), ("unknown", "deepseek"), ("error", "jev")])
def test_failed_state_blocks_before_keys_preflight_and_post(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            status: str, provider: str) -> None:
    output, _, _ = _run_fixture(tmp_path, provider=provider, status=status, attempts=1)
    _patch_prepared_validation(monkeypatch)
    monkeypatch.setattr(followup.jev, "_key", lambda: pytest.fail("key must not be loaded"))
    matrix = _FakeMatrix()
    monkeypatch.setattr(followup, "MatrixBackend", lambda: matrix)
    monkeypatch.setattr(followup.jev, "_post", lambda *_args: pytest.fail("post must not run"))
    report = asyncio.run(followup._run_async(output, 1))
    assert report["status"] == "blocked"
    assert report["campaign_status"] == "blocked"
    assert matrix.preflight_calls == matrix.calls == 0


def test_previous_complete_replays_without_new_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = _valid_deepseek_receipt()
    output, key, _ = _run_fixture(tmp_path, status="complete", attempts=1, cost_usd="0.000054", usage=receipt["usage"])
    ledger = json.loads((output / "ledger.json").read_text())
    ledger["requests"][key].update({"response": receipt, "receipt_id": followup._receipt_id(receipt), "review": followup.parse_review(receipt["text"], ["case-1::c1"])})
    followup._save_ledger(output, ledger)
    _patch_prepared_validation(monkeypatch)
    monkeypatch.setattr(followup, "MatrixBackend", lambda: pytest.fail("matrix must not run"))
    report = asyncio.run(followup._run_async(output, 1))
    assert report["summary"]["attempted_requests"] == 1


@pytest.mark.parametrize("provider,field,value", [
    ("jev", "amount", "0.149"), ("jev", "count", "35"),
    ("deepseek", "amount", "0.749"), ("deepseek", "count", "23"),
])
def test_budget_and_request_caps_stop_before_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  provider: str, field: str, value: str) -> None:
    attempts = int(value) if field == "count" else 1
    cost = value if field == "amount" else None
    output, _, _ = _run_fixture(tmp_path, provider=provider, attempts=attempts, cost_usd=cost,
                                usage={"input_tokens": 1, "output_tokens": 1} if cost else None)
    _patch_prepared_validation(monkeypatch)
    matrix = _FakeMatrix(_valid_deepseek_receipt())
    monkeypatch.setattr(followup, "MatrixBackend", lambda: matrix)
    monkeypatch.setattr(followup.jev, "_key", lambda: pytest.fail("key must not be loaded"))
    monkeypatch.setattr(followup.jev, "_post", lambda *_args: pytest.fail("post must not run"))
    report = asyncio.run(followup._run_async(output, 1))
    assert report["status"] == "blocked"
    assert matrix.calls == 0


def test_deepseek_unknown_usage_saves_receipt_and_reservation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, key, _ = _run_fixture(tmp_path)
    _patch_prepared_validation(monkeypatch)
    receipt = {"text": "", "raw": {"request_id": "unknown"}, "usage": {}, "cost_usd": None, "cost_known": False}
    matrix = _FakeMatrix(receipt)
    monkeypatch.setattr(followup, "MatrixBackend", lambda: matrix)
    report = asyncio.run(followup._run_async(output, 1))
    ledger = json.loads((output / "ledger.json").read_text())
    row = ledger["requests"][key]
    assert row["status"] == "unknown" and row["response"] == receipt
    assert report["summary"]["by_backend"]["deepseek"]["unknown_reserved_usd"] == "0.3342336"


def test_deepseek_valid_usage_bad_json_keeps_cost_and_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, key, _ = _run_fixture(tmp_path)
    _patch_prepared_validation(monkeypatch)
    receipt = _valid_deepseek_receipt(text="not json")
    matrix = _FakeMatrix(receipt)
    monkeypatch.setattr(followup, "MatrixBackend", lambda: matrix)
    report = asyncio.run(followup._run_async(output, 1))
    row = json.loads((output / "ledger.json").read_text())["requests"][key]
    assert row["status"] == "error" and row["response"] == receipt
    assert row["cost_usd"] == "0.000054"
    assert report["summary"]["by_backend"]["deepseek"]["unknown_reserved_usd"] == "0"


def test_deepseek_success_refresh_revalidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, key, _ = _run_fixture(tmp_path)
    _patch_prepared_validation(monkeypatch)
    matrix = _FakeMatrix(_valid_deepseek_receipt())
    monkeypatch.setattr(followup, "MatrixBackend", lambda: matrix)
    report = asyncio.run(followup._run_async(output, 1))
    assert report["records"][0]["status"] == "evaluated"
    ledger = json.loads((output / "ledger.json").read_text())
    ledger["requests"][key]["review"]["case-1::c1"]["verdict"] = "revise"
    followup._save_ledger(output, ledger)
    assert followup.refresh_report(output)["records"][0]["status"] == "error"


def test_jev_input_over_reserved_limit_keeps_receipt_and_reservation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, key, _ = _run_fixture(tmp_path, provider="jev")
    _patch_prepared_validation(monkeypatch)
    raw = {"usage": {"input_tokens": 65537, "output_tokens": 1}}
    receipt = {"status_code": 200, "request_id": "jev-request", "json": raw}
    async def fake_post(*_args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return raw, receipt
    monkeypatch.setattr(followup.jev, "_key", lambda: "key")
    monkeypatch.setattr(followup.jev, "_post", fake_post)
    monkeypatch.setattr(followup, "MatrixBackend", lambda: _FakeMatrix())
    report = asyncio.run(followup._run_async(output, 1))
    row = json.loads((output / "ledger.json").read_text())["requests"][key]
    assert row["status"] == "unknown" and row["response"] == receipt
    assert report["summary"]["by_backend"]["jev"]["unknown_reserved_usd"] == "0.002752512"


def test_float_cost_from_transport_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, _, _ = _run_fixture(tmp_path)
    _patch_prepared_validation(monkeypatch)
    matrix = _FakeMatrix(_valid_deepseek_receipt())
    monkeypatch.setattr(followup, "MatrixBackend", lambda: matrix)
    report = asyncio.run(followup._run_async(output, 1))
    assert report["records"][0]["status"] == "evaluated"


def test_deepseek_usage_rejects_undercharge_and_boolean_tokens() -> None:
    receipt = _valid_deepseek_receipt()
    receipt["cost_usd"] = 0.0
    with pytest.raises(followup.FollowupError):
        followup._deepseek_usage(receipt)
    receipt = _valid_deepseek_receipt()
    receipt["usage"]["input_tokens"] = True
    with pytest.raises(followup.FollowupError):
        followup._deepseek_usage(receipt)


def test_admission_block_reason_is_persisted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, _, _ = _run_fixture(tmp_path, provider="deepseek", attempts=1, cost_usd="0.749",
                                usage={"input_tokens": 1, "output_tokens": 1})
    _patch_prepared_validation(monkeypatch)
    matrix = _FakeMatrix()
    monkeypatch.setattr(followup, "MatrixBackend", lambda: matrix)
    report = asyncio.run(followup._run_async(output, 1))
    ledger = json.loads((output / "ledger.json").read_text())
    assert report["campaign_status"] == ledger["campaign_status"] == "blocked"
    assert report["campaign_error"] == ledger["campaign_error"]
    assert report["campaign_error"]
