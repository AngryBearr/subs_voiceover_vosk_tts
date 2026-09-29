"""Bounded, offline-first follow-up experiment for a frozen Jev replay.

Preparation only reads the saved replay.  Network calls are made by ``run``
through the existing Jev transport and MatrixBackend, one durable request at a
time.  The module intentionally has no generator, queue, fallback provider, or
persistent database.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from filelock import FileLock

from utils import jev_replay as jev
from utils.length_check import check_length
from utils.sh08_matrix_models import MODEL_SPECS
from utils.sh08_matrix_transport import MatrixBackend
from utils.shortening_html import live_html_report
from utils.shortening_prompts import (
    ARBITER_SYSTEM_PROMPT,
    PROMPT_VERSION,
    build_review_prompt,
    parse_review,
    review_schema,
)
from utils.shortening_resume import atomic_json, canonical_hash, _read_json


SCHEMA = "jev_followup.v1"
VERSION = SCHEMA
JEV_MODEL = "jev-1.13.0"
DEEPSEEK_MODEL = "deepseek-flash"
DEEPSEEK_MODE = "off"
JEV_RATE = Decimal("0.042")
JEV_CAP = Decimal("0.15")
JEV_RESERVATION_TOKENS = 65_536
JEV_RESERVATION = Decimal(JEV_RESERVATION_TOKENS) * JEV_RATE / Decimal(1_000_000)
JEV_MAX_REQUESTS = 35
DEEPSEEK_INPUT_RATE = Decimal("0.30")
DEEPSEEK_OUTPUT_RATE = Decimal("1.20")
DEEPSEEK_CAP = Decimal("0.75")
DEEPSEEK_RESERVATION = (
    Decimal(1_048_576) * DEEPSEEK_INPUT_RATE / Decimal(1_000_000)
    + Decimal(16_384) * DEEPSEEK_OUTPUT_RATE / Decimal(1_000_000)
)
DEEPSEEK_MAX_REQUESTS = 23
PAYLOAD_LIMIT = 120_000
EXPECTED_CASES = {
    "bec0de4b2abc:cue-679": 5,
    "2c58c71203c3:cue-210": 2,
    "bec0de4b2abc:cue-504": 3,
    "16e2689c67a5:cue-618": 3,
    "16e2689c67a5:cue-94": 5,
    "16e2689c67a5:cue-353": 5,
}
USER_CONTROLS = frozenset({"16e2689c67a5:cue-94", "16e2689c67a5:cue-353"})
CONTINUATION_SCHEMA = "jev_followup_continuation.v1"
INVALID_ANSWER_POLICY_SCHEMA = "jev_followup_invalid_answer_policy.v1"
INVALID_ANSWER_POLICY_NAME = "quarantine_known_billed_jev_argmax_inconsistency"
INVALID_ANSWER_POLICY_FILE = "invalid_answer_policy.json"
INVALID_ANSWER_POLICY_REASON = "operator-approved quarantine of known billed Jev argmax/probability inconsistencies; no retry or acceptance"
INVALID_JEV_ERROR = "Jev choice is inconsistent with probabilities"
CONTINUATION_REASON = "known-billed invalid Jev response; no retry; continue pending only"
CONTINUATION_CODE_FILE = "jev_followup.py"
CONTINUATION_NOTICE = "Manual acknowledgment: known-billed invalid Jev response; not retried; response validity is unchanged."


class FollowupError(ValueError):
    """Saved input, request state, or provider accounting is not trusted."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    try:
        return _sha_bytes(path.read_bytes())
    except OSError as exc:
        raise FollowupError(f"cannot hash {path}: {exc}") from exc


def _load(path: Path, label: str) -> Any:
    try:
        return _read_json(path, label)
    except Exception as exc:
        raise FollowupError(str(exc)) from exc


def _envelope(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("checksum"), str):
        raise FollowupError(f"{label} envelope is invalid")
    body = {key: item for key, item in value.items() if key != "checksum"}
    if canonical_hash(body) != value["checksum"]:
        raise FollowupError(f"{label} checksum mismatch")
    return value


def _decimal(value: Any, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise FollowupError(f"{label} is invalid") from exc
    if not result.is_finite() or result < 0:
        raise FollowupError(f"{label} is invalid")
    return result


def _money(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.00000000001")), "f").rstrip("0").rstrip(".") or "0"


def _code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "jev_followup.py", "jev_replay.py", "sh08_matrix_models.py",
        "sh08_matrix_transport.py", "shortening_prompts.py", "shortening_resume.py",
        "length_check.py", "deepseek_transport.py", "llm_types.py",
    )
    result: dict[str, str] = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FollowupError(f"required code file is missing: {name}")
        result[name] = _sha_file(path)
    return result


def _source_artifact_hashes(source_dir: Path) -> dict[str, str]:
    # The follow-up source is deliberately the original manifest plus its
    # complete ledger.  The old HTML/report are derived data and are not an
    # input to selection.
    names = ("manifest.json", "ledger.json")
    result: dict[str, str] = {}
    for name in names:
        path = source_dir / name
        if not path.is_file():
            raise FollowupError(f"source artifact is missing: {name}")
        result[name] = _sha_file(path)
    return result


def _source_cases(source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cases = source.get("manifest", {}).get("cases")
    if not isinstance(cases, list) or any(not isinstance(row, Mapping) or not isinstance(row.get("case_id"), str) for row in cases):
        raise FollowupError("source manifest cases are invalid")
    return {str(row["case_id"]): deepcopy(dict(row)) for row in cases}


def _source_requests(source: Mapping[str, Any]) -> dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]]:
    manifest = source["manifest"]
    ledger = source["ledger"]
    descriptors = manifest.get("requests")
    rows = ledger.get("requests")
    if not isinstance(descriptors, list) or len(descriptors) != 48 or not isinstance(rows, Mapping) or len(rows) != 48:
        raise FollowupError("source replay must contain exactly 48 requests")
    result: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    for descriptor_value in descriptors:
        if not isinstance(descriptor_value, Mapping):
            raise FollowupError("source descriptor is invalid")
        descriptor = deepcopy(dict(descriptor_value))
        case_id = descriptor.get("case_id")
        order = descriptor.get("order")
        payload = descriptor.get("payload")
        if not isinstance(case_id, str) or order not in (1, 2) or not isinstance(payload, Mapping):
            raise FollowupError("source descriptor identity is invalid")
        if canonical_hash(payload) != descriptor.get("body_hash"):
            raise FollowupError(f"source payload hash mismatch: {case_id}/{order}")
        key = jev._request_key(descriptor)
        row = rows.get(key)
        if not isinstance(row, Mapping) or row.get("status") != "complete":
            raise FollowupError(f"source request is not complete: {case_id}/{order}")
        response = row.get("response")
        raw = response.get("json") if isinstance(response, Mapping) else None
        answers = row.get("answers")
        if not isinstance(raw, Mapping) or not isinstance(answers, Mapping):
            raise FollowupError(f"source receipt is incomplete: {case_id}/{order}")
        parsed = jev.parse_jev_response(raw, payload.get("questions", {}))
        if parsed != answers:
            raise FollowupError(f"source stored answers mismatch: {case_id}/{order}")
        identity = (case_id, int(order))
        if identity in result:
            raise FollowupError(f"duplicate source request: {case_id}/{order}")
        result[identity] = (descriptor, deepcopy(dict(row)))
    expected = {(case_id, order) for case_id in _source_cases(source) for order in (1, 2)}
    if set(result) != expected:
        raise FollowupError("source requests do not cover every case in both orders")
    return result


def _load_source(source_dir: Path) -> dict[str, Any]:
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FollowupError(f"source directory is missing: {source_dir}")
    actual_artifacts = _source_artifact_hashes(source_dir)
    manifest = _envelope(_load(source_dir / "manifest.json", "source manifest"), "source manifest")
    ledger = _envelope(_load(source_dir / "ledger.json", "source ledger"), "source ledger")
    if ledger.get("manifest_checksum") != manifest.get("checksum"):
        raise FollowupError("source ledger is not bound to source manifest")
    if manifest.get("schema") != jev.SCHEMA or manifest.get("model") != JEV_MODEL:
        raise FollowupError("source Jev manifest identity is invalid")
    if len(manifest.get("requests", [])) != 48:
        raise FollowupError("source manifest is not the complete 48-request replay")
    source = {"manifest": manifest, "ledger": ledger, "artifact_hashes": actual_artifacts}
    _source_requests(source)
    return source


def _case_decision(case: Mapping[str, Any], descriptor: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    response = row["response"]["json"]
    answers = jev.parse_jev_response(response, descriptor["payload"]["questions"])
    decision = jev.decide(case, answers)
    best = answers.get("best", {}).get("choice") if isinstance(answers.get("best"), Mapping) else None
    safety = answers.get(f"safety_{best}", {}) if isinstance(best, str) else {}
    return {
        "answers": answers,
        "accepted": bool(decision.get("accepted")),
        "selected_candidate_id": decision.get("selected_candidate_id"),
        "accepted_text": decision.get("accepted_text"),
        "best": best,
        "safety_choice": safety.get("choice") if isinstance(safety, Mapping) else None,
        "receipt_id": row.get("receipt_id"),
    }


def _derive_selection(source: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = _source_cases(source)
    requests = _source_requests(source)
    selected: list[dict[str, Any]] = []
    reasons: dict[str, list[str]] = {}
    historical: dict[str, dict[int, dict[str, Any]]] = {}
    for case_id in sorted(cases):
        case = cases[case_id]
        pair: dict[int, dict[str, Any]] = {}
        for order in (1, 2):
            descriptor, row = requests[(case_id, order)]
            pair[order] = _case_decision(case, descriptor, row)
        historical[case_id] = pair
        why: list[str] = []
        if any(item.get("best") in {candidate.get("id") for candidate in case.get("candidates", []) if isinstance(candidate, Mapping)}
               and item.get("safety_choice") == "reject" for item in pair.values()):
            why.append("best_candidate_safety_reject")
        accepted_count = sum(1 for item in pair.values() if item["accepted"])
        if accepted_count == 1:
            why.append("accepted_in_exactly_one_order")
        if accepted_count == 0:
            why.append("neither_order_accepted")
        if case_id in USER_CONTROLS:
            why.append("explicit_user_control")
        if why:
            reasons[case_id] = why
            selected.append(deepcopy(case))
    selected_ids = {str(case["case_id"]) for case in selected}
    if selected_ids != set(EXPECTED_CASES):
        raise FollowupError(f"selection IDs differ from frozen protocol: {sorted(selected_ids)}")
    for case in selected:
        case_id = str(case["case_id"])
        candidates = case.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != EXPECTED_CASES[case_id]:
            raise FollowupError(f"candidate count differs for {case_id}")
    return selected, {"reasons": reasons, "historical": historical, "user_controls": sorted(USER_CONTROLS)}


def _candidate_by_id(case: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    candidates = case.get("candidates")
    if not isinstance(candidates, list):
        raise FollowupError("case candidates are missing")
    matches = [item for item in candidates if isinstance(item, Mapping) and item.get("id") == candidate_id]
    if len(matches) != 1:
        raise FollowupError(f"candidate is not unique: {candidate_id}")
    return deepcopy(dict(matches[0]))


def _old_order(source: Mapping[str, Any], case_id: str, order: int) -> dict[str, Any]:
    descriptor, _ = _source_requests(source)[(case_id, order)]
    return descriptor


def _validate_candidate_window(state: Mapping[str, Any], candidate: Mapping[str, Any], window: Any) -> None:
    original = state.get("original_window")
    position = state.get("target_position")
    if not isinstance(original, list) or not isinstance(window, list) or window != window or window.__class__ is not list:
        raise FollowupError("candidate window shape is invalid")
    if window.__len__() != original.__len__() or isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < len(original):
        raise FollowupError("candidate window target is invalid")
    for index, (old, new) in enumerate(zip(original, window)):
        if not isinstance(old, Mapping) or not isinstance(new, Mapping):
            raise FollowupError("candidate window row is invalid")
        if index == position:
            if new.get("text") != candidate.get("text") or new.get("marked_target") is not True:
                raise FollowupError("candidate window target is not frozen")
            old_copy, new_copy = dict(old), dict(new)
            old_copy.pop("text", None)
            new_copy.pop("text", None)
            if old_copy != new_copy:
                raise FollowupError("candidate window changes more than target text")
        elif dict(old) != dict(new):
            raise FollowupError("candidate window changes a neighbor")


def _repeat_record(case: Mapping[str, Any], descriptor: Mapping[str, Any], old_payload_hash: str) -> dict[str, Any]:
    payload = deepcopy(descriptor["payload"])
    return {
        "case_id": case["case_id"], "source": case["source"], "start": case["start"], "end": case["end"],
        "request_kind": "repeat", "order": descriptor["order"], "candidate_id": None,
        "original": case["original"], "context_original_window": deepcopy(case["context"]),
        "target_position": case["target_position"], "max_chars": case["max_chars"], "length_basis": case["length_basis"],
        "candidates": deepcopy(case["candidates"]), "model": JEV_MODEL, "provider": "typesafe",
        "model_id": JEV_MODEL, "mode": "Choice", "status": "pending", "answers": {},
        "verdict": None, "selected_candidate_id": None, "accepted_text": None,
        "receipt_id": None, "error": None, "old_payload_hash": old_payload_hash, "payload_hash_old": old_payload_hash,
        "payload_hash_new": canonical_hash(payload),
    }


def _isolated_descriptor(case: Mapping[str, Any], old: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(old["payload"]))
    state = payload.get("state")
    questions = payload.get("questions")
    candidate_id = candidate.get("id")
    if not isinstance(state, Mapping) or not isinstance(questions, Mapping) or not isinstance(candidate_id, str):
        raise FollowupError("source Jev payload is incomplete")
    old_windows = state.get("candidate_windows")
    matching = [item for item in old_windows if isinstance(item, Mapping) and item.get("candidate_id") == candidate_id] if isinstance(old_windows, list) else []
    if len(matching) != 1:
        raise FollowupError(f"source candidate window is missing: {candidate_id}")
    safety_key = f"safety_{candidate_id}"
    if safety_key not in questions:
        raise FollowupError(f"source safety question is missing: {safety_key}")
    _validate_candidate_window(state, candidate, matching[0].get("window"))
    isolated_state = deepcopy(dict(state))
    isolated_state["candidate_windows"] = [deepcopy(dict(matching[0]))]
    fit = state.get("precomputed_length_fit")
    if not isinstance(fit, Mapping) or candidate_id not in fit:
        raise FollowupError("source precomputed length fit is missing")
    isolated_state["precomputed_length_fit"] = {candidate_id: deepcopy(fit[candidate_id])}
    payload["state"] = isolated_state
    payload["questions"] = {safety_key: deepcopy(questions[safety_key])}
    payload["model"] = JEV_MODEL
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > PAYLOAD_LIMIT:
        raise FollowupError("isolated payload exceeds sanity limit")
    return {
        "case_id": case["case_id"], "order": None, "candidate_id": candidate_id,
        "request_kind": "isolated_safety", "payload": payload,
        "body_hash": canonical_hash(payload), "old_payload_hash": canonical_hash(old["payload"]),
        "payload_hash_old": canonical_hash(old["payload"]), "payload_hash_new": canonical_hash(payload),
    }


def _review_descriptor(case: Mapping[str, Any], old: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    payload = old["payload"]
    state = payload.get("state") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping) or not isinstance(state.get("original_window"), list):
        raise FollowupError("source Jev state is incomplete")
    candidate_id = candidate.get("id")
    if not isinstance(candidate_id, str):
        raise FollowupError("candidate ID is invalid")
    context: list[dict[str, Any]] = []
    target_position = state.get("target_position")
    center_count = 0
    for row in state["original_window"]:
        if not isinstance(row, Mapping) or not isinstance(row.get("cue_id"), str):
            raise FollowupError("source context row is invalid")
        item = deepcopy(dict(row))
        marked = item.pop("marked_target", False)
        if marked is True:
            center_count += 1
            item["cue_id"] = f"{case['case_id']}::{candidate_id}"
            item["is_target"] = True
        elif "marked_target" in item:
            raise FollowupError("source context marker is invalid")
        context.append(item)
    if center_count != 1 or not isinstance(target_position, int) or not 0 <= target_position < len(context) or context[target_position].get("is_target") is not True:
        raise FollowupError("source context has no unique marked target")
    cue_id = f"{case['case_id']}::{candidate_id}"
    target = {
        "cue_id": cue_id, "original": state.get("original_target", case["original"]), "candidate": candidate["text"],
        "max_chars": state.get("max_chars", case["max_chars"]), "actual_chars": candidate["actual_chars"],
        "length_basis": state.get("length_basis", case["length_basis"]),
    }
    if target["original"] != case["original"] or target["max_chars"] != case["max_chars"] or target["length_basis"] != case["length_basis"]:
        raise FollowupError("source review target contract differs")
    if check_length(str(candidate["text"]), int(target["max_chars"]), str(target["length_basis"]))["length"] != candidate["actual_chars"]:
        raise FollowupError("candidate actual length is not source-derived")
    user_prompt = build_review_prompt(context, [target])
    schema = review_schema()
    return {
        "case_id": case["case_id"], "order": None, "candidate_id": candidate_id,
        "request_kind": "independent_review", "system_prompt": ARBITER_SYSTEM_PROMPT,
        "user_prompt": user_prompt, "schema": schema, "prompt_version": PROMPT_VERSION,
        "model": DEEPSEEK_MODEL, "mode": DEEPSEEK_MODE, "spec": deepcopy(MODEL_SPECS["deepseek"]),
        "prompt_hash": canonical_hash({"system": ARBITER_SYSTEM_PROMPT, "user": user_prompt, "schema": schema, "version": PROMPT_VERSION}),
        "target": target, "context": context, "target_position": target_position,
        "old_payload_hash": canonical_hash(payload),
    }


def _request_key(descriptor: Mapping[str, Any]) -> str:
    return canonical_hash({key: descriptor.get(key) for key in ("request_kind", "case_id", "order", "candidate_id")})


def _save_ledger(output_dir: Path, ledger: Mapping[str, Any]) -> None:
    body = deepcopy(dict(ledger))
    body.pop("checksum", None)
    body["checksum"] = canonical_hash(body)
    atomic_json(Path(output_dir) / "ledger.json", body)


def _manifest_checksum(body: Mapping[str, Any]) -> str:
    return canonical_hash({key: value for key, value in body.items() if key != "checksum"})


def _freeze_descriptor(case: Mapping[str, Any], descriptor: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(descriptor))
    result["record"] = deepcopy(dict(record))
    result["descriptor_hash"] = canonical_hash({key: value for key, value in result.items() if key != "descriptor_hash"})
    return result


def _validate_manifest_static(manifest: Mapping[str, Any], output_dir: Path) -> None:
    if manifest.get("schema") != SCHEMA or manifest.get("version") != VERSION:
        raise FollowupError("follow-up manifest schema is invalid")
    if manifest.get("model") != JEV_MODEL or manifest.get("deepseek_spec") != MODEL_SPECS["deepseek"]:
        raise FollowupError("follow-up model contract changed")
    if manifest.get("requests_sha256") != canonical_hash(manifest.get("requests")):
        raise FollowupError("follow-up requests hash mismatch")
    if manifest.get("checksum") != _manifest_checksum(manifest):
        raise FollowupError("follow-up manifest checksum mismatch")
    limits = manifest.get("limits")
    expected_limits = {
        "jev_cap_usd": _money(JEV_CAP), "jev_max_requests": JEV_MAX_REQUESTS,
        "jev_reservation_usd": _money(JEV_RESERVATION), "deepseek_cap_usd": _money(DEEPSEEK_CAP),
        "deepseek_max_requests": DEEPSEEK_MAX_REQUESTS, "deepseek_reservation_usd": _money(DEEPSEEK_RESERVATION),
    }
    if limits != expected_limits:
        raise FollowupError("follow-up budget limits changed")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise FollowupError("follow-up source reference is missing")
    source_dir = Path(str(source.get("directory", "")))
    actual = _source_artifact_hashes(source_dir)
    if actual != source.get("artifact_hashes"):
        raise FollowupError("source artifact changed")
    if output_dir.stat().st_mode & 0o077:
        raise FollowupError("follow-up directory is not private")


def _load_policy_file(output_dir: Path) -> dict[str, Any] | None:
    path = Path(output_dir) / INVALID_ANSWER_POLICY_FILE
    if not path.is_file():
        return None
    return _envelope(_load(path, "invalid-answer policy"), "invalid-answer policy")


def _load_continuation_file(output_dir: Path) -> dict[str, Any] | None:
    path = Path(output_dir) / "continuation.json"
    if not path.is_file():
        return None
    return _envelope(_load(path, "continuation"), "continuation")


def _validate_continuation_code_migration(manifest: Mapping[str, Any], output_dir: Path) -> dict[str, Any] | None:
    frozen = manifest.get("code_hashes")
    current = _code_hashes()
    if frozen == current:
        return _load_continuation_file(output_dir)
    approval = _load_continuation_file(output_dir)
    if approval is None:
        raise FollowupError("follow-up code changed without continuation approval")
    if approval.get("original_manifest_checksum") != manifest.get("checksum"):
        raise FollowupError("continuation is not bound to the original manifest")
    if approval.get("old_code_hashes") != frozen or approval.get("new_code_hashes") != current:
        raise FollowupError("continuation code hash migration is invalid")
    if not isinstance(frozen, Mapping) or not isinstance(current, Mapping) or set(frozen) != set(current):
        raise FollowupError("continuation code hash map is invalid")
    changed = {key for key in frozen if frozen[key] != current[key]}
    if changed != {CONTINUATION_CODE_FILE}:
        raise FollowupError("continuation changed code outside utils/jev_followup.py")
    return approval


def _validate_manifest(manifest: Mapping[str, Any], output_dir: Path) -> None:
    _validate_manifest_static(manifest, output_dir)
    if _load_policy_file(output_dir) is None:
        _validate_continuation_code_migration(manifest, output_dir)


def _validate_ledger(manifest: Mapping[str, Any], ledger: Mapping[str, Any]) -> None:
    if ledger.get("manifest_checksum") != manifest.get("checksum"):
        raise FollowupError("follow-up ledger is not bound to manifest")
    requests = manifest.get("requests")
    ledger_rows = ledger.get("requests")
    if not isinstance(requests, list) or not isinstance(ledger_rows, Mapping):
        raise FollowupError("follow-up request ledger is invalid")
    expected = {_request_key(item): item for item in requests if isinstance(item, Mapping)}
    if len(expected) != len(requests) or set(expected) != set(ledger_rows):
        raise FollowupError("follow-up ledger request set differs from manifest")
    for key, row in ledger_rows.items():
        if not isinstance(row, Mapping) or row.get("descriptor") != expected[key]:
            raise FollowupError("follow-up ledger descriptor mismatch")
        if row.get("status") not in {"pending", "started", "complete", "error", "unknown"}:
            raise FollowupError("follow-up ledger status is invalid")


def _load_prepared(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    output_dir = Path(output_dir)
    manifest = _envelope(_load(output_dir / "manifest.json", "follow-up manifest"), "follow-up manifest")
    _validate_manifest(manifest, output_dir)
    ledger = _envelope(_load(output_dir / "ledger.json", "follow-up ledger"), "follow-up ledger")
    _validate_ledger(manifest, ledger)
    policy = _load_policy_file(output_dir)
    if policy is not None:
        _validate_policy(manifest, ledger, output_dir, policy)
    else:
        _validate_continuation(manifest, ledger, output_dir)
    return dict(manifest), dict(ledger)


def _record_for(manifest: Mapping[str, Any], descriptor: Mapping[str, Any]) -> dict[str, Any]:
    record = descriptor.get("record")
    if not isinstance(record, Mapping):
        raise FollowupError("request record is missing")
    return deepcopy(dict(record))


def _receipt_id(receipt: Any) -> str | None:
    if not isinstance(receipt, Mapping):
        return None
    return canonical_hash(receipt)


def _save_current_receipt(row: dict[str, Any], receipt: Any) -> None:
    """Persist the current transport receipt before inspecting its contents."""
    if isinstance(receipt, Mapping):
        saved = deepcopy(dict(receipt))
    else:
        saved = {"raw": deepcopy(receipt)}
    row["response"] = saved
    row["receipt_id"] = _receipt_id(saved)


def _jev_usage(response: Mapping[str, Any]) -> tuple[dict[str, int], Decimal]:
    inputs, outputs, cost = jev._usage(response)
    return {"input_tokens": inputs, "output_tokens": outputs}, cost


def _invalid_jev_row(descriptor: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[str, str, list[str]]:
    if row.get("status") != "error":
        raise FollowupError("continuation target is not an error row")
    if row.get("provider") != "jev":
        raise FollowupError("continuation target is not a Jev row")
    if row.get("attempts") != 1:
        raise FollowupError("continuation target must have exactly one attempt")
    if row.get("error") != INVALID_JEV_ERROR:
        raise FollowupError("continuation target error is not the known Jev validation error")
    receipt = row.get("response")
    if not isinstance(receipt, Mapping) or receipt.get("status_code") != 200:
        raise FollowupError("continuation target has no complete HTTP 200 receipt")
    raw = receipt.get("json")
    if not isinstance(raw, Mapping) or raw.get("model") != JEV_MODEL:
        raise FollowupError("continuation target response model is invalid")
    try:
        usage, cost = _jev_usage(raw)
    except jev.JevReplayError as exc:
        raise FollowupError(f"continuation target usage is invalid: {exc}") from exc
    if row.get("usage") != usage or row.get("cost_usd") != _money(cost) or row.get("cost_kind") != "estimated_usage":
        raise FollowupError("continuation target usage or charged cost is invalid")
    if row.get("receipt_id") != _receipt_id(receipt):
        raise FollowupError("continuation target receipt checksum is invalid")
    answers = raw.get("answers")
    questions = descriptor.get("payload", {}).get("questions") if isinstance(descriptor.get("payload"), Mapping) else None
    if not isinstance(answers, Mapping) or not isinstance(questions, Mapping) or set(answers) != set(questions):
        raise FollowupError("continuation target answer IDs are invalid")
    offending: list[str] = []
    for question_id, question in questions.items():
        one_question = {question_id: question}
        one_answer = {question_id: answers.get(question_id)}
        candidate = {"model": raw.get("model"), "answers": one_answer, "usage": raw.get("usage")}
        try:
            jev.parse_jev_response(candidate, one_question)
        except jev.JevReplayError as exc:
            if str(exc) == INVALID_JEV_ERROR:
                offending.append(str(question_id))
            else:
                raise FollowupError(f"continuation target has another Jev schema error: {exc}") from exc
    if len(offending) != 1:
        raise FollowupError("continuation target has no choice/probability mismatch")
    return str(row["receipt_id"]), canonical_hash(raw), offending


def _validate_continuation(
    manifest: Mapping[str, Any],
    ledger: Mapping[str, Any],
    output_dir: Path,
    *,
    effective_code_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    approval = _load_continuation_file(output_dir)
    if approval is None:
        return None
    if approval.get("schema") != CONTINUATION_SCHEMA:
        raise FollowupError("continuation schema is invalid")
    if approval.get("run_id") != manifest.get("run_id"):
        raise FollowupError("continuation run identity is invalid")
    if approval.get("original_manifest_checksum") != manifest.get("checksum"):
        raise FollowupError("continuation manifest checksum is invalid")
    frozen_code_hashes = manifest.get("code_hashes")
    current_code_hashes = dict(effective_code_hashes) if effective_code_hashes is not None else _code_hashes()
    if approval.get("old_code_hashes") != frozen_code_hashes or approval.get("new_code_hashes") != current_code_hashes:
        raise FollowupError("continuation code hash migration is invalid")
    if not isinstance(frozen_code_hashes, Mapping) or not isinstance(current_code_hashes, Mapping):
        raise FollowupError("continuation code hash map is invalid")
    changed_code = {key for key in frozen_code_hashes if frozen_code_hashes[key] != current_code_hashes[key]}
    if changed_code - {CONTINUATION_CODE_FILE}:
        raise FollowupError("continuation changed code outside utils/jev_followup.py")
    if approval.get("original_manifest_file_sha256") != _sha_file(Path(output_dir) / "manifest.json"):
        raise FollowupError("original manifest file changed")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or approval.get("original_source_artifact_hashes") != source.get("artifact_hashes"):
        raise FollowupError("continuation source identity is invalid")
    if approval.get("original_source_manifest_checksum") != source.get("manifest_checksum"):
        raise FollowupError("continuation source manifest checksum is invalid")
    if approval.get("caps") != manifest.get("limits"):
        raise FollowupError("continuation caps differ from the original campaign")
    if approval.get("reason") != CONTINUATION_REASON:
        raise FollowupError("continuation reason is invalid")
    acknowledged = approval.get("acknowledged_request_ids")
    if not isinstance(acknowledged, list) or len(acknowledged) != 1:
        raise FollowupError("continuation must acknowledge exactly one request")
    item = acknowledged[0]
    if not isinstance(item, Mapping) or not isinstance(item.get("request_id"), str):
        raise FollowupError("continuation request binding is invalid")
    request_id = str(item["request_id"])
    row = ledger.get("requests", {}).get(request_id) if isinstance(ledger.get("requests"), Mapping) else None
    descriptor = next((value for value in manifest.get("requests", [])
                       if isinstance(value, Mapping) and _request_key(value) == request_id), None)
    if not isinstance(row, Mapping) or not isinstance(descriptor, Mapping):
        raise FollowupError("continuation request is not in the prepared ledger")
    receipt_id, response_hash, offending = _invalid_jev_row(descriptor, row)
    if item.get("error_receipt_id") != receipt_id or item.get("canonical_response_hash") != response_hash:
        raise FollowupError("continuation receipt binding is invalid")
    if approval.get("offending_question_ids") != offending:
        raise FollowupError("continuation offending answer record is invalid")
    return approval


def _policy_rows(manifest: Mapping[str, Any], ledger: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any], str]]:
    rows: list[tuple[str, Mapping[str, Any], Mapping[str, Any], str]] = []
    descriptors = manifest.get("requests")
    ledger_rows = ledger.get("requests")
    if not isinstance(descriptors, list) or not isinstance(ledger_rows, Mapping):
        raise FollowupError("invalid-answer policy request identities are invalid")
    by_key = {_request_key(item): item for item in descriptors if isinstance(item, Mapping)}
    for request_id, row in ledger_rows.items():
        descriptor = by_key.get(request_id)
        if not isinstance(descriptor, Mapping) or not isinstance(row, Mapping):
            raise FollowupError("invalid-answer policy request identity is invalid")
        rows.append((str(request_id), descriptor, row, str(row.get("status"))))
    return rows


def _validate_policy(
    manifest: Mapping[str, Any], ledger: Mapping[str, Any], output_dir: Path, policy: Mapping[str, Any],
) -> set[str]:
    if policy.get("schema") != INVALID_ANSWER_POLICY_SCHEMA or policy.get("policy_name") != INVALID_ANSWER_POLICY_NAME:
        raise FollowupError("invalid-answer policy identity is invalid")
    if policy.get("main_reason") != INVALID_ANSWER_POLICY_REASON:
        raise FollowupError("invalid-answer policy reason is invalid")
    policy_path = Path(output_dir) / INVALID_ANSWER_POLICY_FILE
    if policy_path.stat().st_mode & 0o777 != 0o600:
        raise FollowupError("invalid-answer policy permissions are invalid")
    if policy.get("run_id") != manifest.get("run_id") or policy.get("original_manifest_checksum") != manifest.get("checksum"):
        raise FollowupError("invalid-answer policy manifest binding is invalid")
    manifest_path = Path(output_dir) / "manifest.json"
    if policy.get("original_manifest_file_sha256") != _sha_file(manifest_path):
        raise FollowupError("invalid-answer policy manifest file binding is invalid")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise FollowupError("invalid-answer policy source binding is invalid")
    if policy.get("original_source_artifact_hashes") != source.get("artifact_hashes"):
        raise FollowupError("invalid-answer policy source artifact binding is invalid")
    if policy.get("original_source_manifest_checksum") != source.get("manifest_checksum"):
        raise FollowupError("invalid-answer policy source manifest binding is invalid")
    if policy.get("limits") != manifest.get("limits"):
        raise FollowupError("invalid-answer policy limits differ from the original campaign")
    continuation_path = Path(output_dir) / "continuation.json"
    if not continuation_path.is_file() or policy.get("previous_continuation_file_sha256") != _sha_file(continuation_path):
        raise FollowupError("invalid-answer policy continuation binding is invalid")
    continuation = _load_continuation_file(output_dir)
    if continuation is None or policy.get("previous_continuation_checksum") != continuation.get("checksum"):
        raise FollowupError("invalid-answer policy continuation checksum is invalid")
    previous_code_hashes = continuation.get("new_code_hashes")
    if policy.get("previous_approval_code_hashes") != previous_code_hashes or not isinstance(previous_code_hashes, Mapping):
        raise FollowupError("invalid-answer policy historical approval binding is invalid")
    _validate_continuation(manifest, ledger, output_dir, effective_code_hashes=previous_code_hashes)
    current_code_hashes = _code_hashes()
    if policy.get("current_code_hashes") != current_code_hashes:
        raise FollowupError("invalid-answer policy current code hash binding is invalid")
    if set(previous_code_hashes) != set(current_code_hashes):
        raise FollowupError("invalid-answer policy code hash map is invalid")
    changed = {key for key in current_code_hashes if previous_code_hashes[key] != current_code_hashes[key]}
    if changed - {CONTINUATION_CODE_FILE}:
        raise FollowupError("invalid-answer policy changed code outside utils/jev_followup.py")

    listed = policy.get("currently_quarantined")
    if not isinstance(listed, list) or any(not isinstance(item, Mapping) for item in listed):
        raise FollowupError("invalid-answer policy quarantined rows are invalid")
    listed_ids = {str(item.get("request_id")) for item in listed}
    if len(listed_ids) != len(listed):
        raise FollowupError("invalid-answer policy quarantined rows are duplicated")
    actual_invalid: set[str] = set()
    for request_id, descriptor, row, status in _policy_rows(manifest, ledger):
        if status in {"started", "unknown"}:
            raise FollowupError("invalid-answer policy cannot authorize started or unknown rows")
        if status == "error":
            try:
                _, response_hash, _ = _invalid_jev_row(descriptor, row)
            except FollowupError as exc:
                raise FollowupError(f"invalid-answer policy found an unauthorized error row: {exc}") from exc
            actual_invalid.add(request_id)
            item = next((value for value in listed if value.get("request_id") == request_id), None)
            if request_id in listed_ids and (not isinstance(item, Mapping) or item.get("raw_receipt_hash") != response_hash):
                raise FollowupError("invalid-answer policy raw receipt binding is invalid")
    if not listed_ids.issubset(actual_invalid):
        raise FollowupError("invalid-answer policy does not cover its quarantined rows")
    records = _refresh_records(manifest, ledger)
    for (_, _, _, status), record in zip(_policy_rows(manifest, ledger), records):
        if status == "complete" and record.get("status") == "error":
            raise FollowupError("invalid-answer policy found an unexpected stored response error")
    return actual_invalid


def _acknowledged_request_ids(manifest: Mapping[str, Any], ledger: Mapping[str, Any], output_dir: Path) -> set[str]:
    policy = _load_policy_file(output_dir)
    if policy is not None:
        return _validate_policy(manifest, ledger, output_dir, policy)
    approval = _validate_continuation(manifest, ledger, output_dir)
    if approval is None:
        return set()
    return {str(approval["acknowledged_request_ids"][0]["request_id"])}


def _deepseek_usage(receipt: Mapping[str, Any]) -> tuple[dict[str, int], Decimal] | None:
    cost_known = receipt.get("cost_known")
    if not isinstance(cost_known, bool):
        raise FollowupError("DeepSeek receipt cost_known is invalid")
    cost_value = receipt.get("cost_usd")
    if cost_value is not None:
        supplied_cost = _decimal(cost_value, "DeepSeek receipt cost")
    else:
        supplied_cost = None
    usage = receipt.get("usage")
    if not isinstance(usage, Mapping):
        if cost_known:
            raise FollowupError("DeepSeek receipt usage is missing")
        return None
    inputs, outputs = usage.get("input_tokens"), usage.get("output_tokens")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (inputs, outputs)):
        raise FollowupError("DeepSeek receipt usage is invalid")
    if not cost_known:
        return None
    if supplied_cost is None:
        raise FollowupError("DeepSeek receipt cost is missing")
    expected = (Decimal(inputs) * DEEPSEEK_INPUT_RATE + Decimal(outputs) * DEEPSEEK_OUTPUT_RATE) / Decimal(1_000_000)
    if not math.isclose(float(supplied_cost), float(expected), rel_tol=1e-9, abs_tol=1e-9):
        raise FollowupError("DeepSeek receipt cost does not match usage")
    return {"input_tokens": inputs, "output_tokens": outputs}, expected


def _admit(ledger: Mapping[str, Any], provider: str) -> None:
    cap = JEV_CAP if provider == "jev" else DEEPSEEK_CAP
    max_requests = JEV_MAX_REQUESTS if provider == "jev" else DEEPSEEK_MAX_REQUESTS
    reservation = JEV_RESERVATION if provider == "jev" else DEEPSEEK_RESERVATION
    attempted = 0
    spent = Decimal(0)
    unknown = Decimal(0)
    for row in ledger.get("requests", {}).values():
        if not isinstance(row, Mapping) or row.get("provider") != provider:
            continue
        attempted += int(row.get("attempts", 0) or 0)
        known = isinstance(row.get("usage"), Mapping) and row.get("cost_usd") is not None
        if known:
            spent += _decimal(row["cost_usd"], "known request cost")
        if row.get("status") in {"started", "unknown"} or row.get("status") == "error" and not known:
            unknown += reservation
    if attempted >= max_requests or spent + unknown + reservation > cap:
        raise FollowupError(f"{provider} budget or request cap reached")


def _record_error(record: dict[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    record["status"] = "error"
    record["error"] = str(row.get("error") or f"durable request state: {row.get('status')}")[:2000]
    record["receipt_id"] = row.get("receipt_id")
    return record


def _refresh_records(manifest: Mapping[str, Any], ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for descriptor in manifest["requests"]:
        key = _request_key(descriptor)
        row = ledger["requests"][key]
        record = _record_for(manifest, descriptor)
        status = row.get("status")
        record["receipt_id"] = row.get("receipt_id")
        if status == "complete":
            receipt = row.get("response")
            try:
                if descriptor["request_kind"] == "repeat":
                    if not isinstance(receipt, Mapping) or not isinstance(receipt.get("json"), Mapping):
                        raise FollowupError("complete Jev receipt has no JSON")
                    answers = jev.parse_jev_response(receipt["json"], descriptor["payload"]["questions"])
                    if answers != row.get("answers"):
                        raise FollowupError("stored Jev answers mismatch")
                    case = next(case for case in manifest["cases"] if case["case_id"] == descriptor["case_id"])
                    decision = jev.decide(case, answers)
                    record.update({"status": "accepted" if decision["accepted"] else "unresolved", "answers": answers,
                                   "selected_candidate_id": decision["selected_candidate_id"], "accepted_text": decision["accepted_text"],
                                   "verdict": answers.get(f"safety_{answers.get('best', {}).get('choice')}", {}).get("choice") if isinstance(answers.get("best"), Mapping) else None})
                elif descriptor["request_kind"] == "isolated_safety":
                    if not isinstance(receipt, Mapping) or not isinstance(receipt.get("json"), Mapping):
                        raise FollowupError("complete isolated receipt has no JSON")
                    answers = jev.parse_jev_response(receipt["json"], descriptor["payload"]["questions"])
                    if answers != row.get("answers"):
                        raise FollowupError("stored isolated answers mismatch")
                    record.update({"status": "evaluated", "answers": answers,
                                   "verdict": next(iter(answers.values()))["choice"]})
                else:
                    text = receipt.get("text") if isinstance(receipt, Mapping) else None
                    parsed = parse_review(text, [descriptor["target"]["cue_id"]]) if isinstance(text, str) else None
                    if parsed is None or parsed != row.get("review"):
                        raise FollowupError("stored DeepSeek review mismatch")
                    review = parsed[descriptor["target"]["cue_id"]]
                    record.update({"status": "evaluated", "verdict": review["verdict"], "issues": review["issues"], "advice": review["advice"]})
            except Exception as exc:
                record["status"] = "error"
                record["error"] = str(exc)[:2000]
        elif status in {"error", "unknown", "started"}:
            _record_error(record, row)
        records.append(record)
    return records


def _summary(manifest: Mapping[str, Any], ledger: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_backend: dict[str, dict[str, Any]] = {
        "jev": {"attempted": 0, "completed": 0, "input_tokens": 0, "output_tokens": 0,
                "cost_usd_estimated": "0", "unknown_reserved_usd": "0"},
        "deepseek": {"attempted": 0, "completed": 0, "input_tokens": 0, "output_tokens": 0,
                     "cost_usd": "0", "cost_kind": "estimated_usage", "unknown_reserved_usd": "0"},
    }
    for row in ledger["requests"].values():
        provider = row.get("provider")
        if provider not in by_backend:
            continue
        backend = by_backend[provider]
        if int(row.get("attempts", 0) or 0) > 0:
            backend["attempted"] += 1
        if row.get("status") == "complete":
            backend["completed"] += 1
        usage = row.get("usage")
        if isinstance(usage, Mapping):
            backend["input_tokens"] += int(usage.get("input_tokens", 0))
            backend["output_tokens"] += int(usage.get("output_tokens", 0))
        known = isinstance(usage, Mapping) and row.get("cost_usd") is not None
        if known:
            if provider == "jev":
                backend["cost_usd_estimated"] = _money(_decimal(backend["cost_usd_estimated"], "Jev summary") + _decimal(row["cost_usd"], "Jev request cost"))
            else:
                backend["cost_usd"] = _money(_decimal(backend["cost_usd"], "DeepSeek summary") + _decimal(row["cost_usd"], "DeepSeek request cost"))
        if row.get("status") in {"started", "unknown"} or (row.get("status") == "error" and not known):
            field = "unknown_reserved_usd"
            backend[field] = _money(_decimal(backend[field], "unknown reservation") + (JEV_RESERVATION if provider == "jev" else DEEPSEEK_RESERVATION))
    return {
        "cases": 6, "records": 58,
        "attempted_requests": sum(int(row.get("attempts", 0) or 0) for row in ledger["requests"].values()),
        "completed_requests": sum(row.get("status") == "complete" for row in ledger["requests"].values()),
        "errors": sum(row.get("status") in {"error", "unknown", "started"} for row in ledger["requests"].values()),
        "pending": sum(row.get("status") == "pending" for row in ledger["requests"].values()),
        "campaign_status": ledger.get("campaign_status"),
        "campaign_error": ledger.get("campaign_error"),
        "by_backend": by_backend,
    }


def refresh_report(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    manifest, ledger = _load_prepared(output_dir)
    policy = _load_policy_file(output_dir)
    records = _refresh_records(manifest, ledger)
    summary = _summary(manifest, ledger, records)
    acknowledged = _acknowledged_request_ids(manifest, ledger, output_dir)
    for descriptor, record in zip(manifest["requests"], records):
        if _request_key(descriptor) in acknowledged:
            record.update({"manual_acknowledgment": CONTINUATION_REASON, "manual_acknowledged": True, "not_retried": True,
                           "acknowledged_invalid_response": True})
    summary["acknowledged_invalid_responses"] = len(acknowledged)
    invalid_only = bool(acknowledged and summary["pending"] == 0 and summary["errors"] == len(acknowledged))
    summary["completed_with_invalid_response"] = invalid_only
    summary["clean_complete"] = bool(summary["pending"] == 0 and summary["errors"] == 0)
    summary["remaining_work"] = summary["pending"]
    attempted = summary["attempted_requests"]
    errors = summary["errors"]
    complete = summary["completed_requests"] == 58
    blocked = ledger.get("campaign_status") == "blocked" and (not acknowledged or summary["errors"] > len(acknowledged))
    status = "complete" if complete else "blocked" if blocked else "partial" if attempted else manifest.get("status", "prepared")
    campaign_status = ledger.get("campaign_status")
    if invalid_only:
        campaign_status = "completed_with_invalid_response"
    notices = deepcopy(manifest.get("notices", [])) if isinstance(manifest.get("notices"), list) else []
    if acknowledged:
        notices.append(CONTINUATION_NOTICE)
    if policy is not None:
        notices.append("Invalid-answer policy applied: quarantined Jev responses remain errors and are not semantic acceptance.")
    report = {
        "schema": SCHEMA, "run_id": manifest["run_id"], "status": status,
        "campaign_status": campaign_status, "campaign_error": ledger.get("campaign_error"),
        "backend_consumed": attempted > 0, "summary": summary, "records": records,
        "notices": notices,
        "invalid_answer_policy_applied": policy is not None,
    }
    atomic_json(Path(output_dir) / "report.json", report)
    return report


def prepare(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Freeze the six-case, 58-record campaign without contacting a provider."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    source_dir = Path(source_dir)
    source = _load_source(source_dir)
    cases, selection = _derive_selection(source)
    descriptors: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda item: str(item["case_id"])):
        case_id = str(case["case_id"])
        for order in (1, 2):
            old = _old_order(source, case_id, order)
            old_payload_hash = canonical_hash(old["payload"])
            record = _repeat_record(case, old, old_payload_hash)
            descriptor = {"request_kind": "repeat", "case_id": case_id, "order": order, "candidate_id": None,
                          "payload": deepcopy(old["payload"]), "body_hash": old_payload_hash, "old_payload_hash": old_payload_hash}
            descriptors.append(_freeze_descriptor(case, descriptor, record))
            records.append(record)
    for case in sorted(cases, key=lambda item: str(item["case_id"])):
        case_id = str(case["case_id"])
        old = _old_order(source, case_id, 1)
        for candidate in case["candidates"]:
            if not isinstance(candidate, Mapping):
                raise FollowupError("candidate is invalid")
            isolated = _isolated_descriptor(case, old, candidate)
            record = _repeat_record(case, {"payload": isolated["payload"], "order": None}, canonical_hash(old["payload"]))
            record.update({"request_kind": "isolated_safety", "order": None, "candidate_id": candidate["id"],
                           "candidates": [deepcopy(dict(candidate))], "context_original_window": deepcopy(old["payload"]["state"]["original_window"]),
                           "model": JEV_MODEL, "provider": "typesafe", "model_id": JEV_MODEL, "mode": "Choice"})
            descriptor = {key: value for key, value in isolated.items()}
            descriptors.append(_freeze_descriptor(case, descriptor, record))
            records.append(record)
    for case in sorted(cases, key=lambda item: str(item["case_id"])):
        case_id = str(case["case_id"])
        old = _old_order(source, case_id, 1)
        for candidate in case["candidates"]:
            if not isinstance(candidate, Mapping):
                raise FollowupError("candidate is invalid")
            review = _review_descriptor(case, old, candidate)
            record = _repeat_record(case, {"payload": old["payload"], "order": None}, canonical_hash(old["payload"]))
            record.update({"request_kind": "independent_review", "order": None, "candidate_id": candidate["id"],
                           "candidates": [deepcopy(dict(candidate))], "context_original_window": deepcopy(review["context"]),
                           "target_position": review["target_position"], "model": DEEPSEEK_MODEL, "provider": "deepseek",
                           "model_id": DEEPSEEK_MODEL, "mode": DEEPSEEK_MODE, "answers": {},
                           "issues": [], "advice": "", "old_payload_hash": canonical_hash(old["payload"])})
            descriptor = {key: value for key, value in review.items()}
            descriptors.append(_freeze_descriptor(case, descriptor, record))
            records.append(record)
    if len(records) != 58 or len(descriptors) != 58:
        raise FollowupError("follow-up descriptor count is not 58")
    run_id = _sha_bytes((SCHEMA + _now() + canonical_hash(descriptors)).encode())
    source_reference = {
        "directory": str(source_dir.resolve()), "artifact_hashes": source["artifact_hashes"],
        "manifest_checksum": source["manifest"]["checksum"],
        "baseline_cases": deepcopy(source["manifest"]["cases"]),
        "raw_receipts": [{"request_key": jev._request_key(descriptor), "receipt_id": row.get("receipt_id")}
                         for descriptor, row in _source_requests(source).values()],
    }
    body: dict[str, Any] = {
        "schema": SCHEMA, "version": VERSION, "run_id": run_id, "status": "prepared", "model": JEV_MODEL,
        "deepseek_spec": deepcopy(MODEL_SPECS["deepseek"]), "deepseek_model": DEEPSEEK_MODEL, "deepseek_mode": DEEPSEEK_MODE,
        "source": source_reference, "cases": cases, "selection": selection,
        "selection_predicate": "best candidate safety reject OR accepted in exactly one order OR neither order accepted",
        "user_controls_not_representative_sample": sorted(USER_CONTROLS), "requests": descriptors,
        "requests_sha256": canonical_hash(descriptors), "code_hashes": _code_hashes(),
        "limits": {"jev_cap_usd": _money(JEV_CAP), "jev_max_requests": JEV_MAX_REQUESTS, "jev_reservation_usd": _money(JEV_RESERVATION),
                    "deepseek_cap_usd": _money(DEEPSEEK_CAP), "deepseek_max_requests": DEEPSEEK_MAX_REQUESTS,
                    "deepseek_reservation_usd": _money(DEEPSEEK_RESERVATION)},
        "payload_identity": "canonical JSON identity; historical Jev wire bytes were not stored",
        "created_at": _now(), "notices": ["isolated safety changes candidate competition and question packing; it is not a pure causal context test."],
    }
    body["checksum"] = _manifest_checksum(body)
    output_dir.mkdir(parents=False, mode=0o700)
    output_dir.chmod(0o700)
    atomic_json(output_dir / "manifest.json", body)
    ledger: dict[str, Any] = {"schema": SCHEMA, "run_id": run_id, "manifest_checksum": body["checksum"], "campaign_status": "prepared", "requests": {}}
    for descriptor in descriptors:
        key = _request_key(descriptor)
        provider = "jev" if descriptor["request_kind"] in {"repeat", "isolated_safety"} else "deepseek"
        ledger["requests"][key] = {"key": key, "descriptor": descriptor, "provider": provider, "status": "pending", "attempts": 0,
                                    "usage": None, "cost_usd": None, "cost_kind": None, "response": None, "receipt_id": None, "answers": {}, "review": None, "error": None}
    _save_ledger(output_dir, ledger)
    return refresh_report(output_dir)


def _create_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing existing immutable file: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def authorize_continuation(output_dir: Path, invalid_request_id: str | Sequence[str]) -> dict[str, Any]:
    """Record one explicit, offline approval for a billed invalid Jev response."""
    output_dir = Path(output_dir)
    continuation_path = output_dir / "continuation.json"
    if continuation_path.exists():
        raise FileExistsError(f"refusing existing continuation: {continuation_path}")
    request_ids = [invalid_request_id] if isinstance(invalid_request_id, str) else list(invalid_request_id)
    if len(request_ids) != 1 or not isinstance(request_ids[0], str) or not request_ids[0]:
        raise FollowupError("exactly one invalid request ID must be authorized")
    if not output_dir.is_dir():
        raise FollowupError(f"follow-up directory is missing: {output_dir}")
    manifest = _envelope(_load(output_dir / "manifest.json", "follow-up manifest"), "follow-up manifest")
    _validate_manifest_static(manifest, output_dir)
    ledger = _envelope(_load(output_dir / "ledger.json", "follow-up ledger"), "follow-up ledger")
    _validate_ledger(manifest, ledger)
    request_id = request_ids[0]
    row = ledger["requests"].get(request_id)
    descriptor = next((value for value in manifest["requests"]
                       if isinstance(value, Mapping) and _request_key(value) == request_id), None)
    if not isinstance(row, Mapping) or not isinstance(descriptor, Mapping):
        raise FollowupError("invalid request ID is not in the prepared ledger")
    _, response_hash, offending = _invalid_jev_row(descriptor, row)
    old_code_hashes = manifest.get("code_hashes")
    new_code_hashes = _code_hashes()
    if not isinstance(old_code_hashes, Mapping) or set(old_code_hashes) != set(new_code_hashes):
        raise FollowupError("original code hash map is invalid")
    changed = {key for key in old_code_hashes if old_code_hashes[key] != new_code_hashes[key]}
    if changed != set() and changed != {CONTINUATION_CODE_FILE}:
        raise FollowupError("authorization detected code changes outside utils/jev_followup.py")
    body: dict[str, Any] = {
        "schema": CONTINUATION_SCHEMA,
        "run_id": manifest["run_id"],
        "original_manifest_checksum": manifest["checksum"],
        "original_manifest_file_sha256": _sha_file(output_dir / "manifest.json"),
        "original_source_artifact_hashes": deepcopy(manifest["source"]["artifact_hashes"]),
        "original_source_manifest_checksum": manifest["source"].get("manifest_checksum"),
        "old_code_hashes": deepcopy(dict(old_code_hashes)),
        "new_code_hashes": deepcopy(new_code_hashes),
        "acknowledged_request_ids": [{
            "request_id": request_id,
            "error_receipt_id": row["receipt_id"],
            "canonical_response_hash": response_hash,
        }],
        "offending_question_ids": offending,
        "reason": CONTINUATION_REASON,
        "caps": deepcopy(manifest["limits"]),
        "created_at": _now(),
    }
    body["checksum"] = canonical_hash(body)
    _create_immutable_json(continuation_path, body)
    return {"status": "authorized", "run_id": manifest["run_id"], "continuation": str(continuation_path)}


def authorize_invalid_answer_policy(output_dir: Path) -> dict[str, Any]:
    """Create the immutable offline policy for the currently billed invalid Jev rows."""
    output_dir = Path(output_dir)
    policy_path = output_dir / INVALID_ANSWER_POLICY_FILE
    if policy_path.exists():
        raise FileExistsError(f"refusing existing immutable file: {policy_path}")
    if not output_dir.is_dir():
        raise FollowupError(f"follow-up directory is missing: {output_dir}")
    manifest = _envelope(_load(output_dir / "manifest.json", "follow-up manifest"), "follow-up manifest")
    _validate_manifest_static(manifest, output_dir)
    ledger = _envelope(_load(output_dir / "ledger.json", "follow-up ledger"), "follow-up ledger")
    _validate_ledger(manifest, ledger)
    continuation_path = output_dir / "continuation.json"
    continuation = _load_continuation_file(output_dir)
    if continuation is None:
        raise FollowupError("invalid-answer policy requires the existing continuation approval")
    previous_code_hashes = continuation.get("new_code_hashes")
    if not isinstance(previous_code_hashes, Mapping):
        raise FollowupError("existing continuation approval code hashes are invalid")
    _validate_continuation(manifest, ledger, output_dir, effective_code_hashes=previous_code_hashes)
    current_code_hashes = _code_hashes()
    if set(previous_code_hashes) != set(current_code_hashes):
        raise FollowupError("invalid-answer policy code hash map is invalid")
    changed = {key for key in current_code_hashes if previous_code_hashes[key] != current_code_hashes[key]}
    if changed - {CONTINUATION_CODE_FILE}:
        raise FollowupError("authorization detected code changes outside utils/jev_followup.py")

    quarantined: list[dict[str, str]] = []
    invalid_count = 0
    for request_id, descriptor, row, status in _policy_rows(manifest, ledger):
        if status in {"started", "unknown"}:
            raise FollowupError("invalid-answer policy cannot authorize started or unknown rows")
        if status != "error":
            continue
        try:
            _, response_hash, _ = _invalid_jev_row(descriptor, row)
        except FollowupError as exc:
            raise FollowupError(f"invalid-answer policy found an unauthorized error row: {exc}") from exc
        quarantined.append({"request_id": request_id, "raw_receipt_hash": response_hash})
        invalid_count += 1
    if invalid_count != 2:
        raise FollowupError("invalid-answer policy requires exactly two existing invalid Jev rows")
    records = _refresh_records(manifest, ledger)
    for (_, _, _, status), record in zip(_policy_rows(manifest, ledger), records):
        if status == "complete" and record.get("status") == "error":
            raise FollowupError("invalid-answer policy found an unexpected stored response error")
    body: dict[str, Any] = {
        "schema": INVALID_ANSWER_POLICY_SCHEMA,
        "policy_name": INVALID_ANSWER_POLICY_NAME,
        "run_id": manifest["run_id"],
        "original_manifest_checksum": manifest["checksum"],
        "original_manifest_file_sha256": _sha_file(output_dir / "manifest.json"),
        "original_source_artifact_hashes": deepcopy(manifest["source"]["artifact_hashes"]),
        "original_source_manifest_checksum": manifest["source"].get("manifest_checksum"),
        "previous_continuation_file_sha256": _sha_file(continuation_path),
        "previous_continuation_checksum": continuation["checksum"],
        "previous_approval_code_hashes": deepcopy(dict(previous_code_hashes)),
        "current_code_hashes": deepcopy(current_code_hashes),
        "limits": deepcopy(manifest["limits"]),
        "currently_quarantined": quarantined,
        "main_reason": INVALID_ANSWER_POLICY_REASON,
        "created_at": _now(),
    }
    body["checksum"] = canonical_hash(body)
    _create_immutable_json(policy_path, body)
    return {"status": "authorized", "run_id": manifest["run_id"], "policy": str(policy_path)}


async def _run_async(output_dir: Path, max_new_requests: int) -> dict[str, Any]:
    with FileLock(str(Path(output_dir).resolve().parent / f".{Path(output_dir).name}.run.lock"), timeout=0):
        manifest, ledger = _load_prepared(Path(output_dir))
        policy = _load_policy_file(Path(output_dir))
        if manifest.get("status") != "prepared":
            raise FollowupError("follow-up manifest is not prepared")
        acknowledged = _acknowledged_request_ids(manifest, ledger, Path(output_dir))
        previous_error = next((row for request_id, row in ledger["requests"].items()
                               if isinstance(row, Mapping) and row.get("status") in {"started", "unknown", "error"}
                               and request_id not in acknowledged), None)
        if previous_error is not None or (ledger.get("campaign_status") == "blocked" and not acknowledged):
            if ledger.get("campaign_status") != "blocked":
                ledger["campaign_status"] = "blocked"
                ledger["campaign_error"] = "previous started/unknown/error request requires manual decision"
                _save_ledger(output_dir, ledger)
            return refresh_report(Path(output_dir))
        if max_new_requests == 0 or not any(row.get("status") == "pending" for row in ledger["requests"].values() if isinstance(row, Mapping)):
            return refresh_report(Path(output_dir))
        started_new = 0
        async with MatrixBackend() as matrix:
            for descriptor in manifest["requests"]:
                if started_new >= max_new_requests:
                    break
                key = _request_key(descriptor)
                row = ledger["requests"][key]
                if row["status"] != "pending":
                    continue
                provider = row["provider"]
                if provider == "deepseek":
                    preflight = matrix.preflight("deepseek")
                    if not preflight.get("ok"):
                        ledger["campaign_status"] = "blocked"
                        ledger["campaign_error"] = str(preflight.get("error") or "deepseek preflight failed")[:2000]
                        _save_ledger(output_dir, ledger)
                        break
                try:
                    _admit(ledger, provider)
                except Exception as exc:
                    ledger["campaign_status"] = "blocked"
                    ledger["campaign_error"] = str(exc)[:2000]
                    _save_ledger(output_dir, ledger)
                    break
                if provider == "jev":
                    try:
                        secret = jev._key()
                    except Exception as exc:
                        ledger["campaign_status"] = "blocked"
                        ledger["campaign_error"] = str(exc)[:2000]
                        _save_ledger(output_dir, ledger)
                        break
                else:
                    secret = None
                row["status"] = "started"
                row["attempts"] = int(row.get("attempts", 0)) + 1
                row["started_at"] = _now()
                _save_ledger(output_dir, ledger)
                started_new += 1
                receipt: Mapping[str, Any] | None = None
                try:
                    if provider == "jev":
                        raw, receipt_value = await jev._post(descriptor, str(secret))
                        receipt = receipt_value if isinstance(receipt_value, Mapping) else None
                        _save_current_receipt(row, receipt_value)
                        _save_ledger(output_dir, ledger)
                        usage, cost = _jev_usage(raw)
                        row.update({"usage": usage, "cost_usd": _money(cost), "cost_kind": "estimated_usage"})
                        _save_ledger(output_dir, ledger)
                        answers = jev.parse_jev_response(raw, descriptor["payload"]["questions"])
                        row.update({"status": "complete", "answers": answers})
                    else:
                        receipt_value = await matrix("deepseek", descriptor["system_prompt"], descriptor["user_prompt"], descriptor["schema"])
                        receipt = receipt_value if isinstance(receipt_value, Mapping) else None
                        _save_current_receipt(row, receipt_value)
                        _save_ledger(output_dir, ledger)
                        if not isinstance(receipt_value, Mapping):
                            raise FollowupError("DeepSeek receipt is invalid")
                        accounted = _deepseek_usage(receipt_value)
                        if accounted is not None:
                            usage, cost = accounted
                            row.update({"usage": usage, "cost_usd": _money(cost), "cost_kind": "estimated_usage"})
                        _save_ledger(output_dir, ledger)
                        if receipt_value.get("error"):
                            raise FollowupError(str(receipt_value["error"]))
                        text = receipt_value.get("text")
                        parsed = parse_review(text, [descriptor["target"]["cue_id"]]) if isinstance(text, str) else None
                        if parsed is None:
                            raise FollowupError("DeepSeek response text is missing")
                        row.update({"status": "complete", "review": parsed})
                    _save_ledger(output_dir, ledger)
                except (jev._ProviderError, jev._UsageUnavailable) as exc:
                    if isinstance(exc, jev._ProviderError) and exc.receipt is not None:
                        _save_current_receipt(row, exc.receipt)
                    row["status"] = "unknown" if getattr(exc, "unknown", False) or isinstance(exc, jev._UsageUnavailable) or row.get("cost_usd") is None else "error"
                    row["error"] = str(exc)[:2000]
                    _save_ledger(output_dir, ledger)
                    break
                except jev.JevReplayError as exc:
                    row["status"] = "error"
                    row["error"] = str(exc)[:2000]
                    _save_ledger(output_dir, ledger)
                    if str(exc) == INVALID_JEV_ERROR and policy is not None:
                        try:
                            _invalid_jev_row(descriptor, row)
                        except FollowupError:
                            pass
                        else:
                            continue
                    break
                except Exception as exc:
                    # A receipt without trusted usage cannot be priced; retain
                    # its reservation as unknown rather than counting zero.
                    row["status"] = "error" if row.get("cost_usd") is not None else "unknown"
                    row["error"] = str(exc)[:2000]
                    _save_ledger(output_dir, ledger)
                    break
    return refresh_report(Path(output_dir))


def run(output_dir: Path, *, max_new_requests: int = 1) -> dict[str, Any]:
    if isinstance(max_new_requests, bool) or not isinstance(max_new_requests, int) or max_new_requests < 0:
        raise ValueError("max_new_requests must be a nonnegative integer")
    with live_html_report(Path(output_dir)):
        return asyncio.run(_run_async(Path(output_dir), max_new_requests))


def report(output_dir: Path) -> dict[str, Any]:
    return refresh_report(Path(output_dir))


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Prepare, run, or report the bounded Jev follow-up")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--source-dir", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--max-new-requests", type=int, default=1)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--output-dir", type=Path, required=True)
    authorize_parser = sub.add_parser("authorize-continuation")
    authorize_parser.add_argument("--output-dir", type=Path, required=True)
    authorize_parser.add_argument("--invalid-request-id", action="append", required=True)
    policy_parser = sub.add_parser("authorize-invalid-answer-policy")
    policy_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source_dir, args.output_dir)
    elif args.command == "run":
        result = run(args.output_dir, max_new_requests=args.max_new_requests)
    elif args.command == "report":
        result = refresh_report(args.output_dir)
    elif args.command == "authorize-continuation":
        result = authorize_continuation(args.output_dir, args.invalid_request_id)
    else:
        result = authorize_invalid_answer_policy(args.output_dir)
    print(json.dumps({"status": result.get("status"), "run_id": result.get("run_id")}, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
