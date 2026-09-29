"""Offline shortening report loading and the real-run publication gate.

This module deliberately knows nothing about model transports.  Transport code
only calls :func:`note_model_request` while inside :func:`live_html_report`.
"""

from __future__ import annotations

import argparse
import contextlib
import contextvars
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
import re
import sys
import tempfile
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

HTML_MARKER = "<!-- subs-shortening-html:v1 -->"
HTML_NAME = "shortening.html"
SIDECAR_NAME = "html_report_metadata.json"
SIDECAR_SCHEMA = "shortening_html_metadata.v1"


class ShorteningHTMLUnknownFormat(ValueError):
    """Raised when a directory is not one of the supported saved formats."""


class _Observation:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.model_requests = 0


_ACTIVE: contextvars.ContextVar[_Observation | None] = contextvars.ContextVar(
    "shortening_html_observation", default=None
)


def note_model_request() -> None:
    """Record an actual model request when a live report scope is active."""

    observation = _ACTIVE.get()
    if observation is not None:
        observation.model_requests += 1


@contextlib.contextmanager
def live_html_report(output_dir: Path) -> Iterator[None]:
    """Publish once on exit when a scoped live transport attempted a request."""

    root = Path(output_dir).resolve()
    observation = _Observation(root)
    token = _ACTIVE.set(observation)
    try:
        yield
    finally:
        try:
            prior = _valid_sidecar(root)
            if observation.model_requests > 0 or prior:
                _publish_safely(root, confirmed_real_run=False)
        finally:
            _ACTIVE.reset(token)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"cannot read JSON artifact: {path.name}") from exc


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_json(value: Any) -> str:
    return _sha_bytes(_canonical(value).encode("utf-8"))


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _maybe_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _integer_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _list_strings(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _safe_error(value: Any) -> str:
    if isinstance(value, BaseException):
        value = type(value).__name__
    if not isinstance(value, str):
        return ""
    value = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [redacted]", value)
    value = re.sub(r"(?i)\bsk-[A-Za-z0-9_-]+", "sk-[redacted]", value)
    value = re.sub(r"(?i)(api[_ -]?key|token|authorization)\s*[:=]\s*[^\s,;]+", r"\1=[redacted]", value)
    return value[:240]


def _detail(label: Any, value: Any) -> dict[str, str]:
    labels = {"source": "Источник", "status": "Статус", "mode": "Режим", "settings": "Настройки",
              "cost": "Стоимость", "cost_status": "Статус стоимости", "remaining": "Осталось", "partial": "Частичный запуск",
              "requests": "Запросы", "cases": "Кейсы", "cost_estimated": "Оценка стоимости",
              "input_tokens": "Входные токены", "output_tokens": "Выходные токены",
              "reasoning_tokens": "Токены рассуждений", "cache_tokens": "Токены кэша",
              "physical": "Физические запросы",
              "limit": "Лимит", "unknown_reserved": "Неизвестный резерв",
              "next_reservation": "Следующий резерв", "stop_reason": "Причина остановки",
              "pricing_snapshot_id": "pricing_snapshot_id", "receipts": "Provider receipt IDs",
              "planned": "Запланировано", "attempted": "Попытки", "completed": "Завершено",
              "errors": "Ошибки", "pending": "Ожидают", "request_id": "Идентификатор запроса",
              "jev_attempted": "Jev: попытки", "jev_completed": "Jev: завершено",
              "jev_input_tokens": "Jev: входные токены", "jev_output_tokens": "Jev: выходные токены",
              "jev_cost_usd_estimated": "Jev: оценочная стоимость", "jev_cost_kind": "Jev: тип стоимости",
              "jev_reserved_usd": "Jev: неизвестный резерв", "deepseek_attempted": "DeepSeek: попытки",
              "deepseek_completed": "DeepSeek: завершено", "deepseek_input_tokens": "DeepSeek: входные токены",
              "deepseek_output_tokens": "DeepSeek: выходные токены", "deepseek_cost_usd": "DeepSeek: стоимость",
              "deepseek_cost_kind": "DeepSeek: тип стоимости", "deepseek_reserved_usd": "DeepSeek: неизвестный резерв"}
    raw_label = _string(label)
    if isinstance(value, (str, int, float, bool)) and not isinstance(value, str):
        value = str(value)
    display = _safe_error(value) if "error" in raw_label.lower() else _string(value)
    return {"label": labels.get(raw_label, raw_label), "value": display}


def _metadata(entries: Mapping[str, Any] | None) -> list[dict[str, str]]:
    if not isinstance(entries, Mapping):
        return []
    allowed = ("source", "status", "mode", "settings", "cost", "cost_status", "remaining", "partial", "limit", "unknown_reserved", "next_reservation", "stop_reason", "pricing_snapshot_id", "receipts", "subquota", "total", "logical", "physical", "requests", "cases", "cost_estimated", "input_tokens", "output_tokens", "reasoning_tokens", "cache_tokens", "planned", "attempted", "completed", "errors", "pending", "jev_attempted", "jev_completed", "jev_input_tokens", "jev_output_tokens", "jev_cost_usd_estimated", "jev_cost_kind", "jev_reserved_usd", "deepseek_attempted", "deepseek_completed", "deepseek_input_tokens", "deepseek_output_tokens", "deepseek_cost_usd", "deepseek_cost_kind", "deepseek_reserved_usd")
    return [_detail(key, entries[key]) for key in allowed if key in entries and isinstance(entries[key], (str, int, float, bool)) and (not isinstance(entries[key], str) or entries[key])]


def _candidate(
    candidate_id: Any = None,
    text: Any = None,
    *,
    phase: str = "initial",
    status: Any = None,
    max_chars: Any = None,
    fits: Any = None,
    selected: Any = False,
    local: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    saved_local = local if isinstance(local, Mapping) else status if isinstance(status, Mapping) else {}
    actual_fit = saved_local.get("fits", saved_local.get("within_length"))
    if actual_fit is None:
        actual_fit = fits
    limit = max_chars if isinstance(max_chars, int) and not isinstance(max_chars, bool) else saved_local.get("max_chars")
    body = _string(text)
    length = saved_local.get("length", saved_local.get("actual_chars"))
    if not isinstance(length, int) or isinstance(length, bool):
        length = None
    status_value = status.get("status") if isinstance(status, Mapping) else status
    return {
        "id": _string(candidate_id), "text": body, "phase": phase if phase in {"initial", "repair", "alternative"} else "initial",
        "status": _string(status_value),
        "length": length, "max_chars": limit if isinstance(limit, int) else None,
        "fits": actual_fit if isinstance(actual_fit, bool) else None,
        "selected": bool(selected),
    }


def _row(
    case_id: Any,
    source: Any,
    original: Any,
    *,
    start: Any = None,
    end: Any = None,
    repetition: Any = None,
    arm: Any = None,
    generator: Any = None,
    generator_mode: Any = None,
    arbiter: Any = None,
    arbiter_mode: Any = None,
    candidates: list[dict[str, Any]] | None = None,
    initial_verdict: Any = None,
    verdict: Any = None,
    accepted: Any = None,
    accepted_text: Any = None,
    status: Any = "unresolved",
    issues: Any = None,
    advice: Any = "",
    context: Any = None,
    receipt_ids: Any = None,
    details: Any = None,
    decision_kind: str | None = None,
    request_kind: str | None = None,
    candidate_id: Any = None,
) -> dict[str, Any]:
    candidate_rows = candidates or []
    return {
        "case_id": _string(case_id), "source": _string(source), "start": _integer_or_none(start), "end": _integer_or_none(end),
        "repetition": _integer_or_none(repetition), "arm": _string(arm), "original": _string(original),
        "generator": _string(generator), "generator_mode": _string(generator_mode), "arbiter": _string(arbiter), "arbiter_mode": _string(arbiter_mode),
        "decision_kind": decision_kind if decision_kind in {"arbiter", "selector", "unreviewed", "evaluation"} else ("arbiter" if _string(arbiter) else ("selector" if _string(arm) else "unreviewed")),
        "request_kind": request_kind if request_kind in {"repeat", "isolated_safety", "independent_review"} else None,
        "candidate_id": _string(candidate_id) if candidate_id is not None else None,
        "candidates": candidate_rows, "initial_verdict": _string(initial_verdict), "verdict": _string(verdict),
        "accepted": accepted if isinstance(accepted, bool) else None, "accepted_text": _string(accepted_text) if accepted_text is not None else None,
        "status": _string(status) or "unresolved", "issues": _list_strings(issues), "advice": _safe_error(advice),
        "context": [_context_item(item) for item in context] if isinstance(context, list) else [],
        "receipt_ids": _list_strings(receipt_ids), "details": details if isinstance(details, list) else [],
    }


def _context_item(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"text": "", "start": None, "end": None}
    return {"text": _string(value.get("text")), "start": _integer_or_none(value.get("start")), "end": _integer_or_none(value.get("end"))}


def _safe_context(value: Any) -> list[dict[str, Any]]:
    return [_context_item(item) for item in value] if isinstance(value, list) else []


def _mode(specs: Any, model: Any) -> str:
    if not isinstance(specs, Mapping) or not isinstance(model, str):
        return ""
    spec = specs.get(model)
    return _string(spec.get("variant")) if isinstance(spec, Mapping) else ""


def _model_identity(value: Any) -> tuple[str, str]:
    if isinstance(value, str):
        return value, ""
    if isinstance(value, Mapping):
        return _string(value.get("model", value.get("model_id", value.get("id")))), _string(value.get("variant", value.get("mode")))
    return "", ""


def _case_lookup(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    selection = manifest.get("selection")
    cases = selection.get("cases") if isinstance(selection, Mapping) else manifest.get("cases")
    return {str(row.get("case_id")): dict(row) for row in cases if isinstance(row, Mapping) and isinstance(row.get("case_id"), str)} if isinstance(cases, list) else {}


def _source_name(manifest: Mapping[str, Any], source_id: Any) -> str:
    source = manifest.get("source")
    sources = source.get("sources") if isinstance(source, Mapping) else manifest.get("sources")
    item = sources.get(source_id) if isinstance(sources, Mapping) else next((row for row in sources if isinstance(row, Mapping) and row.get("source_id") == source_id), None) if isinstance(sources, list) else None
    if isinstance(item, Mapping):
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else item
        for key in ("path", "json_path", "filename", "source_file"):
            value = metadata.get(key) if isinstance(metadata, Mapping) else None
            if isinstance(value, str) and value:
                return Path(value).name
    return _string(source_id)


def _case_status(case: Mapping[str, Any], fallback: str = "pending") -> str:
    if case.get("status") == "invalid":
        return "invalid"
    if case.get("max_chars") == 0:
        return "zero"
    return fallback


def _contract_status(value: Any, fallback: str = "unresolved") -> str:
    status = _string(value)
    if status in {"accepted", "rejected", "unresolved", "pending", "error", "cannot_shorten", "zero", "invalid", "not_reviewed", "evaluated"}:
        return status
    if status in {"transport_error", "schema_error"}:
        return "error"
    if status == "actual_fits":
        return "not_reviewed"
    if status in {"zero_prediction_unresolved", "zero_prediction"}:
        return "zero"
    if status in {"invalid_case", "invalid"}:
        return "invalid"
    if status in {"cannot_shorten", "candidate_absent"}:
        return "cannot_shorten" if status == "cannot_shorten" else fallback
    return fallback


def _own_checksum(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        return
    checksum = value.get("checksum")
    if isinstance(checksum, str):
        body = dict(value)
        body.pop("checksum", None)
        if _sha_json(body) != checksum:
            raise ValueError(f"{label} checksum mismatch")


def _validate_saved_bindings(root: Path, manifest: Mapping[str, Any], raw_results: Any) -> None:
    """Validate optional saved-file bindings, never current source/code hashes."""

    manifest_bytes = (root / "manifest.json").read_bytes() if (root / "manifest.json").is_file() else b""
    ledger_path = root / "ledger.json"
    ledger_bytes = ledger_path.read_bytes() if ledger_path.is_file() else b""
    bindings: Mapping[str, Any] = raw_results if isinstance(raw_results, Mapping) else {}
    if not bindings and isinstance(raw_results, list):
        bindings = next((row for row in raw_results if isinstance(row, Mapping) and any(key in row for key in ("manifest_checksum", "manifest_sha256", "ledger_checksum", "ledger_sha256"))), {})
    has_ledger_binding = any(isinstance(bindings.get(key), str) for key in ("ledger_checksum", "ledger_sha256"))
    if has_ledger_binding and not ledger_bytes:
        raise ValueError("case results references missing ledger")
    expected = bindings.get("manifest_checksum")
    if isinstance(expected, str) and expected != _string(manifest.get("checksum")):
        raise ValueError("case results manifest checksum mismatch")
    expected = bindings.get("manifest_sha256")
    if isinstance(expected, str) and expected != _sha_bytes(manifest_bytes):
        raise ValueError("case results manifest SHA mismatch")
    expected = bindings.get("ledger_checksum")
    if isinstance(expected, str):
        ledger = _read_json(ledger_path)
        _own_checksum(ledger, "ledger")
        body = dict(ledger) if isinstance(ledger, Mapping) else {}
        body.pop("checksum", None)
        if _sha_json(body) != expected:
            raise ValueError("case results ledger checksum mismatch")
    expected = bindings.get("ledger_sha256")
    if isinstance(expected, str) and expected != _sha_bytes(ledger_bytes):
        raise ValueError("case results ledger SHA mismatch")
    expected_ledger = manifest.get("ledger_sha256")
    if isinstance(expected_ledger, str) and expected_ledger != _sha_bytes(ledger_bytes):
        raise ValueError("manifest ledger SHA mismatch")


def _rows_value(value: Any, label: str) -> list[dict[str, Any]]:
    if isinstance(value, list) and all(isinstance(row, Mapping) for row in value):
        return [dict(row) for row in value]
    if isinstance(value, Mapping):
        for key in ("rows", "results", "case_results"):
            rows = value.get(key)
            if isinstance(rows, list) and all(isinstance(row, Mapping) for row in rows):
                _own_checksum(value, label)
                return [dict(row) for row in rows]
    raise ValueError(f"{label} has unknown row shape")


def _branch_candidates(branch: Mapping[str, Any], max_chars: Any, *, phase: str = "initial") -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, item in enumerate(branch.get("candidates", []) if isinstance(branch.get("candidates"), list) else []):
        if isinstance(item, Mapping):
            local = item.get("local_fit") if isinstance(item.get("local_fit"), Mapping) else item.get("local") if isinstance(item.get("local"), Mapping) else None
            result.append(_candidate(item.get("candidate_id", item.get("id", index)), item.get("text"), phase=phase, status=item.get("status", ""), max_chars=max_chars, fits=local.get("fits", local.get("within_length")) if local else None, selected=item.get("selected", False), local=local))
    text = branch.get("candidate", branch.get("text"))
    if isinstance(text, str) and not any(item["text"] == text for item in result):
        local = branch.get("local_fit") if isinstance(branch.get("local_fit"), Mapping) else branch.get("local") if isinstance(branch.get("local"), Mapping) else None
        result.insert(0, _candidate(branch.get("candidate_id", branch.get("id", "initial")), text, phase=phase, status=branch.get("status"), max_chars=max_chars, fits=local.get("fits", local.get("within_length")) if local else None, selected=False, local=local))
    return result


def _branch_row(case: Mapping[str, Any], branch: Mapping[str, Any], manifest: Mapping[str, Any], repetition: Any, arm: str) -> dict[str, Any]:
    specs = manifest.get("full_specs", {})
    generator_key = branch.get("generator")
    arbiter_key = branch.get("arbiter")
    generator, generator_mode = _frozen_model(generator_key, specs)
    arbiter, arbiter_mode = _frozen_model(arbiter_key, specs)
    max_chars = case.get("max_chars")
    initial = _branch_candidates(branch, max_chars)
    initial_verdict = branch.get("verdict")
    repair = branch.get("repair") if isinstance(branch.get("repair"), Mapping) else None
    if repair:
        repair_text = repair.get("text")
        if isinstance(repair_text, str):
            repair_local = repair.get("local_fit") if isinstance(repair.get("local_fit"), Mapping) else repair.get("local") if isinstance(repair.get("local"), Mapping) else None
            initial.append(_candidate(repair.get("candidate_id", "repair"), repair_text, phase="repair", status=repair.get("status", ""), max_chars=max_chars, fits=repair_local.get("fits", repair_local.get("within_length")) if repair_local else None, local=repair_local))
    recheck = repair.get("recheck") if repair else None
    verdict = recheck.get("verdict") if isinstance(recheck, Mapping) else initial_verdict
    accepted = False
    accepted_text: str | None = None
    accepted_id: str | None = None
    initial_text = _string(branch.get("candidate", branch.get("text")))
    repair_text = _string(repair.get("text")) if isinstance(repair, Mapping) else ""
    branch_local = branch.get("local_fit") if isinstance(branch.get("local_fit"), Mapping) else branch.get("local") if isinstance(branch.get("local"), Mapping) else None
    if branch.get("accepted") is True and initial_verdict == "accept" and initial_text.strip() and branch_local and branch_local.get("fits", branch_local.get("within_length")) is True:
        accepted = True
        accepted_text = initial_text
        accepted_id = _string(branch.get("candidate_id")) or None
    elif isinstance(recheck, Mapping) and recheck.get("accepted") is True and recheck.get("verdict") == "accept" and repair_text.strip() and isinstance(repair, Mapping) and isinstance(repair.get("local_fit"), Mapping) and repair["local_fit"].get("fits", repair["local_fit"].get("within_length")) is True:
        accepted = True
        accepted_text = repair_text
        accepted_id = _string(repair.get("candidate_id")) or None
    if accepted:
        phase = "repair" if isinstance(recheck, Mapping) and recheck.get("accepted") is True else "initial"
        marked = False
        for item in initial:
            if item.get("phase") == phase and accepted_id and item.get("id") == accepted_id:
                item["selected"] = True
                marked = True
        if not marked:
            for item in initial:
                if item.get("phase") == phase and item["text"] == accepted_text:
                    item["selected"] = True
                    break
    status = 'accepted' if accepted else _case_status(case, _status_from_branch(branch))
    details: list[dict[str, str]] = []
    if accepted and isinstance(recheck, Mapping):
        details.extend(_detail("Первичные замечания", "; ".join(_list_strings(branch.get("issues")))) for _ in [0] if _list_strings(branch.get("issues")))
        if _string(branch.get("advice")):
            details.append(_detail("Первичный совет", branch.get("advice")))
        if _list_strings(recheck.get("issues")):
            details.append(_detail("Замечания после повтора", "; ".join(_list_strings(recheck.get("issues")))))
        if _string(recheck.get("advice")):
            details.append(_detail("Совет после повтора", recheck.get("advice")))
    elif not accepted and branch.get("repair_status"):
        details.append(_detail("Повтор", branch.get("repair_reason", branch.get("repair_status"))))
    elif accepted and isinstance(repair, Mapping) and not (isinstance(recheck, Mapping) and recheck.get("accepted") is True):
        details.append(_detail("Повтор", repair.get("status", branch.get("repair_status", "не подтверждён"))))
    receipts = [branch.get(key) for key in ("receipt_id", "origin_receipt_id", "review_receipt_id", "candidate_origin_receipt_id") if isinstance(branch.get(key), str)]
    return _row(case.get("case_id"), _source_name(manifest, case.get("source_id", "")), case.get("original", case.get("text")), start=case.get("start"), end=case.get("end"), repetition=repetition, arm=arm, generator=generator, generator_mode=generator_mode, arbiter=arbiter, arbiter_mode=arbiter_mode, candidates=initial, initial_verdict=initial_verdict, verdict=verdict, accepted=accepted, accepted_text=accepted_text, status=status, issues=(recheck.get("issues") if accepted and isinstance(recheck, Mapping) else branch.get("issues")), advice=(recheck.get("advice", "") if accepted and isinstance(recheck, Mapping) else branch.get("advice", branch.get("reason"))), context=case.get("context"), receipt_ids=receipts, details=details)


def _frozen_model(key: Any, specs: Any) -> tuple[str, str]:
    name = _string(key)
    spec = specs.get(name) if isinstance(specs, Mapping) else None
    if isinstance(spec, Mapping):
        model, mode = _model_identity(spec)
        return model or name, mode or _string(spec.get("variant"))
    return name, ""


def _status_from_branch(branch: Mapping[str, Any]) -> str:
    reason = _string(branch.get('reason'))
    if reason in {'invalid_case', 'invalid'}:
        return 'invalid'
    if reason in {'zero_prediction_unresolved', 'zero_prediction'}:
        return 'zero'
    if reason == 'cannot_shorten':
        return 'cannot_shorten'
    if branch.get('parse_error'):
        return 'error'
    status = _string(branch.get('status'))
    if status in {'cannot_shorten', 'zero', 'invalid', 'error', 'unresolved', 'pending', 'rejected', 'not_reviewed'}:
        return status
    if branch.get('verdict') in {'reject', 'revise', 'impossible'}:
        return 'rejected'
    return 'unresolved'


def _adapt_sh08(root: Path, manifest: dict[str, Any], raw_results: Any) -> dict[str, Any]:
    if manifest.get("schema") != "sh08_matrix_manifest.v1":
        raise ShorteningHTMLUnknownFormat("unknown SH08 manifest schema")
    _own_checksum(manifest, "SH08 manifest")
    _validate_saved_bindings(root, manifest, raw_results)
    cases = _case_lookup(manifest)
    rows = _rows_value(raw_results, "SH08 case results")
    by_case: dict[str, dict[str, Any]] = {}
    for item in rows:
        case_id = item.get("case_id", item.get("global_case_id"))
        if isinstance(case_id, str):
            by_case[case_id] = item
    scope = manifest.get("approved_scope") if isinstance(manifest.get("approved_scope"), Mapping) else {}
    repetitions = int(scope.get("repetitions", 1)) if isinstance(scope.get("repetitions", 1), int) else 1
    pairs = scope.get("pairs") if isinstance(scope.get("pairs"), list) else manifest.get("pairs", [])
    output_rows: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            arm = f"{_string(pair[0])}->{_string(pair[1])}"
            for case_id, case in sorted(cases.items(), key=lambda item: (item[1].get("source_id", ""), item[1].get("position", 0), item[0])):
                source_row = by_case.get(case_id, {})
                branches = source_row.get("branches", []) if isinstance(source_row.get("branches"), list) else []
                branch = next((item for item in branches if isinstance(item, Mapping) and item.get("rep", repetition) == repetition and item.get("generator") == pair[0] and item.get("arbiter") == pair[1] and item.get("stage") in {"review", "recheck"}), None)
                generation = next((item for item in branches if isinstance(item, Mapping) and item.get('rep', repetition) == repetition and item.get('generator') == pair[0] and item.get('stage') == 'generation'), None)
                if isinstance(branch, Mapping) and isinstance(generation, Mapping) and generation.get('reason') == 'cannot_shorten':
                    branch = dict(branch)
                    branch['reason'] = 'cannot_shorten'
                if branch is None:
                    status = _contract_status(generation.get("reason"), "") if isinstance(generation, Mapping) else ""
                    if not status:
                        status = _contract_status(source_row.get("status"), "pending")
                    if isinstance(generation, Mapping) and generation.get("text"):
                        branch = dict(generation)
                        branch["arbiter"] = pair[1]
                        branch["verdict"] = None
                    else:
                        branch = None
                if branch is None:
                    generator, generator_mode = _frozen_model(pair[0], manifest.get("full_specs"))
                    arbiter, arbiter_mode = _frozen_model(pair[1], manifest.get("full_specs"))
                    output_rows.append(_row(case_id, _source_name(manifest, case.get("source_id")), case.get("original", case.get("text")), start=case.get("start"), end=case.get("end"), repetition=repetition, arm=arm, generator=generator, generator_mode=generator_mode, arbiter=arbiter, arbiter_mode=arbiter_mode, status=_case_status(case, status), context=case.get("context")))
                else:
                    output_rows.append(_branch_row(case, branch, manifest, repetition, arm))
    if not output_rows:
        raise ValueError("SH08 selection contains no supported cases or pairs")
    run_report = _cheap_report_metadata(root)
    metadata_source = dict(manifest)
    metadata_source.update(run_report)
    document = _document(root, 'Сравнение сокращений — SH08', manifest.get("run_id", root.name), _run_status(manifest, run_report), output_rows, _summary_metadata(metadata_source, len(cases), repetitions, pairs))
    notice = _analysis_notice(root)
    if notice:
        document["notices"].append(notice)
    return document


def _analysis_notice(root: Path) -> str:
    path = root / "analysis" / "report.json"
    if not path.is_file() or path.is_symlink():
        return ""
    try:
        analysis = _read_json(path)
    except ValueError:
        return "Не удалось прочитать анализ; производный статус скрыт"
    if not isinstance(analysis, Mapping):
        return "Неизвестная форма анализа; производный статус скрыт"
    provenance = analysis.get("provenance")
    inputs = provenance.get("inputs") if isinstance(provenance, Mapping) else None
    expected: dict[str, str] = {}
    if isinstance(inputs, Mapping):
        for name, value in inputs.items():
            normalized_name = str(name).lower().replace("_sha256", "").replace("_hash", "")
            if isinstance(value, str):
                expected[normalized_name] = value
            elif isinstance(value, Mapping) and isinstance(value.get("sha256"), str):
                expected[normalized_name] = value["sha256"]
    required = {"manifest": "manifest.json", "ledger": "ledger.json", "case_results": "case_results.json", "report": "report.json"}
    actual = {name: _sha_bytes((root / filename).read_bytes()) for name, filename in required.items() if (root / filename).is_file() and not (root / filename).is_symlink()}
    if set(actual) != set(required) or any(name not in expected or expected[name] != actual[name] for name in required):
        return "Анализ устарел или не имеет всех 4 подтверждённых входов; производный статус скрыт"
    termination = analysis.get("termination") if isinstance(analysis.get("termination"), Mapping) else {}
    raw_status = _string(termination.get('raw_execution_status')) or _string(analysis.get('raw_status')) or _string(_cheap_report_metadata(root).get('status')) or 'неизвестен'
    derived = _string(termination.get('derived_offline_status')) or _string(analysis.get('derived_offline_status')) or _string(analysis.get('derived_termination'))
    if raw_status or derived:
        note = f"Анализ: исходный статус {raw_status}; офлайн-вывод {derived or 'неизвестен'} (проверены 4 входных файла)"
        if derived == "budget_exhausted":
            note += ". Достигнут лимит подписки. В исходном отчёте статус ошибочно обозначен как blocked_unknown; сохранённые результаты не изменялись."
        return note
    return "Анализ: проверены 4 входных файла; производный статус доступен"


def _cheap_report_metadata(root: Path) -> Mapping[str, Any]:
    for name in ("report.json", "run_trialreport.json", "run_trial_report.json"):
        path = root / name
        if path.is_file() and not path.is_symlink():
            try:
                value = _read_json(path)
            except ValueError:
                return {}
            return value if isinstance(value, Mapping) else {}
    return {}


def _run_status(manifest: Mapping[str, Any], report: Mapping[str, Any]) -> str:
    status = _string(report.get("status")) or _string(report.get("run_state"))
    if status:
        return status
    status = _string(manifest.get("status"))
    return "partial" if status in {"prepare", "prepared"} else status or "partial"


def _original_timing(record: Mapping[str, Any]) -> tuple[Any, Any]:
    for key in ("source_analysis", "budget", "source"):
        value = record.get(key)
        if isinstance(value, Mapping):
            start, end = value.get("start"), value.get("end")
            if isinstance(start, int) or isinstance(end, int):
                return start, end
    return record.get("start"), record.get("end")


def _history_candidates(record: Mapping[str, Any], max_chars: Any) -> tuple[list[dict[str, Any]], str, str]:
    candidates: list[dict[str, Any]] = []
    initial_verdict = ""
    final_verdict = ""
    history = record.get("history") if isinstance(record.get("history"), list) else []
    for index, event in enumerate(history):
        if not isinstance(event, Mapping):
            continue
        event_stage = event.get("stage")
        if event_stage not in {"generation", "repair"}:
            event_stage = None
        stage = "repair" if event_stage == "repair" else "initial"
        text = event.get("text", event.get("candidate"))
        generator_ok = event.get("generator_status") in {None, "ok"}
        if event_stage and isinstance(text, str) and generator_ok and event.get("status") not in {"error", "cannot_shorten"}:
            local = event.get("local") if isinstance(event.get("local"), Mapping) else None
            candidates.append(_candidate(event.get("candidate_id", index), text, phase=stage, status=event.get("status", ""), max_chars=max_chars, fits=local.get("within_length", local.get("fits")) if local else None, local=local))
        review = event.get("review") if isinstance(event.get("review"), Mapping) else event.get("judgment") if isinstance(event.get("judgment"), Mapping) else event if event.get("verdict") else None
        if isinstance(review, Mapping) and isinstance(review.get("verdict"), str):
            if not initial_verdict:
                initial_verdict = review["verdict"]
            final_verdict = review["verdict"]
    final = record.get("final_review")
    if isinstance(final, Mapping) and isinstance(final.get("verdict"), str):
        final_verdict = final["verdict"]
    candidate = record.get("candidate")
    record_local = record.get("local") if isinstance(record.get("local"), Mapping) else record.get("local_fit") if isinstance(record.get("local_fit"), Mapping) else None
    if isinstance(candidate, str) and record_local is not None and not any(item["text"] == candidate for item in candidates):
        candidates.append(_candidate("candidate", candidate, status=record.get("status"), max_chars=max_chars, fits=record_local.get("within_length", record_local.get("fits")), local=record_local))
    return candidates, initial_verdict, final_verdict


def _primary_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() and parsed >= 0 else None


def _primary_money(value: Any) -> str | None:
    parsed = _primary_decimal(value)
    if parsed is None:
        return None
    text = format(parsed, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _primary_count(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return str(value)


def _primary_model(value: Any) -> tuple[str, str]:
    model, mode = _model_identity(value)
    return model, mode.upper() if mode else ""


def _is_primary_trial(report: Mapping[str, Any]) -> bool:
    metadata = report.get("metadata")
    openrouter = metadata.get("openrouter") if isinstance(metadata, Mapping) else None
    budget = report.get("budget")
    return isinstance(openrouter, Mapping) and isinstance(openrouter.get("pricing_snapshot"), Mapping) and isinstance(budget, Mapping)


def _primary_reported_cost(stages: Mapping[str, Any], names: tuple[str, ...]) -> str | None:
    values: list[Decimal] = []
    for name in names:
        stage = stages.get(name)
        if not isinstance(stage, Mapping):
            return None
        value = _primary_decimal(stage.get("openrouter_reported_cost_usd"))
        if value is None:
            return None
        values.append(value)
    return _primary_money(sum(values, Decimal(0)))


def _primary_cost_value(cost: Mapping[str, Any], stages: Mapping[str, Any]) -> str | None:
    direct = cost.get("direct")
    if isinstance(direct, Mapping):
        value = _primary_money(direct.get("openrouter_reported_cost_usd"))
        if value is not None:
            return value
    stage_values = []
    for name in ("generation", "review", "repair", "recheck"):
        stage = stages.get(name)
        if not isinstance(stage, Mapping):
            return None
        value = _primary_decimal(stage.get("openrouter_reported_cost_usd"))
        if value is None:
            return None
        stage_values.append(value)
    return _primary_money(sum(stage_values, Decimal(0)))


def _primary_token_value(cost: Mapping[str, Any], stages: Mapping[str, Any], key: str) -> str | None:
    direct = cost.get("direct")
    if isinstance(direct, Mapping):
        value = _primary_count(direct.get(key))
        if value is not None:
            return value
    values: list[int] = []
    for name in ("generation", "review", "repair", "recheck"):
        stage = stages.get(name)
        value = stage.get(key) if isinstance(stage, Mapping) else None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        values.append(value)
    return str(sum(values))


def _primary_receipt_ids(report: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    cost = report.get("cost") if isinstance(report.get("cost"), Mapping) else {}
    containers = [report, cost, report.get("budget"), report.get("metadata")]
    if isinstance(cost, Mapping) and isinstance(cost.get("stages"), Mapping):
        containers.extend(value for value in cost["stages"].values() if isinstance(value, Mapping))
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        for key in ("provider_reported_receipt_ids", "provider_reported_receipts", "provider_receipt_ids", "receipt_ids", "generation_ids"):
            raw = container.get(key)
            if isinstance(raw, list):
                values.extend(item for item in raw if isinstance(item, str) and item)
            elif isinstance(raw, str) and raw:
                values.append(raw)
    records = report.get("records")
    if isinstance(records, list):
        for record in records:
            history = record.get("history") if isinstance(record, Mapping) else None
            if not isinstance(history, list):
                continue
            for event in history:
                if not isinstance(event, Mapping):
                    continue
                for key in ("provider_receipt_id", "receipt_id", "generation_id"):
                    value = event.get(key)
                    if isinstance(value, str) and value:
                        values.append(value)
    return list(dict.fromkeys(values))


def _primary_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    metadata = report.get("metadata") if isinstance(report.get("metadata"), Mapping) else {}
    openrouter = metadata.get("openrouter") if isinstance(metadata.get("openrouter"), Mapping) else {}
    roles = openrouter.get("roles") if isinstance(openrouter.get("roles"), Mapping) else report.get("model", {})
    model = report.get("model") if isinstance(report.get("model"), Mapping) else {}
    generator, generator_mode = _primary_model(model.get("generator", roles.get("generator")))
    arbiter, arbiter_mode = _primary_model(model.get("arbiter", roles.get("arbiter")))

    budget = report.get("budget") if isinstance(report.get("budget"), Mapping) else {}
    cost = report.get("cost") if isinstance(report.get("cost"), Mapping) else {}
    stages = cost.get("stages") if isinstance(cost.get("stages"), Mapping) else {}
    total = _primary_cost_value(cost, stages)
    generator_cost = _primary_reported_cost(stages, ("generation", "repair"))
    arbiter_cost = _primary_reported_cost(stages, ("review", "recheck"))
    cost_text = (
        f"API reported USD: total ${total or 'unknown'}; "
        f"generation+repair ${generator_cost or 'unknown'}; "
        f"review+recheck ${arbiter_cost or 'unknown'}; reservation is not spend"
    )

    settings = [
        f"generator: {generator or 'unknown'} {generator_mode}".rstrip(),
        f"arbiter: {arbiter or 'unknown'} {arbiter_mode}".rstrip(),
    ]
    token_labels = (("input_tokens", "input"), ("output_tokens", "output"),
                    ("reasoning_tokens", "reasoning"), ("cache_read_tokens", "cache"))
    token_values = []
    for key, label in token_labels:
        value = _primary_token_value(cost, stages, key)
        if value is not None:
            token_values.append(f"{label} tokens: {value}")
    settings.extend(token_values)

    snapshot = openrouter.get("pricing_snapshot") if isinstance(openrouter.get("pricing_snapshot"), Mapping) else {}
    snapshot_models = snapshot.get("models") if isinstance(snapshot.get("models"), Mapping) else {}
    route_names: list[str] = []
    fee_sources: list[str] = []
    for role in (roles.get("generator"), roles.get("arbiter")) if isinstance(roles, Mapping) else ():
        role_model, _ = _model_identity(role)
        entry = snapshot_models.get(role_model) if isinstance(snapshot_models, Mapping) else None
        if not isinstance(entry, Mapping) and isinstance(role, Mapping):
            entry = snapshot_models.get(role.get("model")) if isinstance(snapshot_models, Mapping) else None
        if isinstance(entry, Mapping):
            value = entry.get("provider_name") or entry.get("provider_slug")
            if isinstance(value, str) and value and value not in route_names:
                route_names.append(value)
            fee = entry.get("request_fee_source")
            if isinstance(fee, str) and fee and fee not in fee_sources:
                fee_sources.append(fee)
    request_fee = _string(budget.get("request_fee_assumption")) or ", ".join(fee_sources)
    if request_fee:
        settings.append(f"pinned route: {'/'.join(route_names) or 'unknown'}; request fee: {request_fee}")
    run_config = metadata.get("config", {}).get("run") if isinstance(metadata.get("config"), Mapping) else {}
    if isinstance(run_config, Mapping) and isinstance(run_config.get("length_tools"), bool):
        settings.append(f"extra tools: {'yes' if run_config['length_tools'] else 'no'}")

    result: dict[str, Any] = {
        "status": report.get("status", report.get("run_state", "partial")),
        "settings": "; ".join(settings),
        "cost": cost_text,
        "cost_status": f"API reported usage; accounting_complete={cost.get('direct', {}).get('accounting_complete') if isinstance(cost.get('direct'), Mapping) else 'unknown'}; budget={budget.get('status', 'unknown')}",
    }
    for key, metadata_key in (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens"),
                              ("reasoning_tokens", "reasoning_tokens"), ("cache_read_tokens", "cache_tokens")):
        value = _primary_token_value(cost, stages, key)
        if value is not None:
            result[metadata_key] = value
    for key, value in (("limit", budget.get("limit_usd")), ("unknown_reserved", budget.get("unknown_reserved_usd")),
                       ("remaining", budget.get("remaining_usd")), ("next_reservation", budget.get("next_reservation_usd"))):
        if value is not None:
            rendered = _primary_money(value)
            result[key] = f"${rendered}" if rendered is not None else "unknown"
    physical = _primary_count(budget.get("physical_calls"))
    maximum = _primary_count(budget.get("max_requests"))
    if physical is not None or maximum is not None:
        result["physical"] = f"{physical or 'unknown'} / {maximum or 'unknown'}"
    if budget.get("unknown_reserved_usd") is not None:
        result["unknown_reserved"] += " (reservation, not spend)"
    reason = _string(budget.get("blocked_reason"))
    if not reason and isinstance(report.get("budget_stop"), Mapping):
        reason = _string(report["budget_stop"].get("reason"))
    if reason:
        result["stop_reason"] = reason
    snapshot_id = _string(snapshot.get("id")) or _string(budget.get("pricing_snapshot_id"))
    if snapshot_id:
        result["pricing_snapshot_id"] = snapshot_id
    receipts = _primary_receipt_ids(report)
    if receipts:
        result["receipts"] = ", ".join(receipts)
    blocked = reason or _string(budget.get("status"))
    if blocked.startswith("blocked") or _run_status({}, report) != "completed":
        result["partial"] = (
            f"Partial run: budget/accounting state {blocked or 'incomplete'}; "
            "unresolved lines keep original text; no public quality claim"
        )
    return result


def _has_final_audio_evidence(report: Mapping[str, Any]) -> bool:
    for record in report.get("records", []) if isinstance(report.get("records"), list) else []:
        if not isinstance(record, Mapping):
            continue
        final = record.get("final_audio")
        if isinstance(final, Mapping) and any(key in final for key in ("duration_sec", "final_duration_sec")):
            return True
        evidence = record.get("timing_evidence")
        audio = evidence.get("audio") if isinstance(evidence, Mapping) else None
        if isinstance(audio, Mapping) and isinstance(audio.get("final"), Mapping):
            return True
    return False


def _primary_notices(report: Mapping[str, Any]) -> list[str]:
    notices = []
    if not _has_final_audio_evidence(report):
        notices.append("Длительность финальной озвучки ещё не проверена; нерешённые реплики сохранены оригинальными.")
    budget = report.get("budget") if isinstance(report.get("budget"), Mapping) else {}
    if _string(budget.get("status")).startswith("blocked") or _run_status({}, report) != "completed":
        notices.append("Частичный запуск: бюджет или учёт остановил дальнейшие запросы; публичная оценка качества не заявляется.")
    return notices


def _adapt_trial(root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError("trial report records are missing")
    rows: list[dict[str, Any]] = []
    model = report.get("model") if isinstance(report.get("model"), Mapping) else {}
    generator, generator_mode = _model_identity(model.get("generator", model.get("model")))
    arbiter, arbiter_mode = _model_identity(model.get("arbiter"))
    generator_mode = generator_mode or _string(model.get("variant"))
    arbiter_mode = arbiter_mode or _string(model.get("arbiter_variant"))
    primary_report = _is_primary_trial(report)
    for record in records:
        if not isinstance(record, Mapping):
            continue
        candidates, initial_verdict, verdict = _history_candidates(record, record.get("budget", {}).get("max_chars") if isinstance(record.get("budget"), Mapping) else None)
        final = record.get("final_review") if isinstance(record.get("final_review"), Mapping) else {}
        final_text = record.get("final_text")
        matching = next((item for item in candidates if item["text"] == final_text), None)
        final_local = final.get("local") if isinstance(final.get("local"), Mapping) else final.get("local_fit") if isinstance(final.get("local_fit"), Mapping) else None
        valid = bool(record.get("status") == "accepted" and final.get("verdict") == "accept" and isinstance(final_text, str) and final_text.strip() and matching is not None)
        saved_fit = final_local.get("within_length", final_local.get("fits")) if final_local is not None else matching.get("fits") if primary_report and matching is not None else None
        valid = valid and saved_fit is True
        if matching is not None and matching.get("fits") is False:
            valid = False
        raw_status = _string(record.get("status"))
        status = "accepted" if valid else ("unresolved" if raw_status in {"accepted", "manual"} else "not_reviewed" if raw_status == "actual_fits" else _contract_status(raw_status, "pending"))
        start, end = _original_timing(record)
        rows.append(_row(record.get("cue_id", record.get("case_id")), record.get("source_id", ""), record.get("original"), start=start, end=end, repetition=record.get("repetition"), arm="standard", generator=generator, generator_mode=generator_mode, arbiter=arbiter, arbiter_mode=arbiter_mode, candidates=candidates, initial_verdict=initial_verdict, verdict=verdict, accepted=valid, accepted_text=final_text if valid else None, status=status, issues=final.get("issues"), advice=final.get("advice", record.get("failure", "")), context=record.get("context"), receipt_ids=[event.get("receipt_id") for event in record.get("history", []) if isinstance(event, Mapping) and isinstance(event.get("receipt_id"), str)], details=[]))
    if not rows:
        raise ValueError("trial report contains no records")
    if _is_primary_trial(report):
        document = _document(
            root, "Shortening trial", report.get("run_id", root.name),
            _run_status({}, report), rows, _primary_metadata(report),
        )
        document["notices"].extend(_primary_notices(report))
        return document
    return _document(root, "Shortening trial", report.get("run_id", root.name), _run_status({}, report), rows, _summary_metadata(report, len(rows), 1, []))


def _jev_scalar(value: Any) -> str:
    """Serialize one saved Jev scalar without exporting private response data."""

    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
        return ", ".join(str(item) for item in value)
    return ""


def _jev_model_identity(provenance: Any) -> tuple[str, str]:
    if not isinstance(provenance, Mapping):
        return "", ""
    return _string(provenance.get("generator_model")), _string(provenance.get("generator_mode"))


def _jev_provenances(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values: list[Mapping[str, Any]] = []
    candidates = record.get("candidates") if isinstance(record.get("candidates"), list) else []
    for candidate in candidates:
        provenance = candidate.get("provenance") if isinstance(candidate, Mapping) else None
        if isinstance(provenance, list):
            values.extend(item for item in provenance if isinstance(item, Mapping))
    return values


def _jev_probability(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    probabilities = value.get("probabilities")
    if not isinstance(probabilities, Mapping):
        return ""
    return _canonical({str(key): item for key, item in probabilities.items() if isinstance(item, (int, float)) and not isinstance(item, bool)})


def _jev_answer(answers: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    answer = answers.get(key)
    return answer if isinstance(answer, Mapping) else {}


def _jev_details(record: Mapping[str, Any], answers: Mapping[str, Any]) -> list[dict[str, str]]:
    details: list[dict[str, str]] = []
    details.append(_detail("Jev mode", "Choice"))
    details.append(_detail("Порядок", "presentation order (not an independent experiment)"))
    best = _jev_answer(answers, "best")
    for label, value in (("Jev best choice", best.get("choice")), ("Jev best probabilities", _jev_probability(best)),
                         ("Jev best confidence", best.get("confidence"))):
        display = _jev_scalar(value)
        if display:
            details.append(_detail(label, display))
    candidates = record.get("candidates") if isinstance(record.get("candidates"), list) else []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = _string(candidate.get("id"))
        answer = _jev_answer(answers, f"safety_{candidate_id}")
        if not answer:
            continue
        prefix = f"Jev safety {candidate_id}"
        for label, value in ((f"{prefix} choice", answer.get("choice")), (f"{prefix} probabilities", _jev_probability(answer)),
                             (f"{prefix} confidence", answer.get("confidence"))):
            display = _jev_scalar(value)
            if display:
                details.append(_detail(label, display))

    candidates = record.get("candidates") if isinstance(record.get("candidates"), list) else []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("provenance"), list):
            continue
        candidate_id = _string(candidate.get("id"))
        for provenance in candidate["provenance"]:
            if not isinstance(provenance, Mapping):
                continue
            historic = provenance.get("historic_sol")
            if not isinstance(historic, list):
                continue
            for historic_item in historic:
                if not isinstance(historic_item, Mapping):
                    continue
                payload = {
                    "candidate_id": candidate_id,
                    "origin_receipt_id": provenance.get("origin_receipt_id"),
                    "source_run": provenance.get("source_run"),
                    "rep": provenance.get("rep"),
                    "stage": provenance.get("stage"),
                    "historic_sol": historic_item,
                }
                details.append(_detail("Исторический Sol", _canonical(payload)))
    source = _jev_scalar(record.get("source"))
    if source:
        details.append(_detail("source", source))
    request_id = _jev_scalar(record.get("request_id"))
    if request_id:
        details.append(_detail("request_id", request_id))
    receipt_id = _jev_scalar(record.get("receipt_id", record.get("receiptid")))
    if receipt_id and receipt_id != request_id:
        details.append(_detail("Jev receipt id", receipt_id))
    errors = _jev_scalar(record.get("error"))
    if errors:
        details.append(_detail("Jev error", _safe_error(errors)))
    return details


def _jev_candidate(candidate: Mapping[str, Any], max_chars: Any) -> dict[str, Any]:
    candidate_id = candidate.get("id")
    result = _candidate(candidate_id, candidate.get("text"), phase="alternative", max_chars=max_chars,
                        fits=candidate.get("fits"), local={"actual_chars": candidate.get("actual_chars"), "fits": candidate.get("fits")})
    if isinstance(candidate.get("actual_chars"), int) and not isinstance(candidate.get("actual_chars"), bool):
        result["actual_chars"] = candidate["actual_chars"]
    return result


def _jev_summary_metadata(report: Mapping[str, Any], case_count: int) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    planned = summary.get("records")
    attempted = summary.get("attempted_requests")
    completed = summary.get("completed_requests")
    unknown_reserved = summary.get("unknown_reserved_usd")
    input_tokens = summary.get("usage", {}).get("input_tokens") if isinstance(summary.get("usage"), Mapping) else None
    metadata: dict[str, Any] = {
        "status": report.get("status", "partial"),
        "cases": summary.get("cases", case_count),
        "logical": f"{attempted} / {planned}" if isinstance(attempted, int) and isinstance(planned, int) else "",
        "cost": f"estimated ${summary.get('cost_usd_estimated')}" if summary.get("cost_usd_estimated") is not None else "",
        "settings": f"input tokens: {input_tokens}" if isinstance(input_tokens, int) else "",
    }
    if report.get("status") == "partial":
        metadata["partial"] = f"attempted {attempted}; completed {completed}; unknown reserved ${unknown_reserved}"
    return metadata


def small_adapt_jev(root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the saved Jev replay contract without exposing transport artifacts."""

    if report.get("schema") != "jev_replay.v1":
        raise ShorteningHTMLUnknownFormat("unknown Jev report schema")
    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError("Jev report records are missing")
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        candidates_raw = record.get("candidates") if isinstance(record.get("candidates"), list) else []
        max_chars = record.get("max_chars")
        candidates = [_jev_candidate(candidate, max_chars) for candidate in candidates_raw if isinstance(candidate, Mapping)]
        selected_id = record.get("selected_candidate_id")
        matching_candidates = [candidate for candidate in candidates if candidate.get("id") == selected_id]
        selected = matching_candidates[0] if len(matching_candidates) == 1 else None
        accepted_text = record.get("accepted_text")
        answers = record.get("answers") if isinstance(record.get("answers"), Mapping) else {}
        best = _jev_answer(answers, "best")
        safety = _jev_answer(answers, f"safety_{_string(selected_id)}") if selected is not None else {}
        accepted = bool(record.get("status") == "accepted" and selected is not None
                        and best.get("choice") == selected_id
                        and safety.get("choice") == "accept"
                        and isinstance(accepted_text, str) and bool(accepted_text.strip())
                        and accepted_text == selected.get("text") and selected.get("fits") is True)
        for candidate in candidates:
            candidate["selected"] = candidate is selected and accepted
        provenance_values = _jev_provenances(record)
        models = []
        modes = []
        for provenance in provenance_values:
            model, mode = _jev_model_identity(provenance)
            if model and model not in models:
                models.append(model)
            if mode and mode not in modes:
                modes.append(mode)
        generator = "/".join(models)
        generator_mode = "/".join(modes)
        receipt_id = record.get("receipt_id")
        row_status = "accepted" if accepted else _contract_status(record.get("status"), "pending")
        if not accepted and row_status == "accepted":
            row_status = "unresolved"
        context_value = record.get("context")
        if isinstance(context_value, list):
            context_value = [item if isinstance(item, Mapping) else {"text": item} for item in context_value]
        row = _row(record.get("case_id"), record.get("source", ""),
                   record.get("original"), start=record.get("start"), end=record.get("end"),
                   repetition=record.get("order"), generator=generator, generator_mode=generator_mode,
                   arbiter=_string(report.get("model")), arbiter_mode="Choice",
                   candidates=candidates, accepted=accepted, accepted_text=accepted_text if accepted else None,
                   status=row_status,
                   context=context_value, receipt_ids=[receipt_id] if isinstance(receipt_id, str) else [],
                   details=_jev_details(record, answers), decision_kind="selector")
        rows.append(row)
    if not rows:
        raise ValueError("Jev report contains no records")
    document = _document(root, "Эксперимент Jev replay", report.get("run_id", root.name), _run_status({}, report), rows,
                         _jev_summary_metadata(report, len(rows)))
    document["rows"] = rows
    document["notices"].extend([
        "Экспериментальная рекомендация Jev; итоговые субтитры не изменены",
        "Варианты собраны из прошлых реальных запусков, не из одного best-of-5",
        "Совпадение с историческим Sol не является проверкой истинности; независимая смысловая проверка не выполнена",
    ])
    return document


def _followup_baseline_answers(root: Path, record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read only the frozen, structured Jev answers from the producer manifest."""

    path = root / "manifest.json"
    try:
        manifest = _read_json(path)
    except (OSError, ValueError):
        return {}
    if not isinstance(manifest, Mapping) or manifest.get("schema") != "jev_followup.v1":
        return {}
    selection = manifest.get("selection")
    historical = selection.get("historical") if isinstance(selection, Mapping) else None
    case_history = historical.get(record.get("case_id")) if isinstance(historical, Mapping) else None
    if not isinstance(case_history, Mapping):
        return {}
    order = record.get("order") if record.get("request_kind") == "repeat" else 1
    entry = case_history.get(str(order), case_history.get(order))
    answers = entry.get("answers") if isinstance(entry, Mapping) else None
    return answers if isinstance(answers, Mapping) else {}


def _followup_baseline_context(root: Path, record: Mapping[str, Any]) -> Any:
    path = root / "manifest.json"
    try:
        manifest = _read_json(path)
    except (OSError, ValueError):
        return record.get("context_original_window")
    cases = manifest.get("cases") if isinstance(manifest, Mapping) else None
    if isinstance(cases, list):
        case = next((item for item in cases if isinstance(item, Mapping) and item.get("case_id") == record.get("case_id")), None)
        if isinstance(case, Mapping) and isinstance(case.get("context"), list):
            return case["context"]
    return record.get("context_original_window")


def _followup_context_json(value: Any) -> str:
    """Keep source context markers and mapped target text, not provider payloads."""

    if not isinstance(value, list):
        return ""
    safe: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        row: dict[str, Any] = {}
        for key in ("cue_id", "text", "start", "end"):
            raw = item.get(key)
            if isinstance(raw, (str, int, float)) and not isinstance(raw, bool):
                row[key] = raw
        for key in ("marked_target", "is_target"):
            if item.get(key) is True:
                row[key] = True
        safe.append(row)
    return _canonical(safe) if safe else ""


def _followup_current_context(record: Mapping[str, Any], candidate: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    value = record.get("context_original_window")
    context = deepcopy(value) if isinstance(value, list) else []
    if record.get("request_kind") != "isolated_safety" or not isinstance(candidate, Mapping):
        return context
    text = candidate.get("text")
    if not isinstance(text, str):
        return context
    for item in context:
        if isinstance(item, Mapping) and item.get("marked_target") is True:
            item["text"] = text
            item["is_target"] = True
    return context


def _followup_structured_details(label: str, answers: Mapping[str, Any]) -> list[dict[str, str]]:
    details: list[dict[str, str]] = []
    for question_id, value in answers.items():
        if not isinstance(question_id, str) or not isinstance(value, Mapping):
            continue
        choice = _jev_scalar(value.get("choice"))
        probabilities = _jev_probability(value)
        confidence = _jev_scalar(value.get("confidence"))
        for suffix, display in (("choice", choice), ("probabilities", probabilities), ("confidence", confidence)):
            if display:
                details.append(_detail(f"{label} {question_id} {suffix}", display))
    return details


def _followup_request_label(request_kind: Any) -> str:
    return {
        "repeat": "повтор без изменения входных данных",
        "isolated_safety": "изолированная оценка",
        "independent_review": "независимая проверка",
    }.get(_string(request_kind), "неизвестный запрос")


def _followup_details(
    record: Mapping[str, Any],
    baseline_answers: Mapping[str, Any],
    current_answers: Mapping[str, Any],
    base: list[dict[str, str]],
    baseline_context: Any,
    current_context: Any,
) -> list[dict[str, str]]:
    details = list(base)
    details.append(_detail("Тип запроса", _followup_request_label(record.get("request_kind"))))
    candidate_id = record.get("candidate_id")
    if isinstance(candidate_id, str):
        details.append(_detail("candidate_id", candidate_id))
    old_hash = record.get("old_payload_hash")
    if isinstance(old_hash, str):
        details.append(_detail("source baseline payload hash", old_hash))
    for label, value in (("Контекст исходного запроса", baseline_context), ("Контекст текущего запроса", current_context)):
        serialized = _followup_context_json(value)
        if serialized:
            details.append(_detail(label, serialized))
    details.extend(_followup_structured_details("Текущий Jev", current_answers))
    details.extend(_followup_structured_details("Базовый Jev", baseline_answers))
    return details


def _jev_provenance_models(record: Mapping[str, Any]) -> tuple[str, str]:
    models: list[str] = []
    modes: list[str] = []
    for provenance in _jev_provenances(record):
        model, mode = _jev_model_identity(provenance)
        if model and model not in models:
            models.append(model)
        if mode and mode not in modes:
            modes.append(mode)
    return "/".join(models), "/".join(modes)


def _followup_candidate(record: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    values = record.get("candidates")
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], Mapping):
        return None, None
    candidate = dict(values[0])
    candidate_id = candidate.get("id")
    if not isinstance(candidate_id, str):
        return None, None
    return candidate, candidate_id


def _adapt_jev_followup(root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the exact ``jev_followup.v1`` producer report without selecting evaluations."""

    if report.get("schema") != "jev_followup.v1":
        raise ShorteningHTMLUnknownFormat("unknown Jev follow-up report schema")
    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError("Jev follow-up report records are missing")
    rows: list[dict[str, Any]] = []
    for record_value in records:
        if not isinstance(record_value, Mapping):
            continue
        request_kind = record_value.get("request_kind")
        if request_kind not in {"repeat", "isolated_safety", "independent_review"}:
            raise ValueError("Jev follow-up request kind is invalid")
        record = dict(record_value)
        baseline_answers = _followup_baseline_answers(root, record)
        candidate, candidate_id = _followup_candidate(record)
        current_context = _followup_current_context(record, candidate)
        baseline_context = _followup_baseline_context(root, record)
        current_answers = record.get("answers") if isinstance(record.get("answers"), Mapping) else {}
        receipt_ids = [record.get("receipt_id")] if isinstance(record.get("receipt_id"), str) else []

        if request_kind == "repeat":
            # The existing replay adapter contains the strict best/safety/fit/text
            # checks.  Keep it as the single acceptance implementation.
            one_report = {
                "schema": "jev_replay.v1", "run_id": report.get("run_id", root.name),
                "status": report.get("status", "partial"), "model": record.get("model", ""),
                "records": [record],
            }
            row = small_adapt_jev(root, one_report)["rows"][0]
            row["repetition"] = record.get("order")
            row["request_kind"] = request_kind
            row["candidate_id"] = None
            row["context"] = current_context
            row["arbiter"] = _string(record.get("model"))
            row["arbiter_mode"] = _string(record.get("mode"))
            row["details"] = _followup_details(record, baseline_answers, current_answers, row.get("details", []), baseline_context, current_context)
            rows.append(row)
            continue

        generator, generator_mode = _jev_provenance_models(record)
        model = _string(record.get("model"))
        mode = _string(record.get("mode"))
        if request_kind == "independent_review":
            base_details = [item for item in _jev_details(record, {}) if item.get("label") not in {"Jev mode", "Порядок"}]
            issues = record.get("issues") if isinstance(record.get("issues"), list) else []
            advice = record.get("advice") if isinstance(record.get("advice"), str) else ""
        else:
            base_details = [item for item in _jev_details(record, current_answers) if item.get("label") != "Порядок"]
            issues = []
            advice = ""
        status = _contract_status(record.get("status"), "pending")
        rows.append(_row(
            record.get("case_id"), record.get("source", ""), record.get("original"),
            start=record.get("start"), end=record.get("end"), repetition=None,
            generator=generator, generator_mode=generator_mode, arbiter=model,
            arbiter_mode=mode, candidates=[_jev_candidate(candidate, record.get("max_chars"))] if candidate is not None else [],
            verdict=record.get("verdict"), accepted=False, accepted_text=None, status=status,
            issues=issues, advice=advice, context=current_context, receipt_ids=receipt_ids,
            details=_followup_details(record, baseline_answers, current_answers, base_details, baseline_context, current_context),
            decision_kind="evaluation", request_kind=request_kind, candidate_id=candidate_id,
        ))
    if not rows:
        raise ValueError("Jev follow-up report contains no records")
    metadata = _jev_followup_summary_metadata(report)
    document = _document(root, "Эксперимент Jev follow-up", report.get("run_id", root.name), _run_status({}, report), rows, metadata)
    document["notices"].extend([
        "Приватный экспериментальный диагностический отчёт по 6 кейсам; выборка смещена правилами отбора.",
        "Итоговые субтитры не изменялись; оценка варианта не означает выбора финального текста.",
        "Независимая проверка DeepSeek использует другое семейство моделей и не является эталоном истины.",
        "Чистый причинный эффект порядка не доказан: изоляция одновременно меняет упаковку вопросов.",
    ])
    return document


def _jev_followup_summary_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    metadata: dict[str, Any] = {
        "status": report.get("status", "partial"), "cases": summary.get("cases"),
        "planned": summary.get("records"), "attempted": summary.get("attempted_requests"),
        "completed": summary.get("completed_requests"), "errors": summary.get("errors"),
        "pending": summary.get("pending"),
    }
    backends = summary.get("by_backend") if isinstance(summary.get("by_backend"), Mapping) else {}
    for name in ("jev", "deepseek"):
        backend = backends.get(name) if isinstance(backends.get(name), Mapping) else {}
        metadata.update({
            f"{name}_attempted": backend.get("attempted"), f"{name}_completed": backend.get("completed"),
            f"{name}_input_tokens": backend.get("input_tokens"), f"{name}_output_tokens": backend.get("output_tokens"),
            f"{name}_reserved_usd": backend.get("unknown_reserved_usd"),
        })
        if name == "jev":
            metadata["jev_cost_usd_estimated"] = backend.get("cost_usd_estimated")
            metadata["jev_cost_kind"] = "estimated"
        else:
            metadata["deepseek_cost_usd"] = backend.get("cost_usd")
            metadata["deepseek_cost_kind"] = backend.get("cost_kind")
    if report.get("status") in {"partial", "blocked"}:
        metadata["partial"] = f"attempted {metadata.get('attempted')}; completed {metadata.get('completed')}"
        campaign_error = _safe_error(report.get("campaign_error"))
        if campaign_error:
            metadata["partial"] += f"; {campaign_error}"
    return metadata


def _adapt_abc(root: Path, manifest: Mapping[str, Any], raw_results: Any) -> dict[str, Any]:
    _own_checksum(manifest, "experiment manifest")
    _validate_saved_bindings(root, manifest, raw_results)
    rows = _rows_value(raw_results, "experiment case results")
    cases = _case_lookup(manifest)
    specs = manifest.get("full_specs", {}) if isinstance(manifest, Mapping) else {}
    output: list[dict[str, Any]] = []
    model_dict = manifest.get("models") if isinstance(manifest.get("models"), Mapping) else {}
    root_generator, root_generator_mode = _model_identity(model_dict.get("generator")) if isinstance(model_dict, Mapping) else ("", "")
    selector, selector_mode = _model_identity(model_dict.get("selector")) if isinstance(model_dict, Mapping) else ("", "")
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        case_id = item.get("global_case_id", item.get("case_id"))
        case = dict(cases.get(case_id, {})) if isinstance(case_id, str) else {}
        budget = item.get("budget") if isinstance(item.get("budget"), Mapping) else {}
        if "max_chars" not in case and isinstance(budget.get("max_chars"), int):
            case["max_chars"] = budget["max_chars"]
        if "available_ms" not in case and isinstance(budget.get("available_ms"), int):
            case["available_ms"] = budget["available_ms"]
        for timing_key in ("start", "end"):
            if timing_key not in case and isinstance(item.get(timing_key), int):
                case[timing_key] = item[timing_key]
        original = item.get("original", case.get("original", case.get("text")))
        standard = item.get("B") if isinstance(item.get("B"), Mapping) else {}
        c_arm = item.get("C") if isinstance(item.get("C"), Mapping) else {}
        repetition = item.get("repetition")
        source_id = item.get("source_id", case.get("source_id"))
        context = item.get("context", case.get("context"))
        child = _standard_child(root, repetition, source_id, case_id)
        generator = root_generator
        generator_mode = root_generator_mode
        provenance = c_arm.get("provenance") if isinstance(c_arm.get("provenance"), Mapping) else item.get("provenance")
        if not generator and isinstance(provenance, Mapping):
            saved = provenance.get("generator")
            generator, generator_mode = _model_identity(saved)
        if not generator:
            provenance = standard.get("provenance") if isinstance(standard.get("provenance"), Mapping) else item.get("A", {}).get("provenance") if isinstance(item.get("A"), Mapping) else None
            generator, generator_mode = _model_identity(provenance.get("generator", provenance) if isinstance(provenance, Mapping) else None)

        # A is deliberately an unreviewed proposal, never an arbiter decision.
        a_arm = item.get("A") if isinstance(item.get("A"), Mapping) else {}
        a_text = a_arm.get("text")
        a_local = a_arm.get("local") if isinstance(a_arm.get("local"), Mapping) else a_arm.get("local_fit") if isinstance(a_arm.get("local_fit"), Mapping) else None
        output.append(_row(case_id, _source_name(manifest, source_id), original, start=case.get("start"), end=case.get("end"), repetition=repetition, arm="A", generator=generator, generator_mode=generator_mode, candidates=[_candidate("A", a_text, status=a_arm.get("status"), max_chars=case.get("max_chars"), local=a_local)] if isinstance(a_text, str) else [], accepted=False, status="unreviewed" if isinstance(a_text, str) else _contract_status(a_arm.get("status"), "pending"), context=context, decision_kind="unreviewed"))

        # B comes from the real standard child when available.  A flat B text is
        # accepted only when the saved writer explicitly marked it accepted.
        if child is not None:
            b_row = dict(child)
            b_row.update({"arm": "B", "decision_kind": "arbiter", "generator": child.get("generator") or generator,
                          "generator_mode": child.get("generator_mode") or generator_mode})
            output.append(b_row)
        else:
            b_text = standard.get("text")
            b_local = standard.get("local") if isinstance(standard.get("local"), Mapping) else standard.get("local_fit") if isinstance(standard.get("local_fit"), Mapping) else None
            b_ok = standard.get("status") == "accepted" and isinstance(b_text, str) and bool(b_text.strip())
            output.append(_row(case_id, _source_name(manifest, source_id), original, start=case.get("start"), end=case.get("end"), repetition=repetition, arm="B", generator=generator, generator_mode=generator_mode, arbiter="", candidates=[_candidate("B", b_text, status=standard.get("status"), max_chars=case.get("max_chars"), local=b_local)] if isinstance(b_text, str) else [], initial_verdict=standard.get("verdict"), verdict=standard.get("verdict"), accepted=b_ok, accepted_text=b_text if b_ok else None, status="accepted" if b_ok else _contract_status(standard.get("status"), "pending"), issues=standard.get("issues"), advice=standard.get("advice", ""), context=context, decision_kind="arbiter"))

        raw_candidates = c_arm.get("candidates", []) if isinstance(c_arm.get("candidates"), list) else []
        selected_id = c_arm.get("selected_candidate_id")
        selected_count = sum(isinstance(candidate, Mapping) and candidate.get("candidate_id") == selected_id for candidate in raw_candidates)
        selected = next((candidate for candidate in raw_candidates if isinstance(candidate, Mapping) and candidate.get("candidate_id") == selected_id), None) if selected_count == 1 else None
        c_candidates: list[dict[str, Any]] = []
        for candidate in raw_candidates:
            if not isinstance(candidate, Mapping) or not isinstance(candidate.get("candidate_id"), str):
                continue
            local = candidate.get("local") if isinstance(candidate.get("local"), Mapping) else candidate.get("local_fit") if isinstance(candidate.get("local_fit"), Mapping) else None
            c_candidates.append(_candidate(candidate["candidate_id"], candidate.get("text"), phase="alternative", status=candidate.get("status", ""), max_chars=case.get("max_chars"), local=local, fits=local.get("within_length", local.get("fits")) if local else None, selected=candidate is selected))
        selected_local = selected.get("local") if isinstance(selected, Mapping) and isinstance(selected.get("local"), Mapping) else selected.get("local_fit") if isinstance(selected, Mapping) and isinstance(selected.get("local_fit"), Mapping) else None
        c_valid = bool(c_arm.get("status") == "accepted" and selected is not None and isinstance(selected.get("text"), str) and selected.get("text").strip() and selected_local is not None and selected_local.get("valid") is True and selected_local.get("within_length") is True and (not isinstance(c_arm.get("selected_text"), str) or c_arm.get("selected_text") == selected.get("text")))
        output.append(_row(case_id, _source_name(manifest, source_id), original, start=case.get("start"), end=case.get("end"), repetition=repetition, arm="C", generator=generator, generator_mode=generator_mode, arbiter=selector, arbiter_mode=selector_mode, candidates=c_candidates, verdict=c_arm.get("verdict"), accepted=c_valid, accepted_text=selected.get("text") if c_valid and selected else None, status="accepted" if c_valid else _contract_status(c_arm.get("status"), "unresolved"), issues=c_arm.get("issues"), advice=c_arm.get("advice", ""), context=context, decision_kind="selector"))
    if not output:
        raise ValueError("experiment contains no case rows")
    return _document(root, "A/B/C shortening experiment", manifest.get("run_id", root.name), _run_status(manifest, _cheap_report_metadata(root)), output, _summary_metadata(manifest, len(output), 1, []))


def _standard_child(root: Path, repetition: Any, source_id: Any, case_id: Any) -> dict[str, Any] | None:
    if not isinstance(repetition, int) or not isinstance(source_id, str):
        return None
    path = root / f"r{repetition}" / "standard" / source_id / "report.json"
    try:
        if path.is_symlink() or path.resolve().relative_to(root) is None or not path.is_file():
            return None
        report = _read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(report, Mapping) or not isinstance(report.get("records"), list):
        return None
    selected = next((record for record in report["records"] if isinstance(record, Mapping) and record.get("cue_id", record.get("case_id")) in {case_id, str(case_id).split(":")[-1]}), None)
    if selected is None:
        return None
    one = dict(report)
    one["records"] = [selected]
    return _adapt_trial(root, one)["rows"][0]


def _legacy_child_reports(root: Path) -> dict[str, Any] | None:
    """Combine old standard child reports without following links outside root."""

    combined: list[dict[str, Any]] = []
    for path in sorted(root.rglob("report.json")):
        if path.parent == root or path.is_symlink():
            continue
        try:
            if path.resolve().relative_to(root) is None:
                continue
            value = _read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(value, Mapping) or not isinstance(value.get("records"), list):
            continue
        document = _adapt_trial(root, value)
        repetition = next((int(part[1:]) for part in path.parts if part.startswith("r") and part[1:].isdigit()), None)
        model = path.parent.parent.name if path.parent.name.startswith("r") else path.parent.name
        for row in document["rows"]:
            if repetition is not None:
                row["repetition"] = repetition
            if model and not row.get("generator"):
                row["generator"] = model
            row["arm"] = "standard"
        combined.extend(document["rows"])
    if not combined:
        return None
    summary: Mapping[str, Any] = {}
    try:
        loaded = _read_json(root / "experiment_summary.json")
        summary = loaded if isinstance(loaded, Mapping) else {}
    except ValueError:
        pass
    return _document(root, "Shortening experiment (legacy aggregation)", root.name, _run_status(summary, summary), combined, {"status": _run_status(summary, summary), "settings": f"{len(combined)} child rows"})


def _summary_metadata(value: Mapping[str, Any], count: int, repetitions: int, pairs: Any) -> dict[str, Any]:
    summary = value.get("summary") if isinstance(value.get("summary"), Mapping) else {}
    metadata: dict[str, Any] = {"status": value.get("status", summary.get("status", "partial")), "settings": f"{count} rows; {repetitions} repetitions"}
    if pairs:
        metadata["settings"] = f"{count} unique cases × {repetitions} repetitions × {len(pairs)} pairs"
    cost = summary.get("costs") if isinstance(summary.get("costs"), Mapping) else summary.get("cost") if isinstance(summary.get("cost"), Mapping) else value.get("costs") if isinstance(value.get("costs"), Mapping) else value.get("cost")
    if isinstance(cost, Mapping):
        known = cost.get("known_sum_usd")
        if isinstance(known, (int, float, str)) and not isinstance(known, bool):
            metadata["cost"] = f"Известная сумма ${_string(known) or str(known)}; остальные вызовы могут быть не оценены"
        direct = cost.get("direct") if isinstance(cost.get("direct"), Mapping) else cost
        if isinstance(direct, Mapping) and isinstance(direct.get("known_subtotal_usd"), (int, float)):
            metadata["cost"] = f"Известная сумма ${direct['known_subtotal_usd']:.4f}; остальные вызовы могут быть не оценены"
        upper = cost.get("deepseek_conservative_upper_usd")
        provider = cost.get("provider_actual_usd")
        ds = cost.get("deepseek_upper_usd", cost.get("deepseek_conservative_upper_usd"))
        if provider is not None or ds is not None:
            metadata["cost"] = f"{metadata.get('cost', 'Известная сумма недоступна')}; провайдер фактически {provider if provider is not None else '—'}, верхняя оценка DeepSeek {ds if ds is not None else '—'}"
        elif upper is not None:
            metadata["cost"] = f"{metadata.get('cost', 'Известная сумма недоступна')}; верхняя оценка DeepSeek {upper}"
        if cost.get("subscription_unpriced") is True:
            metadata["cost"] = f"{metadata.get('cost', 'Известная сумма недоступна')}; Стоимость подписки отдельно не оценена"
        if cost.get("global_actual_qualifier"):
            metadata["cost_status"] = _string(cost.get("global_actual_qualifier"))
        for key, label in (("subquota_count", "subquota"), ("subscription_quota_units", "subquota"), ("total_count", "total"), ("global_quota_units", "total"), ("logical_count", "logical"), ("logical_prompts", "logical"), ("physical_calls", "physical"), ("physical_calls_observed", "physical"), ("unknown_call_count", "logical")):
            if isinstance(cost.get(key), (int, float, str)) and not isinstance(cost.get(key), bool):
                metadata[label] = f"{key}: {cost[key]} (счётчик, не счёт-фактура)"
    return metadata


def _document(root: Path, title: str, run_id: Any, status: Any, rows: list[dict[str, Any]], metadata: Mapping[str, Any]) -> dict[str, Any]:
    rows.sort(key=lambda row: (_string(row.get("source")), _string(row.get("case_id")), row.get("repetition") if isinstance(row.get("repetition"), int) else -1, _string(row.get("arm"))))
    return {"title": title, "run_id": _string(run_id) or root.name, "status": _string(status) or "partial", "notices": [], "metadata": _metadata(metadata), "rows": rows}


def _reject_fake(root: Path) -> None:
    names = {"manifest.json", "experiment_manifest.json", "experiment_summary.json", "case_results.json", "ledger.json", "report.json", "run_trialreport.json", "run_trial_report.json", "run_manifest.json"}
    evidence = False
    for evidence_name in ("report.json", "case_results.json", "ledger.json"):
        path = root / evidence_name
        if not path.is_file() or path.is_symlink():
            continue
        try:
            value = _read_json(path)
        except ValueError:
            continue
        if value:
            if evidence_name == "report.json" and isinstance(value, Mapping) and value.get("schema") == "jev_replay.v1":
                if value.get("status") in {"prepared", "blocked"}:
                    raise ValueError("prepared Jev artifact is not publishable")
                if value.get("backend_consumed") is False:
                    raise ValueError("offline backend artifact is not publishable")
            if evidence_name == "report.json" and isinstance(value, Mapping) and value.get("schema") == "jev_followup.v1":
                summary = value.get("summary") if isinstance(value.get("summary"), Mapping) else {}
                attempted = summary.get("attempted_requests")
                if value.get("status") == "prepared" or value.get("backend_consumed") is not True or not isinstance(attempted, int) or attempted <= 0:
                    raise ValueError("prepared or zero-attempt Jev follow-up is not publishable")
            if evidence_name == "report.json" and isinstance(value, Mapping) and isinstance(value.get("records"), list):
                evidence = True
            elif evidence_name == "case_results.json" and (isinstance(value, list) or isinstance(value, Mapping) and isinstance(value.get("rows"), list)):
                evidence = True
            elif evidence_name == "ledger.json" and isinstance(value, Mapping) and ("requests" in value or "calls" in value):
                evidence = True
    if not evidence:
        for path in root.rglob("report.json"):
            if path == root / "report.json" or path.is_symlink() or not path.is_file():
                continue
            try:
                value = _read_json(path)
            except ValueError:
                continue
            if isinstance(value, Mapping) and isinstance(value.get("records"), list) and value["records"]:
                evidence = True
                break
    for path in root.iterdir() if root.exists() else []:
        if path.name not in names or not path.is_file():
            continue
        try:
            value = _read_json(path)
        except ValueError:
            continue
        def visit(item: Any, key: str = "") -> None:
            if isinstance(item, Mapping):
                for child_key, child in item.items():
                    lowered = str(child_key).lower()
                    if lowered in {"synthetic_fixture", "simulation_mode"} and child is True:
                        raise ValueError("fake or simulation artifact is not publishable")
                    if lowered == "backend_consumed" and child is False:
                        raise ValueError("offline backend artifact is not publishable")
                    if lowered == "dry_run" and child is True:
                        raise ValueError("dry-run artifact is not publishable")
                    if lowered in {"status", "mode", "execution_kind"} and isinstance(child, str) and child.lower() in {"dry_run", "dry-run", "simulation", "offline"}:
                        raise ValueError("offline or preparation artifact is not publishable")
                    visit(child, lowered)
            elif isinstance(item, list):
                for child in item:
                    visit(child, key)
        visit(value)
    for name in ("manifest.json", "experiment_manifest.json"):
        path = root / name
        if path.is_file() and not evidence:
            try:
                value = _read_json(path)
            except ValueError:
                continue
            if isinstance(value, Mapping) and _string(value.get("status")).lower() in {"prepare", "prepared"}:
                raise ValueError("preparation artifact has no real report evidence")


def load_report_document(output_dir: Path) -> dict[str, Any]:
    """Load a known saved artifact into the strict renderer data contract."""

    root = Path(output_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    _reject_fake(root)
    manifest_path = root / "manifest.json"
    case_path = root / "case_results.json"
    report_path = root / "report.json"
    if not report_path.exists():
        for candidate in (root / "run_trialreport.json", root / "run_trial_report.json"):
            if candidate.exists():
                report_path = candidate
                break
    if report_path.exists():
        report = _read_json(report_path)
        if isinstance(report, Mapping) and report.get("schema") == "jev_followup.v1":
            return _adapt_jev_followup(root, report)
        if isinstance(report, Mapping) and report.get("schema") == "jev_replay.v1":
            return small_adapt_jev(root, report)
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if isinstance(manifest, Mapping) and manifest.get("schema") == "sh08_matrix_manifest.v1" and case_path.exists():
            return _adapt_sh08(root, dict(manifest), _read_json(case_path))
        if isinstance(manifest, Mapping) and case_path.exists() and any(key in manifest for key in ("models", "sample_case_ids", "corpus_sha256")):
            return _adapt_abc(root, manifest, _read_json(case_path))
        if isinstance(manifest, Mapping) and manifest.get("schema") not in (None, "sh08_matrix_manifest.v1"):
            raise ShorteningHTMLUnknownFormat(f"unknown manifest schema: {manifest.get('schema')}")
    if report_path.exists():
        report = _read_json(report_path)
        if isinstance(report, Mapping) and isinstance(report.get("records"), list):
            return _adapt_trial(root, report)
        raise ShorteningHTMLUnknownFormat("unknown report schema")
    if (root / "experiment_manifest.json").exists() and (root / "experiment_summary.json").exists():
        legacy = _legacy_child_reports(root)
        if legacy is not None:
            return legacy
        summary = _read_json(root / "experiment_summary.json")
        manifest = _read_json(root / "experiment_manifest.json")
        metadata = manifest.get("case_metadata") if isinstance(manifest, Mapping) else None
        if isinstance(metadata, list):
            rows = [_row(item.get("case_id"), item.get("source_id", ""), item.get("original"), start=item.get("start"), end=item.get("end"), repetition=item.get("repetition"), status=_case_status(item), context=item.get("context")) for item in metadata if isinstance(item, Mapping)]
            if rows:
                status_value = _run_status(summary if isinstance(summary, Mapping) else {}, summary if isinstance(summary, Mapping) else {})
                return _document(root, "Shortening experiment (legacy aggregation)", root.name, status_value, rows, {"status": status_value, "settings": "child data unavailable; case metadata retained"})
        raise ShorteningHTMLUnknownFormat("legacy aggregation has no supported child reports or case metadata")
    if case_path.exists() and (root / "experiment_manifest.json").exists():
        manifest = _read_json(root / "experiment_manifest.json")
        if isinstance(manifest, Mapping):
            return _adapt_abc(root, manifest, _read_json(case_path))
    raise ShorteningHTMLUnknownFormat("no supported shortening artifact found")


def _artifact_files(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    allowed_root = {"manifest.json", "experiment_manifest.json", "experiment_summary.json", "case_results.json", "report.json", "ledger.json", "selection.json", "source_snapshots.json", "run_trialreport.json", "run_trial_report.json", "run_manifest.json"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name in {HTML_NAME, SIDECAR_NAME}:
            continue
        try:
            if path.resolve().relative_to(root) is None:
                continue
            relative = path.relative_to(root)
            if len(relative.parts) == 1 and path.name not in allowed_root:
                continue
            if path.name == "report.json":
                try:
                    value = _read_json(path)
                except ValueError:
                    continue
                if not isinstance(value, Mapping) or not (isinstance(value.get("records"), list) or value.get("status")):
                    continue
            if any(part in {"analysis", "source_snapshots"} for part in relative.parts) and path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            if path.name not in allowed_root and not any(part in {"analysis", "source_snapshots", "selection"} for part in relative.parts) and path.name != "report.json":
                continue
            result[relative.as_posix()] = _sha_bytes(path.read_bytes())
        except (OSError, ValueError):
            continue
    return result


def _valid_sidecar(root: Path) -> bool:
    path = root / SIDECAR_NAME
    html = root / HTML_NAME
    if not path.is_file() or not html.is_file():
        return False
    try:
        value = _read_json(path)
        return isinstance(value, Mapping) and value.get("schema") == SIDECAR_SCHEMA and value.get("schema_version") == 1 and value.get("root") == str(root) and value.get("execution_kind") in {"live_model_run", "explicit_user_confirmed_backfill"} and value.get("live_run") is True and HTML_MARKER in html.read_text(encoding="utf-8") and value.get("html_sha256") == _sha_bytes(html.read_bytes())
    except (OSError, ValueError, UnicodeError):
        return False


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_html_report(output_dir: Path, *, confirmed_real_run: bool = False) -> Path | None:
    """Render and atomically publish only an observed or explicitly confirmed run."""

    root = Path(output_dir).resolve()
    observation = _ACTIVE.get()
    observed = observation is not None and observation.root == root and observation.model_requests > 0
    if not confirmed_real_run and not observed and not _valid_sidecar(root):
        return None
    if confirmed_real_run:
        _reject_fake(root)
    return _publish_safely(root, confirmed_real_run=confirmed_real_run, raise_errors=True)


def _publish_safely(root: Path, *, confirmed_real_run: bool, raise_errors: bool = False) -> Path | None:
    try:
        document = load_report_document(root)
        from utils.shortening_html_view import render_report

        html = render_report(document)
        if not isinstance(html, str):
            raise TypeError("render_report must return str")
        if HTML_MARKER not in html:
            html = HTML_MARKER + "\n" + html
        path = root / HTML_NAME
        if path.exists():
            try:
                existing = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                existing = ""
            if HTML_MARKER not in existing:
                raise FileExistsError("refusing to overwrite foreign shortening.html")
        _atomic_bytes(path, html.encode("utf-8"))
        metadata = {"schema": SIDECAR_SCHEMA, "schema_version": 1, "execution_kind": "live_model_run" if not confirmed_real_run else "explicit_user_confirmed_backfill", "source": "explicit_live_entrypoint" if not confirmed_real_run else "explicit_user_confirmed_backfill", "live_run": True, "confirmed_real_run": confirmed_real_run, "root": str(root), "source_files_sha256": _artifact_files(root), "html_sha256": _sha_bytes(html.encode("utf-8")), "rowcount": len(document.get("rows", [])), "status": document.get("status", "partial")}
        _atomic_bytes(root / SIDECAR_NAME, (_canonical(metadata) + "\n").encode("utf-8"))
        return path
    except Exception as exc:
        if raise_errors:
            raise
        try:
            warnings.warn(f"shortening HTML report not published ({type(exc).__name__}) at {root}", RuntimeWarning, stacklevel=2)
        except Exception:
            pass
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish an offline shortening HTML report")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--confirmed-real-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = publish_html_report(args.output_dir, confirmed_real_run=args.confirmed_real_run)
    except Exception as exc:
        print(f"Не удалось опубликовать отчёт: {_safe_error(exc) or type(exc).__name__}", file=sys.stderr)
        return 1
    if result is not None:
        print(result)
        return 0
    print("Публикация запрещена: нужен реальный запрос модели или --confirmed-real-run.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
