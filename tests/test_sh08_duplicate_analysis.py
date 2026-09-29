from __future__ import annotations

import json

from utils.sh08_duplicate_analysis import analyze
from utils.shortening_resume import canonical_hash


def _request(rep: int, stage: str, generator: str, arbiter: str | None, text: str, targets: list[dict[str, object]], context: list[dict[str, object]] | None = None) -> tuple[str, dict[str, object]]:
    descriptor: dict[str, object] = {
        "rep": rep, "stage": stage, "generator": generator, "arbiter": arbiter,
        "batch": "b", "spec": {"model": generator}, "system_prompt": "system",
        "user_prompt": "header\nInput JSON:\n" + json.dumps({"context": context or [], "targets": targets}),
        "schema": {"type": "object"},
    }
    key = canonical_hash(descriptor)
    return key, {"key": key, "status": "complete", "descriptor": descriptor, "response": {"text": text}, "text": text}


def _sources(request_rows: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    case_ids = [f"case-{n}" for n in range(60)]
    rows = [{"case_id": case_id, "branches": []} for case_id in case_ids]
    selection = {"cases": [{"case_id": case_id} for case_id in case_ids]}
    requests = {str(row["key"]): row for row in request_rows}
    ledger: dict[str, object] = {"requests": requests}
    ledger["checksum"] = canonical_hash(ledger)
    case_results: dict[str, object] = {"rows": rows, "ledger_checksum": ledger["checksum"]}
    case_results["checksum"] = canonical_hash(case_results)
    return case_results, ledger, selection


def _generation_payload(text: str, case_id: str = "case-0", status: str = "ok") -> str:
    return json.dumps({"results": [{"cue_id": case_id, "text": text, "status": status}]})


def test_independent_exact_duplicate_and_changed_proposal_are_benchmark_evidence() -> None:
    targets = [{"cue_id": "case-0", "original": "old", "max_chars": 10, "length_basis": "x"}]
    rows = []
    for rep, text in ((1, "same"), (2, "same")):
        key, row = _request(rep, "generation", "luna", None, _generation_payload(text), targets)
        rows.append(row)
    for rep, text in ((1, "left"), (2, "right")):
        key, row = _request(rep, "generation", "deepseek", None, _generation_payload(text), targets)
        rows.append(row)
    case_results, ledger, selection = _sources(rows)
    report = analyze(case_results, ledger, selection)
    luna = report["generation"]["per_generator"]["luna"]
    deepseek = report["generation"]["per_generator"]["deepseek"]
    assert luna["exact_duplicate_count"] == 1
    assert luna["intersection_ok_denominator"] == 1
    assert deepseek["exact_duplicate_count"] == 0
    assert report["generation"]["total"]["benchmark_evidence_only"] is True


def test_missing_malformed_and_whitespace_only_are_distinguished() -> None:
    target = {"cue_id": "case-0", "original": "old", "max_chars": 10, "length_basis": "x"}
    rows = []
    key, row = _request(1, "generation", "luna", None, _generation_payload("a  b"), [target])
    rows.append(row)
    key, row = _request(2, "generation", "luna", None, _generation_payload("a b"), [target])
    rows.append(row)
    key, row = _request(1, "generation", "deepseek", None, "not-json", [target])
    rows.append(row)
    key, row = _request(2, "generation", "deepseek", None, "", [target])
    row["response"] = {}
    row["text"] = None
    rows.append(row)
    case_results, ledger, selection = _sources(rows)
    report = analyze(case_results, ledger, selection)
    assert report["generation"]["per_generator"]["luna"]["exact_duplicate_count"] == 0
    assert report["generation"]["per_generator"]["luna"]["whitespace_normalized_duplicate_count"] == 1
    deepseek = report["generation"]["per_generator"]["deepseek"]
    assert deepseek["intersection_ok_denominator"] == 0
    assert deepseek["unavailable_response_targets"] == 1
    assert deepseek["parse_error_targets"] == 1


def test_repair_duplicate_is_not_proven_avoidable_with_different_arbiter_context() -> None:
    target = {"cue_id": "case-0", "original": "old", "candidate": "prior", "max_chars": 10, "length_basis": "x"}
    repair_target = dict(target)
    repair_target["prior_candidate"] = "prior"
    initial_key, initial = _request(1, "review", "luna", "sol", '{"results":[{"cue_id":"case-0","verdict":"revise","issues":[],"advice":"retry"}]}', [target], [{"cue_id": "context", "text": "one"}])
    repair_key, repair = _request(1, "repair", "luna", "sol", _generation_payload("prior"), [repair_target], [{"cue_id": "context", "text": "one"}])
    recheck_target = dict(target)
    recheck_target["candidate"] = "prior"
    recheck_key, recheck = _request(1, "recheck", "luna", "sol", '{"results":[{"cue_id":"case-0","verdict":"accept","issues":[],"advice":""}]}', [recheck_target], [{"cue_id": "context", "text": "different"}])
    case_results, ledger, selection = _sources([initial, repair, recheck])
    report = analyze(case_results, ledger, selection)
    evidence = report["repairs"]["duplicate_repair_evidence"][0]
    assert evidence["recheck_sent"] is True
    assert evidence["avoidability"] == "unknown"
    assert evidence["recheck_scope"] == "different_scope"


def test_repair_duplicate_with_different_arbiter_is_not_proven_avoidable() -> None:
    target = {"cue_id": "case-0", "original": "old", "candidate": "prior", "max_chars": 10, "length_basis": "x"}
    repair_target = dict(target)
    repair_target["prior_candidate"] = "prior"
    _, initial = _request(1, "review", "luna", "sol", '{"results":[{"cue_id":"case-0","verdict":"revise","issues":[],"advice":"retry"}]}', [target])
    _, repair = _request(1, "repair", "luna", "sol", _generation_payload("prior"), [repair_target])
    _, recheck = _request(1, "recheck", "luna", "luna", '{"results":[{"cue_id":"case-0","verdict":"accept","issues":[],"advice":""}]}', [target])
    case_results, ledger, selection = _sources([initial, repair, recheck])
    evidence = analyze(case_results, ledger, selection)["repairs"]["duplicate_repair_evidence"][0]
    assert evidence["avoidability"] == "unknown"
    assert evidence["recheck_sent"] is False
