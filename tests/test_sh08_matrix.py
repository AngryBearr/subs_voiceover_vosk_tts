from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

import utils.sh08_matrix as matrix


def _manifest() -> dict[str, Any]:
    cases = []
    for number in range(60):
        case_id = f"source-{number // 10}:cue-{number}"
        cases.append({
            "case_id": case_id,
            "source_id": f"source-{number // 10}",
            "cue_id": case_id,
            "old_id": f"cue-{number}",
            "new_id": case_id,
            "position": number,
            "source_index": number + 1,
            "text": "original text",
            "original": "original text",
            "start": number * 1000,
            "end": number * 1000 + 1000,
            "available_ms": 1000,
            "max_chars": 30,
            "status": "needs_shortening",
            "risk": {"severity": "easy"},
            "length_basis": "display_text",
            "context": [{"cue_id": case_id, "text": "original text", "start": number * 1000, "end": number * 1000 + 1000}],
            "rank_sha256": "x",
        })
    batches = [{"batch_id": "batch-0", "case_ids": [row["case_id"] for row in cases[:6]]}]
    batches.extend({"batch_id": f"batch-{number}", "case_ids": [row["case_id"] for row in cases[start:start + 27]]} for number, start in ((1, 6), (2, 33)))
    manifest = {
        "schema": "sh08_matrix_manifest.v1",
        "version": matrix.VERSION,
        "approved_scope": {"cases": 60, "repetitions": 2, "pairs": [list(pair) for pair in matrix.PAIR_ORDER]},
        "selection": {"version": matrix.VERSION, "cases": cases, "selected_case_ids": [row["case_id"] for row in cases]},
        "full_specs": matrix.DEFAULT_SPECS,
        "batches": batches,
        "code_hashes": {},
    }
    manifest["checksum"] = matrix.canonical_hash(manifest)
    return manifest


def test_reservation_uses_advertised_context_and_full_output() -> None:
    expected = (matrix.Decimal(1_048_576) * matrix.Decimal("0.30") / matrix.Decimal(1_000_000) + matrix.Decimal(16_384) * matrix.Decimal("1.20") / matrix.Decimal(1_000_000)) * matrix.Decimal("1.10")
    assert matrix._reservation(matrix.DEFAULT_SPECS["deepseek"]) == expected


@pytest.mark.parametrize("stage,arbiter,expected", [("generation", None, "luna"), ("repair", "deepseek", "luna"), ("review", "deepseek", "deepseek"), ("recheck", "deepseek", "deepseek")])
def test_descriptor_transport_key_follows_stage(stage: str, arbiter: str | None, expected: str) -> None:
    descriptor = matrix._request_descriptor(_manifest(), rep=1, stage=stage, generator="luna", arbiter=arbiter, batch="batch-0", system="s", user="u", schema={})
    assert descriptor["model_key"] == expected
    assert descriptor["generator"] == "luna"
    assert descriptor["arbiter"] == arbiter


def test_generation_cannot_shorten_is_not_candidate() -> None:
    manifest = _manifest()
    payload = {"results": [{"cue_id": manifest["selection"]["cases"][0]["cue_id"], "text": "original text", "status": "cannot_shorten"}]}
    parsed = matrix.parse_generation_partial(json.dumps(payload), [payload["results"][0]["cue_id"]])
    assert parsed.results[payload["results"][0]["cue_id"]]["status"] == "cannot_shorten"


def test_context_target_flag_is_union() -> None:
    merged = matrix._merge_context([{"cue_id": "x", "text": "x", "is_target": False}, {"cue_id": "x", "text": "x", "is_target": True}])
    assert merged == [{"cue_id": "x", "text": "x", "is_target": True}]


def test_manifest_selection_keeps_source_qualified_ids() -> None:
    manifest = _manifest()
    assert len(manifest["selection"]["cases"]) == 60
    assert all(":" in row["cue_id"] for row in manifest["selection"]["cases"])


def test_admission_refuses_direct_cap_before_backend(tmp_path: Path) -> None:
    manifest = _manifest()
    ledger = {"schema": "sh08_matrix_ledger.v1", "requests": {}}
    admission = matrix._Admission(tmp_path, manifest, ledger, {"deepseek": matrix.DEFAULT_SPECS["deepseek"]})
    reserve = matrix._reservation(matrix.DEFAULT_SPECS["deepseek"])
    for number in range(14):
        ledger["requests"][str(number)] = {"status": "complete", "cost_usd": str(reserve), "quota_units": 1, "subscription": False}
    with pytest.raises(matrix.BudgetLimited):
        admission.admit({"model_key": "deepseek", "payload_bytes": 1}, key="next")


def test_smoke_reuses_cached_response_on_full_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    manifest = _manifest()
    matrix._save_ledger(output_dir, {"schema": "sh08_matrix_ledger.v1", "manifest_checksum": manifest["checksum"], "requests": {}})
    monkeypatch.setattr(matrix, "_read_manifest", lambda _: manifest)
    calls: list[tuple[str, str]] = []

    async def fake(model_key: str, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        calls.append((model_key, system))
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        ids = [row["cue_id"] for row in payload["targets"]]
        if schema == matrix.generation_schema():
            text = '{"results":[' + ','.join('{"cue_id":' + repr(case_id).replace("'", '"') + ',"text":"short","status":"ok"}' for case_id in ids) + ']}'
        else:
            text = '{"results":[' + ','.join('{"cue_id":' + repr(case_id).replace("'", '"') + ',"verdict":"accept","issues":[],"advice":""}' for case_id in ids) + ']}'
        spec = matrix.MODEL_SPECS[model_key]
        return {"text": text, "raw": text, "calls": [], "usage": {"input_tokens": 1, "output_tokens": 1}, "cost_usd": 0, "cost_known": True, "error": None, "physical_calls_observed": 1, "model": spec["model"], "provider": spec["provider"], "elapsed_sec": 0}

    result = asyncio.run(matrix.run_matrix(output_dir, fake, smoke=True))
    assert result["status"] == "smoke_complete"
    assert len(calls) > 1
    first_count = len(calls)
    asyncio.run(matrix.run_matrix(output_dir, fake, smoke=True))
    assert len(calls) == first_count
    full_result = asyncio.run(matrix.run_matrix(output_dir, fake, smoke=False))
    assert full_result["status"] == "complete"
    completed_count = len(calls)
    asyncio.run(matrix.run_matrix(output_dir, fake, smoke=False))
    assert len(calls) == completed_count
    (output_dir / "case_results.json").write_text("{}", encoding="utf-8")
    reported = matrix.report_matrix(output_dir)
    assert reported["firstpass_counts"] == full_result["firstpass_counts"]
    assert len(calls) == completed_count


def test_no_permission_cli_never_imports_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    manifest = _manifest()
    monkeypatch.setattr(matrix, "_read_manifest", lambda _: manifest)
    monkeypatch.setattr(matrix, "_load_ledger", lambda *_args, **_kwargs: {"schema": "sh08_matrix_ledger.v1", "requests": {}})
    monkeypatch.setattr(matrix, "_write_outputs", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(matrix, "_cli", matrix._cli)
    monkeypatch.setattr("sys.argv", ["sh08_matrix", "run", "--output-dir", str(output_dir)])
    assert matrix._cli() != 0


@pytest.mark.parametrize('currency', [0.0, 0.0025])
def test_currency_usage_is_not_a_token_counter(currency: float) -> None:
    response = {'cost_known': True, 'cost_usd': currency, 'usage': {'input_tokens': 10, 'output_tokens': 5, 'cost': currency}}
    cost, known = matrix._usage_cost(response, matrix.MODEL_SPECS['gemini'])
    assert known and cost == matrix.Decimal(str(currency))


def test_initial_acceptance_is_retained_in_final_report() -> None:
    manifest = _manifest()
    row = {'case_id': manifest['selection']['cases'][0]['case_id'], 'branches': [{'stage': 'review', 'rep': 1, 'generator': 'luna', 'arbiter': 'sol', 'status': 'complete', 'accepted': True, 'verdict': 'accept', 'case_id': manifest['selection']['cases'][0]['case_id'], 'local_fit': {'fits': True}}]}
    report = matrix._report(manifest, {'requests': {}}, [row], status='complete')
    assert report['finalaccepted_counts']['luna->sol'] == 1
