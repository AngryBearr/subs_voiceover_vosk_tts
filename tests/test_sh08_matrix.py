from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
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


def test_restricted_prepare_is_fresh_fixed_scope_and_freezes_low_high_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prepare both restricted arms from local fixtures without importing receipts."""
    corpus_dir = tmp_path / "corpus"
    input_root = tmp_path / "inputs"
    pilot_root = tmp_path / "pilot-inputs"
    corpus_dir.mkdir()
    input_root.mkdir()
    pilot_root.mkdir()
    items_by_raw: dict[str, list[dict[str, Any]]] = {}

    def make_source(source_id: str, root: Path, *, path: str, split: str) -> dict[str, Any]:
        items = [{"index": number + 1, "text": f"{source_id} cue {number}"} for number in range(10)]
        raw_path = root / path
        raw_path.write_text(source_id, encoding="utf-8")
        items_by_raw[str(raw_path.resolve())] = items
        json_name = f"{source_id}.json"
        (corpus_dir / json_name).write_text(json.dumps(items), encoding="utf-8")
        return {
            "source_id": source_id,
            "path": path,
            "json_path": json_name,
            "sha256": matrix._sha_file(raw_path),
            "parsed_sha256": matrix.canonical_hash(items),
            "split": split,
        }

    sources = [
        make_source(f"pilot-{number}", input_root, path=f"pilot-{number}.srt", split="pilot")
        for number in range(3)
    ]
    test_sources = [
        make_source(f"source-{number}", input_root, path=f"source-{number}.srt", split="test")
        for number in range(7)
    ]
    historical = make_source(
        "historical", input_root, path="Survivor.S44E11.srt", split="test"
    )
    sources.extend(test_sources)
    sources.append(historical)
    pilot_rows = []
    for number, source in enumerate(sources[:3]):
        pilot_path = pilot_root / f"pilot-{number}.srt"
        pilot_path.write_text((input_root / source["path"]).read_text(encoding="utf-8"), encoding="utf-8")
        items_by_raw[str(pilot_path.resolve())] = items_by_raw[
            str((input_root / source["path"]).resolve())
        ]
        pilot_rows.append({"filename": pilot_path.name, "sha256": source["sha256"]})

    corpus_path = corpus_dir / "manifest.json"
    corpus_path.write_text(
        json.dumps({
            "input_root": str(input_root),
            "planner_config": {"cps": 13.0, "max_speed": 1.5, "gap_ms": 200, "context_before": 3, "context_after": 3},
            "sources": sources,
        }),
        encoding="utf-8",
    )
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(json.dumps({"corpus_root": str(pilot_root), "sources": pilot_rows}), encoding="utf-8")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps({
            "identity": {
                "engine": "fixture", "voice": 8, "rate": 1.5,
                "model_config_sha256": "config", "model_fingerprint": "model",
                "normalizer": "vosk_punctuation_v1_no_ruaccent", "normalizer_sha256": "normalizer",
            },
            "status": "ok",
            "normalizer": "vosk_punctuation_v1_no_ruaccent",
            "input_character_basis": "normalized_tts_text",
            "rate": 1.5,
            "models": {"char_linear": {"name": "char_linear", "rate": 1.5, "coefficients": [0.0, 1.0]}},
            "char_linear_guard_sec": 0.0,
        }),
        encoding="utf-8",
    )
    model_path = tmp_path / "model"
    approval_path = tmp_path / "approval.json"
    frozen_path = tmp_path / "frozen-selection.json"
    approval = json.loads(
        (Path(__file__).resolve().parents[1] / "docs/sh08_followup_protocol_20260915.json").read_text(
            encoding="utf-8"
        )
    )
    approval["selection_reference"] = {"path": str(frozen_path), "sha256": "placeholder"}
    approval["permitted_output_dirs"] = {
        "low": str(tmp_path / "low"),
        "high": str(tmp_path / "high"),
    }
    approval_path.write_text(json.dumps(approval), encoding="utf-8")

    def fake_parse(path: str) -> list[dict[str, Any]]:
        return deepcopy(items_by_raw[str(Path(path).resolve())])

    def fake_plans(items: list[dict[str, Any]], _planner: object) -> list[Any]:
        return [
            SimpleNamespace(
                cue_id=f"cue-{number}", position=number, start=number * 1000, end=number * 1000 + 1000,
                available_ms=1000, max_chars=30, status="needs_shortening", severity="easy",
                reason="fixture", short_timing_risk=False, length_basis="display_text",
            )
            for number in range(len(items))
        ]

    monkeypatch.setattr("utils.srt_to_json.parse_srt_file", fake_parse)
    monkeypatch.setattr(matrix, "plan_subtitles", fake_plans)
    monkeypatch.setattr(
        matrix,
        "context_items",
        lambda items, plans, cue_ids, planner: [
            {"cue_id": cue_id, "text": items[plans[0].position]["text"], "start": 0, "end": 1000}
            for cue_id in cue_ids
        ],
    )
    fixture_identity = {
        "engine": "fixture", "voice": 8, "rate": 1.5,
        "model_config_sha256": "config", "model_fingerprint": "model",
        "normalizer": "vosk_punctuation_v1_no_ruaccent", "normalizer_sha256": "normalizer",
    }
    monkeypatch.setattr(matrix, "_profile_identity", lambda profile, model: fixture_identity)
    monkeypatch.setattr(matrix, "_code_hashes", lambda **kwargs: {"fixture.py": "fixture"})
    monkeypatch.setattr(matrix, "_import_receipts", lambda approval, manifest: {"schema": "sh08_matrix_ledger.v1", "requests": {}})

    def fake_validate(approved: dict[str, Any], selection: Any = None) -> dict[str, Any]:
        if selection is None:
            return {}
        frozen_path.write_text(json.dumps(selection), encoding="utf-8")
        return {"selection": selection, "selection_reference": {"path": str(frozen_path), "sha256": matrix._sha_file(frozen_path)}}

    monkeypatch.setattr(matrix, "_validate_restricted_protocol", fake_validate)
    manifests = {}
    for effort in ("low", "high"):
        output_dir = tmp_path / effort
        manifests[effort] = matrix.prepare_manifest(
            output_dir,
            corpus_manifest_path=corpus_path,
            pilot_manifest_path=pilot_path,
            profile_path=profile_path,
            model_path=model_path,
            input_root=input_root,
            approval_path=approval_path,
            muse_effort=effort,
        )

    for effort, manifest in manifests.items():
        assert len(manifest["selection"]["cases"]) == 60
        assert manifest["approved_scope"]["repetitions"] == 2
        assert manifest["approved_scope"]["pairs"] == [["muse", "sol"]]
        assert [len(batch["case_ids"]) for batch in manifest["batches"]] == [6, 27, 27]
        assert manifest["muse_effort"] == effort
        assert manifest["full_specs"] == matrix.effective_specs(effort)
        assert manifest["limits"]["direct_api_cap_usd"] == "1.50"
        assert manifest["limits"]["subscription_conservative_attempt_units"] == 75
        assert manifest["limits"]["total_attempt_units"] == 150
        assert manifest["carryover"] == matrix._zero_carryover()
        assert manifest["receipt_import"] == {"enabled": False}
        assert matrix._load_ledger(tmp_path / effort)["requests"] == {}
    assert manifests["low"]["full_specs"]["muse"]["variant"] == "low"
    assert manifests["high"]["full_specs"]["muse"]["variant"] == "high"
    assert matrix.DEFAULT_SPECS["muse"]["variant"] == "low"


def test_restricted_real_protocol_prepare_and_read_manifest(tmp_path: Path) -> None:
    """Validate and round-trip both restricted arms using only local fixtures."""
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "docs/sh08_followup_protocol_20260915.json"
    corpus_path = root / "output/shortening_corpus_20260910_v2/manifest.json"
    pilot_path = root / "docs/sh08_pilot_manifest.json"
    profile_path = root / "output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json"
    selection_path = root / "output/sh08_eightpair_20260914_r3/selection.json"
    if not all(path.is_file() for path in (protocol_path, corpus_path, pilot_path, profile_path, selection_path)):
        pytest.skip("requires local SH08 baseline fixtures")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    corpus_root = Path(corpus["input_root"])
    pilot_root = Path(pilot["corpus_root"])
    required = [root / str(pilot["model"]["path"]), profile_path]
    required.extend(corpus_path.parent / source["json_path"] for source in corpus["sources"])
    required.extend(corpus_root / source["path"] for source in corpus["sources"])
    required.extend(pilot_root / source["filename"] for source in pilot["sources"])
    if not all(path.exists() for path in required):
        pytest.skip("requires all local SH08 corpus and model fixtures")

    approval = json.loads(protocol_path.read_text(encoding="utf-8"))
    approval["permitted_output_dirs"] = {
        "low": str(tmp_path / "low"),
        "high": str(tmp_path / "high"),
    }
    copied_protocol = tmp_path / "protocol.json"
    copied_protocol.write_text(json.dumps(approval), encoding="utf-8")
    for effort in ("low", "high"):
        output_dir = tmp_path / effort
        prepared = matrix.prepare_manifest(
            output_dir,
            approval_path=copied_protocol,
            muse_effort=effort,
        )
        assert prepared["execution_authorized"] is True
        assert matrix._read_manifest(output_dir) == prepared
