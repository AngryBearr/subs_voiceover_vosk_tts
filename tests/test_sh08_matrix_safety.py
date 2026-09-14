from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
import socket
from typing import Any

import pytest

import utils.sh08_matrix as matrix
from utils import sh08_matrix_transport as transport

from tests.test_sh08_matrix import _manifest


@pytest.fixture(autouse=True)
def block_real_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make an accidental network path fail loudly in every safety test."""

    real_socket = socket.socket

    def blocked_socket(*args: Any, **kwargs: Any) -> Any:
        # asyncio uses an AF_UNIX self-pipe; only internet sockets are forbidden.
        family = args[0] if args else kwargs.get("family", socket.AF_INET)
        if family == socket.AF_UNIX:
            return real_socket(*args, **kwargs)
        raise AssertionError("real network access is forbidden in SH08 safety tests")

    monkeypatch.setattr(socket, "socket", blocked_socket)


def _new_run(tmp_path: Path, manifest: dict[str, Any]) -> Path:
    output_dir = tmp_path / "matrix"
    output_dir.mkdir()
    matrix._save_ledger(
        output_dir,
        {
            "schema": "sh08_matrix_ledger.v1",
            "manifest_checksum": manifest["checksum"],
            "requests": {},
        },
    )
    return output_dir


def _descriptor(
    manifest: dict[str, Any],
    *,
    model_key: str = "deepseek",
    rep: int = 1,
    stage: str = "generation",
    generator: str | None = None,
    arbiter: str | None = None,
    batch: str = "batch-0",
) -> dict[str, Any]:
    selected_generator = generator or model_key
    return matrix._request_descriptor(
        manifest,
        rep=rep,
        stage=stage,
        generator=selected_generator,
        arbiter=arbiter,
        batch=batch,
        system="system",
        user="user",
        schema={},
    )


def _response(
    model_key: str,
    *,
    cost: float = 0.001,
    input_tokens: int = 1,
    output_tokens: int = 1,
    error: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    physical: int | None = 1,
    tool_calls: int = 0,
) -> dict[str, Any]:
    spec = matrix.DEFAULT_SPECS[model_key]
    return {
        "text": "{}",
        "raw": "{}",
        "calls": [{"status": "success", "tool_calls": tool_calls}],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "cost_usd": cost,
        "cost_known": True,
        "cost_kind": "test",
        "error": error,
        "physical_calls_observed": physical if physical is not None else None,
        "model": model or spec["model"],
        "provider": provider or spec["provider"],
        "elapsed_sec": 0.0,
    }


def test_openrouter_receipt_finishes_admission_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise mocked HTTP through the real transport, admission, and ledger reload."""

    manifest = _manifest()
    output_dir = _new_run(tmp_path, manifest)
    seen: list[dict[str, Any]] = []

    async def fake_post(*, headers: dict[str, str], body: dict[str, Any]) -> tuple[int, bytes]:
        assert headers["Authorization"] == "Bearer fake-openrouter-key"
        seen.append(body)
        return 200, json.dumps(
            {
                "model": "google/gemini-3.8-flash",
                "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 2000, "cost": 0.004125},
            }
        ).encode()

    backend = transport.MatrixBackend(openrouter_no_byok=True)
    # Override credentials directly so neither the process environment nor .env is consulted.
    backend._credentials = {"OPENROUTER_API_KEY": "fake-openrouter-key"}
    monkeypatch.setattr(backend, "_post_openrouter", fake_post)
    receipt = asyncio.run(backend("gemini", "system", "user", {"type": "object"}))

    assert len(seen) == 1
    assert isinstance(receipt["cost_usd"], float)
    assert receipt["cost_known"] is True
    descriptor = matrix._request_descriptor(
        manifest,
        rep=1,
        stage="generation",
        generator="gemini",
        arbiter=None,
        batch="batch-0",
        system="system",
        user="user",
        schema={"type": "object"},
    )
    key = matrix.canonical_hash(descriptor)
    assert key == matrix.canonical_hash(descriptor)
    admission = matrix._Admission(output_dir, manifest, matrix._load_ledger(output_dir), matrix.DEFAULT_SPECS)
    assert admission.admit(descriptor, key=key) is None
    completed = admission.finish(key, receipt)
    assert completed["status"] == "complete"
    resumed = matrix._load_ledger(output_dir, manifest=manifest)
    assert resumed["requests"][key]["status"] == "complete"
    assert resumed["requests"][key]["response"]["cost_usd"] == pytest.approx(0.004125)


@pytest.mark.parametrize("model_key", tuple(matrix.DEFAULT_SPECS))
@pytest.mark.parametrize("failure", ["missing_cost", "error", "mismatch", "tool_call"])
def test_unknown_accounting_is_durable_and_blocks_every_following_model(
    tmp_path: Path, model_key: str, failure: str
) -> None:
    manifest = _manifest()
    output_dir = _new_run(tmp_path, manifest)
    ledger = matrix._load_ledger(output_dir, manifest=manifest)
    admission = matrix._Admission(output_dir, manifest, ledger, matrix.DEFAULT_SPECS)
    descriptor = _descriptor(manifest, model_key=model_key)
    key = matrix.canonical_hash(descriptor)
    assert admission.admit(descriptor, key=key) is None

    invalid = _response(model_key, physical=None if matrix.DEFAULT_SPECS[model_key]["subscription"] else 1)
    if failure == "missing_cost":
        invalid["cost_usd"] = None
        invalid["cost_known"] = False
    elif failure == "error":
        invalid["error"] = "provider_error"
    elif failure == "mismatch":
        invalid["model"] = "another-model"
    else:
        invalid["calls"] = [{"status": "success", "tool_calls": 1}]

    with pytest.raises(matrix.UnknownRequest):
        admission.finish(key, invalid)
    durable = matrix._load_ledger(output_dir, manifest=manifest)
    assert durable["requests"][key]["status"] == "unknown"
    assert durable["requests"][key]["key"] == key
    assert durable["requests"][key]["descriptor"] == descriptor

    next_model = "gemini" if model_key != "gemini" else "luna"
    next_descriptor = _descriptor(manifest, model_key=next_model, rep=2)
    next_key = matrix.canonical_hash(next_descriptor)
    assert next_key != key
    with pytest.raises(matrix.UnknownRequest):
        admission.admit(next_descriptor, key=next_key)


def _complete_row(manifest: dict[str, Any], model_key: str, number: int, *, cost: float) -> tuple[str, dict[str, Any]]:
    descriptor = {
        "model_key": model_key,
        "rep": 99,
        "stage": "fixture",
        "number": number,
    }
    key = matrix.canonical_hash(descriptor)
    spec = matrix.DEFAULT_SPECS[model_key]
    input_tokens = 1_048_576 if model_key == "deepseek" and cost > 0 else 0
    output_tokens = 16_384 if model_key == "deepseek" and cost > 0 else 0
    response = _response(
        model_key,
        cost=cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        physical=None if spec["subscription"] else 1,
    )
    row = {
        "key": key,
        "status": "complete",
        "descriptor": descriptor,
        "reservation_usd": "0",
        "cost_usd": str(cost),
        "quota_units": spec["quota_units"],
        "subscription": spec["subscription"],
        "response": response,
        "text": response["text"],
        "raw": response["raw"],
        "calls": response["calls"],
        "usage": response["usage"],
        "cost_known": True,
    }
    return key, row


def _admission_at_cap(tmp_path: Path, model_key: str, rows: dict[str, dict[str, Any]]) -> tuple[matrix._Admission, dict[str, Any], bytes]:
    manifest = _manifest()
    output_dir = _new_run(tmp_path, manifest)
    ledger = {
        "schema": "sh08_matrix_ledger.v1",
        "manifest_checksum": manifest["checksum"],
        "requests": rows,
    }
    matrix._save_ledger(output_dir, ledger)
    before = (output_dir / "ledger.json").read_bytes()
    return matrix._Admission(output_dir, manifest, ledger, matrix.DEFAULT_SPECS), _descriptor(manifest, model_key=model_key), before


def test_pure_admit_blocks_subscription_300_units_before_ledger_write(tmp_path: Path) -> None:
    manifest = _manifest()
    rows = dict(_complete_row(manifest, "luna", number, cost=0.0) for number in range(50))
    admission, descriptor, before = _admission_at_cap(tmp_path, "luna", rows)
    assert admission._totals() == (matrix.Decimal(0), matrix.Decimal(0), 300, 300)
    with pytest.raises(matrix.BudgetLimited):
        admission.admit(descriptor, key=matrix.canonical_hash(descriptor))
    assert len(admission.ledger["requests"]) == 50
    assert (admission.output_dir / "ledger.json").read_bytes() == before


def test_pure_admit_blocks_total_600_units_before_ledger_write(tmp_path: Path) -> None:
    manifest = _manifest()
    rows = dict(_complete_row(manifest, "deepseek", number, cost=0.0) for number in range(600))
    admission, descriptor, before = _admission_at_cap(tmp_path, "deepseek", rows)
    assert admission._totals() == (matrix.Decimal(0), matrix.Decimal(0), 0, 600)
    with pytest.raises(matrix.BudgetLimited):
        admission.admit(descriptor, key=matrix.canonical_hash(descriptor))
    assert len(admission.ledger["requests"]) == 600
    assert (admission.output_dir / "ledger.json").read_bytes() == before


def test_pure_admit_blocks_five_dollar_direct_cap_before_ledger_write(tmp_path: Path) -> None:
    manifest = _manifest()
    one_cost = float(
        (
            matrix.Decimal(1_048_576) * matrix.Decimal("0.30")
            + matrix.Decimal(16_384) * matrix.Decimal("1.20")
        )
        / matrix.Decimal(1_000_000)
    )
    rows = dict(_complete_row(manifest, "deepseek", number, cost=one_cost) for number in range(16))
    admission, descriptor, before = _admission_at_cap(tmp_path, "deepseek", rows)
    known, pending, sub, total = admission._totals()
    assert known > matrix.Decimal("5")
    assert pending == matrix.Decimal(0) and sub == 0 and total == 16
    with pytest.raises(matrix.BudgetLimited):
        admission.admit(descriptor, key=matrix.canonical_hash(descriptor))
    assert len(admission.ledger["requests"]) == 16
    assert (admission.output_dir / "ledger.json").read_bytes() == before


@pytest.fixture
def unit_read_manifest(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Build a frozen, local-only manifest shape without weakening read validation."""

    manifest = _manifest()
    manifest["approved_scope"].update({"smoke_cases": 6, "smoke_repetition": 1})
    corpus_dir = tmp_path / "corpus"
    input_root = tmp_path / "inputs"
    corpus_dir.mkdir()
    input_root.mkdir()
    corpus_path = corpus_dir / "manifest.json"
    source_copy = corpus_dir / "source.json"
    source_path = input_root / "source.srt"
    snapshots_path = tmp_path / "source_snapshots.json"
    approval_path = tmp_path / "approval.json"
    profile_path = tmp_path / "profile.json"
    corpus_path.write_text("{}", encoding="utf-8")
    source_copy.write_text("[]", encoding="utf-8")
    source_path.write_text("1\n00:00:00,000 --> 00:00:01,000\ntext\n", encoding="utf-8")
    snapshots_path.write_text("{}", encoding="utf-8")
    approval_path.write_text("{}", encoding="utf-8")
    profile_path.write_text("{}", encoding="utf-8")
    raw_hash = matrix._sha_file(source_path)
    parsed_hash = matrix.canonical_hash([])
    manifest.update(
        {
            "selection_sha256": matrix.canonical_hash(manifest["selection"]),
            "source": {
                "corpus_manifest": str(corpus_path),
                "corpus_manifest_sha256": matrix._sha_file(corpus_path),
                "input_root": str(input_root),
                "sources": {
                    "source-0": {
                        "raw_sha256": raw_hash,
                        "parsed_sha256": parsed_hash,
                        "metadata": {"path": "source.srt", "json_path": "source.json"},
                    }
                },
                "source_snapshots_sha256": matrix.canonical_hash({}),
            },
            "approval": {"path": str(approval_path), "sha256": matrix._sha_file(approval_path)},
            "profile": {"path": str(profile_path), "sha256": matrix._sha_file(profile_path)},
            "limits": {
                "direct_api_cap_usd": str(matrix.DIRECT_CAP),
                "subscription_conservative_attempt_units": matrix.SUBSCRIPTION_CAP_UNITS,
                "total_attempt_units": matrix.TOTAL_CAP_UNITS,
                "max_tokens": matrix.MAX_TOKENS,
                "payload_bytes": matrix.PAYLOAD_LIMIT,
            },
            "prompts": {
                "version": matrix.PROMPT_VERSION,
                "generator_system": matrix.GENERATOR_SYSTEM_PROMPT,
                "arbiter_system": matrix.ARBITER_SYSTEM_PROMPT,
                "generation_schema": matrix.generation_schema(),
                "review_schema": matrix.review_schema(),
            },
            "user_consent": {
                "no_byok_confirmed": True,
                "muse_data_approved": True,
                "placeholder": False,
                "approved_source": "unit fixture",
                "approved_scope": "SH08 matrix",
            },
            "code_hashes": matrix._code_hashes(),
            "opencode_required_version": matrix.OPENCODE_REQUIRED_VERSION,
        }
    )
    manifest["checksum"] = matrix.canonical_hash({key: value for key, value in manifest.items() if key != "checksum"})
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output_dir / "selection.json").write_text(json.dumps(manifest["selection"]), encoding="utf-8")
    (output_dir / "source_snapshots.json").write_text("{}", encoding="utf-8")
    matrix._save_ledger(
        output_dir,
        {
            "schema": "sh08_matrix_ledger.v1",
            "manifest_checksum": manifest["checksum"],
            "requests": {},
        },
    )
    return output_dir, manifest


@pytest.mark.parametrize("tamper", ["profile", "selection", "source", "code", "ledger"])
def test_tampered_frozen_inputs_fail_before_backend(
    unit_read_manifest: tuple[Path, dict[str, Any]],
    tamper: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir, manifest = unit_read_manifest
    calls: list[str] = []

    async def forbidden_backend(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("backend")
        raise AssertionError("backend must not start for tampered frozen input")

    if tamper == "profile":
        Path(manifest["profile"]["path"]).write_text("tampered", encoding="utf-8")
    elif tamper == "selection":
        selection = deepcopy(manifest["selection"])
        selection["cases"][0]["text"] = "tampered"
        (output_dir / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    elif tamper == "source":
        source_path = Path(manifest["source"]["input_root"]) / "source.srt"
        source_path.write_text("tampered", encoding="utf-8")
    elif tamper == "code":
        original_sha_file = matrix._sha_file

        def changed_code_hash(path: Path) -> str:
            digest = original_sha_file(path)
            return "0" * 64 if path.name == "sh08_matrix.py" else digest

        monkeypatch.setattr(matrix, "_sha_file", changed_code_hash)
    else:
        ledger = {
            "schema": "sh08_matrix_ledger.v1",
            "manifest_checksum": manifest["checksum"],
            "requests": {},
        }
        descriptor = _descriptor(manifest)
        key = matrix.canonical_hash(descriptor)
        row = _complete_row(manifest, "deepseek", 0, cost=0.0)[1]
        row["descriptor"] = descriptor
        ledger["requests"] = {key: row}
        ledger["requests"][key]["key"] = key
        ledger["requests"][key]["descriptor"]["model_key"] = "luna"
        ledger["checksum"] = matrix.canonical_hash(ledger)
        (output_dir / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(matrix.MatrixError):
        asyncio.run(matrix.run_matrix(output_dir, forbidden_backend))
    assert calls == []


def test_full_replay_orders_first_passes_repairs_and_report_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    for case in manifest["selection"]["cases"]:
        case["max_chars"] = 0
    for batch in manifest["batches"]:
        for case_id in batch["case_ids"][:2]:
            next(case for case in manifest["selection"]["cases"] if case["case_id"] == case_id)["max_chars"] = 30
    manifest["selection"]["cases"][4]["status"] = "invalid"
    cannot_case = manifest["batches"][0]["case_ids"][1]
    output_dir = _new_run(tmp_path, manifest)
    monkeypatch.setattr(matrix, "_read_manifest", lambda _: manifest)
    events: list[tuple[str, str, list[str]]] = []
    records: list[tuple[str, list[str], dict[str, Any]]] = []
    review_count = 0
    batch_first = {batch["batch_id"]: batch["case_ids"][0] for batch in manifest["batches"]}
    case_to_batch = {
        case_id: batch["batch_id"]
        for batch in manifest["batches"]
        for case_id in batch["case_ids"]
    }
    first_pass_review_count = 0
    candidate_review_pairs = [pair for pair in matrix.PAIR_ORDER if pair[0] in {"luna", "deepseek"}]

    async def fake_backend(model: str, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        nonlocal review_count, first_pass_review_count
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        targets = payload["targets"]
        ids = [str(target["cue_id"]) for target in targets]
        if schema == matrix.generation_schema():
            repair = "prior_candidate" in targets[0]
            events.append(("repair" if repair else "generation", model, ids))
            records.append((model, ids, deepcopy(schema)))
            results = []
            for target in targets:
                case_id = str(target["cue_id"])
                if not repair and (model in {"gemini", "muse"} or case_id == cannot_case):
                    results.append({"cue_id": case_id, "text": target["original"], "status": "cannot_shorten"})
                else:
                    results.append({"cue_id": case_id, "text": "fixed" if repair else "short", "status": "ok"})
            text = json.dumps({"results": results})
        else:
            first_pass = first_pass_review_count < 36
            repetition = 1 if first_pass_review_count < 18 else 2
            pair = candidate_review_pairs[review_count % len(candidate_review_pairs)] if first_pass else ("recheck", "recheck")
            review_count += 1
            if first_pass:
                first_pass_review_count += 1
            events.append(("review", model, ids))
            records.append((model, ids, deepcopy(schema)))
            results = []
            for target in targets:
                case_id = str(target["cue_id"])
                revise = first_pass and pair in {("luna", "deepseek"), ("deepseek", "luna")} and case_id == batch_first[case_to_batch[case_id]]
                results.append({"cue_id": case_id, "verdict": "revise" if revise else "accept", "issues": [], "advice": ""})
            text = json.dumps({"results": results})
        spec = matrix.MODEL_SPECS[model]
        return _response(model, cost=0.0, physical=None if spec["subscription"] else 1) | {"text": text, "raw": text}

    result = asyncio.run(matrix.run_matrix(output_dir, fake_backend))
    assert result["status"] == "complete"
    assert first_pass_review_count == 36
    assert len(records) == len(events)
    assert all(isinstance(schema, dict) for _, _, schema in records)
    first_repair = next(index for index, event in enumerate(events) if event[0] == "repair")
    assert all(event[0] != "repair" for event in events[:first_repair])
    assert all(event[0] != "review" or index < first_repair for index, event in enumerate(events) if index < first_repair)

    case_body = json.loads((output_dir / "case_results.json").read_text(encoding="utf-8"))
    rows = case_body["rows"]
    assert all(len(row["branches"]) == 24 for row in rows)
    assert all(
        len({(branch.get("rep"), branch.get("stage"), branch.get("generator"), branch.get("arbiter")) for branch in row["branches"]}) == 24
        for row in rows
    )
    for case_id in (manifest["batches"][0]["case_ids"][0], manifest["batches"][1]["case_ids"][0], manifest["batches"][2]["case_ids"][0]):
        for generator in ("luna", "deepseek"):
            for repetition in (1, 2):
                branches = [
                    branch
                    for row in rows
                    if row["case_id"] == case_id
                    for branch in row["branches"]
                    if branch.get("stage") == "review" and branch.get("generator") == generator and branch.get("rep") == repetition
                ]
                assert len(branches) == 3
                assert len({branch["candidate_hash"] for branch in branches}) == 1
                assert sum(branch["verdict"] == "revise" for branch in branches) == 1
                assert all(branch.get("repair", {}).get("recheck", {}).get("accepted") is True for branch in branches if branch["verdict"] == "revise")
    revisions = [
        branch
        for row in rows
        for branch in row["branches"]
        if branch.get("stage") == "review" and branch.get("verdict") == "revise"
    ]
    assert len(revisions) == 12
    assert all(
        (branch["generator"], branch["arbiter"]) in {("luna", "deepseek"), ("deepseek", "luna")}
        and branch["case_id"] == batch_first[branch["batch"]]
        for branch in revisions
    )
    report = result
    assert report["eligible_denominator"] == 6
    assert report["zero_count"] == 53 and report["invalid_count"] == 1
    assert all(counts["denominator"] == 60 for counts in report["firstpass_counts"].values())
    assert any(counts["cannot_shorten"] > 0 for counts in report["firstpass_counts"].values())

    ledger = matrix._load_ledger(output_dir, manifest=manifest)
    descriptors = [row["descriptor"] for row in ledger["requests"].values()]
    assert len(descriptors) == len({matrix.canonical_hash(descriptor) for descriptor in descriptors})
    repair_descriptors = [descriptor for descriptor in descriptors if descriptor.get("stage") == "repair"]
    assert repair_descriptors and all(descriptor["generator"] == descriptor["model_key"] for descriptor in repair_descriptors)
    assert {descriptor["rep"] for descriptor in descriptors} == {1, 2}

    call_count = len(events)
    (output_dir / "case_results.json").write_text("corrupted", encoding="utf-8")
    reported = matrix.report_matrix(output_dir)
    assert reported["firstpass_counts"] == result["firstpass_counts"]
    (output_dir / "case_results.json").unlink()
    reported_after_delete = matrix.report_matrix(output_dir)
    assert reported_after_delete["firstpass_counts"] == result["firstpass_counts"]
    assert len(events) == call_count


def test_report_keeps_invalid_zero_out_of_zero_count() -> None:
    manifest = _manifest()
    manifest["selection"]["cases"][0].update({"max_chars": 0, "status": "invalid"})
    manifest["selection"]["cases"][1]["max_chars"] = 0

    report = matrix._report(manifest, {"requests": {}}, [], status="complete")

    assert report["eligible_denominator"] == 58
    assert report["zero_count"] == 1
    assert report["invalid_count"] == 1
    assert all(
        counts["denominator"] == 60
        and counts["zero"] == 1
        and counts["invalid"] == 1
        and counts["zero"] + counts["invalid"] + counts["missing"] == 60
        for counts in report["firstpass_counts"].values()
    )


def test_cli_validates_manifest_before_preflight_and_always_closes_context(
    unit_read_manifest: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir, _ = unit_read_manifest
    events: list[str] = []

    class FakeBackend:
        def __init__(self, **kwargs: Any) -> None:
            events.append("construct")

        async def __aenter__(self) -> "FakeBackend":
            events.append("enter")
            return self

        async def __aexit__(self, *args: Any) -> None:
            events.append("exit")

        def preflight(self, model_key: str | None = None) -> dict[str, Any]:
            events.append(f"preflight:{model_key}")
            return {"ok": True}

    real_read_manifest = matrix._read_manifest

    def traced_read_manifest(path: Path) -> dict[str, Any]:
        events.append("manifest")
        return real_read_manifest(path)

    async def fake_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("run")
        return {"status": "complete"}

    monkeypatch.setattr(matrix, "_read_manifest", traced_read_manifest)
    monkeypatch.setattr(matrix, "run_matrix", fake_run)
    monkeypatch.setattr(transport, "MatrixBackend", FakeBackend)
    monkeypatch.setattr("sys.argv", ["sh08_matrix", "run", "--output-dir", str(output_dir), "--allow-muse-data", "--openrouter-no-byok"])

    assert matrix._cli() == 0
    assert events[-1] == "exit"
    assert events.index("manifest") < events.index("preflight:deepseek")


def test_cli_invalid_manifest_skips_credentials_preflight_but_closes_context(
    unit_read_manifest: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir, manifest = unit_read_manifest
    Path(manifest["profile"]["path"]).write_text("tampered", encoding="utf-8")
    events: list[str] = []

    class FakeBackend:
        def __init__(self, **kwargs: Any) -> None:
            events.append("construct")

        async def __aenter__(self) -> "FakeBackend":
            events.append("enter")
            return self

        async def __aexit__(self, *args: Any) -> None:
            events.append("exit")

        def preflight(self, model_key: str | None = None) -> dict[str, Any]:
            events.append(f"preflight:{model_key}")
            raise AssertionError("credentials/backend preflight must not run")

    monkeypatch.setattr(transport, "MatrixBackend", FakeBackend)
    monkeypatch.setattr("sys.argv", ["sh08_matrix", "run", "--output-dir", str(output_dir), "--allow-muse-data", "--openrouter-no-byok"])

    assert matrix._cli() == 2
    assert "preflight:deepseek" not in events
    assert events[-1] == "exit"
