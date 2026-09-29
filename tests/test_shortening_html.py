from __future__ import annotations

import json
import asyncio
import sys
import types
import warnings
from html import escape
from pathlib import Path
from typing import Any, Mapping

import pytest

from utils import jev_followup as followup
from utils import jev_replay as jev
from utils.length_check import check_length
from utils.shortening_html import (
    HTML_MARKER,
    live_html_report,
    load_report_document,
    note_model_request,
    publish_html_report,
)
from utils.shortening_html_view import render_report
from utils.shortening_html import _analysis_notice, _artifact_files, _candidate, _case_status, _sha_bytes, _sha_json, _status_from_branch
from utils.llm_types import CallRecord, DetailedResponse
from utils.opencode_transport import OpenCodePromptUsage
from utils.shortening_budget import BudgetPolicy
from utils.shortening_plan import PlannerConfig, plan_subtitles
from utils.shortening_runner import ModelSpec, RunConfig, run_trial


def _renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("utils.shortening_html_view")

    def render_report(document: dict[str, object]) -> str:
        return HTML_MARKER + "\n" + json.dumps(document, ensure_ascii=False)

    module.render_report = render_report  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils.shortening_html_view", module)


def _failing_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("utils.shortening_html_view")

    def render_report(document: dict[str, object]) -> str:
        raise RuntimeError("renderer")

    module.render_report = render_report  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils.shortening_html_view", module)


def _trial(root: Path, *, fake: bool = False) -> None:
    (root / "report.json").write_text(json.dumps({
        "status": "partial",
        "model": {"generator": "saved-model", "variant": "low"},
        "records": [{
            "cue_id": "cue-2", "source_id": "source", "original": "Original <script>",
            "start": 100, "end": 200, "status": "accepted", "candidate": "Short",
            "final_text": "Short", "budget": {"max_chars": 20},
            "final_review": {"verdict": "accept", "local": {"within_length": True}, "advice": "Bearer secret"},
            "history": [{"stage": "generation", "text": "Short", "local": {"within_length": True}}],
        }],
        "metadata": {"synthetic_fixture": True} if fake else {"mode": "live"},
    }), encoding="utf-8")


def test_loader_keeps_saved_model_and_redacts_error(tmp_path: Path) -> None:
    _trial(tmp_path)
    document = load_report_document(tmp_path)
    row = document["rows"][0]
    assert row["generator"] == "saved-model"
    assert row["generator_mode"] == "low"
    assert row["accepted"] is True
    assert "Bearer secret" not in json.dumps(document)


def test_fake_report_is_rejected(tmp_path: Path) -> None:
    _trial(tmp_path, fake=True)
    with pytest.raises(ValueError, match="fake"):
        load_report_document(tmp_path)


def test_primary_trial_report_shows_reported_costs_budget_stop_and_source_text(
    tmp_path: Path,
) -> None:
    from tests.test_shortening_budget import _snapshot

    snapshot = _snapshot()
    generator = ModelSpec("openrouter", "meta/muse-spark-1.3-contributor", "low")
    arbiter = ModelSpec("openrouter", "openai/gpt-5.6-sol", "medium")
    roles = {"generator": vars(generator), "arbiter": vars(arbiter)}
    metadata = {
        "roles": roles,
        "config": {"run": {"length_tools": False}},
        "openrouter": {"roles": roles, "max_output_tokens": 16_384, "pricing_snapshot": snapshot},
        "budget": {"limit_usd": "5", "max_requests": 1000},
    }
    policy = BudgetPolicy.from_snapshot("5", snapshot)
    items = [
        {"index": 0, "start": 0, "end": 1000, "text": "first original line that needs shortening"},
        {"index": 1, "start": 1000, "end": 2000, "text": "second original line that needs shortening"},
    ]
    planner = PlannerConfig(cps=5, max_speed=1.0, length_basis="normalized_tts_text")
    plans = plan_subtitles(items, planner)
    source = json.loads(json.dumps(items))
    calls = {"generation": 0, "review": 0, "repair": 0, "recheck": 0}

    async def fake_request(
        spec: ModelSpec, _: str, user: str, schema: dict[str, Any], __: bool,
    ) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        cue_ids = [row["cue_id"] for row in payload["targets"]]
        stage = "review" if "verdict" in json.dumps(schema) else "generation"
        if stage == "generation" and "prior_candidate" in user:
            stage = "repair"
        calls[stage] += 1
        if stage in {"generation", "repair"}:
            body = {"results": [{"cue_id": cue_id, "text": "ok", "status": "ok"} for cue_id in cue_ids]}
            cost = 0.1
        else:
            body = {"results": [{
                "cue_id": cue_id,
                "verdict": "accept" if cue_id == cue_ids[0] else "revise",
                "issues": [] if cue_id == cue_ids[0] else ["needs repair"],
                "advice": "",
            } for cue_id in cue_ids]}
            cost = 4.0
        usage = OpenCodePromptUsage(input_tokens=12, output_tokens=3, reasoning_tokens=2, cache_read_tokens=1, cost=cost)
        call = CallRecord(
            "openrouter", spec.model, spec.variant, "2026-09-20T00:00:00Z", 0.1,
            usage, "success", requested_model=spec.model, requested_variant=spec.variant,
            pricing_snapshot_id=snapshot["id"], generation_id=f"receipt-{stage}",
        )
        return DetailedResponse(json.dumps(body), [call])

    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=[plan.cue_id for plan in plans],
        planner_config=planner, run_config=RunConfig(), generator=generator,
        arbiter=arbiter, output_dir=tmp_path / "run", request_fn=fake_request,
        report_metadata=metadata, budget_policy=policy,
    ))
    assert calls == {"generation": 1, "review": 1, "repair": 1, "recheck": 0}
    assert items == source
    assert report["budget"]["status"] == "blocked_by_budget"
    assert report["budget"]["next_reservation_usd"] == "4.85576"
    document = load_report_document(tmp_path / "run")
    rendered = render_report(document)
    metadata_values = {item["label"]: item["value"] for item in document["metadata"]}
    assert "API reported USD" in metadata_values["Стоимость"]
    assert "$4.2" in metadata_values["Стоимость"]
    assert "$4" in metadata_values["Стоимость"]
    assert metadata_values["Следующий резерв"] == "$4.85576"
    assert "reservation, not spend" in metadata_values["Следующий резерв"] or "reservation is not spend" in metadata_values["Стоимость"]
    assert "$0" in metadata_values["Неизвестный резерв"]
    assert metadata_values["Причина остановки"] == "reservation_exceeds_limit"
    assert "meta/muse-spark-1.3-contributor LOW" in metadata_values["Настройки"]
    assert "openai/gpt-5.6-sol MEDIUM" in metadata_values["Настройки"]
    assert "Частичный запуск" in rendered
    assert "quality claim" in rendered
    assert "Длительность финальной озвучки ещё не проверена" in rendered
    assert "pricing_snapshot_id" in rendered
    assert "Meta/OpenAI" in rendered
    assert "extra tools: no" in rendered
    rows = {row["case_id"]: row for row in document["rows"]}
    accepted = next(row for row in rows.values() if row["accepted"] is True)
    unresolved = next(row for row in rows.values() if row["accepted"] is False)
    assert accepted["accepted_text"] == "ok"
    assert unresolved["accepted_text"] is None
    assert unresolved["original"] == next(item["text"] for item in source if item["index"] == 1)


def test_observer_publishes_only_after_model_request_and_resets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _renderer(monkeypatch)
    _trial(tmp_path)
    assert publish_html_report(tmp_path) is None
    with live_html_report(tmp_path):
        note_model_request()
    html_path = tmp_path / "shortening.html"
    assert html_path.exists()
    assert HTML_MARKER in html_path.read_text(encoding="utf-8")
    assert publish_html_report(tmp_path) is not None


def test_observer_preserves_primary_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _renderer(monkeypatch)
    _trial(tmp_path)
    with pytest.raises(RuntimeError, match="primary"):
        with live_html_report(tmp_path):
            note_model_request()
            raise RuntimeError("primary")


def test_observer_renderer_warning_does_not_mask_primary_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _failing_renderer(monkeypatch)
    _trial(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeError, match="primary"):
            with live_html_report(tmp_path):
                note_model_request()
                raise RuntimeError("primary")


def test_observer_renderer_failure_does_not_raise_on_successful_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _failing_renderer(monkeypatch)
    _trial(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with live_html_report(tmp_path):
            note_model_request()
    assert not (tmp_path / "shortening.html").exists()


def test_foreign_html_is_not_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _renderer(monkeypatch)
    _trial(tmp_path)
    (tmp_path / "shortening.html").write_text("foreign", encoding="utf-8")
    with pytest.raises(FileExistsError):
        publish_html_report(tmp_path, confirmed_real_run=True)
    assert (tmp_path / "shortening.html").read_text(encoding="utf-8") == "foreign"


def test_sh08_repair_acceptance_and_exhaustive_missing_row(tmp_path: Path) -> None:
    manifest = {
        "schema": "sh08_matrix_manifest.v1",
        "approved_scope": {"repetitions": 1, "pairs": [["luna", "deepseek"]]},
        "selection": {"cases": [{"case_id": "source:cue-0", "source_id": "source", "position": 0, "original": "Original", "start": 0, "end": 1000, "max_chars": 12, "context": []}]},
        "full_specs": {"luna": {"variant": "medium"}, "deepseek": {"variant": "off"}},
        "status": "partial",
    }
    branch = {"case_id": "source:cue-0", "rep": 1, "stage": "review", "generator": "luna", "arbiter": "deepseek", "status": "complete", "candidate": "first", "accepted": False, "verdict": "revise", "local_fit": {"fits": True}, "repair": {"text": "final", "local_fit": {"fits": True}, "recheck": {"accepted": True, "verdict": "accept"}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "case_results.json").write_text(json.dumps([{ "case_id": "source:cue-0", "branches": [branch] }]), encoding="utf-8")
    row = load_report_document(tmp_path)["rows"][0]
    assert row["accepted_text"] == "final"
    assert row["initial_verdict"] == "revise"


def test_prepared_sh08_uses_report_metadata_and_canonical_ledger_binding(tmp_path: Path) -> None:
    manifest = {
        "schema": "sh08_matrix_manifest.v1", "status": "prepared", "run_id": "run",
        "approved_scope": {"repetitions": 1, "pairs": [["luna", "deepseek"]]},
        "selection": {"cases": [{"case_id": "s:c", "source_id": "s", "position": 0, "original": "old", "max_chars": 5}]},
        "full_specs": {"luna": {"model": "saved/luna", "variant": "medium"}, "deepseek": {"model": "saved/judge", "variant": "low"}},
    }
    manifest["checksum"] = _sha_json(manifest)
    ledger = {"schema": "ledger", "requests": {"r": {"status": "complete"}}}
    ledger["checksum"] = _sha_json(ledger)
    case_body = {"schema": "case", "manifest_checksum": manifest["checksum"], "ledger_checksum": _sha_json({key: value for key, value in ledger.items() if key != "checksum"}), "rows": [{"case_id": "s:c", "branches": []}]}
    case_body["checksum"] = _sha_json(case_body)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    (tmp_path / "case_results.json").write_text(json.dumps(case_body), encoding="utf-8")
    (tmp_path / "report.json").write_text(json.dumps({"status": "blocked_unknown", "costs": {"known_sum_usd": 0.1013308035, "subquota_count": 300, "logical_count": 89, "subscription_unpriced": True}, "records": []}), encoding="utf-8")
    document = load_report_document(tmp_path)
    assert document["status"] == "blocked_unknown"
    assert document["rows"][0]["generator"] == "saved/luna"
    metadata = {item["label"]: item["value"] for item in document["metadata"]}
    assert "prepared" not in json.dumps(metadata)
    assert "0.1013308035" in metadata["Стоимость"]
    assert metadata["subquota"].startswith("subquota_count: 300")
    assert metadata["logical"].startswith("logical_count: 89")
    assert "Стоимость подписки отдельно не оценена" in metadata["Стоимость"]


@pytest.mark.parametrize(("case", "expected"), [({"status": "invalid", "max_chars": 0}, "invalid"), ({"max_chars": 0}, "zero")])
def test_case_status_prioritizes_invalid_over_zero(case: dict[str, object], expected: str) -> None:
    assert _case_status(case) == expected


def test_sh08_branch_statuses_preserve_invalid_zero_and_cannot_shorten() -> None:
    assert _status_from_branch({"parse_error": True}) == "error"


def test_sh08_real_shape_maps_invalid_zero_and_generation_cannot_shorten(tmp_path: Path) -> None:
    manifest = {
        "schema": "sh08_matrix_manifest.v1",
        "approved_scope": {"repetitions": 1, "pairs": [["luna", "deepseek"]]},
        "selection": {"cases": [
            {"case_id": "s:invalid", "source_id": "s", "position": 0, "original": "invalid", "max_chars": 0, "status": "invalid"},
            {"case_id": "s:zero", "source_id": "s", "position": 1, "original": "zero", "max_chars": 0, "status": "valid"},
            {"case_id": "s:cannot", "source_id": "s", "position": 2, "original": "cannot", "max_chars": 5},
        ]},
        "full_specs": {"luna": {"variant": "medium"}, "deepseek": {"variant": "off"}},
    }
    review = lambda case_id: {"case_id": case_id, "rep": 1, "stage": "review", "generator": "luna", "arbiter": "deepseek", "status": "unresolved", "reason": "candidate_absent"}
    results = [
        {"case_id": "s:invalid", "branches": [review("s:invalid")]},
        {"case_id": "s:zero", "branches": [review("s:zero")]},
        {"case_id": "s:cannot", "branches": [
            {"case_id": "s:cannot", "rep": 1, "stage": "generation", "generator": "luna", "reason": "cannot_shorten"},
            review("s:cannot"),
        ]},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "case_results.json").write_text(json.dumps(results), encoding="utf-8")
    rows = {row["case_id"]: row for row in load_report_document(tmp_path)["rows"]}
    assert {case_id: rows[case_id]["status"] for case_id in rows} == {
        "s:invalid": "invalid", "s:zero": "zero", "s:cannot": "cannot_shorten",
    }


def test_analysis_notice_uses_verified_termination_statuses(tmp_path: Path) -> None:
    artifacts = {
        "manifest.json": {"schema": "sh08_matrix_manifest.v1"},
        "ledger.json": {"schema": "ledger"},
        "case_results.json": [],
        "report.json": {"status": "blocked_unknown", "records": []},
    }
    for name, value in artifacts.items():
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    inputs = {name.removesuffix(".json"): _sha_bytes((tmp_path / name).read_bytes()) for name in artifacts}
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "report.json").write_text(json.dumps({
        "status": "offline_evidence_analysis",
        "termination": {"raw_execution_status": "blocked_unknown", "derived_offline_status": "budget_exhausted"},
        "provenance": {"inputs": {f"{name}_sha256": digest for name, digest in inputs.items()}},
    }), encoding="utf-8")
    notice = _analysis_notice(tmp_path)
    assert "Анализ: исходный статус blocked_unknown; офлайн-вывод budget_exhausted" in notice
    assert "Достигнут лимит подписки. В исходном отчёте статус ошибочно обозначен как blocked_unknown; сохранённые результаты не изменялись." in notice
    assert "offline_evidence_analysis" not in notice


def test_trial_never_invents_final_candidate_and_maps_manual(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text(json.dumps({
        "run_state": "complete", "records": [{
            "cue_id": "c", "original": "original", "status": "manual", "final_text": "invented",
            "budget": {"max_chars": 8}, "history": [{"stage": "review", "text": "not-a-proposal", "verdict": "accept"}],
        }],
    }), encoding="utf-8")
    row = load_report_document(tmp_path)["rows"][0]
    assert row["status"] == "unresolved"
    assert row["accepted"] is False
    assert row["candidates"] == []


def test_abc_keeps_three_arms_duplicate_text_ids_and_saved_modes(tmp_path: Path) -> None:
    manifest = {
        "models": {"generator": {"model": "writer/saved", "variant": "high"}, "selector": {"model": "selector/saved", "variant": "medium"}},
        "cases": [{"case_id": "s:c", "source_id": "s", "original": "original", "max_chars": 10}],
    }
    candidates = [{"candidate_id": f"canonical-{index}", "text": "same", "local": {"valid": True, "within_length": True, "length": 4}} for index in range(5)]
    results = [{"global_case_id": "s:c", "source_id": "s", "repetition": 1, "original": "original", "budget": {"max_chars": 10},
                "context": [{"text": "normal"}], "A": {"text": "a", "status": "unreviewed"},
                "B": {"text": "b", "status": "accepted"}, "C": {"status": "accepted", "selected_candidate_id": "canonical-3", "selected_text": "same", "candidates": candidates,
                "provenance": {"generator": {"model": "replay/writer", "variant": "low"}}}}]
    (tmp_path / "experiment_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "case_results.json").write_text(json.dumps(results), encoding="utf-8")
    rows = load_report_document(tmp_path)["rows"]
    assert [row["arm"] for row in rows] == ["A", "B", "C"]
    c_row = rows[2]
    assert len(c_row["candidates"]) == 5
    assert [candidate["selected"] for candidate in c_row["candidates"]].count(True) == 1
    assert c_row["accepted_text"] == "same"
    assert rows[0]["generator"] == "writer/saved"
    assert c_row["arbiter"] == "selector/saved"


def test_artifact_hashes_ignore_audio_environment_and_foreign_symlinks(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "source.wav").write_bytes(b"audio")
    (tmp_path / ".env").write_text("TOKEN=secret", encoding="utf-8")
    (tmp_path / "calls.jsonl").write_text("secret", encoding="utf-8")
    assert set(_artifact_files(tmp_path)) == {"manifest.json"}


def test_candidate_length_comes_from_saved_local_measurement() -> None:
    candidate = _candidate("c", "Не люблю, когда мне указывают.", local={"length": 30, "max_chars": 23, "fits": False}, status="candidate")
    assert candidate["length"] == 30
    assert candidate["fits"] is False


def test_legacy_children_keep_saved_model_and_summary_status(tmp_path: Path) -> None:
    child = tmp_path / "r2" / "standard" / "source"
    child.mkdir(parents=True)
    (tmp_path / "experiment_manifest.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    (tmp_path / "experiment_summary.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    (child / "report.json").write_text(json.dumps({"run_state": "complete", "model": {"generator": {"model": "saved/legacy", "variant": "low"}}, "records": [{"cue_id": "c", "original": "old", "status": "manual", "history": []}]}), encoding="utf-8")
    document = load_report_document(tmp_path)
    assert document["status"] == "complete"
    assert document["rows"][0]["generator"] == "saved/legacy"


def _jev_report(*, status: str = "partial", selected_id: str = "c1", safety: str = "accept") -> dict[str, object]:
    return {
        "schema": "jev_replay.v1",
        "run_id": "jev-run",
        "status": status,
        "model": "jev-saved-model",
        "mode": "Choice: selection + per-candidate safety",
        "backend_consumed": True,
        "summary": {"records": 48, "attempted_requests": 1, "completed_requests": 1,
                     "unknown_reserved_usd": "0", "cases": 1, "cost_usd_estimated": "0.02",
                     "usage": {"input_tokens": 123}},
        "records": [{
            "case_id": "source:cue-1", "source": "source", "order": 1, "original": "Original <script>",
            "context": [{"text": "Context & neighbor", "start": 1, "end": 2}], "max_chars": 20,
            "candidates": [
                {"id": "c1", "text": "Short <b>", "actual_chars": 9, "fits": True,
                 "provenance": [{"generator_model": "muse/saved", "generator_mode": "low",
                                 "source_run": "low", "rep": 1, "stage": "generation",
                                 "origin_receipt_id": "origin-1",
                                 "historic_sol": [{"verdict": "accept", "issues": [], "advice": "Sol <note>",
                                                    "review_receipt": "review-1"}]}]},
                {"id": "c2", "text": "Unsafe", "actual_chars": 6, "fits": False,
                  "provenance": [{"generator_model": "muse/saved", "generator_mode": "high",
                                  "source_run": "high", "rep": 1, "stage": "generation",
                                  "origin_receipt_id": "origin-2", "historic_sol": []}]},
            ],
            "answers": {
                "best": {"choice": selected_id, "probabilities": {"c1": 0.8, "c2": 0.2}, "confidence": 0.8},
                "safety_c1": {"choice": safety, "probabilities": {"accept": 0.7, "reject": 0.3}, "confidence": 0.7},
            },
            "selected_candidate_id": selected_id, "accepted_text": "Short <b>",
            "status": "accepted", "request_id": "req-1", "receipt_id": "receipt-1",
        }],
    }


def test_jev_prepared_artifact_is_rejected(tmp_path: Path) -> None:
    report = _jev_report(status="prepared")
    report["backend_consumed"] = False
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="prepared|offline"):
        load_report_document(tmp_path)


@pytest.mark.parametrize("status", ["partial", "complete"])
def test_jev_actual_report_maps_acceptance_and_summary(tmp_path: Path, status: str) -> None:
    (tmp_path / "report.json").write_text(json.dumps(_jev_report(status=status)), encoding="utf-8")
    document = load_report_document(tmp_path)
    row = document["rows"][0]
    assert document["status"] == status
    assert row["accepted"] is True
    assert row["accepted_text"] == "Short <b>"
    assert row["generator"] == "muse/saved"
    assert row["generator_mode"] == "low/high"
    assert row["arbiter"] == "jev-saved-model"
    assert row["arbiter_mode"] == "Choice"
    assert row["repetition"] == 1
    assert all(candidate["phase"] == "alternative" for candidate in row["candidates"])
    assert {item["label"] for item in row["details"]} >= {"Jev best choice", "Jev best probabilities", "Jev best confidence"}
    metadata = {item["label"]: item["value"] for item in document["metadata"]}
    assert metadata["logical"] == "1 / 48"
    assert metadata["Кейсы"] == "1"
    assert metadata["Настройки"] == "input tokens: 123"
    assert metadata["Стоимость"] == "estimated $0.02"


def test_jev_probability_and_historic_notes_are_escaped_and_not_advice(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text(json.dumps(_jev_report()), encoding="utf-8")
    document = load_report_document(tmp_path)
    row = document["rows"][0]
    assert row["issues"] == []
    assert row["advice"] == ""
    assert any(item["label"] == "Исторический Sol" and "origin-1" in item["value"] for item in row["details"])
    from utils.shortening_html_view import render_report
    rendered = render_report(document)
    assert "Short &lt;b&gt;" in rendered
    assert "Sol &lt;note&gt;" in rendered
    assert "Jev best probabilities" in rendered
    assert "Совет" not in rendered


@pytest.mark.parametrize("selected_id,safety", [("unknown", "accept"), ("c2", "reject")])
def test_jev_unknown_or_unsafe_selection_has_no_accepted_text(tmp_path: Path, selected_id: str, safety: str) -> None:
    (tmp_path / "report.json").write_text(json.dumps(_jev_report(selected_id=selected_id, safety=safety)), encoding="utf-8")
    row = load_report_document(tmp_path)["rows"][0]
    assert row["accepted"] is False
    assert row["accepted_text"] is None


def test_jev_mismatching_best_or_missing_safety_is_unresolved(tmp_path: Path) -> None:
    report = _jev_report()
    record = report["records"][0]
    assert isinstance(record, dict)
    record["answers"]["best"]["choice"] = "c2"
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    row = load_report_document(tmp_path)["rows"][0]
    assert row["status"] == "unresolved"
    assert row["accepted"] is False
    assert row["accepted_text"] is None


def _write_jev_source_run(root: Path, mode: str) -> None:
    root.mkdir()
    raw = root / "source.srt"
    raw.write_text("historical source", encoding="utf-8")
    source_rows = [{"text": f"Original {index}", "start": index, "end": index + 1, "index": index} for index in range(24)]
    snapshots = {"source": source_rows}
    cases = [{
        "case_id": f"source:cue-{index}", "source_id": "source", "position": index,
        "source_index": index, "original": row["text"], "start": row["start"], "end": row["end"],
        "max_chars": 80, "length_basis": "display_text",
    } for index, row in enumerate(source_rows)]
    ledger_requests: dict[str, object] = {}
    result_rows: list[dict[str, object]] = []
    for index, case in enumerate(cases):
        text = f"Short {mode} {index}"
        receipt_id = f"{mode}-receipt-{index}"
        ledger_requests[receipt_id] = {
            "key": receipt_id, "status": "complete", "attempts": 1,
            "descriptor": {"spec": {"model": f"muse/{mode}", "variant": mode}},
            "response": {"text": text}, "usage": {"input_tokens": 10, "output_tokens": 2},
            "estimated_cost_usd": "0.00000042",
        }
        fit = check_length(text, 80, "display_text")
        branches = [
            {"stage": "generation", "status": "candidate", "text": text,
             "candidate_hash": jev._candidate_hash(text), "origin": receipt_id,
             "local_fit": fit, "rep": 1},
            {"stage": "review", "text": text, "candidate_hash": jev._candidate_hash(text),
             "verdict": "accept", "issues": [], "advice": "", "review_receipt_id": f"{mode}-review-{index}", "rep": 1},
        ]
        result_rows.append({**case, "branches": branches})

    selection = {"cases": cases}
    source = {"input_root": str(root), "sources": {"source": {
        "metadata": {"path": raw.name}, "raw_sha256": _sha_bytes(raw.read_bytes()),
    }}}
    manifest = {
        "run_id": f"historical-{mode}", "selection_sha256": _sha_json(selection),
        "source": {**source, "source_snapshots_sha256": _sha_json(snapshots)},
        "full_specs": {"muse": {"model": f"muse/{mode}", "variant": mode}}, "muse_effort": mode,
    }
    ledger = {"schema": jev.SCHEMA, "run_id": manifest["run_id"], "requests": ledger_requests}
    ledger["checksum"] = _sha_json({key: value for key, value in ledger.items() if key != "checksum"})
    case_body = {"rows": result_rows, "ledger_checksum": ledger["checksum"]}
    case_body["checksum"] = _sha_json(case_body)
    manifest["checksum"] = _sha_json(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    (root / "source_snapshots.json").write_text(json.dumps(snapshots), encoding="utf-8")
    (root / "case_results.json").write_text(json.dumps(case_body), encoding="utf-8")
    (root / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")


def test_jev_production_shape_to_html_without_automatic_fake_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    low_dir, high_dir = tmp_path / "low", tmp_path / "high"
    _write_jev_source_run(low_dir, "low")
    _write_jev_source_run(high_dir, "high")
    output_dir = tmp_path / "jev"
    jev.prepare(output_dir, low_dir=low_dir, high_dir=high_dir)

    async def fake_post(descriptor: dict[str, object], _: str) -> tuple[dict[str, object], dict[str, object]]:
        questions = descriptor["payload"]["questions"]
        candidate_ids = [key for key in questions["best"]["criteria"] if key.startswith("c")]
        best = candidate_ids[0]
        answers: dict[str, object] = {}
        for question_id, question in questions.items():
            options = list(question["criteria"])
            choice = best if question_id == "best" else "accept" if question_id == f"safety_{best}" else options[0]
            answers[question_id] = {"type": "choice", "choice": choice,
                                    "probabilities": {option: 1 / len(options) for option in options}, "confidence": 0.5}
        response = {"model": jev.MODEL, "answers": answers, "usage": {"input_tokens": 100, "output_tokens": 20}}
        return response, {"status_code": 200, "request_id": "fake-request", "json": response, "response": {"text": "ok"}}

    monkeypatch.setattr(jev, "_key", lambda: "fake-key")
    monkeypatch.setattr(jev, "_post", fake_post)
    report = jev.run(output_dir, max_new_requests=1)
    assert not (output_dir / "shortening.html").exists()
    report = jev.refresh_report(output_dir)
    assert report["summary"]["records"] == 48
    assert report["summary"]["attempted_requests"] == 1
    assert report["summary"]["completed_requests"] == 1
    assert report["summary"]["usage"]["input_tokens"] == 100
    document = load_report_document(output_dir)
    accepted = next(row for row in document["rows"] if row["status"] == "accepted")
    assert accepted["generator"] == "muse/low/muse/high"
    assert accepted["generator_mode"] == "low/high"
    assert any(item["label"] == "Исторический Sol" for item in accepted["details"])
    assert any(item["label"] == "Jev best probabilities" for item in accepted["details"])
    assert {item["label"]: item["value"] for item in document["metadata"]}["logical"] == "1 / 48"
    assert "100" in {item["label"]: item["value"] for item in document["metadata"]}["Настройки"]
    assert "<table" in render_report(document)


def _fake_jev_response(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    questions = descriptor["payload"]["questions"]
    record = descriptor["record"]
    candidates = record["candidates"]
    best = next(candidate["id"] for candidate in candidates if candidate.get("fits") is True)
    answers: dict[str, dict[str, Any]] = {}
    for question_id, question in questions.items():
        options = list(question["criteria"])
        if question_id == "best":
            choice = best
        elif question_id == f"safety_{best}":
            choice = "accept"
        else:
            choice = "reject" if "reject" in options else options[0]
        if len(options) == 1:
            probabilities = {options[0]: 1.0}
        else:
            probabilities = {option: 0.2 / (len(options) - 1) for option in options}
            probabilities[choice] = 0.8
        answers[question_id] = {
            "type": "choice", "choice": choice, "probabilities": probabilities, "confidence": 0.8,
        }
    return {"model": jev.MODEL, "answers": answers, "usage": {"input_tokens": 100, "output_tokens": 20}}


def _populate_followup_ledger(output_dir: Path) -> dict[str, Any]:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    ledger = json.loads((output_dir / "ledger.json").read_text(encoding="utf-8"))
    selected: dict[str, str] = {}
    for descriptor in manifest["requests"]:
        request_kind = descriptor["request_kind"]
        if request_kind not in selected:
            selected[request_kind] = followup._request_key(descriptor)

    for request_kind in ("repeat", "isolated_safety"):
        key = selected[request_kind]
        row = ledger["requests"][key]
        descriptor = row["descriptor"]
        raw = _fake_jev_response(descriptor)
        receipt = {
            "status_code": 200, "request_id": f"fake-{request_kind}", "json": raw,
        }
        row.update({
            "status": "complete", "attempts": 1, "usage": {"input_tokens": 100, "output_tokens": 20},
            "cost_usd": followup._money(followup.Decimal("0.0000042")),
            "response": receipt, "receipt_id": followup._receipt_id(receipt),
            "answers": jev.parse_jev_response(raw, descriptor["payload"]["questions"]),
        })

    review_key = selected["independent_review"]
    review_row = ledger["requests"][review_key]
    review_descriptor = review_row["descriptor"]
    advice = 'DeepSeek <script>alert("x")</script> & advice'
    review = {"cue_id": review_descriptor["target"]["cue_id"], "verdict": "accept", "issues": [], "advice": advice}
    input_tokens, output_tokens = 100, 20
    deepseek_cost = followup._money(
        (followup.Decimal(input_tokens) * followup.DEEPSEEK_INPUT_RATE
         + followup.Decimal(output_tokens) * followup.DEEPSEEK_OUTPUT_RATE) / followup.Decimal(1_000_000)
    )
    review_receipt = {
        "status_code": 200, "request_id": "fake-independent-review",
        "text": json.dumps({"results": [review]}, ensure_ascii=False),
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "cost_known": True, "cost_usd": deepseek_cost,
    }
    review_row.update({
        "status": "complete", "attempts": 1, "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "cost_usd": deepseek_cost, "cost_kind": "provider_usage", "response": review_receipt,
        "receipt_id": followup._receipt_id(review_receipt),
    })
    review_row["review"] = followup.parse_review(review_receipt["text"], [review["cue_id"]])

    error_key = next(key for key in ledger["requests"] if key not in selected.values() and ledger["requests"][key]["provider"] == "deepseek")
    error_row = ledger["requests"][error_key]
    error_row.update({"status": "error", "attempts": 0, "error": "synthetic durable error"})
    ledger["campaign_status"] = "partial"
    ledger["campaign_error"] = "synthetic campaign error"
    followup._save_ledger(output_dir, ledger)
    return followup.refresh_report(output_dir)


def test_jev_followup_producer_to_html_contract_uses_real_source_and_no_selection_invented(
    tmp_path: Path,
) -> None:
    source = Path("output/jev_replay_20260919")
    if not (source / "manifest.json").is_file() or not (source / "ledger.json").is_file():
        pytest.skip("saved Jev replay source is not present")

    output_dir = tmp_path / "followup"
    prepared = followup.prepare(source, output_dir)
    assert prepared["summary"]["records"] == 58
    assert not (output_dir / "shortening.html").exists()
    with pytest.raises(ValueError, match="prepared|zero-attempt"):
        load_report_document(output_dir)

    report = _populate_followup_ledger(output_dir)
    assert report["status"] == "partial"
    assert report["campaign_status"] == "partial"
    assert report["campaign_error"] == "synthetic campaign error"
    assert report["summary"]["records"] == 58
    assert report["summary"]["attempted_requests"] == 3
    assert report["summary"]["completed_requests"] == 3
    assert report["summary"]["errors"] == 1

    document = load_report_document(output_dir)
    assert not (output_dir / "shortening.html").exists()
    assert len({row["case_id"] for row in document["rows"]}) == 6
    metadata = {item["label"]: item["value"] for item in document["metadata"]}
    assert metadata["Запланировано"] == "58"
    assert metadata["Попытки"] == "3"
    assert metadata["Завершено"] == "3"
    assert metadata["Jev: оценочная стоимость"] == "0.0000084"
    assert metadata["Jev: неизвестный резерв"] == "0"
    assert metadata["DeepSeek: стоимость"] == "0.000054"
    assert metadata["DeepSeek: неизвестный резерв"] == followup._money(followup.DEEPSEEK_RESERVATION)

    for row in document["rows"]:
        assert "muse" in row["generator"].lower()
        assert row["generator_mode"] in {"low", "high", "low/high"}
        assert row["request_kind"] in {"repeat", "isolated_safety", "independent_review"}
        assert all(
            set(context) <= {"cue_id", "text", "start", "end", "marked_target", "is_target"}
            for context in row["context"]
        )
        if row["request_kind"] in {"repeat", "isolated_safety"}:
            assert row["arbiter"] == followup.JEV_MODEL
            assert row["arbiter_mode"] == "Choice"
        else:
            assert row["arbiter"] == followup.DEEPSEEK_MODEL
            assert row["arbiter_mode"] == followup.DEEPSEEK_MODE
        if row["status"] in {"pending", "error"}:
            assert row["accepted"] is False
            assert row["accepted_text"] is None

    records = report["records"]
    accepted_repeat = next(record for record in records if record["request_kind"] == "repeat" and record["status"] == "accepted")
    assert accepted_repeat["accepted_text"] is not None
    assert accepted_repeat["answers"]["best"]["choice"] == accepted_repeat["selected_candidate_id"]
    assert accepted_repeat["answers"][f"safety_{accepted_repeat['selected_candidate_id']}"]["choice"] == "accept"
    for record in records:
        if record["request_kind"] == "repeat" and record["accepted_text"] is not None:
            assert record["status"] == "accepted"

    evaluated = [row for row in document["rows"] if row["status"] == "evaluated"]
    assert {row["request_kind"] for row in evaluated} == {"isolated_safety", "independent_review"}
    assert all(row["decision_kind"] == "evaluation" and row["accepted_text"] is None for row in evaluated)
    repeat = next(row for row in document["rows"] if row["request_kind"] == "repeat")
    assert any(item["label"] == "Jev best probabilities" for item in repeat["details"])

    rendered = render_report(document)
    assert not (output_dir / "shortening.html").exists()
    assert escape('DeepSeek <script>alert("x")</script> & advice', quote=True) in rendered


def test_blocked_jev_followup_report_shows_reason_counts_and_no_isolated_order(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text(json.dumps({
        "schema": "jev_followup.v1",
        "run_id": "blocked-run",
        "status": "blocked",
        "backend_consumed": True,
        "campaign_error": "budget cap exceeded",
        "summary": {
            "cases": 1, "records": 1, "attempted_requests": 1,
            "completed_requests": 0, "errors": 1, "pending": 0,
            "by_backend": {},
        },
        "records": [{
            "request_kind": "isolated_safety", "case_id": "case-1", "source": "source",
            "original": "Original", "status": "error", "model": followup.JEV_MODEL,
            "mode": "Choice", "answers": {}, "candidates": [], "error": "budget cap exceeded; Bearer dummysecret",
        }],
    }), encoding="utf-8")

    document = load_report_document(tmp_path)
    metadata = {item["label"]: item["value"] for item in document["metadata"]}
    assert metadata["Частичный запуск"] == "attempted 1; completed 0; budget cap exceeded"
    assert document["status"] == "blocked"
    assert all(item["label"] != "Порядок" for item in document["rows"][0]["details"])
    assert all(row["accepted_text"] is None for row in document["rows"])

    rendered = render_report(document)
    assert "Заблокировано" in rendered
    assert "budget cap exceeded" in rendered
    assert "attempted 1; completed 0" in rendered
    assert "Bearer dummysecret" not in rendered
    assert "Bearer [redacted]" in rendered
    assert '<script>alert("x")</script>' not in rendered
