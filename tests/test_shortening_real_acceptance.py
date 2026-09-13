"""Opt-in, offline SH05/SH11 acceptance against the frozen real corpus.

This module intentionally exercises only preparation and the review IO/apply/
reanalysis APIs.  Its edit is a mechanically generated test fixture, not a
semantic or human-quality claim.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import time
from typing import Any, Callable

from filelock import FileLock
import pytest

from utils.apply_review import apply_review
from utils.llm_types import DetailedResponse
from utils.json_to_srt import json_to_srt
from utils.review_subtitles import reanalyze
from utils.shortening_plan import plan_subtitles
from utils.shortening_review import ReviewError, build_review_state, make_review_queue
from utils.shortening_review_io import load_review_bundle, write_derived_bundle
from utils.shortening_resume import canonical_hash
from utils.shorten_file import _LazyRequest, prepare_run
from utils.shortening_runner import ModelSpec, RunConfig, run_trial
from utils.vosk_calibration import model_metadata, profile_identity


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name("shortening_real_acceptance_manifest.json")
ACCEPTANCE_ENV = "SUBS_REAL_ACCEPTANCE"
OUTPUT_ENV = "SUBS_REAL_ACCEPTANCE_OUTPUT"
LOG_PATH = Path("/tmp/opencode/sh05-real-acceptance-final.log")

REQUIRED_PHASES = (
    "prepare_srt", "build_baseline_state", "write_baseline_bundle", "load_baseline_bundle",
    "prepare_json", "json_bundle", "apply_simulated_fixture", "apply_invariants",
    "reanalyze_cached", "reanalyze_forced", "neighbor_text_fault",
    "neighbor_timing_fault", "negative_apply", "output_refusal_and_lock", "source_profile_mismatch",
)


class OfflineGuardError(BaseException):
    """BaseException-derived guard so application exception handling cannot hide it."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fresh_output(requested: Path) -> Path:
    requested = requested.resolve()
    output_parent = (REPO_ROOT / "output").resolve()
    if requested.parent != output_parent:
        raise AssertionError(f"acceptance output must be directly under {output_parent}: {requested}")
    candidate = requested
    number = 2
    while candidate.exists():
        candidate = requested.with_name(f"{requested.name}_r{number}")
        number += 1
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.mkdir()
    return candidate


def _manifest() -> dict[str, Any]:
    value = _read_json(MANIFEST_PATH)
    if not isinstance(value, dict) or not isinstance(value.get("sources"), list) or len(value["sources"]) != 4:
        raise AssertionError("manifest must contain exactly four sources")
    return value


def _args(source: Path, output: Path, profile: Path, model: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input=source, output_dir=output, model="deepseek/deepseek-flash", profile=profile,
        model_path=model, voice=8, rate=1.5, text_mode="punctuation", variant="off",
        length_tools=False, dry_run=False, timeout=300.0, max_tokens=None,
        gen_batch_size=6, review_batch_size=12, max_repairs=1, max_context_chars=20000,
    )


def _install_guards(monkeypatch: pytest.MonkeyPatch, attempts: list[dict[str, Any]]) -> None:
    def blocked(kind: str) -> Callable[..., Any]:
        def guard(*args: Any, **kwargs: Any) -> Any:
            attempts.append({"kind": kind, "args": [str(x) for x in args[:2]]})
            raise OfflineGuardError(f"offline guard blocked {kind}")
        return guard

    monkeypatch.setattr(socket.socket, "connect", blocked("socket.connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", blocked("socket.connect_ex"))
    monkeypatch.setattr(socket, "create_connection", blocked("socket.create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", blocked("socket.getaddrinfo"))

    import utils.opencode_clean as clean
    import utils.shortening_audio as audio
    import utils.shortening_backends as backends
    import utils.shorten_file as shorten
    import utils.vosk_calibration as calibration

    monkeypatch.setattr(clean.CleanOpenCodeServer, "start", blocked("CleanOpenCodeServer.start"))
    monkeypatch.setattr(backends.LiveBackend, "__call__", blocked("LiveBackend.__call__"))
    monkeypatch.setattr(_LazyRequest, "_make_backend", blocked("_LazyRequest._make_backend"))
    monkeypatch.setattr(shorten, "verify_original_timing", blocked("shorten_file.verify_original_timing"))
    monkeypatch.setattr(audio, "verify_original_timing", blocked("shortening_audio.verify_original_timing"))
    monkeypatch.setattr(audio, "verify_audio", blocked("shortening_audio.verify_audio"))
    monkeypatch.setattr(audio, "_invoke_synthesis", blocked("shortening_audio._invoke_synthesis"))
    monkeypatch.setattr(calibration, "_invoke_synthesis", blocked("vosk_calibration._invoke_synthesis"))

    # Import the actual synthesis module before patching its callable entrypoints.
    from synthesize import synthesize as synthesis
    assert callable(synthesis.synthesize_text)
    assert callable(synthesis.create_synth)
    monkeypatch.setattr(synthesis, "synthesize_text", blocked("synthesize.synthesize_text"))
    monkeypatch.setattr(synthesis, "create_synth", blocked("synthesize.create_synth"))


def _display(item: dict[str, Any]) -> str:
    text = item.get("text")
    if isinstance(text, str):
        return text
    if isinstance(text, list) and all(isinstance(part, str) for part in text):
        return " ".join(text)
    return ""


def _phase(phases: dict[str, bool], name: str, function: Callable[[], Any]) -> Any:
    started = time.monotonic()
    try:
        value = function()
        phases[name] = True
        return value
    finally:
        # Timings are populated by the caller; this marker also makes a failed
        # phase visibly distinct from a phase which was never entered.
        phases.setdefault(f"{name}_elapsed_sec", round(time.monotonic() - started, 6))


def _assert_srt_roundtrip(bundle: Path, source: list[dict[str, Any]], invalid_ids: list[str]) -> str | None:
    report_path = bundle / "reanalysis_report.json"
    report = _read_json(report_path)
    exported = bundle / "subtitles.srt"
    if not exported.exists():
        if invalid_ids:
            return str(report.get("export_error"))
        raise AssertionError("SRT export silently missing")
    from utils.srt_to_json import parse_srt_file
    parsed = parse_srt_file(str(exported))
    assert len(parsed) == len(source), "SRT export dropped source rows"
    for expected, actual in zip(sorted(source, key=lambda x: (x.get("start", 0), x.get("index", 0))), parsed):
        assert (expected.get("start"), expected.get("end"), _display(expected)) == (actual.get("start"), actual.get("end"), _display(actual))
    # This helper is an assertion only.  In particular, it must not invent an
    # export error when the product exporter successfully serialized a source
    # containing invalid timing.  The acceptance summary records that fact as
    # a source-timing warning, separately from the product report.
    return report.get("export_error")


def _mechanical_candidate(text: str, cap: int, basis: str) -> str | None:
    from utils.length_check import prepare_length_text
    tokens = text.split()
    for end in range(1, len(tokens) + 1):
        candidate = " ".join(tokens[:end])
        if candidate != text and len(prepare_length_text(candidate, basis)) <= cap:
            return candidate
    return None


def _historical_fixture(source: list[dict[str, Any]], plans: list[Any], historical: dict[str, Any]) -> dict[str, Any]:
    from utils.length_check import prepare_length_text

    by_position = {plan.position: plan for plan in plans}
    entries: list[dict[str, Any]] = []
    for record in historical.get("records", []):
        if record.get("cue_id") not in {"cue-204", "cue-261", "cue-663", "cue-797"}:
            continue
        position = record.get("position")
        if not isinstance(position, int) or position not in by_position:
            continue
        plan = by_position[position]
        budget = record.get("budget") if isinstance(record.get("budget"), dict) else {}
        source_item = source[position]
        exact = (record.get("position") == position and record.get("source_index") == source_item.get("index")
                 and record.get("original") == plan.text
                 and budget.get("start") == plan.start and budget.get("end") == plan.end)
        entries.append({"cue_id": record.get("cue_id"), "exact_source_match": exact,
                        "historical_candidate": record.get("candidate"),
                        "historical_max_chars": budget.get("max_chars"),
                        "current_max_chars": plan.max_chars,
                        "current_budget_accepts": exact and isinstance(record.get("candidate"), str) and
                        len(prepare_length_text(record["candidate"], plan.length_basis)) <= plan.max_chars})
    return {
        "records": entries,
        "report_sha256": _sha256(Path(historical["_path"])) if historical.get("_path") else None,
        "historical_accepted_current_budget_rejected": sum(
            item["exact_source_match"] and not item["current_budget_accepts"] for item in entries
        ),
        "historical_accepted_current_budget_eligible": sum(
            item["exact_source_match"] and item["current_budget_accepts"] for item in entries
        ),
    }


def _negative_apply(
    run: Path, queue: dict[str, Any], root: Path, state: dict[str, Any], invalid_ids: list[str],
    positive_entry_id: str, positive_replacement: str,
) -> dict[str, dict[str, Any]]:
    """Reject malformed queues atomically while retaining one valid approval."""
    del state
    base = copy.deepcopy(queue)
    positive = next(entry for entry in base["entries"] if entry["cue_id"] == positive_entry_id)
    positive.update({"approved": True, "decision": "replace", "replacement_text": positive_replacement})
    bad_id = next(entry["cue_id"] for entry in base["entries"] if entry["cue_id"] != positive_entry_id)
    variants: list[tuple[str, dict[str, Any], str]] = []

    unknown = copy.deepcopy(base)
    unknown["entries"].append({"cue_id": "cue-unknown"})
    variants.append(("unknown_id", unknown, "unknown or duplicate edit queue entry"))
    duplicate = copy.deepcopy(base)
    duplicate["entries"].append(copy.deepcopy(positive))
    variants.append(("duplicate_id", duplicate, "unknown or duplicate edit queue entry"))
    stale = copy.deepcopy(base)
    stale["snapshot_id"] = "stale-snapshot"
    stale_entry = next(e for e in stale["entries"] if e["cue_id"] == bad_id)
    stale_entry["current_context_fingerprint"] = "stale-row-context"
    variants.append(("stale_snapshot", stale, "edit queue snapshot mismatch"))
    immutable = copy.deepcopy(base)
    immutable_entry = next(e for e in immutable["entries"] if e["cue_id"] == bad_id)
    immutable_entry["max_chars"] = int(immutable_entry["max_chars"]) + 1
    variants.append(("immutable_cap", immutable, "edit queue immutable field mismatch"))
    context = copy.deepcopy(base)
    context_entry = next(e for e in context["entries"] if e["cue_id"] == bad_id)
    context_entry["current_context_fingerprint"] = "changed-context"
    variants.append(("immutable_context", context, "edit queue immutable field mismatch"))
    overlong = copy.deepcopy(base)
    overlong_entry = next(e for e in overlong["entries"] if e["cue_id"] == bad_id)
    overlong_entry.update({"approved": True, "decision": "replace", "replacement_text": "x" * (int(overlong_entry["max_chars"]) + 100)})
    variants.append(("overlong", overlong, "replacement does not fit its budget"))
    if invalid_ids:
        invalid = copy.deepcopy(base)
        invalid_entry = next(entry for entry in invalid["entries"] if entry["cue_id"] == invalid_ids[0])
        invalid_entry.update({"approved": True, "decision": "replace", "replacement_text": "Да."})
        variants.append(("invalid_timing", invalid, "invalid cue cannot be approved through text queue"))

    def files_snapshot() -> dict[str, bytes]:
        return {str(path.relative_to(run)): path.read_bytes() for path in run.rglob("*") if path.is_file() and path.name != ".run.lock"}

    results: dict[str, dict[str, Any]] = {}
    for name, payload, expected_message in variants:
        path = root / f"negative_{name}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        destination = root / f"negative_output_{name}"
        before = files_snapshot()
        try:
            apply_review(run, path, destination)
        except ReviewError as exc:
            error = str(exc)
            accepted = False
            results[name] = {"accepted": accepted, "error_class": type(exc).__name__, "error": error, "expected": expected_message}
            assert expected_message in error, f"negative apply reason changed for {name}: {error}"
        except BaseException:
            raise
        else:
            results[name] = {"accepted": True, "error_class": None, "error": None, "expected": expected_message}
        assert not results[name]["accepted"] and not destination.exists(), f"negative apply unexpectedly produced output: {name}"
        assert files_snapshot() == before, f"negative apply mutated baseline files: {name}"
    return results


def _run_file(entry: dict[str, Any], corpus: Path, profile: Path, model: Path, root: Path,
              monkeypatch: pytest.MonkeyPatch, attempts: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    name = str(entry["path"])
    slug = re.sub(r"[^A-Za-z0-9]+", "_", Path(name).stem).strip("_")
    file_root = root / slug
    file_root.mkdir()
    phases: dict[str, bool] = {}
    started = time.monotonic()
    source = corpus / name
    source_before = source.read_bytes()
    historical_path = REPO_ROOT / manifest["historical_report"]["path"]
    historical_before = {p: _sha256(p) for p in historical_path.parent.iterdir() if p.is_file()}
    summary: dict[str, Any] = {"source": name, "sourcehash": entry["sha256"], "count": entry["count"], "allphasepassbools": phases}
    try:
        assert source.is_file(), f"missing real source: {source}"
        assert _sha256(source) == entry["sha256"]
        assert _sha256(profile) == manifest["profile"]["sha256"]
        actual_model = model_metadata(model)
        assert actual_model["fingerprint"] == manifest["model"]["fingerprint"]
        assert actual_model["config_sha256"] == manifest["model"]["config_sha256"]
        identity = profile_identity(actual_model, 8, 1.5)
        assert identity["model_fingerprint"] == manifest["model"]["fingerprint"]

        prepared = _phase(phases, "prepare_srt", lambda: prepare_run(_args(source, file_root / "must_not_exist", profile, model)))
        assert len(prepared.items) == entry["count"]
        assert prepared.input_format == "srt" and prepared.text_mode == "punctuation"
        assert prepared.planner_config.length_basis == "normalized_tts_text"
        assert not (file_root / "must_not_exist").exists()
        plans = prepared.plans
        all_ids = [plan.cue_id for plan in plans]
        assert all_ids == [f"cue-{i}" for i in range(len(plans))]
        assert len(set(all_ids)) == len(all_ids)
        assert all(plan.position == i for i, plan in enumerate(plans))
        invalid_ids = [plan.cue_id for plan in plans if plan.status == "invalid"]
        if entry.get("invalid_ids"):
            assert invalid_ids == entry["invalid_ids"]
        assert len(prepared.target_ids) == len([p for p in plans if p.status in {"needs_shortening", "short_timing"}])
        raw_hash = prepared.input_bytes_sha256
        parsed_hash = canonical_hash(prepared.items)
        raw_bytes = source.read_bytes()
        available_ms = [int(plan.available_ms) for plan in plans]
        summary.update({"rawhash": raw_hash, "parsedcanonicalhash": parsed_hash, "sourcecuecount": len(prepared.items),
                        "profile_sha256": manifest["profile"]["sha256"],
                        "historical_report_sha256": manifest["historical_report"]["sha256"],
                        "selectedcount": len(prepared.target_ids), "invalidIDs": invalid_ids,
                        "zero_character_budget_count": sum(plan.max_chars == 0 for plan in plans if plan.status != "invalid"),
                        "invalid_zero_character_budget_count": sum(plan.max_chars == 0 for plan in plans if plan.status == "invalid"),
                        "available_window_diagnostics": {"min_available_ms": min(available_ms), "max_available_ms": max(available_ms),
                            "count_over_10000_ms": sum(value > 10000 for value in available_ms),
                            "count_over_60000_ms": sum(value > 60000 for value in available_ms),
                            "count_nonpositive_ms": sum(value <= 0 for value in available_ms),
                            "labels": "diagnostic_only; not an error and not a claim that all source windows are normal"},
                        "multiline": sum(isinstance(item.get("text"), list) and len(item["text"]) > 1 for item in prepared.items),
                        "newline_formats": {"crlf": raw_bytes.count(b"\r\n"), "lf": raw_bytes.count(b"\n") - raw_bytes.count(b"\r\n"), "cr_only": raw_bytes.count(b"\r") - raw_bytes.count(b"\r\n")},
                        "retainedinvalidcounts": {"source": len(prepared.items), "parsed": len(prepared.items), "invalid": len(invalid_ids), "invalid_ids": invalid_ids},
                        "riskcounts": {status: sum(plan.status == status for plan in plans) for status in {"fits", "needs_shortening", "short_timing", "invalid"}}})

        metadata = copy.deepcopy(prepared.metadata)
        metadata["offline_acceptance"] = {"not_for_production": True, "simulated_user_actions": True,
            "semantic_quality_evidence": False, "network_forbidden": True,
            "source": "real_frozen_corpus", "label": "offline_acceptance"}
        baseline_state = _phase(phases, "build_baseline_state", lambda: build_review_state(
            source_items=copy.deepcopy(prepared.items), current_items=copy.deepcopy(prepared.items), plans=plans,
            records=[], planner_config=prepared.planner_config, metadata=metadata,
            selected_ids=list(prepared.target_ids)))
        assert all(not row["text_changed"] and row["measured_duration_sec"] is None and row["measured_fit"] is None for row in baseline_state["outcomes"])
        assert all(row["validation_source"] in {None, "prediction"} for row in baseline_state["outcomes"])
        queue = make_review_queue(baseline_state)
        assert {row["cue_id"] for row in baseline_state["outcomes"] if row["outcome"] not in {"unchanged_and_fits", "accepted_correction"}} == {row["cue_id"] for row in queue["entries"]}
        review_headings = [line for line in ("\n".join(f"### {row['cue_id']}" for row in queue["entries"])).splitlines() if line]
        assert len(review_headings) == len(queue["entries"])
        baseline_dir = _phase(phases, "write_baseline_bundle", lambda: write_derived_bundle(
            file_root / "baseline_srt", baseline_state,
            {"offline_acceptance": True, "not_for_production": True, "source": "real_frozen_corpus"}, kind="reanalysis"))
        baseline_bundle = file_root / "baseline_srt"
        review_markdown = (baseline_bundle / "review.md").read_text(encoding="utf-8")
        assert all(len(re.findall(rf"^### {re.escape(row['cue_id'])} ", review_markdown, re.MULTILINE)) == 1 for row in queue["entries"])
        loaded = _phase(phases, "load_baseline_bundle", lambda: load_review_bundle(baseline_bundle, source_path=source))
        assert loaded["state"]["source_items"] == prepared.items
        report_before_roundtrip = _sha256(baseline_bundle / "reanalysis_report.json")
        export_error = _assert_srt_roundtrip(baseline_bundle, prepared.items, invalid_ids)
        assert _sha256(baseline_bundle / "reanalysis_report.json") == report_before_roundtrip

        json_source = file_root / "full_source_utf8_bom.json"
        json_source.write_bytes(b"\xef\xbb\xbf" + json.dumps(prepared.items, ensure_ascii=False, indent=2).encode("utf-8"))
        json_prepared = _phase(phases, "prepare_json", lambda: prepare_run(_args(json_source, file_root / "json_must_not_exist", profile, model)))
        assert json_prepared.items == prepared.items and len(json_prepared.plans) == len(prepared.plans)
        assert json_prepared.target_ids == prepared.target_ids
        json_metadata = copy.deepcopy(json_prepared.metadata); json_metadata["offline_acceptance"] = metadata["offline_acceptance"]
        json_state = build_review_state(source_items=copy.deepcopy(json_prepared.items), current_items=copy.deepcopy(json_prepared.items), plans=json_prepared.plans, records=[], planner_config=json_prepared.planner_config, metadata=json_metadata, selected_ids=json_prepared.target_ids)
        json_bundle = file_root / "baseline_json"
        write_derived_bundle(json_bundle, json_state, {"offline_acceptance": True, "not_for_production": True, "source": "real_frozen_corpus"}, kind="reanalysis")
        assert load_review_bundle(json_bundle, source_path=json_source)["state"]["source_items"] == prepared.items
        phases["json_bundle"] = True

        historical = _read_json(historical_path)
        historical["_path"] = str(historical_path)
        assert _sha256(historical_path) == manifest["historical_report"]["sha256"]
        fixture = _historical_fixture(prepared.items, plans, historical) if name == "Survivor.S44E11.srt" else {"records": [], "report_sha256": None}
        for item in fixture["records"]:
            assert item["exact_source_match"]
        eligible = next((plan for plan in plans if plan.cue_id in set(queue["entries"][i]["cue_id"] for i in range(len(queue["entries"]))) and plan.status != "invalid" and plan.max_chars > 0 and _mechanical_candidate(plan.text, plan.max_chars, plan.length_basis)), None)
        assert eligible is not None, "no positive full-file mechanical apply candidate"
        replacement = _mechanical_candidate(eligible.text, eligible.max_chars, eligible.length_basis)
        assert replacement is not None
        edited = copy.deepcopy(queue)
        by_id = {row["cue_id"]: row for row in edited["entries"]}
        by_id[eligible.cue_id].update({"approved": True, "decision": "replace", "replacement_text": replacement})
        other = next(row for row in edited["entries"] if row["cue_id"] != eligible.cue_id)
        other["replacement_text"] = "deliberately ignored test mutation"
        edit_path = file_root / "simulated_human_actions.json"
        edit_path.write_text(json.dumps(edited, ensure_ascii=False, indent=2), encoding="utf-8")
        application_dir = file_root / "application"
        applied = _phase(phases, "apply_simulated_fixture", lambda: apply_review(baseline_bundle, edit_path, application_dir, source_path=source))
        applied_state = load_review_bundle(application_dir, source_path=source)["state"]
        assert applied_state["current_items"][eligible.position] != prepared.items[eligible.position]
        for pos, (before, after) in enumerate(zip(prepared.items, applied_state["current_items"])):
            if pos != eligible.position:
                assert before == after
        assert applied_state["metadata"]["offline_acceptance"]["not_for_production"] is True
        assert applied_state["metadata"]["offline_acceptance"]["simulated_user_actions"] is True
        assert applied_state["current_items"][eligible.position].get("index") == prepared.items[eligible.position].get("index")
        summary["test-only-approval-provenance"] = {"source": "human_fixture", "semantic_validation": False,
            "label": "SIMULATED_HUMAN_ACTION_FIXTURE", "mechanical_test_mutation_not_semantically_approved": True,
            "cue_id": eligible.cue_id, "original_length": len(eligible.text), "replacement_length": len(replacement)}
        assert all(not row.get("approved", False) for row in edited["entries"] if row["cue_id"] != eligible.cue_id)
        phases["apply_invariants"] = True

        unchanged_dir = file_root / "reanalyze_cached"
        cached = _phase(phases, "reanalyze_cached", lambda: reanalyze(application_dir, unchanged_dir, source_path=source))
        cached_state = load_review_bundle(unchanged_dir, source_path=source)["state"]
        cached_row = next(row for row in cached_state["outcomes"] if row["cue_id"] == eligible.cue_id)
        assert cached_row["validation_source"] == "human" and cached_row["reusable_checks"]["semantic"]
        assert cached["report"]["provenance"]["force_recheck"] is False
        forced_dir = file_root / "reanalyze_forced"
        forced = _phase(phases, "reanalyze_forced", lambda: reanalyze(application_dir, forced_dir, source_path=source, force_recheck=True))
        forced_state = load_review_bundle(forced_dir, source_path=source)["state"]
        forced_row = next(row for row in forced_state["outcomes"] if row["cue_id"] == eligible.cue_id)
        assert forced_row["validation_source"] is None and forced_row["measured_duration_sec"] is None and not forced_row["reusable_checks"]["semantic"]
        assert forced_state["source_items"] == cached_state["source_items"] and forced_state["current_items"] == cached_state["current_items"]
        assert forced_state["initial_intervention_ids"] == cached_state["initial_intervention_ids"]
        assert forced_state["summary"]["initial_intervention_count"] == cached_state["summary"]["initial_intervention_count"]
        fault = copy.deepcopy(applied_state["current_items"])
        neighbor = next((i for i in range(max(0, eligible.position - 3), min(len(fault), eligible.position + 4)) if i != eligible.position and _display(fault[i])), None)
        assert neighbor is not None
        fault[neighbor]["text"] = _display(fault[neighbor]) + " [controlled-fault]"
        fault_json = file_root / "controlled_neighbor_text_fault.json"
        fault_json.write_text(json.dumps(fault, ensure_ascii=False, indent=2), encoding="utf-8")
        fault_dir = file_root / "reanalyze_neighbor_text_fault"
        reanalyze(application_dir, fault_dir, current_json=fault_json, source_path=source)
        fault_state = load_review_bundle(fault_dir, source_path=source)["state"]
        fault_row = next(row for row in fault_state["outcomes"] if row["cue_id"] == eligible.cue_id)
        assert fault_row["current_context_fingerprint"] != cached_row["current_context_fingerprint"]
        assert eligible.cue_id in {entry["cue_id"] for entry in make_review_queue(fault_state)["entries"]}
        phases["neighbor_text_fault"] = True
        timing_fault = copy.deepcopy(applied_state["current_items"])
        timing_fault[neighbor]["end"] = int(timing_fault[neighbor]["end"]) + 1
        timing_json = file_root / "controlled_neighbor_timing_fault.json"
        timing_json.write_text(json.dumps(timing_fault, ensure_ascii=False, indent=2), encoding="utf-8")
        timing_dir = file_root / "reanalyze_neighbor_timing_fault"
        reanalyze(application_dir, timing_dir, current_json=timing_json, source_path=source)
        timing_state = load_review_bundle(timing_dir, source_path=source)["state"]
        timing_row = next(row for row in timing_state["outcomes"] if row["cue_id"] == eligible.cue_id)
        current_plans = plan_subtitles(timing_state["current_items"], prepared.planner_config)
        expected_plan = next(plan for plan in current_plans if plan.position == eligible.position)
        assert timing_row["current_context_fingerprint"] != cached_row["current_context_fingerprint"]
        assert timing_row["max_chars"] == expected_plan.max_chars
        phases["neighbor_timing_fault"] = True

        negatives = _negative_apply(baseline_bundle, queue, file_root, baseline_state, invalid_ids, eligible.cue_id, replacement)
        phases["negative_apply"] = True
        refusal = file_root / "output_refusal"
        apply_review(baseline_bundle, edit_path, refusal, source_path=source)
        with pytest.raises(ReviewError, match="output directory already exists"):
            apply_review(baseline_bundle, edit_path, refusal, source_path=source)
        lock_path = baseline_bundle / ".run.lock"
        with FileLock(str(lock_path), timeout=0):
            with pytest.raises(ReviewError, match="locked"):
                apply_review(baseline_bundle, edit_path, file_root / "locked_output", source_path=source)
        phases["output_refusal_and_lock"] = True
        copy_root = file_root / "mismatch_copies"
        copy_root.mkdir()
        copied_source = copy_root / source.name; shutil.copy2(source, copied_source)
        copied_profile = copy_root / profile.name; shutil.copy2(profile, copied_profile)
        copy_prepared = prepare_run(_args(copied_source, copy_root / "prepared_unused", copied_profile, model))
        copy_state = build_review_state(source_items=copy.deepcopy(copy_prepared.items), current_items=copy.deepcopy(copy_prepared.items), plans=copy_prepared.plans, records=[], planner_config=copy_prepared.planner_config, metadata=copy_prepared.metadata, selected_ids=copy_prepared.target_ids)
        copy_bundle = copy_root / "bundle"; write_derived_bundle(copy_bundle, copy_state, {"offline_acceptance": True, "not_for_production": True}, kind="reanalysis")
        copied_source_bytes = copied_source.read_bytes()
        copied_source.write_bytes(copied_source_bytes + b"\n")
        with pytest.raises(ReviewError, match="raw input hash does not match review metadata"):
            load_review_bundle(copy_bundle, source_path=copied_source)
        copied_source.write_bytes(copied_source_bytes)
        copied_profile.write_bytes(copied_profile.read_bytes() + b"\n")
        with pytest.raises(ReviewError, match="calibration profile (file does not match review metadata|hash does not match stored profile)"):
            load_review_bundle(copy_bundle, source_path=copied_source)
        phases["source_profile_mismatch"] = True
        summary.update({"formatcounts": {"srt": len(prepared.items), "json": len(json_prepared.items)},
                        "baselineoutcomecounts": baseline_state["summary"]["counts"], "queuecount": len(queue["entries"]),
                        "negativecaseresults": negatives, "historicalcandidatecoverage": fixture,
                         "source_profile_mismatch_coverage": "source and profile mismatch tested on this real file",
                        "elapsedtimes": {key: value for key, value in phases.items() if key.endswith("_elapsed_sec")},
                        "network/backend/synthesisattempts": attempts, "new_paid_calls": 0, "new_cost": 0,
                         "quality_assessment": "not_performed", "export_error": export_error,
                         "product_report_export_error": export_error,
                         "source_timing_warning": bool(invalid_ids),
                         "retained_zero_length_ids": invalid_ids,
                         "retained_zero_length_count": len(invalid_ids)})
    finally:
        summary["allphasepassbools"] = {key: value for key, value in phases.items() if isinstance(value, bool)}
        summary.setdefault("elapsedtimes", {})["total_sec"] = round(time.monotonic() - started, 6)
        summary.setdefault("network/backend/synthesisattempts", attempts)
        (file_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        assert source.read_bytes() == source_before
        assert {p: _sha256(p) for p in historical_before} == historical_before
    assert set(summary["allphasepassbools"]) == set(REQUIRED_PHASES)
    assert all(summary["allphasepassbools"].get(key) is True for key in REQUIRED_PHASES)
    assert not attempts
    return summary


def _write_aggregate(root: Path, exitstatus: int) -> None:
    failure_log = root / "pytest_failure_log.txt"
    if not failure_log.exists():
        failure_log.write_text(f"pytest session exit status: {exitstatus}\n", encoding="utf-8")
    summaries = [_read_json(path) for path in sorted(root.glob("*/summary.json"))]
    start = _read_json(root / "acceptance_start.json")
    all_four_pass = _all_four_pass(summaries, start["expected_source_hashes"], exitstatus)
    runner_path = root / "runner_acceptance.json"
    runner = _read_json(runner_path) if runner_path.exists() else {"passed": False, "status": "not_run"}
    assert _sha256(Path(__file__)) == start["harness_sha256"]
    assert _sha256(MANIFEST_PATH) == start["manifest_sha256"]
    aggregate = {
        "schema_version": 1, "exitstatus": exitstatus, "session_exit_status": exitstatus,
        "all_four_pass": all_four_pass, "aggregate_acceptance_pass": bool(all_four_pass and runner.get("passed") is True),
        "required_phase_keys": list(REQUIRED_PHASES),
        "corpus_selection_rationale": "three large/diverse files plus the fourth historical anchor",
        "corpus_totals": {
            "source_count_all_four": sum(int(item.get("count", 0)) for item in summaries),
            "source_count_first_three": sum(int(item.get("count", 0)) for item in summaries[:3]),
            "expected_source_count_all_four": 4419,
            "expected_source_count_first_three": 3525,
            "zero_character_budget_count_valid_all_four": sum(int(item.get("zero_character_budget_count", 0)) for item in summaries),
            "invalid_zero_character_budget_count_all_four": sum(int(item.get("invalid_zero_character_budget_count", 0)) for item in summaries),
        },
        "baseline_queue_note": "baseline queue is the unresolved full-file taxonomy, not an LLM/manual fraction",
        "quality_claim": "no independent semantic, TTS, or audio-quality claim; quality_assessment is not_performed",
        "source_summaries": summaries, "new_paid_calls": 0, "new_cost": 0,
        "frozen_external_hashes": {"sources": {item["path"]: item["sha256"] for item in _manifest()["sources"]},
                                    "profile": _manifest()["profile"]["sha256"],
                                    "historical_report": _manifest()["historical_report"]["sha256"]},
        "harness_sha256": start["harness_sha256"], "manifest_sha256": start["manifest_sha256"],
        "captured_harness_sha256": start["harness_sha256"], "captured_manifest_sha256": start["manifest_sha256"],
        "runner_acceptance": runner,
        "log_path": str(LOG_PATH),
    }
    (root / "report.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _all_four_pass(summaries: list[dict[str, Any]], expected_hashes: list[str], exitstatus: int) -> bool:
    hashes = [str(item.get("sourcehash")) for item in summaries]
    return bool(
        exitstatus == 0 and len(summaries) == 4 and len(set(hashes)) == 4
        and set(hashes) == set(expected_hashes)
        and sum(int(item.get("count", 0)) for item in summaries) == 4419
        and sum(int(item.get("count", 0)) for item in summaries[:3]) == 3525
        and sum(int(item.get("zero_character_budget_count", 0)) for item in summaries) == 906
        and all(set(item.get("allphasepassbools", {})) == set(REQUIRED_PHASES)
                and all(item.get("allphasepassbools", {}).get(key) is True for key in REQUIRED_PHASES)
                for item in summaries)
    )


def test_aggregate_gate_rejects_missing_phase_fourth_source_and_nonzero_exit() -> None:
    counts = [1523, 1007, 995, 894]
    zero_counts = [300, 200, 200, 206]
    summaries = [{"sourcehash": f"h{i}", "count": counts[i], "zero_character_budget_count": zero_counts[i],
                  "allphasepassbools": {key: True for key in REQUIRED_PHASES}} for i in range(4)]
    assert _all_four_pass(summaries, [f"h{i}" for i in range(4)], 0)
    missing_phase = copy.deepcopy(summaries)
    del missing_phase[0]["allphasepassbools"][REQUIRED_PHASES[-1]]
    assert not _all_four_pass(missing_phase, [f"h{i}" for i in range(4)], 0)
    assert not _all_four_pass(summaries[:3], [f"h{i}" for i in range(4)], 0)
    assert not _all_four_pass(summaries, [f"h{i}" for i in range(4)], 1)


def test_offline_guards_hit_real_entrypoints_without_loading_models(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[dict[str, Any]] = []
    _install_guards(monkeypatch, attempts)
    import utils.shortening_backends as backends
    import utils.shortening_audio as audio
    backend = object.__new__(backends.LiveBackend)
    with pytest.raises(OfflineGuardError, match="LiveBackend.__call__"):
        asyncio.run(backend(ModelSpec("fixture", "offline-acceptance", "off"), "", {}, False))
    with pytest.raises(OfflineGuardError, match="shortening_audio._invoke_synthesis"):
        audio._invoke_synthesis(lambda **_: None, "", Path("blocked.wav"), voice=8, speech_rate=1.5, model_path=Path("."), synth_instance=None)
    assert {attempt["kind"] for attempt in attempts} == {"LiveBackend.__call__", "shortening_audio._invoke_synthesis"}
    assert not any(attempt["kind"].startswith("socket.") for attempt in attempts)


@pytest.fixture(scope="session")
def real_acceptance_root(request: pytest.FixtureRequest) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    if os.environ.get(ACCEPTANCE_ENV) != "1":
        pytest.skip("SUBS_REAL_ACCEPTANCE=1 is required for the real-corpus offline acceptance")
    manifest = _manifest()
    output_value = os.environ.get(OUTPUT_ENV)
    if not output_value:
        raise AssertionError(f"{OUTPUT_ENV} must name a fresh directory under repo/output")
    corpus = Path(os.environ.get("SUBS_REAL_CORPUS_ROOT", manifest["corpus_root_default"]))
    profile = REPO_ROOT / manifest["profile"]["path"]
    model = REPO_ROOT / manifest["model"]["path"]
    assert corpus.is_dir() and profile.is_file() and model.is_dir()
    root = _fresh_output(Path(output_value))
    (root / "acceptance_start.json").write_text(json.dumps({
        "harness_sha256": _sha256(Path(__file__)),
        "manifest_sha256": _sha256(MANIFEST_PATH),
        "expected_source_hashes": [str(item["sha256"]) for item in manifest["sources"]],
    }, indent=2) + "\n", encoding="utf-8")
    request.addfinalizer(lambda: _write_aggregate(root, 0 if not request.session.testsfailed else 1))
    return manifest, corpus, profile, model, root


@pytest.mark.parametrize("source_entry", _manifest()["sources"], ids=lambda item: str(item["path"]))
def test_real_shortening_acceptance(source_entry: dict[str, Any], real_acceptance_root: tuple[dict[str, Any], Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, corpus, profile, model, root = real_acceptance_root
    attempts: list[dict[str, Any]] = []
    _install_guards(monkeypatch, attempts)
    _run_file(source_entry, corpus, profile, model, root, monkeypatch, attempts, manifest)


def test_real_full_runner_checkpoint_resume(
    real_acceptance_root: tuple[dict[str, Any], Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the complete frozen S44E07 file through the runner with a mock transport."""
    manifest, corpus, profile, model, root = real_acceptance_root
    entry = next(item for item in manifest["sources"] if item["path"] == "Survivor.S44E07.srt")
    source = corpus / str(entry["path"])
    runner_root = root / "runner_full_Survivor_S44E07"
    attempts: list[dict[str, Any]] = []
    _install_guards(monkeypatch, attempts)
    prepared = prepare_run(_args(source, runner_root / "must_not_exist", profile, model))
    assert len(prepared.items) == 995
    primary_before = [(item.get("text"), item.get("start"), item.get("end"), item.get("index")) for item in prepared.items]
    request_count = 0

    async def fixture_request(spec: ModelSpec, _system: str, user: str, _schema: dict[str, Any], _allow_tools: bool) -> DetailedResponse:
        nonlocal request_count
        request_count += 1
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        targets = payload["targets"]
        if spec.provider != "fixture" or spec.model != "offline-acceptance" or spec.variant != "off":
            raise AssertionError(f"unexpected model spec: {spec}")
        if "Generate one result" in user:
            results = [{"cue_id": row["cue_id"], "text": row["original"], "status": "cannot_shorten"} for row in targets]
        else:
            results = [{"cue_id": row["cue_id"], "verdict": "impossible", "issues": [], "advice": "offlinefixture"} for row in targets]
        return DetailedResponse(json.dumps({"results": results}, ensure_ascii=False), [])

    def offline_timing(plans: list[Any], _output_dir: Path) -> dict[str, dict[str, Any]]:
        return {plan.cue_id: {"error": {"type": "OfflineAcceptanceNoSynthesis"}} for plan in plans}

    config = RunConfig(gen_batch_size=1000, review_batch_size=1000, max_repairs=0, max_context_chars=20000)
    metadata = copy.deepcopy(prepared.metadata)
    metadata["offline_acceptance"] = {
        "fixture_transport": True, "not_for_production": True, "semantic_quality_evidence": False,
        "network_forbidden": True, "quality_evidence": "none", "source": "real_frozen_corpus",
    }
    generator = ModelSpec("fixture", "offline-acceptance", "off")
    report = asyncio.run(run_trial(
        items=copy.deepcopy(prepared.items), plans=prepared.plans, target_ids=prepared.target_ids,
        planner_config=prepared.planner_config, run_config=config, generator=generator, arbiter=generator,
        output_dir=runner_root, request_fn=fixture_request, original_timing_verifier=offline_timing,
        report_metadata=metadata,
    ))
    assert report["run_state"] == "completed"
    assert len(report["outcomes"]) == 995
    subtitles = _read_json(runner_root / "subtitles.json")
    assert [(item.get("text"), item.get("start"), item.get("end"), item.get("index")) for item in subtitles] == primary_before
    assert report["summary"]["invalid_input_count"] == 2
    assert all(report["outcomes"][index]["current_timing"] == report["outcomes"][index]["source_timing"] for index in (70, 356))
    assert report["metadata"]["offline_acceptance"] == metadata["offline_acceptance"]
    assert report["cost"]["direct"]["known_subtotal_usd"] == 0
    assert report["cost"]["direct"]["input_tokens"] == 0
    assert report["cost"]["direct"]["output_tokens"] == 0
    assert load_review_bundle(runner_root, source_path=source)["state"]["source_items"] == prepared.items
    assert not attempts
    assert request_count > 0
    first_bytes = {path.name: path.read_bytes() for path in runner_root.iterdir()
                   if path.is_file() and path.name not in {".run.lock", "report.json"}}
    first_requests = (runner_root / "requests.jsonl").read_bytes()
    first_calls = (runner_root / "calls.jsonl").read_bytes() if (runner_root / "calls.jsonl").exists() else b""

    async def must_not_call(*_args: Any, **_kwargs: Any) -> DetailedResponse:
        raise AssertionError("resume must not call the fixture transport")

    resumed = asyncio.run(run_trial(
        items=copy.deepcopy(prepared.items), plans=prepared.plans, target_ids=prepared.target_ids,
        planner_config=prepared.planner_config, run_config=config, generator=generator, arbiter=generator,
        output_dir=runner_root, request_fn=must_not_call, original_timing_verifier=offline_timing,
        report_metadata=metadata, resume=True,
    ))
    assert request_count == len([line for line in first_requests.splitlines() if json.loads(line)["state"] == "start"])
    assert (runner_root / "requests.jsonl").read_bytes() == first_requests
    resumed_calls = (runner_root / "calls.jsonl").read_bytes() if (runner_root / "calls.jsonl").exists() else b""
    assert resumed_calls == first_calls
    assert {path.name: path.read_bytes() for path in runner_root.iterdir()
            if path.is_file() and path.name not in {".run.lock", "report.json"}} == first_bytes
    assert (runner_root / "subtitles.json").read_bytes() == first_bytes["subtitles.json"]
    assert (runner_root / "subtitles.srt").read_bytes() == first_bytes["subtitles.srt"]
    assert (runner_root / "review.json").read_bytes() == first_bytes["review.json"]
    assert (runner_root / "outcomes.json").read_bytes() == first_bytes["outcomes.json"]
    assert resumed["summary"]["status_counts"] == report["summary"]["status_counts"]
    assert resumed["summary"]["request_counts"] == report["summary"]["request_counts"]
    assert report["summary"]["request_counts"]
    (root / "runner_acceptance.json").write_text(json.dumps({
        "passed": True, "mode": "mock transport on real full S44E07 file; not an actual LLM or SH08 pilot",
        "fixture_request_count": request_count, "new_resume_fixture_calls": 0,
        "actual_paid_requests": 0, "actual_network_requests": 0, "calls_journal_bytes_unchanged": True,
        "offline_timing_error": "OfflineAcceptanceNoSynthesis", "outcome_count": len(report["outcomes"]),
        "all_invalid_retained": report["summary"]["invalid_input_count"] == 2,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    # Kept as a documentation hook for pytest plugin loading; the fixture
    # finalizer writes the aggregate because test modules are not plugins.
    return None
