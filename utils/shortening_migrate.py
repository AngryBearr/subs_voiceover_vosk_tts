"""Offline, fail-closed migration of the September Sol6 generation-only episode."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from utils.shorten_file import _parser, prepare_run
from utils.shortening_budget import BudgetPolicy
from utils.shortening_html import publish_html_report
from utils.shortening_plan import CuePlan, PlannerConfig, make_batches
from utils.shortening_prompts import parse_generation_partial
from utils.shortening_resume import (
    ResumeError, RunJournal, _read_json, atomic_json, canonical_hash, ensure_manifest,
    inspect_journal, load_baseline, load_manifest_contract,
)
from utils.shortening_runner import (
    _local_check, _manifest_contract, _restore_baseline, _validate_timing_evidence, ModelSpec, RunConfig,
)

VERSION = "sol6-generation-v1"
SOURCE_ITEMS_HASH = "27f678347bb0a177294f7f23a55aef0a309491f6d561bb56a1d2efb44b21950b"
SOURCE_BYTES_HASH = "1a8bee086a0afebb288095b618cf2d152ff96721bc26c77241baf606c9d984fa"
SOURCE_REPORT_HASH = "d1ef89d58f8d88f7686343ebfb76722e3261d2e12adaa95da6b9cb1f40ad4850"
SOURCE_STARTS = 121
SOURCE_PROPOSALS = 500
SOURCE_AUDIO_ROWS = 425
SOURCE_SELECTED = 1219
SOURCE_ITEMS = 1523
SOURCE_ACTUAL_FITS = 124
SOURCE_UNKNOWN_ID = "c966047a268147149ab9976bd511589a"
SOURCE_KNOWN_USD = "0.073219707"
SOURCE_RESERVED_USD = "0.319488"
SOURCE_UNKNOWN_COUNT = 3
EVIDENCE = ("run_manifest.json", "resume_baseline.json", "requests.jsonl", "calls.jsonl", "report.json", "subtitles.json", "preflight_audio/audio_report.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_source(source: Path, destination: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], BudgetPolicy, RunJournal, int]:
    contract = load_manifest_contract(source)
    frozen = contract.get("budget_policy")
    metadata = contract.get("report_metadata")
    if (contract.get("source", {}).get("sha256") != SOURCE_ITEMS_HASH
            or contract.get("prompt_version") != "shortening-v4"
            or contract.get("models") != {
                "generator": asdict(ModelSpec("openrouter", "meta/muse-spark-1.3-contributor", "low")),
                "arbiter": asdict(ModelSpec("openrouter", "openai/gpt-6-sol", "high")),
            } or contract.get("has_original_timing_verifier") is not True
            or not isinstance(frozen, dict) or not isinstance(metadata, dict)
            or frozen.get("limit_usd") != "10" or frozen.get("max_requests") != 1000
            or frozen.get("max_unknown_calls") != 3
            or contract.get("run") != {"gen_batch_size": 6, "review_batch_size": 12,
                                       "max_repairs": 1, "max_context_chars": 20000, "length_tools": False}
            or metadata.get("input", {}).get("sha256") != SOURCE_BYTES_HASH):
        raise ResumeError("not the supported generation-only Sol6 source contract")
    if frozen.get("pricing_snapshot_id") != frozen.get("snapshot", {}).get("id") or metadata.get("openrouter", {}).get("pricing_snapshot") != frozen["snapshot"]:
        raise ResumeError("source pricing snapshot differs")
    if (metadata.get("budget") != {
            "limit_usd": frozen["limit_usd"], "max_requests": frozen["max_requests"],
            "reservation_basis": "published_context_and_max_rates", "max_unknown_calls": frozen["max_unknown_calls"],
            "retry_policy": "one queued recovery per failed cue/stage; no automatic replay of completed successes",
        } or metadata.get("config", {}).get("planner") != contract.get("planner")
            or metadata.get("config", {}).get("run") != {**contract["run"], "timeout": 300.0, "max_tokens": None}
            or metadata.get("roles") != contract["models"]):
        raise ResumeError("source metadata differs from frozen pricing, planner or run policy")
    policy = BudgetPolicy.from_snapshot("10", frozen["snapshot"], max_requests=1000, max_unknown_calls=1000)
    # Re-plan the exact external input and verify the calibration and model identity
    # without invoking audio, pricing endpoints, or any live transport.
    input_path = Path(metadata["input"]["absolute_path"])
    profile = Path(metadata["calibration"]["path"])
    args = _parser().parse_args([
        str(input_path), "--output-dir", str(destination), "--profile", str(profile),
        "--model-path", str(metadata["model_path"]), "--voice", str(metadata["voice"]),
        "--rate", str(metadata["rate"]), "--budget-usd", "10",
    ])
    prepared = prepare_run(args)
    if (prepared.input_bytes_sha256 != SOURCE_BYTES_HASH
            or canonical_hash(prepared.items) != SOURCE_ITEMS_HASH
            or prepared.metadata["calibration"] != metadata["calibration"]
            or prepared.metadata["model_metadata"] != metadata["model_metadata"]
            or prepared.metadata["input"] != metadata["input"]
            or prepared.metadata["roles"] != metadata["roles"]
            or prepared.planner_config != PlannerConfig(**contract["planner"])
            or [asdict(p) for p in prepared.plans] != contract["initial_plans"]
            or prepared.target_ids != contract["target_ids"]):
        raise ResumeError("source, profile, cue selection or timing identity changed")
    baseline = load_baseline(source)
    if baseline is None:
        raise ResumeError("source baseline is missing")
    records, plans = _restore_baseline(baseline, prepared.items, prepared.target_ids,
                                       prepared.planner_config, prepared.plans)
    preflight = _read_json(source / "preflight_audio/audio_report.json", "preflight audio_report.json")
    audio_rows = preflight.get("records") if isinstance(preflight, dict) else None
    preflight_ids = {row.get("cue_id") for row in audio_rows} if isinstance(audio_rows, list) else set()
    by_plan = {plan.cue_id: plan for plan in plans}
    baseline_by_id = {row["cue_id"]: row for row in records}
    if (not isinstance(audio_rows, list) or len(audio_rows) != SOURCE_AUDIO_ROWS or len(preflight_ids) != SOURCE_AUDIO_ROWS
            or any(cue_id not in baseline_by_id or baseline_by_id[cue_id].get("timing_evidence", {}).get("audio") != audio
                   or not _validate_timing_evidence(by_plan[cue_id], baseline_by_id[cue_id]["timing_evidence"])[0]
                   or (_validate_timing_evidence(by_plan[cue_id], baseline_by_id[cue_id]["timing_evidence"])[1]
                       != (baseline_by_id[cue_id]["status"] == "actual_fits"))
                   for cue_id, audio in ((row.get("cue_id"), row) for row in audio_rows))):
        raise ResumeError("preflight timing evidence differs from the frozen baseline")
    # This importer only supports the known real source snapshot. The strict
    # byte pin covers untouched records and copied summary fields too; future
    # live resumes instead validate the independent ancestral journal binding.
    if _sha(source / "report.json") != SOURCE_REPORT_HASH:
        raise ResumeError("source report snapshot differs from validated generation-only episode")
    report = _read_json(source / "report.json", "report.json")
    if (report.get("model") != contract["models"] or report.get("config") != {
            "planner": contract["planner"], "run": contract["run"]}
            or report.get("metadata") != metadata or report.get("selected_ids") != prepared.target_ids
            or report.get("source_manifest", {}).get("sha256") != SOURCE_ITEMS_HASH
            or report.get("run_state") != "incomplete" or len(report.get("records", [])) != len(records)
            or any(row.get("cue_id") != expected["cue_id"] or row.get("original") != expected["original"]
                   or row.get("final_text") != expected["original"]
                   or row.get("source_index") != expected["source_index"]
                   or row.get("timing_evidence") != expected.get("timing_evidence")
                   for row, expected in zip(report["records"], records))):
        raise ResumeError("saved source report differs from frozen source/baseline")
    subtitles = _read_json(source / "subtitles.json", "subtitles.json")
    if (not isinstance(subtitles, list) or len(subtitles) != len(prepared.items)
            or any(item.get("index") != original.get("index") or item.get("text") != original.get("text")
                   or item.get("start") != original.get("start") or item.get("end") != original.get("end")
                   for item, original in zip(subtitles, prepared.items))):
        raise ResumeError("saved subtitles differ from the unmodified source cues")
    journal = RunJournal.readonly_snapshot(source)
    inspection = inspect_journal(source, policy)
    starts = list(journal._starts.values())
    if (len(starts) != SOURCE_STARTS or len(journal._completes) != SOURCE_STARTS - 1
            or len(inspection.unknown_requests) != 1
            or inspection.unknown_requests[0]["request_id"] != SOURCE_UNKNOWN_ID
            or journal._events or any(row["stage"] != "generation" for row in starts)
            or any(row.get("status") == "accepted" or row.get("final_review") is not None
                   for row in report["records"])):
        raise ResumeError("source is not the supported unreviewed generation prefix")
    eligible = [plan.cue_id for plan in plans if plan.cue_id in prepared.target_ids and plan.status == "needs_shortening"]
    batches = make_batches(prepared.items, plans, eligible, prepared.planner_config, 6, 20000)
    if len(batches) < len(starts):
        raise ResumeError("source generation batches are incomplete")
    by_id = {row["cue_id"]: row for row in report["records"]}
    proposed = 0
    for index, start in enumerate(starts):
        if start["target_ids"] != batches[index] or start["model"] != contract["models"]["generator"]:
            raise ResumeError("source request order or target identities differ")
        complete = journal._completes.get(start["request_id"])
        if complete is None:
            if index != len(starts) - 1 or any(by_id[cue_id]["history"] for cue_id in batches[index]):
                raise ResumeError("unresolved request has a reported result")
            continue
        parsed = parse_generation_partial(complete["text"], batches[index]) if complete["error"] is None and complete["text"] is not None else None
        for cue_id in batches[index]:
            row = by_id[cue_id]
            history = row.get("history")
            if not isinstance(history, list) or len(history) != 1 or history[0].get("stage") != "generation":
                raise ResumeError("source report generation history differs from journal")
            event = history[0]
            if parsed is None:
                if row.get("status") != "error" or event.get("failure") != "transport":
                    raise ResumeError("source transport error was relabeled")
            elif cue_id in parsed.errors:
                if row.get("status") != "error" or event.get("failure") != "schema":
                    raise ResumeError("source parse error was relabeled")
            else:
                result = parsed.results[cue_id]
                candidate = result["text"] if result["status"] == "ok" else row["original"]
                if (event.get("generator_status") != result["status"]
                        or event.get("candidate") != candidate
                        or event.get("local") != _local_check(candidate, next(p for p in plans if p.cue_id == cue_id))):
                    raise ResumeError("saved proposal differs from completed response")
                if result["status"] == "ok" and row.get("candidate") != candidate:
                    raise ResumeError("saved candidate differs from completed response")
                if result["status"] == "cannot_shorten" and row.get("candidate") != (
                    None if row.get("failure") == "protocol_misuse" else candidate
                ):
                    raise ResumeError("saved cannot-shorten result differs from completed response")
                if result["status"] == "ok":
                    proposed += 1
    if (proposed != SOURCE_PROPOSALS or sum(bool(row.get("history")) for row in report["records"]) != sum(len(batch) for batch in batches[:SOURCE_STARTS - 1])
            or sum(row.get("status") == "actual_fits" for row in report["records"]) != SOURCE_ACTUAL_FITS
            or len(prepared.items) != SOURCE_ITEMS or len(prepared.target_ids) != SOURCE_SELECTED):
        raise ResumeError("source proposal or timing totals differ")
    budget = policy.summary(inspection.calls, inspection.failures)
    if (budget["physical_calls"] != SOURCE_STARTS or budget["known_spent_usd"] != SOURCE_KNOWN_USD
            or budget["unknown_reserved_usd"] != SOURCE_RESERVED_USD or budget["unknown_call_count"] != SOURCE_UNKNOWN_COUNT):
        raise ResumeError("source ledger does not match the retained whole-lineage budget")
    return contract, baseline, report, policy, journal, proposed


def migrate(source: Path, destination: Path) -> Path:
    """Make one new, guarded offline lineage; never modify the ancestor."""
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents or destination.exists():
        raise FileExistsError(f"migration destination already exists: {destination}")
    if not (source / ".run.lock").is_file() or (source / ".run.lock").is_symlink():
        raise ResumeError("source run lock is missing or unsafe; do not create files in the ancestor")
    try:
        lock = FileLock(str(source / ".run.lock"), timeout=0)
        lock.acquire()
    except Timeout as exc:
        raise ResumeError("source is locked by another process") from exc
    owned = False
    try:
        contract, baseline, report, policy, journal, proposed = _verified_source(source, destination)
        hashes = {name: _sha(source / name) for name in EVIDENCE}
        new_contract = copy.deepcopy(contract)
        new_contract["run"]["scheduling"] = "windowed"
        new_contract["budget_policy"]["max_unknown_calls"] = new_contract["budget_policy"]["max_requests"]
        meta = new_contract["report_metadata"]
        meta["config"]["run"]["scheduling"] = "windowed"
        meta["budget"]["max_unknown_calls"] = meta["budget"]["max_requests"]
        meta["budget"]["retry_policy"] = "one explicit replacement per unknown request; one queued recovery per failed cue/stage"
        meta["migration"] = {"version": VERSION, "parent_contract_hash": canonical_hash(contract),
                             "source_files_sha256": hashes, "inherited_generation_starts": SOURCE_STARTS,
                             "inherited_file_sizes": {name: (source / name).stat().st_size for name in (
                                 "requests.jsonl", "calls.jsonl", "resume_baseline.json", "preflight_audio/audio_report.json")},
                             "inherited_proposals": proposed,
                             "policy_changes": ["stage_major->windowed", "unknown_ceiling_3->physical_cap_1000"]}
        from utils.shortening_runner import _module_hashes
        new_contract["module_sha256"] = _module_hashes(primary=True)
        # Enforce exact equivalence with the normal resume contract builder.
        run = RunConfig(**new_contract["run"])
        expected = _manifest_contract(new_contract["source"]["items"],
            [CuePlan(**row) for row in new_contract["initial_plans"]], new_contract["target_ids"],
            PlannerConfig(**new_contract["planner"]), run,
            ModelSpec(**new_contract["models"]["generator"]), ModelSpec(**new_contract["models"]["arbiter"]),
            meta, True, policy)
        if new_contract != expected:
            raise ResumeError("migrated contract is not the current runner contract")
        destination.mkdir(parents=True)
        owned = True
        (destination / ".migration-incomplete").write_text(VERSION + "\n", encoding="utf-8")
        for name in ("resume_baseline.json", "requests.jsonl", "calls.jsonl", "subtitles.json", "preflight_audio/audio_report.json"):
            (destination / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, destination / name)
        if any(_sha(destination / name) != hashes[name] for name in ("resume_baseline.json", "requests.jsonl", "calls.jsonl", "subtitles.json", "preflight_audio/audio_report.json")):
            raise ResumeError("copied source evidence differs")
        calls, counts, failures = journal.accounting()
        summary = policy.summary([call for group in calls.values() for call in group], failures)
        report = copy.deepcopy(report)
        report["config"]["run"] = new_contract["run"]
        report["metadata"] = meta
        report["run_state"] = "incomplete"
        report["budget"] = summary
        report["summary"]["budget"] = copy.deepcopy(summary)
        report["summary"]["request_counts"] = {stage: counts.get(stage, 0) for stage in ("generation", "review", "repair", "recheck")}
        report["summary"]["request_failures"] = failures
        report["summary"]["accounting_complete"] = False
        report["summary"]["unknown_request_ids"] = journal.unresolved_request_ids
        report["resume"] = {"schema_version": 1, "status": "incomplete", "unknown_requests": journal.unknown_requests, "replayed_request_count": 0}
        report.pop("review_snapshot_id", None)  # Derived review view changes with the new contract.
        atomic_json(destination / "report.json", report)
        ensure_manifest(destination, new_contract, resume=False)
        # Real requests are validated above; this is a saved-run backfill,
        # not a live request observed by the HTML context manager.
        if publish_html_report(destination, confirmed_real_run=True) is None:
            raise ResumeError("inherited real HTML could not be published")
        (destination / ".migration-incomplete").unlink()
        return destination / "shortening.html"
    except BaseException:
        if owned:
            shutil.rmtree(destination)
        raise
    finally:
        lock.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    result = migrate(args.source, args.destination)
    print(json.dumps({"report": str(result), "status": "inherited_partial_unknown_blocked"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
