"""CLI for deterministic planning and optional live shortening trials."""

from __future__ import annotations

import argparse
import asyncio
import copy
import functools
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.deepseek_limits import (
    DEEPSEEK_LIMIT_POLICY_VERSION,
    DEEPSEEK_PROJECT_MAX_TOKENS,
    DEEPSEEK_PROVIDER_ADVERTISED_CAP_LABEL,
    resolve_deepseek_token_limit,
)
from utils.deepseek_transport import DEEPSEEK_CURRENT_MODEL, DEEPSEEK_MODEL_CHOICES
from utils.shortening_costs import (
    CURRENT_PRICE_SNAPSHOT_ID,
    DEEPSEEK_RATES_USD_PER_MILLION,
    PRICE_SNAPSHOTS,
    PRICE_SNAPSHOT_DATE,
    PRICE_SOURCE,
    project_500,
)
from utils.shortening_backends import LiveBackend
from utils.opencode_clean import CleanOpenCodeServer
from utils.shortening_plan import CuePlan, PlannerConfig, plan_subtitles, select_sample
from utils.shortening_timing import apply_timing_profile
from utils.shortening_runner import ModelSpec, RunConfig, run_trial
from utils.shortening_audio import TEXT_MODES, verify_original_timing
from utils.shortening_prompts import ARBITER_SYSTEM_PROMPT, GENERATOR_SYSTEM_PROMPT, PROMPT_VERSION


GENERATOR_CHOICES = {
    "deepseek:off": ModelSpec("deepseek", DEEPSEEK_CURRENT_MODEL, "off"),
    "deepseek:low": ModelSpec("deepseek", DEEPSEEK_CURRENT_MODEL, "low"),
    "deepseek:high": ModelSpec("deepseek", DEEPSEEK_CURRENT_MODEL, "high"),
    "deepseek:max": ModelSpec("deepseek", DEEPSEEK_CURRENT_MODEL, "max"),
    "opencode:none": ModelSpec("opencode", "openai/gpt-5.6-luna", "none"),
    "opencode:low": ModelSpec("opencode", "openai/gpt-5.6-luna", "low"),
    "opencode:medium": ModelSpec("opencode", "openai/gpt-5.6-luna", "medium"),
    "opencode:high": ModelSpec("opencode", "openai/gpt-5.6-luna", "high"),
}


def _load_items(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("input must be a JSON array of objects")
    return value


def _selection(plans: list[Any], args: argparse.Namespace) -> list[str]:
    if args.per_severity == 0:
        return [plan.cue_id for plan in plans if plan.status in {"needs_shortening", "short_timing"}]
    return select_sample(plans, args.per_severity, args.seed, args.short_count)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _frozen_selection(
    path: Path,
    plans: list[CuePlan],
    args: argparse.Namespace,
    source_hash: str,
) -> list[str]:
    manifest = _load_json_object(path, "selection manifest")
    declared_hash = manifest.get("sha256", manifest.get("input_hash"))
    if declared_hash != source_hash:
        raise ValueError("selection manifest source hash does not match input")
    selection = manifest.get("selection")
    selection_object = selection if isinstance(selection, dict) else {}
    selected = selection_object.get("selected_ids", manifest.get("target_ids"))
    if not isinstance(selected, list) or any(not isinstance(value, str) for value in selected):
        raise ValueError("selection manifest selected_ids must be a list of strings")
    if len(set(selected)) != len(selected):
        raise ValueError("selection manifest selected_ids must not contain duplicates")
    values = manifest.get("planner")
    if not isinstance(values, dict):
        raise ValueError("selection manifest planner is required")
    for name, expected in (("cps", args.cps), ("gap_ms", args.gap_ms), ("max_speed", args.max_speed)):
        if values.get(name) != expected:
            raise ValueError(f"selection manifest {name} is incompatible with arguments")
    plans_by_id = {plan.cue_id: plan for plan in plans}
    unknown = [cue_id for cue_id in selected if cue_id not in plans_by_id]
    if unknown:
        raise ValueError(f"selection manifest contains unknown cue_id(s): {', '.join(unknown)}")
    invalid = [cue_id for cue_id in selected if plans_by_id[cue_id].status not in {"needs_shortening", "short_timing"}]
    if invalid:
        raise ValueError(f"selection manifest contains non-target cue_id(s): {', '.join(invalid)}")
    return list(selected)


def _model_key(spec: ModelSpec) -> str:
    return f"{spec.provider}-{spec.variant}"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(handle)
    temporary = Path(raw)
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_args(args: argparse.Namespace) -> None:
    integer_ranges = (("repetitions", args.repetitions, 1), ("per_severity", args.per_severity, 0),
                      ("short_count", args.short_count, 0), ("generation_batch_size", args.generation_batch_size, 1),
                      ("review_batch_size", args.review_batch_size, 1), ("max_repairs", args.max_repairs, 0))
    for name, value, minimum in integer_ranges:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    if args.max_repairs > 2:
        raise ValueError("max_repairs must be between 0 and 2")
    for name, value in (("cps", args.cps), ("max_speed", args.max_speed), ("timeout", args.timeout)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 1.0 <= float(args.max_speed) <= 1.5:
        raise ValueError("max_speed must be between 1.0 and 1.5")
    if isinstance(args.gap_ms, bool) or not isinstance(args.gap_ms, int) or args.gap_ms < 0:
        raise ValueError("gap_ms must be non-negative")
    if args.max_tokens is not None:
        for effort in (None, "off", "low", "high", "max"):
            resolve_deepseek_token_limit(effort, args.max_tokens)
    if args.generators and len(set(args.generators)) != len(args.generators):
        raise ValueError("duplicate --generator choices are not allowed")
    if isinstance(args.preflight_voice, bool) or not isinstance(args.preflight_voice, int) or args.preflight_voice < 0:
        raise ValueError("preflight_voice must be a non-negative integer")


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _error_type(exc: BaseException) -> str:
    return type(exc).__name__


def _known_cost_value(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) and value >= 0 else None


def _summary_payload(args: argparse.Namespace, target_ids: list[str], reports: list[dict[str, Any]], *, status: str) -> dict[str, Any]:
    aggregate: dict[str, list[dict[str, Any]]] = {}
    for row in reports:
        key = f"{row['model']['provider']}:{row['model']['variant']}"
        aggregate.setdefault(key, []).append(row)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "source": str(args.input.resolve()),
        "selected_ids": target_ids,
        "repetitions": args.repetitions,
        "trials": reports,
        "by_model_variant": {},
        "experiment_spent_known_subtotal_usd": sum(
            value for row in reports
            if (value := _known_cost_value(row["cost"]["direct"].get("known_subtotal_usd"))) is not None
        ),
        "experiment_accounting_complete": all(
            row["cost"]["direct"].get("accounting_complete", False) for row in reports
        ),
        "spent_note": "This is an explicit experiment ledger total, not a production cost or a sum used to choose a model.",
    }
    for key, rows in aggregate.items():
        acceptance = [row["summary"]["accepted_count"] / row["summary"]["selected_count"] if row["summary"]["selected_count"] else 0.0 for row in rows]
        manual = [row["summary"]["manual_fraction"] for row in rows]
        known = [
            value for row in rows
            if (value := _known_cost_value(row["cost"]["direct"].get("known_subtotal_usd"))) is not None
        ]
        projections = [row["cost"].get("projection_500", {}) for row in rows]
        known_projection = [row.get("projected_known_cost_usd") for row in projections]
        summary["by_model_variant"][key] = {
            "acceptance_mean": sum(acceptance) / len(acceptance), "acceptance_min": min(acceptance), "acceptance_max": max(acceptance),
            "manual_fraction_mean": sum(manual) / len(manual), "manual_fraction_min": min(manual), "manual_fraction_max": max(manual),
            "direct_known_cost_mean": sum(known) / len(known) if known else None,
            "direct_known_cost_min": min(known, default=None), "direct_known_cost_max": max(known, default=None),
            "direct_known_cost_sample_count": len(known),
            "direct_accounting_complete": [row["cost"]["direct"].get("accounting_complete", False) for row in rows],
            "projection_known_cost_mean": (sum(value for value in known_projection if value is not None) / len([value for value in known_projection if value is not None])
                                            if any(value is not None for value in known_projection) else None),
            "projection_known_cost_min": min((value for value in known_projection if value is not None), default=None),
            "projection_known_cost_max": max((value for value in known_projection if value is not None), default=None),
            "projection_known_costs": known_projection,
            "projection_accounting_complete": [projection.get("accounting_complete", False) for projection in projections],
            "direct_usage_costs": [row["cost"]["direct"] for row in rows],
            "projection_500": projections,
            "meaning_scores_are_arbiter_verdicts": True,
        }
    return summary


def _manifest(items: list[dict[str, Any]], plans: list[Any], target_ids: list[str], args: argparse.Namespace,
               *, timing_profile: dict[str, Any] | None = None, selection_manifest: dict[str, str] | None = None) -> dict[str, Any]:
    source = args.input.resolve()
    requested_deepseek_model = getattr(args, "deepseek_model", DEEPSEEK_CURRENT_MODEL)
    source_indices = [item.get("index") for item in items if isinstance(item, dict) and item.get("index") is not None]
    duplicate_indices = sorted({index for index in source_indices if source_indices.count(index) > 1}, key=str)
    duplicate_entries = sum(max(0, source_indices.count(index) - 1) for index in set(source_indices))
    stale_analysis = [
        plan.cue_id for plan in plans
        if isinstance(items[plan.position].get("analysis"), dict)
        and (
            "used_gap_sec" in items[plan.position]["analysis"]
            or "extended_duration_sec" in items[plan.position]["analysis"]
            or any(
                key in items[plan.position]["analysis"] and items[plan.position]["analysis"][key] not in (None, args.gap_ms, args.gap_ms / 1000)
                for key in ("gap_ms", "min_gap_ms", "gap_before_next_ms")
            )
        )
    ]
    sizes = [{"cue_id": plan.cue_id, "source_index": items[plan.position].get("index"), "actual_chars": len(plan.text)} for plan in plans if plan.cue_id in target_ids]
    deepseek_limits = {
        "requested_override": args.max_tokens,
        "effective_by_mode": {
            mode: resolve_deepseek_token_limit(mode, args.max_tokens)
            for mode in ("off", "low", "high", "max")
        },
        "limit_policy_version": DEEPSEEK_LIMIT_POLICY_VERSION,
        "project_conservative_ceiling": DEEPSEEK_PROJECT_MAX_TOKENS,
        "provider_advertised_cap_label": DEEPSEEK_PROVIDER_ADVERTISED_CAP_LABEL,
        "project_conservative_ceiling_note": "Conservative project ceiling; exact provider maximum is not independently verified.",
        "model_requested": requested_deepseek_model,
        "provider_declared_version": PRICE_SNAPSHOTS[CURRENT_PRICE_SNAPSHOT_ID]["provider_declared_version"],
        "provider_version_attested": False,
        "reasoning_effort_note": "Requested variant is recorded; provider response does not attest internal reasoning effort.",
    }
    selected_generators = [
        asdict_model(
            ModelSpec(spec.provider, requested_deepseek_model, spec.variant)
            if spec.provider == "deepseek" else spec
        )
        for spec in (GENERATOR_CHOICES[value] for value in (args.generators or sorted(GENERATOR_CHOICES)))
    ]
    result = {
        "schema_version": 2,
        "status": "planned",
        "created_at": _utc_now(),
        "input": str(args.input), "absolute_input": str(source), "sha256": _source_hash(args.input),
        "input_hash": _source_hash(args.input),
        "dry_run": not args.execute,
        "seed": args.seed, "per_severity": args.per_severity, "short_count": args.short_count,
        "selection": {"selected_ids": target_ids, "selected_count": len(target_ids),
                       "status_counts": {status: sum(plan.status == status for plan in plans) for status in {plan.status for plan in plans}},
                       "severity_counts": {severity: sum(plan.severity == severity for plan in plans) for severity in {plan.severity for plan in plans}}},
        "target_ids": target_ids, "sample_actual_sizes": sizes,
        "plans": [asdict_plan(plan) for plan in plans],
        "generators": selected_generators,
        "arbiter": asdict_model(ModelSpec("opencode", "openai/gpt-5.6-sol", "medium")),
        "repetitions": args.repetitions,
        "run_config": {"gen_batch_size": args.generation_batch_size, "review_batch_size": args.review_batch_size,
                        "max_repairs": args.max_repairs, "length_tools": args.length_tools,
                         "max_tokens": args.max_tokens, "timeout": args.timeout,
                         "max_tokens_requested": args.max_tokens,
                         "effective_max_tokens_by_mode": deepseek_limits["effective_by_mode"],
                         "limit_policy_version": DEEPSEEK_LIMIT_POLICY_VERSION,
                          "project_conservative_ceiling": DEEPSEEK_PROJECT_MAX_TOKENS,
                          "provider_advertised_cap_label": DEEPSEEK_PROVIDER_ADVERTISED_CAP_LABEL},
        "planner": {"cps": args.cps, "gap_ms": args.gap_ms, "max_speed": args.max_speed,
                    "cps_note": "planned text heuristic includes spaces; it is not measured audio CPS"},
        "prompt_version": PROMPT_VERSION,
        "prompt_hashes": {"generator_system": hashlib.sha256(GENERATOR_SYSTEM_PROMPT.encode()).hexdigest(),
                          "arbiter_system": hashlib.sha256(ARBITER_SYSTEM_PROMPT.encode()).hexdigest()},
        "saved_prompts": {"generator_system": GENERATOR_SYSTEM_PROMPT, "arbiter_system": ARBITER_SYSTEM_PROMPT},
        "pricing": {"snapshot_id": CURRENT_PRICE_SNAPSHOT_ID, "date": PRICE_SNAPSHOT_DATE,
                     "source": PRICE_SOURCE, "table": DEEPSEEK_RATES_USD_PER_MILLION,
                     "note": PRICE_SNAPSHOTS[CURRENT_PRICE_SNAPSHOT_ID]["note"]},
        "deepseek_limits": deepseek_limits,
        "warnings": {"duplicate_source_indices": duplicate_indices,
                      "duplicate_source_index_count": len(duplicate_indices),
                      "duplicate_source_entry_count": duplicate_entries,
                      "stale_analysis_cue_ids": stale_analysis,
                      "stale_analysis_count": len(stale_analysis),
                      "analysis_warning": "Existing analysis may use a different gap policy (for example 300 ms versus the new 200 ms); plans are recomputed." if stale_analysis else None},
    }
    if timing_profile is not None:
        result.update({
            "timing_profile": timing_profile,
            "timing_profile_path": str(args.timing_profile),
            "timing_profile_hash": _source_hash(args.timing_profile),
        })
    if selection_manifest is not None:
        result.update({
            "selection_manifest_path": selection_manifest["path"],
            "selection_manifest_hash": selection_manifest["sha256"],
        })
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or run accountable subtitle shortening trials")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="opt in to live model calls")
    parser.add_argument("--per-severity", type=int, default=4)
    parser.add_argument("--short-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--generator", action="append", choices=sorted(GENERATOR_CHOICES), dest="generators")
    parser.add_argument("--generation-batch-size", type=int, default=6)
    parser.add_argument("--review-batch-size", type=int, default=12)
    parser.add_argument("--max-repairs", type=int, default=1,
                        help="shared extra recovery-round budget (0..2) for generation and review")
    parser.add_argument("--length-tools", action="store_true")
    parser.add_argument("--cps", type=float, default=13.0)
    parser.add_argument("--gap-ms", type=int, default=200)
    parser.add_argument("--max-speed", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--deepseek-model", choices=DEEPSEEK_MODEL_CHOICES, default=DEEPSEEK_CURRENT_MODEL)
    parser.add_argument("--timing-profile", type=Path, help="apply an explicit measured-audio diagnostic profile")
    parser.add_argument("--selection-manifest", type=Path, help="freeze IDs from a prior selection manifest")
    parser.add_argument("--preflight-model-path", type=Path, help="opt-in Vosk model used for original timing preflight")
    parser.add_argument("--preflight-voice", type=int, default=8)
    parser.add_argument("--preflight-text-mode", choices=list(TEXT_MODES), default="punctuation")
    return parser


async def _execute(args: argparse.Namespace, items: list[dict[str, Any]], plans: list[Any], target_ids: list[str]) -> int:
    generators = [
        ModelSpec(spec.provider, args.deepseek_model, spec.variant) if spec.provider == "deepseek" else spec
        for spec in (GENERATOR_CHOICES[value] for value in (args.generators or sorted(GENERATOR_CHOICES)))
    ]
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not target_ids:
        manifest.update({"status": "complete", "dry_run": False, "completed_at": _utc_now(), "completed_trials": 0})
        _write(manifest_path, manifest)
        _write(args.output_dir / "experiment_summary.json", _summary_payload(args, target_ids, [], status="complete"))
        return 0
    manifest.update({"status": "running", "dry_run": False, "started_at": _utc_now()})
    _write(manifest_path, manifest)
    planner = PlannerConfig(cps=args.cps, max_speed=args.max_speed, gap_ms=args.gap_ms)
    if args.preflight_model_path is not None:
        if not args.preflight_model_path.exists() or not args.preflight_model_path.is_dir():
            raise FileNotFoundError(f"preflight model path does not exist or is not a directory: {args.preflight_model_path}")
    deepseek_needed = any(spec.provider == "deepseek" for spec in generators)
    if deepseek_needed:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
    server: CleanOpenCodeServer | None = None
    backend: LiveBackend
    try:
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY") if deepseek_needed else None
        if deepseek_needed and not deepseek_key:
            raise RuntimeError("DeepSeek API key preflight failed")
        variants = {spec.variant for spec in generators if spec.provider == "opencode"}
        required = {"openai/gpt-5.6-sol": {"medium"}}
        if variants:
            required["openai/gpt-5.6-luna"] = variants
        server = CleanOpenCodeServer(enable_length_tool=args.length_tools, required_models=required, timeout=args.timeout)
        backend = LiveBackend(server, deepseek_api_key=deepseek_key,
                              timeout=args.timeout, max_tokens=args.max_tokens)
        manifest["runtime_metadata"] = backend.runtime_metadata
        _write(manifest_path, manifest)
        inherited_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            server.start()
        finally:
            if inherited_key is not None:
                os.environ["DEEPSEEK_API_KEY"] = inherited_key
        config = RunConfig(gen_batch_size=args.generation_batch_size, review_batch_size=args.review_batch_size,
                           max_repairs=args.max_repairs, length_tools=args.length_tools)
        arbiter = ModelSpec("opencode", "openai/gpt-5.6-sol", "medium")
        original_timing_verifier = (
            functools.partial(
                verify_original_timing,
                model_path=args.preflight_model_path,
                voice=args.preflight_voice,
                speech_rate=planner.max_speed,
                text_mode=args.preflight_text_mode,
            )
            if args.preflight_model_path is not None
            else None
        )
        reports: list[dict[str, Any]] = []
        for generator in generators:
            for repetition in range(1, args.repetitions + 1):
                trial_dir = args.output_dir / _model_key(generator) / f"r{repetition}"
                report = await run_trial(items=copy.deepcopy(items), plans=plans, target_ids=target_ids,
                                         planner_config=planner, run_config=config, generator=generator,
                                         arbiter=arbiter, output_dir=trial_dir, request_fn=backend,
                                         original_timing_verifier=original_timing_verifier)
                report["source_manifest"].update({
                    "path": str(args.input), "absolute_path": str(args.input.resolve()),
                    "sha256": _source_hash(args.input), "parent_manifest": str(manifest_path.resolve()),
                })
                if args.timing_profile is not None:
                    report["timing_profile"] = {
                        "path": str(args.timing_profile),
                        "sha256": _source_hash(args.timing_profile),
                    }
                if args.selection_manifest is not None:
                    report["selection_manifest"] = {
                        "path": str(args.selection_manifest),
                        "sha256": _source_hash(args.selection_manifest),
                    }
                direct = report.get("cost", {}).get("direct", {})
                repairs = report.get("summary", {}).get("repairs", 0)
                selected = report.get("summary", {}).get("selected_count", len(target_ids))
                report.setdefault("cost", {})["projection_500"] = project_500(
                    selected_count=selected,
                    tokens={key: direct.get(key) for key in ("input_tokens", "output_tokens", "reasoning_tokens")},
                    known_cost_usd=direct.get("known_subtotal_usd"),
                    request_counts={"observed": len(report.get("cost", {}).get("stages", {}))},
                    observed_repair_fraction=(repairs / selected if selected else 0.0),
                    accounting_complete=bool(direct.get("accounting_complete", False)),
                    gen_batch_size=args.generation_batch_size,
                    review_batch_size=args.review_batch_size,
                )
                _write(trial_dir / "report.json", report)
                reports.append({"model": asdict_model(generator), "repetition": repetition,
                                "summary": report["summary"], "cost": report["cost"]})
                _write(args.output_dir / "experiment_summary.json", _summary_payload(args, target_ids, reports, status="running"))
                manifest["completed_trials"] = len(reports)
                _write(manifest_path, manifest)
        summary = _summary_payload(args, target_ids, reports, status="complete")
        _write(args.output_dir / "experiment_summary.json", summary)
        manifest.update({"status": "complete", "completed_at": _utc_now(), "completed_trials": len(reports)})
        _write(manifest_path, manifest)
        return 0
    except BaseException as exc:
        manifest.update({"status": "error", "error_type": _error_type(exc), "completed_trials": len(locals().get("reports", []))})
        _write(manifest_path, manifest)
        if locals().get("reports") is not None:
            _write(args.output_dir / "experiment_summary.json", _summary_payload(args, target_ids, locals()["reports"], status="error"))
        raise
    finally:
        if server is not None:
            server.stop()


def asdict_model(spec: ModelSpec) -> dict[str, str]:
    return {"provider": spec.provider, "model": spec.model, "variant": spec.variant}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_args(args)
        if args.output_dir.exists():
            raise FileExistsError(f"output directory already exists: {args.output_dir}")
        items = _load_items(args.input)
        planner = PlannerConfig(cps=args.cps, max_speed=args.max_speed, gap_ms=args.gap_ms)
        plans = plan_subtitles(items, planner)
        source_hash = _source_hash(args.input)
        timing_profile: dict[str, Any] | None = None
        if args.timing_profile is not None:
            timing_profile = _load_json_object(args.timing_profile, "timing profile")
            plans = apply_timing_profile(plans, timing_profile, source_sha256=source_hash, config=planner)
        selection_metadata: dict[str, str] | None = None
        if args.selection_manifest is not None:
            target_ids = _frozen_selection(args.selection_manifest, plans, args, source_hash)
            selection_metadata = {"path": str(args.selection_manifest), "sha256": _source_hash(args.selection_manifest)}
        else:
            target_ids = _selection(plans, args)
        manifest = _manifest(items, plans, target_ids, args,
                             timing_profile=timing_profile, selection_manifest=selection_metadata)
        args.output_dir.mkdir(parents=True)
        _write(args.output_dir / "manifest.json", manifest)
        _write(args.output_dir / "plan.json", [asdict_plan(plan) for plan in plans])
        if not args.execute:
            manifest.update({"status": "complete", "completed_at": _utc_now(), "completed_trials": 0})
            _write(args.output_dir / "manifest.json", manifest)
            print(json.dumps(manifest["selection"], ensure_ascii=False, sort_keys=True))
            return 0
        from utils.shortening_html import live_html_report
        with live_html_report(args.output_dir):
            return asyncio.run(_execute(args, items, plans, target_ids))
    except Exception as exc:
        print(f"shortening failed: {_error_type(exc)}", file=sys.stderr)
        return 1


def asdict_plan(plan: Any) -> dict[str, Any]:
    return {field.name: getattr(plan, field.name) for field in __import__("dataclasses").fields(plan)}


if __name__ == "__main__":
    raise SystemExit(main())
