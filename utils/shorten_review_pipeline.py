"""Run Flash shortening and union-target Pro review as one pipeline."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from utils.analyze_text import join_text_lines
from utils.review_shortened_subtitles_deepseek import review_subtitles
from utils.shorten_helpers import (calculate_budget, hydrate_context_timing, load_api_key,
                                   load_json, refresh_analysis, run_iterative_shortening,
                                   save_json, validate_context_source)
from utils.shorten_subtitles_deepseek import make_deepseek_shorten_func, usage_summary
from utils.review_shortened_subtitles_deepseek import duration_profile_metadata
from utils.shortening_domain import CalibratedDurationModel, DurationSelectionPolicy, load_calibrated_duration_profile
from utils.verify_subtitles_opencode import verify_items_live

Items = List[Dict[str, Any]]


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    output_dir: Path
    api_key: str
    context_source: Optional[Path] = None
    flash_model: str = "deepseek-v4-flash"
    pro_model: str = "deepseek-v4-pro"
    threshold: float = 1.5
    avg_chars_per_sec: float = 13.0
    flash_max_iterations: int = 1
    flash_batch_size: int = 6
    flash_concurrency: int = 3
    pro_thinking_mode: str = "auto"
    pro_batch_size: int = 4
    pro_concurrency: int = 3
    pro_risk_threshold: float = 35.0
    pro_context_window: int = 3
    pro_max_input_tokens: int = 50_000
    base_url: str = "https://api.deepseek.com"
    duration_profile: Optional[CalibratedDurationModel] = None
    duration_fit_ratio: float = 1.0
    semantic_barrier_enabled: bool = False
    semantic_barrier_units_enabled: bool = True
    semantic_barrier_unit_max_cues: int = 3
    semantic_barrier_unit_max_gap_sec: float = 0.3
    semantic_barrier_model: str = "openai/gpt-5.6-luna"
    semantic_barrier_server_url: Optional[str] = None
    semantic_barrier_hostname: str = "127.0.0.1"
    semantic_barrier_port: Optional[int] = None
    semantic_barrier_batch_size: int = 4
    semantic_barrier_concurrency: int = 1
    semantic_barrier_context_window: int = 3
    semantic_barrier_transport_retries: int = 1
    semantic_barrier_schema_retries: int = 1
    semantic_barrier_structured_output: bool = False


FlashStage = Callable[[Items, PipelineConfig, Path, str], tuple[Items, Dict[str, Any]]]
ProStage = Callable[[Items, Items, PipelineConfig], tuple[Items, Dict[str, Any], Dict[str, Any]]]
SemanticBarrierStage = Callable[[Items, Items, Optional[Items], PipelineConfig], tuple[Items, Dict[str, Any], Dict[str, Any]]]


class PipelineStageError(RuntimeError):
    """Indicate that a stage failed after prior artifacts were persisted."""


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _selection_policy(config: PipelineConfig) -> Optional[DurationSelectionPolicy]:
    if config.duration_profile is None:
        return None
    return DurationSelectionPolicy(config.duration_profile, config.duration_fit_ratio)


def _critical_count(items: Items, config: PipelineConfig) -> int:
    analyzed = refresh_analysis(
        copy.deepcopy(items), config.avg_chars_per_sec, config.threshold,
        preserve_saved_timing=config.context_source is not None,
    )
    return sum(_is_critical(item, config.threshold, config.duration_profile, config.duration_fit_ratio,
                             config.avg_chars_per_sec)
               for item in analyzed["items"])


def _carry_stable_timing(source: Items, targets: Items) -> Items:
    """Keep target text/schema while carrying timing captured from source."""
    source_by_index = {item.get("index"): item for item in source}
    carried = copy.deepcopy(targets)
    for target in carried:
        source_item = source_by_index.get(target.get("index"))
        if source_item is None:
            continue
        source_analysis = source_item.get("analysis", {})
        target_analysis = target.setdefault("analysis", {})
        for field in ("available_duration_sec", "effective_duration_sec", "is_checked"):
            if field in source_analysis:
                target_analysis[field] = copy.deepcopy(source_analysis[field])
    return carried


def _changed_count(original: Items, current: Items) -> int:
    original_text = {item.get("index"): join_text_lines(item.get("text", "")).strip() for item in original}
    return sum(join_text_lines(item.get("text", "")).strip() != original_text.get(item.get("index"))
               for item in current)


def _is_critical(item: Dict[str, Any], threshold: float,
                 duration_profile: Optional[CalibratedDurationModel] = None,
                 duration_fit_ratio: float = 1.0,
                 avg_chars_per_sec: float = 13.0) -> bool:
    analysis = item.get("analysis", {})
    if duration_profile is not None:
        budget = calculate_budget(item, threshold, avg_chars_per_sec)
        return bool(analysis.get("is_checked") and budget.effective_duration_sec > 0
                    and duration_profile.predict_seconds(join_text_lines(item.get("text", "")))
                    > budget.effective_duration_sec * duration_fit_ratio)
    extended = analysis.get("extended_mismatch_ratio")
    mismatch = analysis.get("mismatch_ratio")
    effective = extended if isinstance(extended, (int, float)) else mismatch
    return bool(analysis.get("is_checked", False) and isinstance(effective, (int, float))
                and effective > threshold)


def _mark_shortening(original: Items, flash: Items, final: Items,
                       pro_report: Dict[str, Any], config: PipelineConfig,
                       barrier_report: Optional[Dict[str, Any]] = None) -> tuple[List[int], Items]:
    original_by_index = {item.get("index"): item for item in original}
    flash_by_index = {item.get("index"): item for item in flash}
    selected = set(pro_report.get("selected_indices", []))
    outcomes = {value.get("index"): value for value in pro_report.get("outcomes", [])}
    barrier_selected = set(barrier_report.get("selected_indices", [])) if barrier_report else set()
    barrier_verified = set(barrier_report.get("verified_indices", [])) if barrier_report else set()
    barrier_targets = {value.get("index"): value for value in barrier_report.get("targets", [])} if barrier_report else {}
    unresolved: List[int] = []
    for item in final:
        index = item.get("index")
        source = original_by_index[index]
        flash_item = flash_by_index[index]
        original_text = join_text_lines(source.get("text", "")).strip()
        flash_text = join_text_lines(flash_item.get("text", "")).strip()
        final_text = join_text_lines(item.get("text", "")).strip()
        budget = calculate_budget(item, config.threshold, config.avg_chars_per_sec)
        required = (_is_critical(source, config.threshold, config.duration_profile, config.duration_fit_ratio,
                                 config.avg_chars_per_sec) or
                    len(original_text) > calculate_budget(
                        source, config.threshold, config.avg_chars_per_sec
                    ).max_chars)
        outcome = outcomes.get(index, {})
        failed_review = index in selected and outcome.get("outcome") not in {
            "verified", "accepted", "rewrite"
        }
        over_budget = len(final_text) > budget.max_chars
        over_duration = (required and config.duration_profile is not None and budget.effective_duration_sec > 0
                         and config.duration_profile.predict_seconds(final_text)
                         > budget.effective_duration_sec * config.duration_fit_ratio)
        barrier_fallback = barrier_targets.get(index, {}).get("fallback")
        if barrier_fallback:
            status, reason = "unresolved", ("semantic_barrier_rejected" if barrier_fallback in {"verdict_fail", "verdict_uncertain"}
                                             else "semantic_barrier_fallback")
        elif len(final_text.split()) <= 1 and over_budget:
            status, reason = "unresolved", "minimum_text_exceeds_budget"
        elif failed_review:
            error = str(outcome.get("error", ""))
            quality_failure = error == "decision_unresolved" or error.startswith("quality_flags:")
            status, reason = "unresolved", "quality_unresolved" if quality_failure else "review_fallback"
        elif over_duration:
            status, reason = "unresolved", "still_over_duration"
        elif over_budget:
            status, reason = "unresolved", "still_over_budget"
        elif required and index not in selected:
            status, reason = "unresolved", "review_selection_missing"
        elif required:
            status, reason = "resolved", "verified_within_budget"
        else:
            status, reason = "not_required", "within_budget"
        if status == "unresolved":
            unresolved.append(int(index))
        shortening = {"required_initially": required,
                              "flash_changed": flash_text != original_text,
                              "pro_selected": index in selected,
                              "max_chars": budget.max_chars, "final_chars": len(final_text),
                              "status": status, "reason": reason}
        if barrier_report is not None:
            shortening["semantic_barrier_selected"] = index in barrier_selected
            shortening["semantic_barrier_verified"] = index in barrier_verified
        item["shortening"] = shortening
    return unresolved, final


def _default_flash_stage(items: Items, config: PipelineConfig, output_dir: Path,
                         stem: str) -> tuple[Items, Dict[str, Any]]:
    usage: Dict[str, int] = {}
    shorten = make_deepseek_shorten_func(
        config.api_key, config.base_url, config.flash_concurrency, usage
    )
    output = run_iterative_shortening(
        items, shorten, config.flash_model, config.threshold, 3,
        config.flash_max_iterations, 3, output_dir, stem,
        reasoning_effort=None, avg_chars_per_sec=config.avg_chars_per_sec,
        batch_size=config.flash_batch_size,
        context_items=load_json(config.context_source) if config.context_source else None,
        selection_policy=_selection_policy(config),
    )
    summary = usage_summary(usage)
    if config.duration_profile is not None:
        summary["selection_mode"] = "calibrated_post_silence_duration"
        summary["duration_profile"] = duration_profile_metadata(config.duration_profile, config.duration_fit_ratio)
    return output, summary


def _default_pro_stage(original: Items, shortened: Items,
                       config: PipelineConfig) -> tuple[Items, Dict[str, Any], Dict[str, Any]]:
    return asyncio.run(review_subtitles(
        original, shortened, config.api_key, config.pro_model,
        config.pro_thinking_mode, config.pro_risk_threshold,
        config.pro_context_window, config.pro_batch_size,
        config.pro_max_input_tokens, 3, config.avg_chars_per_sec,
        config.threshold, config.base_url,
        concurrency=config.pro_concurrency,
        enable_planner=True,
        context_source=load_json(config.context_source) if config.context_source else None,
        duration_profile=config.duration_profile,
        duration_fit_ratio=config.duration_fit_ratio,
    ))


def _default_semantic_barrier_stage(original: Items, candidate: Items,
                                    context_items: Optional[Items],
                                    config: PipelineConfig) -> tuple[Items, Dict[str, Any], Dict[str, Any]]:
    return verify_items_live(
        original, candidate, model=config.semantic_barrier_model, context_items=context_items,
        context_window=config.semantic_barrier_context_window,
        batch_size=config.semantic_barrier_batch_size, concurrency=config.semantic_barrier_concurrency,
        transport_retries=config.semantic_barrier_transport_retries,
        schema_retries=config.semantic_barrier_schema_retries,
        semantic_units=config.semantic_barrier_units_enabled,
        semantic_unit_max_cues=config.semantic_barrier_unit_max_cues,
        semantic_unit_max_gap_sec=config.semantic_barrier_unit_max_gap_sec,
        review_report=None, server_url=config.semantic_barrier_server_url,
        hostname=config.semantic_barrier_hostname, port=config.semantic_barrier_port,
        structured_output=config.semantic_barrier_structured_output,
    )


def _validate_semantic_barrier_config(config: PipelineConfig) -> None:
    if not config.semantic_barrier_enabled:
        return
    if not config.semantic_barrier_model or config.semantic_barrier_context_window < 0:
        raise ValueError("invalid semantic barrier configuration")
    if (isinstance(config.semantic_barrier_unit_max_cues, bool)
            or not isinstance(config.semantic_barrier_unit_max_cues, int)
            or not 1 <= config.semantic_barrier_unit_max_cues <= 5):
        raise ValueError("semantic_barrier_unit_max_cues must be an integer between 1 and 5")
    if (isinstance(config.semantic_barrier_unit_max_gap_sec, bool)
            or not isinstance(config.semantic_barrier_unit_max_gap_sec, (int, float))
            or not math.isfinite(float(config.semantic_barrier_unit_max_gap_sec))
            or config.semantic_barrier_unit_max_gap_sec < 0):
        raise ValueError("semantic_barrier_unit_max_gap_sec must be finite and non-negative")
    if not 1 <= config.semantic_barrier_batch_size <= 100 or not 1 <= config.semantic_barrier_concurrency <= 100:
        raise ValueError("invalid semantic barrier configuration")
    if not 0 <= config.semantic_barrier_transport_retries <= 5 or config.semantic_barrier_schema_retries not in {0, 1}:
        raise ValueError("invalid semantic barrier configuration")
    if config.semantic_barrier_server_url is not None and not config.semantic_barrier_server_url.strip():
        raise ValueError("server_url_required")
    if config.semantic_barrier_server_url is None and config.semantic_barrier_hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("auto-start hostname must be localhost or 127.0.0.1")
    if config.semantic_barrier_port is not None and not 1 <= config.semantic_barrier_port <= 65535:
        raise ValueError("port must be between 1 and 65535")


def run_pipeline(config: PipelineConfig, flash_stage: FlashStage = _default_flash_stage,
                 pro_stage: ProStage = _default_pro_stage,
                 semantic_barrier_stage: SemanticBarrierStage = _default_semantic_barrier_stage) -> Path:
    """Run both stages, persist artifacts, and return the reviewed final path."""
    _selection_policy(config)
    _validate_semantic_barrier_config(config)
    original = load_json(config.input_path)
    context_items = load_json(config.context_source) if config.context_source else None
    if context_items is not None:
        validate_context_source(original, context_items)
        original = hydrate_context_timing(
            original, context_items, config.avg_chars_per_sec, config.threshold
        )
    original_for_pro = copy.deepcopy(original)
    stem = config.input_path.stem
    flash_dir = config.output_dir / "flash"
    pro_dir = config.output_dir / "pro"
    flash_usage_path = flash_dir / f"{stem}.usage.json"
    pro_usage_path = pro_dir / f"{stem}.usage.json"
    pro_report_path = pro_dir / f"{stem}.report.json"
    pipeline_usage_path = config.output_dir / f"{stem}.pipeline.usage.json"
    pipeline_report_path = config.output_dir / f"{stem}.pipeline.report.json"
    final_path = config.output_dir / f"{stem}_reviewed.json"
    barrier_dir = config.output_dir / "semantic_barrier"
    prebarrier_path = barrier_dir / f"{stem}_reviewed_prebarrier.json"
    barrier_path = barrier_dir / f"{stem}_reviewed_independent_verified.json"
    barrier_report_path = barrier_dir / f"{stem}_reviewed_independent_verified.report.json"
    barrier_usage_path = barrier_dir / f"{stem}_reviewed_independent_verified.usage.json"

    critical_start = _critical_count(original, config)
    print("Flash stage: starting", flush=True)
    flash_output, flash_usage = flash_stage(copy.deepcopy(original), config, flash_dir, stem)
    if context_items is not None:
        flash_output = _carry_stable_timing(original, flash_output)
    print("Flash stage: completed", flush=True)
    flash_final_path = flash_dir / f"{stem}_final.json"
    if not flash_final_path.exists():
        save_json(flash_final_path, flash_output)
    _write_json(flash_usage_path, flash_usage)
    flash_changed = _changed_count(original, flash_output)
    flash_critical = _critical_count(flash_output, config)
    base_report: Dict[str, Any] = {
        "status": "pro_pending", "total": len(original),
        "context_mode": "full" if config.context_source else "sparse",
        "context_source": str(config.context_source) if config.context_source else None,
        "flash": {"critical_start": critical_start, "changed": flash_changed,
                  "remaining_critical": flash_critical},
        "paths": {"flash_final": str(flash_final_path), "flash_usage": str(flash_usage_path),
                  "final": str(final_path), "pro_report": str(pro_report_path),
                   "pro_usage": str(pro_usage_path)},
    }
    if config.duration_profile is not None:
        base_report["duration_profile"] = duration_profile_metadata(
            config.duration_profile, config.duration_fit_ratio
        )
    try:
        print("Pro stage: starting", flush=True)
        reviewed, pro_report, pro_usage = pro_stage(
            copy.deepcopy(original_for_pro), copy.deepcopy(flash_output), config
        )
        print("Pro stage: completed", flush=True)
    except Exception as exc:
        print(f"Pro stage: error: {type(exc).__name__}: {exc}", flush=True)
        base_report.update({"status": "pro_failed", "error": f"{type(exc).__name__}: {exc}"})
        _write_json(pipeline_report_path, base_report)
        raise PipelineStageError(f"Pro review failed; Flash result kept at {flash_final_path}") from exc

    if context_items is not None:
        reviewed = _carry_stable_timing(original, reviewed)
    _write_json(pro_usage_path, pro_usage)
    _write_json(pro_report_path, pro_report)
    base_report["pro"] = pro_report
    barrier_report: Optional[Dict[str, Any]] = None
    barrier_usage: Optional[Dict[str, Any]] = None
    if config.semantic_barrier_enabled:
        save_json(prebarrier_path, reviewed)
        base_report["paths"]["semantic_barrier_prebarrier"] = str(prebarrier_path)
        try:
            reviewed, barrier_report, barrier_usage = semantic_barrier_stage(
                copy.deepcopy(original), copy.deepcopy(reviewed),
                copy.deepcopy(context_items) if context_items is not None else None, config
            )
            save_json(barrier_path, reviewed)
            base_report["paths"]["semantic_barrier"] = str(barrier_path)
            _write_json(barrier_report_path, barrier_report)
            base_report["paths"]["semantic_barrier_report"] = str(barrier_report_path)
            _write_json(barrier_usage_path, barrier_usage)
            base_report["paths"]["semantic_barrier_usage"] = str(barrier_usage_path)
        except Exception as exc:
            base_report.update({"status": "semantic_barrier_failed",
                                "error": f"{type(exc).__name__}: {exc}"})
            _write_json(pipeline_report_path, base_report)
            raise PipelineStageError(
                f"Semantic barrier failed; Pro result is kept at prebarrier path {prebarrier_path}"
            ) from exc
    refreshed = refresh_analysis(
        reviewed, config.avg_chars_per_sec, config.threshold,
        preserve_saved_timing=config.context_source is not None,
    )
    unresolved_indices, final_items = _mark_shortening(
        original, flash_output, refreshed["items"], pro_report, config, barrier_report
    )
    save_json(final_path, final_items)
    combined_usage = {
        "estimated_cost_usd": round(float(flash_usage.get("estimated_cost_usd", 0)) +
                                    float(pro_usage.get("estimated_cost_usd", 0)), 8),
        "flash": flash_usage, "pro": pro_usage,
    }
    if barrier_report is not None and barrier_usage is not None:
        combined_usage["semantic_barrier"] = barrier_usage
    if config.duration_profile is not None:
        combined_usage["duration_profile"] = duration_profile_metadata(
            config.duration_profile, config.duration_fit_ratio
        )
    _write_json(pipeline_usage_path, combined_usage)
    union_unresolved = sorted(set(unresolved_indices) | set(barrier_report.get("unresolved_indices", [])) if barrier_report else set(unresolved_indices))
    base_report.update({
        "status": "completed_with_unresolved" if union_unresolved else "completed", "pro": pro_report,
        "final_critical": _critical_count(refreshed["items"], config),
        "unresolved_count": len(union_unresolved), "unresolved_indices": union_unresolved,
        "paths": {**base_report["paths"], "pipeline_usage": str(pipeline_usage_path),
                  "pipeline_report": str(pipeline_report_path)},
    })
    if barrier_report is not None:
        base_report["semantic_barrier"] = barrier_report
        base_report["semantic_barrier_unresolved_indices"] = barrier_report.get("unresolved_indices", [])
        base_report["shortening_unresolved_indices"] = unresolved_indices
        base_report["status"] = "completed_with_unresolved" if union_unresolved else "completed"
    _write_json(pipeline_report_path, base_report)
    print(f"Final reviewed JSON: {final_path}")
    print(f"Stages: critical {critical_start} -> {flash_critical} -> "
          f"{_critical_count(refreshed['items'], config)}; Flash changed {flash_changed}; "
          f"Pro selected {pro_report.get('selected', 0)}, fallback {len(pro_report.get('fallback', []))}")
    return final_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shorten with Flash, then review changed and remaining-critical subtitles with Pro."
    )
    parser.add_argument("input")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--flash-model", default="deepseek-v4-flash")
    parser.add_argument("--pro-model", default="deepseek-v4-pro")
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--avg-chars-per-sec", type=float, default=13.0)
    parser.add_argument("--context-source")
    parser.add_argument("--flash-max-iterations", type=int, default=1)
    parser.add_argument("--flash-batch-size", type=int, default=6)
    parser.add_argument("--flash-concurrency", type=int, default=3)
    parser.add_argument(
        "--pro-thinking-mode", choices=["disabled", "high", "max", "auto"], default="auto",
        help="Deprecated compatibility option; fixed Pro stage routing is always enforced",
    )
    parser.add_argument("--pro-batch-size", type=int, default=4)
    parser.add_argument("--pro-concurrency", type=int, default=3)
    parser.add_argument(
        "--pro-risk-threshold", type=float, default=35.0,
        help="Deprecated compatibility option; fixed Pro stage routing is always enforced",
    )
    parser.add_argument("--pro-context-window", type=int, default=3)
    parser.add_argument("--pro-max-input-tokens", type=int, default=50_000)
    parser.add_argument("--duration-profile", type=Path)
    parser.add_argument("--duration-fit-ratio", type=float, default=1.0)
    parser.add_argument("--semantic-barrier", action="store_true")
    parser.add_argument("--semantic-barrier-units", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--semantic-barrier-unit-max-cues", type=int, default=3)
    parser.add_argument("--semantic-barrier-unit-max-gap-sec", type=float, default=0.3)
    parser.add_argument("--semantic-barrier-model", default="openai/gpt-5.6-luna")
    parser.add_argument("--semantic-barrier-server-url")
    parser.add_argument("--semantic-barrier-hostname", default="127.0.0.1")
    parser.add_argument("--semantic-barrier-port", type=int)
    parser.add_argument("--semantic-barrier-batch-size", type=int, default=4)
    parser.add_argument("--semantic-barrier-concurrency", type=int, default=1)
    parser.add_argument("--semantic-barrier-context-window", type=int, default=3)
    parser.add_argument("--semantic-barrier-transport-retries", type=int, default=1)
    parser.add_argument("--semantic-barrier-schema-retries", type=int, choices=[0, 1], default=1)
    parser.add_argument("--semantic-barrier-structured-output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", "-o", default="output/shorten_review")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if (isinstance(args.duration_fit_ratio, bool) or not math.isfinite(args.duration_fit_ratio)
                or args.duration_fit_ratio <= 0):
            raise ValueError("duration_fit_ratio must be finite and positive")
        duration_profile = (load_calibrated_duration_profile(args.duration_profile)
                            if args.duration_profile is not None else None)
        barrier_config = PipelineConfig(
            input_path=Path(args.input), output_dir=Path(args.output_dir), api_key="",
            semantic_barrier_enabled=args.semantic_barrier,
            semantic_barrier_units_enabled=args.semantic_barrier_units,
            semantic_barrier_unit_max_cues=args.semantic_barrier_unit_max_cues,
            semantic_barrier_unit_max_gap_sec=args.semantic_barrier_unit_max_gap_sec,
            semantic_barrier_model=args.semantic_barrier_model,
            semantic_barrier_server_url=args.semantic_barrier_server_url,
            semantic_barrier_hostname=args.semantic_barrier_hostname,
            semantic_barrier_port=args.semantic_barrier_port,
            semantic_barrier_batch_size=args.semantic_barrier_batch_size,
            semantic_barrier_concurrency=args.semantic_barrier_concurrency,
            semantic_barrier_context_window=args.semantic_barrier_context_window,
            semantic_barrier_transport_retries=args.semantic_barrier_transport_retries,
             semantic_barrier_schema_retries=args.semantic_barrier_schema_retries,
             semantic_barrier_structured_output=args.semantic_barrier_structured_output,
        )
        _validate_semantic_barrier_config(barrier_config)
    except ValueError as exc:
        print(f"ERROR: invalid pipeline configuration: {exc}", file=sys.stderr)
        return 2
    api_key = args.api_key or load_api_key("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DeepSeek API key not found.", file=sys.stderr)
        return 1
    config = PipelineConfig(
        input_path=Path(args.input), output_dir=Path(args.output_dir), api_key=api_key,
        context_source=Path(args.context_source) if args.context_source else None,
        flash_model=args.flash_model, pro_model=args.pro_model, threshold=args.threshold,
        avg_chars_per_sec=args.avg_chars_per_sec, flash_max_iterations=args.flash_max_iterations,
        flash_batch_size=args.flash_batch_size, flash_concurrency=args.flash_concurrency,
        pro_thinking_mode=args.pro_thinking_mode, pro_batch_size=args.pro_batch_size,
        pro_concurrency=args.pro_concurrency,
        pro_risk_threshold=args.pro_risk_threshold, pro_context_window=args.pro_context_window,
        pro_max_input_tokens=args.pro_max_input_tokens, base_url=args.base_url,
        duration_profile=duration_profile, duration_fit_ratio=args.duration_fit_ratio,
        semantic_barrier_enabled=args.semantic_barrier,
        semantic_barrier_units_enabled=args.semantic_barrier_units,
        semantic_barrier_unit_max_cues=args.semantic_barrier_unit_max_cues,
        semantic_barrier_unit_max_gap_sec=args.semantic_barrier_unit_max_gap_sec,
        semantic_barrier_model=args.semantic_barrier_model,
        semantic_barrier_server_url=args.semantic_barrier_server_url,
        semantic_barrier_hostname=args.semantic_barrier_hostname,
        semantic_barrier_port=args.semantic_barrier_port,
        semantic_barrier_batch_size=args.semantic_barrier_batch_size,
        semantic_barrier_concurrency=args.semantic_barrier_concurrency,
        semantic_barrier_context_window=args.semantic_barrier_context_window,
        semantic_barrier_transport_retries=args.semantic_barrier_transport_retries,
         semantic_barrier_schema_retries=args.semantic_barrier_schema_retries,
         semantic_barrier_structured_output=args.semantic_barrier_structured_output,
     )
    try:
        run_pipeline(config)
    except PipelineStageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
