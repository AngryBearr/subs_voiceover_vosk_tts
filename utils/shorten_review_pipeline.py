"""Run Flash shortening and union-target Pro review as one pipeline."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
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


FlashStage = Callable[[Items, PipelineConfig, Path, str], tuple[Items, Dict[str, Any]]]
ProStage = Callable[[Items, Items, PipelineConfig], tuple[Items, Dict[str, Any], Dict[str, Any]]]


class PipelineStageError(RuntimeError):
    """Indicate that a stage failed after prior artifacts were persisted."""


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _critical_count(items: Items, config: PipelineConfig) -> int:
    analyzed = refresh_analysis(
        copy.deepcopy(items), config.avg_chars_per_sec, config.threshold,
        preserve_saved_timing=config.context_source is not None,
    )
    return len(analyzed["critical_items"])


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


def _is_critical(item: Dict[str, Any], threshold: float) -> bool:
    analysis = item.get("analysis", {})
    extended = analysis.get("extended_mismatch_ratio")
    mismatch = analysis.get("mismatch_ratio")
    effective = extended if isinstance(extended, (int, float)) else mismatch
    return bool(analysis.get("is_checked", False) and isinstance(effective, (int, float))
                and effective > threshold)


def _mark_shortening(original: Items, flash: Items, final: Items,
                      pro_report: Dict[str, Any], config: PipelineConfig) -> tuple[List[int], Items]:
    original_by_index = {item.get("index"): item for item in original}
    flash_by_index = {item.get("index"): item for item in flash}
    selected = set(pro_report.get("selected_indices", []))
    outcomes = {value.get("index"): value for value in pro_report.get("outcomes", [])}
    unresolved: List[int] = []
    for item in final:
        index = item.get("index")
        source = original_by_index[index]
        flash_item = flash_by_index[index]
        original_text = join_text_lines(source.get("text", "")).strip()
        flash_text = join_text_lines(flash_item.get("text", "")).strip()
        final_text = join_text_lines(item.get("text", "")).strip()
        budget = calculate_budget(item, config.threshold, config.avg_chars_per_sec)
        required = (_is_critical(source, config.threshold) or
                    len(original_text) > calculate_budget(
                        source, config.threshold, config.avg_chars_per_sec
                    ).max_chars)
        outcome = outcomes.get(index, {})
        failed_review = index in selected and outcome.get("outcome") not in {
            "verified", "accepted", "rewrite"
        }
        over_budget = len(final_text) > budget.max_chars
        if len(final_text.split()) <= 1 and over_budget:
            status, reason = "unresolved", "minimum_text_exceeds_budget"
        elif failed_review:
            error = str(outcome.get("error", ""))
            quality_failure = error == "decision_unresolved" or error.startswith("quality_flags:")
            status, reason = "unresolved", "quality_unresolved" if quality_failure else "review_fallback"
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
        item["shortening"] = {"required_initially": required,
                              "flash_changed": flash_text != original_text,
                              "pro_selected": index in selected,
                              "max_chars": budget.max_chars, "final_chars": len(final_text),
                              "status": status, "reason": reason}
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
    )
    return output, usage_summary(usage)


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
    ))


def run_pipeline(config: PipelineConfig, flash_stage: FlashStage = _default_flash_stage,
                 pro_stage: ProStage = _default_pro_stage) -> Path:
    """Run both stages, persist artifacts, and return the reviewed final path."""
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
    refreshed = refresh_analysis(
        reviewed, config.avg_chars_per_sec, config.threshold,
        preserve_saved_timing=config.context_source is not None,
    )
    unresolved_indices, final_items = _mark_shortening(
        original, flash_output, refreshed["items"], pro_report, config
    )
    save_json(final_path, final_items)
    _write_json(pro_usage_path, pro_usage)
    _write_json(pro_report_path, pro_report)
    combined_usage = {
        "estimated_cost_usd": round(float(flash_usage.get("estimated_cost_usd", 0)) +
                                    float(pro_usage.get("estimated_cost_usd", 0)), 8),
        "flash": flash_usage, "pro": pro_usage,
    }
    _write_json(pipeline_usage_path, combined_usage)
    base_report.update({
        "status": "completed_with_unresolved" if unresolved_indices else "completed", "pro": pro_report,
        "final_critical": len(refreshed["critical_items"]),
        "unresolved_count": len(unresolved_indices), "unresolved_indices": unresolved_indices,
        "paths": {**base_report["paths"], "pipeline_usage": str(pipeline_usage_path),
                  "pipeline_report": str(pipeline_report_path)},
    })
    _write_json(pipeline_report_path, base_report)
    print(f"Final reviewed JSON: {final_path}")
    print(f"Stages: critical {critical_start} -> {flash_critical} -> "
          f"{len(refreshed['critical_items'])}; Flash changed {flash_changed}; "
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
    parser.add_argument("--output-dir", "-o", default="output/shorten_review")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
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
    )
    try:
        run_pipeline(config)
    except PipelineStageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
