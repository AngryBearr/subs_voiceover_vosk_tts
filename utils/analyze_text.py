"""Legacy-compatible deterministic subtitle timing analysis."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from utils.shortening_plan import PlannerConfig, analysis_for_text, plan_subtitles


DEFAULT_AVG_CHARS_PER_SEC = 13.0
DEFAULT_MAX_MISMATCH_RATIO = 1.5
MIN_DURATION_SEC = 1.0


def join_text_lines(text_field: Any) -> str:
    """Convert text field (list of strings or string) to a single string."""
    if isinstance(text_field, list):
        return " ".join(str(t) for t in text_field)
    return str(text_field)


def analyze_subtitles(
    items: List[Dict[str, Any]],
    avg_chars_per_sec: float,
    max_mismatch_ratio: float,
) -> Dict[str, Any]:
    """Attach the shared planner analysis without changing source fields."""
    config = PlannerConfig(cps=avg_chars_per_sec, max_speed=max_mismatch_ratio, gap_ms=200)
    plans = plan_subtitles(items, config)
    critical_items: List[Dict[str, Any]] = []
    checked_count = 0

    for item, plan in zip(items, plans):
        analysis = analysis_for_text(plan, plan.text, config)
        timing_valid = (
            plan.start is not None
            and plan.end is not None
            and plan.end > plan.start
            and not plan.invalid_timing
        )
        if timing_valid:
            assert plan.start is not None and plan.end is not None
            duration_sec = (plan.end - plan.start) / 1000.0
            analysis["duration_sec"] = duration_sec
            analysis["is_short_segment"] = duration_sec < MIN_DURATION_SEC
        if plan.status == "invalid":
            analysis.update(
                {
                    "is_checked": False,
                    "is_critical": False,
                    "reason": "invalid_timing" if plan.invalid_timing else "invalid_text",
                    "invalid_timing": plan.invalid_timing,
                }
            )
        else:
            analysis["is_checked"] = True
            checked_count += 1
            if analysis["is_critical"]:
                critical_items.append(item)
        item["analysis"] = analysis

    return {"items": items, "checked_count": checked_count, "critical_items": critical_items}


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze subtitle timing vs text length. Input JSON is read-only; "
            "the optional output receives a new analyzed copy."
        )
    )
    parser.add_argument("input", help="Input JSON file; it is never modified.")
    parser.add_argument(
        "--output", "-o",
        help="Optional output JSON path. Without it, analysis is printed and no file is written.",
    )
    parser.add_argument(
        "--avg-chars-per-sec", "--cps", type=float, default=DEFAULT_AVG_CHARS_PER_SEC,
        help=f"Nominal text estimate in characters/second (default: {DEFAULT_AVG_CHARS_PER_SEC}).",
    )
    parser.add_argument(
        "--max-mismatch-ratio", "--threshold", type=float, default=DEFAULT_MAX_MISMATCH_RATIO,
        help=(
            "Allowed timing speed ratio from 1.0 through 1.5 (default: 1.5); "
            "values outside that range are rejected."
        ),
    )
    args = parser.parse_args(argv)
    items = load_json(args.input)
    try:
        result = analyze_subtitles(
            items,
            avg_chars_per_sec=args.avg_chars_per_sec,
            max_mismatch_ratio=args.max_mismatch_ratio,
        )
    except ValueError as exc:
        parser.error(str(exc))
    critical_items = result["critical_items"]
    invalid_timing_count = sum(
        isinstance(item.get("analysis"), dict) and bool(item["analysis"].get("invalid_timing"))
        for item in items
    )
    print(
        f"Проверено сегментов: {result['checked_count']}. "
        f"Критичных: {len(critical_items)}. Неверных таймингов: {invalid_timing_count}."
    )
    if critical_items:
        print("ВНИМАНИЕ: обнаружены субтитры с прогнозируемым превышением бюджета.")
        for item in critical_items:
            idx = item.get("index")
            text_value = join_text_lines(item.get("text", ""))
            analysis = item.get("analysis", {})
            ratio = analysis.get("mismatch_ratio")
            duration = analysis.get("duration_sec")
            estimate = analysis.get("estimated_sec")
            print(f"  - #{idx}: оценка {estimate:.2f}с при окне {duration:.2f}с (отношение {ratio:.2f}x). Текст: '{text_value}'")
    if args.output:
        save_json(args.output, result["items"])
    return 1 if critical_items else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
