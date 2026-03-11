from __future__ import annotations

import argparse
import json
from typing import Any, Mapping


DEFAULT_AVG_CHARS_PER_SEC = 13.0
DEFAULT_MAX_MISMATCH_RATIO = 1.5
MIN_DURATION_SEC = 1.0
MIN_GAP_BEFORE_NEXT_SEC = 0.3
MIN_WORDS_SHORT_SEGMENT = 3


def coerce_text_lines(text_field: Any) -> list[str]:
    """Normalize subtitle text to a list of non-empty lines."""

    if text_field is None:
        return []
    if isinstance(text_field, list):
        return [str(part).strip() for part in text_field if str(part).strip()]

    text_value = str(text_field)
    if not text_value.strip():
        return []

    split_lines = [line.strip() for line in text_value.splitlines() if line.strip()]
    if split_lines:
        return split_lines
    return [text_value.strip()]


def join_text_lines(text_field: Any, separator: str = " ") -> str:
    """Join subtitle text into one analysis string."""

    return separator.join(coerce_text_lines(text_field)).strip()


def analyze_subtitles(
    items: list[dict[str, Any]],
    avg_chars_per_sec: float,
    max_mismatch_ratio: float,
) -> dict[str, Any]:
    """Analyze subtitle timing and attach analysis metadata in-place."""

    critical_items: list[dict[str, Any]] = []
    checked_count = 0

    for position, item in enumerate(items):
        start_ms = _coerce_int(item.get("start"))
        end_ms = _coerce_int(item.get("end"))
        duration_ms = _coerce_int(item.get("duration"))

        if duration_ms is None and start_ms is not None and end_ms is not None:
            duration_ms = end_ms - start_ms

        if duration_ms is None:
            item["analysis"] = _make_empty_analysis()
            continue

        duration_sec = max(duration_ms, 0) / 1000.0
        text_value = join_text_lines(item.get("text"))
        words_count = len(text_value.split()) if text_value else 0
        is_short_segment = duration_sec < MIN_DURATION_SEC

        if is_short_segment and words_count < MIN_WORDS_SHORT_SEGMENT:
            item["analysis"] = {
                "duration_sec": duration_sec,
                "estimated_sec": None,
                "mismatch_ratio": None,
                "diff_sec": None,
                "extended_duration_sec": None,
                "extended_mismatch_ratio": None,
                "used_gap_sec": None,
                "words_count": words_count,
                "is_short_segment": True,
                "is_checked": False,
                "is_critical": False,
            }
            continue

        estimated_sec = len(text_value) / avg_chars_per_sec if avg_chars_per_sec > 0 else 0.0
        mismatch_ratio = estimated_sec / duration_sec if duration_sec > 0 else float("inf")
        diff_sec = estimated_sec - duration_sec

        extended_duration_sec: float | None = None
        extended_mismatch_ratio: float | None = None
        used_gap_sec: float | None = None

        if mismatch_ratio > max_mismatch_ratio and position + 1 < len(items):
            next_item = items[position + 1]
            next_start_ms = _coerce_int(next_item.get("start"))
            if next_start_ms is not None and end_ms is not None:
                gap_sec = (next_start_ms - end_ms) / 1000.0
                extra_sec = gap_sec - MIN_GAP_BEFORE_NEXT_SEC
                if extra_sec > 0:
                    extended_duration_sec = duration_sec + extra_sec
                    used_gap_sec = extra_sec
                    if extended_duration_sec > 0:
                        extended_mismatch_ratio = estimated_sec / extended_duration_sec

        effective_ratio = (
            extended_mismatch_ratio if extended_mismatch_ratio is not None else mismatch_ratio
        )
        is_critical = effective_ratio > max_mismatch_ratio

        item["analysis"] = {
            "duration_sec": duration_sec,
            "estimated_sec": estimated_sec,
            "mismatch_ratio": mismatch_ratio,
            "diff_sec": diff_sec,
            "extended_duration_sec": extended_duration_sec,
            "extended_mismatch_ratio": extended_mismatch_ratio,
            "used_gap_sec": used_gap_sec,
            "words_count": words_count,
            "is_short_segment": is_short_segment,
            "is_checked": True,
            "is_critical": is_critical,
        }
        checked_count += 1
        if is_critical:
            critical_items.append(item)

    return {
        "items": items,
        "checked_count": checked_count,
        "critical_items": critical_items,
    }


def load_json(path: str) -> list[dict[str, Any]]:
    """Load subtitle JSON as a list of objects."""

    with open(path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of subtitle items")

    items: list[dict[str, Any]] = []
    for value in data:
        if not isinstance(value, Mapping):
            raise ValueError("Each subtitle item must be an object")
        items.append(dict(value))
    return items


def save_json(path: str, data: list[dict[str, Any]]) -> None:
    """Save subtitle JSON with stable formatting."""

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    """Run CLI analysis for subtitle JSON."""

    parser = argparse.ArgumentParser(
        description="Analyze subtitle timing vs text length and mark over-threshold items."
    )
    parser.add_argument("input", help="Input JSON file with subtitle items")
    parser.add_argument(
        "--output",
        "-o",
        help="Optional output JSON file. If omitted, only analysis summary is printed.",
    )
    parser.add_argument(
        "--avg-chars-per-sec",
        "--cps",
        type=float,
        default=DEFAULT_AVG_CHARS_PER_SEC,
        help=f"Average characters per second (default: {DEFAULT_AVG_CHARS_PER_SEC}).",
    )
    parser.add_argument(
        "--max-mismatch-ratio",
        "--threshold",
        type=float,
        default=DEFAULT_MAX_MISMATCH_RATIO,
        help=f"Allowed mismatch ratio threshold (default: {DEFAULT_MAX_MISMATCH_RATIO}).",
    )
    args = parser.parse_args(argv)

    items = load_json(args.input)
    result = analyze_subtitles(
        items,
        avg_chars_per_sec=args.avg_chars_per_sec,
        max_mismatch_ratio=args.max_mismatch_ratio,
    )

    critical_items = result["critical_items"]
    print(
        "[analyze] "
        f"items={len(result['items'])} "
        f"checked={result['checked_count']} "
        f"critical={len(critical_items)}"
    )

    if args.output:
        save_json(args.output, result["items"])

    return 1 if critical_items else 0


def _make_empty_analysis() -> dict[str, Any]:
    return {
        "duration_sec": None,
        "estimated_sec": None,
        "mismatch_ratio": None,
        "diff_sec": None,
        "extended_duration_sec": None,
        "extended_mismatch_ratio": None,
        "used_gap_sec": None,
        "words_count": None,
        "is_short_segment": False,
        "is_checked": False,
        "is_critical": False,
    }


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
