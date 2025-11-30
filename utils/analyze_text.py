import argparse
import json
import sys
from typing import Any, Dict, List


DEFAULT_AVG_CHARS_PER_SEC = 13.0
DEFAULT_MAX_MISMATCH_RATIO = 1.3
MIN_DURATION_SEC = 1.0
MIN_GAP_BEFORE_NEXT_SEC = 0.3
MIN_WORDS_SHORT_SEGMENT = 3


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
    """Analyze subtitle items and return stats.

    Each item gets an `analysis` field:
    {
        "duration_sec": float,
        "estimated_sec": float,
        "mismatch_ratio": float,
        "diff_sec": float,
        "extended_duration_sec": float | None,
        "extended_mismatch_ratio": float | None,
        "used_gap_sec": float | None,
        "words_count": int | None,
        "is_short_segment": bool,
        "is_checked": bool,
        "is_critical": bool,
    }
    """

    critical_items: List[Dict[str, Any]] = []
    checked_count = 0

    for idx, item in enumerate(items):
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            # Nothing to analyze
            item["analysis"] = {
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
            continue

        duration_ms = end - start
        duration_sec = duration_ms / 1000.0

        text_value = join_text_lines(item.get("text", ""))
        words_count = len(text_value.split()) if text_value else 0
        is_short_segment = duration_sec < MIN_DURATION_SEC

        # Decide if we should check this segment at all
        should_check = True
        if is_short_segment and words_count < MIN_WORDS_SHORT_SEGMENT:
            should_check = False

        if not should_check:
            item["analysis"] = {
                "duration_sec": duration_sec,
                "estimated_sec": None,
                "mismatch_ratio": None,
                "diff_sec": None,
                "extended_duration_sec": None,
                "extended_mismatch_ratio": None,
                "used_gap_sec": None,
                "words_count": words_count,
                "is_short_segment": is_short_segment,
                "is_checked": False,
                "is_critical": False,
            }
            continue

        estimated_sec = len(text_value) / avg_chars_per_sec if avg_chars_per_sec > 0 else 0.0

        if duration_sec > 0:
            mismatch_ratio = estimated_sec / duration_sec
        else:
            mismatch_ratio = float("inf")

        diff_sec = estimated_sec - duration_sec

        # Try to extend into the gap before next subtitle if needed
        extended_duration_sec: float | None = None
        extended_mismatch_ratio: float | None = None
        used_gap_sec: float | None = None

        if mismatch_ratio > max_mismatch_ratio and idx + 1 < len(items):
            next_item = items[idx + 1]
            next_start = next_item.get("start")
            if next_start is not None:
                gap_ms = next_start - end
                gap_sec = gap_ms / 1000.0
                extra_sec = gap_sec - MIN_GAP_BEFORE_NEXT_SEC
                if extra_sec > 0:
                    extended_duration_sec = duration_sec + extra_sec
                    used_gap_sec = extra_sec
                    if extended_duration_sec > 0:
                        extended_mismatch_ratio = estimated_sec / extended_duration_sec

        # Final decision on criticality uses extended ratio if available
        effective_ratio = extended_mismatch_ratio if extended_mismatch_ratio is not None else mismatch_ratio
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


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze subtitle timing vs text length and mark critical "
            "segments that are too long for their slots."
        )
    )
    parser.add_argument("input", help="Input JSON file with subtitles")
    parser.add_argument(
        "--output",
        "-o",
        help=(
            "Optional output JSON file. If omitted, input file will not be "
            "modified; analysis will only be printed."
        ),
    )
    parser.add_argument(
        "--avg-chars-per-sec",
        "--cps",
        type=float,
        default=DEFAULT_AVG_CHARS_PER_SEC,
        help=(
            "Average characters per second for TTS (default: "
            f"{DEFAULT_AVG_CHARS_PER_SEC})."
        ),
    )
    parser.add_argument(
        "--max-mismatch-ratio",
        "--threshold",
        type=float,
        default=DEFAULT_MAX_MISMATCH_RATIO,
        help=(
            "Maximum allowed ratio estimated_duration/target_duration before "
            "segment is considered critical (default: "
            f"{DEFAULT_MAX_MISMATCH_RATIO})."
        ),
    )

    args = parser.parse_args(argv)

    items = load_json(args.input)

    result = analyze_subtitles(
        items,
        avg_chars_per_sec=args.avg_chars_per_sec,
        max_mismatch_ratio=args.max_mismatch_ratio,
    )

    critical_items = result["critical_items"]
    checked_count = result["checked_count"]

    print(
        f"Проверено сегментов (>= {MIN_DURATION_SEC:.1f}с): {checked_count}. "
        f"Критичных: {len(critical_items)}."
    )

    if critical_items:
        print("ОШИБКА: обнаружены субтитры, которые физически не влезают в свои слоты.")
        for item in critical_items:
            idx = item.get("index")
            text_value = join_text_lines(item.get("text", ""))
            a = item.get("analysis", {})
            ratio = a.get("mismatch_ratio")
            dur = a.get("duration_sec")
            est = a.get("estimated_sec")
            print(
                f"  - #{idx}: длина текста {est:.2f}с при слоте {dur:.2f}с "
                f"(в {ratio:.2f}x длиннее). Текст: '{text_value}'"
            )

    if args.output:
        save_json(args.output, result["items"])

    # Exit with non-zero code if there are critical items
    if critical_items:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
