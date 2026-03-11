from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_text import (
    DEFAULT_AVG_CHARS_PER_SEC,
    DEFAULT_MAX_MISMATCH_RATIO,
    analyze_subtitles,
    load_json,
    save_json,
)


def main(argv: list[str] | None = None) -> int:
    """Rebuild analysis metadata for subtitle JSON."""

    parser = argparse.ArgumentParser(
        description="Rebuild subtitle analysis for a shortened subtitle JSON file."
    )
    parser.add_argument("input", help="Input subtitle JSON file")
    parser.add_argument(
        "--output",
        "-o",
        help="Optional output JSON path. Defaults to overwriting the input file.",
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
        help=f"Critical ratio threshold (default: {DEFAULT_MAX_MISMATCH_RATIO}).",
    )
    args = parser.parse_args(argv)

    output_path = args.output or args.input
    items = load_json(args.input)
    result = analyze_subtitles(
        items,
        avg_chars_per_sec=args.avg_chars_per_sec,
        max_mismatch_ratio=args.max_mismatch_ratio,
    )
    save_json(output_path, result["items"])

    critical_count = len(result["critical_items"])
    print(
        "[reanalyze] "
        f"items={len(result['items'])} "
        f"checked={result['checked_count']} "
        f"critical={critical_count} "
        f"output={Path(output_path).resolve()}"
    )
    return 1 if critical_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
