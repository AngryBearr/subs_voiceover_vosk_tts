import json
from pathlib import Path
from typing import Any, Dict, List

from .compress_critical_segments import process_file


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Shorten critical subtitle segments in analyzed JSON using agent compression."
        )
    )
    parser.add_argument("input", help="Input JSON with analyzed subtitles")
    parser.add_argument(
        "-o", "--output", required=True, help="Path to save updated JSON",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    items = process_file(input_path)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":  # pragma: no cover
    main()
