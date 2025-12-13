import argparse
import json
from typing import Any, Dict, List

from .process_critical_segments import process_json_items


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply compressed texts to critical subtitle segments and "
            "recompute timing analysis."
        )
    )
    parser.add_argument("input", help="Input JSON with analyzed subtitles")
    parser.add_argument(
        "compressed",
        help=(
            "JSON mapping from index to new text. Keys may be strings or ints; "
            "values must be strings."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to save updated JSON",
    )

    args = parser.parse_args()

    items = load_json(args.input)
    compressed_raw: Dict[str, str] = load_json(args.compressed)

    # Нормализуем ключи к int, значения к str
    compressed_texts: Dict[int, str] = {}
    for k, v in compressed_raw.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):  # pragma: no cover - защитный код
            continue
        compressed_texts[idx] = str(v)

    updated = process_json_items(items, compressed_texts)
    save_json(args.output, updated)


if __name__ == "__main__":  # pragma: no cover
    main()
