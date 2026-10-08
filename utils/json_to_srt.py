from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _ms_to_srt_timestamp(ms: int) -> str:
    """Convert milliseconds to SRT timestamp HH:MM:SS,mmm."""
    if ms < 0:
        raise ValueError("Negative timestamp not allowed")
    hours = ms // 3_600_000
    remainder = ms % 3_600_000
    minutes = remainder // 60_000
    seconds = (remainder % 60_000) // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _load_json_objects(json_path: str) -> List[Dict[str, Any]]:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Support both JSON array and JSONL (ndjson) formats
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Top-level JSON must be an array of subtitle objects")
        return data

    # Try JSON Lines
    objs: List[Dict[str, Any]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {i}: {e}")
        objs.append(obj)
    return objs


def json_to_srt(json_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """Convert JSON (array or JSONL) of subtitle entries back to SRT.

    Expected entry fields:
      - index (optional)
      - start (ms)
      - end (ms)
      - duration (optional)
      - text (list[str] or str)

    Returns SRT content as string if `output_path` is None, otherwise writes file and returns None.
    """
    entries = _load_json_objects(json_path)

    if not entries:
        logging.warning("No subtitle entries found in JSON: %s", json_path)
        srt_content = ""
        if output_path:
            Path(output_path).write_text(srt_content, encoding="utf-8")
            logging.info("Wrote empty SRT to %s", output_path)
            return None
        return srt_content

    # Determine ordering: prefer explicit `index` if present, otherwise by `start`.
    have_index = all(("index" in e and isinstance(e["index"], int)) for e in entries)
    if have_index:
        entries_sorted = sorted(entries, key=lambda e: e["index"])
    else:
        entries_sorted = sorted(entries, key=lambda e: int(e.get("start", 0)))

    blocks: List[str] = []

    for idx, entry in enumerate(entries_sorted, start=1):
        index = int(entry.get("index", idx))

        try:
            start_ms = int(entry["start"])
            end_ms = int(entry["end"])
        except KeyError as e:
            raise ValueError(f"Missing required time field: {e}")
        except (TypeError, ValueError):
            raise ValueError("`start` and `end` must be integers (milliseconds)")

        if end_ms < start_ms:
            raise ValueError(f"End time is before start time for index {index} ({start_ms} > {end_ms})")

        start_ts = _ms_to_srt_timestamp(start_ms)
        end_ts = _ms_to_srt_timestamp(end_ms)

        text_val = entry.get("text", "")
        if isinstance(text_val, list):
            lines = [str(x) for x in text_val]
            text_block = "\n".join(lines)
        else:
            text_block = str(text_val)

        block = f"{index}\n{start_ts} --> {end_ts}\n{text_block}"
        blocks.append(block)

    srt_content = "\n\n".join(blocks) + "\n"

    if output_path:
        Path(output_path).write_text(srt_content, encoding="utf-8")
        logging.info("SRT output saved to: %s", output_path)
        return None

    return srt_content


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert JSON (array or JSONL) of subtitles to SRT format")
    parser.add_argument("json_file", help="Path to JSON or JSONL file with subtitle entries")
    parser.add_argument("-o", "--output", help="Output SRT file path (optional)")

    args = parser.parse_args()
    logging.basicConfig(level="INFO", format="[%(levelname)s] %(message)s")

    try:
        result = json_to_srt(args.json_file, args.output)
    except Exception as e:
        logging.error("%s", e)
        raise SystemExit(1)

    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
