from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.text_normalizer import DEFAULT_BATCH_SIZE, DEFAULT_SEPARATOR, _load_runorm, normalize_subtitles_runorm_only
from utils.text_sanitize import sanitize_text


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list")
    return data  # type: ignore[return-value]


def _item_text_to_str(item: Dict[str, Any]) -> str:
    text_val = item.get("text", []) or []
    if isinstance(text_val, list):
        return " ".join(str(t) for t in text_val if t is not None).strip()
    return str(text_val).strip()


def _maybe_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def build_chunks(
    items: List[Dict[str, Any]],
    *,
    limit: int = 100,
    group_size: int = 5,
) -> List[Dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if group_size <= 0:
        raise ValueError("group_size must be > 0")

    selected = items[:limit]
    chunks: List[Dict[str, Any]] = []

    chunk_index = 1
    for i in range(0, len(selected), group_size):
        group = selected[i : i + group_size]
        if not group:
            continue

        lines: List[str] = []
        source_indices: List[int] = []
        starts: List[int] = []
        ends: List[int] = []

        for it in group:
            raw = _item_text_to_str(it)
            if raw:
                # Keep per-line sanitation so we can preserve newlines when joining.
                lines.append(sanitize_text(raw))
            idx = _maybe_int(it.get("index"))
            if idx is not None:
                source_indices.append(idx)
            st = _maybe_int(it.get("start"))
            en = _maybe_int(it.get("end"))
            if st is not None:
                starts.append(st)
            if en is not None:
                ends.append(en)

        chunk_text = "\n".join([ln for ln in lines if ln])

        out: Dict[str, Any] = {
            "index": chunk_index,
            "text": [chunk_text],
            "source_indices": source_indices,
        }
        if starts:
            out["start"] = min(starts)
        if ends:
            out["end"] = max(ends)
        if "start" in out and "end" in out and isinstance(out["start"], int) and isinstance(out["end"], int):
            out["duration"] = int(out["end"] - out["start"])

        chunks.append(out)
        chunk_index += 1

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Take first N subtitle items, group by K, join text with newlines, and normalize with RUNorm only (no RUAccent)."
        )
    )
    parser.add_argument("input", type=Path, help="Input subtitles JSON (list)")
    parser.add_argument("output", type=Path, help="Output JSON path")
    parser.add_argument("--limit", type=int, default=100, help="Number of subtitle records to take")
    parser.add_argument("--group-size", type=int, default=5, help="How many records per chunk")
    parser.add_argument("--no-runorm", action="store_true", help="Skip RUNorm normalization (debug)")
    parser.add_argument(
        "--norm-device",
        default="cpu",
        choices=["CUDA", "cuda", "CPU", "cpu"],
        help="Device for RUNorm (used only if not preloaded)",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="RUNorm batch size")
    parser.add_argument("--separator", default=DEFAULT_SEPARATOR, help="Separator used during batch normalization")

    args = parser.parse_args()

    items = _load_json_list(args.input)
    chunks = build_chunks(items, limit=int(args.limit), group_size=int(args.group_size))

    if args.no_runorm:
        out_items = chunks
    else:
        normalizer = _load_runorm(device=str(args.norm_device))
        out_items = normalize_subtitles_runorm_only(
            chunks,
            normalizer,
            batch_size=int(args.batch_size),
            separator=str(args.separator),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)

    print(f"[prepare] Wrote {args.output} (chunks={len(out_items)})")


if __name__ == "__main__":
    main()
