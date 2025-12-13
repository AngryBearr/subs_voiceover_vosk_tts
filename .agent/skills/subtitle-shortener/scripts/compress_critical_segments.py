import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


try:
    from utils.analyze_text import (
        analyze_subtitles,
        DEFAULT_AVG_CHARS_PER_SEC,
        DEFAULT_MAX_MISMATCH_RATIO,
    )
except Exception:  # pragma: no cover
    analyze_subtitles = None  # type: ignore[assignment]
    DEFAULT_AVG_CHARS_PER_SEC = 13.0
    DEFAULT_MAX_MISMATCH_RATIO = 1.3


@dataclass
class SegmentView:
    index: int
    text: str
    duration_sec: float
    estimated_sec: float
    mismatch_ratio: float | None
    extended_mismatch_ratio: float | None
    combined_mismatch_ratio: float | None


def load_items(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def join_text_lines(text_field: Any) -> str:
    if isinstance(text_field, list):
        return " ".join(str(t) for t in text_field)
    return str(text_field)


def build_context_window(items: List[Dict[str, Any]], pos: int, window: int = 3) -> str:
    start = max(0, pos - window)
    end = min(len(items), pos + window + 1)
    parts: List[str] = []
    for item in items[start:end]:
        text = join_text_lines(item.get("text", ""))
        if text.strip():
            parts.append(text)
    return " ".join(parts)


def get_combined_mismatch_ratio(analysis: Dict[str, Any]) -> float | None:
    """Return the best-case mismatch ratio for decision-making.

    We treat `extended_mismatch_ratio` as "with using the gap to next subtitle",
    and `mismatch_ratio` as "text-only". For selection / prioritization we use
    the minimal (best-case) value, because it's the most realistic achievable
    ratio when there is a gap.
    """

    mismatch_ratio = analysis.get("mismatch_ratio")
    extended_ratio = analysis.get("extended_mismatch_ratio")

    if mismatch_ratio is None and extended_ratio is None:
        return None
    if mismatch_ratio is None:
        return extended_ratio
    if extended_ratio is None:
        return mismatch_ratio
    return min(mismatch_ratio, extended_ratio)


def collect_critical_segments(
    items: List[Dict[str, Any]],
    window: int = 3,
    min_combined_ratio: float = 1.5,
) -> Dict[int, Dict[str, Any]]:
    segments: Dict[int, Dict[str, Any]] = {}
    for pos, item in enumerate(items):
        analysis = item.get("analysis") or {}
        if not analysis.get("is_critical"):
            continue

        combined_ratio = get_combined_mismatch_ratio(analysis)
        if combined_ratio is None or combined_ratio < min_combined_ratio:
            continue

        idx = item.get("index")
        if idx is None:
            continue

        duration_sec = analysis.get("duration_sec")
        estimated_sec = analysis.get("estimated_sec")

        mismatch_ratio = analysis.get("mismatch_ratio")
        extended_ratio = analysis.get("extended_mismatch_ratio")

        segments[idx] = {
            "context": build_context_window(items, pos, window=window),
            "index": idx,
            "text": join_text_lines(item.get("text", "")),
            "duration_sec": duration_sec,
            "estimated_sec": estimated_sec,
            "mismatch_ratio": mismatch_ratio,
            "extended_mismatch_ratio": extended_ratio,
            "combined_mismatch_ratio": combined_ratio,
        }

    return segments


def recompute_analysis_for_position(
    items: List[Dict[str, Any]],
    pos: int,
    avg_chars_per_sec: float = DEFAULT_AVG_CHARS_PER_SEC,
    max_mismatch_ratio: float = DEFAULT_MAX_MISMATCH_RATIO,
) -> None:
    """Recompute analysis for a single item and (optionally) its next neighbor.

    `extended_mismatch_ratio` depends on the next subtitle timing, so we analyze
    a 1-2 item slice to get consistent results for the target item.
    """
    if analyze_subtitles is None:
        raise RuntimeError("utils.analyze_text.analyze_subtitles is not available")
    if pos < 0 or pos >= len(items):
        return

    end = min(len(items), pos + 2)
    slice_items = [dict(items[i]) for i in range(pos, end)]
    result = analyze_subtitles(slice_items, avg_chars_per_sec, max_mismatch_ratio)
    # Update only the target item
    items[pos]["analysis"] = result["items"][0].get("analysis")


def should_continue_compression(
    analysis: Dict[str, Any],
    target_ratio: float,
) -> bool:
    """Return True if the segment still needs more compression.

    We use the combined ratio (min(mismatch_ratio, extended_mismatch_ratio)).
    """
    combined = get_combined_mismatch_ratio(analysis)
    if combined is None:
        return False
    return combined > target_ratio


def is_too_aggressive_compression(
        old_analysis: Dict[str, Any],
        new_analysis: Dict[str, Any],
        low_ratio_threshold: float = 1.0,
        strong_overflow_threshold: float = 2.1,
) -> bool:
        """Heuristic check for over‑aggressive compression.

        - old_analysis: analysis до сжатия сегмента.
        - new_analysis: analysis после сжатия и пересчёта.
        - low_ratio_threshold: если новый combined < этого порога, текст
            считается очень сильно ужатым (обычно < 1.0).
        - strong_overflow_threshold: если исходный combined был сильно выше
            этого порога (например, > 2.1), то агрессивное снижение до < 1.0
            допустимо, чтобы вернуться в безопасный диапазон.

        Логика:
            * если новый combined < low_ratio_threshold и старый combined
                ≤ strong_overflow_threshold, считаем такое сжатие слишком
                агрессивным и рекомендуем сделать мягче;
            * во всех остальных случаях считаем сжатие допустимым.
        """

        old_combined = get_combined_mismatch_ratio(old_analysis)
        new_combined = get_combined_mismatch_ratio(new_analysis)

        if old_combined is None or new_combined is None:
                return False

        if new_combined < low_ratio_threshold and old_combined <= strong_overflow_threshold:
                return True
        return False


def build_agent_payload(segments: Dict[int, Dict[str, Any]]) -> str:
    payload = {"segments": list(segments.values())}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_agent_response(raw: str) -> Dict[int, str]:
    data = json.loads(raw)
    result: Dict[int, str] = {}
    for k, v in data.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        result[idx] = str(v)
    return result


def apply_compressed_texts(
    items: List[Dict[str, Any]],
    compressed_texts: Dict[int, str],
) -> List[Dict[str, Any]]:
    for item in items:
        idx = item.get("index")
        if idx is None or idx not in compressed_texts:
            continue
        item["text"] = [compressed_texts[idx]]
    return items


def process_file(path: Path) -> List[Dict[str, Any]]:
    """High-level helper for the agent.

    Steps:
    1) Load items
    2) Collect critical segments with context
    3) Build compact JSON payload for the agent
    4) Ask agent to return mapping {index: "compressed text"}
    5) Apply new texts to items and return updated list
    """

    items = load_items(path)
    segments = collect_critical_segments(items)

    # Step 3: build payload for the agent
    payload = build_agent_payload(segments)

    # NOTE: The actual agent call is not implemented here.
    # In the skill runtime, Claude reads this script and uses
    # `payload` inside its own prompt, then obtains `agent_raw`.
    # Here we just expose the pieces needed for that workflow.
    _ = payload  # to avoid lints in this context

    # This function is intended to be patched by the agent flow:
    # - Agent loads items
    # - Uses collect_critical_segments/build_agent_payload in its prompt
    # - Calls parse_agent_response on model output
    # - Calls apply_compressed_texts and then external analysis recomputation
    return items
