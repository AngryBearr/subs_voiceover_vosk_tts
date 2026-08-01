"""Shared logic for subtitle shortening scripts.

Contains constants, data classes, context building, response parsing,
validation, selection, and the iterative shortening loop used by both
the DeepSeek and OpenCode implementations.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from utils.analyze_text import analyze_subtitles, join_text_lines
from utils.shortening_domain import CharacterRateBudgetEstimator, DurationSelectionPolicy, TimingWindow

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 1.5
DEFAULT_CONTEXT_WINDOW = 3
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_CONCURRENCY = 5
DEFAULT_AVG_CHARS_PER_SEC = 13.0
DEFAULT_BATCH_SIZE = 5
MAX_BATCH_SIZE = 25
DEFAULT_STUCK_THRESHOLD = 5
DEFAULT_EARLY_STOP_PATIENCE = 5
MIN_WORDS_RESULT = 3

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_SHARED = (
    "Сокращай русские субтитры строго короче original и не длиннее max_chars. "
    "Сохраняй смысловой каркас: кто действует, действие и объект, отрицание, модальность, "
    "причину/уступку, сравнение, время/лицо, число и охват альтернатив, сущности, числа и "
    "межстрочные ссылки. Контекст нужен только для грамматики и референтов: не переноси из него факты. "
    "Можно убрать повторы и несущественные детали. Если эквивалентный вариант не помещается, верни original без изменений. "
    "Не объясняй и не показывай рассуждения. Верни только JSON вида "
    '{"results":[{"index":1,"shortened_text":"текст"}]}.'
)
SYSTEM_PROMPT_SHORTEN = SYSTEM_PROMPT_SHARED
SYSTEM_PROMPT_REPHRASE = SYSTEM_PROMPT_SHARED

SYSTEM_PROMPT = SYSTEM_PROMPT_SHORTEN


def get_system_prompt(strategy: str, min_words: int) -> str:
    """Return the immutable cache-friendly system prompt."""
    return SYSTEM_PROMPT_SHARED


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ShortenResult:
    index: int
    original_text: str
    shortened_text: str
    success: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class ShorteningBudget:
    effective_duration_sec: float
    current_chars: int
    max_chars: int
    required_reduction_percent: float


def calculate_budget(
    item: Dict[str, Any], target_ratio: float, avg_chars_per_sec: float
) -> ShorteningBudget:
    """Calculate a stable character budget from saved effective duration."""
    _, duration = _stable_durations(item)
    text = join_text_lines(item.get("text", ""))
    window = TimingWindow.from_item(item)
    max_chars = CharacterRateBudgetEstimator(avg_chars_per_sec).max_chars(window, target_ratio)
    reduction = max(0.0, (1.0 - max_chars / max(1, len(text))) * 100.0)
    return ShorteningBudget(float(duration), len(text), max_chars, round(reduction, 1))


def target_min_words(text: str, configured_min_words: int) -> int:
    """Relax the minimum for an already short three-word subtitle."""
    word_count = len(text.split())
    if word_count <= 1:
        return 1
    if word_count <= configured_min_words:
        return max(2, word_count - 1)
    return configured_min_words


def _stable_durations(item: Dict[str, Any]) -> tuple[float, float]:
    """Return immutable base/effective durations, deriving legacy values once."""
    analysis = item.get("analysis", {})
    window = TimingWindow.from_item(item)
    analysis.update(window.as_analysis_fields())
    return window.base_duration_sec, window.effective_duration_sec


def refresh_analysis(
    items: List[Dict[str, Any]], avg_chars_per_sec: float, threshold: float,
    preserve_saved_timing: bool = False,
) -> Dict[str, Any]:
    """Refresh each item while preserving legacy effective timing without drift."""
    saved = [(_stable_durations(item), bool(item.get("analysis", {}).get("is_checked"))) for item in items]
    timed_result = analyze_subtitles(items, avg_chars_per_sec, threshold)
    critical: List[Dict[str, Any]] = []
    checked_count = 0
    for index, item in enumerate(timed_result["items"]):
        analysis = item.get("analysis", {})
        has_timing = (item.get("end", 0) - item.get("start", 0)) > 0
        (base, effective), was_checked = saved[index]
        if has_timing and not preserve_saved_timing:
            base, effective = _stable_durations(item)
        elif preserve_saved_timing:
            analysis["is_checked"] = was_checked
        elif base > 0 and was_checked:
            analysis["is_checked"] = True
        if not analysis.get("is_checked") or base <= 0 or effective <= 0:
            continue
        checked_count += 1
        estimated = len(join_text_lines(item.get("text", ""))) / avg_chars_per_sec
        base_ratio = estimated / base
        effective_ratio = estimated / effective
        analysis["available_duration_sec"] = base
        analysis["effective_duration_sec"] = effective
        analysis["duration_sec"] = base
        if effective != base:
            analysis["extended_duration_sec"] = effective
            analysis["used_gap_sec"] = effective - base
        else:
            analysis["extended_duration_sec"] = None
            analysis["used_gap_sec"] = None
        analysis["estimated_sec"] = estimated
        analysis["diff_sec"] = estimated - base
        analysis["mismatch_ratio"] = base_ratio
        analysis["extended_mismatch_ratio"] = effective_ratio if effective != base else None
        analysis["words_count"] = len(join_text_lines(item.get("text", "")).split())
        analysis["is_short_segment"] = base < 1.0
        analysis["is_critical"] = effective_ratio > threshold
        if analysis["is_critical"]:
            critical.append(item)
    return {"items": timed_result["items"], "checked_count": checked_count, "critical_items": critical}


def hydrate_context_timing(
    items: List[Dict[str, Any]],
    context_items: List[Dict[str, Any]],
    avg_chars_per_sec: float,
    threshold: float,
) -> List[Dict[str, Any]]:
    """Copy real-neighbor timing from a full context onto sparse target items."""
    targets = copy.deepcopy(items)
    context = copy.deepcopy(context_items)
    validate_context_source(targets, context)
    context_result = refresh_analysis(context, avg_chars_per_sec, threshold)
    context_by_index = {int(item["index"]): item for item in context_result["items"]}
    hydrated: List[Dict[str, Any]] = []
    for target in targets:
        source = context_by_index[int(target["index"])]
        source_analysis = source.get("analysis", {})
        target_analysis = target.setdefault("analysis", {})
        target_analysis["available_duration_sec"] = source_analysis.get("available_duration_sec")
        target_analysis["effective_duration_sec"] = source_analysis.get("effective_duration_sec")
        target_analysis["is_checked"] = source_analysis.get("is_checked", False)
        hydrated.append(target)
    return hydrated


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------


def build_context(
    items: List[Dict[str, Any]],
    target_idx: int,
    context_window: int,
) -> str:
    """Build context string with surrounding subtitles (no timestamps)."""
    start = max(0, target_idx - context_window)
    end = min(len(items), target_idx + context_window + 1)

    lines: List[str] = []
    for i in range(start, end):
        item = items[i]
        idx = item.get("index", i + 1)
        text = join_text_lines(item.get("text", ""))

        if i == target_idx:
            lines.append(f">>> ЦЕЛЕВОЙ СУБТИТР #{idx}:")
            lines.append(f">>> {text}")
            lines.append("<<<")
        else:
            role = "ДО" if i < target_idx else "ПОСЛЕ"
            lines.append(f"[{role}] Субтитр #{idx}: {text}")

    return "\n".join(lines)


def build_batch_context(
    items: List[Dict[str, Any]],
    target_indices: List[int],
    context_window: int,
    context_items: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Serialize the union of target context exactly once as compact JSON."""
    if not target_indices:
        return "[]"
    source = context_items or items
    source_positions = {int(item["index"]): pos for pos, item in enumerate(source)}
    overlay = {int(item["index"]): join_text_lines(item.get("text", "")) for item in items}
    wanted: Set[int] = set()
    for idx in target_indices:
        number = int(items[idx]["index"])
        pos = source_positions[number]
        wanted.update(range(max(0, pos - context_window), min(len(source), pos + context_window + 1)))
    payload = [{"index": source[pos]["index"],
                "text": overlay.get(int(source[pos]["index"]), join_text_lines(source[pos].get("text", "")))}
               for pos in sorted(wanted)]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def validate_context_source(targets: List[Dict[str, Any]], source: List[Dict[str, Any]]) -> None:
    """Validate unique source indices and exact target/source originals."""
    indexed: Dict[int, str] = {}
    for item in source:
        index = item.get("index")
        if not isinstance(index, int):
            raise ValueError("context source: missing_or_invalid_index")
        if index in indexed:
            raise ValueError(f"context source: duplicate_index:{index}")
        indexed[index] = join_text_lines(item.get("text", "")).strip()
    for item in targets:
        index = item.get("index")
        if not isinstance(index, int) or index not in indexed:
            raise ValueError(f"context source: missing_target_index:{index}")
        if indexed[index] != join_text_lines(item.get("text", "")).strip():
            raise ValueError(f"context source: text_mismatch:{index}")


def build_user_prompt(context: str, min_words: int) -> str:
    """Build user prompt with context (single subtitle, backward compat)."""
    return (
        f"{context}\n\n"
        f"Сократи ЦЕЛЕВОЙ СУБТИТР по смыслу. "
        f"Минимум {min_words} слов в результате."
    )


def build_batch_user_prompt(
    batch_context: str,
    min_words: int,
    strategy: str = "shorten",
    targets: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build user prompt for a batch of subtitles."""
    task = "Перефразируй компактнее." if strategy == "rephrase" else "Сократи."

    target_json = json.dumps(targets or [], ensure_ascii=False, separators=(",", ":"))
    return (
        f"Контекст:{batch_context}\nЦели и budgets:{target_json}\n"
        f"{task} Соблюдай min_words каждой цели."
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_shortened_response(response: Optional[str]) -> Optional[str]:
    """Parse model response and extract shortened text (single subtitle)."""
    if not response:
        return None

    try:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        data = json.loads(text)
        shortened = data.get("shortened_text", "").strip()
        if shortened:
            return shortened
    except json.JSONDecodeError:
        pass

    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])
            shortened = data.get("shortened_text", "").strip()
            if shortened:
                return shortened
    except (json.JSONDecodeError, ValueError):
        pass

    if not response.startswith("{") and len(response.split()) >= MIN_WORDS_RESULT:
        return response.strip()

    return None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract JSON from a string, handling markdown wrappers."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        )

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def parse_batch_response(
    response: Optional[str],
    expected_indices: List[int],
    items: List[Dict[str, Any]],
) -> Dict[int, str]:
    """Parse batch response and return dict {0-based index: shortened_text}.

    The model returns subtitle numbers (from the ``index`` field of each
    item) as ``"index"`` in the JSON.  This function maps them back to
    0-based positions in *items*.
    """
    if not response:
        return {}

    number_to_idx: Dict[int, int] = {}
    for idx in expected_indices:
        item = items[idx]
        sub_number = item.get("index", idx + 1)
        number_to_idx[sub_number] = idx

    data = _extract_json(response)
    if data is None:
        return {}

    results_list = data.get("results")
    if not isinstance(results_list, list):
        if "shortened_text" in data and len(expected_indices) == 1:
            text = str(data["shortened_text"]).strip()
            if text:
                return {expected_indices[0]: text}
        return {}

    results_map: Dict[int, str] = {}
    for r in results_list:
        if not isinstance(r, dict):
            continue
        sub_number = r.get("index")
        text = str(r.get("shortened_text", "")).strip()
        if sub_number is None or not text:
            continue
        try:
            sub_number = int(sub_number)
        except (ValueError, TypeError):
            continue
        idx = number_to_idx.get(sub_number)
        if idx is not None:
            results_map[idx] = text

    return results_map


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_shortened_text(
    original: str,
    shortened: str,
    min_words: int = MIN_WORDS_RESULT,
    max_chars: Optional[int] = None,
) -> bool:
    """Validate that shortened text meets requirements.

    Strict rules:
    - Must not be empty or only punctuation.
    - Must not contain ``...`` unless the original also does.
    - Must have at least *min_words* words (with repetition exception).
    """
    return validation_error(original, shortened, min_words, max_chars) is None


def validation_error(original: str, shortened: str, min_words: int, max_chars: Optional[int]) -> Optional[str]:
    """Return a precise, cheap validation error or ``None``."""
    if not shortened:
        return "missing"
    if shortened.strip() == original.strip():
        return "unchanged"
    if len(shortened) >= len(original):
        return "not_shorter"
    if max_chars is not None and len(shortened) > max_chars:
        return f"too_long:{len(shortened)}>{max_chars}"

    cleaned = (
        shortened.replace(".", "")
        .replace(",", "")
        .replace("!", "")
        .replace("?", "")
        .strip()
    )
    if not cleaned:
        return "punctuation_only"

    if "..." in shortened and "..." not in original:
        return "ellipsis_added"

    if set(re.findall(r"\d+(?:[.,]\d+)?", original)) - set(re.findall(r"\d+(?:[.,]\d+)?", shortened)):
        return "number_missing"
    negations = {word for word in ("не", "нет", "никогда", "нельзя") if re.search(rf"\b{word}\b", original, re.I)}
    if any(not re.search(rf"\b{word}\b", shortened, re.I) for word in negations):
        return "negation_missing"

    words = shortened.split()
    if len(words) < min_words:
        if len(words) >= 2:
            unique_words = set(w.lower().strip(".,!?;:") for w in words)
            if len(unique_words) <= 1:
                return None
        return "too_few_words"

    return None


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_subtitles_for_shortening(
    items: List[Dict[str, Any]],
    threshold: float,
    exclude_indices: Optional[Set[int]] = None,
    selection_policy: Optional[DurationSelectionPolicy] = None,
) -> List[int]:
    """Select subtitle indices that need shortening.

    Returns list of indices (0-based) into the items list.
    Subtitles in *exclude_indices* are skipped (e.g. stuck subtitles).
    """
    exclude = exclude_indices or set()
    selected: List[int] = []
    for i, item in enumerate(items):
        if i in exclude:
            continue
        analysis = item.get("analysis", {})
        if not analysis.get("is_checked", False):
            continue

        if selection_policy is not None:
            window = TimingWindow.from_item(item)
            if (window.effective_duration_sec > 0
                    and selection_policy.model.predict_seconds(item.get("text", ""))
                    > window.effective_duration_sec * selection_policy.fit_ratio):
                selected.append(i)
            continue

        mismatch = analysis.get("mismatch_ratio")
        extended = analysis.get("extended_mismatch_ratio")

        effective = extended if extended is not None else mismatch

        if effective is not None and effective > threshold:
            selected.append(i)

    return selected


# ---------------------------------------------------------------------------
# Apply results
# ---------------------------------------------------------------------------


def apply_shortening(
    items: List[Dict[str, Any]],
    results: List[ShortenResult],
) -> List[Dict[str, Any]]:
    """Apply shortening results to items, return new copy."""
    new_items = copy.deepcopy(items)

    for result in results:
        if result.success:
            item = new_items[result.index]
            original_text = item.get("text", "")
            if isinstance(original_text, list):
                item["text"] = [result.shortened_text]
            else:
                item["text"] = result.shortened_text

    return new_items


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def save_json(path: Path, data: List[Dict[str, Any]]) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> List[Dict[str, Any]]:
    """Load data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------


def load_api_key(env_var: str = "DEEPSEEK_API_KEY") -> Optional[str]:
    """Load API key from .env file or environment variable."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key == env_var:
                            os.environ.setdefault(key, value)

    return os.environ.get(env_var)


# ---------------------------------------------------------------------------
# Common CLI arguments
# ---------------------------------------------------------------------------


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common CLI arguments shared by both scripts."""
    parser.add_argument("input", help="Input JSON file with analyzed subtitles")
    parser.add_argument("--context-source", help="Optional full-episode analyzed JSON used only for context")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Mismatch ratio threshold for shortening (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help=f"Number of surrounding subtitles for context (default: {DEFAULT_CONTEXT_WINDOW})",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum number of iterations (default: {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=MIN_WORDS_RESULT,
        help=f"Minimum words in shortened text (default: {MIN_WORDS_RESULT})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output/shortened",
        help="Output directory (default: output/shortened)",
    )
    parser.add_argument(
        "--avg-chars-per-sec", type=float, default=DEFAULT_AVG_CHARS_PER_SEC,
        help=f"TTS speed used for budgets (default: {DEFAULT_AVG_CHARS_PER_SEC})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            f"Number of subtitles per API call "
            f"(default: {DEFAULT_BATCH_SIZE}, max: {MAX_BATCH_SIZE})"
        ),
    )
    parser.add_argument(
        "--stuck-threshold",
        type=int,
        default=DEFAULT_STUCK_THRESHOLD,
        help=(
            f"Iterations without change before skipping subtitle "
            f"(default: {DEFAULT_STUCK_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help=(
            "Stop after N consecutive iterations with < 2 successes "
            "(default: disabled)"
        ),
    )


# ---------------------------------------------------------------------------
# Iterative shortening loop
# ---------------------------------------------------------------------------

ShortenFunc = Callable[..., List[ShortenResult]]


def run_iterative_shortening(
    items: List[Dict[str, Any]],
    shorten_func: ShortenFunc,
    model: str,
    threshold: float,
    context_window: int,
    max_iterations: int,
    min_words: int,
    output_dir: Path,
    stem: str,
    reasoning_effort: Optional[str] = None,
    avg_chars_per_sec: float = DEFAULT_AVG_CHARS_PER_SEC,
    batch_size: int = DEFAULT_BATCH_SIZE,
    stuck_threshold: int = DEFAULT_STUCK_THRESHOLD,
    early_stop_patience: Optional[int] = None,
    context_items: Optional[List[Dict[str, Any]]] = None,
    selection_policy: Optional[DurationSelectionPolicy] = None,
) -> List[Dict[str, Any]]:
    """Run the iterative shortening loop.

    Features:
    - Batching: send multiple subtitles per API call.
    - Adaptive strategy: ``shorten`` for iterations 1-2, ``rephrase`` from 3+.
    - Stuck detection: skip subtitles unchanged for *stuck_threshold* iterations
      (counter resets on strategy change).
    - Early stopping: stop if fewer than 2 successes for
      *early_stop_patience* consecutive iterations (counter resets on
      strategy change).

    Args:
        items: Subtitle items (will be modified in-place via copies).
        shorten_func: Function that shortens subtitles.
        model: Model identifier.
        threshold: Mismatch ratio threshold.
        context_window: Number of surrounding subtitles for context.
        max_iterations: Maximum iterations.
        min_words: Minimum words in result.
        output_dir: Directory to save intermediate results.
        stem: Base filename stem for outputs.
        reasoning_effort: Optional reasoning effort level.
        batch_size: Number of subtitles per API call.
        stuck_threshold: Iterations without change before skipping.
        early_stop_patience: Stop after N iterations with < 2 successes.

    Returns:
        Final list of subtitle items.
    """
    batch_size = max(1, min(batch_size, MAX_BATCH_SIZE))

    print("\nRunning initial analysis...")
    preserve_saved_timing = context_items is not None
    if context_items is not None:
        items = hydrate_context_timing(items, context_items, avg_chars_per_sec, threshold)
    result = refresh_analysis(items, avg_chars_per_sec, threshold, preserve_saved_timing)
    items = result["items"]
    critical_count = len(select_subtitles_for_shortening(items, threshold, selection_policy=selection_policy))
    print(f"Critical subtitles: {critical_count}")

    analyzed_path = output_dir / f"{stem}_analyzed.json"
    save_json(analyzed_path, items)
    print(f"Saved analyzed: {analyzed_path}")

    current_items = items
    stuck_count: Dict[int, int] = {}
    prev_texts: Dict[int, str] = {}
    current_strategy = "shorten"
    patience_counter = 0

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{max_iterations} [strategy: {current_strategy}]")
        print(f"{'='*60}")

        new_strategy = "rephrase" if iteration >= 3 else "shorten"
        if new_strategy != current_strategy:
            print(f"Strategy changed: {current_strategy} -> {new_strategy}")
            current_strategy = new_strategy
            stuck_count = {}
            prev_texts = {}
            patience_counter = 0

        stuck_indices = {idx for idx, cnt in stuck_count.items() if cnt >= stuck_threshold}
        target_indices = select_subtitles_for_shortening(
            current_items, threshold, stuck_indices, selection_policy
        )

        if not target_indices:
            if stuck_indices:
                remaining = select_subtitles_for_shortening(current_items, threshold, selection_policy=selection_policy)
                if not remaining:
                    print("No subtitles need shortening. Done!")
                    break
                print(
                    f"All {len(remaining)} remaining subtitles are stuck. Done!"
                )
                break
            print("No subtitles need shortening. Done!")
            break

        print(
            f"Subtitles to shorten: {len(target_indices)}"
            f" (stuck: {len(stuck_indices)})"
        )

        iter_prev_texts: Dict[int, str] = {}
        for idx in target_indices:
            iter_prev_texts[idx] = join_text_lines(
                current_items[idx].get("text", "")
            )

        shorten_args = (
            current_items,
            target_indices,
            model,
            context_window,
            min_words,
            reasoning_effort,
            current_strategy,
            batch_size,
            avg_chars_per_sec,
            threshold,
        )
        results = shorten_func(*shorten_args, context_items) if context_items is not None else shorten_func(*shorten_args)

        success_count = sum(1 for r in results if r.success)
        changed_count = sum(
            1
            for r in results
            if r.success and r.shortened_text.strip() != r.original_text.strip()
        )
        print(f"\nShortened: {success_count}/{len(results)} (changed: {changed_count})")

        current_items = apply_shortening(current_items, results)

        for idx in target_indices:
            new_text = join_text_lines(current_items[idx].get("text", ""))
            if new_text == iter_prev_texts.get(idx):
                stuck_count[idx] = stuck_count.get(idx, 0) + 1
            else:
                stuck_count[idx] = 0

        print("Re-analyzing...")
        result = refresh_analysis(
            current_items, avg_chars_per_sec, threshold, preserve_saved_timing
        )
        current_items = result["items"]
        critical_after = len(select_subtitles_for_shortening(
            current_items, threshold, selection_policy=selection_policy
        ))
        print(f"Critical after re-analysis: {critical_after}")

        iter_path = output_dir / f"{stem}_iter{iteration:02d}.json"
        save_json(iter_path, current_items)
        print(f"Saved: {iter_path}")

        if early_stop_patience is not None:
            if changed_count < 2:
                patience_counter += 1
            else:
                patience_counter = 0
            if patience_counter >= early_stop_patience:
                print(
                    f"\nEarly stop: {early_stop_patience} consecutive "
                    f"iterations with < 2 changes"
                )
                break

    final_path = output_dir / f"{stem}_final.json"
    save_json(final_path, current_items)
    print(f"\nFinal result: {final_path}")

    final_result = refresh_analysis(
        current_items, avg_chars_per_sec, threshold, preserve_saved_timing
    )
    final_critical = len(select_subtitles_for_shortening(
        final_result["items"], threshold, selection_policy=selection_policy
    ))
    print(f"\nSummary:")
    print(f"  Total subtitles: {len(current_items)}")
    print(f"  Critical at start: {critical_count}")
    print(f"  Critical at end: {final_critical}")

    return current_items
