"""Iterative subtitle shortening using LLM.

Two modes:
  - opencode: sequential calls via opencode CLI
  - deepseek: parallel calls via DeepSeek API (openai-compatible)

Usage:
  uv run -m utils.shorten_subtitles input.json --mode opencode --model deepseek-v4-flash
  uv run -m utils.shorten_subtitles input.json --mode deepseek --model deepseek-v4-flash
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.analyze_text import analyze_subtitles, join_text_lines

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 1.5
DEFAULT_CONTEXT_WINDOW = 3
DEFAULT_MAX_ITERATIONS = 15
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_CONCURRENCY = 5
DEFAULT_REASONING_EFFORT = "max"
MIN_WORDS_RESULT = 3

SYSTEM_PROMPT = """\
Ты - помощник по сокращению субтитров для озвучки.

На вход ты получаешь:
- КОНТЕКСТ: субтитры до и после целевого (с таймкодами и номерами)
- ЦЕЛЕВОЙ СУБТИТР: субтитр, который нужно сократить

ЗАДАЧА: Сократи ЦЕЛЕВОЙ субтитр по смыслу, сохранив основную идею и тон.

ПРАВИЛА:
1. Сокращение ТОЛЬКО по смыслу, не механическое.
2. ЗАПРЕЩЕНО:
   - Просто добавлять "..." или обрезать слова на полуслове
   - Использовать скрипты, формулы или шаблоны
   - Менять порядок слов или смыслประโยка
3. Результат должен содержать минимум {min_words} слова (кроме случаев \
повторов, например "Вперёд, вперёд!").
4. Сохраняй тон и стиль оригинала.
5. Если субтитр уже достаточно короткий и его нельзя сократить без потери \
смысла - верни его без изменений.

Отвечай ТОЛЬКО валидным JSON без markdown-обёрток:
{{"shortened_text": "сокращённый текст здесь"}}
"""


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


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------


def build_context(
    items: List[Dict[str, Any]],
    target_idx: int,
    context_window: int,
) -> str:
    """Build context string with surrounding subtitles."""
    start = max(0, target_idx - context_window)
    end = min(len(items), target_idx + context_window + 1)

    lines: List[str] = []
    for i in range(start, end):
        item = items[i]
        idx = item.get("index", i + 1)
        start_ms = item.get("start", 0)
        end_ms = item.get("end", 0)
        text = join_text_lines(item.get("text", ""))

        ts = f"[{start_ms/1000:.1f}s - {end_ms/1000:.1f}s]"

        if i == target_idx:
            lines.append(f">>> ЦЕЛЕВОЙ СУБТИТР #{idx} {ts}:")
            lines.append(f">>> {text}")
            lines.append("<<<")
        else:
            role = "ДО" if i < target_idx else "ПОСЛЕ"
            lines.append(f"[{role}] Субтитр #{idx} {ts}: {text}")

    return "\n".join(lines)


def build_user_prompt(context: str, min_words: int) -> str:
    """Build user prompt with context."""
    return (
        f"{context}\n\n"
        f"Сократи ЦЕЛЕВОЙ СУБТИТР по смыслу. "
        f"Минимум {min_words} слов в результате."
    )


# ---------------------------------------------------------------------------
# Model callers
# ---------------------------------------------------------------------------


def shorten_via_opencode(
    model: str,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: Optional[str] = None,
) -> Optional[str]:
    """Call opencode CLI to shorten a subtitle."""
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    cmd = [
        "opencode",
        "run",
        "--model", model,
        "--format", "json",
    ]
    if reasoning_effort:
        cmd.extend(["--variant", reasoning_effort])
    cmd.append(full_prompt)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  opencode error: {result.stderr.strip()}", file=sys.stderr)
            return None

        # Parse JSON output from opencode
        output = result.stdout.strip()
        try:
            # Try to parse as JSON (opencode format)
            data = json.loads(output)
            # Extract assistant message from JSON events
            if isinstance(data, list):
                for event in data:
                    if isinstance(event, dict) and event.get("type") == "assistant":
                        content = event.get("content", "")
                        if content:
                            return content
            elif isinstance(data, dict):
                # Single message format
                content = data.get("content", "")
                if content:
                    return content
        except json.JSONDecodeError:
            # Not JSON, return as-is
            pass

        return output
    except FileNotFoundError:
        print("  opencode not found in PATH", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print("  opencode timeout", file=sys.stderr)
        return None


def shorten_via_deepseek_sync(
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    reasoning_effort: Optional[str] = None,
) -> Optional[str]:
    """Call DeepSeek API synchronously."""
    try:
        from openai import OpenAI
    except ImportError:
        print("  openai package not installed", file=sys.stderr)
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        extra_body = {}
        if reasoning_effort:
            extra_body["reasoning_effort"] = reasoning_effort

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            extra_body=extra_body if extra_body else None,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  DeepSeek API error: {e}", file=sys.stderr)
        return None


async def shorten_via_deepseek_async(
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    semaphore: Optional[asyncio.Semaphore] = None,
    reasoning_effort: Optional[str] = None,
) -> Optional[str]:
    """Call DeepSeek API asynchronously."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("  openai package not installed", file=sys.stderr)
        return None

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    try:
        extra_body = {}
        if reasoning_effort:
            extra_body["reasoning_effort"] = reasoning_effort

        if semaphore:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=200,
                    extra_body=extra_body if extra_body else None,
                )
        else:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=200,
                extra_body=extra_body if extra_body else None,
            )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  DeepSeek API error: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_shortened_response(response: Optional[str]) -> Optional[str]:
    """Parse model response and extract shortened text."""
    if not response:
        return None

    # Try to parse as JSON
    try:
        # Handle potential markdown code blocks
        text = response.strip()
        if text.startswith("```"):
            # Remove markdown code block
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        data = json.loads(text)
        shortened = data.get("shortened_text", "").strip()
        if shortened:
            return shortened
    except json.JSONDecodeError:
        pass

    # Fallback: try to find JSON in the response
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

    # Last resort: return the whole response if it looks like plain text
    if not response.startswith("{") and len(response.split()) >= MIN_WORDS_RESULT:
        return response.strip()

    return None


def validate_shortened_text(
    original: str,
    shortened: str,
    min_words: int = MIN_WORDS_RESULT,
) -> bool:
    """Validate that shortened text meets requirements."""
    if not shortened:
        return False

    # Check it's not just "..." or similar punctuation
    cleaned = shortened.replace(".", "").replace(",", "").replace("!", "").replace("?", "").strip()
    if not cleaned:
        return False

    words = shortened.split()
    if len(words) < min_words:
        # Check for repeated words (like "Вперёд, вперёд, вперёд!")
        # Need at least 2 words to consider it a repetition pattern
        if len(words) >= 2:
            unique_words = set(w.lower().strip(".,!?;:") for w in words)
            if len(unique_words) <= 1:
                # All words are the same repetition - allowed
                return True
        # Too few words and not a repetition pattern
        return False

    # Check it's not just truncated with "..."
    if shortened.endswith("...") and not original.endswith("..."):
        # Check if it's just original truncated
        before_ellipsis = shortened[:-3].strip()
        if original.startswith(before_ellipsis) and len(shortened) < len(original):
            return False

    return True


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_subtitles_for_shortening(
    items: List[Dict[str, Any]],
    threshold: float,
) -> List[int]:
    """Select subtitle indices that need shortening.

    Returns list of indices (0-based) into the items list.
    """
    selected: List[int] = []
    for i, item in enumerate(items):
        analysis = item.get("analysis", {})
        if not analysis.get("is_checked", False):
            continue

        mismatch = analysis.get("mismatch_ratio")
        extended = analysis.get("extended_mismatch_ratio")

        # Use extended if available, otherwise use mismatch
        effective = extended if extended is not None else mismatch

        if effective is not None and effective > threshold:
            selected.append(i)

    return selected


# ---------------------------------------------------------------------------
# Main shortening logic
# ---------------------------------------------------------------------------


def shorten_subtitles_opencode(
    items: List[Dict[str, Any]],
    target_indices: List[int],
    model: str,
    context_window: int,
    min_words: int,
    reasoning_effort: Optional[str] = None,
) -> List[ShortenResult]:
    """Shorten subtitles sequentially via opencode CLI."""
    system_prompt = SYSTEM_PROMPT.format(min_words=min_words)
    results: List[ShortenResult] = []

    for i, target_idx in enumerate(target_indices):
        item = items[target_idx]
        original_text = join_text_lines(item.get("text", ""))
        print(f"  [{i+1}/{len(target_indices)}] Subtitle #{item.get('index', target_idx+1)}: {original_text[:50]}...")

        context = build_context(items, target_idx, context_window)
        user_prompt = build_user_prompt(context, min_words)

        response = shorten_via_opencode(model, system_prompt, user_prompt, reasoning_effort)
        shortened = parse_shortened_response(response)

        if shortened and validate_shortened_text(original_text, shortened, min_words):
            results.append(ShortenResult(
                index=target_idx,
                original_text=original_text,
                shortened_text=shortened,
                success=True,
            ))
            print(f"    -> Shortened: {shortened[:50]}...")
        else:
            results.append(ShortenResult(
                index=target_idx,
                original_text=original_text,
                shortened_text=original_text,
                success=False,
                error="Failed to get valid shortened text",
            ))
            print(f"    -> Failed, keeping original")

    return results


async def shorten_subtitles_deepseek(
    items: List[Dict[str, Any]],
    target_indices: List[int],
    model: str,
    context_window: int,
    min_words: int,
    api_key: str,
    base_url: str,
    concurrency: int,
    reasoning_effort: Optional[str] = None,
) -> List[ShortenResult]:
    """Shorten subtitles in parallel via DeepSeek API."""
    system_prompt = SYSTEM_PROMPT.format(min_words=min_words)
    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(target_idx: int, position: int) -> ShortenResult:
        item = items[target_idx]
        original_text = join_text_lines(item.get("text", ""))
        print(f"  [{position}/{len(target_indices)}] Subtitle #{item.get('index', target_idx+1)}: {original_text[:50]}...")

        context = build_context(items, target_idx, context_window)
        user_prompt = build_user_prompt(context, min_words)

        response = await shorten_via_deepseek_async(
            model, system_prompt, user_prompt, api_key, base_url, semaphore, reasoning_effort
        )
        shortened = parse_shortened_response(response)

        if shortened and validate_shortened_text(original_text, shortened, min_words):
            print(f"    -> Shortened: {shortened[:50]}...")
            return ShortenResult(
                index=target_idx,
                original_text=original_text,
                shortened_text=shortened,
                success=True,
            )
        else:
            print(f"    -> Failed, keeping original")
            return ShortenResult(
                index=target_idx,
                original_text=original_text,
                shortened_text=original_text,
                success=False,
                error="Failed to get valid shortened text",
            )

    tasks = [process_one(idx, i+1) for i, idx in enumerate(target_indices)]
    results = await asyncio.gather(*tasks)
    return list(results)


def apply_shortening(
    items: List[Dict[str, Any]],
    results: List[ShortenResult],
) -> List[Dict[str, Any]]:
    """Apply shortening results to items, return new copy."""
    new_items = copy.deepcopy(items)

    for result in results:
        if result.success:
            item = new_items[result.index]
            # Preserve as list if original was list
            original_text = item.get("text", "")
            if isinstance(original_text, list):
                item["text"] = [result.shortened_text]
            else:
                item["text"] = result.shortened_text

    return new_items


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
# CLI
# ---------------------------------------------------------------------------


def load_api_key() -> Optional[str]:
    """Load API key from .env file or environment variable."""
    # Try to load from .env file
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
                        if key == "DEEPSEEK_API_KEY":
                            os.environ.setdefault(key, value)

    return os.environ.get("DEEPSEEK_API_KEY")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Iteratively shorten subtitles using LLM."
    )
    parser.add_argument("input", help="Input JSON file with analyzed subtitles")
    parser.add_argument(
        "--mode",
        choices=["opencode", "deepseek"],
        default="opencode",
        help="Mode: opencode CLI or DeepSeek API (default: opencode)",
    )
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
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max parallel API calls for deepseek mode (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output/shortened",
        help="Output directory (default: output/shortened)",
    )
    parser.add_argument(
        "--api-key",
        help="DeepSeek API key (overrides .env and env var)",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="DeepSeek API base URL",
    )
    parser.add_argument(
        "--analyze-args",
        default="",
        help="Extra args for analyze_text (e.g. '--avg-chars-per-sec 13')",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        choices=["minimal", "low", "medium", "high", "max"],
        help=f"Reasoning effort for model (default: {DEFAULT_REASONING_EFFORT})",
    )

    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load API key for deepseek mode
    api_key = args.api_key
    if args.mode == "deepseek" and not api_key:
        api_key = load_api_key()
        if not api_key:
            print("ERROR: DeepSeek API key not found.", file=sys.stderr)
            print("Set DEEPSEEK_API_KEY in .env file or environment.", file=sys.stderr)
            return 1

    # Load input
    print(f"Loading: {input_path}")
    items = load_json(input_path)
    print(f"Total subtitles: {len(items)}")

    # Initial analysis
    print("\nRunning initial analysis...")
    result = analyze_subtitles(items, avg_chars_per_sec=13.0, max_mismatch_ratio=1.3)
    items = result["items"]
    critical_count = len(result["critical_items"])
    print(f"Critical subtitles: {critical_count}")

    # Save analyzed version
    stem = input_path.stem
    analyzed_path = output_dir / f"{stem}_analyzed.json"
    save_json(analyzed_path, items)
    print(f"Saved analyzed: {analyzed_path}")

    # Iterative shortening
    current_items = items
    for iteration in range(1, args.max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{args.max_iterations}")
        print(f"{'='*60}")

        # Select subtitles for shortening
        target_indices = select_subtitles_for_shortening(current_items, args.threshold)

        if not target_indices:
            print("No subtitles need shortening. Done!")
            break

        print(f"Subtitles to shorten: {len(target_indices)}")

        # Shorten
        if args.mode == "opencode":
            results = shorten_subtitles_opencode(
                current_items,
                target_indices,
                args.model,
                args.context_window,
                args.min_words,
                args.reasoning_effort,
            )
        else:
            results = asyncio.run(shorten_subtitles_deepseek(
                current_items,
                target_indices,
                args.model,
                args.context_window,
                args.min_words,
                api_key,
                args.base_url,
                args.concurrency,
                args.reasoning_effort,
            ))

        # Apply results
        success_count = sum(1 for r in results if r.success)
        print(f"\nShortened: {success_count}/{len(results)}")

        current_items = apply_shortening(current_items, results)

        # Re-analyze
        print("Re-analyzing...")
        result = analyze_subtitles(current_items, avg_chars_per_sec=13.0, max_mismatch_ratio=1.3)
        current_items = result["items"]
        critical_after = len(result["critical_items"])
        print(f"Critical after re-analysis: {critical_after}")

        # Save iteration result
        iter_path = output_dir / f"{stem}_iter{iteration:02d}.json"
        save_json(iter_path, current_items)
        print(f"Saved: {iter_path}")

    # Save final result
    final_path = output_dir / f"{stem}_final.json"
    save_json(final_path, current_items)
    print(f"\nFinal result: {final_path}")

    # Summary
    final_result = analyze_subtitles(current_items, avg_chars_per_sec=13.0, max_mismatch_ratio=1.3)
    final_critical = len(final_result["critical_items"])
    print(f"\nSummary:")
    print(f"  Total subtitles: {len(current_items)}")
    print(f"  Critical at start: {critical_count}")
    print(f"  Critical at end: {final_critical}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
