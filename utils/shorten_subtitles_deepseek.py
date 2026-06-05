"""Subtitle shortening using DeepSeek API (openai-compatible).

Usage:
  uv run -m utils.shorten_subtitles_deepseek input.json --model deepseek-v4-flash
  uv run -m utils.shorten_subtitles_deepseek input.json --model deepseek-v4-flash --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any, Dict, List, Optional

from utils.shorten_helpers import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    ShortenFunc,
    ShortenResult,
    add_common_args,
    build_context,
    build_user_prompt,
    load_api_key,
    load_json,
    parse_shortened_response,
    run_iterative_shortening,
    validate_shortened_text,
)
from utils.analyze_text import join_text_lines


# ---------------------------------------------------------------------------
# DeepSeek API callers
# ---------------------------------------------------------------------------


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
            max_tokens=2000,
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
                    max_tokens=2000,
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
                max_tokens=2000,
                extra_body=extra_body if extra_body else None,
            )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  DeepSeek API error: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ShortenFunc adapter
# ---------------------------------------------------------------------------


def make_deepseek_shorten_func(
    api_key: str,
    base_url: str,
    concurrency: int,
) -> ShortenFunc:
    """Create a ShortenFunc adapter for DeepSeek API."""

    def _shorten(
        items: List[Dict[str, Any]],
        target_indices: List[int],
        model: str,
        context_window: int,
        min_words: int,
        reasoning_effort: Optional[str] = None,
    ) -> List[ShortenResult]:
        return asyncio.run(shorten_subtitles_deepseek(
            items, target_indices, model, context_window, min_words,
            api_key, base_url, concurrency, reasoning_effort,
        ))

    return _shorten


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Iteratively shorten subtitles using DeepSeek API."
    )
    add_common_args(parser)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max parallel API calls (default: {DEFAULT_CONCURRENCY})",
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

    args = parser.parse_args(argv)

    from pathlib import Path

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load API key
    api_key = args.api_key
    if not api_key:
        api_key = load_api_key("DEEPSEEK_API_KEY")
        if not api_key:
            print("ERROR: DeepSeek API key not found.", file=sys.stderr)
            print("Set DEEPSEEK_API_KEY in .env file or environment.", file=sys.stderr)
            return 1

    # Load input
    print(f"Loading: {input_path}")
    items = load_json(input_path)
    print(f"Total subtitles: {len(items)}")

    # Create shorten function
    shorten_func = make_deepseek_shorten_func(api_key, args.base_url, args.concurrency)

    # Run
    stem = input_path.stem
    run_iterative_shortening(
        items=items,
        shorten_func=shorten_func,
        model=args.model,
        threshold=args.threshold,
        context_window=args.context_window,
        max_iterations=args.max_iterations,
        min_words=args.min_words,
        output_dir=output_dir,
        stem=stem,
        reasoning_effort=args.reasoning_effort,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
