"""Subtitle shortening using DeepSeek API (openai-compatible).

Usage:
  uv run -m utils.shorten_subtitles_deepseek input.json --model deepseek-v4-flash
  uv run -m utils.shorten_subtitles_deepseek input.json --model deepseek-v4-flash --concurrency 10
  uv run -m utils.shorten_subtitles_deepseek input.json --batch-size 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.shorten_helpers import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCURRENCY,
    DEFAULT_MODEL,
    MAX_BATCH_SIZE,
    ShortenFunc,
    ShortenResult,
    add_common_args,
    build_batch_context,
    build_batch_user_prompt,
    calculate_budget,
    get_system_prompt,
    load_api_key,
    load_json,
    parse_batch_response,
    run_iterative_shortening,
    target_min_words,
    validate_shortened_text,
    validation_error,
    validate_context_source,
)
from utils.analyze_text import join_text_lines
from utils.shortening_domain import DurationSelectionPolicy, load_calibrated_duration_profile


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
    max_tokens: int = 4000,
) -> Optional[str]:
    """Call DeepSeek API synchronously."""
    try:
        from openai import OpenAI
    except ImportError:
        print("  openai package not installed", file=sys.stderr, flush=True)
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        extra_body: Dict[str, Any] = {"thinking": {"type": "disabled"}}
        if reasoning_effort in {"high", "max"}:
            extra_body = {"thinking": {"type": "enabled"}, "reasoning_effort": reasoning_effort}

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            extra_body=extra_body,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  DeepSeek API error: {e}", file=sys.stderr)
        return None


def dynamic_max_tokens(max_chars: List[int]) -> int:
    """Set a conservative Cyrillic-safe cap including per-result JSON overhead."""
    return max(192, min(4096, sum(max_chars) + 80 * len(max_chars) + 32))


def _is_retryable_api_error(exc: BaseException) -> bool:
    """Classify rate limits, server failures and known transport errors."""
    status = getattr(exc, "status_code", None)
    if status == 429 or isinstance(status, int) and status >= 500:
        return True
    retryable_names = {
        "APITimeoutError", "APIConnectionError", "ConnectError", "ConnectTimeout",
        "ReadTimeout", "WriteTimeout", "PoolTimeout", "NetworkError",
    }
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in retryable_names or isinstance(
            current, (TimeoutError, OSError, asyncio.TimeoutError)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def usage_values(usage: Any) -> Dict[str, int]:
    """Normalize DeepSeek/OpenAI usage objects."""
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "prompt_cache_hit_tokens": int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0),
        "prompt_cache_miss_tokens": int(getattr(usage, "prompt_cache_miss_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(details, "reasoning_tokens", 0) or 0),
    }


def usage_summary(total: Dict[str, int]) -> Dict[str, Any]:
    """Add a DeepSeek V4 Flash direct-API cost estimate."""
    cost = (total.get("prompt_cache_miss_tokens", 0) * 0.14 +
            total.get("prompt_cache_hit_tokens", 0) * 0.0028 +
            total.get("completion_tokens", 0) * 0.28) / 1_000_000
    return {**total, "estimated_cost_usd": round(cost, 8),
            "pricing_model": "deepseek-v4-flash", "pricing_usd_per_million": {
        "input_cache_miss": 0.14, "input_cache_hit": 0.0028, "output": 0.28}}


async def shorten_via_deepseek_async(
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    semaphore: Optional[asyncio.Semaphore] = None,
    reasoning_effort: Optional[str] = None,
    max_tokens: int = 4000,
    client: Any = None,
    usage_total: Optional[Dict[str, int]] = None,
    api_retries: int = 2,
) -> Optional[str]:
    """Call DeepSeek API asynchronously."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("  openai package not installed", file=sys.stderr)
        return None

    client = client or AsyncOpenAI(
        api_key=api_key, base_url=base_url, timeout=60.0, max_retries=0
    )
    kwargs: Dict[str, Any] = {"model": model, "messages": [
        {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "max_tokens": max_tokens, "response_format": {"type": "json_object"}}
    if reasoning_effort in {"high", "max"}:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}, "reasoning_effort": reasoning_effort}
    else:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    for attempt in range(api_retries + 1):
        try:
            if semaphore:
                async with semaphore:
                    response = await client.chat.completions.create(**kwargs)
            else:
                response = await client.chat.completions.create(**kwargs)
            if usage_total is not None:
                for key, value in usage_values(response.usage).items():
                    usage_total[key] = usage_total.get(key, 0) + value
            return response.choices[0].message.content.strip()
        except Exception as exc:
            if not _is_retryable_api_error(exc) or attempt >= api_retries:
                print(
                    f"  DeepSeek API error: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                return None
            await asyncio.sleep((2 ** attempt) + random.random() * 0.2)
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
    strategy: str = "shorten",
    batch_size: int = DEFAULT_BATCH_SIZE,
    avg_chars_per_sec: float = 13.0,
    target_ratio: float = 1.0,
    usage_total: Optional[Dict[str, int]] = None,
    client: Any = None,
    context_items: Optional[List[Dict[str, Any]]] = None,
) -> List[ShortenResult]:
    """Shorten subtitles in parallel via DeepSeek API with batching."""
    system_prompt = get_system_prompt(strategy, min_words)
    semaphore = asyncio.Semaphore(concurrency)
    owns_client = client is None
    if owns_client:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=60.0, max_retries=0
        )
    batch_size = max(1, min(batch_size, MAX_BATCH_SIZE))

    batches = [
        target_indices[i : i + batch_size]
        for i in range(0, len(target_indices), batch_size)
    ]
    total_batches = len(batches)

    async def process_batch(
        batch_indices: List[int], batch_pos: int
    ) -> List[ShortenResult]:
        sub_numbers = [
            items[idx].get("index", idx + 1) for idx in batch_indices
        ]
        print(
            f"  [{batch_pos}/{total_batches}] Batch: "
            f"#{', #'.join(str(n) for n in sub_numbers)}"
        )

        budgets = {idx: calculate_budget(items[idx], target_ratio, avg_chars_per_sec) for idx in batch_indices}
        target_minimums = {
            idx: target_min_words(
                join_text_lines(items[idx].get("text", "")), min_words
            )
            for idx in batch_indices
        }
        pending = list(batch_indices)
        completed: Dict[int, ShortenResult] = {}
        deferred: Dict[int, ShortenResult] = {}
        last_errors: Dict[int, str] = {}
        feedback = ""
        for targeted_attempt in range(2):
            batch_context = build_batch_context(items, pending, context_window, context_items)
            targets = []
            for idx in pending:
                target = {
                    "index": items[idx].get("index", idx + 1),
                    **budgets[idx].__dict__,
                    "min_words": target_minimums[idx],
                }
                if targeted_attempt > 0 and last_errors.get(idx, "").startswith("too_long:"):
                    retry_max_chars = max(1, int(budgets[idx].max_chars * 0.9))
                    target["max_chars"] = retry_max_chars
                    target["required_reduction_percent"] = round(
                        max(
                            0.0,
                            (1.0 - retry_max_chars / budgets[idx].current_chars)
                            * 100.0,
                        ),
                        1,
                    )
                targets.append(target)
            user_prompt = build_batch_user_prompt(batch_context, min_words, strategy, targets) + feedback
            response = await shorten_via_deepseek_async(
            model,
            system_prompt,
            user_prompt,
            api_key,
            base_url,
            semaphore,
            reasoning_effort,
            dynamic_max_tokens([budgets[idx].max_chars for idx in pending]), client, usage_total,
            )
            results_map = parse_batch_response(response, pending, items)
            failed: List[int] = []
            errors: List[str] = []
            for idx in pending:
                original_text = join_text_lines(items[idx].get("text", ""))
                shortened = results_map.get(idx, "")
                error = validation_error(
                    original_text,
                    shortened,
                    target_minimums[idx],
                    budgets[idx].max_chars,
                )
                if error is None:
                    completed[idx] = ShortenResult(idx, original_text, shortened, True)
                    last_errors.pop(idx, None)
                elif error == "unchanged":
                    deferred[idx] = ShortenResult(
                        idx, original_text, original_text, False, "safe_deferred_unchanged"
                    )
                    last_errors.pop(idx, None)
                else:
                    failed.append(idx)
                    last_errors[idx] = error
                    errors.append(f"#{items[idx].get('index', idx + 1)}:{error}")
            pending = failed
            if not pending:
                break
            feedback = "\nИсправь только перечисленные цели. Ошибки: " + ", ".join(errors)

        results: List[ShortenResult] = []
        for idx in batch_indices:
            original_text = join_text_lines(items[idx].get("text", ""))
            results.append(completed.get(
                idx,
                deferred.get(idx, ShortenResult(
                    idx,
                    original_text,
                    original_text,
                    False,
                    last_errors.get(idx, "targeted_retry_failed"),
                )),
            ))

        batch_success = sum(1 for r in results if r.success)
        unresolved = [
            f"#{items[r.index].get('index', r.index + 1)}:{r.error}"
            for r in results
            if not r.success
        ]
        if unresolved:
            print(f"    unresolved: {', '.join(unresolved)}")
        print(
            f"    -> {batch_success}/{len(results)} shortened"
        )
        return results

    tasks = [
        process_batch(batch, i + 1) for i, batch in enumerate(batches)
    ]
    try:
        batch_results = await asyncio.gather(*tasks)
        return [r for batch in batch_results for r in batch]
    finally:
        if owns_client:
            await client.close()


# ---------------------------------------------------------------------------
# ShortenFunc adapter
# ---------------------------------------------------------------------------


def make_deepseek_shorten_func(
    api_key: str,
    base_url: str,
    concurrency: int,
    usage_total: Optional[Dict[str, int]] = None,
) -> ShortenFunc:
    """Create a ShortenFunc adapter for DeepSeek API."""

    def _shorten(
        items: List[Dict[str, Any]],
        target_indices: List[int],
        model: str,
        context_window: int,
        min_words: int,
        reasoning_effort: Optional[str] = None,
        strategy: str = "shorten",
        batch_size: int = DEFAULT_BATCH_SIZE,
        runtime_avg_chars_per_sec: float = 13.0,
        runtime_target_ratio: float = 1.0,
        context_items: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ShortenResult]:
        return asyncio.run(
            shorten_subtitles_deepseek(
                items,
                target_indices,
                model,
                context_window,
                min_words,
                api_key,
                base_url,
                concurrency,
                reasoning_effort,
                strategy,
                batch_size,
                runtime_avg_chars_per_sec,
                runtime_target_ratio,
                usage_total,
                context_items=context_items,
            )
        )

    return _shorten


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the direct-DeepSeek CLI parser, including experimental flags."""
    parser = argparse.ArgumentParser(
        description="Iteratively shorten subtitles using DeepSeek API."
    )
    add_common_args(parser)
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Max parallel API calls (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument("--api-key", help="DeepSeek API key (overrides .env and env var)")
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="DeepSeek API base URL")
    parser.add_argument(
        "--thinking-effort", choices=["high", "max"], default=None,
        help="Explicitly enable DeepSeek thinking (default: disabled)",
    )
    parser.add_argument("--duration-profile", type=Path, help="Opt-in calibrated post-silence duration profile")
    parser.add_argument("--duration-fit-ratio", type=float, default=1.0,
                        help="Required duration fit ratio (default: 1.0)")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    selection_policy: DurationSelectionPolicy | None = None
    try:
        if args.duration_fit_ratio <= 0 or not math.isfinite(args.duration_fit_ratio):
            raise ValueError("duration-fit-ratio must be finite and positive")
        if args.duration_profile is not None:
            selection_policy = DurationSelectionPolicy(
                load_calibrated_duration_profile(args.duration_profile), args.duration_fit_ratio
            )
    except ValueError as exc:
        print(f"ERROR: invalid duration selection configuration: {exc}", file=sys.stderr)
        return 2

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key
    if not api_key:
        api_key = load_api_key("DEEPSEEK_API_KEY")
        if not api_key:
            print("ERROR: DeepSeek API key not found.", file=sys.stderr)
            print("Set DEEPSEEK_API_KEY in .env file or environment.", file=sys.stderr)
            return 1

    print(f"Loading: {input_path}")
    items = load_json(input_path)
    context_items = load_json(Path(args.context_source)) if args.context_source else None
    if context_items is not None:
        validate_context_source(items, context_items)
    print(f"Total subtitles: {len(items)}")

    usage_total: Dict[str, int] = {}
    shorten_func = make_deepseek_shorten_func(api_key, args.base_url, args.concurrency, usage_total)

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
        reasoning_effort=args.thinking_effort,
        avg_chars_per_sec=args.avg_chars_per_sec,
        batch_size=args.batch_size,
        stuck_threshold=args.stuck_threshold,
        early_stop_patience=args.early_stop_patience,
        context_items=context_items,
        selection_policy=selection_policy,
    )
    summary = usage_summary(usage_total)
    summary["selection_mode"] = (
        "calibrated_post_silence_duration" if selection_policy is not None else "legacy_mismatch_ratio"
    )
    summary["fit_ratio"] = args.duration_fit_ratio
    if selection_policy is not None:
        model = selection_policy.model
        summary["duration_profile"] = {
            "measurement": model.measurement, "voice": model.voice, "rate": model.rate,
            "coefficients": {
                "intercept_sec": model.intercept_sec,
                "seconds_per_budget_char": model.seconds_per_budget_char,
                "seconds_per_punctuation": model.seconds_per_punctuation,
            },
            "training": {"sample_count": model.sample_count, "episode_count": model.episode_count},
            "validation": {"method": model.validation_method, "mae_sec": model.validation_mae_sec,
                           "rmse_sec": model.validation_rmse_sec, "r_squared": model.validation_r_squared},
        }
    summary["context_mode"] = "full" if context_items is not None else "sparse"
    summary["context_source"] = str(Path(args.context_source)) if args.context_source else None
    usage_path = output_dir / f"{stem}_final.usage.json"
    usage_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Usage/cost: {summary} (saved: {usage_path})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
