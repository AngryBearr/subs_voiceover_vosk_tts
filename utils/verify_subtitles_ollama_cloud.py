"""Standalone semantic-unit subtitle verification through Ollama Cloud."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

from utils.ollama_cloud_transport import normalize_ollama_cloud_model_id, request_strict_json
from utils.opencode_transport import OpenCodePromptResult
from utils.shorten_helpers import load_api_key, load_json, validate_context_source
from utils.verify_subtitles_opencode import UNIT_VERIFIER_SYSTEM_PROMPT, _validate_items, _validate_runtime_args, verify_items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed Ollama Cloud semantic-unit subtitle verifier.")
    parser.add_argument("original")
    parser.add_argument("candidate")
    parser.add_argument("--context-source")
    parser.add_argument("--api-key")
    parser.add_argument("--model", required=True, help="Explicit Ollama Cloud catalog model ID.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--context-window", type=int, default=3)
    parser.add_argument("--transport-retries", type=int, default=1)
    parser.add_argument("--schema-retries", type=int, choices=[0, 1], default=1)
    parser.add_argument("--semantic-unit-max-cues", type=int, default=3)
    parser.add_argument("--semantic-unit-max-gap-sec", type=float, default=0.3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="output/ollama_cloud_semantic_verified")
    return parser


def _validate_config(args: argparse.Namespace) -> None:
    _validate_runtime_args(args.model, args.context_window, args.batch_size, args.concurrency, args.transport_retries,
                           args.schema_retries, args.semantic_unit_max_cues, args.semantic_unit_max_gap_sec)
    normalize_ollama_cloud_model_id(args.model)
    if (isinstance(args.timeout, bool) or not isinstance(args.timeout, (int, float))
            or not math.isfinite(float(args.timeout)) or args.timeout <= 0):
        raise ValueError("invalid_timeout")
    if (isinstance(args.temperature, bool) or not isinstance(args.temperature, (int, float))
            or not math.isfinite(float(args.temperature)) or not 0 <= args.temperature <= 2):
        raise ValueError("invalid_temperature")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
        raise ValueError("invalid_seed")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_config(args)
        original = load_json(Path(args.original))
        candidate = load_json(Path(args.candidate))
        context = load_json(Path(args.context_source)) if args.context_source else None
        _validate_items(original, candidate)
        if context is not None:
            validate_context_source(original, context)
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    api_key = args.api_key if args.api_key is not None else load_api_key("OLLAMA_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        print("OLLAMA_API_KEY is required", file=sys.stderr)
        return 1

    async def request(prompt: str, model: str) -> Optional[OpenCodePromptResult]:
        return await request_strict_json(api_key=api_key, model=model, system_prompt=UNIT_VERIFIER_SYSTEM_PROMPT,
                                         user_prompt=prompt, timeout=args.timeout, temperature=args.temperature, seed=args.seed)

    output, report, usage = asyncio.run(verify_items(
        original, candidate, model=args.model, context_items=context, context_window=args.context_window,
        batch_size=args.batch_size, concurrency=args.concurrency, transport_retries=args.transport_retries,
        schema_retries=args.schema_retries, request_callable=request, semantic_units=True,
        semantic_unit_max_cues=args.semantic_unit_max_cues, semantic_unit_max_gap_sec=args.semantic_unit_max_gap_sec,
        backend="ollama_cloud", accounting="subscription", cost_is_billing_authoritative=False,
    ))
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = Path(args.candidate).stem + "_ollama_cloud_verified"
    for suffix, value in ((".json", output), (".report.json", report), (".usage.json", usage)):
        (destination / f"{stem}{suffix}").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
