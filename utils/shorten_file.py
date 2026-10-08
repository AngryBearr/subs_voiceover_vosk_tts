"""Run one ordinary full-file subtitle shortening job.

The command performs all input, model, and frozen-calibration checks before it
creates an output directory or constructs a live transport.  The same model
specification is passed to generation and review by the existing runner.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import functools
import hashlib
import json
import math
import os
import re
import signal
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from utils.deepseek_limits import resolve_deepseek_token_limit
from utils.llm_types import DetailedResponse
from utils import openrouter_transport
from utils.shortening_audio import verify_original_timing
from utils.shortening_backends import DEEPSEEK_VARIANTS, OPENAI_MODELS, LiveBackend
from utils.shortening_plan import CuePlan, PlannerConfig, plan_subtitles
from utils.shortening_runner import ModelSpec, RunConfig, run_trial
from utils.shortening_budget import BudgetPolicy
from utils.shortening_resume import ResumeError, UnknownRequestsError, load_manifest_contract, validate_inherited_evidence
from utils.opencode_clean import CleanOpenCodeServer
from utils.srt_to_json import SRTParseError, parse_srt_file
from utils.vosk_calibration import (
    model_metadata,
    profile_identity,
    validate_character_profile,
)


MODEL_CHOICES = (
    "openai/gpt-5.6-luna",
    "openai/gpt-5.6-sol",
    "deepseek/deepseek-flash",
    "deepseek/deepseek-v4-flash",
)
DEFAULT_MODEL_PATH = Path("models/vosk-model-tts-ru-0.10-multi")
DEFAULT_VOICE = 8
DEFAULT_RATE = 1.5
PRIMARY_GENERATOR = ModelSpec("openrouter", "meta/muse-spark-1.3-contributor", "low")
PRIMARY_ARBITER = ModelSpec("openrouter", "openai/gpt-6-sol", "high")
PRIMARY_MAX_OUTPUT_TOKENS = 16_384


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_decimal_string(value: Any) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _asdict_model(spec: ModelSpec) -> dict[str, str]:
    return asdict(spec)


def _asdict_plan(plan: CuePlan) -> dict[str, Any]:
    return asdict(plan)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_source(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"input file does not exist: {path}")
    raw = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            value = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("input JSON is not valid UTF-8 JSON") from exc
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise ValueError("input JSON must be an array of objects")
        return value, "json", _sha256_bytes(raw)
    if suffix == ".srt":
        # parse_srt_file itself uses utf-8-sig and raises SRTParseError for a
        # malformed block instead of silently dropping it.
        try:
            block_text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError as exc:
            raise ValueError("SRT is not valid UTF-8") from exc
        items = parse_srt_file(str(path))
        block_count = len([block for block in re.split(r"\n{2,}", block_text.strip()) if block.strip()])
        if not isinstance(items, list) or len(items) != block_count:
            raise ValueError("SRT parser entry count does not match source block count")
        return items, "srt", _sha256_bytes(raw)
    raise ValueError("input format must have a .json or .srt suffix")


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite and positive" if positive else f"{name} must be finite")
    return result


def _validate_args(args: argparse.Namespace) -> tuple[float, ModelSpec]:
    rate = _finite_number(getattr(args, "rate", DEFAULT_RATE), "rate")
    if not 1.0 <= rate <= 1.5:
        raise ValueError("rate must be between 1.0 and 1.5")
    voice = getattr(args, "voice", DEFAULT_VOICE)
    if isinstance(voice, bool) or not isinstance(voice, int) or voice < 0:
        raise ValueError("voice must be a non-negative integer")
    model = getattr(args, "model", None)
    primary = model is None
    timeout = getattr(args, "timeout", None)
    _finite_number(300.0 if timeout is None else timeout,
                   "timeout", positive=True)
    for name in ("gen_batch_size", "review_batch_size", "max_context_chars"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    max_repairs = getattr(args, "max_repairs")
    if isinstance(max_repairs, bool) or not isinstance(max_repairs, int) or not 0 <= max_repairs <= 2:
        raise ValueError("max_repairs must be between 0 and 2")
    if getattr(args, "text_mode", "punctuation") != "punctuation":
        raise ValueError("text_mode must be punctuation")
    max_tokens = getattr(args, "max_tokens", None)
    if primary:
        if max_repairs != 1:
            raise ValueError("primary OpenRouter recovery is fixed at max_repairs=1")
        if (300.0 if timeout is None else float(timeout)) > 300.0:
            raise ValueError("primary OpenRouter timeout must not exceed 300 seconds")
        if getattr(args, "variant", None) is not None:
            raise ValueError("--variant is legacy-only and cannot override the primary OpenRouter roles")
        if bool(getattr(args, "length_tools", False)):
            raise ValueError("--length-tools is legacy-only and disabled for the primary OpenRouter roles")
        if max_tokens is not None:
            raise ValueError("--max-tokens is legacy-only; primary output is fixed at 16384")
        return rate, PRIMARY_GENERATOR
    if max_tokens is not None:
        resolve_deepseek_token_limit("off", max_tokens)
    if model not in MODEL_CHOICES:
        raise ValueError("a supported qualified model ID is required")
    variant = getattr(args, "variant", None)
    if model.startswith("openai/"):
        selected_variant = "medium" if variant is None else variant
        if selected_variant not in OPENAI_MODELS[model]:
            raise ValueError(f"unsupported model variant: {model}:{selected_variant}")
        spec = ModelSpec("opencode", model, selected_variant)
    else:
        selected_variant = "off" if variant is None else variant
        if selected_variant not in DEEPSEEK_VARIANTS:
            raise ValueError(f"unsupported model variant: {model}:{selected_variant}")
        spec = ModelSpec("deepseek", model.split("/", 1)[1], selected_variant)
    return rate, spec


def _validate_model_directory(path: Path, voice: int) -> dict[str, Any]:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"model path does not exist or is not a directory: {path}")
    required = (path / "model.onnx", path / "dictionary", path / "config.json")
    missing = [item.name for item in required if not item.is_file()]
    if missing:
        raise ValueError(f"model directory is missing required files: {', '.join(missing)}")
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Vosk model config.json is not valid UTF-8 JSON") from exc
    if not isinstance(config, dict):
        raise ValueError("Vosk model config.json must contain an object")
    speaker_map = config.get("speaker_id_map")
    if isinstance(speaker_map, dict) and speaker_map:
        ids = {
            value for value in speaker_map.values()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        if ids and voice not in ids:
            raise ValueError(f"voice {voice} is not available in Vosk model config")
    elif isinstance(config.get("num_speakers"), int) and not isinstance(config.get("num_speakers"), bool):
        count = int(config["num_speakers"])
        if count > 0 and voice >= count:
            raise ValueError(f"voice {voice} is not available in Vosk model config")
    return model_metadata(path)


def _select_profile(raw: dict[str, Any], rate: float) -> dict[str, Any]:
    if "profiles" not in raw:
        return raw
    if raw.get("schema_version") != 1 or raw.get("status") != "complete":
        raise ValueError("calibration report is incomplete or has an unsupported schema")
    profiles = raw.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("calibration report profiles must be an object")
    matches: list[dict[str, Any]] = []
    for key, value in profiles.items():
        try:
            key_rate = float(key)
        except (TypeError, ValueError):
            raise ValueError("calibration report contains a non-numeric profile rate") from None
        if not math.isfinite(key_rate):
            raise ValueError("calibration report contains a non-finite profile rate")
        if key_rate == rate:
            if not isinstance(value, dict):
                raise ValueError("selected calibration profile must be an object")
            matches.append(value)
    if len(matches) != 1:
        raise ValueError("calibration report must contain exactly one profile for the requested rate")
    return matches[0]


def _load_profile(path: Path, rate: float) -> tuple[dict[str, Any], str]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"profile file does not exist: {path}")
    raw_bytes = path.read_bytes()
    try:
        raw = json.loads(raw_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("profile is not valid UTF-8 JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("profile must contain a JSON object")
    profile = _select_profile(raw, rate)
    if profile.get("rate") != rate:
        raise ValueError("selected calibration profile rate does not match --rate")
    return profile, _sha256_bytes(raw_bytes)


def _voice(args: argparse.Namespace) -> int:
    return int(getattr(args, "voice", DEFAULT_VOICE))


@dataclass(frozen=True)
class PreparedRun:
    """Fully validated, offline description of one ordinary run."""

    input_path: Path
    input_format: str
    input_bytes_sha256: str
    items: list[dict[str, Any]]
    plans: list[CuePlan]
    target_ids: list[str]
    output_dir: Path
    model_path: Path
    voice: int
    rate: float
    timeout: float
    run_time_limit_sec: float | None
    max_tokens: int | None
    text_mode: str
    profile_path: Path
    profile_file_sha256: str
    profile: dict[str, Any]
    identity: dict[str, Any]
    model: ModelSpec
    planner_config: PlannerConfig
    run_config: RunConfig
    metadata: dict[str, Any]
    dry_run: bool
    resume: bool = False
    retry_unknown_ids: tuple[str, ...] = ()
    arbiter: ModelSpec | None = None
    budget_policy: BudgetPolicy | None = None
    primary: bool = False


def prepare_run(args: argparse.Namespace) -> PreparedRun:
    """Validate and plan a run without credentials, servers, inference, or output writes."""
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    profile_path = Path(args.profile)
    resume = bool(getattr(args, "resume", False))
    retry_unknown_ids = tuple(getattr(args, "retry_unknown_ids", ()) or ())
    if retry_unknown_ids and not resume:
        raise ResumeError("--retry-unknown requires --resume")
    if resume and bool(getattr(args, "dry_run", False)):
        raise ResumeError("--resume cannot be combined with --dry-run")
    if resume:
        if not output_dir.exists() or not output_dir.is_dir():
            raise ResumeError(f"resume output directory does not exist or is not a directory: {output_dir}")
    elif output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if output_dir.resolve() in {input_path.resolve(), profile_path.resolve()}:
        raise ValueError("output directory must not be the input or profile path")
    rate, model = _validate_args(args)
    primary = getattr(args, "model", None) is None
    budget_usd = getattr(args, "budget_usd", None)
    budget_limit: str | None = None
    max_requests = getattr(args, "max_requests", 1000)
    if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests <= 0:
        raise ValueError("max_requests must be a positive integer")
    if not primary and budget_usd is not None:
        raise ValueError("--budget-usd is for the primary OpenRouter route; omit --model")
    if not primary and max_requests != 1000:
        raise ValueError("--max-requests is for the primary OpenRouter route; omit --model")
    if primary and not bool(getattr(args, "dry_run", False)):
        if budget_usd is None:
            raise ValueError("--budget-usd is required for the primary live run")
    run_time_limit_sec = getattr(args, "run_time_limit", None)
    if run_time_limit_sec is not None:
        if not primary:
            raise ValueError("--run-time-limit is primary-only")
        run_time_limit_sec = _finite_number(run_time_limit_sec, "run_time_limit", positive=True)
    if budget_usd is not None:
        try:
            from decimal import Decimal, InvalidOperation
            budget_decimal = Decimal(str(budget_usd))
        except (InvalidOperation, ValueError):
            raise ValueError("budget_usd must be a finite positive decimal") from None
        if not budget_decimal.is_finite() or budget_decimal <= 0:
            raise ValueError("budget_usd must be a finite positive decimal")
        budget_limit = _canonical_decimal_string(budget_decimal)
    voice = _voice(args)
    items, input_format, input_bytes_sha256 = _load_source(input_path)
    model_path = Path(getattr(args, "model_path", DEFAULT_MODEL_PATH))
    frozen_model_metadata = _validate_model_directory(model_path, voice)
    profile, profile_file_sha256 = _load_profile(profile_path, rate)
    identity = profile_identity(frozen_model_metadata, voice, rate)
    validate_character_profile(profile, expected_identity=identity)
    planner_config = PlannerConfig(
        length_basis="normalized_tts_text",
        calibration_profile=copy.deepcopy(profile),
        calibration_identity=copy.deepcopy(identity),
        max_speed=rate,
    )
    plans = plan_subtitles(items, planner_config)
    target_ids = [
        plan.cue_id for plan in plans
        if plan.status in {"needs_shortening", "short_timing"}
    ]
    run_config = RunConfig(
        gen_batch_size=args.gen_batch_size,
        review_batch_size=args.review_batch_size,
        max_repairs=args.max_repairs,
        max_context_chars=args.max_context_chars,
        length_tools=args.length_tools,
        scheduling="windowed" if primary else "stage_major",
    )
    profile_canonical = _canonical_json(profile)
    profile_sha256 = _sha256_bytes(profile_canonical.encode("utf-8"))
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "input": {
            "path": str(input_path),
            "absolute_path": str(input_path.resolve()),
            "sha256": input_bytes_sha256,
            "format": input_format,
        },
        "calibration": {
            "path": str(profile_path),
            "absolute_path": str(profile_path.resolve()),
            "file_sha256": profile_file_sha256,
            "profile_sha256": profile_sha256,
            "rate": rate,
            "identity": copy.deepcopy(identity),
            "profile": copy.deepcopy(profile),
        },
        "model_path": str(model_path),
        "model_metadata": frozen_model_metadata,
        "voice": voice,
        "rate": rate,
        "text_mode": "punctuation",
        "length_basis": planner_config.length_basis,
        "config": {"planner": asdict(planner_config), "run": asdict(run_config)},
    }
    timeout = 300.0 if getattr(args, "timeout", None) is None else float(args.timeout)
    metadata["config"]["run"].update({
        "timeout": timeout,
        "max_tokens": getattr(args, "max_tokens", None),
        "run_time_limit_sec": run_time_limit_sec,
    })
    if primary:
        metadata["roles"] = {"generator": _asdict_model(PRIMARY_GENERATOR), "arbiter": _asdict_model(PRIMARY_ARBITER)}
        metadata["openrouter"] = {"roles": copy.deepcopy(metadata["roles"]), "max_output_tokens": PRIMARY_MAX_OUTPUT_TOKENS}
        metadata["budget"] = {
            "limit_usd": budget_limit,
            "max_requests": max_requests,
            "reservation_basis": "published_context_and_max_rates",
            "max_unknown_calls": max_requests,
            "retry_policy": "one explicit replacement per unknown request; one queued recovery per failed cue/stage",
        }
    frozen_metadata = metadata
    if primary and resume:
        contract = load_manifest_contract(output_dir)
        if isinstance(contract.get("report_metadata"), dict) and "migration" in contract["report_metadata"]:
            validate_inherited_evidence(output_dir, contract["report_metadata"]["migration"])
        frozen_metadata_value = contract.get("report_metadata")
        if not isinstance(frozen_metadata_value, dict):
            raise ResumeError("primary resume manifest has no frozen report metadata")
        frozen_openrouter = frozen_metadata_value.get("openrouter")
        frozen_budget = frozen_metadata_value.get("budget")
        if not isinstance(frozen_openrouter, dict) or not isinstance(frozen_budget, dict):
            raise ResumeError("primary resume manifest has no frozen OpenRouter budget contract")
        if frozen_openrouter.get("roles") != metadata["openrouter"]["roles"]:
            raise ResumeError("resume OpenRouter roles differ from the frozen contract")
        if frozen_openrouter.get("max_output_tokens") != PRIMARY_MAX_OUTPUT_TOKENS:
            raise ResumeError("resume OpenRouter output cap differs from the frozen contract")
        if frozen_budget.get("max_requests") != max_requests:
            raise ResumeError("resume max_requests differs from the frozen contract")
        if frozen_budget.get("max_unknown_calls") != max_requests or contract.get("run", {}).get("scheduling") != "windowed":
            raise ResumeError("older primary scheduling/unknown policy cannot be resumed by this CLI")
        if frozen_metadata_value.get("config", {}).get("run", {}).get("run_time_limit_sec") != run_time_limit_sec:
            raise ResumeError("resume run-time limit differs from the frozen contract")
        requested_budget = budget_limit
        if requested_budget != frozen_budget.get("limit_usd"):
            raise ResumeError("resume budget differs from the frozen contract")
        frozen_metadata = copy.deepcopy(frozen_metadata_value)
    return PreparedRun(
        input_path=input_path,
        input_format=input_format,
        input_bytes_sha256=input_bytes_sha256,
        items=items,
        plans=plans,
        target_ids=target_ids,
        output_dir=output_dir,
        model_path=model_path,
        voice=voice,
        rate=rate,
        timeout=timeout,
        run_time_limit_sec=run_time_limit_sec,
        max_tokens=getattr(args, "max_tokens", None),
        text_mode="punctuation",
        profile_path=profile_path,
        profile_file_sha256=profile_file_sha256,
        profile=profile,
        identity=identity,
        model=model,
        planner_config=planner_config,
        run_config=run_config,
        metadata=frozen_metadata,
        dry_run=bool(getattr(args, "dry_run", False)),
        resume=resume,
        retry_unknown_ids=retry_unknown_ids,
        arbiter=PRIMARY_ARBITER if primary else model,
        primary=primary,
    )


def _dry_report(prepared: PreparedRun) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_state": "dry_run",
        "model": {"generator": _asdict_model(prepared.model), "arbiter": _asdict_model(prepared.arbiter or prepared.model)},
        "config": prepared.metadata["config"],
        "metadata": prepared.metadata,
        "plans": [_asdict_plan(plan) for plan in prepared.plans],
        "selected_ids": list(prepared.target_ids),
        "selected_count": len(prepared.target_ids),
        "srt_renumbering": "SRT export uses canonical chronological numbering; JSON indexes remain unchanged.",
        "evidence_statement": "No LLM, acoustic, or final-audio evidence has been collected; this is offline planning only.",
        "dry_run_note": "Use a separate new output directory for a live run; resume requires a completed immutable manifest and baseline.",
    }


class _LazyRequest:
    """Create credentials, backend, and OpenCode only when the first request exists."""

    def __init__(self, prepared: PreparedRun) -> None:
        self.prepared = prepared
        self.server: CleanOpenCodeServer | None = None
        self.backend: LiveBackend | None = None
        self.server_started = False

    def _make_backend(self) -> LiveBackend:
        if self.backend is not None:
            return self.backend
        if self.prepared.primary:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
            key = os.environ.get("OPENROUTER_API_KEY")
            snapshot = self.prepared.metadata.get("openrouter", {}).get("pricing_snapshot")
            if not isinstance(snapshot, dict):
                raise RuntimeError("frozen OpenRouter pricing snapshot is missing")
            self.backend = LiveBackend(
                None, openrouter_api_key=key, openrouter_pricing_snapshot=snapshot,
                timeout=self.prepared.timeout,
            )
        elif self.prepared.model.provider == "opencode":
            self.server = CleanOpenCodeServer(
                enable_length_tool=self.prepared.run_config.length_tools,
                required_models={self.prepared.model.model: {self.prepared.model.variant}},
                timeout=min(self.prepared.timeout, 45.0),
            )
            self.server.start()
            self.server_started = True
            self.backend = LiveBackend(self.server, timeout=self.prepared.timeout, max_tokens=self.prepared.max_tokens)
        else:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
            key = os.environ.get("DEEPSEEK_API_KEY")
            self.backend = LiveBackend(None, deepseek_api_key=key, timeout=self.prepared.timeout, max_tokens=self.prepared.max_tokens)
        return self.backend

    async def __call__(self, spec: ModelSpec, system: str, user: str, schema: dict[str, Any], allow_tool: bool) -> DetailedResponse:
        return await self._make_backend()(spec, system, user, schema, allow_tool)

    def close(self) -> None:
        server = self.server
        self.server = None
        self.server_started = False
        self.backend = None
        if server is not None:
            try:
                server.stop()
            except Exception:
                pass


async def execute_run(prepared: PreparedRun, *, stop_requested: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Execute exactly one runner trial, retaining its checkpoints and exports."""
    lazy = _LazyRequest(prepared)
    verifier = functools.partial(
        verify_original_timing,
        model_path=prepared.model_path,
        voice=prepared.voice,
        speech_rate=prepared.rate,
        text_mode="punctuation",
    )
    budget_policy: BudgetPolicy | None = None
    if prepared.primary:
        import copy as _copy
        openrouter = prepared.metadata.setdefault("openrouter", {})
        if prepared.resume:
            snapshot = openrouter.get("pricing_snapshot")
            if not isinstance(snapshot, dict):
                raise ResumeError("frozen OpenRouter pricing snapshot is missing")
            await openrouter_transport.validate_current_pricing(snapshot, timeout=30.0)
        else:
            snapshot = await openrouter_transport.fetch_pricing_snapshot(timeout=30.0)
            openrouter["pricing_snapshot"] = _copy.deepcopy(snapshot)
        budget = prepared.metadata.get("budget")
        if not isinstance(budget, dict) or not isinstance(budget.get("limit_usd"), str):
            raise ValueError("primary budget contract is missing")
        budget_policy = BudgetPolicy.from_snapshot(
            budget["limit_usd"], snapshot, max_requests=int(budget.get("max_requests", 1000)),
            max_unknown_calls=budget.get("max_unknown_calls", 0),
        )
    from utils.shortening_html import live_html_report
    with live_html_report(prepared.output_dir):
        try:
            return await run_trial(
                items=copy.deepcopy(prepared.items),
                plans=prepared.plans,
                target_ids=prepared.target_ids,
                planner_config=prepared.planner_config,
                run_config=prepared.run_config,
                generator=prepared.model,
                arbiter=prepared.arbiter or prepared.model,
                output_dir=prepared.output_dir,
                request_fn=lazy,
                original_timing_verifier=verifier,
                report_metadata=prepared.metadata,
                resume=prepared.resume,
                retry_unknown_ids=prepared.retry_unknown_ids,
                budget_policy=budget_policy,
                stop_requested=stop_requested,
            )
        finally:
            lazy.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shorten one complete subtitle file using frozen Vosk calibration")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--voice", type=int, default=DEFAULT_VOICE)
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE)
    parser.add_argument("--text-mode", choices=("punctuation",), default="punctuation")
    parser.add_argument("--variant", help=argparse.SUPPRESS)
    parser.add_argument("--length-tools", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-unknown", dest="retry_unknown_ids", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-tokens", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--budget-usd", type=str)
    parser.add_argument("--max-requests", type=int, default=1000)
    parser.add_argument("--run-time-limit", type=float, default=None,
                        help="primary only: stop before the next request after this many seconds")
    parser.add_argument("--gen-batch-size", type=int, default=6)
    parser.add_argument("--review-batch-size", type=int, default=12)
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--max-context-chars", type=int, default=20000)
    return parser


def main(argv: list[str] | None = None) -> int:
    prepared: PreparedRun | None = None
    try:
        args = _parser().parse_args(argv)
        prepared = prepare_run(args)
        if prepared.dry_run:
            prepared.output_dir.mkdir(parents=True)
            _write_json(prepared.output_dir / "preflight.json", _dry_report(prepared))
            print(json.dumps({
                "run_state": "dry_run", "output_dir": str(prepared.output_dir),
                "roles": _dry_report(prepared)["model"], "selected_count": len(prepared.target_ids),
                "known_spent_usd": None, "unknown_reserved_usd": None, "budget_stop_reason": None,
            }, ensure_ascii=False, sort_keys=True))
            return 0
        if prepared.primary:
            stop_flag = threading.Event()
            deadline = (time.monotonic() + prepared.run_time_limit_sec
                        if prepared.run_time_limit_sec is not None else None)
            active_task: asyncio.Task[dict[str, Any]] | None = None

            def on_signal(_signum: int, _frame: Any) -> None:
                if stop_flag.is_set():
                    if active_task is not None:
                        active_task.cancel()
                    else:
                        raise KeyboardInterrupt
                else:
                    stop_flag.set()

            def should_stop() -> bool:
                return stop_flag.is_set() or (deadline is not None and time.monotonic() >= deadline)

            async def run_primary() -> dict[str, Any]:
                nonlocal active_task
                active_task = asyncio.current_task()
                return await execute_run(prepared, stop_requested=should_stop)

            previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                for sig in previous:
                    signal.signal(sig, on_signal)
                report = asyncio.run(run_primary())
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
        else:
            report = asyncio.run(execute_run(prepared))
        summary = report.get("summary", {})
        outcome_summary = report.get("outcome_summary", {})
        budget = report.get("budget", {})
        print(json.dumps({
            "run_state": report.get("run_state"), "output_dir": str(prepared.output_dir),
            "roles": report.get("model"), "selected_count": summary.get("selected_count", len(prepared.target_ids)),
            "resolved_count": summary.get("resolved_count", 0), "manual_count": summary.get("manual_count", 0),
            "review_unresolved_count": outcome_summary.get("unresolved_count", 0),
            "intervention_count": outcome_summary.get("intervention_count", 0),
            "known_spent_usd": budget.get("known_spent_usd"),
            "unknown_reserved_usd": budget.get("unknown_reserved_usd"),
            "budget_stop_reason": budget.get("reason"),
        }, ensure_ascii=False, sort_keys=True))
        return 0 if report.get("run_state") == "completed" else 1
    except UnknownRequestsError as exc:
        ids = ", ".join(exc.request_ids)
        print(
            f"shorten_file blocked: unknown request IDs: {ids}. "
            "A provider may have billed an unfinished attempt; rerun with "
            "--resume --retry-unknown REQUEST_ID only after reviewing the risk.",
            file=sys.stderr,
        )
        return 1 if prepared is not None and prepared.primary else 2
    except asyncio.CancelledError:
        print("shorten_file interrupted during a request; unknown billing retained in the journal", file=sys.stderr)
        return 130
    except SRTParseError as exc:
        message = re.sub(r"(?i)(bearer\s+|sk[-_])[A-Za-z0-9_.-]+", "[redacted]", str(exc))
        print(f"shorten_file failed: {type(exc).__name__}: {message[:2000]}", file=sys.stderr)
        return 1
    except Exception as exc:
        lines = re.sub(r"(?i)(bearer\s+|sk[-_])[A-Za-z0-9_.-]+", "[redacted]", str(exc)).splitlines()
        message = lines[0] if lines else "no error message"
        print(f"shorten_file failed: {type(exc).__name__}: {message[:240]}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
