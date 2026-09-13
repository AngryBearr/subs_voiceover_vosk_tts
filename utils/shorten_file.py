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
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from utils.deepseek_limits import resolve_deepseek_token_limit
from utils.llm_types import DetailedResponse
from utils.shortening_audio import verify_original_timing
from utils.shortening_backends import DEEPSEEK_VARIANTS, OPENAI_MODELS, LiveBackend
from utils.shortening_plan import CuePlan, PlannerConfig, plan_subtitles
from utils.shortening_runner import ModelSpec, RunConfig, run_trial
from utils.shortening_resume import ResumeError, UnknownRequestsError
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
    _finite_number(getattr(args, "timeout", 300.0), "timeout", positive=True)
    for name in ("gen_batch_size", "review_batch_size", "max_context_chars"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    max_repairs = getattr(args, "max_repairs")
    if isinstance(max_repairs, bool) or not isinstance(max_repairs, int) or not 0 <= max_repairs <= 2:
        raise ValueError("max_repairs must be between 0 and 2")
    max_tokens = getattr(args, "max_tokens")
    if max_tokens is not None:
        resolve_deepseek_token_limit("off", max_tokens)
    if getattr(args, "text_mode") != "punctuation":
        raise ValueError("text_mode must be punctuation")
    model = getattr(args, "model", None)
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
    metadata["config"]["run"].update({
        "timeout": float(args.timeout),
        "max_tokens": args.max_tokens,
    })
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
        timeout=float(args.timeout),
        max_tokens=args.max_tokens,
        text_mode="punctuation",
        profile_path=profile_path,
        profile_file_sha256=profile_file_sha256,
        profile=profile,
        identity=identity,
        model=model,
        planner_config=planner_config,
        run_config=run_config,
        metadata=metadata,
        dry_run=bool(getattr(args, "dry_run", False)),
        resume=resume,
        retry_unknown_ids=retry_unknown_ids,
    )


def _dry_report(prepared: PreparedRun) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_state": "dry_run",
        "model": {"generator": _asdict_model(prepared.model), "arbiter": _asdict_model(prepared.model)},
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
        if self.prepared.model.provider == "opencode":
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


async def execute_run(prepared: PreparedRun) -> dict[str, Any]:
    """Execute exactly one runner trial, retaining its checkpoints and exports."""
    lazy = _LazyRequest(prepared)
    verifier = functools.partial(
        verify_original_timing,
        model_path=prepared.model_path,
        voice=prepared.voice,
        speech_rate=prepared.rate,
        text_mode="punctuation",
    )
    try:
        return await run_trial(
            items=copy.deepcopy(prepared.items),
            plans=prepared.plans,
            target_ids=prepared.target_ids,
            planner_config=prepared.planner_config,
            run_config=prepared.run_config,
            generator=prepared.model,
            arbiter=prepared.model,
            output_dir=prepared.output_dir,
            request_fn=lazy,
            original_timing_verifier=verifier,
            report_metadata=prepared.metadata,
            resume=prepared.resume,
            retry_unknown_ids=prepared.retry_unknown_ids,
        )
    finally:
        lazy.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shorten one complete subtitle file using frozen Vosk calibration")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--voice", type=int, default=DEFAULT_VOICE)
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE)
    parser.add_argument("--text-mode", choices=("punctuation",), default="punctuation")
    parser.add_argument("--variant")
    parser.add_argument("--length-tools", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-unknown", dest="retry_unknown_ids", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--gen-batch-size", type=int, default=6)
    parser.add_argument("--review-batch-size", type=int, default=12)
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--max-context-chars", type=int, default=20000)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        prepared = prepare_run(args)
        if prepared.dry_run:
            prepared.output_dir.mkdir(parents=True)
            _write_json(prepared.output_dir / "preflight.json", _dry_report(prepared))
            print(json.dumps({"run_state": "dry_run", "output_dir": str(prepared.output_dir), "selected_count": len(prepared.target_ids)}, ensure_ascii=False, sort_keys=True))
            return 0
        report = asyncio.run(execute_run(prepared))
        summary = report.get("summary", {})
        print(json.dumps({"run_state": report.get("run_state"), "output_dir": str(prepared.output_dir), "selected_count": summary.get("selected_count", len(prepared.target_ids)), "resolved_count": summary.get("resolved_count", 0), "manual_count": summary.get("manual_count", 0)}, ensure_ascii=False, sort_keys=True))
        return 0 if report.get("run_state") == "completed" else 1
    except UnknownRequestsError as exc:
        ids = ", ".join(exc.request_ids)
        print(
            f"shorten_file blocked: unknown request IDs: {ids}. "
            "A provider may have billed an unfinished attempt; rerun with "
            "--resume --retry-unknown REQUEST_ID only after reviewing the risk.",
            file=sys.stderr,
        )
        return 2
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
