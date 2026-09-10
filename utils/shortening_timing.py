"""Diagnostic character-budget profiles derived from measured trial audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import replace
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from pathlib import Path
from typing import Any

from utils.shortening_plan import CuePlan, PlannerConfig


PROFILE_SCHEMA_VERSION = 1
PROFILE_VERSION = 1
MAX_SPEECH_RATE = Decimal("1.5")


def _finite_number(value: Any, name: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{name} must be a finite number") from None
    if not result.is_finite() or (positive and result <= 0):
        raise ValueError(f"{name} must be finite and positive" if positive else f"{name} must be finite")
    return result


def _optional_duration(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() and result > 0 else None


def _source(report: dict[str, Any], key: str) -> dict[str, Any]:
    value = report.get(key)
    return value if isinstance(value, dict) else {}


def _record_value(record: dict[str, Any], key: str, nested: str | None = None) -> Any:
    value = record.get(key)
    if value is not None:
        return value
    if nested is not None:
        nested_value = record.get(nested)
        if isinstance(nested_value, dict):
            return nested_value.get(key)
    return None


def _cue_records(report: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{label}.records must be a list")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("cue_id"), str):
            raise ValueError(f"{label}.records must contain cue_id objects")
        cue_id = record["cue_id"]
        if cue_id in result:
            raise ValueError(f"duplicate cue_id: {cue_id}")
        result[cue_id] = record
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_hash(audio_report: dict[str, Any], trial_report: dict[str, Any]) -> str:
    source = _source(trial_report, "source_manifest")
    source_hash = source.get("sha256")
    if not isinstance(source_hash, str) or not source_hash:
        raise ValueError("trial_report.source_manifest.sha256 is required")
    declared_input_hash = audio_report.get("source_input_sha256")
    if not isinstance(declared_input_hash, str) or not declared_input_hash:
        raise ValueError("audio source_input_sha256 is required")
    if declared_input_hash != source_hash:
        raise ValueError("audio source_input_sha256 does not match trial source hash")
    return source_hash


def _audio_metadata(audio_report: dict[str, Any]) -> tuple[Decimal, str, str, Any]:
    rate = _finite_number(audio_report.get("speech_rate"), "speech_rate", positive=True)
    if rate > MAX_SPEECH_RATE:
        raise ValueError("speech_rate must be no greater than 1.5")
    model_path = audio_report.get("model_path")
    preprocessing = audio_report.get("preprocessing")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("model_path metadata is required")
    if not isinstance(preprocessing, str) or not preprocessing:
        raise ValueError("preprocessing metadata is required")
    voice = audio_report.get("voice")
    if isinstance(voice, bool) or not isinstance(voice, int) or voice < 0:
        raise ValueError("voice metadata must be a non-negative integer")
    return rate, model_path, preprocessing, voice


def _candidate_duration(
    audio_record: dict[str, Any],
    cue_id: str,
    candidate_key: str,
    direct_key: str,
) -> Decimal | None:
    candidate = audio_record.get(candidate_key)
    direct_raw = audio_record.get(direct_key)
    direct_duration = _optional_duration(direct_raw)
    if direct_raw is not None and direct_duration is None:
        raise ValueError(f"{cue_id}.{direct_key} must be a positive finite duration")
    if candidate is not None and not isinstance(candidate, dict):
        raise ValueError(f"{cue_id}.{candidate_key} must be an object")
    nested_raw = candidate.get("duration_sec") if isinstance(candidate, dict) else None
    nested_duration = (
        _optional_duration(nested_raw)
        if isinstance(candidate, dict)
        else None
    )
    if nested_raw is not None and nested_duration is None:
        raise ValueError(f"{cue_id}.{candidate_key}.duration_sec must be a positive finite duration")
    if direct_duration is not None and nested_duration is not None and direct_duration != nested_duration:
        raise ValueError(f"{cue_id}.{direct_key} does not match {cue_id}.{candidate_key}.duration_sec")
    if direct_duration is not None and candidate is None:
        raise ValueError(f"{cue_id}.{direct_key} has no matching {cue_id}.{candidate_key}")
    return nested_duration if nested_duration is not None else direct_duration


def build_timing_profile(
    audio_report: dict[str, Any],
    trial_report: dict[str, Any],
    *,
    margin: float = 0.95,
) -> dict[str, Any]:
    """Build an explicitly diagnostic, observation-based timing profile."""
    if not isinstance(audio_report, dict) or not isinstance(trial_report, dict):
        raise TypeError("audio_report and trial_report must be objects")
    margin_decimal = _finite_number(margin, "margin", positive=True)
    if margin_decimal > Decimal("1"):
        raise ValueError("margin must be no greater than 1")
    source_hash = _validate_source_hash(audio_report, trial_report)
    rate, model_path, preprocessing, voice = _audio_metadata(audio_report)
    trial_records = _cue_records(trial_report, "trial_report")
    audio_records = _cue_records(audio_report, "audio_report")

    overrides: dict[str, dict[str, Any]] = {}
    kept_fit_ids: list[str] = []
    unresolved: list[str] = []
    for cue_id, audio in audio_records.items():
        if cue_id not in trial_records:
            raise ValueError(f"audio cue_id is not present in trial report: {cue_id}")
        trial = trial_records[cue_id]
        budget = trial.get("budget")
        if not isinstance(budget, dict):
            raise ValueError(f"{cue_id}.budget is required")
        trial_available = _finite_number(budget.get("available_ms"), f"{cue_id}.budget.available_ms")
        if trial_available != trial_available.to_integral_value():
            raise ValueError(f"{cue_id}.budget.available_ms must be an integer")
        audio_available_raw = audio.get("available_ms")
        if audio_available_raw is not None:
            audio_available = _finite_number(audio_available_raw, f"{cue_id}.available_ms")
            if audio_available != audio_available.to_integral_value():
                raise ValueError(f"{cue_id}.available_ms must be an integer")
            if audio_available != trial_available:
                raise ValueError(f"{cue_id}.available_ms does not match trial budget")
        available = trial_available
        original = trial.get("original")
        if not isinstance(original, str):
            raise ValueError(f"{cue_id}.trial original text is required")
        audio_original_text = audio.get("original_text")
        if audio_original_text is not None and audio_original_text != original:
            raise ValueError(f"{cue_id}.audio original_text does not match trial original")
        final_text = trial.get("final_text")
        audio_final_text = audio.get("final_text")
        if audio_final_text is not None and audio_final_text != final_text:
            raise ValueError(f"{cue_id}.audio final_text does not match trial final_text")
        if audio_final_text is not None and not isinstance(audio_final_text, str):
            raise ValueError(f"{cue_id}.audio final_text must be text")
        if final_text is not None and not isinstance(final_text, str):
            raise ValueError(f"{cue_id}.trial final_text must be text")
        if (
            audio.get("error") is not None
            or audio.get("error_type") is not None
            or audio.get("status") == "error"
        ):
            unresolved.append(cue_id)
            continue
        final_duration = _candidate_duration(audio, cue_id, "final", "final_duration_sec")
        if final_duration is not None and not isinstance(final_text, str):
            raise ValueError(f"{cue_id}.final duration has no matching trial final_text")
        old_max = budget.get("max_chars")
        if isinstance(old_max, bool) or not isinstance(old_max, int) or old_max < 0:
            raise ValueError(f"{cue_id}.budget.max_chars must be a non-negative integer")
        if available <= 0 or final_duration is None:
            if cue_id not in unresolved:
                unresolved.append(cue_id)
            continue
        window = available / Decimal("1000")
        if final_duration <= window:
            kept_fit_ids.append(cue_id)
            continue

        original_len = len(original)
        original_duration = _candidate_duration(audio, cue_id, "original", "original_duration_sec")
        final_len = len(final_text) if isinstance(final_text, str) else None
        candidates: list[Decimal] = []
        if original_len is not None and original_duration is not None:
            candidates.append(Decimal(original_len) * window / original_duration)
        if final_len is not None:
            candidates.append(Decimal(final_len) * window / final_duration)
        candidates.append(Decimal(old_max))
        proposed_decimal = min(candidates) * margin_decimal
        proposed = int(proposed_decimal.to_integral_value(rounding=ROUND_FLOOR))
        proposed = max(0, min(old_max, proposed))
        if proposed >= old_max:
            continue
        overrides[cue_id] = {
            "max_chars": proposed,
            "previous_max_chars": old_max,
            "available_ms": int(available),
            "original_chars": original_len,
            "original_duration_sec": float(original_duration) if original_duration is not None else None,
            "final_chars": final_len,
            "final_duration_sec": float(final_duration),
            "reason": "measured_overflow",
        }

    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_version": PROFILE_VERSION,
        "source_input_sha256": source_hash,
        "expected_speech_rate": float(rate),
        "speech_rate": float(rate),
        "voice": voice,
        "preprocessing": preprocessing,
        "model_path": model_path,
        "overrides": overrides,
        "kept_fit_ids": kept_fit_ids,
        "unresolved_synthesis_ids": unresolved,
        "margin": float(margin_decimal),
        "diagnostic_only": True,
    }


def _profile_version(profile: dict[str, Any]) -> None:
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION or profile.get("profile_version") != PROFILE_VERSION:
        raise ValueError("unsupported timing profile schema/version")


def _status_for(plan: CuePlan, max_chars: int) -> tuple[str, str, float]:
    if plan.status == "invalid":
        return plan.status, plan.severity, plan.reduction_fraction
    length = len(plan.text)
    reduction = max(0.0, 1.0 - (max_chars / length)) if length else 0.0
    if length <= max_chars:
        status = "fits"
    elif len(plan.text.split()) <= 2:
        status = "short_timing"
    else:
        status = "needs_shortening"
    if status == "fits":
        severity = "fits"
    elif reduction <= 0.20:
        severity = "easy"
    elif reduction <= 0.40:
        severity = "medium"
    else:
        severity = "hard"
    return status, severity, reduction


def apply_timing_profile(
    plans: list[CuePlan],
    profile: dict[str, Any],
    *,
    source_sha256: str,
    config: PlannerConfig,
) -> list[CuePlan]:
    """Apply lower measured caps without mutating plans or source data."""
    if not isinstance(config, PlannerConfig):
        raise TypeError("config must be a PlannerConfig")
    if not isinstance(profile, dict):
        raise TypeError("profile must be an object")
    _profile_version(profile)
    if profile.get("source_input_sha256") != source_sha256:
        raise ValueError("timing profile source hash does not match input")
    rate = profile.get("expected_speech_rate", profile.get("speech_rate"))
    if _finite_number(rate, "expected_speech_rate") != _finite_number(config.max_speed, "max_speed"):
        raise ValueError("timing profile speech rate is incompatible with planner max_speed")
    overrides = profile.get("overrides")
    if not isinstance(overrides, dict):
        raise ValueError("timing profile overrides must be an object")
    plan_by_id: dict[str, CuePlan] = {}
    for plan in plans:
        if not isinstance(plan, CuePlan):
            raise TypeError("plans must contain CuePlan objects")
        if plan.cue_id in plan_by_id:
            raise ValueError(f"duplicate plan cue_id: {plan.cue_id}")
        plan_by_id[plan.cue_id] = plan
    result = list(plans)
    for cue_id, override in overrides.items():
        if not isinstance(cue_id, str) or cue_id not in plan_by_id:
            raise ValueError(f"unknown override cue_id: {cue_id}")
        if not isinstance(override, dict):
            raise ValueError(f"override for {cue_id} must be an object")
        plan = plan_by_id[cue_id]
        available = override.get("available_ms")
        if isinstance(available, bool) or not isinstance(available, int) or available < 0:
            raise ValueError(f"override {cue_id}.available_ms must be a non-negative integer")
        if available != plan.available_ms:
            raise ValueError(f"timing profile evidence window changed for {cue_id}")
        maximum = override.get("max_chars")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0 or maximum > plan.max_chars:
            raise ValueError(f"override {cue_id}.max_chars is invalid")
        previous = override.get("previous_max_chars")
        if previous is not None and previous != plan.max_chars:
            raise ValueError(f"override {cue_id}.previous_max_chars does not match plan")
        if plan.status == "invalid":
            continue
        status, severity, reduction = _status_for(plan, maximum)
        result[plan.position if 0 <= plan.position < len(result) and result[plan.position].cue_id == cue_id else result.index(plan)] = replace(
            plan, max_chars=maximum, reduction_fraction=reduction, severity=severity, status=status
        )
    return result


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"output file already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a diagnostic timing profile from measured audio")
    parser.add_argument("audio_report", type=Path)
    parser.add_argument("--trial-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"output file already exists: {args.output}")
    audio = json.loads(args.audio_report.read_text(encoding="utf-8"))
    trial = json.loads(args.trial_report.read_text(encoding="utf-8"))
    if not isinstance(audio, dict) or not isinstance(trial, dict):
        raise ValueError("reports must contain JSON objects")
    source = _source(audio, "source_report")
    source_path = source.get("absolute_path")
    if not isinstance(source_path, str) or not Path(source_path).is_absolute():
        raise ValueError("audio source_report.absolute_path is required")
    if Path(source_path).resolve() != args.trial_report.resolve():
        raise ValueError("audio source_report.absolute_path does not match trial report file")
    declared_hash = source.get("sha256")
    actual_hash = _sha256(args.trial_report)
    if declared_hash != actual_hash:
        raise ValueError("audio source_report hash does not match trial report file")
    profile = build_timing_profile(audio, trial, margin=args.margin)
    _write_new(args.output, profile)
    print(json.dumps({"overrides": len(profile["overrides"]), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
