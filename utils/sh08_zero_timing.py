"""Measure the frozen SH08 zero-capacity originals with local Vosk TTS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from utils.shortening_audio import verify_original_timing
from utils.shortening_plan import CuePlan
from utils.vosk_calibration import (
    model_metadata,
    profile_identity,
    validate_character_profile,
)


VOICE = 8
SPEECH_RATE = 1.5
TEXT_MODE = "punctuation"
EXPECTED_ZERO_CASES = 19
FROZEN_MODEL = Path("models/vosk-model-tts-ru-0.10-multi")
FROZEN_PROFILE = Path(
    "output/vosk_calibration_20260910/profiles/profile_voice_8_rate_1.5.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _zero_cases(selection: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    raw_cases = selection.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("selection.cases must be a list")
    zero_cases = [
        row for row in raw_cases
        if isinstance(row, Mapping) and row.get("max_chars") == 0
    ]
    invalid_excluded = sum(row.get("status") == "invalid" for row in zero_cases)
    selected = [row for row in zero_cases if row.get("status") != "invalid"]
    if len(zero_cases) != EXPECTED_ZERO_CASES:
        raise ValueError(
            f"expected {EXPECTED_ZERO_CASES} zero-capacity cases, got {len(zero_cases)}"
        )
    if invalid_excluded != 1 or len(selected) != EXPECTED_ZERO_CASES - 1:
        raise ValueError(
            "expected exactly one invalid zero-capacity case and 18 valid cases"
        )
    return [dict(row) for row in selected], len(zero_cases), invalid_excluded


def _validate_case(row: Mapping[str, Any], index: int) -> None:
    label = f"zero-capacity case {index}"
    case_id = row.get("case_id")
    if not isinstance(case_id, str) or not case_id or ":" not in case_id:
        raise ValueError(f"{label} has malformed case_id")
    position = row.get("position")
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise ValueError(f"{label} has malformed position")
    original = row.get("original")
    if not isinstance(original, str) or not original.strip():
        raise ValueError(f"{label} has malformed original text")
    for name in ("start", "end", "available_ms"):
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} has malformed {name}")
        try:
            finite = math.isfinite(float(value))
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{label} has malformed {name}")
    if row["start"] < 0 or row["end"] <= row["start"] or row["available_ms"] <= 0:
        raise ValueError(f"{label} has a nonpositive timing window")
    if type(row.get("max_chars")) is not int or row["max_chars"] != 0:
        raise ValueError(f"{label} does not have zero max_chars")
    if not isinstance(row.get("status"), str) or row["status"] == "invalid":
        raise ValueError(f"{label} has malformed status")
    if row.get("length_basis") not in {"display_text", "normalized_tts_text"}:
        raise ValueError(f"{label} has malformed length_basis")
    risk = row.get("risk")
    if not isinstance(risk, Mapping):
        raise ValueError(f"{label} has malformed risk")
    if not isinstance(risk.get("severity"), str) or not isinstance(risk.get("reason"), str):
        raise ValueError(f"{label} has malformed risk severity/reason")
    if not isinstance(risk.get("short_timing"), bool):
        raise ValueError(f"{label} has malformed risk short_timing")


def build_plans(cases: list[dict[str, Any]]) -> list[CuePlan]:
    """Map frozen rows directly to plans without recalculating their budgets."""
    seen: set[str] = set()
    plans: list[CuePlan] = []
    for index, row in enumerate(cases):
        _validate_case(row, index)
        case_id = row["case_id"]
        if case_id in seen:
            raise ValueError(f"duplicate case ID: {case_id}")
        seen.add(case_id)
        risk = row["risk"]
        plans.append(
            CuePlan(
                cue_id=case_id,
                position=row["position"],
                text=row["original"],
                start=row["start"],
                end=row["end"],
                available_ms=row["available_ms"],
                max_chars=0,
                required_ratio=None,
                reduction_fraction=0.0,
                severity=risk["severity"],
                status=row["status"],
                short_timing_risk=risk["short_timing"],
                reason=risk["reason"],
                length_basis=row["length_basis"],
            )
        )
    return plans


def _assert_declared_compatibility(
    declared: Mapping[str, Any],
    *,
    profile_path: Path,
    profile_hash: str,
    model_path: Path,
    identity: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    declared_identity = declared.get("identity")
    if declared_identity is not None and declared_identity != identity:
        raise ValueError("declared profile identity is incompatible")
    declared_path = declared.get("path")
    if declared_path is not None and Path(str(declared_path)).resolve() != profile_path.resolve():
        raise ValueError("declared profile path is incompatible")
    declared_hash = declared.get("sha256")
    if declared_hash is not None and declared_hash != profile_hash:
        raise ValueError("declared profile hash is incompatible")
    model = declared.get("model")
    if isinstance(model, Mapping):
        declared_model_path = model.get("path")
        if declared_model_path is not None and Path(str(declared_model_path)).resolve() != model_path.resolve():
            raise ValueError("declared model path is incompatible")
        declared_metadata = model.get("metadata")
        if declared_metadata is not None and declared_metadata != metadata.get("artifacts"):
            raise ValueError("declared model metadata is incompatible")


def _validate_calibration(
    profile_path: Path,
    model_path: Path,
    profile: dict[str, Any],
    selection_path: Path,
    selection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not profile_path.is_file():
        raise FileNotFoundError(f"profile does not exist: {profile_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"model path does not exist or is not a directory: {model_path}")
    metadata = model_metadata(model_path)
    identity = profile_identity(metadata, VOICE, SPEECH_RATE)
    validate_character_profile(profile, expected_identity=identity)
    profile_hash = _sha256(profile_path)

    declarations: list[Mapping[str, Any]] = []
    for key in ("profile", "profile_metadata", "calibration"):
        value = selection.get(key)
        if isinstance(value, Mapping):
            declarations.append(value)
    manifest_path = selection_path.with_name("manifest.json")
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        value = manifest.get("profile")
        if isinstance(value, Mapping):
            declarations.append(value)
    for declaration in declarations:
        _assert_declared_compatibility(
            declaration,
            profile_path=profile_path,
            profile_hash=profile_hash,
            model_path=model_path,
            identity=identity,
            metadata=metadata,
        )
    return metadata, identity, profile_hash


def _classification(evidence: Mapping[str, Any]) -> str:
    if (
        evidence.get("error") is not None
        or evidence.get("error_type") is not None
        or evidence.get("status") == "error"
    ):
        return "unverified"
    audio = evidence.get("audio")
    if isinstance(audio, Mapping) and audio.get("status") == "error":
        return "unverified"
    fits = audio.get("original_fits") if isinstance(audio, Mapping) else None
    if type(fits) is bool:
        return "fits" if fits else "overrun"
    duration = evidence.get("duration_sec")
    window = evidence.get("available_ms")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and isinstance(window, (int, float)) and not isinstance(window, bool):
        if math.isfinite(float(duration)) and math.isfinite(float(window)):
            return "fits" if float(duration) <= float(window) / 1000 else "overrun"
    return "unverified"


def _evidence_rows(plans: list[CuePlan], measured: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plan in plans:
        evidence = measured.get(plan.cue_id)
        evidence = evidence if isinstance(evidence, Mapping) else {}
        audio = evidence.get("audio")
        original_audio = audio.get("original") if isinstance(audio, Mapping) else None
        row: dict[str, Any] = {
            "case_id": plan.cue_id,
            "position": plan.position,
            "original": plan.text,
            "start": plan.start,
            "end": plan.end,
            "available_ms": plan.available_ms,
            "duration_sec": evidence.get("duration_sec"),
            "audio_path": original_audio.get("path") if isinstance(original_audio, Mapping) else None,
            "status": _classification(evidence),
            "verifier_status": evidence.get("status"),
        }
        for key in ("error", "error_type"):
            if key in evidence:
                row[key] = evidence[key]
        rows.append(row)
    return rows


def run(
    selection_path: Path,
    profile_path: Path,
    model_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the local original-only timing check and save its compact summary."""
    selection = _load_json(selection_path)
    cases, zero_count, invalid_excluded = _zero_cases(selection)
    plans = build_plans(cases)
    profile = _load_json(profile_path)
    metadata, identity, profile_hash = _validate_calibration(
        profile_path, model_path, profile, selection_path, selection
    )
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    measured = verify_original_timing(
        plans,
        output_dir,
        model_path=model_path,
        voice=VOICE,
        speech_rate=SPEECH_RATE,
        text_mode=TEXT_MODE,
    )
    evidence = _evidence_rows(plans, measured)
    counts = {name: sum(row["status"] == name for row in evidence) for name in ("fits", "overrun", "unverified")}
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "source": {"path": str(selection_path.resolve()), "sha256": _sha256(selection_path)},
        "profile": {"path": str(profile_path.resolve()), "sha256": profile_hash, "identity": identity},
        "model": {
            "path": str(model_path.resolve()),
            "config_sha256": metadata.get("config_sha256"),
            "fingerprint": metadata.get("fingerprint"),
        },
        "config": {"voice": VOICE, "speech_rate": SPEECH_RATE, "text_mode": TEXT_MODE},
        "selection": {
            "zero_capacity_total": zero_count,
            "invalid_excluded": invalid_excluded,
            "valid_measurements": len(plans),
        },
        "counts": counts,
        "evidence": evidence,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run(args.selection, args.profile, args.model_path, args.output_dir)
    print(json.dumps(summary["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
