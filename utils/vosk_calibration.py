"""Reproducible, source-disjoint Vosk duration calibration.

This module measures WAVs produced by the existing Vosk TTS wrapper.  It does
not make claims about pronunciation, intelligibility, or subtitle meaning.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import soundfile as sf
from scipy.optimize import nnls

from utils.vosk_text import normalize_vosk_text
from utils.text_metrics import count_words


NORMALIZER_NAME = "vosk_punctuation_v1_no_ruaccent"
SCHEMA_VERSION = 1
MODEL_NAMES = ("baseline", "char_linear", "text_features")
_IDENTITY_FIELDS = (
    "engine",
    "voice",
    "rate",
    "model_config_sha256",
    "model_fingerprint",
    "normalizer",
    "normalizer_sha256",
)
_PUNCTUATION = set(",;:.!?")
_LATIN_RE = re.compile(r"[A-Za-z]")
_TAG_RE = re.compile(r"<[^>]+>")
_SECRET_RE = re.compile(r"(?i)(bearer\s+|sk[-_])[A-Za-z0-9_.-]+")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _normalizer_source_hash() -> str:
    source_path = inspect.getsourcefile(normalize_vosk_text)
    if source_path is None:
        raise RuntimeError("could not locate normalizer source")
    return _sha256_file(Path(source_path))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_error(error: BaseException) -> str:
    message = _SECRET_RE.sub("[redacted]", str(error))
    message = re.sub(r"(?:[A-Za-z]:)?[/\\][^\s]+", "[path]", message)
    return message[:1000]


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def duration_features(text: str) -> dict[str, float]:
    """Return deterministic count features for normalized TTS text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = normalize_vosk_text(text)
    normalized_char_count = float(len(normalized))
    russian_vowel_count = float(sum(character.lower() in "аеёиоуыэюя" for character in normalized))
    return {
        "intercept": 1.0,
        "char_count": normalized_char_count,
        "normalized_char_count": normalized_char_count,
        "nonspace_chars": float(sum(not character.isspace() for character in normalized)),
        "word_count": float(count_words(normalized)),
        "vowel_count": russian_vowel_count,
        "russian_vowel_count": russian_vowel_count,
        "punctuation_count": float(sum(character in _PUNCTUATION for character in normalized)),
        "digit_count": float(sum(character.isdigit() for character in normalized)),
    }


def _report_flags(text: str) -> dict[str, bool]:
    normalized = normalize_vosk_text(text)
    return {
        "has_digits": any(character.isdigit() for character in normalized),
        "has_latin": bool(_LATIN_RE.search(normalized)),
        "has_tags": bool(_TAG_RE.search(normalized)),
    }


def _feature_vector(text: str, model_name: str) -> np.ndarray:
    features = duration_features(text)
    if model_name == "char_linear":
        return np.array([features["intercept"], features["char_count"]], dtype=float)
    if model_name == "text_features":
        return np.array(
            [
                features["intercept"],
                features["nonspace_chars"],
                features["vowel_count"],
                features["word_count"],
                features["punctuation_count"],
                features["digit_count"],
            ],
            dtype=float,
        )
    raise ValueError(f"unknown duration model: {model_name}")


def _baseline(text: str, rate: float) -> float:
    return len(text) / (13.0 * rate)


def _quantile_higher(values: Iterable[float], quantile: float) -> float:
    values_array = np.asarray(list(values), dtype=float)
    if values_array.size == 0:
        return 0.0
    try:
        return float(np.quantile(values_array, quantile, method="higher"))
    except TypeError:  # pragma: no cover - compatibility with old numpy
        return float(np.quantile(values_array, quantile, interpolation="higher"))


def _mae(actual: list[float], predicted: list[float]) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted)))) if actual else float("nan")


def _fit_model(rows: list[dict[str, Any]], model_name: str, rate: float) -> dict[str, Any]:
    if model_name == "baseline":
        return {"name": model_name, "rate": rate, "coefficients": [], "mae_fit": _mae([r["duration_sec"] for r in rows], [_baseline(r["display_text"], rate) for r in rows])}
    matrix = np.vstack([_feature_vector(r["tts_text"], model_name) for r in rows])
    target = np.asarray([float(row["duration_sec"]) for row in rows], dtype=float)
    coefficients, residual_norm = nnls(matrix, target)
    predictions = matrix @ coefficients
    return {
        "name": model_name,
        "rate": rate,
        "coefficients": [float(value) for value in coefficients],
        "nnls_residual_norm": float(residual_norm),
        "mae_fit": float(np.mean(np.abs(target - predictions))),
    }


def _model_prediction(model: dict[str, Any], text: str, rate: float) -> float:
    name = str(model["name"])
    if name == "baseline":
        return _baseline(text, rate)
    coefficients = np.asarray(model.get("coefficients", []), dtype=float)
    vector = _feature_vector(text, name)
    if coefficients.size != vector.size:
        raise ValueError(f"invalid coefficients for {name}")
    result = float(vector @ coefficients)
    return max(0.0, result)


def _fit_profile(rows: list[dict[str, Any]], rate: float) -> dict[str, Any]:
    successful_fit = [row for row in rows if row["split"] == "fit" and row["status"] == "ok"]
    successful_calibration = [row for row in rows if row["split"] == "calibration" and row["status"] == "ok"]
    fit_texts = {row["tts_text"] for row in successful_fit}
    calibration_texts = {row["tts_text"] for row in successful_calibration}
    provenance: dict[str, Any] = {
        "successful_fit_rows": len(successful_fit),
        "successful_calibration_rows": len(successful_calibration),
        "fit_source_count": len({row["source_id"] for row in successful_fit}),
        "calibration_source_count": len({row["source_id"] for row in successful_calibration}),
        "distinct_fit_texts": len(fit_texts),
        "distinct_calibration_texts": len(calibration_texts),
        "minimum_distinct_fit_texts": 10,
        "minimum_distinct_calibration_texts": 5,
    }
    if len(fit_texts) < 10 or len(calibration_texts) < 5:
        return {"rate": rate, "status": "profile_error", "error": "insufficient distinct successful fit/calibration texts", "training_provenance": provenance}

    models = {name: _fit_model(successful_fit, name, rate) for name in MODEL_NAMES}
    calibration_actual = [float(row["duration_sec"]) for row in successful_calibration]
    calibration_mae: dict[str, float] = {}
    for name in MODEL_NAMES:
        calibration_mae[name] = _mae(calibration_actual, [_model_prediction(models[name], row["tts_text" if name != "baseline" else "display_text"], rate) for row in successful_calibration])
        models[name]["mae_calibration"] = calibration_mae[name]
    chosen_name = min(MODEL_NAMES, key=lambda name: (calibration_mae[name], MODEL_NAMES.index(name)))
    chosen = models[chosen_name]
    residuals = [actual - _model_prediction(chosen, row["tts_text" if chosen_name != "baseline" else "display_text"], rate) for actual, row in zip(calibration_actual, successful_calibration)]
    guard = max(0.0, _quantile_higher(residuals, 0.90))
    char_residuals = [actual - _model_prediction(models["char_linear"], row["tts_text"], rate) for actual, row in zip(calibration_actual, successful_calibration)]
    char_guard = max(0.0, _quantile_higher(char_residuals, 0.90))
    return {
        "rate": rate,
        "status": "ok",
        "chosen_model": chosen_name,
        "models": models,
        "guard_sec": float(guard),
        "char_linear_guard_sec": float(char_guard),
        "empirical_target_coverage": 0.90,
        "mae_calibration": float(calibration_mae[chosen_name]),
        "training_provenance": provenance,
        "normalizer": NORMALIZER_NAME,
        "input_character_basis": "normalized_tts_text",
        "source_disjoint": True,
    }


def assert_profile_compatible(profile: dict[str, Any], expected_identity: dict[str, Any]) -> None:
    """Reject profiles whose frozen engine identity differs from the caller's configuration."""
    if not isinstance(profile, dict):
        raise ValueError("profile is not an object")
    if not isinstance(expected_identity, dict):
        raise TypeError("expected_identity must be an object")
    actual_identity = profile.get("identity")
    if not isinstance(actual_identity, dict):
        raise ValueError("profile has no identity")
    for field in _IDENTITY_FIELDS:
        if field not in actual_identity:
            raise ValueError(f"profile identity is missing {field}")
        if field not in expected_identity:
            raise ValueError(f"expected identity is missing {field}")
        actual = actual_identity[field]
        expected = expected_identity[field]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"profile identity mismatch for {field}")


def predict_duration(
    profile: dict[str, Any],
    text: str,
    *,
    expected_identity: dict[str, Any],
    guarded: bool = False,
) -> float:
    """Predict duration after checking identity; text is normalized by the profile normalizer."""
    assert_profile_compatible(profile, expected_identity)
    if not isinstance(profile, dict) or profile.get("status") != "ok":
        raise ValueError("profile is invalid or has profile_error status")
    rate = _finite_float(profile.get("rate"), "profile rate")
    model_name = str(profile.get("chosen_model"))
    models = profile.get("models")
    if not isinstance(models, dict) or model_name not in models:
        raise ValueError("profile has no chosen model")
    normalized = normalize_vosk_text(text)
    prediction = _model_prediction(models[model_name], text if model_name == "baseline" else normalized, rate)
    if guarded:
        prediction += max(0.0, _finite_float(profile.get("guard_sec", 0.0), "profile guard"))
    return max(0.0, float(prediction))


def max_chars_for_window(
    profile: dict[str, Any],
    available_ms: int,
    *,
    expected_identity: dict[str, Any],
) -> int | None:
    """Return a conservative normalized-character cap for a compatible profile."""
    assert_profile_compatible(profile, expected_identity)
    if not isinstance(profile, dict) or profile.get("status") != "ok":
        raise ValueError("profile is invalid or has profile_error status")
    if isinstance(available_ms, bool) or not isinstance(available_ms, int):
        raise ValueError("available_ms must be a non-negative integer")
    if available_ms < 0:
        raise ValueError("available_ms must be a non-negative integer")
    try:
        target = available_ms / 1000.0
        model = profile["models"]["char_linear"]
        coefficients = model["coefficients"]
        intercept, slope = float(coefficients[0]), float(coefficients[1])
        guard = max(0.0, float(profile.get("char_linear_guard_sec", profile.get("guard_sec", 0.0))))
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("profile has invalid char_linear coefficients") from exc
    if not all(math.isfinite(value) for value in (target, intercept, slope, guard)) or slope <= 0:
        return None
    return max(0, int(math.floor(max(0.0, (target - intercept - guard) / slope))))


def _validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be an object")
    cases = manifest.get("measurement_cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest.measurement_cases must be a non-empty list")
    allowed_splits = {"fit", "calibration", "test"}
    declared_sources: dict[str, tuple[str, str]] = {}
    sources_declared = "sources" in manifest
    if sources_declared:
        sources = manifest["sources"]
        if not isinstance(sources, list):
            raise ValueError("manifest.sources must be a list")
        for index, raw_source in enumerate(sources):
            if not isinstance(raw_source, dict):
                raise ValueError(f"manifest source {index} is not an object")
            source_id = raw_source.get("source_id")
            source_split = raw_source.get("split")
            source_sha = raw_source.get("sha256", raw_source.get("source_sha256"))
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(f"manifest source {index} lacks source_id")
            if source_split not in allowed_splits:
                raise ValueError(f"manifest source {index} has invalid split")
            if not isinstance(source_sha, str) or not source_sha:
                raise ValueError(f"manifest source {index} lacks sha256")
            if source_id in declared_sources:
                raise ValueError(f"manifest sources repeat source_id: {source_id}")
            declared_sources[source_id] = (source_split, source_sha)
    source_locations: dict[str, str] = {}
    source_hash_locations: dict[str, str] = {}
    text_locations: dict[str, str] = {}
    case_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(cases):
        if not isinstance(raw, dict):
            raise ValueError(f"measurement case {index} is not an object")
        split = raw.get("split")
        if split not in allowed_splits:
            raise ValueError(f"measurement case {index} has invalid split")
        source_id = str(raw.get("source_id", ""))
        source_sha = str(raw.get("source_sha256", ""))
        text = raw.get("text")
        if not source_id or not source_sha or not isinstance(text, str) or not text.strip():
            raise ValueError(f"measurement case {index} lacks source or text identity")
        case_id = str(raw.get("case_id", f"case-{index}"))
        if case_id in case_ids:
            raise ValueError(f"measurement cases repeat case_id: {case_id}")
        case_ids.add(case_id)
        if sources_declared:
            declaration = declared_sources.get(source_id)
            if declaration is None:
                raise ValueError(f"measurement case {index} references undeclared source: {source_id}")
            if declaration != (split, source_sha):
                raise ValueError(f"measurement case {index} source split/source_sha256 disagrees with declared source: {source_id}")
        normalized = normalize_vosk_text(text)
        text_hash = _sha256_bytes(normalized.encode("utf-8"))
        if not isinstance(raw.get("text_hash"), str) or raw["text_hash"] != text_hash:
            raise ValueError(f"measurement case {index} text_hash does not match normalized text")
        for value, locations, label in ((source_id, source_locations, "source_id"), (source_sha, source_hash_locations, "source_sha256"), (text_hash, text_locations, "normalized text")):
            old_split = locations.get(value)
            if old_split is not None and old_split != split:
                raise ValueError(f"{label} crosses source-disjoint splits: {value}")
            locations[value] = split
        validated.append({
            **raw,
            "case_id": case_id,
            "source_id": source_id,
            "source_sha256": source_sha,
            "split": split,
            "text": text,
            "tts_text": normalized,
            "text_hash": text_hash,
        })
    return validated


def _model_metadata(model_path: Path) -> dict[str, Any]:
    artifacts: list[dict[str, str]] = []
    if model_path.is_dir():
        for artifact in sorted(path for path in model_path.rglob("*") if path.is_file()):
            artifacts.append({"name": str(artifact.relative_to(model_path)), "sha256": _sha256_file(artifact)})
    artifact_identities = [{"name": item["name"], "sha256": item["sha256"]} for item in artifacts]
    return {
        "path": str(model_path),
        "artifacts": artifacts,
        "config_sha256": next((item["sha256"] for item in artifacts if item["name"] == "config.json"), None),
        "fingerprint": _json_hash(artifact_identities),
    }


def _profile_identity(model_metadata: dict[str, Any], voice: int, rate: float) -> dict[str, Any]:
    return {
        "engine": "vosk",
        "voice": int(voice),
        "rate": rate,
        "model_config_sha256": model_metadata.get("config_sha256"),
        "model_fingerprint": model_metadata["fingerprint"],
        "normalizer": NORMALIZER_NAME,
        "normalizer_sha256": _normalizer_source_hash(),
    }


def model_metadata(model_path: Path) -> dict[str, Any]:
    """Return the frozen metadata identity for a Vosk model directory."""
    return _model_metadata(model_path)


def profile_identity(model_metadata: dict[str, Any], voice: int, rate: float) -> dict[str, Any]:
    """Return the frozen profile identity for model, voice, and rate."""
    return _profile_identity(model_metadata, voice, rate)


def validate_character_profile(
    profile: dict[str, Any], *, expected_identity: dict[str, Any]
) -> None:
    """Strictly validate a frozen normalized character-linear calibration profile."""
    if not isinstance(profile, dict):
        raise ValueError("profile is not an object")
    if not isinstance(expected_identity, dict):
        raise TypeError("expected_identity must be an object")
    assert_profile_compatible(profile, expected_identity)
    identity = profile.get("identity")
    if type(identity) is not type(expected_identity) or identity != expected_identity:
        raise ValueError("profile identity mismatch")
    if profile.get("status") != "ok":
        raise ValueError("profile is invalid or has profile_error status")
    if profile.get("normalizer") != NORMALIZER_NAME:
        raise ValueError("profile has invalid normalizer")
    if profile.get("input_character_basis") != "normalized_tts_text":
        raise ValueError("profile has invalid input character basis")

    def finite_number(value: Any) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(float(value))
        except (OverflowError, ValueError):
            return False

    profile_rate = profile.get("rate")
    identity_rate = expected_identity.get("rate")
    if not finite_number(profile_rate) or profile_rate <= 0:
        raise ValueError("profile rate must be finite and positive")
    if type(profile_rate) is not type(identity_rate) or profile_rate != identity_rate:
        raise ValueError("profile rate does not match identity")
    models = profile.get("models")
    if not isinstance(models, dict):
        raise ValueError("profile models are invalid")
    char_linear = models.get("char_linear")
    if not isinstance(char_linear, dict) or char_linear.get("name") != "char_linear":
        raise ValueError("profile char_linear model is invalid")
    model_rate = char_linear.get("rate")
    if type(model_rate) is not type(profile_rate) or model_rate != profile_rate:
        raise ValueError("profile char_linear rate mismatch")
    coefficients = char_linear.get("coefficients")
    if not isinstance(coefficients, list) or len(coefficients) != 2:
        raise ValueError("profile char_linear coefficients are invalid")
    if any(not finite_number(value) for value in coefficients):
        raise ValueError("profile char_linear coefficients are invalid")
    if coefficients[0] < 0 or coefficients[1] <= 0:
        raise ValueError("profile char_linear coefficients are invalid")
    guard = profile.get("char_linear_guard_sec")
    if not finite_number(guard) or guard < 0:
        raise ValueError("profile char_linear guard is invalid")


def predict_character_duration(
    profile: dict[str, Any], text: str, *, expected_identity: dict[str, Any]
) -> float:
    """Predict duration with the validated unguarded normalized character model."""
    validate_character_profile(profile, expected_identity=expected_identity)
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    coefficients = profile["models"]["char_linear"]["coefficients"]
    normalized = normalize_vosk_text(text)
    return float(coefficients[0] + coefficients[1] * len(normalized))


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "scipy", "soundfile", "vosk-tts"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _invoke_synthesis(function: Callable[..., Any], text: str, path: Path, *, voice: int, rate: float, model_path: Path, synth_instance: Any) -> Any:
    supplied: dict[str, Any] = {"text": text, "voice": voice, "speech_rate": rate, "model_path": str(model_path), "output_path": str(path), "synth": synth_instance}
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**supplied)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return function(**supplied)
    return function(**{name: value for name, value in supplied.items() if name in signature.parameters})


def _audio_info(path: Path) -> tuple[int, int, float]:
    info = sf.info(str(path))
    frames = int(info.frames)
    samplerate = int(info.samplerate)
    if frames < 0 or samplerate <= 0:
        raise ValueError("invalid WAV frame metadata")
    return frames, samplerate, frames / samplerate


def _metric_bundle(actual: list[float], predicted: list[float], guard: float = 0.0) -> dict[str, Any]:
    if not actual:
        return {"sample_count": 0, "mae": None, "median_abs_error": None, "p90_abs_error": None, "mean_signed_error": None, "underprediction_count": 0, "underprediction_fraction_over_0_1_sec": None, "guarded_coverage": None, "guarded_overshoot_distribution": {}}
    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)
    errors = actual_array - predicted_array
    overshoot = np.maximum(0.0, predicted_array + guard - actual_array)
    return {
        "sample_count": len(actual),
        "mae": float(np.mean(np.abs(errors))),
        "median_abs_error": float(np.median(np.abs(errors))),
        "p90_abs_error": _quantile_higher(np.abs(errors), 0.90),
        "mean_signed_error": float(np.mean(errors)),
        "underprediction_count": int(np.sum(errors > 0)),
        "underprediction_fraction_over_0_1_sec": float(np.mean(errors > 0.1)),
        "guarded_coverage": float(np.mean(actual_array <= predicted_array + guard)),
        "guarded_overshoot_distribution": {"mean_sec": float(np.mean(overshoot)), "p90_sec": _quantile_higher(overshoot, 0.90), "max_sec": float(np.max(overshoot))},
    }


def _length_band(text: str) -> str:
    length = len(text)
    if length <= 15:
        return "<=15"
    if length <= 35:
        return "16..35"
    if length <= 60:
        return "36..60"
    if length <= 100:
        return "61..100"
    return ">100"


def _write_report_text(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Калибровка длительности Vosk", "", f"Статус: {report.get('status')}", ""]
    summary = report.get("summary", {})
    lines.append(f"Измерений: {summary.get('logical_observations', 0)}; уникальных WAV: {summary.get('unique_wav_count', 0)}.")
    lines.append(f"Ошибки: {summary.get('failure_fraction', 0.0):.4f} от всех попыток.")
    for rate, profile in sorted(report.get("profiles", {}).items(), key=lambda item: float(item[0])):
        lines.append("")
        lines.append(f"## Скорость {rate}")
        lines.append(f"Выбранная модель: {profile.get('chosen_model', 'нет')}.")
        lines.append(f"Профиль: {profile.get('status')}.")
        if profile.get("status") == "ok":
            lines.append(f"MAE калибровки: {profile.get('mae_calibration')}; guard: {profile.get('guard_sec')} сек.")
    lines.append("")
    lines.append("Показатели являются ограниченной проверкой отложенной выборки и не гарантируют произношение или разбор цифр.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def measure_corpus(
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    model_path: Path,
    voice: int = 8,
    rates: tuple[float, ...] = (1.0, 1.5),
    synthesis_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Measure every manifest case at every requested rate and fit profiles."""
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    if not rates:
        raise ValueError("rates must not be empty")
    if isinstance(voice, bool):
        raise ValueError("voice must not be bool")
    if any(isinstance(rate, bool) for rate in rates):
        raise ValueError("rates must not contain bool")
    normalized_rates = tuple(_finite_float(rate, "rate") for rate in rates)
    if any(rate <= 0 for rate in normalized_rates):
        raise ValueError("rates must be > 0")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    cases = _validate_manifest(manifest)
    output_dir.mkdir(parents=True)
    wav_dir = output_dir / "wav"
    profiles_dir = output_dir / "profiles"
    wav_dir.mkdir()
    profiles_dir.mkdir()
    (output_dir / "test_predictions.jsonl").touch()
    manifest_hash = str(manifest.get("_manifest_file_sha256") or _json_hash({key: value for key, value in manifest.items() if key != "_manifest_file_sha256"}))
    model_metadata = _model_metadata(model_path)
    partition_counts = {split: sum(case["split"] == split for case in cases) for split in ("fit", "calibration", "test")}
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "normalizer": NORMALIZER_NAME,
        "normalizer_version": NORMALIZER_NAME,
        "source_code_sha256": _sha256_file(Path(__file__)),
        "manifest_file_sha256": manifest_hash,
        "source_hashes": sorted({str(case["source_sha256"]) for case in cases}),
        "model": model_metadata,
        "vosk_config_sha256": model_metadata["config_sha256"],
        "voice": int(voice),
        "rates": list(normalized_rates),
        "package_versions": _package_versions(),
        "measurement_cases": len(cases),
        "expected_partition": partition_counts,
        "expected_partition_label": "observed",
    }
    _atomic_json(output_dir / "metadata.json", metadata)
    rows: list[dict[str, Any]] = []
    unique_paths: dict[tuple[str, int, float], dict[str, Any]] = {}
    attempted = 0
    synth_instance: Any = None
    measurements_path = output_dir / "measurements.jsonl"
    progress_path = output_dir / "progress.json"
    started = time.time()
    try:
        if synthesis_fn is None:
            if not model_path.exists() or not model_path.is_dir():
                raise FileNotFoundError(f"model path does not exist or is not a directory: {model_path}")
            from synthesize.synthesize import create_synth, synthesize_text

            synth_instance = create_synth(str(model_path), device="cpu")
            synthesis_fn = synthesize_text
    except Exception as error:
        failed = {**metadata, "status": "failed", "error": _safe_error(error), "summary": {"logical_observations": 0, "unique_wav_count": 0, "failure_fraction": 1.0}}
        _atomic_json(progress_path, failed)
        _atomic_json(profiles_dir / "report.json", failed)
        _write_report_text(profiles_dir / "results.md", failed)
        return failed

    with measurements_path.open("w", encoding="utf-8") as measurements:
        for case in cases:
            for rate in normalized_rates:
                attempted += 1
                key = (str(model_path.resolve()), int(voice), rate, case["text_hash"])
                row: dict[str, Any] = {
                    "case_id": case["case_id"], "source_id": case["source_id"], "split": case["split"], "position": case.get("position"),
                    "source_sha256": case["source_sha256"], "available_ms": case.get("available_ms"),
                    "text": case["text"], "display_text": case["text"], "tts_text": case["tts_text"], "voice": int(voice), "rate": rate,
                    "text_hash": case["text_hash"], "features": duration_features(case["tts_text"]), "flags": _report_flags(case["tts_text"]),
                    "status": "error", "error": None, "cause": None,
                }
                wall_start = time.perf_counter()
                cpu_start = time.process_time()
                try:
                    if key in unique_paths:
                        cached = unique_paths[key]
                        if not Path(cached["absolute_path"]).exists():
                            raise FileNotFoundError("cached WAV is missing")
                        row.update(cached)
                        row["reused_file"] = True
                        row["elapsed_wall_sec"] = 0.0
                        row["elapsed_cpu_sec"] = 0.0
                    else:
                        file_name = f"{case['text_hash']}_{int(voice)}_{rate:g}.wav"
                        wav_path = wav_dir / file_name
                        result = _invoke_synthesis(synthesis_fn, case["tts_text"], wav_path, voice=int(voice), rate=rate, model_path=model_path, synth_instance=synth_instance)
                        returned = Path(result) if isinstance(result, (str, Path)) else wav_path
                        if not wav_path.exists() and returned != wav_path and returned.exists():
                            raise FileNotFoundError("synthesizer did not write requested WAV path")
                        if not wav_path.exists():
                            raise FileNotFoundError("generated WAV is missing")
                        frames, samplerate, duration = _audio_info(wav_path)
                        relative_wav_path = str(wav_path.relative_to(output_dir))
                        cached = {"wavpath": relative_wav_path, "wav_path_relative": relative_wav_path, "absolute_path": str(wav_path), "frames": frames, "samplerate": samplerate, "duration_sec": duration, "reused_file": False}
                        unique_paths[key] = cached
                        row.update(cached)
                        row["elapsed_wall_sec"] = max(0.0, time.perf_counter() - wall_start)
                        row["elapsed_cpu_sec"] = max(0.0, time.process_time() - cpu_start)
                    row["status"] = "ok"
                    row["error"] = None
                    row["cause"] = None
                except Exception as error:
                    row["status"] = "error"
                    row["error"] = _safe_error(error)
                    row["cause"] = "synthesis_or_wav_measurement"
                    row["elapsed_wall_sec"] = max(0.0, time.perf_counter() - wall_start)
                    row["elapsed_cpu_sec"] = max(0.0, time.process_time() - cpu_start)
                row.pop("absolute_path", None)
                rows.append(row)
                measurements.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                measurements.flush()
                _atomic_json(progress_path, {**metadata, "status": "running", "attempted": attempted, "logical_observations": len(rows), "unique_wav_count": len(unique_paths), "elapsed_wall_sec": time.time() - started})

    profiles: dict[str, Any] = {}
    evaluation: dict[str, Any] = {}
    for rate in normalized_rates:
        rate_rows = [row for row in rows if row["rate"] == rate]
        profile = _fit_profile(rate_rows, rate)
        profile["voice"] = int(voice)
        profile["identity"] = _profile_identity(model_metadata, int(voice), rate)
        profiles[str(rate)] = profile
        _atomic_json(profiles_dir / f"profile_voice_{int(voice)}_rate_{rate:g}.json", profile)
        test_rows = [row for row in rate_rows if row["split"] == "test" and row["status"] == "ok"]
        fixed_models: dict[str, Any] = {}
        if profile.get("status") == "ok":
            for model_name in MODEL_NAMES:
                actual = [float(row["duration_sec"]) for row in test_rows]
                predicted = [_model_prediction(profile["models"][model_name], row["tts_text" if model_name != "baseline" else "display_text"], rate) for row in test_rows]
                fixed_models[model_name] = _metric_bundle(actual, predicted)
            chosen_name = str(profile["chosen_model"])
            chosen_model = profile["models"][chosen_name]
            actual = [float(row["duration_sec"]) for row in test_rows]
            predicted = [_model_prediction(chosen_model, row["tts_text" if chosen_name != "baseline" else "display_text"], rate) for row in test_rows]
            fixed_models["chosen_guarded"] = _metric_bundle(actual, predicted, float(profile["guard_sec"]))
            breakdowns: dict[str, dict[str, dict[str, Any]]] = {"length_bands": {}, "digits_flag": {}}
            for label, groups in (
                ("length_bands", {band: [row for row in test_rows if _length_band(row["display_text"]) == band] for band in ("<=15", "16..35", "36..60", "61..100", ">100")}),
                ("digits_flag", {"digits": [row for row in test_rows if row["flags"]["has_digits"]], "no_digits": [row for row in test_rows if not row["flags"]["has_digits"]]}),
            ):
                for group_name, group_rows in groups.items():
                    group_metrics: dict[str, Any] = {}
                    for model_name in MODEL_NAMES:
                        group_actual = [float(row["duration_sec"]) for row in group_rows]
                        group_predicted = [_model_prediction(profile["models"][model_name], row["tts_text" if model_name != "baseline" else "display_text"], rate) for row in group_rows]
                        group_metrics[model_name] = _metric_bundle(group_actual, group_predicted)
                    group_metrics["chosen_guarded"] = _metric_bundle(
                        [float(row["duration_sec"]) for row in group_rows],
                        [_model_prediction(chosen_model, row["tts_text" if chosen_name != "baseline" else "display_text"], rate) for row in group_rows],
                        float(profile["guard_sec"]),
                    )
                    breakdowns[label][group_name] = group_metrics
            evaluation[str(rate)] = {"selected_model": chosen_name, "test_metrics": fixed_models, "by_length_band": breakdowns["length_bands"], "by_digits_flag": breakdowns["digits_flag"], "sample_count": len(test_rows), "source_count": len({row["source_id"] for row in test_rows})}
            with (output_dir / "test_predictions.jsonl").open("a", encoding="utf-8") as predictions_file:
                for row, prediction in zip(test_rows, predicted):
                    prediction_row = {
                        "case_id": row["case_id"], "source_id": row["source_id"], "split": row["split"], "rate": rate,
                        "actual_duration_sec": row["duration_sec"], "selected_model": chosen_name,
                        "predicted_duration_sec": prediction, "guarded_predicted_duration_sec": prediction + float(profile["guard_sec"]),
                        "baseline_prediction_sec": _baseline(row["display_text"], rate),
                        "char_linear_prediction_sec": _model_prediction(profile["models"]["char_linear"], row["tts_text"], rate),
                        "text_features_prediction_sec": _model_prediction(profile["models"]["text_features"], row["tts_text"], rate),
                    }
                    predictions_file.write(json.dumps(prediction_row, ensure_ascii=False, sort_keys=True) + "\n")
        else:
            evaluation[str(rate)] = {"selected_model": None, "test_metrics": {}, "sample_count": len(test_rows), "source_count": len({row["source_id"] for row in test_rows})}

    summary: dict[str, Any] = {
        "logical_observations": len(rows), "unique_wav_count": len(unique_paths), "reused_file_observations": sum(bool(row.get("reused_file")) for row in rows),
        "attempted": attempted, "successes": sum(row["status"] == "ok" for row in rows), "failures": sum(row["status"] != "ok" for row in rows),
        "failure_fraction": (sum(row["status"] != "ok" for row in rows) / attempted) if attempted else 0.0,
        "failure_fraction_by_split": {split: (sum(row["split"] == split and row["status"] != "ok" for row in rows) / max(1, sum(row["split"] == split for row in rows))) for split in ("fit", "calibration", "test")},
        "partition_counts": {split: sum(row["split"] == split for row in rows) // len(normalized_rates) for split in ("fit", "calibration", "test")},
    }
    paired: dict[str, Any] = {}
    if 1.0 in normalized_rates and 1.5 in normalized_rates:
        first = {(row["case_id"]): row for row in rows if row["rate"] == 1.0 and row["status"] == "ok"}
        second = {(row["case_id"]): row for row in rows if row["rate"] == 1.5 and row["status"] == "ok"}
        ratios = [second[case_id]["duration_sec"] / first[case_id]["duration_sec"] for case_id in sorted(first.keys() & second.keys()) if first[case_id]["duration_sec"] > 0]
        paired = {"case_count": len(ratios), "duration_ratio_1_5_to_1_0": {"mean": float(np.mean(ratios)) if ratios else None, "median": float(np.median(ratios)) if ratios else None, "p10": _quantile_higher(ratios, 0.10) if ratios else None, "p90": _quantile_higher(ratios, 0.90) if ratios else None}}
    report: dict[str, Any] = {**metadata, "status": "complete", "summary": summary, "profiles": profiles, "evaluation": evaluation, "paired_rate_summary": paired, "recommended_for_use": {rate: bool(profile.get("status") == "ok" and evaluation.get(rate, {}).get("test_metrics", {}).get("chosen_guarded", {}).get("mae") is not None and evaluation[rate]["test_metrics"]["chosen_guarded"].get("guarded_coverage", 0.0) >= 0.85 and evaluation[rate]["test_metrics"].get(str(profile.get("chosen_model")), {}).get("mae", float("inf")) < evaluation[rate]["test_metrics"].get("baseline", {}).get("mae", float("inf"))) for rate, profile in profiles.items()}, "note": "Source-disjoint holdout относится только к этому манифесту и не обобщается на русский язык или все голоса; это не гарантия и не проверка качества произношения."}
    _atomic_json(profiles_dir / "report.json", report)
    _write_report_text(profiles_dir / "results.md", report)
    _atomic_json(progress_path, {**metadata, "status": "complete", "summary": summary})
    return report


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure actual Vosk WAV durations and fit source-disjoint profiles")
    parser.add_argument("corpus_manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", type=Path, default=Path("models/vosk-model-tts-ru-0.10-multi"))
    parser.add_argument("--voice", type=int, default=8)
    parser.add_argument("--rates", nargs="+", type=float, default=[1.0, 1.5])
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    if not args.model_path.exists() or not args.model_path.is_dir():
        parser.error(f"model path does not exist or is not a directory: {args.model_path}")
    try:
        with args.corpus_manifest.open("rb") as handle:
            manifest_bytes = handle.read()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest JSON must be an object")
        manifest["_manifest_file_sha256"] = _sha256_bytes(manifest_bytes)
        report = measure_corpus(manifest, args.output_dir, model_path=args.model_path, voice=args.voice, rates=tuple(args.rates))
    except Exception as error:
        parser.error(_safe_error(error))
    print(json.dumps({"status": report.get("status"), "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0 if report.get("status") == "complete" else 1


if __name__ == "__main__":
    sys.exit(_main())
