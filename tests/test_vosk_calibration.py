"""Unit tests for the deterministic Vosk calibration bookkeeping and profiles."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from utils.vosk_calibration import (
    assert_profile_compatible,
    duration_features,
    max_chars_for_window,
    measure_corpus,
    predict_duration,
)
from utils.vosk_text import normalize_vosk_text


def _manifest() -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for split, count, prefix in (("fit", 12, "fit"), ("calibration", 6, "cal"), ("test", 6, "test")):
        for index in range(count):
            text = f"{prefix} строка {index} пример"
            cases.append(
                {
                    "case_id": f"{split}-{index}",
                    "source_id": f"source-{split}",
                    "cue_id": f"cue-{index}",
                    "split": split,
                    "text": text,
                    "position": index,
                    "available_ms": 5000,
                    "source_sha256": f"{split}-hash",
                    "text_hash": hashlib.sha256(normalize_vosk_text(text).encode("utf-8")).hexdigest(),
                }
            )
    return {
        "schema_version": 1,
        "sources": [
            {"source_id": f"source-{split}", "split": split, "sha256": f"{split}-hash"}
            for split in ("fit", "calibration", "test")
        ],
        "measurement_cases": cases,
    }


def _fake_synth(text: str, output_path: str, speech_rate: float, **_: object) -> Path:
    duration = 0.2 + 0.03 * len(normalize_vosk_text(text))
    samplerate = 1000
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(max(1, round(duration * samplerate))), samplerate, subtype="PCM_16")
    return path


def test_duration_features_are_counts_on_normalized_text() -> None:
    features = duration_features("Ёжик, 12!")
    normalized = normalize_vosk_text("Ёжик, 12!")
    assert features["char_count"] == len(normalized)
    assert features["nonspace_chars"] == sum(not char.isspace() for char in normalized)
    assert features["vowel_count"] == 2
    assert features["digit_count"] == 2
    assert features["punctuation_count"] == 2


def test_measurement_fits_profiles_caches_per_rate_and_writes_progress(tmp_path: Path) -> None:
    output = tmp_path / "measurement"
    report = measure_corpus(_manifest(), output, model_path=tmp_path / "missing-model", rates=(1.0, 1.5), synthesis_fn=_fake_synth)
    assert report["status"] == "complete"
    assert report["summary"]["logical_observations"] == 48
    assert report["summary"]["unique_wav_count"] == 48
    assert report["summary"]["reused_file_observations"] == 0
    assert (output / "measurements.jsonl").exists()
    assert (output / "progress.json").exists()
    assert (output / "profiles" / "report.json").exists()
    assert report["profiles"]["1.0"]["status"] == "ok"
    assert report["expected_partition"] == {"fit": 12, "calibration": 6, "test": 6}
    assert report["expected_partition_label"] == "observed"
    assert report["profiles"]["1.0"]["identity"]["engine"] == "vosk"
    char_coefficients = report["profiles"]["1.0"]["models"]["char_linear"]["coefficients"]
    assert char_coefficients[1] == pytest.approx(0.03, abs=0.002)


def test_duplicate_text_reuses_only_same_model_voice_rate(tmp_path: Path) -> None:
    manifest = _manifest()
    cases = manifest["measurement_cases"]
    assert isinstance(cases, list)
    cases[1]["text"] = cases[0]["text"]
    cases[1]["source_id"] = cases[0]["source_id"]
    cases[1]["source_sha256"] = cases[0]["source_sha256"]
    cases[1]["text_hash"] = cases[0]["text_hash"]
    report = measure_corpus(manifest, tmp_path / "out", model_path=tmp_path / "missing", rates=(1.0, 1.5), synthesis_fn=_fake_synth)
    assert report["summary"]["logical_observations"] == 48
    assert report["summary"]["unique_wav_count"] == 46
    assert report["summary"]["reused_file_observations"] == 2


def test_source_and_normalized_text_cross_split_leakage_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest()
    cases = manifest["measurement_cases"]
    assert isinstance(cases, list)
    cases[-1]["source_sha256"] = cases[0]["source_sha256"]
    with pytest.raises(ValueError, match="source_sha256"):
        measure_corpus(manifest, tmp_path / "out", model_path=tmp_path / "missing", synthesis_fn=_fake_synth)

    manifest = _manifest()
    cases = manifest["measurement_cases"]
    assert isinstance(cases, list)
    cases[-1]["text"] = cases[0]["text"]
    cases[-1]["text_hash"] = cases[0]["text_hash"]
    with pytest.raises(ValueError, match="normalized text"):
        measure_corpus(manifest, tmp_path / "other", model_path=tmp_path / "missing", synthesis_fn=_fake_synth)


def test_duplicate_case_id_is_rejected_before_synthesis(tmp_path: Path) -> None:
    manifest = _manifest()
    cases = manifest["measurement_cases"]
    assert isinstance(cases, list)
    cases[1]["case_id"] = cases[0]["case_id"]
    with pytest.raises(ValueError, match="case_id"):
        measure_corpus(manifest, tmp_path / "out", model_path=tmp_path / "missing", synthesis_fn=_fake_synth)


def test_missing_text_hash_is_rejected_before_synthesis(tmp_path: Path) -> None:
    manifest = _manifest()
    cases = manifest["measurement_cases"]
    assert isinstance(cases, list)
    del cases[0]["text_hash"]
    with pytest.raises(ValueError, match="text_hash"):
        measure_corpus(manifest, tmp_path / "out", model_path=tmp_path / "missing", synthesis_fn=_fake_synth)


def test_declared_source_relabel_is_rejected_before_model_initialization(tmp_path: Path) -> None:
    manifest = _manifest()
    cases = manifest["measurement_cases"]
    assert isinstance(cases, list)
    cases[0]["split"] = "test"
    with pytest.raises(ValueError, match="declared source"):
        measure_corpus(manifest, tmp_path / "out", model_path=tmp_path / "missing", synthesis_fn=_fake_synth)


def test_invalid_rate_does_not_overwrite_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="rates"):
        measure_corpus(_manifest(), output, model_path=tmp_path / "missing", rates=(True,), synthesis_fn=_fake_synth)
    with pytest.raises(ValueError, match="voice"):
        measure_corpus(_manifest(), output, model_path=tmp_path / "missing", voice=True, synthesis_fn=_fake_synth)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_test_duration_changes_do_not_change_frozen_profile(tmp_path: Path) -> None:
    manifest = _manifest()
    first = measure_corpus(manifest, tmp_path / "first", model_path=tmp_path / "missing", synthesis_fn=_fake_synth)

    def altered_test_synth(text: str, output_path: str, speech_rate: float, **_: object) -> Path:
        duration = (0.2 + 0.03 * len(normalize_vosk_text(text))) * (5.0 if text.startswith("test ") else 1.0)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.zeros(max(1, round(duration * 1000))), 1000, subtype="PCM_16")
        # Deliberately perturb only test labels after the WAV is generated.
        return path

    changed = _manifest()
    changed_report = measure_corpus(changed, tmp_path / "changed", model_path=tmp_path / "missing", synthesis_fn=altered_test_synth)
    for rate in ("1.0", "1.5"):
        assert first["profiles"][rate] == changed_report["profiles"][rate]


def test_failed_wav_is_counted_and_no_infinite_budget(tmp_path: Path) -> None:
    def missing_wav(**_: object) -> None:
        return None

    report = measure_corpus(_manifest(), tmp_path / "out", model_path=tmp_path / "missing", rates=(1.0,), synthesis_fn=missing_wav)
    assert report["summary"]["failures"] == 24
    assert report["summary"]["failure_fraction"] == 1.0
    assert report["profiles"]["1.0"]["status"] == "profile_error"
    profile = {
        "status": "ok",
        "identity": report["profiles"]["1.0"]["identity"],
        "models": {"char_linear": {"coefficients": [0.0, 0.0]}},
        "char_linear_guard_sec": 0.0,
    }
    assert max_chars_for_window(profile, 1000, expected_identity=profile["identity"]) is None


def test_budget_boundary_and_prediction_use_profile_rate(tmp_path: Path) -> None:
    report = measure_corpus(_manifest(), tmp_path / "out", model_path=tmp_path / "missing", rates=(1.0,), synthesis_fn=_fake_synth)
    profile = report["profiles"]["1.0"]
    identity = profile["identity"]
    text = "новый текст"
    assert_profile_compatible(profile, identity)
    wrong_identity = dict(identity)
    wrong_identity["engine"] = "other"
    with pytest.raises(ValueError, match="engine"):
        assert_profile_compatible(profile, wrong_identity)
    prediction = predict_duration(profile, text, expected_identity=identity)
    assert prediction >= 0
    cap = max_chars_for_window(profile, int((prediction + 2.0) * 1000), expected_identity=identity)
    assert cap is not None and cap >= 0
    assert max_chars_for_window(profile, 0, expected_identity=identity) == 0
    with pytest.raises(ValueError):
        max_chars_for_window(profile, -1, expected_identity=identity)
    with pytest.raises(ValueError):
        max_chars_for_window(profile, True, expected_identity=identity)
    with pytest.raises(ValueError):
        max_chars_for_window(profile, float("nan"), expected_identity=identity)


def test_real_model_initialization_failure_has_failed_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "out"
    report = measure_corpus(_manifest(), output, model_path=tmp_path / "missing-model")
    assert report["status"] == "failed"
    assert report["summary"]["logical_observations"] == 0
    checkpoint = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "failed"
