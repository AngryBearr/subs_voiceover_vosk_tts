from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import utils.sh08_zero_timing as zero_timing
from utils.shortening_audio import verify_original_timing


def _row(case_id: str = "source:cue-1", *, status: str = "short_timing") -> dict[str, Any]:
    return {
        "case_id": case_id,
        "position": 1,
        "original": "Проверка.",
        "start": 100,
        "end": 900,
        "available_ms": 1000,
        "max_chars": 0,
        "status": status,
        "length_basis": "normalized_tts_text",
        "risk": {"severity": "hard", "reason": status, "short_timing": True},
    }


def _selection(*rows: dict[str, Any]) -> dict[str, Any]:
    return {"cases": [*rows] + [_row(f"other-{index}:cue-{index}") for index in range(19 - len(rows))]}


def test_build_plans_preserves_namespaced_identity_text_and_windows() -> None:
    row = _row("episode:cue-42")
    plan = zero_timing.build_plans([row])[0]
    assert plan.cue_id == row["case_id"]
    assert plan.text == row["original"]
    assert (plan.position, plan.start, plan.end, plan.available_ms) == (1, 100, 900, 1000)
    assert plan.max_chars == 0
    assert plan.required_ratio is None
    assert plan.reduction_fraction == 0.0
    assert plan.length_basis == row["length_basis"]
    assert plan.severity == row["risk"]["severity"]
    assert plan.reason == row["risk"]["reason"]
    assert plan.short_timing_risk is True


def test_invalid_zero_case_is_excluded_explicitly() -> None:
    selection = _selection(_row("bad:cue-0", status="invalid"))
    cases, zero_count, invalid_count = zero_timing._zero_cases(selection)
    assert zero_count == 19
    assert invalid_count == 1
    assert len(cases) == 18
    assert all(row["status"] != "invalid" for row in cases)


def test_malformed_and_duplicate_rows_fail_before_synthesis() -> None:
    malformed = _row()
    malformed["available_ms"] = 0
    with pytest.raises(ValueError, match="nonpositive"):
        zero_timing.build_plans([malformed])

    duplicate = _row("same:cue")
    with pytest.raises(ValueError, match="duplicate"):
        zero_timing.build_plans([duplicate, copy.deepcopy(duplicate)])


def _wav_synth(*, output_path: str, **_: Any) -> Path:
    import numpy as np
    import soundfile as sf

    sf.write(output_path, np.zeros(100, dtype=np.float32), 1000, subtype="PCM_16")
    return Path(output_path)


def test_run_uses_existing_verifier_with_fake_synthesis_only_in_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text("{}", encoding="utf-8")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}", encoding="utf-8")
    selection = _selection(_row("episode:cue-42"), _row("bad:cue-0", status="invalid"))
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    profile = {"identity": {"engine": "vosk"}}
    monkeypatch.setattr(zero_timing, "_load_json", lambda path: selection if path == selection_path else profile)
    monkeypatch.setattr(zero_timing, "model_metadata", lambda _: {"config_sha256": "config", "fingerprint": "fingerprint", "artifacts": []})
    monkeypatch.setattr(zero_timing, "profile_identity", lambda *_: {"engine": "vosk"})
    monkeypatch.setattr(zero_timing, "validate_character_profile", lambda *_args, **_kwargs: None)

    def fake_verify(plans: list[Any], output_dir: Path, **kwargs: Any) -> dict[str, dict[str, Any]]:
        return verify_original_timing(
            plans, output_dir, synthesis_fn=_wav_synth, **kwargs
        )

    monkeypatch.setattr(zero_timing, "verify_original_timing", fake_verify)
    summary = zero_timing.run(selection_path, profile_path, model, tmp_path / "result")
    assert summary["selection"] == {"zero_capacity_total": 19, "invalid_excluded": 1, "valid_measurements": 18}
    assert summary["evidence"][0]["case_id"] == "episode:cue-42"
    assert summary["counts"] == {"fits": 18, "overrun": 0, "unverified": 0}
