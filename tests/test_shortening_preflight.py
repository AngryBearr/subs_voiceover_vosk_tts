from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import utils.shortening_audio as shortening_audio
from utils.llm_types import DetailedResponse
from utils.shortening_audio import verify_original_timing
from utils.shortening_plan import CuePlan, PlannerConfig, analysis_for_text, plan_subtitles
from utils.shortening_runner import (
    ModelSpec,
    RunConfig,
    _build_report,
    _apply_preflight,
    _forecast_plan,
    _new_record,
    _validate_timing_evidence,
    run_trial,
)


def _wav_synth(*, text: str, output_path: str, **_: Any) -> Path:
    sf.write(output_path, np.zeros(50, dtype=np.float32), 1000, subtype="PCM_16")
    return Path(output_path)


def _models() -> tuple[ModelSpec, ModelSpec]:
    return ModelSpec("fake", "generator", "off"), ModelSpec("fake", "arbiter", "off")


def _calibrated_planner() -> PlannerConfig:
    identity = {
        "engine": "vosk", "voice": 8, "rate": 1.5,
        "model_config_sha256": None, "model_fingerprint": "fingerprint",
        "normalizer": "vosk_punctuation_v1_no_ruaccent", "normalizer_sha256": "hash",
    }
    profile = {
        "status": "ok", "identity": identity,
        "normalizer": "vosk_punctuation_v1_no_ruaccent",
        "input_character_basis": "normalized_tts_text", "rate": 1.5,
        "models": {"char_linear": {"name": "char_linear", "rate": 1.5, "coefficients": [0.01, 0.005]}},
        "char_linear_guard_sec": 0.06,
    }
    return PlannerConfig(
        cps=13.0, max_speed=1.5, length_basis="normalized_tts_text",
        calibration_profile=profile, calibration_identity=identity,
    )


def test_run_trial_preflight_resolves_short_and_repeated_fit_without_llm(tmp_path: Path) -> None:
    items = [
        {"index": 41, "start": 0, "end": 100, "text": "Да.", "unknown": "keep"},
        {"index": 42, "start": 300, "end": 400, "text": "Пошли, пошли, пошли.", "extra": [1]},
    ]
    source = json.loads(json.dumps(items, ensure_ascii=False))
    planner = PlannerConfig(cps=13.0, max_speed=1.5, gap_ms=200)
    plans = plan_subtitles(items, planner)
    assert [plan.status for plan in plans] == ["short_timing", "needs_shortening"]
    calls = 0

    async def request_fn(*_: Any) -> DetailedResponse:
        nonlocal calls
        calls += 1
        raise AssertionError("measured fits must not call an LLM")

    def verifier(selected: list[CuePlan], output_dir: Path) -> dict[str, dict[str, Any]]:
        return verify_original_timing(
            selected,
            output_dir,
            model_path=tmp_path / "model",
            synthesis_fn=_wav_synth,
            text_mode="raw",
        )

    (tmp_path / "model").mkdir()
    generator, arbiter = _models()
    report = asyncio.run(
        run_trial(
            items=items,
            plans=plans,
            target_ids=[plan.cue_id for plan in plans],
            planner_config=planner,
            run_config=RunConfig(),
            generator=generator,
            arbiter=arbiter,
            output_dir=tmp_path / "trial",
            request_fn=request_fn,
            original_timing_verifier=verifier,
        )
    )

    assert calls == 0
    assert items == source
    assert report["summary"]["accepted_count"] == 0
    assert report["summary"]["actual_fits_count"] == 2
    assert report["summary"]["resolved_count"] == 2
    assert report["summary"]["manual_count"] == 0
    assert all(record["status"] == "actual_fits" for record in report["records"])
    checkpoint = json.loads((tmp_path / "trial" / "subtitles.json").read_text(encoding="utf-8"))
    for item in checkpoint:
        analysis = item["analysis"]
        assert analysis["reason"] == "actual_fit"
        assert analysis["is_critical"] is False
        assert analysis["estimate_source"] == "vosk_measurement"
    assert checkpoint[0]["index"] == 41
    assert checkpoint[0]["unknown"] == "keep"
    assert checkpoint[0]["text"] == "Да."
    assert checkpoint[1]["extra"] == [1]


def test_measured_fit_does_not_transfer_to_different_text() -> None:
    plan = CuePlan(
        "cue-0", 0, "Да.", 0, 100, 100, 1, 3.0, 0.0, "fits", "fits",
        words_count=1, estimate_source="vosk_measurement",
    )
    analysis = analysis_for_text(plan, "Пошли, пошли, пошли.", PlannerConfig())
    assert analysis["estimate_source"] == "nominal_cps_estimate"
    assert analysis["reason"] != "actual_fit"
    assert analysis["is_critical"] is True


def test_calibrated_measured_fit_is_actual_fit_and_repeated_overflow_stays_eligible(
    tmp_path: Path,
) -> None:
    items = [
        {"start": 0, "end": 100, "text": "Да, нет."},
        {"start": 300, "end": 400, "text": "да да да"},
    ]
    planner = _calibrated_planner()
    plans = plan_subtitles(items, planner)
    assert [plan.status for plan in plans] == ["short_timing", "needs_shortening"]
    calls: list[list[str]] = []

    def verifier(selected: list[CuePlan], _: Path) -> dict[str, dict[str, Any]]:
        assert {plan.cue_id for plan in selected} == {"cue-0", "cue-1"}
        return {
            "cue-0": {"original": "Да, нет.", "available_ms": 100, "duration_sec": 0.05},
            "cue-1": {"original": "да да да", "available_ms": 100, "duration_sec": 0.2},
        }

    async def request(_: ModelSpec, __: str, user: str, schema: dict[str, Any], ___: bool) -> DetailedResponse:
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        ids = [row["cue_id"] for row in payload["targets"]]
        calls.append(ids)
        if "verdict" in json.dumps(schema):
            response = {"results": [{"cue_id": cue_id, "verdict": "accept", "issues": [], "advice": ""} for cue_id in ids]}
        else:
            response = {"results": [{"cue_id": cue_id, "text": "да", "status": "ok"} for cue_id in ids]}
        return DetailedResponse(json.dumps(response), [])

    generator, arbiter = _models()
    report = asyncio.run(run_trial(
        items=items, plans=plans, target_ids=["cue-0", "cue-1"], planner_config=planner,
        run_config=RunConfig(max_repairs=0), generator=generator, arbiter=arbiter,
        output_dir=tmp_path / "calibrated", request_fn=request, original_timing_verifier=verifier,
    ))
    first, second = report["records"]
    assert first["status"] == "actual_fits"
    assert second["status"] == "accepted"
    assert calls == [["cue-1"], ["cue-1"]]
    checkpoint = json.loads((tmp_path / "calibrated" / "subtitles.json").read_text(encoding="utf-8"))
    assert checkpoint[0]["analysis"]["estimate_source"] == "vosk_measurement"
    assert checkpoint[0]["analysis"]["estimate_label"] == "vosk_calibration_char_linear"
    assert checkpoint[0]["analysis"]["reason"] == "actual_fit"
    assert checkpoint[0]["analysis"]["is_critical"] is False


@pytest.mark.parametrize("duration", [10**1000, float("inf"), float("nan"), True, 0, -1])
def test_timing_evidence_rejects_unsafe_duration_without_raising(duration: Any) -> None:
    plan = CuePlan("cue-0", 0, "Да.", 0, 100, 100, 1, 3.0, 0.0, "short_timing", "short_timing")
    valid, fits, evidence = _validate_timing_evidence(
        plan, {"original": "Да.", "available_ms": 100, "duration_sec": duration}
    )
    assert valid is False
    assert fits is False
    assert evidence["error"]["type"] == "InvalidDuration"


def test_timing_evidence_rejects_huge_window_and_error_metadata() -> None:
    plan = CuePlan("cue-0", 0, "Да.", 0, 100, 100, 1, 3.0, 0.0, "short_timing", "short_timing")
    for evidence in (
        {"original": "Да.", "available_ms": 10**1000, "duration_sec": 0.01},
        {"original": "Да.", "available_ms": 100, "duration_sec": 0.01, "error_type": "RuntimeError"},
        {"original": "Да.", "available_ms": 100, "duration_sec": 0.01, "status": "error"},
    ):
        valid, fits, _ = _validate_timing_evidence(plan, evidence)
        assert (valid, fits) == (False, False)


def test_preflight_error_string_is_preserved_and_does_not_call_llm(tmp_path: Path) -> None:
    items = [{"start": 0, "end": 100, "text": "Да."}]
    planner = PlannerConfig()
    plan = plan_subtitles(items, planner)[0]
    calls = 0

    async def request_fn(*_: Any) -> DetailedResponse:
        nonlocal calls
        calls += 1
        raise AssertionError("unverified timing must not call an LLM")

    def verifier(*_: Any) -> dict[str, dict[str, Any]]:
        return {plan.cue_id: {"error": "synthesis failed", "error_type": "RuntimeError"}}

    generator, arbiter = _models()
    report = asyncio.run(
        run_trial(
            items=items, plans=[plan], target_ids=[plan.cue_id], planner_config=planner,
            run_config=RunConfig(), generator=generator, arbiter=arbiter,
            output_dir=tmp_path / "error", request_fn=request_fn,
            original_timing_verifier=verifier,
        )
    )
    assert calls == 0
    assert report["records"][0]["status"] == "timing_unverified"
    assert report["records"][0]["timing_evidence"]["error"] == "synthesis failed"


def test_report_manual_fraction_keeps_measured_fits_in_denominator(tmp_path: Path) -> None:
    planner = PlannerConfig()
    generator, arbiter = _models()
    plans = [
        CuePlan(f"cue-{index}", index, "x", 0, 1000, 1000, 13, None, 0, "fits", "fits")
        for index in range(3)
    ]

    def report_for(statuses: list[str]) -> dict[str, Any]:
        records = [
            {
                "status": status,
                "history": [],
                "budget": {"status": "needs_shortening"},
                "initial_accepted": False,
                "final_review": None,
            }
            for status in statuses
        ]
        return _build_report(
            [], plans, [plan.cue_id for plan in plans], planner, RunConfig(), generator, arbiter,
            tmp_path, records, {}, None, "source", run_state="completed",
        )

    mixed = report_for(["actual_fits", "actual_fits", "manual"])["summary"]
    assert mixed["manual_count"] == 1
    assert mixed["manual_fraction"] == pytest.approx(1 / 3)
    assert mixed["resolved_count"] == 2
    assert report_for(["actual_fits"] * 3)["summary"]["manual_fraction"] == 0
    assert report_for(["manual"] * 3)["summary"]["manual_fraction"] == 1


def test_old_repeated_short_plan_is_normalized_and_budget_is_consistent(tmp_path: Path) -> None:
    items = [{"start": 0, "end": 100, "text": "Пошли, пошли, пошли."}]
    planner = PlannerConfig()
    plan = plan_subtitles(items, planner)[0]
    old_plan = CuePlan(
        plan.cue_id, plan.position, plan.text, plan.start, plan.end, plan.available_ms,
        plan.max_chars, plan.required_ratio, plan.reduction_fraction, plan.severity,
        "short_timing", words_count=plan.words_count, repetition_candidate=True,
        short_timing_risk=True,
    )
    generator, arbiter = _models()

    async def fake(*_: Any) -> DetailedResponse:
        return DetailedResponse(
            json.dumps({"results": [{"cue_id": "cue-0", "text": "Пошли", "status": "ok"}]}), []
        )

    # The fake response is intentionally never reviewed; this assertion only checks
    # that the repeated overflow enters the normal shortening path.
    report = asyncio.run(
        run_trial(
            items=items, plans=[old_plan], target_ids=["cue-0"], planner_config=planner,
            run_config=RunConfig(max_repairs=0), generator=generator, arbiter=arbiter,
            output_dir=tmp_path / "old", request_fn=fake,
        )
    )
    assert report["records"][0]["budget"]["status"] == "needs_shortening"


def test_verifier_rejects_duplicate_and_invalid_plans_before_synthesis(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    plan = CuePlan("same", 0, "Да.", 0, 100, 100, 1, 3.0, 0.0, "short_timing", "short_timing")
    with pytest.raises(ValueError, match="duplicate"):
        verify_original_timing([plan, plan], tmp_path / "duplicate", model_path=model, synthesis_fn=_wav_synth)
    invalid = CuePlan("bad", 0, "Да.", 0, 0, 0, 0, None, 0.0, "invalid", "invalid", invalid_timing=True)
    with pytest.raises(ValueError, match="invalid timing"):
        verify_original_timing([invalid], tmp_path / "invalid", model_path=model, synthesis_fn=_wav_synth)


def test_verifier_measures_every_distinct_cue_and_preserves_one_synthesis_error(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    items = [
        {"start": index * 300, "end": index * 300 + 100, "text": f"x{index}."}
        for index in range(7)
    ]
    plans = plan_subtitles(items, PlannerConfig())
    calls: list[str] = []

    def synthesize(*, text: str, output_path: str, **_: Any) -> Path:
        calls.append(text)
        if text == "x3.":
            raise RuntimeError("one cue failed")
        return _wav_synth(text=text, output_path=output_path)

    evidence = verify_original_timing(
        plans,
        tmp_path / "seven",
        model_path=model,
        synthesis_fn=synthesize,
        text_mode="raw",
    )
    assert calls == [f"x{index}." for index in range(7)]
    assert set(evidence) == {f"cue-{index}" for index in range(7)}
    assert all("available_ms" in row for cue, row in evidence.items() if cue != "cue-3")
    assert "available_ms" not in evidence["cue-3"]
    assert evidence["cue-3"]["error"]["type"] == "RuntimeError"
    for row in evidence.values():
        assert row["speech_rate"] == 1.5
        assert row["voice"] == 8
        assert row["model_path"] == str(model)
        assert row["preprocessing"] == "raw_no_ruaccent"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("original_text", "other", "original"),
        ("available_ms", 999, "window"),
        ("cue_id", "other", "cue IDs"),
    ],
)
def test_verifier_rejects_mismatched_measured_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: Any, message: str
) -> None:
    plan = CuePlan("cue-0", 0, "Да.", 0, 100, 100, 1, 3.0, 0.0, "short_timing", "short_timing")

    def fake_verify_audio(*_: Any, **__: Any) -> dict[str, Any]:
        row: dict[str, Any] = {
            "cue_id": "cue-0", "position": 0, "original_text": "Да.", "available_ms": 100,
            "original": {"duration_sec": 0.05},
        }
        row[field] = value
        return {
            "speech_rate": 1.5, "voice": 8, "model_path": "model",
            "preprocessing": "raw_no_ruaccent", "records": [row],
        }

    monkeypatch.setattr(shortening_audio, "verify_audio", fake_verify_audio)
    with pytest.raises(ValueError, match=message):
        shortening_audio.verify_original_timing(
            [plan], tmp_path / field, model_path=tmp_path / "model", synthesis_fn=_wav_synth
        )


def test_zero_fallback_risk_survives_runner_and_only_measurement_yields_actual_fit(tmp_path: Path) -> None:
    config = _calibrated_planner()
    assert config.calibration_profile is not None
    profile = dict(config.calibration_profile, char_linear_guard_sec=1.0)
    config = replace(config, calibration_profile=profile)
    item = {"start": 0, "end": 158, "text": "Да."}
    plan = _forecast_plan(plan_subtitles([item], config)[0])
    assert plan.status == "fits" and plan.short_timing_risk
    assert plan.reason == "calibrated_zero_budget"
    assert plan.max_chars == 3 and plan.calibrated_max_chars == 0
    record = _new_record(item, plan)
    assert record["status"] != "actual_fits"

    def verifier(selected: list[CuePlan], _: Path) -> dict[str, dict[str, Any]]:
        assert selected == [plan]
        return {plan.cue_id: {"original": plan.text, "available_ms": 158, "duration_sec": 0.1}}

    effective, risk_ids = _apply_preflight([plan], {plan.cue_id: record}, verifier, tmp_path)
    assert risk_ids == {plan.cue_id}
    assert record["status"] == "actual_fits"
    assert record["forecast"]["calibrated_max_chars"] == record["budget"]["calibrated_max_chars"] == 0
    assert record["forecast"]["max_chars"] == record["budget"]["max_chars"] == 3
    assert effective[plan.cue_id].estimate_source == "vosk_measurement"
    assert analysis_for_text(effective[plan.cue_id], "Да!", config)["short_timing_risk"]


def test_mocked_zero_fallback_acceptance_is_text_only_and_resume_preserves_raw_cap(tmp_path: Path) -> None:
    config = _calibrated_planner()
    assert config.calibration_profile is not None
    config = replace(config, calibration_profile=dict(config.calibration_profile, char_linear_guard_sec=1.0))
    items = [{"start": 0, "end": 158, "text": "да да да"}]
    plans = plan_subtitles(items, config)
    calls: list[str] = []

    def verifier(_: list[CuePlan], __: Path) -> dict[str, dict[str, Any]]:
        return {"cue-0": {"original": "да да да", "available_ms": 158, "duration_sec": 0.3}}

    async def request(_: ModelSpec, __: str, ___: str, schema: dict[str, Any], ____: bool) -> DetailedResponse:
        if "verdict" in json.dumps(schema):
            calls.append("review")
            result = {"cue_id": "cue-0", "verdict": "accept", "issues": [], "advice": ""}
        else:
            calls.append("generation")
            result = {"cue_id": "cue-0", "text": "да", "status": "ok"}
        return DetailedResponse(json.dumps({"results": [result]}), [])

    generator, arbiter = _models()
    kwargs = dict(
        items=items, plans=plans, target_ids=["cue-0"], planner_config=config,
        run_config=RunConfig(max_repairs=0), generator=generator, arbiter=arbiter,
        output_dir=tmp_path / "fallback", request_fn=request, original_timing_verifier=verifier,
    )
    report = asyncio.run(run_trial(**kwargs))
    record = report["records"][0]
    assert record["status"] == "accepted" and record["final_text"] == "да"
    assert report["summary"]["actual_fits_count"] == 0
    assert record["budget"]["max_chars"] == record["forecast"]["max_chars"] == 3
    assert record["budget"]["calibrated_max_chars"] == record["forecast"]["calibrated_max_chars"] == 0
    saved = json.loads((tmp_path / "fallback/subtitles.json").read_text(encoding="utf-8"))[0]["analysis"]
    assert saved["reason"] == "calibrated_zero_budget" and saved["short_timing_risk"]
    assert saved["budget_basis"] == "nominal_cps_fallback_zero_calibration"
    assert "Audio timing is unverified" in saved["timing_warning"]
    resumed = asyncio.run(run_trial(**kwargs, resume=True))
    assert resumed["records"][0]["budget"] == record["budget"]
    assert calls == ["generation", "review"]  # Resume replays, without new mocked requests.
