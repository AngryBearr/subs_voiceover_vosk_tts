from __future__ import annotations

import asyncio
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
    _validate_timing_evidence,
    run_trial,
)


def _wav_synth(*, text: str, output_path: str, **_: Any) -> Path:
    sf.write(output_path, np.zeros(50, dtype=np.float32), 1000, subtype="PCM_16")
    return Path(output_path)


def _models() -> tuple[ModelSpec, ModelSpec]:
    return ModelSpec("fake", "generator", "off"), ModelSpec("fake", "arbiter", "off")


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
