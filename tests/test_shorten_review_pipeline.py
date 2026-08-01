"""Offline tests for the Flash-to-Pro orchestration pipeline."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import utils.shorten_review_pipeline as pipeline_module
from utils.review_shortened_subtitles_deepseek import select_changed_targets
from utils.shortening_domain import CalibratedDurationModel, DurationSelectionPolicy
from utils.shorten_review_pipeline import (
    PipelineConfig,
    PipelineStageError,
    _critical_count,
    _is_critical,
    _mark_shortening,
    build_parser,
    run_pipeline,
)


def _item(text: str) -> dict[str, Any]:
    return {"index": 1, "text": [text], "start": 0, "end": 1000,
            "analysis": {"is_checked": True, "available_duration_sec": 1.0,
                          "effective_duration_sec": 1.0}}


def _duration_model(char_seconds: float = 1.0) -> CalibratedDurationModel:
    return CalibratedDurationModel(
        intercept_sec=0.0, seconds_per_budget_char=char_seconds, seconds_per_punctuation=0.0,
        measurement="edge_audio_decoded_with_libsndfile_pcm16_wav_emulation_measured_after_production_silence_reduction",
        voice="test-voice", rate="+0%", sample_count=30, episode_count=2,
        validation_method="leave_one_episode_out", validation_mae_sec=0.1,
        validation_rmse_sec=0.2, validation_r_squared=0.8,
    )


def _profile_json() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile_type": "calibrated_post_silence_duration",
        "measurement": "edge_audio_decoded_with_libsndfile_pcm16_wav_emulation_measured_after_production_silence_reduction",
        "voice": "test-voice", "rate": "+0%",
        "coefficients": {"intercept_sec": 0.0, "seconds_per_budget_char": 1.0,
                         "seconds_per_punctuation": 0.0},
        "training": {"sample_count": 30, "episode_count": 2},
        "validation": {"method": "leave_one_episode_out", "mae_sec": 0.1,
                        "rmse_sec": 0.2, "r_squared": 0.8},
    }


def _profile_item(index: int, text: str, duration: float = 10.0,
                  checked: bool = True) -> dict[str, Any]:
    value = {"index": index, "text": [text], "start": 0, "end": int(duration * 1000),
             "analysis": {"is_checked": checked, "available_duration_sec": duration,
                          "effective_duration_sec": duration}}
    return value


def test_cli_defaults() -> None:
    args = build_parser().parse_args(["input.json"])
    assert (args.threshold, args.avg_chars_per_sec) == (1.5, 13.0)
    assert (args.flash_max_iterations, args.flash_batch_size, args.flash_concurrency) == (1, 6, 3)
    assert (args.pro_thinking_mode, args.pro_batch_size, args.pro_context_window) == ("auto", 4, 3)
    assert args.pro_concurrency == 3
    assert (args.pro_risk_threshold, args.pro_max_input_tokens) == (35.0, 50_000)


def test_default_pro_stage_enables_semantic_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_review(*args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        captured.update(kwargs)
        return args[1], {"outcomes": [], "selected_indices": [], "fallback": []}, {}

    monkeypatch.setattr(pipeline_module, "review_subtitles", fake_review)
    config = PipelineConfig(Path("input.json"), Path("output"), "key")
    result = pipeline_module._default_pro_stage([_item("original")], [_item("short")], config)
    assert result[0][0]["text"] == ["short"]
    assert captured["enable_planner"] is True


def test_pipeline_order_original_integrity_refresh_and_combined_reports(tmp_path: Path) -> None:
    input_path = tmp_path / "subs_analyzed.json"
    original = [_item("A very long original subtitle phrase")]
    input_path.write_text(json.dumps(original), encoding="utf-8")
    calls: list[str] = []

    def flash(items: list[dict[str, Any]], config: PipelineConfig, output: Path,
              stem: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        calls.append("flash")
        items[0]["text"] = ["Flash short phrase"]
        return items, {"estimated_cost_usd": 0.1, "pricing_model": "flash", "tokens": 1}

    def pro(original_arg: list[dict[str, Any]], shortened: list[dict[str, Any]],
            config: PipelineConfig) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        calls.append("pro")
        assert original_arg == original
        original_arg[0]["text"] = ["mutation must not escape"]
        shortened[0]["text"] = ["Reviewed phrase"]
        return shortened, {"selected": 1, "selected_indices": [1], "routing": {"disabled": [1]}, "accepted_changed": [1],
                           "accepted_unchanged": [], "fallback": [],
                           "outcomes": [{"index": 1, "outcome": "verified"}]}, {
                               "estimated_cost_usd": 0.2, "pricing_model": "pro", "tokens": 2}

    output_dir = tmp_path / "result"
    final_path = run_pipeline(PipelineConfig(input_path, output_dir, "key"), flash, pro)
    assert calls == ["flash", "pro"]
    assert final_path == output_dir / "subs_analyzed_reviewed.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    assert final[0]["text"] == ["Reviewed phrase"]
    assert final[0]["analysis"]["is_critical"] is False
    assert final[0]["shortening"] == {"required_initially": True, "flash_changed": True,
                                      "pro_selected": True, "max_chars": 19, "final_chars": 15,
                                      "status": "resolved", "reason": "verified_within_budget"}
    usage = json.loads((output_dir / "subs_analyzed.pipeline.usage.json").read_text(encoding="utf-8"))
    assert usage["estimated_cost_usd"] == 0.3
    assert usage["flash"]["pricing_model"] == "flash"
    assert usage["pro"]["pricing_model"] == "pro"
    report = json.loads((output_dir / "subs_analyzed.pipeline.report.json").read_text(encoding="utf-8"))
    assert report["flash"]["changed"] == 1 and report["pro"]["selected"] == 1
    assert report["status"] == "completed" and report["unresolved_count"] == 0


def test_pro_failure_keeps_flash_artifacts_and_report(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps([_item("Original long phrase")]), encoding="utf-8")

    def flash(items: list[dict[str, Any]], config: PipelineConfig, output: Path,
              stem: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        changed = copy.deepcopy(items)
        changed[0]["text"] = ["Flash phrase"]
        return changed, {"estimated_cost_usd": 0.1}

    def failed(*args: Any) -> Any:
        raise RuntimeError("offline failure")

    output = tmp_path / "result"
    with pytest.raises(PipelineStageError, match="Flash result kept"):
        run_pipeline(PipelineConfig(input_path, output, "key"), flash, failed)
    assert (output / "flash/input_final.json").exists()
    report = json.loads((output / "input.pipeline.report.json").read_text(encoding="utf-8"))
    assert report["status"] == "pro_failed"


def test_pipeline_marks_minimum_one_word_budget_unresolved(tmp_path: Path) -> None:
    input_path = tmp_path / "one.json"
    value = _item("Сверхдлинноеслово")
    value["end"] = 100
    value["analysis"].update({"available_duration_sec": 0.1, "effective_duration_sec": 0.1})
    input_path.write_text(json.dumps([value]), encoding="utf-8")

    def flash(items: list[dict[str, Any]], config: PipelineConfig, output: Path,
              stem: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return items, {"estimated_cost_usd": 0}

    def pro(original: list[dict[str, Any]], shortened: list[dict[str, Any]],
            config: PipelineConfig) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        return shortened, {"selected": 1, "selected_indices": [1], "fallback": [1],
                           "outcomes": [{"index": 1, "outcome": "fallback"}]}, {
                               "estimated_cost_usd": 0}

    output = tmp_path / "result"
    final_path = run_pipeline(PipelineConfig(input_path, output, "key"), flash, pro)
    final = json.loads(final_path.read_text(encoding="utf-8"))
    assert final[0]["shortening"]["status"] == "unresolved"
    assert final[0]["shortening"]["reason"] == "minimum_text_exceeds_budget"
    report = json.loads((output / "one.pipeline.report.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed_with_unresolved"
    assert report["unresolved_indices"] == [1]


def test_unselected_not_required_item_does_not_become_unresolved(tmp_path: Path) -> None:
    input_path = tmp_path / "mixed.json"
    required = _item("A very long required subtitle phrase")
    optional = {"index": 2, "text": ["Short"], "start": 1000, "end": 5000,
                "analysis": {"is_checked": True, "available_duration_sec": 4.0,
                             "effective_duration_sec": 4.0}}
    input_path.write_text(json.dumps([required, optional]), encoding="utf-8")

    def flash(items: list[dict[str, Any]], config: PipelineConfig, output: Path,
              stem: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return items, {"estimated_cost_usd": 0}

    def pro(original: list[dict[str, Any]], shortened: list[dict[str, Any]],
            config: PipelineConfig) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        return shortened, {
            "selected": 1,
            "selected_indices": [1],
            "fallback": [1],
            "outcomes": [{"index": 1, "outcome": "unresolved"}],
        }, {"estimated_cost_usd": 0}

    output_dir = tmp_path / "result"
    final_path = run_pipeline(PipelineConfig(input_path, output_dir, "key"), flash, pro)
    final = json.loads(final_path.read_text(encoding="utf-8"))
    assert final[0]["shortening"]["status"] == "unresolved"
    assert final[1]["shortening"]["status"] == "not_required"
    report = json.loads((output_dir / "mixed.pipeline.report.json").read_text(encoding="utf-8"))
    assert report["unresolved_count"] == 1
    assert report["unresolved_indices"] == [1]


def test_pipeline_context_hydrates_reports_and_never_processes_context_only_items(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "sparse.json"
    context_path = tmp_path / "full.json"
    target_text = "A very long target subtitle phrase that is critical"
    sparse = [{"index": 2, "text": [target_text], "start": 1000, "end": 2000,
               "analysis": {"is_checked": True}}]
    context = [
        {"index": 1, "text": ["Before"], "start": 0, "end": 1000},
        {"index": 2, "text": [target_text], "start": 1000, "end": 2000},
        {"index": 3, "text": ["Close next neighbor"], "start": 2200, "end": 3200},
    ]
    input_path.write_text(json.dumps(sparse), encoding="utf-8")
    context_path.write_text(json.dumps(context), encoding="utf-8")
    observed: dict[str, Any] = {}

    def flash(items: list[dict[str, Any]], config: PipelineConfig, output: Path,
              stem: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        observed["flash_len"] = len(items)
        observed["flash_timing"] = items[0]["analysis"]["effective_duration_sec"]
        changed = copy.deepcopy(items)
        changed[0]["text"] = ["Short target phrase"]
        return changed, {"estimated_cost_usd": 0}

    def pro(original: list[dict[str, Any]], shortened: list[dict[str, Any]],
            config: PipelineConfig) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        observed["pro_len"] = len(original)
        observed["pro_timing"] = original[0]["analysis"]["effective_duration_sec"]
        return shortened, {"selected": 1, "selected_indices": [2],
                           "outcomes": [{"index": 2, "outcome": "verified"}]}, {
                               "estimated_cost_usd": 0}

    output_dir = tmp_path / "result"
    config = PipelineConfig(input_path, output_dir, "key", context_source=context_path)
    final_path = run_pipeline(config, flash, pro)
    final = json.loads(final_path.read_text(encoding="utf-8"))

    assert observed == {"flash_len": 1, "flash_timing": 1.0,
                        "pro_len": 1, "pro_timing": 1.0}
    assert final[0]["shortening"]["required_initially"] is True
    assert final[0]["shortening"]["status"] == "resolved"


def test_duration_profile_parser_defaults_and_flags() -> None:
    defaults = build_parser().parse_args(["input.json"])
    assert defaults.duration_profile is None and defaults.duration_fit_ratio == 1.0
    explicit = build_parser().parse_args(["input.json", "--duration-profile", "profile.json",
                                           "--duration-fit-ratio", "1.25"])
    assert explicit.duration_profile == Path("profile.json")
    assert explicit.duration_fit_ratio == 1.25


@pytest.mark.parametrize("argv", [
    ["input.json", "--duration-profile", "missing.json"],
    ["input.json", "--duration-profile", "profile.json", "--duration-fit-ratio", "0"],
    ["input.json", "--duration-profile", "profile.json", "--duration-fit-ratio", "nan"],
])
def test_profile_cli_invalid_before_key_or_pipeline(tmp_path: Path, argv: list[str],
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    profile_path = tmp_path / "profile.json"
    if "missing" not in argv:
        profile_path.write_text("not json", encoding="utf-8")
    argv = [str(profile_path) if value == "profile.json" else value for value in argv]
    monkeypatch.setattr(pipeline_module, "load_api_key", lambda _: pytest.fail("API key lookup"))
    monkeypatch.setattr(pipeline_module, "run_pipeline", lambda *args: pytest.fail("pipeline execution"))
    assert pipeline_module.main(argv) == 2


def test_profile_cli_loads_once_and_keeps_model_identity(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile_json()), encoding="utf-8")
    model = _duration_model()
    calls: list[Path] = []
    captured: list[PipelineConfig] = []

    def loader(path: Path) -> CalibratedDurationModel:
        calls.append(path)
        return model

    def fake_run(config: PipelineConfig) -> Path:
        captured.append(config)
        return tmp_path / "final.json"

    monkeypatch.setattr(pipeline_module, "load_calibrated_duration_profile", loader)
    monkeypatch.setattr(pipeline_module, "run_pipeline", fake_run)
    assert pipeline_module.main(["input.json", "--api-key", "key", "--duration-profile",
                                str(profile_path), "--duration-fit-ratio", "1.25"]) == 0
    assert calls == [profile_path]
    assert captured[0].duration_profile is model
    assert captured[0].duration_fit_ratio == 1.25
    assert not hasattr(captured[0].duration_profile, "path")


@pytest.mark.parametrize("profile, expected_policy", [(None, None), (_duration_model(), "profile")])
def test_default_flash_stage_passes_shared_policy_and_legacy_budget(monkeypatch: pytest.MonkeyPatch,
                                                                     profile: Any,
                                                                     expected_policy: Any) -> None:
    captured: dict[str, Any] = {}
    usage: dict[str, int] = {}

    def fake_factory(*args: Any, **kwargs: Any) -> Any:
        return object()

    def fake_iterative(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        captured.update(kwargs)
        captured["positional"] = args
        return args[0]

    monkeypatch.setattr(pipeline_module, "make_deepseek_shorten_func", fake_factory)
    monkeypatch.setattr(pipeline_module, "run_iterative_shortening", fake_iterative)
    monkeypatch.setattr(pipeline_module, "usage_summary", lambda value: {"tokens": 1})
    config = PipelineConfig(Path("input.json"), Path("output"), "key",
                            duration_profile=profile, duration_fit_ratio=1.25)
    pipeline_module._default_flash_stage([_profile_item(1, "text")], config, Path("output"), "input")
    assert (captured["positional"][3], captured["positional"][4], captured["positional"][5]) == (1.5, 3, 1)
    policy = captured["selection_policy"]
    if expected_policy is None:
        assert policy is None
    else:
        assert isinstance(policy, DurationSelectionPolicy)
        assert policy.model is profile and policy.fit_ratio == 1.25


@pytest.mark.parametrize("profile, expected_ratio", [(None, 1.0), (_duration_model(), 1.25)])
def test_default_pro_stage_forwards_profile_identity(monkeypatch: pytest.MonkeyPatch,
                                                      profile: Any, expected_ratio: float) -> None:
    captured: dict[str, Any] = {}

    async def fake_review(*args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        captured.update(kwargs)
        return args[1], {}, {}

    monkeypatch.setattr(pipeline_module, "review_subtitles", fake_review)
    config = PipelineConfig(Path("input.json"), Path("output"), "key",
                            duration_profile=profile, duration_fit_ratio=expected_ratio)
    pipeline_module._default_pro_stage([_profile_item(1, "original")], [_profile_item(1, "short")], config)
    assert captured["duration_profile"] is profile
    assert captured["duration_fit_ratio"] == expected_ratio


def test_reviewer_profile_selection_replaces_legacy_mismatch() -> None:
    model = _duration_model()
    original = [_profile_item(1, "legacy noncritical", 2.0),
                _profile_item(2, "critical", 10.0),
                _profile_item(3, "changed text", 10.0),
                _profile_item(4, "unchecked", 1.0),
                _profile_item(5, "zero timing", 0.0)]
    current = copy.deepcopy(original)
    current[0]["analysis"]["mismatch_ratio"] = 2.0
    current[1]["analysis"]["mismatch_ratio"] = 2.0
    current[1]["text"] = ["critical"]
    current[2]["text"] = ["changed"]
    current[3]["analysis"]["is_checked"] = False
    current[4]["analysis"]["is_checked"] = True
    targets = select_changed_targets(original, current, duration_profile=model, duration_fit_ratio=1.0)
    assert [target.index for target in targets] == [1, 3]
    assert targets[0].requires_shortening is True
    assert targets[1].requires_shortening is False


def test_critical_helpers_use_calibrated_replacement_and_legacy_without_profile() -> None:
    model = _duration_model()
    critical = _profile_item(1, "long text", 2.0)
    critical["analysis"]["mismatch_ratio"] = 0.1
    assert _is_critical(critical, 1.5, model) is True
    assert _is_critical(critical, 1.5) is False
    assert _critical_count([critical], PipelineConfig(Path("i"), Path("o"), "k",
                                                        duration_profile=model)) == 1
    assert _critical_count([critical], PipelineConfig(Path("i"), Path("o"), "k")) == 0


def test_mark_shortening_calibrated_reasons_and_legacy_shape() -> None:
    model = _duration_model()
    source = _profile_item(1, "original text", 2.0)
    source["analysis"]["mismatch_ratio"] = 0.1
    flash = [copy.deepcopy(source)]
    final = [copy.deepcopy(source)]
    final[0]["text"] = ["ok"]
    report = {"selected_indices": [1], "outcomes": [{"index": 1, "outcome": "verified"}]}
    unresolved, values = _mark_shortening([source], flash, final, report,
                                          PipelineConfig(Path("i"), Path("o"), "k",
                                                         duration_profile=model))
    assert unresolved == [] and values[0]["shortening"]["required_initially"] is True
    assert values[0]["shortening"]["max_chars"] == 39

    over_duration = copy.deepcopy(source)
    over_duration["text"] = ["12345"]
    _, result = _mark_shortening([source], flash, [over_duration],
                                 {"selected_indices": [1], "outcomes": [{"index": 1, "outcome": "verified"}]},
                                 PipelineConfig(Path("i"), Path("o"), "k", duration_profile=model))
    assert result[0]["shortening"]["reason"] == "still_over_duration"
    over_quality = copy.deepcopy(over_duration)
    _, result = _mark_shortening([source], flash, [over_quality],
                                 {"selected_indices": [1], "outcomes": [{"index": 1, "outcome": "unresolved",
                                                                            "error": "quality_flags:object"}]},
                                 PipelineConfig(Path("i"), Path("o"), "k", duration_profile=model))
    assert result[0]["shortening"]["reason"] == "quality_unresolved"

    low_model = _duration_model(0.01)
    over_chars = copy.deepcopy(source)
    over_chars["text"] = ["x " * 21]
    _, result = _mark_shortening([source], flash, [over_chars],
                                 {"selected_indices": [1], "outcomes": [{"index": 1, "outcome": "verified"}]},
                                 PipelineConfig(Path("i"), Path("o"), "k", duration_profile=low_model))
    assert result[0]["shortening"]["reason"] == "still_over_budget"
    one_word = copy.deepcopy(over_chars)
    one_word["text"] = ["x" * 40]
    one_word["analysis"]["effective_duration_sec"] = 0.1
    _, result = _mark_shortening([source], flash, [one_word],
                                 {"selected_indices": [1], "outcomes": [{"index": 1, "outcome": "verified"}]},
                                 PipelineConfig(Path("i"), Path("o"), "k", duration_profile=low_model))
    assert result[0]["shortening"]["reason"] == "minimum_text_exceeds_budget"


def test_pipeline_profile_metadata_counts_and_custom_stage_signatures(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    source = _profile_item(1, "long text", 2.0)
    source["analysis"]["mismatch_ratio"] = 0.1
    input_path.write_text(json.dumps([source]), encoding="utf-8")
    model = _duration_model()

    def flash(items: list[dict[str, Any]], config: PipelineConfig, output: Path,
              stem: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if config.duration_profile is not None:
            assert config.duration_profile is model
        return items, {"estimated_cost_usd": 0}

    def pro(original: list[dict[str, Any]], shortened: list[dict[str, Any]],
            config: PipelineConfig) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        if config.duration_profile is not None:
            assert config.duration_profile is model
        return shortened, {"selected": 1, "selected_indices": [1],
                           "outcomes": [{"index": 1, "outcome": "verified"}]}, {"estimated_cost_usd": 0}

    output = tmp_path / "result"
    run_pipeline(PipelineConfig(input_path, output, "secret-key", duration_profile=model), flash, pro)
    report = json.loads((output / "input.pipeline.report.json").read_text(encoding="utf-8"))
    usage = json.loads((output / "input.pipeline.usage.json").read_text(encoding="utf-8"))
    assert report["duration_profile"]["fit_ratio"] == 1.0
    assert report["flash"]["critical_start"] == 1 and report["final_critical"] == 1
    assert usage["duration_profile"]["voice"] == "test-voice"
    assert "secret-key" not in json.dumps(report) and "profile.json" not in json.dumps(report)

    plain_output = tmp_path / "plain"
    run_pipeline(PipelineConfig(input_path, plain_output, "secret-key"), flash, pro)
    plain_report = json.loads((plain_output / "input.pipeline.report.json").read_text(encoding="utf-8"))
    plain_usage = json.loads((plain_output / "input.pipeline.usage.json").read_text(encoding="utf-8"))
    assert "duration_profile" not in plain_report and "duration_profile" not in plain_usage


def test_programmatic_invalid_profile_ratio_fails_before_stage(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps([_profile_item(1, "text")]), encoding="utf-8")
    called = False

    def flash(*args: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("stage called")

    with pytest.raises(ValueError, match="finite and positive"):
        run_pipeline(PipelineConfig(input_path, tmp_path / "out", "key",
                                    duration_profile=_duration_model(), duration_fit_ratio=0), flash, flash)
    assert called is False


def test_unchecked_profile_over_duration_item_remains_not_required() -> None:
    model = _duration_model()
    source = _profile_item(42, "original", 1.0, checked=False)
    flash = copy.deepcopy(source)
    final = copy.deepcopy(source)
    final["text"] = ["x" * 10]
    unresolved, values = _mark_shortening(
        [source], [flash], [final], {"selected_indices": [], "outcomes": []},
        PipelineConfig(Path("i"), Path("o"), "key", duration_profile=model),
    )
    shortening = values[0]["shortening"]
    assert unresolved == []
    assert shortening["required_initially"] is False
    assert shortening["status"] == "not_required"
    assert shortening["reason"] == "within_budget"
