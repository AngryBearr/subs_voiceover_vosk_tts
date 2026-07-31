"""Offline tests for the Flash-to-Pro orchestration pipeline."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import utils.shorten_review_pipeline as pipeline_module
from utils.shorten_review_pipeline import PipelineConfig, PipelineStageError, build_parser, run_pipeline


def _item(text: str) -> dict[str, Any]:
    return {"index": 1, "text": [text], "start": 0, "end": 1000,
            "analysis": {"is_checked": True, "available_duration_sec": 1.0,
                         "effective_duration_sec": 1.0}}


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
