from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.shorten_subtitles import _summary_payload, main


def test_cli_dry_run_does_not_contact_models(tmp_path: Path) -> None:
    output = tmp_path / "dry"
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{"index": 7, "start": 0, "end": 1000, "text": ["one two three"]}]), encoding="utf-8")
    assert main([str(source), "--output-dir", str(output)]) == 0
    assert (output / "manifest.json").exists()
    assert (output / "plan.json").exists()
    assert not (output / "report.json").exists()


def test_cli_manifest_records_deepseek_limits_and_requested_legacy_model(tmp_path: Path) -> None:
    output = tmp_path / "dry"
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{"index": 7, "start": 0, "end": 1000, "text": ["one two three"]}]), encoding="utf-8")
    assert main([
        str(source), "--output-dir", str(output), "--max-tokens", "123457",
        "--deepseek-model", "deepseek-v4-flash",
    ]) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["deepseek_limits"]["requested_override"] == 123457
    assert manifest["deepseek_limits"]["effective_by_mode"]["high"] == 123457
    assert manifest["deepseek_limits"]["project_conservative_ceiling"] == 384000
    assert manifest["deepseek_limits"]["provider_advertised_cap_label"] == "384K"
    assert "exact provider maximum is not independently verified" in manifest["deepseek_limits"]["project_conservative_ceiling_note"]
    assert manifest["deepseek_limits"]["model_requested"] == "deepseek-v4-flash"
    assert manifest["deepseek_limits"]["provider_version_attested"] is False
    assert {entry["model"] for entry in manifest["generators"] if entry["provider"] == "deepseek"} == {"deepseek-v4-flash"}


def test_cli_preflight_does_not_create_output_and_keeps_runtime_sibling(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{"index": 7, "start": 0, "end": 1000, "text": ["one two three"]}]), encoding="utf-8")
    output = tmp_path / "run"
    sibling = tmp_path / "run.runtime"
    sibling.mkdir()
    marker = sibling / "keep"
    marker.write_text("keep", encoding="utf-8")
    assert main([str(source), "--output-dir", str(output), "--execute", "--repetitions", "0"]) == 1
    assert not output.exists()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_cli_empty_selection_completes_without_server(tmp_path: Path) -> None:
    output = tmp_path / "empty"
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{"index": 7, "start": 0, "end": 1000, "text": ["short"]}]), encoding="utf-8")
    assert main([str(source), "--output-dir", str(output), "--execute"]) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["completed_trials"] == 0
    assert (output / "experiment_summary.json").exists()


def test_summary_payload_aggregates_partial_known_subtotals(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("[]", encoding="utf-8")
    args = argparse.Namespace(input=source, repetitions=2)

    def report(known_subtotal: float | None) -> dict[str, object]:
        return {
            "model": {"provider": "deepseek", "variant": "off"},
            "summary": {"accepted_count": 1, "selected_count": 1, "manual_fraction": 0.0},
            "cost": {
                "direct": {"known_subtotal_usd": known_subtotal, "accounting_complete": False},
                "projection_500": {},
            },
        }

    summary = _summary_payload(args, [], [report(None), report(0.01)], status="complete")
    assert summary["experiment_spent_known_subtotal_usd"] == 0.01
    assert summary["experiment_accounting_complete"] is False
    assert summary["by_model_variant"]["deepseek:off"]["direct_known_cost_mean"] == 0.01

    all_unknown = _summary_payload(args, [], [report(None), report(None)], status="complete")
    assert all_unknown["by_model_variant"]["deepseek:off"]["direct_known_cost_mean"] is None
