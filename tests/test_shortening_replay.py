from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from utils.llm_types import CallRecord, DetailedResponse
from utils.shortening_experiment import _canonical
from utils.shortening_plan import PlannerConfig, plan_subtitles
from utils.shortening_replay import _old_cost_summary, replay_experiment
from utils.shortening_runner import ModelSpec


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _fixture(tmp_path: Path, *, best_of: int = 5, repetitions: int = 2) -> tuple[Path, dict[str, Any], Path]:
    items = [{"index": index, "start": index * 1200, "end": index * 1200 + 600,
              "text": f"A deliberately long immutable subtitle cue number {index}."}
             for index in range(6)]
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "sources").mkdir(parents=True)
    (corpus_dir / "sources" / "s.json").write_text(json.dumps(items), encoding="utf-8")
    plans = plan_subtitles(items, PlannerConfig())
    cases = []
    for plan in plans:
        cases.append({"case_id": f"s:{plan.cue_id}", "source_id": "s", "cue_id": plan.cue_id,
                      "position": plan.position, "source_index": plan.position, "original": plan.text,
                      "max_chars": plan.max_chars, "available_ms": plan.available_ms, "severity": plan.severity,
                      "context": [{"cue_id": f"cue-{i}", "text": item["text"], "start": item["start"], "end": item["end"]}
                                  for i, item in enumerate(items[max(0, plan.position - 3):min(len(items), plan.position + 4)], max(0, plan.position - 3))]})
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "planner_config": {"cps": 13.0, "max_speed": 1.5, "gap_ms": 200, "context_before": 3, "context_after": 3},
        "sources": [{"source_id": "s", "json_path": "sources/s.json", "split": "test",
                     "parsed_sha256": _canonical(items)}],
        "cases": cases,
        "audit_case_ids": [cases[0]["case_id"]],
        "crosscheck_case_ids": [cases[0]["case_id"]],
        "order_check_case_ids": [cases[0]["case_id"]],
    }
    old_root = tmp_path / "old"
    (old_root / "r1" / "best5" / "s").mkdir(parents=True)
    if repetitions == 2:
        (old_root / "r2" / "best5" / "s").mkdir(parents=True)
    old_manifest = {"corpus_sha256": _canonical(manifest), "sample_case_ids": [case["case_id"] for case in cases],
                    "params": {"repetitions": repetitions, "best_of": best_of},
                    "prompt_versions": {"standard": "shortening-v1", "candidates": "candidate-v1"}}
    (old_root / "experiment_manifest.json").write_text(json.dumps(old_manifest), encoding="utf-8")
    old_rows: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        results: list[dict[str, Any]] = []
        for case in cases:
            a_text = f"A-frozen-{repetition}-{case['cue_id']}"
            b_text = f"B-frozen-{repetition}-{case['cue_id']}"
            results.append({"global_case_id": case["case_id"], "local_cue_id": case["cue_id"], "source_id": "s",
                            "repetition": repetition, "original": case["original"], "context": case["context"],
                            "A": {"text": a_text, "status": "unreviewed", "candidate_hash": _sha_text(a_text)},
                            "B": {"text": b_text, "status": "accepted", "candidate_hash": _sha_text(b_text)},
                            "C": {"text": None, "status": "unavailable", "candidates": []}})
        old_rows.extend(results)
        if best_of == 5:
            payload_rows = []
            for case in cases[:5]:
                payload_rows.append({"cue_id": case["cue_id"], "status": "ok",
                                     "candidates": [{"candidate_id": "old", "text": f"short-{case['cue_id']}"}]})
            payload_rows.append({"cue_id": cases[5]["cue_id"], "status": "ok", "candidates": "malformed"})
            payload = json.dumps({"results": payload_rows})
            request_path = old_root / f"r{repetition}" / "best5" / "s" / "requests.jsonl"
            request_path.write_text(json.dumps({"request_id": f"old-{repetition}", "stage": "generation_candidates", "state": "start"})
                                    + "\n" + json.dumps({"request_id": f"old-{repetition}", "stage": "generation_candidates", "state": "complete", "text": payload, "error": None}) + "\n",
                                    encoding="utf-8")
    (old_root / "case_results.json").write_text(json.dumps(old_rows), encoding="utf-8")
    return old_root, manifest, corpus_dir


def _fake_request(calls: list[str], *, unknown: bool = False):
    async def request(_: ModelSpec, __: str, user: str, schema: dict[str, Any], ___: bool) -> DetailedResponse:
        properties = schema["properties"]["results"]["items"]["properties"]
        payload = json.loads(user.split("Input JSON:\n", 1)[1])
        if "selected_candidate_id" in properties:
            calls.append("selection")
            rows = [{"cue_id": target["cue_id"], "selected_candidate_id": "forbidden" if unknown else target["candidates"][0]["candidate_id"],
                     "issues": [], "advice": ""} for target in payload["targets"]]
            return DetailedResponse(json.dumps({"results": rows}), [])
        calls.append("audit")
        rows = []
        for target in payload["targets"]:
            options = target["options"]
            rows.append({"case_id": target["case_id"],
                         "judgments": [{"option_id": option["option_id"], "verdict": "accept", "issues": []} for option in options],
                         "preferred_option_id": options[0]["option_id"] if options else None})
        return DetailedResponse(json.dumps({"results": rows}), [])
    return request


def test_old_cost_attribution_filters_generation_stages_without_dropping_legacy_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    standard = tmp_path / "old" / "r1" / "standard" / "s" / "calls.jsonl"
    best5 = tmp_path / "old" / "r1" / "best5" / "s" / "calls.jsonl"
    standard.parent.mkdir(parents=True)
    best5.parent.mkdir(parents=True)

    def row(stage: str, model: str) -> dict[str, Any]:
        record = asdict(CallRecord("fixture", model, None, "now", 0.0, None, "ok"))
        return {"stage": stage, "record": record}

    standard.write_text(json.dumps(row("generation", "a")) + "\n" + json.dumps(row("review", "b")), encoding="utf-8")
    best5.write_text(json.dumps(row("generation_candidates", "c")) + "\n" + json.dumps(row("selection", "d")), encoding="utf-8")
    costs = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    monkeypatch.setattr("utils.shortening_replay.estimate_record_cost", lambda record: {"known_cost_usd": costs[record.model]})

    old_a = _old_cost_summary([standard], tmp_path / "old", stage="generation")
    old_b = _old_cost_summary([standard], tmp_path / "old")
    old_c = _old_cost_summary([best5], tmp_path / "old", stage="generation_candidates")
    old_selector = _old_cost_summary([best5], tmp_path / "old", stage="selection")

    assert old_a["known_subtotal_usd"] == 1.0
    assert old_b["known_subtotal_usd"] == 3.0
    assert old_c["known_subtotal_usd"] == 3.0
    assert old_selector["known_subtotal_usd"] == 4.0
    assert old_c["stage_filter"] == "generation_candidates"
    assert old_c["source_paths"] == ["r1/best5/s/calls.jsonl"]


def test_old_cost_summary_counts_unparseable_rows_without_using_fallback_cost(tmp_path: Path) -> None:
    calls = tmp_path / "calls.jsonl"
    calls.write_text(json.dumps({"stage": "generation", "record": {"invalid": True},
                                 "cost_estimate": {"known_cost_usd": 99.0}}) + "\n"
                      + json.dumps({"stage": "generation", "cost_estimate": {"known_cost_usd": 50.0}}) + "\n"
                      + "{malformed receipt",
                      encoding="utf-8")

    summary = _old_cost_summary([calls], tmp_path)

    assert summary["physical_calls"] == 3
    assert summary["recorded_calls"] == 0
    assert summary["raw_record_count"] == 3
    assert summary["unknown_call_count"] == 3
    assert summary["error_count"] == 3
    assert summary["known_subtotal_usd"] == 0.0


def test_replay_reuses_invalid_batch_partially_without_generation_and_preserves_repetitions(tmp_path: Path) -> None:
    old_root, manifest, corpus_dir = _fixture(tmp_path)
    calls: list[str] = []
    output = tmp_path / "replayed"
    result = asyncio.run(replay_experiment(old_root, manifest, corpus_dir, output, request_fn=_fake_request(calls)))
    assert result["status"] == "completed"
    assert calls.count("selection") == 2
    assert calls.count("audit") == 6
    assert result["replay_generation_requests"] == 0
    rows = json.loads((output / "case_results.json").read_text(encoding="utf-8"))
    assert {row["repetition"] for row in rows} == {1, 2}
    assert all(row["A"]["text"].startswith("A-frozen") for row in rows)
    assert all(row["B"]["text"].startswith("B-frozen") for row in rows)
    assert sum(row["C"]["status"] == "accepted" for row in rows) == 10
    assert (output / "r1" / "best5" / "s" / "subtitles.srt").exists()
    assert result["cost"]["projection_500"]["denominator"] == 6


def test_replay_unknown_selection_id_fails_closed_and_hash_mismatch_precedes_calls(tmp_path: Path) -> None:
    old_root, manifest, corpus_dir = _fixture(tmp_path)
    calls: list[str] = []
    result = asyncio.run(replay_experiment(old_root, manifest, corpus_dir, tmp_path / "unknown",
                                           request_fn=_fake_request(calls, unknown=True)))
    rows = json.loads((tmp_path / "unknown" / "case_results.json").read_text(encoding="utf-8"))
    assert all(row["C"]["status"] == "error" for row in rows if row["local_cue_id"] != "cue-5")
    assert result["replay_generation_requests"] == 0
    bad_manifest = json.loads(json.dumps(manifest))
    bad_manifest["replay_only_change"] = True
    with pytest.raises(ValueError, match="corpus canonical hash mismatch"):
        asyncio.run(replay_experiment(old_root, bad_manifest, corpus_dir, tmp_path / "bad", request_fn=_fake_request([])))


def test_replay_best_of_zero_skips_selection_and_refuses_overwrite(tmp_path: Path) -> None:
    old_root, manifest, corpus_dir = _fixture(tmp_path, best_of=0, repetitions=1)
    calls: list[str] = []
    output = tmp_path / "ab-only"
    asyncio.run(replay_experiment(old_root, manifest, corpus_dir, output, request_fn=_fake_request(calls)))
    assert "selection" not in calls
    assert all(row["C"]["reason"] == "best_of_disabled" for row in json.loads((output / "case_results.json").read_text(encoding="utf-8")))
    with pytest.raises(FileExistsError):
        asyncio.run(replay_experiment(old_root, manifest, corpus_dir, output, request_fn=_fake_request([])))
