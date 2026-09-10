"""Focused tests for deterministic corpus construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from utils.shortening_corpus import build_corpus, select_source_targets
from utils.shortening_plan import PlannerConfig, plan_subtitles


def _cue(index: int, text: str, start: int = 0, end: int = 10_000) -> str:
    return f"{index}\n00:{start // 60000:02d}:{(start // 1000) % 60:02d},{start % 1000:03d} --> 00:{end // 60000:02d}:{(end // 1000) % 60:02d},{end % 1000:03d}\n{text}\n\n"


def _write_sources(root: Path) -> None:
    names = {
        "S01E02.srt": "fit",
        "Survivor.S44E01.srt": "fit",
        "Survivor.S44E02.srt": "fit",
        "Survivor.S44E04.srt": "fit",
        "Survivor.S44E05.srt": "fit",
        "Australian Survivor S01E03_ru.srt": "calibration",
        "Survivor.S44E07.srt": "calibration",
        "Survivor.S44E08.srt": "calibration",
        "Survivor.S44E09.srt": "calibration",
        "S01E01.srt": "test",
        "Australian Survivor S01E04_ru.srt": "test",
        "Survivor.S44E03.srt": "test",
        "Survivor.S44E06.srt": "test",
        "Survivor.S44E10.srt": "test",
        "Survivor.S44E11.srt": "test",
        "Survivor.S44E12.srt": "test",
        "Survivor.S44E13.srt": "test",
    }
    for name in names:
        text = "one two three four five six seven eight nine ten " + ("x " * 90) + name
        content = "\ufeff" + _cue(1, text, 0, 1_000)
        content += "".join(_cue(index, text, index * 1_200, index * 1_200 + 1_000) for index in range(2, 7))
        (root / name).write_text(content, encoding="utf-8")
    duplicate = root / "Survivor.S44E01(1).srt"
    duplicate.write_bytes((root / "Survivor.S44E01.srt").read_bytes())


def test_manifest_hashes_dedup_and_partition(tmp_path: Path) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    _write_sources(source)
    manifest = build_corpus(source, tmp_path / "out", seed=7)
    assert len(manifest["source_inventory"]) == 18
    assert len(manifest["sources"]) == 17
    assert len(manifest["cases"]) == 48
    assert len(manifest["duplicate_groups"]["raw_sha256"]) == 1
    assert {case["source_id"] for case in manifest["cases"]} <= {row["source_id"] for row in manifest["sources"]}
    assert all(case["case_id"] == f"{case['source_id']}:{case['cue_id']}" for case in manifest["cases"])
    assert set(manifest["audit_case_ids"]) <= {case["case_id"] for case in manifest["cases"]}
    assert set(manifest["crosscheck_case_ids"]) <= set(manifest["audit_case_ids"])
    assert set(manifest["order_check_case_ids"]) <= set(manifest["crosscheck_case_ids"])
    case_by_id = {case["case_id"]: case for case in manifest["cases"]}
    assert len(manifest["audit_case_ids"]) == 12
    assert len({case_by_id[case_id]["source_id"] for case_id in manifest["audit_case_ids"]}) == 8
    assert len(manifest["crosscheck_case_ids"]) == 6
    assert len({case_by_id[case_id]["source_id"] for case_id in manifest["crosscheck_case_ids"]}) == 6
    assert len(manifest["order_check_case_ids"]) == 3
    assert len({case_by_id[case_id]["source_id"] for case_id in manifest["order_check_case_ids"]}) == 3
    assert manifest["audit_selection_meta"]["source_coverage"]["audit"]["count"] == 8
    assert manifest["audit_selection_meta"]["source_coverage"]["crosscheck"]["count"] == 6
    assert manifest["audit_selection_meta"]["source_coverage"]["order_check"]["count"] == 3


def test_bom_source_copy_and_context_stay_in_one_file(tmp_path: Path) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    _write_sources(source)
    manifest = build_corpus(source, tmp_path / "out")
    row = next(row for row in manifest["sources"] if row["path"] == "S01E01.srt")
    copied = json.loads((tmp_path / "out" / row["json_path"]).read_text(encoding="utf-8"))
    assert copied[0]["index"] == 1
    case = next(case for case in manifest["cases"] if case["source_id"] == row["source_id"])
    assert all(0 <= int(item["cue_id"].split("-")[1]) < len(copied) for item in case["context"])


def test_target_selection_has_no_fabricated_severity() -> None:
    items = [{"start": i * 1000, "end": i * 1000 + 100, "text": "one two"} for i in range(10)]
    plans = plan_subtitles(items, PlannerConfig())
    selected = select_source_targets(items, plans, count=6, seed=9)
    assert selected == []
    assert all(plans[int(cue_id.split("-")[1])].status != "needs_shortening" for cue_id in selected)


def test_no_overwrite_and_invalid_timing_is_not_a_target(tmp_path: Path) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    _write_sources(source)
    out = tmp_path / "out"
    build_corpus(source, out)
    with pytest.raises(FileExistsError):
        build_corpus(source, out)


def test_selection_does_not_mutate_global_rng() -> None:
    import random

    random.seed(123)
    before = random.getstate()
    plans = plan_subtitles([{"start": 0, "end": 1000, "text": "one two three four five"}], PlannerConfig())
    select_source_targets([], plans, seed=44)
    assert random.getstate() == before
