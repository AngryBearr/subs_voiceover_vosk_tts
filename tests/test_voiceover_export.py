from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import soundfile as sf

from tests.phase4_fixtures import ORIGINAL, RUNG_TEXTS, add_manual_row, build_world
from tests.test_voiceover_render import fake_stretch
from voiceover.export import export_final, forced_tempo, srt_text, srt_time
from voiceover.overrides import Overrides
from voiceover.resolve import resolve


class StretchSpy:
    def __init__(self) -> None:
        self.calls: dict[str, float] = {}

    def __call__(self, path: Path, tempo: float, sr: int) -> Any:
        self.calls[path.name] = tempo
        return fake_stretch(path, tempo, sr)


def _export(tmp_path: Path, world: SimpleNamespace, spy: StretchSpy | None = None,
            overrides: Overrides | None = None) -> SimpleNamespace:
    resolution = resolve(world.units, world.rows0, world.rows1, world.verdicts, world.ledger, world.placements0,
                         world.policy, world.oracle, world.cues, world.fit_slack_sec, overrides)
    schedule1 = {"passes": resolution.passes, "choices": resolution.choices, "placements": resolution.placements}
    out = tmp_path / "export"
    summary = export_final(out, cues=world.cues, units=world.units, durations0={"rows": world.rows0},
                           durations1={"rows": world.rows1}, schedule1=schedule1,
                           verified={"verdicts": world.verdicts}, policy=world.policy,
                           manifest_extra={"engine_identity": {"name": "fake"}}, stretch=spy or fake_stretch,
                           accepted=overrides.accepted if overrides else frozenset())
    return SimpleNamespace(out=out, summary=summary, resolution=resolution)


def _ledger(result: SimpleNamespace) -> dict[str, dict[str, Any]]:
    return {item["unit_id"]: item for item in json.loads((result.out / "ledger.json").read_text(encoding="utf-8"))}


def test_srt_time_and_blocks() -> None:
    assert srt_time(3_723_456) == "01:02:03,456"
    assert srt_text([(0, 1500, "Раз"), (2000, 2500, "Два")]) == (
        "1\n00:00:00,000 --> 00:00:01,500\nРаз\n\n2\n00:00:02,000 --> 00:00:02,500\nДва\n")


def test_srt_files_for_a_changed_unit(tmp_path: Path) -> None:
    world = build_world(tmp_path, {2: 4.2})
    result = _export(tmp_path, world)
    placement = next(p for p in result.resolution.placements if p["unit_id"] == "unit-0")
    spoken = (result.out / "spoken.srt").read_text(encoding="utf-8")
    blocks = [b.split("\n") for b in spoken.strip().split("\n\n")]
    assert [b[0] for b in blocks] == ["1", "2", "3"]
    assert blocks[0][1] == f"00:00:00,000 --> {srt_time(placement['end_ms'])}"
    assert all(re.fullmatch(r"\d\d:\d\d:\d\d,\d{3} --> \d\d:\d\d:\d\d,\d{3}", b[1]) for b in blocks)
    assert blocks[0][2] == RUNG_TEXTS[2] and blocks[1][2] == "Да, конечно."
    display = [b.split("\n") for b in (result.out / "display.srt").read_text(encoding="utf-8").strip().split("\n\n")]
    assert display[0][1] == "00:00:00,000 --> 00:00:02,000" and display[0][2] == RUNG_TEXTS[2]
    assert display[1][1] == "00:00:03,000 --> 00:00:05,000" and display[1][2] == "Да, конечно."


def test_display_srt_of_a_merged_unit_spans_its_member_cues(tmp_path: Path) -> None:
    world = build_world(tmp_path, {2: 4.2})
    world.cues.append({"cue_id": "cue-3", "position": 3, "index": 4, "start": 2100, "end": 2900, "text": "хвост"})
    world.units[0].update(cue_ids=["cue-0", "cue-3"], merged=True)
    world.rows0[0].update(cue_ids=["cue-0", "cue-3"], merged=True)
    result = _export(tmp_path, world)
    display = [b.split("\n") for b in (result.out / "display.srt").read_text(encoding="utf-8").strip().split("\n\n")]
    assert len(display) == 3  # the two member cues became one entry
    assert display[0][1] == "00:00:00,000 --> 00:00:02,900" and display[0][2] == RUNG_TEXTS[2]
    assert [b[0] for b in display] == ["1", "2", "3"]


def test_forced_tempo_makes_the_clip_end_at_the_placement_end(tmp_path: Path) -> None:
    world = build_world(tmp_path, {1: 4.0, 2: 3.0}, ok={1: False, 2: False})  # nothing usable: deficit stays
    spy = StretchSpy()
    result = _export(tmp_path, world, spy)
    placement = next(p for p in result.resolution.placements if p["unit_id"] == "unit-0")
    assert placement["removed_sec"] > 0.5
    item = _ledger(result)["unit-0"]
    factor = item["final"]["forced_tempo_factor"]
    assert factor is not None and item["status"] == "rejected" and not item["text_changed"]
    assert 5.0 / factor == pytest.approx((placement["end_ms"] - placement["start_ms"]) / 1000.0, abs=1e-3)
    assert item["final"]["total_speed"] == pytest.approx(factor * world.policy.base_rate, abs=1e-3)
    assert spy.calls["u0.wav"] == pytest.approx(factor, abs=1e-6)
    assert forced_tempo({**placement, "removed_sec": 0.0}) is None
    assert result.summary["counts"]["forced"] == 1


def test_ledger_statuses_and_chosen_rung_details(tmp_path: Path) -> None:
    result = _export(tmp_path, build_world(tmp_path, {1: 4.6, 2: 4.2, 3: 3.0}))
    item = _ledger(result)["unit-0"]
    assert item["status"] == "resolved" and item["chosen_rung"] == 2 and item["text_changed"]
    assert item["original_text"] == ORIGINAL and item["final_text"] == RUNG_TEXTS[2]
    assert item["verify"]["core_lost"] == []
    assert item["final"]["removed_sec"] <= 0.05  # a dead-zone sliver: tempo is nudged, nothing is queued
    assert item["final_speech_sec"] == 4.2 and item["original_speech_sec"] == 5.0
    assert set(_ledger(result)) == {"unit-0"}  # untouched, un-forced units are not listed


@pytest.mark.parametrize("name,kwargs,reason", [
    ("unresolved", {"rung_sec": {1: 5.0, 2: 4.9}, "status": "unresolved"}, "не влезло"),
    ("rejected", {"rung_sec": {1: 4.0}, "ok": {1: False}}, "отклонено проверкой смысла"),
    ("lossy", {"rung_sec": {5: 1.0}}, "потеря смысла"),
])
def test_human_queue_rows(tmp_path: Path, name: str, kwargs: dict[str, Any], reason: str) -> None:
    rung_sec = kwargs.pop("rung_sec")
    result = _export(tmp_path, build_world(tmp_path, rung_sec, **kwargs))
    queue = (result.out / "human_queue.md").read_text(encoding="utf-8")
    assert reason in queue and "unit-0" in queue
    assert "unit-1" not in queue
    assert '"unit-0": {"accept": true}' in queue and "Заготовка overrides.json" in queue
    assert "**Пробовали:**" in queue and all(f"«{RUNG_TEXTS[n]}»" in queue for n in rung_sec)
    assert result.summary["human_queue"] == 1


def test_forced_unit_without_candidates_is_queued_and_listed(tmp_path: Path) -> None:
    world = build_world(tmp_path, {}, status="timing_limited")
    world.ledger[0]["rungs"] = []
    result = _export(tmp_path, world)
    queue = (result.out / "human_queue.md").read_text(encoding="utf-8")
    assert "ускорено сверх нормы" in queue and "ускорена в" in queue and "оригинал без изменений" in queue
    assert _ledger(result)["unit-0"]["status"] == "timing_limited"


def test_accepted_forced_unit_without_candidates_leaves_the_queue(tmp_path: Path) -> None:
    world = build_world(tmp_path, {}, status="timing_limited")
    world.ledger[0]["rungs"] = []
    result = _export(tmp_path, world, overrides=Overrides(accepted=frozenset({"unit-0"})))
    assert "Пусто." in (result.out / "human_queue.md").read_text(encoding="utf-8")
    item = _ledger(result)["unit-0"]
    assert item["reviewed"] is True and item["status"] == "timing_limited"
    assert result.summary["counts"]["reviewed"] == 1 and result.summary["human_queue"] == 0


def test_clean_run_has_an_empty_queue(tmp_path: Path) -> None:
    result = _export(tmp_path, build_world(tmp_path, {2: 4.2}))
    assert "Пусто." in (result.out / "human_queue.md").read_text(encoding="utf-8")


def test_manifest_counts_and_track_length(tmp_path: Path) -> None:
    result = _export(tmp_path, build_world(tmp_path, {1: 4.6, 2: 4.2}))
    manifest = json.loads((result.out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["engine_identity"] == {"name": "fake"}
    assert manifest["counts"] == {"units": 3, "changed": 1, "resolved": 1, "unresolved": 0, "rejected": 0,
                                  "timing_limited": 0, "manual": 0, "forced": 0, "forced_tempo_applied": 1,
                                  "lossy": 0, "collisions": 0, "reviewed": 0}
    last_end = max(p["end_ms"] for p in result.resolution.placements)
    info = sf.info(str(result.out / "track.wav"))
    assert info.frames / info.samplerate == pytest.approx(last_end / 1000 + 0.5, abs=0.01)
    assert manifest["track"]["sample_rate"] == 22050 and info.samplerate == 22050
    assert (result.out / "track_guide.md").exists()


def test_export_can_run_twice_into_the_same_directory(tmp_path: Path) -> None:
    world = build_world(tmp_path, {2: 4.2})
    _export(tmp_path, world)
    again = _export(tmp_path, world)
    assert again.summary["files"] and (again.out / "spoken.srt").exists()


def test_reviewed_unit_is_left_out_of_the_queue_but_stays_in_the_ledger(tmp_path: Path) -> None:
    world = build_world(tmp_path, {1: 5.0, 2: 4.9}, status="unresolved")
    result = _export(tmp_path, world, overrides=Overrides(accepted=frozenset({"unit-0"})))
    assert "Пусто." in (result.out / "human_queue.md").read_text(encoding="utf-8")
    item = _ledger(result)["unit-0"]
    assert item["reviewed"] is True and item["status"] == "unresolved"
    assert result.summary["human_queue"] == 0 and result.summary["counts"]["reviewed"] == 1


def test_manual_text_that_fits_is_spoken_and_not_queued(tmp_path: Path) -> None:
    world = build_world(tmp_path, {1: 5.0}, status="unresolved")
    add_manual_row(world, tmp_path, "Но она думает иначе.", 3.4)
    result = _export(tmp_path, world, overrides=Overrides(texts={"unit-0": "Но она думает иначе."}))
    assert "Пусто." in (result.out / "human_queue.md").read_text(encoding="utf-8")
    item = _ledger(result)["unit-0"]
    assert item["status"] == "manual" and item["chosen_rung"] == "manual" and item["final_text"] == "Но она думает иначе."
    assert item["verify"] is None and item["text_changed"]
    assert "Но она думает иначе." in (result.out / "spoken.srt").read_text(encoding="utf-8")
    assert "Но она думает иначе." in (result.out / "track_guide.md").read_text(encoding="utf-8")
    counts = result.summary["counts"]
    assert counts["manual"] == 1 and counts["reviewed"] == 0 and counts["resolved"] == 0


def test_manual_text_that_still_needs_forced_tempo_is_queued(tmp_path: Path) -> None:
    world = build_world(tmp_path, {})
    add_manual_row(world, tmp_path, "Очень длинный ручной текст.", 5.5)
    spy = StretchSpy()
    result = _export(tmp_path, world, spy, Overrides(texts={"unit-0": "Очень длинный ручной текст."}))
    queue = (result.out / "human_queue.md").read_text(encoding="utf-8")
    assert "ручной текст не влезает" in queue and "unit-0" in queue and "ускорено сверх нормы" not in queue
    assert _ledger(result)["unit-0"]["final"]["forced_tempo_factor"] is not None
    assert result.summary["human_queue"] == 1
