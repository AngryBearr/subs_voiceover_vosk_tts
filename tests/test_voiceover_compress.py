from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.fakes import FakeTransport, make_client
from voiceover.compress import (Annotation, CompressionPolicy, Fragment, SceneUnit, UnitJob, annotate_scenes,
                                build_jobs, compress_unit, hard_checks, number_keys, parse_annotation,
                                run_compression, summarize)
from voiceover.oracle import Oracle

ORACLE = Oracle("x", {"intercept": 0.1, "syllables": 0.2, "pauses": 0.0}, 50, 0.1, 0.5, 0.1, 5.0)
POLICY = CompressionPolicy()
ORIGINAL = "Честно говоря, они не дали нам две минуты, ребята."  # 18 syllables -> 3.7 s with ORACLE
SCENE = [SceneUnit("unit-1", "Что там было?"), SceneUnit("unit-2", ORIGINAL), SceneUnit("unit-3", "Совсем мало.")]


def job(text: str = ORIGINAL, band: str = "medium", removed: float = 1.7, syllables_now: int = 18,
        target: int = 11) -> UnitJob:
    return UnitJob("unit-2", ("cue-1",), "scene-0", band, text, 3.7, removed, 1.0, syllables_now, target)


def annotation() -> Annotation:
    return Annotation("unit-2", [
        Fragment("u1", "Честно говоря,", "decor"), Fragment("u2", "они не дали нам две минуты,", "core"),
        Fragment("u3", "ребята.", "address")], {"они": "продюсеры"}, "statement")


def rungs(*texts: str, dropped: dict[int, list[str]] | None = None) -> dict[str, Any]:
    items = []
    previous = ORIGINAL
    for number in range(1, 6):
        text = texts[number - 1] if number <= len(texts) else previous
        items.append({"rung": number, "text": text, "kept": ["u2"], "dropped": (dropped or {}).get(number, []),
                      "compressed": [], "note": "n"})
        previous = text
    return {"rungs": items}


def run_unit(tmp_path: Path, content: dict[str, Any], the_job: UnitJob | None = None):
    transport = FakeTransport(lambda body: content)
    entry = compress_unit(make_client(tmp_path, transport), the_job or job(), annotation(), SCENE, ORACLE, POLICY)
    return entry, transport


# ------------------------------------------------------------ annotation

def test_parse_annotation_accepts_verbatim_fragments() -> None:
    raw = {"units": [{"id": "u1", "text": "Честно говоря,", "role": "decor"},
                     {"id": "u2", "text": "они не дали нам две минуты, ребята.", "role": "core"}],
           "refs": {"они": "продюсеры"}, "act": "statement"}
    result = parse_annotation("unit-2", ORIGINAL, raw)
    assert [f.role for f in result.fragments] == ["decor", "core"]
    assert result.warnings == [] and result.refs == {"они": "продюсеры"}


def test_parse_annotation_warns_on_low_coverage_and_drops_non_substrings() -> None:
    raw = {"units": [{"id": "u1", "text": "они не дали", "role": "core"},
                     {"id": "u2", "text": "выдуманный текст", "role": "support"}]}
    result = parse_annotation("unit-2", ORIGINAL, raw)
    assert [f.id for f in result.fragments] == ["u1"]
    assert any("not a substring" in w for w in result.warnings)
    assert any("cover 3 of 9 words" in w for w in result.warnings)


def test_missing_annotation_becomes_one_core_unit() -> None:
    result = parse_annotation("unit-2", ORIGINAL, None)
    assert result.fragments == [Fragment("u1", ORIGINAL, "core")]
    assert "annotation_missing" in result.warnings


def test_annotate_scenes_one_call_per_scene_and_missing_unit(tmp_path: Path) -> None:
    other = UnitJob("unit-9", ("cue-9",), "scene-1", "easy", "Пока.", 1.0, 0.1, 1.0, 2, 1)
    scenes = {"scene-0": SCENE, "scene-1": [SceneUnit("unit-9", "Пока.")], "scene-2": [SceneUnit("unit-20", "Да.")]}
    prompts: list[str] = []

    def reply(body: dict[str, Any]) -> dict[str, Any]:
        prompts.append(body["messages"][1]["content"])
        return {"units": {"unit-2": {"units": [{"id": "u1", "text": ORIGINAL, "role": "core"}], "refs": {},
                                     "act": "statement"}}}

    transport = FakeTransport(reply)
    result = annotate_scenes(make_client(tmp_path, transport), scenes, [job(), other], POLICY)
    assert transport.calls == 2  # scene-2 has nothing to shorten
    # scenes are annotated in parallel, so find the scene-0 prompt rather than trusting the order
    scene0 = next(p for p in prompts if "[unit-2]" in p)
    assert "* [unit-2] " + ORIGINAL in scene0 and "  [unit-1] Что там было?" in scene0
    assert result["unit-2"].warnings == []
    assert result["unit-9"].warnings == ["annotation_missing"]


# ------------------------------------------------------------ scoring

def test_first_fitting_rung_is_chosen_and_repeats_are_skipped(tmp_path: Path) -> None:
    content = rungs("Они не дали нам две минуты.",       # 10 syl -> 2.1 s, does not fit 2.0
                    "Они не дали две минуты.",            # 9 syl -> 1.9 s, fits
                    "Они не дали две минуты.",            # repeat
                    "Они не дали две минуты.", "Не дали минуты.")
    entry, transport = run_unit(tmp_path, content)
    assert transport.calls == 1
    assert entry.chosen_rung == 2 and entry.status == "resolved" and not entry.lossy
    assert entry.chosen_text == "Они не дали две минуты."
    assert [r.fits for r in entry.rungs[:2]] == [False, True]
    assert entry.rungs[2].checks == ["repeat_of_previous"]
    assert entry.chosen_estimated_sec == pytest.approx(1.9)
    assert entry.target_base_sec == pytest.approx(2.0)
    assert entry.annotation is not None and entry.original_syllables == 18


def test_rung_five_is_lossy(tmp_path: Path) -> None:
    content = rungs("Они не дали нам две минуты.", dropped={}) | {}
    content["rungs"][4]["text"] = "Не дали две минуты."
    entry, _ = run_unit(tmp_path, content)
    assert entry.chosen_rung == 5 and entry.lossy and entry.status == "resolved"


def test_unresolved_when_nothing_fits_picks_the_smallest_estimate(tmp_path: Path) -> None:
    content = rungs("Они не дали нам две минуты, ребята.", "Они не дали нам две минуты.",
                    "Они не дали две минуты.")
    entry, _ = run_unit(tmp_path, content, job(removed=3.0))  # target 0.7 s: nothing fits
    assert entry.status == "unresolved" and entry.chosen_rung == 3
    assert entry.chosen_text == "Они не дали две минуты."
    assert not entry.rungs[0].fits and not entry.rungs[2].fits


def test_rejected_when_nothing_passes_the_checks(tmp_path: Path) -> None:
    content = rungs(*(["Они дали нам две минуты."] * 5))  # drops "не" without declaring it
    entry, _ = run_unit(tmp_path, content)
    assert entry.status == "rejected" and entry.chosen_rung is None
    assert entry.chosen_text == ORIGINAL and entry.chosen_estimated_sec is None
    assert entry.rungs[0].checks == ["negation_lost"]


def test_negation_inside_a_dropped_unit_is_allowed(tmp_path: Path) -> None:
    content = rungs("Они дали нам две минуты.", dropped={1: ["u2"]})
    entry, _ = run_unit(tmp_path, content, job(removed=1.0))
    assert entry.rungs[0].checks == [] and entry.chosen_rung == 1


def test_timing_limited_makes_no_call(tmp_path: Path) -> None:
    def forbidden(body: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("the model must not be called")

    entry = compress_unit(make_client(tmp_path, forbidden), job(band="timing_limited"), None, SCENE, ORACLE, POLICY)
    assert entry.status == "timing_limited" and entry.rungs == [] and entry.chosen_text == ORIGINAL
    assert entry.chosen_rung is None


def test_invalid_answer_structure_is_rejected_not_fatal(tmp_path: Path) -> None:
    entry, _ = run_unit(tmp_path, {"unexpected": 1})
    assert entry.status == "rejected" and [r.checks for r in entry.rungs] == [["missing"]] * 5


def test_run_compression_annotates_then_compresses(tmp_path: Path) -> None:
    def reply(body: dict[str, Any]) -> dict[str, Any]:
        if "Верни JSON такого вида" in body["messages"][1]["content"]:
            return {"units": {}}
        return rungs("Они не дали нам две минуты, ребята.", "Они не дали две минуты.")

    transport = FakeTransport(reply)
    entries = run_compression(make_client(tmp_path, transport), [job()], {"scene-0": SCENE}, ORACLE, POLICY)
    assert transport.calls == 2
    assert entries[0].annotation is not None and "annotation_missing" in entries[0].annotation["warnings"]
    assert entries[0].chosen_rung == 2


# ------------------------------------------------------------ hard checks

@pytest.mark.parametrize("original, candidate, scene, dropped, expected", [
    ("Нас было 12 человек в лодке.", "Нас было человек в лодке.", "", "", "number_lost"),
    ("Нас было двенадцать человек в лодке.", "Нас было много человек в лодке.", "", "", "number_lost"),
    ("Это было в третьем сезоне.", "Это было в сезоне.", "", "", "number_lost"),
    ("Он сказал, что Джон уехал давно.", "Он сказал, что Джон уехал.", "Он сказал, что Джон уехал давно.", "", None),
    ("Он сказал, что Мэтт уехал давно.", "Он сказал, что Билл уехал.", "Он сказал, что Мэтт уехал давно.", "",
     "new_name"),
    ("Он не хочет ехать туда сегодня.", "Он хочет ехать туда.", "", "", "negation_lost"),
    ("Он не хочет ехать туда сегодня.", "Он хочет ехать туда.", "", "не хочет", None),
    ("Он поехал туда сегодня.", "Он поехал туда сегодня же.", "", "", "not_shorter"),
    ("Он поехал туда сегодня.", "Он поехал туда сегодня.", "", "", "unchanged"),
    ("Он поехал туда сегодня.", "Он поехал.\nТуда.", "", "", "multi_line"),
    ("Он поехал туда сегодня.", "  ", "", "", "empty"),
])
def test_hard_checks(original: str, candidate: str, scene: str, dropped: str, expected: str | None) -> None:
    reasons = hard_checks(original, candidate, scene or original, dropped)
    if expected is None:
        assert reasons == []
    else:
        assert expected in reasons


def test_number_keys_handle_digits_cardinals_and_ordinal_endings() -> None:
    assert number_keys("В 3 раза, пять, третьего") == {"d:3", "c:пять", "o:трет"}
    assert number_keys("третий и третьей") == {"o:трет"}
    assert number_keys("четвёртого, пятое") == {"o:четвёрт", "o:пят"}
    assert number_keys("пять девять") == {"c:пять", "c:девять"}


def test_sentence_initial_capital_is_not_a_new_name() -> None:
    assert hard_checks("Он ушёл вчера вечером домой.", "Тогда ушёл домой.", "Он ушёл вчера вечером домой.", "") == []


# ------------------------------------------------------------ jobs and summary

def test_build_jobs_selects_bands_and_orders_by_episode() -> None:
    units = [{"unit_id": f"unit-{i}", "scene_id": "scene-0", "text": f"t{i}"} for i in range(4)]
    bands = ["none", "easy", "timing_limited", "hard"]
    placements = [{"unit_id": u["unit_id"], "cue_ids": ["cue-0"], "text_band": b, "removed_sec": 0.2,
                   "tempo_factor": 1.0, "current_syllables": 8, "target_syllables": 6} for u, b in zip(units, bands)]
    rows = [{"unit_id": u["unit_id"], "display_text": u["text"], "speech_sec": 2.0} for u in units]
    jobs, scenes = build_jobs(units, placements, rows, CompressionPolicy(bands=("easy",)))
    assert [j.unit_id for j in jobs] == ["unit-1", "unit-2"]
    assert len(scenes["scene-0"]) == 4
    assert jobs[0].target_base_sec == pytest.approx(1.8)


def test_summarize_counts() -> None:
    ledger = [
        {"status": "resolved", "chosen_rung": 2, "lossy": False, "original_syllables": 17, "chosen_syllables": 9,
         "speech_sec": 3.5, "chosen_estimated_sec": 1.9, "annotation": {"warnings": []}},
        {"status": "unresolved", "chosen_rung": 5, "lossy": True, "original_syllables": 10, "chosen_syllables": 5,
         "speech_sec": 2.0, "chosen_estimated_sec": 1.1, "annotation": {"warnings": ["w"]}},
        {"status": "timing_limited", "chosen_rung": None, "lossy": False, "original_syllables": 2,
         "chosen_syllables": 2, "speech_sec": 0.7, "chosen_estimated_sec": None, "annotation": None},
    ]
    summary = summarize(ledger)
    assert summary["resolved"] == 1 and summary["unresolved"] == 1 and summary["timing_limited"] == 1
    assert summary["sent_to_model"] == 2 and summary["lossy"] == 1
    assert summary["rung_histogram"] == {"2": 1, "5": 1}
    assert summary["syllables_removed"] == 13 and summary["estimated_sec_saved"] == 2.5
    assert summary["annotation_warnings"] == 1


def test_core_fragment_dropped_is_lossy_even_below_rung_five() -> None:
    from voiceover.compress import Annotation, Fragment, RungResult, _is_lossy
    annotation = Annotation("u", [Fragment("u1", "Теперь", "decor"), Fragment("u2", "у нас гонка", "core")], {}, "exclamation")
    rung = RungResult(4, "Гонка!", 2, 0.4, True, [], "", ["u2"], ["u1"], [])
    assert not _is_lossy(annotation, rung)
    rung_core = RungResult(4, "Теперь!", 2, 0.4, True, [], "", ["u1"], ["u2"], [])
    assert _is_lossy(annotation, rung_core)


def test_negative_pronouns_and_odin_are_not_hard_checked() -> None:
    from voiceover.compress import hard_checks
    assert hard_checks("Я ничего не ела четыре дня.", "Я не ела четыре дня.", "", "") == []
    assert hard_checks("Ещё один кусочек у Вавау!", "Ещё кусочек у Вавау!", "Вавау", "") == []
    assert "negation_lost" in hard_checks("Я не ела четыре дня.", "Я ела четыре дня.", "", "")
