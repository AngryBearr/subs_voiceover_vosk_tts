from __future__ import annotations

import json
from pathlib import Path

import pytest

from voiceover.engine import trim_edges
from voiceover.pipeline import parse_input, stage_parse

import numpy as np


def test_parse_json_assigns_positional_ids(tmp_path: Path) -> None:
    source = tmp_path / "cues.json"
    source.write_text(json.dumps([
        {"index": 7, "start": 100, "end": 900, "text": ["Привет,", "мир"]},
        {"index": 3, "start": 1000, "end": 1500, "text": "Пока"},
    ]), encoding="utf-8")
    cues = parse_input(source)
    assert [c["cue_id"] for c in cues] == ["cue-0", "cue-1"]
    assert cues[0]["text"] == "Привет, мир"
    assert cues[0]["index"] == 7
    assert parse_input(source, limit=1) == cues[:1]


def test_parse_srt(tmp_path: Path) -> None:
    source = tmp_path / "a.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:02,500\nПервая\nстрока\n\n2\n00:00:03,000 --> 00:00:04,000\nВторая\n",
                      encoding="utf-8")
    cues = parse_input(source)
    assert cues[0]["start"] == 1000 and cues[0]["end"] == 2500
    assert cues[0]["text"] == "Первая строка"
    assert len(cues) == 2


def test_stage_parse_refuses_a_different_source_in_same_dir(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    first.write_text(json.dumps([{"start": 0, "end": 10, "text": "a"}]), encoding="utf-8")
    second = tmp_path / "b.json"
    second.write_text(json.dumps([{"start": 0, "end": 10, "text": "b"}]), encoding="utf-8")
    out = tmp_path / "out"
    stage_parse(first, out, None)
    stage_parse(first, out, None)  # idempotent
    with pytest.raises(ValueError):
        stage_parse(second, out, None)


def test_trim_edges_removes_leading_and_trailing_silence() -> None:
    sr = 1000
    audio = np.zeros((3000, 1), dtype=np.float32)
    audio[1000:2000, 0] = 0.5
    trimmed, removed_ms = trim_edges(audio, sr, dbfs=-40.0, pad_ms=100)
    assert trimmed.shape[0] == 1200
    assert removed_ms == 1800
    silent, removed = trim_edges(np.zeros((500, 1), dtype=np.float32), sr, dbfs=-40.0, pad_ms=10)
    assert silent.shape[0] == 500 and removed == 0


def test_text_band_classification() -> None:
    from voiceover.pipeline import text_band
    assert text_band(0.0, 10) == "none"
    assert text_band(0.1, 10) == "easy"
    assert text_band(0.35, 10) == "medium"
    assert text_band(0.5, 10) == "hard"
    assert text_band(0.7, 10) == "timing_limited"
    assert text_band(0.3, 2) == "timing_limited"


def test_fix_homoglyphs_only_touches_mixed_words() -> None:
    from voiceover.engine import fix_homoglyphs
    assert fix_homoglyphs("На Суонe?") == "На Суоне?"
    assert fix_homoglyphs("Hello мир") == "Hello мир"
    assert fix_homoglyphs("Cтоп") == "Стоп"


def test_syllables_to_remove_uses_the_line_own_rate() -> None:
    from voiceover.oracle import Oracle
    from voiceover.pipeline import syllables_to_remove
    oracle = Oracle("x", {"intercept": 0.1, "syllables": 0.2, "pauses": 0.2}, 50, 0.1, 0.5, 0.1, 5.0)
    # 10 syllables in 2.1 s -> 0.2 s each; saving 0.5 s needs 3 syllables (rounded up)
    assert syllables_to_remove(10, 2.1, 0.5, oracle) == 3
    # a slow line (0.4 s per syllable) needs fewer syllables for the same saving
    assert syllables_to_remove(10, 4.1, 0.5, oracle) == 2
    # never more than the line has, never less than one
    assert syllables_to_remove(3, 0.7, 5.0, oracle) == 3
    assert syllables_to_remove(10, 2.1, 0.001, oracle) == 1


def test_text_target_ignores_dead_zone_removals() -> None:
    from voiceover.oracle import Oracle
    from voiceover.pipeline import _text_target
    from voiceover.timeline import Placement
    oracle = Oracle("x", {"intercept": 0.1, "syllables": 0.2, "pauses": 0.2}, 50, 0.1, 0.5, 0.1, 5.0)
    row = {"tts_text": "Абсолютно.", "speech_sec": 0.77}
    tight = Placement("u", 0, 0, 0, 1.0, 1.25, 0.77, 0.76, 0.01, 0.76, 760, 0, "needs_text")
    assert _text_target(row, tight, oracle)["text_band"] == "none"
    real = Placement("u", 0, 0, 0, 1.0, 1.25, 0.77, 0.47, 0.3, 0.47, 470, 0, "needs_text")
    target = _text_target(row, real, oracle)
    assert target["target_syllables"] == 2 and target["text_band"] == "timing_limited"


def test_text_target_scales_removal_to_base_rate() -> None:
    from voiceover.oracle import Oracle
    from voiceover.pipeline import _text_target
    from voiceover.timeline import Placement
    oracle = Oracle("x", {"intercept": 0.0, "syllables": 0.2, "pauses": 0.0}, 50, 0.1, 0.5, 0.1, 5.0)
    # 10 syllables, 2.0 s at base => 0.2 s per syllable. 0.4 s removed at tempo 1.5 is 0.6 s at base => 3 syllables.
    row = {"tts_text": "Абсолютно точно, да да да да.", "speech_sec": 2.0}
    from voiceover.text import syllables
    assert syllables(row["tts_text"]) == 10
    at_tempo = Placement("u", 0, 0, 0, 1.5, 1.875, 2.0 / 1.5, 2.0 / 1.5 - 0.4, 0.4, 0.0, 0, 0, "needs_text")
    assert _text_target(row, at_tempo, oracle)["target_syllables"] == 7


def _compress_fixture() -> tuple[dict, dict, dict, "Oracle", dict]:
    from voiceover.oracle import Oracle
    original = "Честно говоря, они не дали нам две минуты, ребята."
    units = {"units": [{"unit_id": "unit-0", "cue_ids": ["cue-0"], "scene_id": "scene-0", "text": original},
                       {"unit_id": "unit-1", "cue_ids": ["cue-1"], "scene_id": "scene-0", "text": "Совсем мало."}]}
    durations = {"rows": [{"unit_id": "unit-0", "display_text": original, "speech_sec": 3.7},
                          {"unit_id": "unit-1", "display_text": "Совсем мало.", "speech_sec": 0.9}]}
    placements = [
        {"unit_id": "unit-0", "cue_ids": ["cue-0"], "text_band": "medium", "removed_sec": 1.7,
         "tempo_factor": 1.0, "current_syllables": 18, "target_syllables": 10},
        {"unit_id": "unit-1", "cue_ids": ["cue-1"], "text_band": "timing_limited", "removed_sec": 0.5,
         "tempo_factor": 1.0, "current_syllables": 4, "target_syllables": 2}]
    oracle = Oracle("x", {"intercept": 0.1, "syllables": 0.2, "pauses": 0.0}, 50, 0.1, 0.5, 0.1, 5.0)
    return units, durations, {"placements": placements}, oracle, {}


def test_stage_compress_writes_candidates_and_second_run_makes_no_calls(tmp_path: Path) -> None:
    from tests.fakes import FakeTransport, make_client
    from voiceover.compress import CompressionPolicy
    from voiceover.pipeline import read_json, stage_annotate, stage_compress
    units, durations, schedule, oracle, _ = _compress_fixture()

    def reply(body: dict) -> dict:
        if "Верни JSON такого вида" in body["messages"][1]["content"]:
            return {"units": {"unit-0": {"units": [
                {"id": "u1", "text": "Честно говоря,", "role": "decor"},
                {"id": "u2", "text": "они не дали нам две минуты, ребята.", "role": "core"}], "refs": {},
                "act": "statement"}}}
        texts = ["Они не дали нам две минуты, ребята.", "Они не дали нам две минуты."]
        return {"rungs": [{"rung": n, "text": texts[min(n, 2) - 1] if n < 3 else "Они не дали две минуты.",
                           "kept": ["u2"], "dropped": ["u1"], "compressed": [], "note": ""} for n in range(1, 6)]}

    policy = CompressionPolicy()
    out = tmp_path / "out"
    transport = FakeTransport(reply)
    client = make_client(tmp_path, transport)
    annotations = stage_annotate(units, durations, schedule, policy, client, out)
    candidates = stage_compress(units, durations, schedule, oracle, annotations, policy, client, out)
    assert transport.calls == 2  # one scene + one unit; the timing-limited unit is not sent
    assert read_json(out / "candidates.json")["schema"] == "voiceover-candidates-v1"
    assert read_json(out / "annotations.json")["schema"] == "voiceover-annotations-v1"
    by_id = {e["unit_id"]: e for e in candidates["ledger"]}
    assert by_id["unit-0"]["status"] == "resolved" and by_id["unit-0"]["chosen_rung"] == 3
    assert by_id["unit-1"]["status"] == "timing_limited"
    assert candidates["summary"]["resolved"] == 1 and candidates["llm_stats"]["calls_made"] == 1

    # same inputs: nothing is recomputed, even with a fresh client
    again = make_client(tmp_path, FakeTransport(lambda body: {}))
    annotations2 = stage_annotate(units, durations, schedule, policy, again, out)
    stage_compress(units, durations, schedule, oracle, annotations2, policy, again, out)
    assert again.stats()["calls_made"] == 0 and again.stats()["cache_hits"] == 0

    # a new prompt version invalidates the artifacts but the old cache entries stay untouched
    changed = CompressionPolicy(prompt_version="v99")
    stage_annotate(units, durations, schedule, changed, client, out)
    assert transport.calls == 3


# ---------------------------------------------------------------- phase 4

def test_stage_order_ends_with_the_phase_four_stages() -> None:
    from voiceover.pipeline import STAGES
    assert STAGES == ("parse", "units", "calibrate", "synth0", "oracle", "schedule", "annotate", "compress",
                      "measure", "verify", "resolve", "export", "report")


def _measure_inputs(tmp_path: Path) -> tuple[dict, dict]:
    from tests.phase4_fixtures import build_world
    world = build_world(tmp_path, {})
    world.ledger[0]["rungs"][1]["checks"] = ["not_shorter"]  # rung 2 failed a hard check
    world.ledger.append({**world.ledger[0], "unit_id": "unit-1", "status": "timing_limited", "rungs": []})
    return {"ledger": world.ledger}, {"rows": world.rows0}


def test_stage_measure_synthesises_only_candidate_rungs(tmp_path: Path) -> None:
    from tests.fakes import FakeEngine
    from voiceover.pipeline import read_json, stage_measure
    candidates, durations = _measure_inputs(tmp_path)
    engine = FakeEngine()
    out = tmp_path / "out"
    artifact = stage_measure(candidates, durations, engine, out, 2, tmp_path / "tts")
    assert engine.synthesized == ["unit-0#r1", "unit-0#r3", "unit-0#r4", "unit-0#r5"]
    assert artifact["schema"] == "voiceover-durations1-v1"
    assert set(artifact["inputs"]) == {"candidates_sha256", "synth_identity_sha256", "overrides_sha256"}
    assert artifact["inputs"]["overrides_sha256"] is None
    row = artifact["rows"][0]
    assert set(row) == {"unit_id", "rung", "text", "tts_text", "normalization", "normalization_error", "path",
                        "sha256", "speech_sec", "raw_sec", "syllables", "error"}
    assert row["rung"] == 1 and row["syllables"] > 0 and row["speech_sec"] > 0 and Path(row["path"]).exists()
    assert read_json(out / "durations1.json")["rows"] == artifact["rows"]

    again = FakeEngine()
    stage_measure(candidates, durations, again, out, 2, tmp_path / "tts")
    assert again.synthesized == [] and again.prepared_items == 0


def test_stage_measure_retries_only_error_rows(tmp_path: Path) -> None:
    from tests.fakes import FakeEngine
    from tests.phase4_fixtures import RUNG_TEXTS
    from voiceover.pipeline import stage_measure
    candidates, durations = _measure_inputs(tmp_path)
    out = tmp_path / "out"
    first = stage_measure(candidates, durations, FakeEngine(fail_texts={RUNG_TEXTS[3]}), out, 1, tmp_path / "tts")
    assert [r["rung"] for r in first["rows"] if r["error"]] == [3]
    healthy = FakeEngine()
    second = stage_measure(candidates, durations, healthy, out, 1, tmp_path / "tts")
    assert healthy.synthesized == ["unit-0#r3"]
    assert not any(r["error"] for r in second["rows"]) and [r["rung"] for r in second["rows"]] == [1, 3, 4, 5]


_E2E_LONG = "Ну, честно говоря, это очень сложный вопрос для нас всех, правда."
_E2E_RUNGS = ["Честно говоря, это очень сложный вопрос для нас всех, правда.", "Это очень сложный вопрос для нас всех.",
              "Это сложный вопрос для нас.", "Это сложный вопрос.", "Это вопрос."]


def _e2e_source(tmp_path: Path) -> Path:
    phrases = ["Хорошо, пойдём домой.", "Да, конечно, я согласен.", "Нет, это невозможно.", "Мы уже почти приехали."]
    blocks = []
    starts = [4000 * n for n in range(40)]
    for n, start in enumerate(starts):
        text = _E2E_LONG if n == 5 else phrases[n % 4]
        if n == 6:
            start = starts[5] + 1500
        end = start + 1800
        blocks.append(f"{n + 1}\n{_ms(start)} --> {_ms(end)}\n{text}\n")
    source = tmp_path / "episode.srt"
    source.write_text("\n".join(blocks), encoding="utf-8")
    return source


def _ms(value: int) -> str:
    seconds, millis = divmod(value, 1000)
    return f"00:{seconds // 60:02d}:{seconds % 60:02d},{millis:03d}"


def _e2e_reply(body: dict) -> dict:
    user = body["messages"][1]["content"]
    if "Верни JSON такого вида" in user:
        return {"units": {"unit-5": {"units": [
            {"id": "u1", "text": "Ну, честно говоря,", "role": "decor"},
            {"id": "u2", "text": "это очень сложный вопрос", "role": "core"},
            {"id": "u3", "text": "для нас всех, правда.", "role": "support"}], "refs": {}, "act": "statement"}}}
    if "Сокращённая реплика" in user:
        return {"fragments": [{"id": "u2", "kept": True, "why": "ok"}, {"id": "u3", "kept": True, "why": "ok"}]}
    return {"rungs": [{"rung": n, "text": _E2E_RUNGS[n - 1], "kept": ["u2"], "dropped": ["u1"], "compressed": [],
                       "note": ""} for n in range(1, 6)]}


def test_run_end_to_end_with_fakes(tmp_path: Path) -> None:
    from tests.fakes import FakeEngine, FakeTransport
    from tests.test_voiceover_render import fake_stretch
    from voiceover.llm import LLMClient
    from voiceover.pipeline import read_json, run
    config = {"engine": {"name": "fake"}, "policy": {"lead_ms": 360, "lag_ms": 500, "gap_ms": 150},
              "synthesis": {"workers": 1, "cache_dir": str(tmp_path / "tts")},
              "llm": {"cache_dir": str(tmp_path / "llm")}, "compression": {"workers": 2},
              "verify": {"workers": 2}, "export": {"sample_rate": 22050}}
    transport = FakeTransport(_e2e_reply)
    engine = FakeEngine()
    out = tmp_path / "out"
    source = _e2e_source(tmp_path)

    def go(until: str = "report") -> dict:
        return run(source, out, config, until=until, engine_factory=lambda _: engine,
                   llm_factory=lambda cfg: LLMClient(cfg, transport, sleep=lambda _: None), stretch=fake_stretch)

    assert go("measure")["stage"] == "measure"
    summary = go()
    export = out / "export"
    for name in ("track.wav", "track_guide.md", "spoken.srt", "display.srt", "ledger.json", "human_queue.md",
                 "manifest.json"):
        assert (export / name).exists(), name
    for name in ("durations1.json", "verified.json", "schedule1.json"):
        assert (out / name).exists(), name
    assert summary["resolution"]["changed"] == 1 and summary["resolution"]["collisions"] == 0
    assert summary["export"]["counts"]["units"] == 40
    manifest = read_json(export / "manifest.json")
    assert manifest["prompt_versions"]["verify"] == "v2" and manifest["counts"]["changed"] == 1
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## Фаза 4: измеренный пересчёт и экспорт" in report and "## Сокращение текста" in report

    # a rerun recomputes nothing: no synthesis and no model calls
    calls, synthesized = transport.calls, len(engine.synthesized)
    go()
    assert transport.calls == calls and len(engine.synthesized) == synthesized


def test_stage_measure_adds_manual_rows_and_reruns_when_overrides_change(tmp_path: Path) -> None:
    from tests.fakes import FakeEngine, fake_speech_sec
    from voiceover.overrides import Overrides
    from voiceover.pipeline import stage_measure
    candidates, durations = _measure_inputs(tmp_path)
    out = tmp_path / "out"
    manual = Overrides(texts={"unit-0": "Но она думает иначе."})
    engine = FakeEngine()
    artifact = stage_measure(candidates, durations, engine, out, 1, tmp_path / "tts", manual)
    assert engine.synthesized == ["unit-0#r1", "unit-0#r3", "unit-0#r4", "unit-0#r5", "unit-0#manual"]
    row = artifact["rows"][-1]
    assert row["rung"] == "manual" and row["unit_id"] == "unit-0" and row["text"] == "Но она думает иначе."
    assert row["speech_sec"] == fake_speech_sec(row["text"]) and Path(row["path"]).exists()
    assert artifact["inputs"]["overrides_sha256"]

    same = FakeEngine()
    stage_measure(candidates, durations, same, out, 1, tmp_path / "tts", manual)
    assert same.synthesized == []
    changed = FakeEngine()
    rebuilt = stage_measure(candidates, durations, changed, out, 1, tmp_path / "tts",
                            Overrides(texts={"unit-0": "Другой текст."}))
    assert rebuilt["rows"][-1]["text"] == "Другой текст." and rebuilt["inputs"] != artifact["inputs"]


def test_stage_measure_retries_a_failed_manual_row(tmp_path: Path) -> None:
    from tests.fakes import FakeEngine
    from voiceover.overrides import Overrides
    from voiceover.pipeline import stage_measure
    candidates, durations = _measure_inputs(tmp_path)
    out = tmp_path / "out"
    manual = Overrides(texts={"unit-0": "Сломанный текст."})
    first = stage_measure(candidates, durations, FakeEngine(fail_texts={"Сломанный текст."}), out, 1,
                          tmp_path / "tts", manual)
    assert first["rows"][-1]["error"]
    healthy = FakeEngine()
    second = stage_measure(candidates, durations, healthy, out, 1, tmp_path / "tts", manual)
    assert healthy.synthesized == ["unit-0#manual"] and not second["rows"][-1]["error"]


def _e2e_config(tmp_path: Path) -> dict:
    return {"engine": {"name": "fake"}, "policy": {"lead_ms": 360, "lag_ms": 500, "gap_ms": 150},
            "synthesis": {"workers": 1, "cache_dir": str(tmp_path / "tts")},
            "llm": {"cache_dir": str(tmp_path / "llm")}, "compression": {"workers": 2},
            "verify": {"workers": 2}, "export": {"sample_rate": 22050}}


def test_overrides_rerun_measure_resolve_export_and_reuse_verify(tmp_path: Path) -> None:
    from tests.fakes import FakeEngine, FakeTransport
    from tests.test_voiceover_render import fake_stretch
    from voiceover.llm import LLMClient
    from voiceover.pipeline import read_json, run
    transport = FakeTransport(_e2e_reply)
    engine = FakeEngine()
    out = tmp_path / "out"
    source = _e2e_source(tmp_path)

    def go(**kwargs: object) -> dict:
        return run(source, out, _e2e_config(tmp_path), engine_factory=lambda _: engine,
                   llm_factory=lambda cfg: LLMClient(cfg, transport, sleep=lambda _: None), stretch=fake_stretch,
                   **kwargs)

    go()
    before = {name: read_json(out / name) for name in ("durations1.json", "verified.json", "schedule1.json")}
    assert before["durations1.json"]["inputs"]["overrides_sha256"] is None
    assert "Ручные правки" not in (out / "report.md").read_text(encoding="utf-8")
    calls = transport.calls

    manual_text = "Это непростой вопрос."
    (out / "overrides.json").write_text(json.dumps({
        "schema": "voiceover-overrides-v1",
        "units": {"unit-5": {"text": manual_text, "note": "проверено"}, "unit-0": {"accept": True}}},
        ensure_ascii=False), encoding="utf-8")
    summary = go()  # picked up from <out>/overrides.json
    after = {name: read_json(out / name) for name in before}
    for name in ("durations1.json", "schedule1.json"):
        assert after[name]["inputs"] != before[name]["inputs"], name
        assert after[name]["inputs"]["overrides_sha256"], name
    assert after["verified.json"] == before["verified.json"]  # verify reused, manual texts are not verified
    assert transport.calls == calls
    manual_rows = [r for r in after["durations1.json"]["rows"] if r["rung"] == "manual"]
    assert [(r["unit_id"], r["text"]) for r in manual_rows] == [("unit-5", manual_text)]
    choices = {c["unit_id"]: c for c in after["schedule1.json"]["choices"]}
    assert choices["unit-5"]["status"] == "manual" and choices["unit-5"]["chosen_text"] == manual_text
    export = out / "export"
    assert manual_text in (export / "spoken.srt").read_text(encoding="utf-8")
    manifest = read_json(export / "manifest.json")
    assert manifest["counts"]["manual"] == 1 and manifest["stage_inputs"]["overrides"]
    assert summary["resolution"]["overrides"] == {"manual_texts": 1, "accepted": 1}
    assert "Ручные правки применены: ручных текстов 1, принятых единиц 1." in (out / "report.md").read_text(
        encoding="utf-8")

    # unchanged overrides: nothing is rebuilt
    synthesized = len(engine.synthesized)
    go()
    assert len(engine.synthesized) == synthesized
    assert read_json(out / "schedule1.json") == after["schedule1.json"]


def test_run_rejects_unknown_override_units_and_missing_explicit_file(tmp_path: Path) -> None:
    from tests.fakes import FakeEngine, FakeTransport
    from voiceover.llm import LLMClient
    from voiceover.pipeline import run
    out = tmp_path / "out"
    source = _e2e_source(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "voiceover-overrides-v1",
                               "units": {"unit-777": {"text": "x"}, "unit-0": {"accept": True}}}), encoding="utf-8")
    kwargs = dict(until="units", engine_factory=lambda _: FakeEngine(),
                  llm_factory=lambda cfg: LLMClient(cfg, FakeTransport(_e2e_reply), sleep=lambda _: None))
    with pytest.raises(ValueError, match="unit-777"):
        run(source, out, _e2e_config(tmp_path), overrides_path=bad, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not found"):
        run(source, out, _e2e_config(tmp_path), overrides_path=tmp_path / "missing.json", **kwargs)  # type: ignore[arg-type]


def test_run_help_lists_the_overrides_option(capsys: pytest.CaptureFixture[str]) -> None:
    from voiceover.pipeline import main
    with pytest.raises(SystemExit):
        main(["run", "--help"])
    assert "--overrides" in capsys.readouterr().out
