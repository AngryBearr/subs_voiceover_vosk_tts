"""Offline controller checks. All model/normalizer/TTS/tempo calls are mocked."""

from __future__ import annotations

import copy
import json
import math
import wave
from pathlib import Path
from typing import Any

import pytest

from utils import pipeline_batch as batch
from utils.pipeline_text import json_sha256, text_sha256

RENDER_REPORT = batch.publish_report

def write_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(1000)
        stream.writeframes(b"\x10\x00" * round(seconds * 1000))


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    monkeypatch.setattr(batch, "runtime_binding", lambda config: {"offline_fixture": True})
    monkeypatch.setattr(batch, "publish_report", lambda folder, state: None)
    items = [{"index": 11 + i, "start": i * 2000, "end": i * 2000 + 1000, "text": text,
              "preserved": {"custom": i}}
             for i, text in enumerate(["Да.", "Очень длинная исходная реплика с подробностями и смыслом.",
                                       "BAD исходная нормализация", "Хорошая.", "Дальше."])]
    config = {"tts": {"voice": 8, "base_speech_rate": 1.25, "max_total_speed": 1.5},
              "reduce_silence": {"enabled": False}, "resample": {"enabled": False}}
    source = tmp_path / "source.json"
    batch.save_json(source, items)
    folder = tmp_path / "batch"
    batch.prepare(source, folder, config, 0, 3)
    return folder, items, config


def response(folder: Path, proposals: dict[str, str | None], verdicts: dict[str, str] | None = None,
             number: int = 1) -> dict[str, Any]:
    state = batch.load_state(folder)
    request = state["requests"][str(number)]
    result = {key: copy.deepcopy(request[key]) for key in
              ("round", "request_sha256", "binding_sha256", "source_sha256", "plan_sha256", "policy")}
    result.update(schema="pipeline-batch-reviewed-v1", execution="native_live", entries=[])
    for original in request["entries"]:
        candidate = proposals.get(original["cue_id"])
        result["entries"].append({"cue_id": original["cue_id"], "original": original["original"],
            "candidate": candidate, "writer": {"model": "mock", "mode": "unit_writer", "invocation_id": "writer"},
            "review": {"verdict": (verdicts or {}).get(original["cue_id"], "accept" if candidate else "refusal"),
                       "text_sha256": text_sha256(candidate) if candidate is not None else None,
                       "request_sha256": request["request_sha256"], "model": "mock",
                       "mode": "unit_reviewer", "invocation_id": "reviewer"}})
    chosen, decisions = batch.reviewed_selection(folder, state, request, result["entries"])
    proposed = copy.deepcopy(state)
    applied = {row["cue_id"] for row in decisions.values() if row["reason"] == "applied"}
    for cue in proposed["cues"]:
        if cue["cue_id"] in applied:
            cue["chosen_text"] = batch.join_text_lines(chosen[cue["position"]]["text"])
    result["assembly_review"] = {"verdict": "accept", "assembly_sha256": json_sha256(batch.assembled(proposed)),
                                 "request_sha256": request["request_sha256"], "model": "mock",
                                 "mode": "unit_assembled_reviewer", "invocation_id": "assembly-reviewer"}
    return result


def ingest(folder: Path, result: dict[str, Any]) -> dict[str, Any]:
    path = folder.parent / "response.json"
    batch.save_json(path, result)
    return batch.ingest_reviewed(folder, path)


def mocked_engine(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    class MockEngine:
        def __init__(self, config: dict[str, Any]) -> None:
            assert config["tts"]["voice"] == 8

        def normalize(self, item: dict[str, Any]) -> str:
            if "BAD" in item["text"]:
                raise ValueError("RUNorm lexical change")
            return item["text"]

        def synthesize(self, text: str, path: Path, inputs: list[str]) -> None:
            calls.append(text)
            inputs.append(batch.normalize_vosk_text(text))
            # A nominal fit overruns its FULL 1.8 sec window; long originals do too.
            write_wav(path, 2.5 if text == "Да." or len(text) > 35 else 0.8)

    monkeypatch.setattr(batch, "AudioEngine", MockEngine)
    return calls


def test_prepare_full_windows_selection_and_cli_readonly(workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]],
                                                       monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    folder, items, config = workspace
    request = batch.read_json(folder / "request_round1.json")
    assert [row["cue_id"] for row in request["entries"]] == ["cue-1"]
    assert len(request["assembled_context"]) == 4
    assert request["entries"][0]["original_context"][-1]["cue_id"] == "cue-4"
    state = batch.load_state(folder)
    assert state["cues"][-1]["target"]["available_ms"] == 1800  # Includes immediate NEXT source cue.
    assert state["cues"][-1]["original"] == items[3]
    before = {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}
    monkeypatch.setattr(batch, "AudioEngine", lambda config: pytest.fail("Status imported inference"))
    batch.main(["status", "--output-dir", str(folder)])
    assert json.loads(capsys.readouterr().out)["tts_attempts"] == 0
    assert before == {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}
    assert not (folder / "report.html").exists()
    assert batch.prepare(folder.parent / "source.json", folder, config, 0, 3) == state


def test_mixed_fit_repair_queue_only_failures_max_two_and_resume(workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]],
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    folder, _, _ = workspace
    calls = mocked_engine(monkeypatch)
    first = response(folder, {"cue-1": "Короткая осмысленная реплика."})
    ingest(folder, first)
    first_state = batch.audio(folder)
    assert calls == ["Да.", "Короткая осмысленная реплика.", "Хорошая."]
    request = first_state["requests"]["2"]
    assert [row["cue_id"] for row in request["entries"]] == ["cue-0", "cue-2"]
    assert request["entries"][0]["target"]["measured_advisory_max_chars"] == 2
    assert request["entries"][1]["target"]["repair_reason"] == "normalization_failed"
    final = response(folder, {"cue-0": "Да", "cue-2": "Исходная нормализация"}, number=2)
    ingest(folder, final)
    completed = batch.audio(folder)
    assert batch.status_summary(completed)["ready_for_separate_mix"]
    assert len(calls) == 5
    assert all(sum(attempt["tts_attempts"] for attempt in cue["attempts"]) <= 2 for cue in completed["cues"])
    snapshots = {str(path): path.read_bytes() for path in folder.rglob("*.wav")}
    batch.audio(folder)
    ingest(folder, final)
    assert len(calls) == 5 and snapshots == {str(path): path.read_bytes() for path in folder.rglob("*.wav")}
    assert completed["cues"][0]["attempts"][0]["engine_input_text"] == "Да."
    assert completed["cues"][0]["attempts"][0]["pretempo"]["frames"] == 2500
    assert len(completed["cues"][1]["attempts"]) == 1


@pytest.mark.parametrize("key", ["source_sha256", "plan_sha256", "policy", "candidate", "assembly", "independence"])
def test_stale_hashes_fail_before_changes(workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]], key: str) -> None:
    folder, _, _ = workspace
    result = response(folder, {"cue-1": "Коротко."})
    before = (folder / "state.json").read_bytes()
    if key in {"source_sha256", "plan_sha256", "policy"}:
        result[key] = "changed"
    elif key == "candidate":
        result["entries"][0]["candidate"] = "Другое."
    elif key == "assembly":
        result["assembly_review"]["assembly_sha256"] = "stale"
    else:
        result["entries"][0]["review"]["invocation_id"] = "writer"
    with pytest.raises(ValueError):
        ingest(folder, result)
    assert before == (folder / "state.json").read_bytes()


def test_rejected_review_does_not_block_other_original_audio(workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]],
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    folder, items, _ = workspace
    calls = mocked_engine(monkeypatch)
    state = ingest(folder, response(folder, {"cue-1": "Коротко."}, {"cue-1": "revise"}))
    assert state["cues"][1]["chosen_text"] == items[1]["text"]
    assert state["cues"][1]["decisions"][0]["reason"] == "revise"
    state = batch.audio(folder)
    assert len(calls) == 3
    assert state["cues"][3]["attempts"][0]["status"] == "fits"
    assert [row["cue_id"] for row in state["requests"]["2"]["entries"]] == ["cue-0", "cue-1", "cue-2"]


def test_interrupt_unknown_intent_never_retries(workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]],
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    folder, _, _ = workspace
    ingest(folder, response(folder, {"cue-1": "Коротко."}))
    calls = mocked_engine(monkeypatch)

    def interrupt(self: Any, item: dict[str, Any]) -> str:
        raise KeyboardInterrupt()

    monkeypatch.setattr(batch.AudioEngine, "normalize", interrupt)
    with pytest.raises(KeyboardInterrupt):
        batch.audio(folder)
    assert batch.status_summary(batch.load_state(folder))["counts"]["needs_reconciliation"] == 1
    mocked_engine(monkeypatch)
    state = batch.audio(folder)
    assert state["cues"][0]["attempts"][0]["status"] == "started"
    assert not calls
    assert "cue-0" not in [row["cue_id"] for row in state["requests"]["2"]["entries"]]
    assert not batch.status_summary(state)["ready_for_separate_mix"]


def test_round_two_failure_is_final_no_third_loop(workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]],
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    folder, _, _ = workspace
    calls = mocked_engine(monkeypatch)
    ingest(folder, response(folder, {"cue-1": "Коротко."}))
    batch.audio(folder)
    ingest(folder, response(folder, {"cue-0": "Да", "cue-2": "BAD нормализация"}, number=2))
    state = batch.audio(folder)
    assert state["cues"][2]["attempts"][-1]["status"] == "normalization_failed"
    assert set(state["requests"]) == {"1", "2"}
    assert not batch.status_summary(state)["ready_for_separate_mix"]
    count = len(calls)
    batch.audio(folder)
    assert len(calls) == count


def test_nominal_advisory_overrun_is_allowed(workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]]) -> None:
    folder, _, _ = workspace
    candidate = "Достаточно длинная реплика, но короче исходной."
    state = ingest(folder, response(folder, {"cue-1": candidate}))
    assert state["cues"][1]["chosen_text"] == candidate
    assert state["cues"][1]["decisions"][0]["over_advisory_target"]


def test_runtime_source_and_wav_tampering_block_resume(workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]],
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    folder, _, _ = workspace
    monkeypatch.setattr(batch, "runtime_binding", lambda config: {"different": True})
    with pytest.raises(ValueError, match="Runtime"):
        batch.load_state(folder)
    monkeypatch.setattr(batch, "runtime_binding", lambda config: {"offline_fixture": True})
    mocked_engine(monkeypatch)
    ingest(folder, response(folder, {"cue-1": "Коротко."}))
    batch.audio(folder)
    wav = next(folder.rglob("*.wav"))
    old = wav.read_bytes()
    wav.write_bytes(b"changed")
    with pytest.raises(ValueError, match="WAV changed"):
        batch.audio(folder)
    wav.write_bytes(old)
    (folder.parent / "source.json").write_text("[]")
    with pytest.raises(ValueError, match="Source changed"):
        batch.load_state(folder)


def test_measured_tempo_preserves_pretempo_and_total_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from utils import audio_analysis as aa
    raw, pretempo, final = [tmp_path / name for name in ("raw.wav", "pretempo.wav", "final.wav")]
    write_wav(raw, 1.1)
    monkeypatch.setattr(aa, "_ffmpeg_available", lambda: True)
    factors: list[float] = []

    def tempo(source: Path, output: Path, *, speed_factor: float, **kwargs: Any) -> None:
        factors.append(speed_factor)
        assert source == pretempo
        write_wav(output, 1.0)

    monkeypatch.setattr(aa, "_speedup_with_ffmpeg", tempo)
    attempt: dict[str, Any] = {}
    config = {"tts": {"base_speech_rate": 1.25, "max_total_speed": 1.5},
              "reduce_silence": {"enabled": False}, "resample": {"enabled": False}}
    batch.measure_audio(config, raw, pretempo, final, 1.0, attempt)
    assert factors == [1.1]
    assert raw.read_bytes() == pretempo.read_bytes()
    assert attempt["status"] == "accelerated_fits" and attempt["effective_total_speed"] <= 1.5
    assert attempt["pretempo"]["frames"] == 1100 and attempt["final"]["frames"] == 1000


def test_actual_shared_normalization_guard_not_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    from utils import text_normalizer as normalizer

    class BadNorm:
        def norm(self, text: str) -> str:
            return "Другое слово"

    monkeypatch.setattr(normalizer, "_load_runorm", lambda device: BadNorm())
    monkeypatch.setattr(normalizer, "_load_ruaccent", lambda device: object())
    engine = batch.AudioEngine({})
    with pytest.raises(ValueError, match="RUNorm lexical change"):
        engine.normalize({"index": 9, "text": "Исходное слово"})


def test_cli_prepare_and_frozen_config_validation(tmp_path: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batch, "runtime_binding", lambda config: {})
    source = tmp_path / "input.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nДа.\n", encoding="utf-8")
    folder = tmp_path / "run"
    batch.main(["prepare", str(source), "--start", "0", "--end", "0", "--output-dir", str(folder)])
    assert json.loads(capsys.readouterr().out)["cues"] == 1
    config = batch.read_json(folder / "config.json")
    config["tts"]["max_total_speed"] = 1.3
    batch.save_json(folder / "config.json", config)
    with pytest.raises(ValueError, match="Frozen config"):
        batch.load_state(folder)


def test_report_gate_and_escaping(workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]]) -> None:
    folder, _, _ = workspace
    # Test renderer directly, never publish a fake report artifact outside tmp_path.
    state = batch.load_state(folder)
    RENDER_REPORT(folder, state)
    assert not (folder / "report.html").exists()
    state["proposals"]["1"] = {"execution": "dry_run"}
    RENDER_REPORT(folder, state)
    assert not (folder / "report.html").exists()
    state["proposals"].clear()
    state["imports"]["1"] = {"result": {"execution": "native_live", "candidate": "<script>alert(1)</script>"}}
    RENDER_REPORT(folder, state)
    document = (folder / "report.html").read_text()
    assert "<script>" not in document and "&lt;script&gt;" in document and "media-src data:" in document
    assert "<table>" in document and "<details>" in document and 'lang="ru"' in document
    assert "Уместились:" in document and "Стоимость" in document


def test_native_review_input_hashes_rejected_subset_and_frozen_proposals(
    workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]], capsys: Any,
) -> None:
    folder, _, _ = workspace
    final = response(folder, {"cue-1": "Коротко."}, {"cue-1": "revise"})
    proposals = {key: copy.deepcopy(value) for key, value in final.items() if key != "assembly_review"}
    proposals["schema"] = "pipeline-batch-proposals-v1"
    for row in proposals["entries"]:
        del row["review"]
    path = folder.parent / "writer.json"
    batch.save_json(path, proposals)
    batch.main(["review-input", str(path), "--output-dir", str(folder)])
    assert json.loads(capsys.readouterr().out)["native_phases"] == 1
    review_input = batch.read_json(folder / "reviewer_input_round1.json")
    assert review_input["entries"][0]["candidate_sha256"] == text_sha256("Коротко.")
    assert review_input["assembled_context"][1]["text"] == "Коротко."
    per_cue = folder.parent / "per-cue.json"
    batch.save_json(per_cue, final)
    batch.review_input(folder, path, per_cue)
    review_input = batch.read_json(folder / "reviewer_input_round1.json")
    assert review_input["entries"][0]["advisory_check"] == {
        "normalized_chars": 8, "advisory_max_chars": 35, "source": "nominal", "exceeds_advisory": False}
    before = (folder / "reviewer_input_round1.json").read_bytes()
    state_before = (folder / "state.json").read_bytes()
    batch.review_input(folder, path, per_cue)
    assert (folder / "reviewer_input_round1.json").read_bytes() == before
    assert (folder / "state.json").read_bytes() == state_before
    assert review_input["assembly_sha256"] == final["assembly_review"]["assembly_sha256"]
    assert review_input["assembly_sha256"] == json_sha256(batch.assembled(batch.load_state(folder)))
    assert batch.load_state(folder)["native_phase_count"] == 1
    assert ingest(folder, final)["native_phase_count"] == 2
    different = copy.deepcopy(final)
    different["entries"][0]["candidate"] = "Иное."
    different["entries"][0]["review"]["text_sha256"] = text_sha256("Иное.")
    with pytest.raises(ValueError, match="already imported"):
        ingest(folder, different)


def test_single_writer_lock(workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]]) -> None:
    folder, _, _ = workspace
    with batch.locked(folder):
        with pytest.raises(ValueError, match="another process"):
            batch.audio(folder)


def test_final_repair_refusal_exports_only_real_fits_fail_closed(
    workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder, _, _ = workspace
    calls = mocked_engine(monkeypatch)
    ingest(folder, response(folder, {"cue-1": "Коротко."}))
    batch.audio(folder)
    ingest(folder, response(folder, {"cue-0": None, "cue-2": None}, number=2))
    state = batch.audio(folder)
    assert len(calls) == 3 and batch.status_summary(state)["tts_attempts"] == 3
    manifest = batch.read_json(folder / "fits.json")
    assert not manifest["ready_for_separate_mix"]
    assert [row["cue_id"] for row in manifest["entries"]] == ["cue-1", "cue-3"]
    assert {row["reason"] for row in batch.read_json(folder / "exceptions.json")} == {"unresolved", "normalization_failed"}


@pytest.mark.parametrize("candidate,reason", [("24 участника.", "new_numeric_representation"),
                                             ("Коротко.", "assembly_rejected")])
def test_new_numeric_representation_keeps_original_and_assembly_rejection_blocks_changes(
    workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]], candidate: str, reason: str,
) -> None:
    folder, items, _ = workspace
    result = response(folder, {"cue-1": candidate})
    result["assembly_review"]["verdict"] = "revise"
    state = ingest(folder, result)
    assert state["cues"][1]["chosen_text"] == items[1]["text"]
    assert state["cues"][1]["decisions"][0]["reason"] == reason


def test_invalid_source_is_per_cue_exception_no_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batch, "runtime_binding", lambda config: {})
    monkeypatch.setattr(batch, "AudioEngine", lambda config: object())
    path, folder = tmp_path / "source.json", tmp_path / "run"
    batch.save_json(path, [{"index": 7, "start": None, "end": 1000, "text": "Да."}])
    batch.prepare(path, folder, {}, 0, 0)
    state = batch.audio(folder)
    assert batch.status_summary(state)["continuous_duration_sec"] is None
    assert batch.status_summary(state)["counts"] == {"invalid_source": 1}
    assert batch.status_summary(state)["tts_attempts"] == 0
    assert not batch.status_summary(state)["ready_for_separate_mix"]
    assert not (folder / "report.html").exists()


def test_real_runtime_binding_ignores_hf_coordination_but_detects_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any,
) -> None:
    import types

    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(batch, "RUNTIME_FILES", ("controller.py",))
    monkeypatch.setattr(batch.importlib.metadata, "distribution", lambda package: types.SimpleNamespace(
        version="test-version", locate_file=lambda module: tmp_path / "packages" / module))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "controller.py").write_text("# frozen code\n")
    cache = tmp_path / "runorm_cache" / "models--test"
    weight = cache / "snapshots" / "revision-a" / "model.safetensors"
    batch.save_json(weight.parent / "config.json", {"model_type": "test"})
    batch.save_json(weight.parent / "tokenizer.json", {"tokens": ["one"]})
    weight.write_bytes(b"real-model-content")
    source, folder = tmp_path / "source.json", tmp_path / "run"
    batch.save_json(source, [{"index": 1, "start": 0, "end": 1000, "text": "Да."}])
    state = batch.prepare(source, folder, {}, 0, 0)
    assert str(weight.resolve()) in state["binding"]["runtime"]["model_assets"]
    for name in (".locks/marker", ".no_exist/revision-a/tokenizer_config.json", "refs/main",
                 ".cache/huggingface/download/file.metadata", "snapshots/revision-a/file.incomplete"):
        path = cache / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("benign cache touch")
    batch.main(["status", "--output-dir", str(folder)])
    assert json.loads(capsys.readouterr().out)["tts_attempts"] == 0
    weight.write_bytes(b"changed-real-model-content")
    with pytest.raises(ValueError, match="Runtime/model assets changed"):
        batch.load_state(folder)


def test_same_independent_reviewer_can_review_candidates_and_assembly(
    workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]],
) -> None:
    folder, _, _ = workspace
    result = response(folder, {"cue-1": "Коротко."})
    result["assembly_review"]["invocation_id"] = result["entries"][0]["review"]["invocation_id"]
    state = ingest(folder, result)
    assert state["cues"][1]["chosen_text"] == "Коротко."
    assert batch.status_summary(state)["assembly_review_status"] == "accepted_exact_final_assembly"


@pytest.mark.parametrize("verdict", ["revise", "refusal", "accept_stale"])
def test_all_audio_fits_cannot_override_rejected_assembly(
    workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]], monkeypatch: pytest.MonkeyPatch, verdict: str,
) -> None:
    folder, _, _ = workspace
    result = response(folder, {"cue-1": "Коротко."})
    result["assembly_review"]["verdict"] = "accept" if verdict == "accept_stale" else verdict
    state = ingest(folder, result)
    if verdict == "accept_stale":
        state["cues"][1]["chosen_text"] = "Теперь другой текст."
        state["cues"][1]["chosen_sha256"] = text_sha256(state["cues"][1]["chosen_text"])
    for cue in state["cues"]:
        cue["attempts"] = [{"round": 1, "status": "fits", "tts_attempts": 1,
            "final": {"frames": 500, "sample_rate": 1000, "duration_sec": 0.5}}]
    state["requests"]["2"] = batch.make_request(folder, state, 2, [])
    state["imports"]["2"] = {"status": "no_targets"}
    summary = batch.status_summary(state)
    assert summary["automatic_fit_fraction"] == 1.0 and not summary["exceptions"]
    assert not summary["ready_for_separate_mix"]
    assert {row["reason"] for row in summary["mix_blockers"]} == {"assembly_rejected_or_unreviewed"}


def test_zero_writer_targets_trusted_unchanged_originals_need_no_model_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(batch, "runtime_binding", lambda config: {})
    source, folder = tmp_path / "source.json", tmp_path / "run"
    batch.save_json(source, [{"index": 1, "start": 0, "end": 1000, "text": "Да."}])
    config = {"tts": {"voice": 8}, "reduce_silence": {"enabled": False}, "resample": {"enabled": False}}
    batch.prepare(source, folder, config, 0, 0)
    mocked_engine(monkeypatch)
    # This nominal fit is measured unresolved first; make its mock duration fit.
    monkeypatch.setattr(batch.AudioEngine, "synthesize", lambda self, text, path, inputs:
                        (inputs.append(batch.normalize_vosk_text(text)), write_wav(path, 0.8)))
    state = batch.audio(folder)
    summary = batch.status_summary(state)
    assert summary["assembly_review_status"] == "retained_originals_no_model_review_needed"
    assert summary["ready_for_separate_mix"] and summary["native_phases"] == 0
    assert not (folder / "report.html").exists()


@pytest.mark.parametrize("repair,reason", [("Коротко.", "repair_unchanged"),
                                          ("  Коротко.  ", "repair_unchanged"),
                                          ("Снова более длинная реплика.", "repair_not_shorter")])
def test_ineligible_timing_repairs_do_not_resynthesize_and_other_cues_proceed(
    workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]], monkeypatch: pytest.MonkeyPatch,
    repair: str, reason: str,
) -> None:
    folder, _, _ = workspace
    calls = mocked_engine(monkeypatch)
    initial = response(folder, {"cue-1": "Коротко."})
    ingest(folder, initial)
    synthesize = batch.AudioEngine.synthesize

    def timed(self: Any, text: str, path: Path, inputs: list[str]) -> None:
        synthesize(self, text, path, inputs)
        if text == "Коротко.":
            write_wav(path, 2.5)

    monkeypatch.setattr(batch.AudioEngine, "synthesize", timed)
    batch.audio(folder)
    final = response(folder, {"cue-0": "Да", "cue-1": repair, "cue-2": "Исходная нормализация"}, number=2)
    proposal = copy.deepcopy(final)
    proposal["schema"] = "pipeline-batch-proposals-v1"
    del proposal["assembly_review"]
    for row in proposal["entries"]:
        del row["review"]
    proposal_path = folder.parent / "writer2.json"
    batch.save_json(proposal_path, proposal)
    batch.review_input(folder, proposal_path)
    review_input = batch.read_json(folder / "reviewer_input_round2.json")
    row = next(row for row in review_input["entries"] if row["cue_id"] == "cue-1")
    assert row["eligibility_reason"] == reason
    assert review_input["assembled_context"][1]["text"] == "Коротко."
    assert review_input["assembly_sha256"] == final["assembly_review"]["assembly_sha256"]
    ingest(folder, final)
    state = batch.audio(folder)
    assert state["cues"][1]["decisions"][-1]["reason"] == reason
    assert len(state["cues"][1]["attempts"]) == 1
    assert len(calls) == 5 and state["cues"][0]["attempts"][-1]["status"] == "fits"
    assert state["cues"][2]["attempts"][-1]["status"] == "fits"
    assert reason in {row["reason"] for row in batch.status_summary(state)["exceptions"]}


@pytest.mark.parametrize("seconds,ready", [(0.8, False), (0.5, True)])
def test_actual_timeline_overlap_blocks_individually_fitted_source_cues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seconds: float, ready: bool,
) -> None:
    monkeypatch.setattr(batch, "runtime_binding", lambda config: {})
    source, folder = tmp_path / "source.json", tmp_path / "run"
    items = [{"index": 8, "start": 0, "end": 1000, "text": "Да."},
             {"index": 11, "start": 500, "end": 1500, "text": "Нет."}]
    batch.save_json(source, items)
    config = {"tts": {"voice": 8}, "reduce_silence": {"enabled": False}, "resample": {"enabled": False}}
    batch.prepare(source, folder, config, 0, 1)
    mocked_engine(monkeypatch)
    monkeypatch.setattr(batch.AudioEngine, "synthesize", lambda self, text, path, inputs:
                        (inputs.append(batch.normalize_vosk_text(text)), write_wav(path, seconds)))
    state = batch.audio(folder)
    summary = batch.status_summary(state)
    assert summary["automatic_fit_fraction"] == 1.0 and summary["ready_for_separate_mix"] is ready
    assert summary["mix_blockers"] == ([] if ready else [
        {"reason": "final_timeline_overlap", "cue_id": "cue-0", "next_cue_id": "cue-1"}])
    assert batch.read_json(source) == items
    assert batch.read_json(folder / "fits.json")["ready_for_separate_mix"] is ready


def test_cli_rejects_non_root_before_output_or_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="2"):
        batch.main(["status", "--output-dir", str(tmp_path / "missing")])


def test_normalization_repair_can_be_longer_than_current_but_not_identical(
    workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder, _, _ = workspace
    mocked_engine(monkeypatch)
    ingest(folder, response(folder, {"cue-1": "BAD реплика"}))
    batch.audio(folder)
    result = response(folder, {"cue-0": "Да", "cue-1": "Короткая реплика.", "cue-2": "Исходная нормализация"}, number=2)
    state = ingest(folder, result)
    assert state["cues"][1]["decisions"][-1]["reason"] == "applied"
    assert len("Короткая реплика.") > len("BAD реплика")
    assert batch.status_summary(batch.audio(folder))["ready_for_separate_mix"]


@pytest.mark.parametrize("duration", [2, 2.0, 0.5, None, 0, -1, True, "2", math.nan, math.inf])
def test_request_canonical_current_length_and_unchanged_measured_formula(
    workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]], duration: Any,
) -> None:
    folder, _, _ = workspace
    state = batch.load_state(folder)
    cue = copy.deepcopy(state["cues"][1])
    cue["chosen_text"] = "  «Мама»   мама…  "
    cue["attempts"] = [{"status": "unresolved", **({"pretempo": {"duration_sec": duration}} if duration is not None else {})}]
    plan = batch.read_json(folder / "full_plan.json")
    for number in (1, 2):
        if number == 2 and type(duration) is float and not math.isfinite(duration):
            # Frozen request serialization still rejects non-JSON measurements.
            with pytest.raises(ValueError, match="JSON compliant"):
                batch.make_request(folder, state, number, [cue])
            continue
        request = batch.make_request(folder, state, number, [cue])
        row = request["entries"][0]
        assert row["current_text"] == cue["chosen_text"]
        assert row["current_normalized_chars"] == len('"Мама" мама...') == 14  # Not 8 Russian letters.
        assert {key: row["target"][key] for key in cue["target"]} == cue["target"]
        assert row["target"]["available_ms"] == 1800
        assert request["policy"] == plan["policy"]
        assert "before RUNorm/RUAccent" in request["instructions"]
        assert "Under budget does not guarantee audio fit" in request["instructions"]
        valid = type(duration) in (int, float) and math.isfinite(duration) and duration > 0
        if number == 2 and valid:
            old_budget = math.floor(14 * 1800 / 1000 * (1.5 / 1.25) / duration * 0.9)
            assert old_budget == (13 if duration == 2 else 54)
            assert row["target"]["measured_advisory_max_chars"] == old_budget
            assert row["target"]["measured_required_reduction_chars"] == max(0, 14 - old_budget)
            assert "measured advice takes precedence" in request["instructions"]
        else:
            assert "measured_advisory_max_chars" not in row["target"]
            assert "measured_required_reduction_chars" not in row["target"]
    assert batch.read_json(folder / "full_plan.json") == plan


@pytest.mark.parametrize("number,target,candidate,expected", [
    (1, {"max_chars": 7}, "  «Да»  … ", {"normalized_chars": 8, "advisory_max_chars": 7, "source": "nominal", "exceeds_advisory": True}),
    (2, {"max_chars": 100, "measured_advisory_max_chars": 7}, "  «Да»  … ", {"normalized_chars": 8, "advisory_max_chars": 7, "source": "measured", "exceeds_advisory": True}),
    (2, {"max_chars": 100, "measured_advisory_max_chars": 8}, "  «Да»  … ", {"normalized_chars": 8, "advisory_max_chars": 8, "source": "measured", "exceeds_advisory": False}),
    (2, {"max_chars": 100}, "Да.", {"normalized_chars": 3, "advisory_max_chars": None, "source": "unavailable", "exceeds_advisory": None}),
    (1, {"max_chars": 7}, None, {"normalized_chars": None, "advisory_max_chars": 7, "source": "nominal", "exceeds_advisory": None}),
    (2, {"max_chars": 100}, None, {"normalized_chars": None, "advisory_max_chars": None, "source": "unavailable", "exceeds_advisory": None}),
])
def test_candidate_advisory_units_precedence_and_null(
    number: int, target: dict[str, Any], candidate: str | None, expected: dict[str, Any],
) -> None:
    before = copy.deepcopy(target)
    assert batch.candidate_advisory_check(candidate, target, number) == expected
    assert target == before


@pytest.mark.parametrize("candidate,over", [("Достаточно длинная реплика, но короче исходной.", True),
                                           ("Коротко.", False), (None, None)])
def test_review_warning_nonblocking_idempotent_and_not_measured_readiness(
    workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]], candidate: str | None, over: bool | None,
) -> None:
    folder, _, _ = workspace
    final = response(folder, {"cue-1": candidate})
    proposals = copy.deepcopy(final)
    proposals["schema"] = "pipeline-batch-proposals-v1"
    del proposals["assembly_review"]
    for row in proposals["entries"]:
        del row["review"]
    path = folder.parent / "writer.json"
    batch.save_json(path, proposals)
    writer_before = path.read_bytes()
    batch.review_input(folder, path)
    before = (folder / "reviewer_input_round1.json").read_bytes()
    state_before = (folder / "state.json").read_bytes()
    batch.review_input(folder, path)
    assert (folder / "reviewer_input_round1.json").read_bytes() == before
    assert (folder / "state.json").read_bytes() == state_before
    row = batch.read_json(folder / "reviewer_input_round1.json")["entries"][0]
    assert row["candidate"] == candidate and row["advisory_check"]["exceeds_advisory"] is over
    assert row["candidate_sha256"] == (text_sha256(candidate) if candidate is not None else None)
    state = ingest(folder, final)
    if candidate is not None:
        assert state["cues"][1]["chosen_text"] == candidate
        assert state["cues"][1]["decisions"][-1]["reason"] == "applied"
    assert not batch.status_summary(state)["ready_for_separate_mix"]
    assert path.read_bytes() == writer_before


@pytest.mark.parametrize("duration", [4.0, None])
def test_measured_review_advice_does_not_block_accepted_repair(
    workspace: tuple[Path, list[dict[str, Any]], dict[str, Any]], duration: float | None,
) -> None:
    folder, _, _ = workspace
    state = batch.load_state(folder)
    cue = state["cues"][1]
    cue["chosen_text"] = "Длинная осмысленная реплика."
    cue["chosen_sha256"] = text_sha256(cue["chosen_text"])
    cue["attempts"] = [{"round": 1, "status": "unresolved" if duration else "normalization_failed",
                        "tts_attempts": 1 if duration else 0,
                        "text_sha256": cue["chosen_sha256"],
                        **({"pretempo": {"duration_sec": duration}} if duration else {})}]
    request = batch.make_request(folder, state, 2, [cue])
    state["requests"]["2"] = request
    batch.save_json(folder / "request_round2.json", request)
    batch.persist(folder, state)
    candidate = "Короткая реплика."
    final = response(folder, {"cue-1": candidate}, number=2)
    proposals = copy.deepcopy(final)
    proposals["schema"] = "pipeline-batch-proposals-v1"
    del proposals["assembly_review"]
    del proposals["entries"][0]["review"]
    path = folder.parent / "writer2.json"
    batch.save_json(path, proposals)
    batch.review_input(folder, path)
    before = (folder / "reviewer_input_round2.json").read_bytes()
    state_before = (folder / "state.json").read_bytes()
    batch.review_input(folder, path)
    assert (folder / "reviewer_input_round2.json").read_bytes() == before
    assert (folder / "state.json").read_bytes() == state_before
    check = batch.read_json(folder / "reviewer_input_round2.json")["entries"][0]["advisory_check"]
    assert check["normalized_chars"] == len(candidate)
    assert check["source"] == ("measured" if duration else "unavailable")
    assert check["exceeds_advisory"] is (True if duration else None)
    state = ingest(folder, final)
    assert state["cues"][1]["chosen_text"] == candidate
    assert state["cues"][1]["decisions"][-1]["reason"] == "applied"
    assert not batch.status_summary(state)["ready_for_separate_mix"]
