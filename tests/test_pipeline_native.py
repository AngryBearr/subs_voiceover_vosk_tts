"""Offline native-text contracts and a mocked full pipeline; no inference."""

from __future__ import annotations

import builtins
import copy
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import pipeline
from utils.pipeline_text import SpeedPolicy, apply_reviewed_text, prepare_text, resolve_speed_policy, text_sha256, json_sha256


def source_items() -> list[dict[str, Any]]:
    return [{"index": 71, "start": 0, "end": 1000,
             "text": ["Это очень длинная исходная реплика с сохранением смысла."],
             "unknown": {"preserve": [1, 2]}},
            {"index": 88, "start": 1200, "end": 1300, "text": ["Да."]}]


def reviewed(candidate: str | None, verdict: str | None = "accept") -> dict[str, Any]:
    artifact = prepare_text(source_items(), SpeedPolicy(), 13.0)
    artifact["entries"] = artifact["entries"][:1]
    row = artifact["entries"][0]
    row["candidate"] = candidate
    row["review"] = None if verdict is None else {
        "verdict": verdict, "text_sha256": text_sha256(candidate) if candidate is not None else None,
        "model": "openai/gpt-6.1-sol", "mode": "native_separate_reviewer",
    }
    return artifact


@pytest.mark.parametrize("cap", [1.3, 1.5])
def test_one_cap_drives_plan_and_analyzer(cap: float) -> None:
    cfg = {"tts": {"base_speech_rate": 1.25, "max_total_speed": cap},
           "analysis": {"max_mismatch_ratio": 1.1}}
    policy = resolve_speed_policy(cfg)
    prepared = prepare_text(source_items(), policy, 13.0)
    analyzed = pipeline.analyze_subtitles_items(source_items(), max_mismatch_ratio=policy.max_total_speed)
    assert prepared["policy"]["max_total_speed"] == cap
    assert prepared["entries"][0]["target"]["max_chars"] == int(13 * cap)
    assert analyzed["items"][0]["analysis"]["max_chars"] == int(13 * cap)


@pytest.mark.parametrize("base,cap", [(float("nan"), 1.5), (float("inf"), 1.5), (0, 1.5),
                                      (1.5, 1.3), (1.25, float("nan")), (1.25, float("inf")),
                                      (1.25, 1.6), (1, 0.9)])
def test_invalid_speed_rejected(base: float, cap: float) -> None:
    with pytest.raises(ValueError):
        SpeedPolicy(base, cap)


@pytest.mark.parametrize("candidate,verdict,reason", [
    ("Краткая реплика.", "accept", "applied"),
    ("Это длиннее номинального лимита, но короче оригинала.", "accept", "applied"),
    ("Кратко.", "revise", "revise"),
    (None, "refusal", "no_candidate"),
    ("Кратко.", None, "not_reviewed"),
    ("Это " * 25, "accept", "not_shorter"),
    ("24 человека.", "accept", "new_numeric_representation"),
])
def test_semantic_accept_only_and_original_fallbacks(candidate: str | None, verdict: str | None, reason: str) -> None:
    items = source_items()
    frozen = copy.deepcopy(items)
    chosen, report = apply_reviewed_text(items, reviewed(candidate, verdict), SpeedPolicy(), 13.0)
    assert items == frozen
    assert report[0]["reason"] == reason
    assert chosen[0]["text"] == (candidate if reason == "applied" else frozen[0]["text"])
    assert {key: value for key, value in chosen[0].items() if key != "text"} == {
        key: value for key, value in frozen[0].items() if key != "text"}
    assert chosen[1] == frozen[1]
    assert report[1]["reason"] == "not_supplied"


@pytest.mark.parametrize("tamper", ["duplicate", "unknown", "source", "original", "candidate", "policy", "experiment"])
def test_tampering_is_fatal(tamper: str) -> None:
    artifact = reviewed("Кратко.")
    row = artifact["entries"][0]
    if tamper == "duplicate":
        artifact["entries"].append(copy.deepcopy(row))
    elif tamper == "unknown":
        row["cue_id"] = "cue-999"
    elif tamper == "source":
        artifact["source_sha256"] = "bad"
    elif tamper == "original":
        row["original"]["end"] += 1
    elif tamper == "candidate":
        row["candidate"] = "Другая реплика."
    elif tamper == "policy":
        artifact["policy"]["max_total_speed"] = 1.3
    else:
        artifact = {"production_ready": False, "final_results": [row]}
    with pytest.raises(ValueError):
        apply_reviewed_text(source_items(), artifact, SpeedPolicy(), 13.0)


def test_explicit_review_can_replace_nominal_fit_but_invalid_and_unreviewed_stay_original() -> None:
    items = [{"index": 9, "start": 0, "end": 1000, "text": "Да."}]
    artifact = prepare_text(items, SpeedPolicy(), 13.0)
    row = artifact["entries"][0]
    row.update(candidate="Да", review={"verdict": "accept", "text_sha256": text_sha256("Да"),
                                       "model": "Sol", "mode": "native"})
    assert row["target"]["status"] == "fits"
    chosen, report = apply_reviewed_text(items, artifact, SpeedPolicy(), 13.0)
    assert chosen[0]["text"] == "Да" and report[0]["reason"] == "applied"
    row["review"] = None
    chosen, report = apply_reviewed_text(items, artifact, SpeedPolicy(), 13.0)
    assert chosen == items and report[0]["reason"] == "not_reviewed"
    items[0]["end"] = 0
    artifact = prepare_text(items, SpeedPolicy(), 13.0)
    artifact["entries"][0].update(candidate="Да", review={
        "verdict": "accept", "text_sha256": text_sha256("Да"), "model": "Sol", "mode": "native"})
    chosen, report = apply_reviewed_text(items, artifact, SpeedPolicy(), 13.0)
    assert chosen == items and report[0]["reason"] == "source_invalid"


def test_punctuation_only_shortcut_preserved() -> None:
    items = [{"index": 9, "start": 0, "end": 100, "text": "Длинная исходная реплика..."}]
    artifact = prepare_text(items, SpeedPolicy(), 13.0)
    candidate = "Длинная исходная реплика…"
    artifact["entries"][0].update(candidate=candidate, review={
        "verdict": "accept", "text_sha256": text_sha256(candidate), "model": "Sol", "mode": "native"})
    chosen, report = apply_reviewed_text(items, artifact, SpeedPolicy(), 13.0)
    assert chosen == items and report[0]["reason"] == "not_shorter"


def write_srt(path: Path) -> None:
    path.write_text("71\n00:00:00,000 --> 00:00:01,000\nЭто очень длинная исходная реплика с сохранением смысла.\n\n"
                    "88\n00:00:01,200 --> 00:00:01,300\nДа.\n", encoding="utf-8")


def test_prepare_cli_has_no_tts_torch_normalizer_imports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    srt = tmp_path / "source.srt"
    write_srt(srt)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith(("synthesize", "torch", "transformers", "utils.text_normaliz", "utils.audio_")):
            pytest.fail(f"Heavy import during offline prepare: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    pipeline.main([str(srt), "--prepare-text", "--output-dir", str(tmp_path / "prepared"),
                   "--max-total-speed", "1.3"])
    saved = json.loads((tmp_path / "prepared/json/native_text.json").read_text())
    assert len(saved["entries"]) == 2
    assert [row["original"]["index"] for row in saved["entries"]] == [71, 88]
    assert saved["policy"]["max_total_speed"] == 1.3
    assert not (tmp_path / "prepared/segments").exists()
    handoff = pipeline.run_pipeline(srt, output_dir=tmp_path / "no-model", config={"tts": {"model_path": ""}},
                                    prepare_text_only=True)
    assert handoff.exists()


def test_invalid_review_fails_before_import_or_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    srt = tmp_path / "source.srt"
    write_srt(srt)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "experimental"}))
    original_import = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith(("synthesize", "utils.text_normaliz")):
            pytest.fail("Imported before validating handoff")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError, match="pipeline-native-text-v1"):
        pipeline.run_pipeline(srt, output_dir=tmp_path / "out", config={}, reviewed_text_path=bad)
    assert not (tmp_path / "out").exists()


def test_full_cli_reanalyzes_finals_and_records_measured_mock_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from utils import audio_analysis as aa

    srt = tmp_path / "source.srt"
    write_srt(srt)
    prepared_dir = tmp_path / "prepared"
    cfg = {"tts": {"model_path": "not-loaded", "use_analysis_speech_rate": True},
           "reduce_silence": {"enabled": False}, "resample": {"enabled": False}}
    path = pipeline.run_pipeline(srt, output_dir=prepared_dir, config=cfg, prepare_text_only=True)
    artifact = json.loads(path.read_text())
    row = artifact["entries"][0]
    row["candidate"] = "Кратко."
    row["review"] = {"verdict": "accept", "text_sha256": text_sha256("Кратко."),
                     "model": "Sol", "mode": "native_separate_reviewer"}
    path.write_text(json.dumps(artifact))
    calls: list[dict[str, Any]] = []

    def fake_tts(**kwargs: Any) -> None:
        calls.append(kwargs)
        analyzed = json.loads(Path(kwargs["json_path"]).read_text())
        assert analyzed[0]["text"] == "Кратко."
        assert analyzed[0]["analysis"]["estimated_sec"] == len("Кратко.") / 13
        assert kwargs["default_speech_rate"] == 1.25
        assert kwargs["use_analysis_speech_rate"] is False
        (Path(kwargs["output_folder"]) / "71.wav").write_bytes(b"mock audio 71")
        (Path(kwargs["output_folder"]) / "88.wav").write_bytes(b"mock audio 88")

    normalizer = types.ModuleType("utils.text_normalization")
    normalizer.normalize_subtitles_items = lambda items, **kwargs: items  # type: ignore[attr-defined]
    batch = types.ModuleType("synthesize.synthesize_batch")
    batch.synthesize_json_lines = fake_tts  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils.text_normalization", normalizer)
    monkeypatch.setitem(sys.modules, "synthesize.synthesize_batch", batch)
    monkeypatch.setattr(aa, "_wav_duration_sec", lambda wav: 1.05 if wav.name == "71.wav" else 0.13 if wav.name == "88.wav" else 1.0)
    monkeypatch.setattr(aa, "_ffmpeg_available", lambda: True)

    def fake_tempo(src: Path, dst: Path, **kwargs: Any) -> None:
        assert kwargs["speed_factor"] == 1.05
        dst.write_bytes(b"mock post tempo")

    monkeypatch.setattr(aa, "_speedup_with_ffmpeg", fake_tempo)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    pipeline.main([str(srt), "--config", str(config_path), "--output-dir", str(tmp_path / "audio"),
                   "--reviewed-text", str(path)])
    report = json.loads((tmp_path / "audio/json/audio_timing.json").read_text())
    assert [r["status"] for r in report["segments"]] == ["accelerated_fits", "unresolved"]
    assert report["failed_count"] == 1
    original = json.loads((tmp_path / "audio/json/subs.json").read_text())
    assert original[0]["text"] == row["original"]["text"] and "analysis" not in original[0]
    with pytest.raises(ValueError, match="not empty"):
        pipeline.run_pipeline(srt, output_dir=tmp_path / "audio", config=cfg, reviewed_text_path=path)
    assert len(calls) == 1


def test_nominal_fit_measured_overrun_can_use_reviewed_shorter_text_with_same_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from utils import audio_analysis as aa

    srt = tmp_path / "source.srt"
    srt.write_text("9\n00:00:00,000 --> 00:00:01,000\nДа.\n", encoding="utf-8")
    cfg = {"tts": {"model_path": "not-loaded"}, "reduce_silence": {"enabled": False},
           "resample": {"enabled": False}}
    normalizer = types.ModuleType("utils.text_normalization")
    normalizer.normalize_subtitles_items = lambda items, **kwargs: items  # type: ignore[attr-defined]
    batch = types.ModuleType("synthesize.synthesize_batch")

    def fake_tts(**kwargs: Any) -> None:
        items = json.loads(Path(kwargs["json_path"]).read_text())
        assert not items[0]["analysis"]["is_critical"]
        assert kwargs["default_speech_rate"] == 1.25 and not kwargs["use_analysis_speech_rate"]
        is_original = items[0]["text"] == ["Да."]
        (Path(kwargs["output_folder"]) / "9.wav").write_bytes(b"original" if is_original else b"candidate")

    batch.synthesize_json_lines = fake_tts  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils.text_normalization", normalizer)
    monkeypatch.setitem(sys.modules, "synthesize.synthesize_batch", batch)
    monkeypatch.setattr(aa, "_wav_duration_sec", lambda path: 1.3 if path.read_bytes() == b"original" else 0.8)
    monkeypatch.setattr(aa, "_speedup_with_ffmpeg", lambda *args, **kwargs: pytest.fail("No tempo needed/allowed in this case"))
    pipeline.run_pipeline(srt, output_dir=tmp_path / "original-audio", config=cfg)
    original_report = json.loads((tmp_path / "original-audio/json/audio_timing.json").read_text())
    assert original_report["segments"][0]["status"] == "unresolved"
    items = pipeline.parse_srt_file(str(srt))
    artifact = prepare_text(items, resolve_speed_policy(cfg), 13.0)
    assert artifact["entries"][0]["target"]["status"] == "fits"
    artifact["entries"][0].update(candidate="Да", review={"verdict": "accept",
        "text_sha256": text_sha256("Да"), "model": "Sol", "mode": "native_separate_reviewer"})
    reviewed_path = tmp_path / "reviewed.json"
    pipeline._save_json(reviewed_path, artifact)
    pipeline.run_pipeline(srt, output_dir=tmp_path / "reviewed-audio", config=cfg, reviewed_text_path=reviewed_path)
    reviewed_report = json.loads((tmp_path / "reviewed-audio/json/audio_timing.json").read_text())
    assert reviewed_report["segments"][0]["status"] == "fits" and reviewed_report["failed_count"] == 0
    assert json.loads((tmp_path / "reviewed-audio/json/subs_chosen.json").read_text())[0]["text"] == "Да"
    assert json.loads((tmp_path / "reviewed-audio/json/subs.json").read_text()) == items
    assert reviewed_report["segments"][0]["max_total_speed"] == original_report["segments"][0]["max_total_speed"] == 1.5


@pytest.mark.parametrize("engine,failure", [("vosk", None), ("edge", None),
                                           ("vosk", "overrun"), ("vosk", "missing"),
                                           ("vosk", "all_missing"), ("vosk", "invalid")])
def test_ui_wires_one_vosk_cap_and_leaves_edge_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str, failure: str | None,
) -> None:
    import inspect
    from utils import audio_analysis as aa

    monkeypatch.setitem(sys.modules, "gradio", types.ModuleType("gradio"))
    from ui import gradio_app as ui
    # This wiring fixture uses mock bytes; real PCM/hash mix gates are tested
    # separately in test_pipeline_silero_pauses.py, not bypassed in production.
    monkeypatch.setattr(pipeline, "validate_complete_mix", lambda *args, **kwargs: None)

    srt = tmp_path / "source.srt"
    write_srt(srt)
    monkeypatch.setattr(ui, "_now_slug", lambda: engine)
    order: list[str] = []
    opts: list[dict[str, Any]] = []

    def synth(**kwargs: Any) -> None:
        order.append("tts")
        if engine == "vosk":
            assert kwargs["default_speech_rate"] == 1.25 and not kwargs["use_analysis_speech_rate"]
            folder = Path(kwargs["output_folder"])
        else:
            folder = kwargs["out_dir"]
            assert kwargs["rate"] == "+0%"
        for idx in (71, 88):
            if failure == "all_missing" or (idx == 71 and failure == "missing"):
                continue
            (folder / f"{idx}.wav").write_bytes(b"mock UI segment")

    for name, function in (("synthesize.synthesize_batch", "synthesize_json_lines"),
                           ("synthesize.edge_tts_batch", "synthesize_json_to_segments_dir_by_gender")):
        module = types.ModuleType(name)
        setattr(module, function, synth)
        monkeypatch.setitem(sys.modules, name, module)
    resampler = types.ModuleType("utils.audio_resample")
    resampler.resample_wav_dir = lambda *args, **kwargs: order.append("resample")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils.audio_resample", resampler)
    mixer = types.ModuleType("utils.audio_mixer")

    def mix(**kwargs: Any) -> Path:
        if failure is not None:
            pytest.fail("Unresolved/invalid/missing Vosk cues must not be mixed or exported")
        assert kwargs["missing_policy"] == ("error" if engine == "vosk" else "skip")
        order.append("mix")
        return kwargs["output_audio"].with_suffix(".wav")

    mixer.mix_segments = mix  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils.audio_mixer", mixer)
    real_check = aa.check_and_speedup_segments

    def check(*args: Any, **kwargs: Any) -> Any:
        order.append("timing")
        opts.append(kwargs)
        return real_check(*args, **kwargs)

    monkeypatch.setattr(aa, "check_and_speedup_segments", check)
    monkeypatch.setattr(aa, "_wav_duration_sec", lambda path: (
        1.5 if failure == "overrun" and path.name == "71.wav" else
        float("nan") if failure == "invalid" and path.name == "71.wav" else 0.05))
    parameters = {key: None for key in inspect.signature(ui.run_pipeline_from_ui).parameters}
    parameters.update(engine=engine, srt_file_path=str(srt), output_root=str(tmp_path / "ui"),
                      do_analysis=True, avg_chars_per_sec=13, max_mismatch_ratio=1.3,
                      do_runorm=False, do_ruaccent=False, gender_mode="use_gender",
                      vosk_model_dir="not-loaded", vosk_device="cpu", vosk_male_voice_choice="8: male",
                      vosk_female_voice_choice="8: female", base_speech_rate=1.25,
                      use_analysis_speech_rate=True, max_extra_pct=0.2,
                      edge_male_voice="male", edge_female_voice="female", edge_rate="+0%",
                      edge_volume="+0%", edge_pitch="+0Hz", reduce_silence_enabled=False,
                      audio_analysis_enabled=True, resample_enabled=True, resample_target_sr=44100,
                      mix_sample_rate=44100, mix_channels=1, video_duration_sec="",
                      export_wav=False, export_mp3=False)
    # Gradio supplies its input components positionally, in signature order.
    events = list(ui.run_pipeline_from_ui(*parameters.values()))
    if engine == "vosk":
        assert order == ["tts", "resample", "timing"] + ([] if failure else ["mix"])
        assert opts[0]["max_total_speed"] == 1.3 and opts[0]["synthesis_rate"] == 1.25
        if failure:
            report_path = tmp_path / f"ui/run_{engine}/json/audio_timing.json"
            report = json.loads(report_path.read_text())
            failed_count = 2 if failure == "all_missing" else 1
            assert report["failed_count"] == failed_count
            assert report["segments"][0]["status"] == {"overrun": "unresolved", "missing": "missing",
                                                       "all_missing": "missing", "invalid": "invalid"}[failure]
            assert f"Not exported: {failed_count} unresolved/invalid/missing" in events[-1][0]
            assert "segments retained=" in events[-1][0]
            assert "Export complete" not in events[-1][0] and "Done" not in events[-1][0]
            assert events[-1][1] == [str(report_path)]
            assert list((tmp_path / f"ui/run_{engine}/mix").iterdir()) == []
            segments = tmp_path / f"ui/run_{engine}/segments"
            if failure == "all_missing":
                assert list(segments.iterdir()) == []
            else:
                assert (segments / "88.wav").read_bytes() == b"mock UI segment"
        else:
            assert "unresolved/invalid/missing=0" in events[-1][0]
        assert events[-1][1][0].endswith("audio_timing.json")
    else:
        assert order == ["tts", "timing", "resample", "mix"]
        assert "max_total_speed" not in opts[0]


def test_saved_episode_offline_preparation(tmp_path: Path) -> None:
    """Optional local evidence from frozen source items, never shortened exports."""
    root = Path(__file__).resolve().parents[1]
    source_path = root / "output/openrouter_episode_20260930_clean/run_manifest.json"
    if not source_path.exists():
        pytest.skip("Historical source snapshot is not part of a fresh checkout")
    frozen_bytes = source_path.read_bytes()
    items = json.loads(frozen_bytes)["contract"]["source"]["items"]
    assert len(items) == 1523
    prepared = prepare_text(items, SpeedPolicy(), 13.0)
    lower = prepare_text(items, SpeedPolicy(max_total_speed=1.3), 13.0)
    assert [row["original"] for row in prepared["entries"]] == items
    assert prepared["source_sha256"] == json_sha256(items)
    assert len({row["cue_id"] for row in prepared["entries"]}) == 1523
    assert all(row["target"]["max_chars"] >= old["target"]["max_chars"]
               for row, old in zip(prepared["entries"], lower["entries"]))
    assert source_path.read_bytes() == frozen_bytes
    folder = tmp_path / "offline-preparation"
    pipeline._save_json(folder / "native_text_1523.json", prepared)
    pipeline._save_json(folder / "offline_evidence.json", {
        "kind": "offline_saved_source_preparation_not_model_or_audio_evidence",
        "source": str(source_path.relative_to(root)) + "#/contract/source/items",
        "source_sha256": prepared["source_sha256"], "source_unchanged": True,
        "cues": len(items), "unique_ids": 1523, "full_original_coverage": True,
        "model_calls": 0, "tts_calls": 0, "audio_encoding_calls": 0,
        "historical_outputs_modified": False,
        "budgets": {str(artifact["policy"]["max_total_speed"]): {
            "nominal_cps": 13, "character_target_sum": sum(row["target"]["max_chars"] for row in artifact["entries"]),
            "predicted_nonfits": sum(row["target"]["status"] != "fits" for row in artifact["entries"]),
        } for artifact in (lower, prepared)},
        "limitation": "Fresh nominal budgets, not the historical calibrated caps or comparable quality/audio-fit evidence",
    })
