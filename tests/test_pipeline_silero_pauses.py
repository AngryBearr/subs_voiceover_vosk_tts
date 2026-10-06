"""Fixed-policy PCM and pipeline checks; saved receipts, never actual inference."""
from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import pipeline
from utils import audio_analysis as aa, pipeline_batch as batch, silero_pauses as sp
from utils.pipeline_text import text_sha256


def pcm_fixture() -> np.ndarray:
    pcm = np.zeros((16000, 1), dtype=np.int16)
    pcm[1600:4800] = 12000
    pcm[11200:14400] = -12000
    return pcm


def fake_processor(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    def init(self: sp.PauseProcessor) -> None:
        self.identity = {"policy": sp.POLICY}

    def detect(self: sp.PauseProcessor, audio: np.ndarray) -> list[dict[str, int]]:
        calls.append(len(audio))
        return [{"start": 1600, "end": 4800}, {"start": 11200, "end": 14400}]

    monkeypatch.setattr(sp.PauseProcessor, "__init__", init)
    monkeypatch.setattr(sp.PauseProcessor, "detect", detect)
    return calls


def test_guards_speech_edges_and_retained_pcm_bit_exact() -> None:
    pcm = pcm_fixture()
    cuts, speech, agreed = sp.agreed_cuts(pcm, 16000,
        [{"start": 1600, "end": 4800}, {"start": 11200, "end": 14400}], len(pcm))
    assert agreed == [(5440, 10880)]  # 40 ms after / 20 ms before.
    assert cuts == [(6640, 9680)]  # Keep 150 ms split equally on both sides.
    after = sp.cut_pcm(pcm, cuts)
    assert len(after) == 12960
    assert np.array_equal(after[:6640], pcm[:6640])
    assert np.array_equal(after[6640:], pcm[9680:])
    assert np.array_equal(after[:1600], pcm[:1600])
    assert np.array_equal(after[-1600:], pcm[-1600:])
    for a, b in speech:
        shift = sum(y - x for x, y in cuts if y <= a)
        assert np.array_equal(after[a - shift:b - shift], pcm[a:b])


@pytest.mark.parametrize("kind", ["no_activity", "no_quiet", "no_speech", "all_speech", "fragmented"])
def test_no_cut_cases(kind: str) -> None:
    pcm = pcm_fixture()
    spans = [{"start": 1600, "end": 4800}, {"start": 11200, "end": 14400}]
    if kind == "no_activity":
        pcm[:] = 0
    elif kind == "no_quiet":
        pcm[:] = 1000
    elif kind == "no_speech":
        spans = []
    elif kind == "all_speech":
        spans = [{"start": 0, "end": len(pcm)}]
    else:
        spans = [{"start": 0, "end": 6000}, {"start": 7000, "end": 8000},
                 {"start": 9000, "end": 16000}]
    cuts, _, _ = sp.agreed_cuts(pcm, 16000, spans, len(pcm))
    assert not cuts and np.array_equal(sp.cut_pcm(pcm, cuts), pcm)


def test_mapping_rounding_union_and_clipped_end() -> None:
    assert sp.map_speech([{"start": 1, "end": 2}, {"start": 2, "end": 4}], 44100, 11, 4) == [(2, 11)]
    assert sp.map_speech([{"start": 0, "end": 16000}], 44100, 44100, 16001) == [(0, 44100)]


@pytest.mark.parametrize("spans", [[{"start": -1, "end": 1}], [{"start": 1, "end": 1}],
                                   [{"start": 0, "end": 16001}], [{"start": 0.5, "end": 1}],
                                   [{"start": 2, "end": 4}, {"start": 3, "end": 5}]])
def test_invalid_spans_rejected(spans: list[dict[str, int]]) -> None:
    with pytest.raises(ValueError):
        sp.map_speech(spans, 44100, 44100, 16000)


@pytest.mark.parametrize("pcm,sr", [(np.zeros((0, 1), dtype=np.int16), 16000),
                                    (np.zeros((1, 0), dtype=np.int16), 16000),
                                    (np.zeros(1, dtype=np.int16), 16000),
                                    (np.zeros((1, 1), dtype=np.float32), 16000),
                                    (np.zeros((1, 1), dtype=np.int16), 0)])
def test_invalid_pcm_rejected(pcm: np.ndarray, sr: int) -> None:
    with pytest.raises(ValueError):
        sp.detector_audio(pcm, sr)


def test_stereo_energy_any_channel_but_detector_rejects_downmix() -> None:
    pcm = np.repeat(pcm_fixture(), 2, axis=1)
    pcm[:, 1] *= -1
    assert sp.quiet_cores(pcm, 16000) == [(5440, 10880)]
    pcm[7000:7200, 1] = 10000
    assert len(sp.quiet_cores(pcm, 16000)) == 2
    with pytest.raises(ValueError, match="mono"):
        sp.detector_audio(pcm, 16000)
    after = sp.cut_pcm(pcm, [(8000, 8500)])
    assert np.array_equal(after[:8000], pcm[:8000]) and np.array_equal(after[8000:], pcm[8500:])


def test_exclusive_one_pass_atomic_output_and_failed_intent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_processor(monkeypatch)
    source, output = tmp_path / "pretempo.wav", tmp_path / "pauses.wav"
    sf.write(source, pcm_fixture(), 16000, subtype="PCM_16")
    original_hash = sp.digest(source)
    processor = sp.PauseProcessor()
    receipt = processor.process(source, output)
    assert receipt["after_frames"] == 12960 and receipt["saving_ms"] == 190
    assert receipt["output_sha256"] == sp.digest(output) and sp.digest(source) == original_hash
    assert calls == [16000]
    for a, b in [(source, source), (source, output), (output, tmp_path / "twice.wav")]:
        with pytest.raises(ValueError):
            processor.process(a, b)
    assert calls == [16000]
    monkeypatch.setattr(processor, "detect", lambda audio: (_ for _ in ()).throw(RuntimeError("offline failure")))
    failed = tmp_path / "failed.wav"
    with pytest.raises(RuntimeError, match="offline failure"):
        processor.process(source, failed)
    assert not failed.exists()
    assert json.loads(failed.with_suffix(".pauses.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="already"):
        processor.process(source, failed)


def test_no_effect_output_is_source_byte_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_processor(monkeypatch)
    processor = sp.PauseProcessor()
    monkeypatch.setattr(processor, "detect", lambda audio: [])
    source, output = tmp_path / "before.wav", tmp_path / "after.wav"
    sf.write(source, pcm_fixture(), 16000, subtype="PCM_16")
    assert processor.process(source, output)["removed_frames"] == 0
    assert source.read_bytes() == output.read_bytes()


def test_segment_stage_refuses_repeat_and_retains_pretempo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_processor(monkeypatch)
    segments, stage = tmp_path / "segments", tmp_path / "pause_stage"
    segments.mkdir()
    source = segments / "1.wav"
    sf.write(source, pcm_fixture(), 16000, subtype="PCM_16")
    original = source.read_bytes()
    items = [{"index": 1}, {"index": 2}]
    receipts = sp.process_segments(items, segments, stage)
    assert len(receipts) == 1 and calls == [16000]
    assert (stage / "1.pretempo.wav").read_bytes() == original
    assert source.read_bytes() == (stage / "1.pauses.wav").read_bytes()
    with pytest.raises(ValueError, match="already"):
        sp.process_segments(items, segments, stage)
    assert calls == [16000]


def test_loader_unavailable_fails_clearly_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sp.metadata, "version", lambda name: (_ for _ in ()).throw(sp.metadata.PackageNotFoundError(name)))
    with pytest.raises(RuntimeError, match="no automatic downloads"):
        sp.PauseProcessor()


def test_loader_rejects_different_package_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sp.metadata, "version", lambda name: "6.2.4")
    with pytest.raises(RuntimeError, match="Expected silero-vad==6.2.3"):
        sp.PauseProcessor()


def test_model_reused_reset_each_file(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    model = types.SimpleNamespace(reset_states=lambda: events.append("reset"))
    module = types.ModuleType("silero_vad")
    module.get_speech_timestamps = lambda audio, supplied, **kwargs: events.append("detect") or []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "silero_vad", module)
    processor = object.__new__(sp.PauseProcessor)
    processor.model = model
    for _ in range(2):
        assert processor.detect(np.zeros(512, dtype=np.float32)) == []
    assert events == ["reset", "detect", "reset", "detect"]


def test_saved_receipts_reproduce_pcm_without_model_calls() -> None:
    folder = Path("output/pipeline_silero_pauses_20261005")
    if not folder.exists():
        pytest.skip("Saved listening fixtures absent in fresh checkout")
    receipts = {r["cue_id"]: r for r in json.loads((folder / "vad_receipts.json").read_text())}
    expected = [298891, 177759, 165770]
    for preview, frames in zip(json.loads((folder / "previews.json").read_text()), expected):
        pcm, sr = sf.read(preview["source_path"], dtype="int16", always_2d=True)
        receipt = receipts[preview["cue_id"]]
        cuts, speech, agreed = sp.agreed_cuts(pcm, sr, receipt["speech_16k_samples"], receipt["detector_frames"])
        assert [list(span) for span in speech] == receipt["speech_source_samples"]
        assert [list(span) for span in agreed] == receipt["agreed_core_samples"]
        assert [list(span) for span in cuts] == preview["cut_intervals_source_samples"]
        approved, _ = sf.read(preview["after_path"], dtype="int16", always_2d=True)
        assert len(approved) == frames and np.array_equal(sp.cut_pcm(pcm, cuts), approved)


@pytest.mark.parametrize("enabled", [False, True])
def test_batch_processing_order_and_actual_measurement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    fake_processor(monkeypatch)
    from utils import reduce_silence, audio_resample
    order: list[str] = []
    monkeypatch.setattr(reduce_silence, "reduce_silence_file", lambda *args, **kwargs: order.append("legacy"))
    monkeypatch.setattr(audio_resample, "resample_wav_file", lambda *args, **kwargs: order.append("resample"))
    detect = sp.PauseProcessor.detect

    def vad(self: sp.PauseProcessor, audio: np.ndarray) -> list[dict[str, int]]:
        order.append("vad")
        return detect(self, audio)

    monkeypatch.setattr(sp.PauseProcessor, "detect", vad)
    raw, pretempo, final = [tmp_path / name for name in ("raw.wav", "pretempo.wav", "final.wav")]
    sf.write(raw, pcm_fixture(), 16000, subtype="PCM_16")
    original = raw.read_bytes()
    config = {"tts": {"base_speech_rate": 1.25, "max_total_speed": 1.5}, "silero_pauses": {"enabled": enabled}}

    def tempo(source: Path, output: Path, **kwargs: Any) -> tuple[float, float]:
        order.append("tempo")
        assert source.name == ("pauses.wav" if enabled else "pretempo.wav")
        sf.write(output, np.zeros((12000, 1), dtype=np.int16), 16000, subtype="PCM_16")
        return sf.info(source).duration / 0.75, 0.75

    monkeypatch.setattr(aa, "render_bounded_tempo", tempo)
    attempt: dict[str, Any] = {}
    batch.measure_audio(config, raw, pretempo, final, 0.75, attempt)
    assert raw.read_bytes() == original == pretempo.read_bytes()
    if enabled:
        assert order == ["legacy", "resample", "vad", "tempo"]
        assert attempt["required_factor"] == (12960 / 16000) / 0.75
        assert attempt["pretempo"]["frames"] == 16000 and attempt["pauses"]["frames"] == 12960
        assert attempt["status"] == "accelerated_fits"
    else:
        assert order == ["legacy", "resample"]  # Original 1 sec cannot fit remaining 1.2x.
        assert attempt["status"] == "unresolved" and "pauses" not in attempt
    with pytest.raises(ValueError, match="already"):
        batch.measure_audio(config, raw, pretempo, final, 0.75, attempt)


@pytest.mark.parametrize("profile", [{"voice": 0}, {"voice": 8, "voice_female": 0},
                                    {"voice": 8, "model_path": "another-model"}])
def test_other_profiles_not_silently_enabled(profile: dict[str, Any]) -> None:
    settings = {"model_path": "models/vosk-model-tts-ru-0.10-multi", **profile}
    with pytest.raises(ValueError, match="voice 8 only"):
        sp.validate_profile({"tts": settings, "silero_pauses": {"enabled": True}})
    sp.validate_profile({"tts": settings, "silero_pauses": {"enabled": False}})


def test_old_config_does_not_inherit_new_policy_or_mix(tmp_path: Path) -> None:
    old = tmp_path / "old.json"
    old.write_text('{"tts":{"voice":0}}')
    cfg = pipeline.load_pipeline_config(old)
    assert cfg["silero_pauses"]["enabled"] is False and cfg["mix"]["enabled"] is False
    assert cfg["silero_edges"]["enabled"] is False
    chosen = pipeline.load_pipeline_config(Path("pipeline_config_vosk.json"))
    assert chosen["silero_pauses"]["enabled"] and chosen["mix"]["enabled"]
    assert chosen["silero_edges"]["enabled"]
    old.write_text('{"tts":{"voice":8},"silero_pauses":{"enabled":true}}')
    internal_only = pipeline.load_pipeline_config(old)
    assert internal_only["silero_pauses"]["enabled"] and not internal_only["silero_edges"]["enabled"]


def mix_fixture(tmp_path: Path) -> tuple[list[dict[str, Any]], Path, Path]:
    items = [{"index": 1, "start": 200, "end": 500, "text": "Да."},
             {"index": 2, "start": 1200, "end": 1500, "text": "Нет."}]
    folder = tmp_path / "segments"
    folder.mkdir()
    for idx in (1, 2):
        sf.write(folder / f"{idx}.wav", np.full((1600, 1), idx * 1000, dtype=np.int16), 16000, subtype="PCM_16")
    report = tmp_path / "timing.json"
    aa.check_and_speedup_segments(items, folder, synthesis_rate=1.25, max_total_speed=1.5, report_path=report)
    return items, folder, report


@pytest.mark.parametrize("failure", ["missing", "stale", "unfit", "incomplete", "text", "postprocess", "overlap"])
def test_mix_blocks_missing_unfit_stale_and_incomplete(tmp_path: Path, failure: str) -> None:
    items, folder, report = mix_fixture(tmp_path)
    source = copy.deepcopy(items)
    if failure == "missing":
        (folder / "2.wav").unlink()
    elif failure in {"stale", "unfit"}:
        sf.write(folder / "2.wav", np.ones((1600 if failure == "stale" else 32000, 1), dtype=np.int16),
                 16000, subtype="PCM_16")
    elif failure == "incomplete":
        source.append({"index": 3, "start": 2000, "end": 2300, "text": "Да."})
    elif failure == "text":
        items[0]["text"] = "Другой текст."
    elif failure == "postprocess":
        receipt = report.parent / "pauses.json"
        receipt.write_text("[]")
    else:
        items[1]["start"] = 250
        source = copy.deepcopy(items)
        report.unlink()
        aa.check_and_speedup_segments(items, folder, synthesis_rate=1.25, max_total_speed=1.5, report_path=report)
    with pytest.raises(ValueError, match="blocked"):
        pipeline.validate_complete_mix(items, source, folder, report,
                                       pauses_path=report.parent / "pauses.json" if failure == "postprocess" else None)


def test_valid_synthetic_full_timeline_assembly(tmp_path: Path) -> None:
    from utils.audio_mixer import mix_segments
    items, folder, report = mix_fixture(tmp_path)
    pipeline.validate_complete_mix(items, items, folder, report)
    source = tmp_path / "source.json"
    source.write_text(json.dumps(items))
    output = mix_segments(source, folder, tmp_path / "full", sample_rate=16000, channels=1,
                          missing_policy="error", transcode_to=None)
    data, sr = sf.read(output, dtype="int16", always_2d=True)
    assert sr == 16000 and len(data) == 24000
    assert np.count_nonzero(data[:3200]) == 0
    assert np.all(data[3200:4800] == 1000)
    assert np.count_nonzero(data[4800:19200]) == 0
    assert np.all(data[19200:20800] == 2000)
    assert np.count_nonzero(data[20800:]) == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_real_pcm_wiring_and_full_mix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    fake_processor(monkeypatch)
    monkeypatch.setattr(sp, "runtime_identity", lambda: {"policy": sp.POLICY})
    order: list[str] = []
    from utils import audio_resample, reduce_silence
    monkeypatch.setattr(reduce_silence, "reduce_silence_in_dir", lambda *args, **kwargs: order.append("legacy"))
    monkeypatch.setattr(audio_resample, "resample_wav_dir", lambda *args, **kwargs: order.append("resample"))
    normalizer = types.ModuleType("utils.text_normalization")
    normalizer.normalize_subtitles_items = lambda items, **kwargs: items  # type: ignore[attr-defined]
    synth = types.ModuleType("synthesize.synthesize_batch")

    def tts(**kwargs: Any) -> None:
        order.append("tts")
        sf.write(Path(kwargs["output_folder"]) / "1.wav", pcm_fixture(), 16000, subtype="PCM_16")

    synth.synthesize_json_lines = tts  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils.text_normalization", normalizer)
    monkeypatch.setitem(sys.modules, "synthesize.synthesize_batch", synth)
    detect = sp.PauseProcessor.detect

    def vad(self: sp.PauseProcessor, audio: np.ndarray) -> list[dict[str, int]]:
        order.append("vad")
        return detect(self, audio)

    monkeypatch.setattr(sp.PauseProcessor, "detect", vad)
    real_check = aa.check_and_speedup_segments

    def timing(*args: Any, **kwargs: Any) -> Any:
        order.append("timing")
        return real_check(*args, **kwargs)

    monkeypatch.setattr(aa, "check_and_speedup_segments", timing)
    srt = tmp_path / "source.srt"
    srt.write_text("1\n00:00:00,200 --> 00:00:01,500\nДа.\n")
    config = {"tts": {"voice": 8, "model_path": "models/vosk-model-tts-ru-0.10-multi"},
              "silero_pauses": {"enabled": enabled}, "mix": {"enabled": True, "sample_rate": 16000}}
    out = tmp_path / "run"
    pipeline.run_pipeline(srt, output_dir=out, config=config)
    assert order == ["tts", "legacy", "resample"] + (["vad"] if enabled else []) + ["timing"]
    result = json.loads((out / "json/audio_timing.json").read_text())["segments"][0]
    assert result["audio_duration_sec"] == (12960 if enabled else 16000) / 16000
    assert (out / "mix/voiceover.wav").exists()
    if enabled:
        assert sf.info(out / "pause_stage/1.pretempo.wav").frames == 16000
        assert sf.info(out / "pause_stage/1.pauses.wav").frames == 12960
        (out / "pause_stage/1.pauses.wav").write_bytes(b"changed")
        items = json.loads((out / "json/subs_normalized.json").read_text())
        with pytest.raises(ValueError, match="provenance"):
            pipeline.validate_complete_mix(items, pipeline.parse_srt_file(str(srt)), out / "segments",
                out / "json/audio_timing.json", pauses_path=out / "json/silero_pauses.json")


@pytest.mark.parametrize("failure", ["missing", "unfit"])
def test_cli_configured_complete_mix_blocks_failed_cues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       failure: str) -> None:
    normalizer = types.ModuleType("utils.text_normalization")
    normalizer.normalize_subtitles_items = lambda items, **kwargs: items  # type: ignore[attr-defined]
    synth = types.ModuleType("synthesize.synthesize_batch")

    def tts(**kwargs: Any) -> None:
        if failure == "unfit":
            sf.write(Path(kwargs["output_folder"]) / "1.wav", np.full((32000, 1), 1000, dtype=np.int16),
                     16000, subtype="PCM_16")

    synth.synthesize_json_lines = tts  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils.text_normalization", normalizer)
    monkeypatch.setitem(sys.modules, "synthesize.synthesize_batch", synth)
    mixer = types.ModuleType("utils.audio_mixer")
    mixer.mix_segments = lambda *args, **kwargs: pytest.fail("Failed cue must block full export")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils.audio_mixer", mixer)
    srt = tmp_path / "source.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nДа.\n")
    out = tmp_path / "run"
    pipeline.run_pipeline(srt, output_dir=out, config={"tts": {"model_path": "not-loaded"},
        "reduce_silence": {"enabled": False}, "resample": {"enabled": False}, "mix": {"enabled": True}})
    report = json.loads((out / "json/audio_timing.json").read_text())
    assert report["failed_count"] == 1
    assert report["segments"][0]["status"] == ("missing" if failure == "missing" else "unresolved")
    assert not (out / "mix").exists()


@pytest.mark.parametrize("enabled,engine", [(True, "vosk"), (False, "vosk"), (True, "edge")])
def test_ui_order_enable_disable_and_edge_exclusion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  enabled: bool, engine: str) -> None:
    import inspect
    fake_processor(monkeypatch)
    monkeypatch.setattr(sp, "runtime_identity", lambda: {"policy": sp.POLICY})
    monkeypatch.setitem(sys.modules, "gradio", types.ModuleType("gradio"))
    from ui import gradio_app as ui
    from utils import audio_resample, reduce_silence
    order: list[str] = []
    monkeypatch.setattr(reduce_silence, "reduce_silence_in_dir", lambda *args, **kwargs: order.append("legacy"))
    monkeypatch.setattr(audio_resample, "resample_wav_dir", lambda *args, **kwargs: order.append("resample"))
    for name, function in (("synthesize.synthesize_batch", "synthesize_json_lines"),
                           ("synthesize.edge_tts_batch", "synthesize_json_to_segments_dir_by_gender")):
        module = types.ModuleType(name)

        def tts(**kwargs: Any) -> None:
            order.append("tts")
            folder = Path(kwargs["output_folder"]) if "output_folder" in kwargs else kwargs["out_dir"]
            sf.write(folder / "1.wav", pcm_fixture(), 16000, subtype="PCM_16")

        setattr(module, function, tts)
        monkeypatch.setitem(sys.modules, name, module)
    detect = sp.PauseProcessor.detect

    def vad(self: sp.PauseProcessor, audio: np.ndarray) -> list[dict[str, int]]:
        order.append("vad")
        return detect(self, audio)

    monkeypatch.setattr(sp.PauseProcessor, "detect", vad)
    real_check = aa.check_and_speedup_segments

    def timing(*args: Any, **kwargs: Any) -> Any:
        order.append("timing")
        return real_check(*args, **kwargs)

    monkeypatch.setattr(aa, "check_and_speedup_segments", timing)
    srt = tmp_path / "source.srt"
    srt.write_text("1\n00:00:00,200 --> 00:00:01,500\nДа.\n")
    kwargs = {key: None for key in inspect.signature(ui.run_pipeline_from_ui).parameters}
    kwargs.update(engine=engine, srt_file_path=str(srt), output_root=str(tmp_path / "ui"),
        do_analysis=True, avg_chars_per_sec=13, max_mismatch_ratio=1.5,
        do_runorm=False, do_ruaccent=False, gender_mode="use_gender",
        vosk_model_dir="models/vosk-model-tts-ru-0.10-multi", vosk_device="cpu",
        vosk_male_voice_choice="8: male", vosk_female_voice_choice="8: female", base_speech_rate=1.25,
        edge_male_voice="male", edge_female_voice="female", edge_rate="+0%", edge_volume="+0%", edge_pitch="+0Hz",
        reduce_silence_enabled=True, resample_enabled=True, resample_target_sr=16000,
        audio_analysis_enabled=True, mix_sample_rate=16000, mix_channels=1,
        video_duration_sec="", export_wav=True, export_mp3=False, silero_pauses_enabled=enabled)
    kwargs["silero_edges_enabled"] = enabled if engine == "edge" else False
    events = list(ui.run_pipeline_from_ui(**kwargs))
    assert "Export complete" in events[-1][0]
    assert order == (["tts", "legacy", "resample"] + (["vad"] if enabled else []) + ["timing"]
                     if engine == "vosk" else ["tts", "legacy", "timing", "resample"])


def test_native_runtime_contract_binds_policy_model_and_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "runtime"
    root.mkdir()
    helper = root / "helper.py"
    helper.write_text("policy v1")
    monkeypatch.setattr(batch, "ROOT", root)
    monkeypatch.setattr(batch, "RUNTIME_FILES", ("helper.py",))
    monkeypatch.setattr(batch.importlib.metadata, "distribution", lambda name: (_ for _ in ()).throw(
        batch.importlib.metadata.PackageNotFoundError(name)))
    identity: dict[str, Any] = {"policy": copy.deepcopy(sp.POLICY), "official_code": {}, "asset": "local"}
    monkeypatch.setattr(sp, "runtime_identity", lambda: copy.deepcopy(identity))
    config = {"silero_pauses": {"enabled": True}}
    before = batch.runtime_binding(config)
    identity["policy"]["model_sha256"] = "changed"
    assert batch.runtime_binding(config) != before
    identity["policy"] = copy.deepcopy(sp.POLICY)
    helper.write_text("policy v2")
    assert batch.runtime_binding(config) != before
    assert batch.runtime_binding({})["silero_pauses"] == {"enabled": False}


def test_native_saved_binding_is_not_rebound_when_pause_identity_changes(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    helper = tmp_path / "helper.py"
    helper.write_text("bound helper")
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(batch, "RUNTIME_FILES", ("helper.py",))
    monkeypatch.setattr(batch.importlib.metadata, "distribution", lambda name: (_ for _ in ()).throw(
        batch.importlib.metadata.PackageNotFoundError(name)))
    identity: dict[str, Any] = {"policy": copy.deepcopy(sp.POLICY)}
    monkeypatch.setattr(sp, "runtime_identity", lambda: copy.deepcopy(identity))
    source, folder = tmp_path / "source.json", tmp_path / "batch"
    source.write_text(json.dumps([{"index": 1, "start": 0, "end": 1000, "text": "Да."}]))
    cfg = {"tts": {"model_path": "models/vosk-model-tts-ru-0.10-multi", "voice": 8},
           "silero_pauses": {"enabled": True}}
    batch.prepare(source, folder, cfg, 0, 0)
    before = (folder / "state.json").read_bytes()
    batch.load_state(folder)
    identity["policy"]["model_sha256"] = "changed"
    with pytest.raises(ValueError, match="Runtime/model assets changed"):
        batch.load_state(folder)
    assert (folder / "state.json").read_bytes() == before


def edge_pcm() -> np.ndarray:
    pcm = np.zeros((32000, 1), dtype=np.int16)
    pcm[8000:24000] = 4000
    return pcm


def edge_config() -> dict[str, Any]:
    return {"tts": {"voice": 8, "model_path": "models/vosk-model-tts-ru-0.10-multi"},
            "silero_pauses": {"enabled": True}, "silero_edges": {"enabled": True}}


def edge_detector(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    fake_processor(monkeypatch)

    def detect(self: sp.PauseProcessor, audio: np.ndarray) -> list[dict[str, int]]:
        events.append("vad")
        return [{"start": 8000, "end": 24000}]

    monkeypatch.setattr(sp.PauseProcessor, "detect", detect)
    monkeypatch.setattr(sp, "runtime_identity", lambda: {"policy": sp.POLICY})


def edge_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, text: str | None = "Один два три.",
               window: int = 1200) -> tuple[dict[str, Any], dict[str, Any]]:
    edge_detector(monkeypatch, [])
    raw, pauses, edges = [tmp_path / name for name in ("raw.wav", "pauses.wav", "edges.wav")]
    sf.write(raw, edge_pcm(), 16000, subtype="PCM_16")
    internal = sp.PauseProcessor().process(raw, pauses)
    receipt = sp.process_edges(pauses, edges, internal, available_ms=window, chosen_text=text)
    return internal, receipt


@pytest.mark.parametrize("window", [2000, 2200])
def test_edges_already_fitting_and_exact_equal_byte_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       window: int) -> None:
    internal, edge = edge_stage(tmp_path, monkeypatch, window=window)
    assert edge["decision"] == "already_fits" and edge["removed_frames"] == 0
    assert Path(edge["output_path"]).read_bytes() == Path(internal["output_path"]).read_bytes()
    assert edge["after_frames"] == 32000
    # Plenty of removable edges exist, but production must not even plan them.
    assert sp.edge_plan(edge_pcm(), 16000, [(8000, 24000)], False)["left_samples"] == 7360


@pytest.mark.parametrize("text,short,frames", [("Один два три.", False, 17920),
                                              ("Да? Да.", True, 20000), (None, True, 20000)])
def test_edge_overrun_short_guards_and_preserved_speech(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      text: str | None, short: bool, frames: int) -> None:
    internal, edge = edge_stage(tmp_path, monkeypatch, text=text)
    assert edge["decision"] == "applied" and edge["short_protection"] is short
    assert edge["after_frames"] == frames and edge["new_vad_passes"] == 0
    pcm, _ = sf.read(internal["output_path"], dtype="int16", always_2d=True)
    after, _ = sf.read(edge["output_path"], dtype="int16", always_2d=True)
    left, right = edge["plan"]["left_samples"], edge["plan"]["right_samples"]
    assert np.array_equal(after, pcm[left:len(pcm) - right])
    assert np.array_equal(after[8000 - left:24000 - left], pcm[8000:24000])
    sp.validate_edge_receipt(edge, internal, available_ms=1200, chosen_text=text)
    if short:
        assert edge["after_frames"] * 1000 > 1200 * 16000  # Guards are not relaxed to force fit.
    with pytest.raises(ValueError, match="already"):
        sp.process_edges(Path(edge["source_path"]), Path(edge["output_path"]), internal,
                         available_ms=1200, chosen_text=text)
    with pytest.raises(ValueError, match="already"):
        sp.process_edges(Path(edge["output_path"]), tmp_path / "twice.wav", internal,
                         available_ms=1200, chosen_text=text)


@pytest.mark.parametrize("kind", ["both_uncertain", "left_uncertain", "no_speech", "no_activity", "stereo"])
def test_edge_uncertainty_and_undetected_activity_keep_edges(kind: str) -> None:
    pcm = edge_pcm()
    speech = [(8000, 24000)]
    if kind in {"both_uncertain", "left_uncertain"}:
        pcm[4000:6400] = 40  # Quiet onset proxy (-58.3 dBFS), not real speech evidence.
        if kind == "both_uncertain":
            pcm[26000:28000] = 40
    elif kind == "no_speech":
        speech = []
    elif kind == "no_activity":
        pcm[:] = 0
    else:
        pcm = np.concatenate((np.zeros_like(pcm), pcm), axis=1)
    plan = sp.edge_plan(pcm, 16000, speech, False)
    assert plan["left_samples"] == 0
    assert plan["right_samples"] == (6720 if kind == "left_uncertain" else 0)
    if "uncertain" in kind:
        assert plan["left_uncertain"]
    else:
        assert plan["skip_reason"]


def test_edge_coordinates_minimum_cut_and_punctuation_match_preview() -> None:
    assert sp.protected_after_cuts([(1000, 3000), (8000, 12000)], [(4000, 5000), (6000, 6500)], 16000) == [
        (1000, 3000), (6500, 10500)]
    with pytest.raises(ValueError, match="overlaps"):
        sp.protected_after_cuts([(1000, 3000)], [(2000, 4000)], 16000)
    assert sp.normalized_edge_text("Да? Да. +окей, поехали!") == "Да Да +окей поехали"
    assert sp.edge_samples(40, 22051) == 883
    # VAD/RMS union starts at 80 ms: 40 ms guard leaves exactly a 40 ms cut.
    assert sp.edge_plan(edge_pcm(), 16000, [(1280, 24000)], False)["left_samples"] == 640


def test_short_pretempo_duration_uses_larger_guards_even_for_more_words(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    fake_processor(monkeypatch)
    monkeypatch.setattr(sp.PauseProcessor, "detect", lambda self, audio: [{"start": 4800, "end": 11200}])
    pcm = np.zeros((16000, 1), dtype=np.int16)
    pcm[4800:11200] = 4000
    raw, pauses, edges = [tmp_path / name for name in ("raw.wav", "pauses.wav", "edges.wav")]
    sf.write(raw, pcm, 16000, subtype="PCM_16")
    internal = sp.PauseProcessor().process(raw, pauses)
    edge = sp.process_edges(pauses, edges, internal, available_ms=650, chosen_text="Один два три.")
    assert edge["short_protection"] and edge["after_frames"] == 10400
    assert (edge["plan"]["head_guard_samples"], edge["plan"]["tail_guard_samples"]) == (1600, 2400)


@pytest.mark.parametrize("failure", ["text", "window", "missing", "stale_pcm", "internal"])
def test_edge_receipt_stale_text_window_or_audio_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      failure: str) -> None:
    internal, edge = edge_stage(tmp_path, monkeypatch)
    text, window = "Один два три.", 1200
    if failure == "text":
        text = "Один другой текст."
    elif failure == "window":
        window = 1201
    elif failure == "missing":
        Path(edge["output_path"]).unlink()
    elif failure == "stale_pcm":
        Path(edge["output_path"]).write_bytes(b"changed")
    else:
        internal["before_frames"] += 1
    with pytest.raises(ValueError, match="blocked"):
        sp.validate_edge_receipt(edge, internal, available_ms=window, chosen_text=text)


@pytest.mark.parametrize("entry", ["cli", "ui", "batch"])
def test_edge_dependency_error_precedes_tts_and_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    entry: str) -> None:
    cfg = edge_config()
    cfg["silero_pauses"]["enabled"] = False
    srt = tmp_path / "source.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nДа.\n")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="requires silero_pauses.enabled"):
        if entry == "cli":
            pipeline.run_pipeline(srt, output_dir=out, config=cfg)
        elif entry == "batch":
            batch.prepare(srt, out, cfg, 0, 0)
        else:
            import inspect
            monkeypatch.setitem(sys.modules, "gradio", types.ModuleType("gradio"))
            from ui import gradio_app as ui
            kwargs = {key: None for key in inspect.signature(ui.run_pipeline_from_ui).parameters}
            kwargs.update(engine="vosk", srt_file_path=str(srt), output_root=str(out), base_speech_rate=1.25,
                          max_mismatch_ratio=1.5, silero_pauses_enabled=False, silero_edges_enabled=True,
                          vosk_male_voice_choice="8: male", vosk_female_voice_choice="8: female",
                          vosk_model_dir=cfg["tts"]["model_path"])
            list(ui.run_pipeline_from_ui(**kwargs))
    assert not out.exists()


@pytest.mark.parametrize("flag", ["--no-silero-pauses", "--no-silero-edges"])
def test_cli_pause_disable_switches_do_not_require_hidden_vad(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            flag: str) -> None:
    srt = tmp_path / "source.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nДа.\n")
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(pipeline, "run_pipeline", lambda *args, **kwargs: seen.append(kwargs["config"]) or srt)
    pipeline.main([str(srt), "--config", "pipeline_config_vosk.json", "--prepare-text", flag])
    assert seen[0]["silero_edges"]["enabled"] is False
    assert seen[0]["silero_pauses"]["enabled"] is (flag == "--no-silero-edges")


@pytest.mark.parametrize("window,cap", [(1200, 1.5), (1100, 1.5), (800, 1.5), (1100, 1.3)])
def test_native_order_actual_edge_duration_and_bounded_tempo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           window: int, cap: float) -> None:
    from utils import reduce_silence, audio_resample
    events: list[str] = []
    edge_detector(monkeypatch, events)
    monkeypatch.setattr(reduce_silence, "reduce_silence_file", lambda *args, **kwargs: events.append("legacy"))
    monkeypatch.setattr(audio_resample, "resample_wav_file", lambda *args, **kwargs: events.append("resample"))
    real_edges = sp.process_edges

    def edges(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("edges")
        return real_edges(*args, **kwargs)

    monkeypatch.setattr(sp, "process_edges", edges)
    raw, pretempo, final = [tmp_path / name for name in ("raw.wav", "pretempo.wav", "final.wav")]
    sf.write(raw, edge_pcm(), 16000, subtype="PCM_16")
    original = raw.read_bytes()

    def tempo(source: Path, output: Path, **kwargs: Any) -> tuple[float, float]:
        events.append("tempo")
        assert source.name == "edges.wav" and kwargs["max_factor"] == cap / 1.25
        sf.write(output, np.full((window * 16, 1), 4000, dtype=np.int16), 16000, subtype="PCM_16")
        return sf.info(source).duration / (window / 1000), window / 1000

    monkeypatch.setattr(aa, "render_bounded_tempo", tempo)
    attempt: dict[str, Any] = {}
    config = edge_config()
    config["tts"]["max_total_speed"] = cap
    batch.measure_audio(config, raw, pretempo, final, window / 1000, attempt,
                        chosen_text="Один два три.", available_ms=window)
    assert events == ["legacy", "resample", "vad", "edges"] + (["tempo"] if window == 1100 else [])
    assert raw.read_bytes() == original == pretempo.read_bytes()
    assert attempt["pretempo"]["frames"] == attempt["pauses"]["frames"] == 32000
    assert attempt["edges"]["frames"] == 17920
    assert attempt["required_factor"] == max(1, 1.12 / (window / 1000))
    assert attempt["status"] == {1200: "fits", 1100: "accelerated_fits", 800: "unresolved"}[window]
    assert attempt["effective_total_speed"] <= cap


@pytest.mark.parametrize("entry,gap", [("cli", False), ("ui", False), ("cli", True), ("ui", True)])
def test_entry_edge_order_canonical_gap_and_original_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       entry: str, gap: bool) -> None:
    import inspect
    from utils import reduce_silence, audio_resample
    events: list[str] = []
    edge_detector(monkeypatch, events)
    monkeypatch.setattr(reduce_silence, "reduce_silence_in_dir", lambda *args, **kwargs: events.append("legacy"))
    monkeypatch.setattr(audio_resample, "resample_wav_dir", lambda *args, **kwargs: events.append("resample"))
    real_edges, real_timing = sp.process_edges, aa.check_and_speedup_segments

    def edges(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("edges")
        return real_edges(*args, **kwargs)

    def timing(*args: Any, **kwargs: Any) -> Any:
        events.append("timing")
        return real_timing(*args, **kwargs)

    monkeypatch.setattr(sp, "process_edges", edges)
    monkeypatch.setattr(aa, "check_and_speedup_segments", timing)
    normalizer = types.ModuleType("utils.text_normalization")
    normalizer.normalize_subtitles_items = lambda items, **kwargs: [{**item, "text": "О+дин дв+а тр+и."} for item in items]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils.text_normalization", normalizer)
    synth = types.ModuleType("synthesize.synthesize_batch")

    def tts(**kwargs: Any) -> None:
        events.append("tts")
        for item in json.loads(Path(kwargs["json_path"]).read_text()):
            sf.write(Path(kwargs["output_folder"]) / f"{item['index']}.wav", edge_pcm(), 16000, subtype="PCM_16")

    synth.synthesize_json_lines = tts  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "synthesize.synthesize_batch", synth)
    srt = tmp_path / "source.srt"
    srt.write_text("1\n00:00:00,000 --> " + ("00:00:00,900" if gap else "00:00:01,200") +
        "\nОдин два три.\n" + ("\n2\n00:00:02,500 --> 00:00:05,500\nОдин два три.\n" if gap else ""))
    if entry == "cli":
        out = tmp_path / "run"
        cfg = {**edge_config(), "mix": {"enabled": True, "sample_rate": 16000}}
        pipeline.run_pipeline(srt, output_dir=out, config=cfg)
    else:
        monkeypatch.setitem(sys.modules, "gradio", types.ModuleType("gradio"))
        from ui import gradio_app as ui
        monkeypatch.setattr(ui, "_now_slug", lambda: "test")
        kwargs = {key: None for key in inspect.signature(ui.run_pipeline_from_ui).parameters}
        kwargs.update(engine="vosk", srt_file_path=str(srt), output_root=str(tmp_path),
            base_speech_rate=1.25, max_mismatch_ratio=1.5, avg_chars_per_sec=13,
            do_analysis=True, do_runorm=True, do_ruaccent=True, gender_mode="use_gender", batch_size=1,
            vosk_model_dir=edge_config()["tts"]["model_path"], vosk_male_voice_choice="8: male",
            vosk_female_voice_choice="8: female", reduce_silence_enabled=True, resample_enabled=True,
            resample_target_sr=16000, audio_analysis_enabled=True, silero_pauses_enabled=True,
            silero_edges_enabled=True, video_duration_sec="", mix_sample_rate=16000, mix_channels=1,
            export_wav=True, export_mp3=False)
        assert "Export complete" in list(ui.run_pipeline_from_ui(**kwargs))[-1][0]
        out = tmp_path / "run_test"
    assert events == ["tts", "legacy", "resample", "vad", "edges"] + (["vad", "edges"] if gap else []) + ["timing"]
    edge_rows = json.loads((out / "json/silero_edges.json").read_text())
    first = edge_rows[0]
    assert first["punctuation_only_text"] == "Один два три" and first["short_protection"] is False
    assert first["chosen_text_sha256"] == text_sha256("Один два три.")
    assert first["available_ms"] == (2300 if gap else 1200)
    assert first["decision"] == ("already_fits" if gap else "applied")
    assert first["after_frames"] == (32000 if gap else 17920)
    assert (out / "mix/voiceover.wav").exists()
    # Mix must recheck the pre-normalization text and edge receipt, not just final WAV.
    edge_rows[0]["available_ms"] += 1
    (out / "json/silero_edges.json").write_text(json.dumps(edge_rows))
    normalized = json.loads((out / "json/subs_normalized.json").read_text())
    source = pipeline.parse_srt_file(str(srt))
    with pytest.raises(ValueError, match="edge text/window"):
        pipeline.validate_complete_mix(normalized, source, out / "segments", out / "json/audio_timing.json",
            pauses_path=out / "json/silero_pauses.json", edges_path=out / "json/silero_edges.json", edge_items=source)


def test_native_edge_runtime_policy_and_actual_repair_advice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = tmp_path / "helper.py"
    helper.write_text("current")
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(batch, "RUNTIME_FILES", ("helper.py",))
    monkeypatch.setattr(batch.importlib.metadata, "distribution", lambda name: (_ for _ in ()).throw(
        batch.importlib.metadata.PackageNotFoundError(name)))
    monkeypatch.setattr(sp, "runtime_identity", lambda: {"policy": sp.POLICY})
    cfg = edge_config()
    before = batch.runtime_binding(cfg)
    disabled = copy.deepcopy(cfg)
    disabled["silero_edges"]["enabled"] = False
    assert batch.runtime_binding(disabled) != before
    monkeypatch.setattr(sp, "EDGE_POLICY", {**sp.EDGE_POLICY, "id": "changed"})
    assert batch.runtime_binding(cfg) != before
    monkeypatch.setattr(batch, "runtime_binding", lambda config: {})
    source, folder = tmp_path / "source.json", tmp_path / "batch"
    source.write_text(json.dumps([{"index": 1, "start": 0, "end": 1100, "text": "Очень длинная исходная реплика."}]))
    state = batch.prepare(source, folder, cfg, 0, 0)
    cue = state["cues"][0]
    cue["attempts"] = [{"status": "unresolved", "pretempo": {"duration_sec": 2},
                         "pauses": {"duration_sec": 2}, "edges": {"duration_sec": 1.12}}]
    request = batch.make_request(folder, state, 2, [cue])
    chars = len(batch.prepare_length_text(cue["chosen_text"], "normalized_tts_text"))
    import math
    assert request["entries"][0]["target"]["measured_advisory_max_chars"] == math.floor(chars * 1.1 * 1.2 / 1.12 * 0.9)


@pytest.mark.parametrize("failure", ["text", "window"])
def test_native_saved_edge_decision_binding_cannot_be_reinterpreted(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    monkeypatch.setattr(batch, "runtime_binding", lambda config: {})
    source, folder = tmp_path / "source.json", tmp_path / "batch"
    source.write_text(json.dumps([{"index": 1, "start": 0, "end": 1200, "text": "Один два три."}]))
    state = batch.prepare(source, folder, edge_config(), 0, 0)
    cue = state["cues"][0]
    edge = {"chosen_text_sha256": cue["chosen_sha256"], "available_ms": cue["target"]["available_ms"]}
    cue["attempts"] = [{"status": "fits", "text_sha256": cue["chosen_sha256"], "silero_edges": edge}]
    if failure == "text":
        edge["chosen_text_sha256"] = "stale"
    else:
        edge["available_ms"] += 1
    batch.persist(folder, state)
    with pytest.raises(ValueError, match="Edge text/window binding mismatch"):
        batch.load_state(folder)


def test_native_audio_loop_passes_original_text_and_canonical_window(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from utils import reduce_silence, audio_resample
    events: list[str] = []
    edge_detector(monkeypatch, events)
    monkeypatch.setattr(batch, "runtime_binding", lambda config: {})
    monkeypatch.setattr(reduce_silence, "reduce_silence_file", lambda *args, **kwargs: events.append("legacy"))
    monkeypatch.setattr(audio_resample, "resample_wav_file", lambda *args, **kwargs: events.append("resample"))

    class SyntheticEngine:
        def __init__(self, config: dict[str, Any]) -> None:
            pass

        def normalize(self, item: dict[str, Any]) -> str:
            return "О+дин дв+а тр+и."

        def synthesize(self, text: str, path: Path, inputs: list[str]) -> None:
            events.append("tts")
            sf.write(path, edge_pcm(), 16000, subtype="PCM_16")
            inputs.append(batch.normalize_vosk_text(text))

    monkeypatch.setattr(batch, "AudioEngine", SyntheticEngine)
    source, folder = tmp_path / "source.json", tmp_path / "batch"
    source.write_text(json.dumps([{"index": 1, "start": 0, "end": 1200, "text": "Один два три."}]))
    batch.prepare(source, folder, edge_config(), 0, 0)
    state = batch.audio(folder)
    attempt = state["cues"][0]["attempts"][0]
    assert events == ["tts", "legacy", "resample", "vad"]  # One stub detector receipt, no second VAD.
    assert attempt["silero_edges"]["available_ms"] == 1200
    assert attempt["silero_edges"]["chosen_text_sha256"] == text_sha256("Один два три.")
    assert attempt["silero_edges"]["punctuation_only_text"] == "Один два три"
    assert attempt["edges"]["duration_sec"] == attempt["final"]["duration_sec"] == 1.12
    assert attempt["effective_factor"] == 1 and attempt["status"] == "fits"
    assert batch.status_summary(state)["ready_for_separate_mix"]
    assert not (folder / "mix").exists() and not (folder / "report.html").exists()


def test_all_eight_saved_edge_plans_without_any_inference() -> None:
    path = Path("output/pipeline_edge_preview_20261005/results.json")
    if not path.exists():
        pytest.skip("Saved edge listening fixtures absent")
    rows = json.loads(path.read_text())
    assert len(rows) == 8
    for row in rows:
        assert row["receipt"]["chosen_text_sha256"] == row["text_sha256"] == text_sha256(row["text"])
        assert row["receipt"]["model_sha256"] == sp.MODEL_SHA256
        assert all(row["receipt"]["parameters"][key] == value for key, value in sp.PARAMETERS.items())
        original, sr = sf.read(row["pretempo"]["path"], dtype="int16", always_2d=True)
        assert sp.digest(Path(row["pretempo"]["path"])) == row["pretempo"]["sha256"]
        spans = row["receipt"]["speech_16k_samples"]
        cuts, speech, _ = sp.agreed_cuts(original, sr, spans, row["receipt"]["detector_frames"])
        baseline = sp.cut_pcm(original, cuts)
        mapped = sp.protected_after_cuts(speech, cuts, len(original))
        assert [list(span) for span in mapped] == row["padded_vad_baseline_samples"]
        for (a, b), (x, y) in zip(speech, mapped):
            assert np.array_equal(original[a:b], baseline[x:y])
        plan = sp.edge_plan(baseline, sr, mapped, row["short_protection"])
        assert json.loads(json.dumps(plan)) == row["plan"]
        expected = baseline[plan["left_samples"]:len(baseline) - plan["right_samples"]]
        approved, _ = sf.read(row["after_path"], dtype="int16", always_2d=True)
        assert len(expected) == row["after_frames"] and np.array_equal(expected, approved)
