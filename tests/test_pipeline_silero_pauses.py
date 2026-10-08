"""Fixed-policy PCM and pipeline checks; saved receipts, never actual inference."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from utils import audio_analysis as aa, measured_audio as batch, silero_pauses as sp
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
