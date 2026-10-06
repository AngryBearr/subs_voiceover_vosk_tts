"""Test the real text boundary with a fake engine; no model loading or inference."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import wave

import numpy as np
import pytest

from synthesize import synthesize as single
from synthesize import synthesize_batch as batch
from utils import audio_resample

RAW = '  «Теп+ерь ш+оу» — “два”…\n'
EXPECTED = '"Теп+ерь ш+оу" - "два"...'


class FakeEngine:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, str, int, float]] = []
        self.fail = fail

    def synth(self, *, text: str, oname: str, speaker_id: int, speech_rate: float) -> None:
        self.calls.append(("synth", text, speaker_id, speech_rate))
        if self.fail:
            raise KeyError("unit-test-failure")
        with wave.open(oname, "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(22050)
            stream.writeframes(np.zeros(100, dtype=np.int16).tobytes())

    def synth_audio(self, *, text: str, speaker_id: int, speech_rate: float) -> np.ndarray:
        self.calls.append(("synth_audio", text, speaker_id, speech_rate))
        return np.zeros(100, dtype=np.int16)


@pytest.mark.parametrize("sample_rate", [None, 44100])
def test_each_api_receives_one_canonicalization_with_stress_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_rate: int | None,
) -> None:
    engine: Any = FakeEngine()
    observed: list[str] = []
    captured: list[str] = []
    normalize = single.normalize_vosk_text

    def spy(text: str) -> str:
        observed.append(text)
        return normalize(text)

    monkeypatch.setattr(single, "normalize_vosk_text", spy)
    output = tmp_path / "test.wav"
    result = single.synthesize_text(RAW, voice=8, speech_rate=1.25, output_path=str(output),
                                    output_sample_rate=sample_rate, synth=engine, tts_input_texts=captured)
    assert result == output and result.exists()
    assert RAW == '  «Теп+ерь ш+оу» — “два”…\n'
    assert observed == [RAW] and captured == [EXPECTED]
    assert engine.calls == [("synth" if sample_rate is None else "synth_audio", EXPECTED, 8, 1.25)]


def test_resample_fallback_does_not_canonicalize_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine: Any = FakeEngine()
    observed: list[str] = []
    normalize = single.normalize_vosk_text

    def spy(text: str) -> str:
        observed.append(text)
        return normalize(text)

    def fail_resample(*args: Any, **kwargs: Any) -> np.ndarray:
        raise RuntimeError("unit-test-resample-failure")

    monkeypatch.setattr(single, "normalize_vosk_text", spy)
    monkeypatch.setattr(audio_resample, "resample_int16_scipy", fail_resample)
    single.synthesize_text(RAW, output_path=str(tmp_path / "test.wav"), output_sample_rate=44100, synth=engine)
    assert observed == [RAW]
    assert [call[:2] for call in engine.calls] == [("synth_audio", EXPECTED), ("synth", EXPECTED)]


@pytest.mark.parametrize("fail", [False, True])
def test_batch_returns_and_persists_engine_input_without_mutating_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool,
) -> None:
    engine = FakeEngine(fail=fail)
    monkeypatch.setattr(batch, "create_synth", lambda **kwargs: engine)
    source = tmp_path / "source.json"
    original = json.dumps([{"index": 71, "text": [RAW]}], ensure_ascii=False)
    source.write_text(original, encoding="utf-8")
    output = tmp_path / "segments"
    evidence = batch.synthesize_json_lines(str(source), model_path="not-loaded", output_folder=str(output),
                                           voice=8, default_speech_rate=1.25, output_naming="index")
    assert source.read_text(encoding="utf-8") == original
    assert evidence == json.loads((output / "tts_inputs.json").read_text(encoding="utf-8"))
    assert evidence[0]["source_text"] == RAW and evidence[0]["tts_input_text"] == EXPECTED
    assert evidence[0]["index"] == 71 and evidence[0]["file_id"] == 71
    assert evidence[0]["status"] == ("failed" if fail else "synthesized")
    assert engine.calls == [("synth", EXPECTED, 8, 1.25)]
