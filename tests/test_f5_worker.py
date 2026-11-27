"""The F5 worker's direct synthesis path (no thread per line), its reference cache and the compile option.

No F5 model here: a fake ``utils_infer`` module, model, vocoder and torchaudio on CPU torch."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch

from voiceover.f5_worker import DirectF5Synthesizer, compile_transformer, cross_fade

RATE, HOP = 24000, 256


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.transformer = object()

    def sample(self, *, cond: torch.Tensor, text: list[str], duration: int, steps: int, cfg_strength: float,
               sway_sampling_coef: float) -> tuple[torch.Tensor, None]:
        self.calls.append({"cond": cond, "text": text, "duration": duration, "steps": steps, "cfg": cfg_strength,
                           "sway": sway_sampling_coef, "grad": torch.is_grad_enabled()})
        return torch.full((1, duration, 100), float(len(self.calls)), dtype=torch.float16), None


class FakeVocoder:
    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        assert mel.dtype == torch.float32 and mel.shape[1] == 100  # (1, mel channels, frames)
        return torch.full((1, mel.shape[2] * HOP), 0.5) * mel[0, 0, 0]


class FakeTorchaudio:
    def __init__(self, audio: torch.Tensor, rate: int) -> None:
        self.audio, self.rate, self.loads = audio, rate, 0
        self.transforms = SimpleNamespace(Resample=self._resample)

    def load(self, path: str) -> tuple[torch.Tensor, int]:
        self.loads += 1
        return self.audio.clone(), self.rate

    @staticmethod
    def _resample(src: int, dst: int) -> Any:
        return lambda audio: audio[:, ::src // dst] if src > dst else audio


def fake_utils(pieces: list[str] | None = None) -> SimpleNamespace:
    def infer_process(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the direct path must not start F5's threaded infer_process")

    return SimpleNamespace(target_rms=0.1, target_sample_rate=RATE, hop_length=HOP, device="cpu", cfg_strength=2.0,
                           sway_sampling_coef=-1.0, cross_fade_duration=0.15, infer_process=infer_process,
                           convert_char_to_pinyin=lambda texts: [list(t) for t in texts],
                           chunk_text=lambda text, max_chars: list(pieces) if pieces else [text.strip()])


def make(audio: torch.Tensor, rate: int = RATE, pieces: list[str] | None = None
         ) -> tuple[DirectF5Synthesizer, FakeModel, FakeTorchaudio]:
    model, ta = FakeModel(), FakeTorchaudio(audio, rate)
    return DirectF5Synthesizer(model, FakeVocoder(), fake_utils(pieces), torch, ta), model, ta


def test_one_piece_is_one_sample_call_in_this_thread() -> None:
    synth, model, _ = make(torch.full((1, RATE * 2), 0.5))  # 2 s, loud reference
    wave, rate = synth.synthesize("ref.wav", "Hello there.", "— Прив+ет.", 32, 3.5)
    assert rate == RATE and len(model.calls) == 1 and synth.direct_lines == 1 and synth.split_lines == 0
    call = model.calls[0]
    assert call["steps"] == 32 and call["cfg"] == 2.0 and call["sway"] == -1.0 and not call["grad"]
    assert call["duration"] == int(3.5 * RATE / HOP)  # fix_duration as F5 turns it into mel frames
    assert call["text"] == [list("Hello there. — Прив+ет.")]  # F5 adds a space after a one-byte final character
    assert tuple(call["cond"].shape) == (1, RATE * 2)  # the raw reference: model.sample computes its mel
    ref_frames = RATE * 2 // HOP
    assert isinstance(wave, np.ndarray) and len(wave) == (int(3.5 * RATE / HOP) - ref_frames) * HOP  # reference cut
    assert np.allclose(wave, 0.5)  # loud reference: no loudness rescale


def test_reference_is_loaded_once_per_job_and_quiet_ones_are_raised() -> None:
    synth, model, ta = make(torch.full((2, RATE), 0.02), rate=48000)  # stereo, quiet, 48 kHz
    synth.synthesize("ref.wav", "Привет", "— Да.", 16, 2.0)
    synth.synthesize("ref.wav", "Привет", "— Нет.", 16, 2.0)
    assert ta.loads == 1 and len(synth.refs) == 1
    ref = synth.refs[("ref.wav", "Привет")]
    assert tuple(ref.audio.shape) == (1, RATE // 2) and torch.allclose(ref.audio, torch.full_like(ref.audio, 0.1))
    assert ref.ref_text == "Привет"  # last character is two bytes in UTF-8: no space added
    wave, _ = synth.synthesize("ref.wav", "Привет", "— Да.", 16, 2.0)
    assert np.allclose(wave, 0.5 * 3 * 0.02 / 0.1)  # the take brought back to the reference's loudness
    synth.synthesize("other.wav", "Привет", "— Да.", 16, 2.0)
    assert ta.loads == 2  # another clip, another load


def test_long_text_pieces_run_in_order_with_their_share_of_the_length() -> None:
    synth, model, _ = make(torch.full((1, RATE), 0.5), pieces=["Первая часть,", "вторая часть подлиннее."])
    wave, _ = synth.synthesize("ref.wav", "Hello.", "— Первая часть, вторая часть подлиннее.", 32, 6.0)
    assert synth.split_lines == 1 and [c["text"][0][-3:] for c in model.calls] == [list("ть,"), list("ее.")]
    ref_sec = (RATE // HOP) * HOP / RATE
    weights = [len("Первая часть,".encode()), len("вторая часть подлиннее.".encode())]
    allocatable = 6.0 - ref_sec + 0.15
    assert [c["duration"] for c in model.calls] == [int((ref_sec + allocatable * w / sum(weights)) * RATE / HOP)
                                                     for w in weights]
    first = (model.calls[0]["duration"] - RATE // HOP) * HOP
    second = (model.calls[1]["duration"] - RATE // HOP) * HOP
    assert len(wave) == first + second - int(0.15 * RATE)  # cross-faded


def test_cross_fade_matches_f5() -> None:
    a, b = np.ones(10, dtype=np.float32), np.zeros(8, dtype=np.float32)
    out = cross_fade([a, b], 4)
    assert len(out) == 14 and np.allclose(out[6:10], np.linspace(1, 0, 4)) and out[-1] == 0
    assert len(cross_fade([a, b], 0)) == 18  # no overlap: plain concatenation
    assert cross_fade([a], 4) is a


def test_no_text_is_an_error() -> None:
    synth, _, _ = make(torch.full((1, RATE), 0.5))
    synth.utils.chunk_text = lambda text, max_chars: []
    with pytest.raises(ValueError):
        synth.synthesize("ref.wav", "Hello.", "  ", 32, 2.0)


def test_compile_wraps_the_transformer_with_dynamic_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch._dynamo.config, "cache_size_limit", 8)  # restored after the test
    model = FakeModel()
    seen: list[tuple[Any, dict[str, Any]]] = []
    original = model.transformer
    compile_transformer(model, torch, lambda module, **kw: seen.append((module, kw)) or "compiled")
    assert model.transformer == "compiled" and seen == [(original, {"dynamic": True})]
    assert torch._dynamo.config.cache_size_limit >= 64


def test_compile_option_reaches_the_worker_and_changes_the_cache(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from voiceover import engine_f5
    from voiceover.engine import PreparedText

    python = tmp_path / "python"
    python.write_text("")
    wav = tmp_path / "ref.wav"
    sf.write(str(wav), np.zeros(RATE, dtype=np.float32), RATE)
    wav.with_suffix(".txt").write_text("Hello there.")
    plain = engine_f5.F5Engine({"voice": str(wav), "python": str(python)})
    compiled = engine_f5.F5Engine({"voice": str(wav), "python": str(python), "compile": True})
    assert not plain.compile and "compile" not in plain.synth_identity()  # existing caches stay valid
    assert compiled.synth_identity()["compile"] is True and compiled.synth_identity() != plain.synth_identity()
    jobs: list[dict[str, Any]] = []

    def fake_run_worker(env: str, args: list[str], log_path: Path, **kwargs: Any) -> None:
        job = json.loads(Path(args[0]).read_text(encoding="utf-8"))
        jobs.append(job)
        Path(args[1]).write_text(json.dumps([{"key": i["key"], "error": "skipped"} for i in job["items"]]))

    monkeypatch.setattr(engine_f5, "run_worker", fake_run_worker)
    compiled.synthesize_many([PreparedText("c1", "a", "Прив+ет.", "ok")], tmp_path / "cache")
    assert jobs[0]["compile"] is True
