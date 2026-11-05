"""Overlapping GPU jobs only when the card has room (gpu.parallel); Demucs falls back to the CPU. Fakes only."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stt import gpu_budget
from stt import transcribe as tr

GB = 1024.0


def card(monkeypatch: pytest.MonkeyPatch, total_gb: float | None, free_gb: float | None = None) -> None:
    """The card the policy sees: total and free GB, or no card at all."""
    found = None if total_gb is None else (total_gb * GB, (total_gb if free_gb is None else free_gb) * GB)
    monkeypatch.setattr(gpu_budget, "query_vram", lambda: found)
    gpu_budget.reset_cache()


# ---- the policy

@pytest.mark.parametrize("total_gb, expected", [
    (16, {"ocr": True, "mix": True, "compress": True}),
    (12, {"ocr": True, "mix": True, "compress": True}),
    (10, {"ocr": False, "mix": True, "compress": True}),
    (9, {"ocr": False, "mix": False, "compress": True}),
    (8, {"ocr": False, "mix": False, "compress": False}),
    (6, {"ocr": False, "mix": False, "compress": False}),
])
def test_auto_follows_the_size_of_the_card(monkeypatch: pytest.MonkeyPatch, total_gb: int,
                                           expected: dict[str, bool]) -> None:
    card(monkeypatch, total_gb)
    assert {name: gpu_budget.decide(name, "auto").allowed for name in gpu_budget.OVERLAPS} == expected


def test_unknown_card_runs_one_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    card(monkeypatch, None)
    for name in gpu_budget.OVERLAPS:
        decision = gpu_budget.decide(name, "auto")
        assert not decision.allowed and "unknown" in decision.reason


def test_off_never_and_on_always(monkeypatch: pytest.MonkeyPatch) -> None:
    card(monkeypatch, 16)
    assert not any(gpu_budget.decide(name, "off").allowed for name in gpu_budget.OVERLAPS)
    card(monkeypatch, None)
    assert all(gpu_budget.decide(name, "on").allowed for name in gpu_budget.OVERLAPS)
    assert gpu_budget.normalize_mode(None) == "auto" and gpu_budget.normalize_mode("ON") == "on"
    assert gpu_budget.normalize_mode(True) == "on" and gpu_budget.normalize_mode(False) == "off"
    assert gpu_budget.normalize_mode("sometimes") == "auto"


def test_little_free_memory_at_the_start_means_one_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    card(monkeypatch, 16, free_gb=1.0)  # another program holds the card
    for name in gpu_budget.OVERLAPS:
        decision = gpu_budget.decide(name, "auto")
        assert not decision.allowed and "free now" in decision.reason


def test_reason_names_the_card_and_what_happens_instead(monkeypatch: pytest.MonkeyPatch) -> None:
    card(monkeypatch, 8)
    decision = gpu_budget.decide("ocr", "auto")
    assert "8.0 GB GPU" in decision.reason and "OCR after ASR, one at a time" in decision.reason


def test_a_job_that_is_not_on_the_gpu_always_overlaps(monkeypatch: pytest.MonkeyPatch) -> None:
    card(monkeypatch, None)
    assert gpu_budget.decide("compress", "auto", uses_gpu=False).allowed  # a software encoder
    assert not gpu_budget.decide("compress", "off", uses_gpu=False).allowed


def test_total_is_asked_once_free_every_time(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter([(8192.0, 8000.0), (16384.0, 1000.0)])
    gpu_budget.reset_cache()
    monkeypatch.setattr(gpu_budget, "query_vram", lambda: next(answers))
    assert gpu_budget.gpu_memory() == (8192.0, 8000.0)
    assert gpu_budget.gpu_memory() == (8192.0, 1000.0)


def test_record_peak_keeps_the_largest(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "vram_peaks.json"
    gpu_budget.record_peak(path, "demucs", 2500.4)
    gpu_budget.record_peak(path, "demucs", 2000)
    gpu_budget.record_peak(path, "nvenc", 700)
    gpu_budget.record_peak(path, "none", None)
    gpu_budget.record_peak(None, "x", 5)
    assert json.loads(path.read_text()) == {"demucs": 2500, "nvenc": 700}


# ---- Demucs: out of memory

def test_oom_waits_retries_once_then_uses_the_cpu() -> None:
    devices: list[str] = []
    sleeps: list[float] = []

    def run(device: str) -> str:
        devices.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return "done on cpu"

    assert gpu_budget.run_with_cpu_fallback(run, "separation", wait_s=7, sleep=sleeps.append) == "done on cpu"
    assert devices == ["cuda", "cuda", "cpu"] and sleeps == [7]


def test_oom_then_success_on_the_retry_stays_on_the_gpu() -> None:
    devices: list[str] = []

    def run(device: str) -> str:
        devices.append(device)
        if len(devices) == 1:
            raise RuntimeError("CUDA error: out of memory")
        return device

    assert gpu_budget.run_with_cpu_fallback(run, "separation", sleep=lambda s: None) == "cuda"
    assert devices == ["cuda", "cuda"]


def test_other_errors_are_not_swallowed() -> None:
    def run(device: str) -> str:
        raise ValueError("bad audio")

    with pytest.raises(ValueError, match="bad audio"):
        gpu_budget.run_with_cpu_fallback(run, "separation", sleep=lambda s: None)


def test_separated_voices_falls_back_to_the_cpu_with_the_same_result(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import stt.separate
    from stt import mix as stt_mix

    created: list[str] = []

    class Separator:
        rate = 44100

        def __init__(self, device: str = "cuda") -> None:
            self.device = device
            created.append(device)

        def separate_long(self, stereo: np.ndarray) -> np.ndarray:
            if self.device == "cuda":
                raise RuntimeError("CUDA out of memory")
            return (stereo * 0.5).astype(np.float32)

    monkeypatch.setattr(stt.separate, "VocalSeparator", Separator)
    monkeypatch.setattr(gpu_budget.time, "sleep", lambda s: None)
    original = np.random.default_rng(0).uniform(-1, 1, (4410, 2)).astype(np.float32)
    voices = stt_mix.separated_voices(original, 44100, tmp_path / "separation" / "v.npy", tmp_path / "peaks.json")
    assert created == ["cuda", "cuda", "cpu"]
    np.testing.assert_allclose(voices, original * 0.5)
    assert (tmp_path / "separation" / "v.npy").exists()  # cached as always


# ---- the on-screen text scan in transcribe

class _Stop(Exception):
    pass


def _args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    media = tmp_path / "ep.mkv"
    tmp_path.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"x")
    captured: list[argparse.Namespace] = []
    real = tr.transcribe
    tr.transcribe = lambda a: captured.append(a) or tmp_path  # type: ignore[assignment]
    try:
        tr.main([str(media), "--out", str(tmp_path / "out"), *extra])
    finally:
        tr.transcribe = real  # type: ignore[assignment]
    return captured[0]


def _data(name: str) -> dict[str, Any]:
    if name == "diarize":
        return {"turns": [], "embeddings": {"a": [0.0]}, "diarize_s": 1.0, "peak_vram_mb": 1}
    if name.startswith("asr."):
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi",
                              "words": [{"start": 0.0, "end": 1.0, "text": "hi"}]}], "transcribe_s": 1.0}
    if name == "voice":
        return {"run_s": 1.0, "speakers": {}}
    if name == "ocr":
        return {"captions": [], "events": [], "ocr_s": 0.1, "peak_vram_mb": 500}
    raise AssertionError(name)


def _scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> tuple[list[str], list[Any]]:
    """Order of the stages (with their thread) up to the caption merge, and what the merge was given."""
    import stt.ocr

    events: list[str] = []
    seen: list[Any] = []
    monkeypatch.setattr(tr, "prepare_audio", lambda *a, **k: (60.0, False))

    def run_stage(name: str, env: str, args: list[str], out_path: Path, log_dir: Path, force: bool,
                  *, count_step: bool = True) -> dict[str, Any]:
        events.append(f"{name}:{'bg' if threading.current_thread().name.startswith('stage-') else 'main'}")
        return _data(name)

    def reclassify(data: Any, words: Any) -> Any:
        seen.append(data)
        raise _Stop

    monkeypatch.setattr(tr, "run_stage", run_stage)
    monkeypatch.setattr(stt.ocr, "reclassify", reclassify)
    with pytest.raises(_Stop):
        tr.transcribe(_args(tmp_path, "--asr", "whisper-large-v3", "--refine-speakers", "none", *extra))
    return events, seen


def test_ocr_runs_beside_the_models_on_a_big_card_and_after_them_on_a_small_one(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    card(monkeypatch, 16)
    big, big_seen = _scan(tmp_path / "a", monkeypatch)
    card(monkeypatch, 8)
    small, small_seen = _scan(tmp_path / "b", monkeypatch)
    assert "ocr:bg" in big and "ocr:main" not in big
    assert small == ["diarize:main", "asr.whisper-large-v3:main", "ocr:main", "voice:main"]  # as before the speed-up
    assert sorted(e.split(":")[0] for e in big) == sorted(e.split(":")[0] for e in small)  # the same stages
    assert big_seen == small_seen == [_data("ocr")]  # the same scan reaches the merge


@pytest.mark.parametrize("mode, background", [("off", False), ("on", True), ("auto", False)])
def test_gpu_parallel_option_of_transcribe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
                                           background: bool) -> None:
    card(monkeypatch, 8)
    events, _ = _scan(tmp_path, monkeypatch, "--gpu-parallel", mode)
    assert ("ocr:bg" in events) is background


def test_unknown_card_scans_after_the_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    card(monkeypatch, None)
    events, _ = _scan(tmp_path, monkeypatch)
    assert "ocr:main" in events and "ocr:bg" not in events


def test_a_cached_scan_is_read_in_the_background_whatever_the_card(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    card(monkeypatch, 8)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "ocr.json").write_text("{}", encoding="utf-8")
    events, _ = _scan(tmp_path, monkeypatch)
    assert "ocr:bg" in events


# ---- configuration

def test_config_and_ui_default_to_auto() -> None:
    from ui import settings

    config = json.loads((Path(__file__).resolve().parents[1] / "stt_config.json").read_text(encoding="utf-8"))
    assert config["gpu"] == {"parallel": "auto"}
    assert settings.FALLBACKS[("gpu", "parallel")] == "auto"
    setting = next(s for s in settings.SETTINGS if s.path == ("gpu", "parallel"))
    assert set(setting.choices.values()) == {"auto", "on", "off"} and "по очереди" in setting.info
