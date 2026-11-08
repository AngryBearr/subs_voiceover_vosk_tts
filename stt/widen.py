"""Bandwidth extension of a cloned voice-over track for seasons recorded with a narrow band (Survivor S1: AAC 48 kbit/s,
nothing above ~5.2 kHz).

ESpeech copies the acoustics of its reference clip, so the clones of an old season end where the references end. Before
the mix, the engine's whole voice track is run once through LavaSR v2 (speech bandwidth extension, code MIT, weights
Apache-2.0; ~50 MB, 60x real time on the CPU), which draws the band above the references' edge and returns 48 kHz.
Measured 7.11.2026 (`docs/experiments.md`): the clones' similarity to the person stays (r3 0.462 vs 0.473 in the original
band); by ear, blind, every one of 12 S01 picks was a widened version, on S48 (wide band already) no preference.

The decision is made from the episode's reference clips (``speaker_voices.<engine>.json``): their median band edge under
``max_band_hz``. LavaSR lives in ``lava_env`` (``bash scripts/setup_envs.sh lava``): its vocos fork conflicts with
f5-tts. Without the environment the track is mixed as it is.

    lava_env/bin/python -m stt.widen --worker IN.wav OUT.wav --edge 5200   # what ``widen_track`` runs
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("stt.widen")

REPO_ROOT = Path(__file__).resolve().parents[1]
LAVA_REPO = "YatharthS/LavaSR"
LAVA_REVISION = "b98dc8be472da45ab7b6346ad7997e1dfeb5911d"  # the weights measured on 7.11.2026
MAX_BAND_HZ = 7600.0  # references whose band ends below this are narrowband (S1: ~5.2 kHz, S48: 9.7-11.6 kHz)
FLOOR_DB = 45.0  # the band edge: the highest frequency within this of the loudest band of the long-term spectrum


def band_edge(audio: np.ndarray, sample_rate: int, floor_db: float = FLOOR_DB) -> float:
    """The highest frequency (Hz) whose long-term level, smoothed over ~200 Hz, is within ``floor_db`` of the loudest
    band. Speech energy sits low, so a cumulative-energy rule would put a wide-band voice at 3-4 kHz; this one finds the
    edge of what the recording kept."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n_fft, hop = 2048, 512
    if len(audio) < n_fft:
        return sample_rate / 2
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop] * np.hanning(n_fft)
    level = 10 * np.log10((np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(axis=0) + 1e-12)
    width = max(1, int(200 / (sample_rate / n_fft)))
    smooth = np.convolve(level, np.ones(width) / width, mode="same")
    above = np.nonzero(smooth > smooth.max() - floor_db)[0]
    freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    return float(freqs[above[-1]]) if len(above) else sample_rate / 2


def clip_of(voice: str) -> Path | None:
    """The reference WAV of a voice entry ("<path>.wav#<hash>"); None for TeraTTS / Vosk voices."""
    path = Path(voice.split("#", 1)[0])
    return path if path.suffix.lower() == ".wav" and path.exists() else None


def references_edge(voices_json: Path) -> float | None:
    """Median band edge of the reference clips an engine used for an episode (``speaker_voices.<engine>.json``: speaker
    id -> voice); None when it names no clip."""
    import soundfile as sf

    try:
        voices: dict[str, Any] = json.loads(voices_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    edges = []
    for clip in {c for c in (clip_of(str(v)) for v in voices.values()) if c is not None}:
        audio, rate = sf.read(clip, dtype="float32", always_2d=False)
        edges.append(band_edge(audio, rate))
    return statistics.median(edges) if edges else None


def lava_python(root: Path = REPO_ROOT) -> Path:
    return root / "lava_env" / "bin" / "python"


def widen_track(track: Path, out: Path, edge_hz: float, python: Path | None = None) -> bool:
    """``out``: ``track`` with the band above ``edge_hz`` drawn by LavaSR (48 kHz). False when ``lava_env`` is missing or
    the worker fails (the caller mixes the plain track)."""
    python = python or lava_python()
    if not python.exists():
        log.warning("widen: %s is missing (bash scripts/setup_envs.sh lava): the voice track keeps its narrow band", python)
        return False
    tmp = out.with_name(out.stem + ".tmp.wav")
    cmd = [str(python), "-m", "stt.widen", "--worker", str(track), str(tmp), "--edge", f"{edge_hz:.0f}"]
    done = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if done.returncode or not tmp.exists():
        log.warning("widen: LavaSR failed (%s): the voice track keeps its narrow band", (done.stderr or "").strip()[-400:])
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(out)
    return True


WINDOW_S, OVERLAP_S = 20.0, 0.5  # the track in 20 s pieces, cross-faded over 0.5 s (LavaSR's own batches have no overlap)


def _worker(src: Path, dst: Path, edge_hz: float) -> None:
    """Inside ``lava_env``: LavaSR v2 at 16 kHz input (the content ends below 8 kHz), the band drawn from just under the
    edge, no denoiser (it changed the voice: the reference's similarity 0.835 -> 0.713)."""
    import soundfile as sf
    import torch
    import torchaudio
    from huggingface_hub import snapshot_download
    from LavaSR.model import LavaEnhance2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LavaEnhance2(snapshot_download(LAVA_REPO, revision=LAVA_REVISION), device)
    model.load_audio(str(src), input_sr=16000, duration=0.1, cutoff=max(3000, int(edge_hz - 200)))  # sets the cutoff
    audio, rate = sf.read(src, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    x = torchaudio.functional.resample(torch.from_numpy(audio), rate, 16000).numpy()
    step, overlap = int(WINDOW_S * 16000), int(OVERLAP_S * 16000)
    out = np.zeros(len(x) * 3 + 48000, dtype=np.float32)
    fade = np.linspace(0.0, 1.0, overlap * 3, dtype=np.float32)
    for start in range(0, len(x), step):
        piece = x[max(0, start - overlap): start + step]
        with torch.inference_mode():
            y = model.enhance(torch.from_numpy(piece).unsqueeze(0).to(device), denoise=False).cpu().numpy().reshape(-1)
        a = max(0, start - overlap) * 3
        y = y[: len(out) - a]
        if start:  # cross-fade the overlap with the end of the previous piece
            n = min(len(fade), len(y))
            y[:n] = y[:n] * fade[:n] + out[a:a + n] * (1.0 - fade[:n])
        out[a:a + len(y)] = y
    sf.write(dst, np.clip(out[: len(x) * 3], -1.0, 1.0), 48000)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bandwidth extension of a voice-over track (LavaSR v2, in lava_env)")
    parser.add_argument("--worker", nargs=2, metavar=("IN", "OUT"), type=Path, required=True)
    parser.add_argument("--edge", type=float, required=True, help="Band edge of the track's voices (Hz)")
    args = parser.parse_args(argv)
    _worker(args.worker[0], args.worker[1], args.edge)


if __name__ == "__main__":
    sys.exit(main())
