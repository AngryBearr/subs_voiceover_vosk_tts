"""Voices vs everything else: Hybrid Demucs (torchaudio, MUSDB-HQ + extra data) on the GPU.

Used to clean voice-cloning references (``stt.voice_bank``) and to duck only the original
voices under the Russian voice-over, keeping music and effects (``stt.mix``). Long audio is
processed in overlapping chunks joined with linear cross-fades.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class VocalSeparator:
    """The vocal stem of stereo audio at ``rate`` (44.1 kHz)."""

    def __init__(self, device: str = "cuda") -> None:
        import torch
        import torchaudio

        bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
        self._torch = torch
        self.device = device
        self.model: Any = bundle.get_model().to(device).eval()
        self.rate = int(bundle.sample_rate)
        self.vocals = list(self.model.sources).index("vocals")

    def _vocals(self, stereo: np.ndarray) -> np.ndarray:
        x = self._torch.from_numpy(np.ascontiguousarray(stereo, dtype=np.float32)).to(self.device)
        ref = x.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with self._torch.inference_mode():
            vocals = self.model(((x - mean) / std)[None])[0, self.vocals] * std + mean
        return vocals.float().cpu().numpy()

    def __call__(self, stereo: np.ndarray) -> tuple[np.ndarray, float]:
        """(vocals, accompaniment level relative to the vocals in dB) of a short (2, n) stretch."""
        vocals = self._vocals(stereo)
        rest = stereo - vocals
        level = 10 * np.log10(float((rest ** 2).mean() + 1e-12) / float((vocals ** 2).mean() + 1e-12))
        return vocals, round(level, 1)

    def separate_long(self, stereo: np.ndarray, chunk_s: float = 30.0, overlap_s: float = 1.0) -> np.ndarray:
        """Vocals of a long (2, n) recording, chunk by chunk."""
        n = stereo.shape[1]
        chunk, overlap = int(chunk_s * self.rate), int(overlap_s * self.rate)
        out = np.zeros_like(stereo, dtype=np.float32)
        weight = np.zeros(n, dtype=np.float32)
        start = 0
        while start < n:
            end = min(n, start + chunk)
            part = self._vocals(stereo[:, start:end])
            ramp = np.ones(end - start, dtype=np.float32)
            fade = min(overlap, (end - start) // 2)
            if start > 0 and fade:
                ramp[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
            if end < n and fade:
                ramp[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
            out[:, start:end] += part * ramp
            weight[start:end] += ramp
            if end == n:
                break
            start = end - overlap
        return out / np.maximum(weight, 1e-6)
