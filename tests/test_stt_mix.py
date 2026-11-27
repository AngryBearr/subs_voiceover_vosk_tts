from __future__ import annotations

import numpy as np

from stt.mix import mix


def test_voices_only_ducking_keeps_music() -> None:
    rate = 1000
    n = 4 * rate
    music = np.full((n, 2), 0.2, dtype=np.float32)
    voices = np.zeros((n, 2), dtype=np.float32)
    voices[:, :] = 0.2
    voice = np.zeros(n, dtype=np.float32)
    voice[rate:3 * rate] = 0.5  # the Russian voice speaks in the middle
    original = music + voices
    both = mix(original, voice, rate, duck_db=12.0)
    only = mix(original, voice, rate, duck_db=12.0, voices=voices, rest_duck_db=3.0)
    mid = slice(int(1.8 * rate), int(2.2 * rate))  # ducking fully engaged
    background_all = float((both[mid, 0] - voice[mid]).mean())  # original after ducking everything
    background_voices = float((only[mid, 0] - voice[mid]).mean())
    assert abs(background_all - 0.4 * 10 ** (-12 / 20)) < 0.01
    assert abs(background_voices - (0.2 * 10 ** (-3 / 20) + 0.2 * 10 ** (-12 / 20))) < 0.01
    assert np.allclose(only[: rate // 2], original[: rate // 2])  # no voice-over: original untouched


def _whole_track_mix(original: np.ndarray, voice: np.ndarray, rate: int, voices: np.ndarray | None,
                     extra: list[tuple[np.ndarray, float]], duck_db: float, rest_duck_db: float,
                     voice_gain_db: float) -> np.ndarray:
    """The mix as one expression over whole tracks (the formula before it went block by block)."""
    from stt.mix import speech_envelope

    length = max([len(original), len(voice)] + [len(e) for e, _ in extra])
    original = np.pad(original, ((0, length - len(original)), (0, 0)))
    voice = np.pad(voice, (0, length - len(voice)))
    env = speech_envelope(voice, rate)
    duck = 1.0 - (1.0 - 10 ** (-duck_db / 20)) * env
    if voices is not None:
        voices = np.pad(voices[:length], ((0, length - min(length, len(voices))), (0, 0)))
        rest = 1.0 - (1.0 - 10 ** (-rest_duck_db / 20)) * env
        out = (original - voices) * rest[:, None] + voices * duck[:, None]
    else:
        out = original * duck[:, None]
    out = out + (voice * 10 ** (voice_gain_db / 20))[:, None]
    for layer, gain_db in extra:
        out += (np.pad(layer, (0, length - len(layer))) * 10 ** (gain_db / 20))[:, None]
    return out.astype(np.float32)


def test_block_mix_equals_the_whole_track_formula(monkeypatch) -> None:
    """Blocks (here 997 samples, so they straddle every boundary) give bit for bit the whole-track result,
    with tracks of different lengths, a shorter vocal stem and an extra layer."""
    from stt import mix as mix_module

    monkeypatch.setattr(mix_module, "MIX_BLOCK", 997)
    rng = np.random.default_rng(1)
    rate = 1000
    original = (rng.standard_normal((5300, 2)) * 0.3).astype(np.float32)
    voice = (rng.standard_normal(5600) * 0.5).astype(np.float32)
    voice[:2000] = 0.0  # a silent start: the ducking envelope changes inside the track
    voices = (original * 0.5)[:5100].copy()
    extra = [((rng.standard_normal(4000) * 0.2).astype(np.float32), -3.0)]
    for stem in (voices, None):
        got = mix(original, voice, rate, duck_db=12.0, voice_gain_db=1.5, extra=extra, voices=stem, rest_duck_db=3.0)
        want = _whole_track_mix(original, voice, rate, stem, extra, 12.0, 3.0, 1.5)
        assert got.dtype == np.float32 and got.shape == want.shape and np.array_equal(got, want)


def test_limiter_in_place_matches_the_copy(monkeypatch) -> None:
    from stt import mix as mix_module

    monkeypatch.setattr(mix_module, "MIX_BLOCK", 997)
    rng = np.random.default_rng(2)
    audio = (rng.standard_normal((4000, 2)) * 0.6).astype(np.float32)  # peaks well over the ceiling
    copy = mix_module.limit_peaks(audio, 1000)
    assert copy is not audio and float(np.abs(copy).max()) <= 0.97 + 1e-6
    work = audio.copy()
    inplace = mix_module.limit_peaks(work, 1000, inplace=True)
    assert inplace is work and np.array_equal(inplace, copy)
    loud, _, _ = mix_module.finish_mix(audio.copy(), 1000, None, ceiling=0.97, inplace=True)
    assert np.array_equal(loud, copy)


def test_true_peak_limiter_block_seams_and_inplace_history(monkeypatch) -> None:
    """Tiny blocks must match a single block, including peaks immediately around every seam."""
    from scipy import signal

    from stt import mix as mix_module

    rate = 48000
    t = np.arange(12000) / rate
    audio = np.stack([0.8 * np.sin(2 * np.pi * 12000 * t + np.pi / 4),
                      0.3 * np.sin(2 * np.pi * 997 * t)], axis=1).astype(np.float32)
    for seam in range(997, len(audio), 997):
        audio[seam - 1:seam + 2, 1] = [1.7, -1.6, 1.5]
    saved = audio.copy()
    ceiling = 10 ** (-1.15 / 20)
    monkeypatch.setattr(mix_module, "MIX_BLOCK", len(audio))
    reference = mix_module.limit_peaks(audio, rate, ceiling)
    np.testing.assert_array_equal(audio, saved)
    resample = signal.resample_poly
    calls: list[int] = []

    def bounded_resample(data: np.ndarray, *args, **kwargs) -> np.ndarray:
        calls.append(len(data))
        # 5 ms lookahead: two reaches plus FIR support on each side, independent of episode length.
        assert len(data) <= mix_module.MIX_BLOCK + 2 * (2 * 240 + 16)
        return resample(data, *args, **kwargs)

    monkeypatch.setattr(signal, "resample_poly", bounded_resample)
    for block_size in (997, 37):  # also exercise history longer than a block
        monkeypatch.setattr(mix_module, "MIX_BLOCK", block_size)
        work = audio.copy()
        out = mix_module.limit_peaks(work, rate, ceiling, inplace=True)
        assert out is work and out.shape == audio.shape and out.dtype == np.float32
        np.testing.assert_allclose(out, reference, atol=2e-7, rtol=0)
        assert float(np.abs(resample(out, 4, 1, axis=0, window=("kaiser", 8.6))).max()) <= ceiling + 1e-6
    assert len(calls) > 100
