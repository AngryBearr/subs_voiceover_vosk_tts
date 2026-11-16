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
    loud, _, _ = mix_module.finish_mix(audio.copy(), 1000, None, inplace=True)
    assert np.array_equal(loud, copy)
