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
