"""TeraTTS engine: blended voices (no model needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voiceover.engine_teratts import TeraTTSEngine, style_dir


def _release(tmp_path: Path) -> Path:
    release = tmp_path / "release"
    for voice, value in (("ru_f1", 1.0), ("ru_f2", 3.0)):
        (release / "styles" / voice).mkdir(parents=True)
        np.save(release / "styles" / voice / "style_ttl.npy", np.full((1, 50, 256), value, np.float32))
        np.save(release / "styles" / voice / "style_dp.npy", np.full((1, 8, 16), value, np.float32))
    return release


def test_blend_is_the_mean_of_its_voices_and_made_once(tmp_path: Path) -> None:
    release = _release(tmp_path)
    assert style_dir(release, "ru_f1", tmp_path / "styles") == release / "styles" / "ru_f1"
    blend = style_dir(release, "ru_f1+ru_f2", tmp_path / "styles")
    assert blend == tmp_path / "styles" / "ru_f1+ru_f2"
    assert np.load(blend / "style_ttl.npy").shape == (1, 50, 256)
    assert float(np.load(blend / "style_dp.npy").mean()) == pytest.approx(2.0)
    mtime = (blend / "style_ttl.npy").stat().st_mtime_ns
    style_dir(release, "ru_f1+ru_f2", tmp_path / "styles")
    assert (blend / "style_ttl.npy").stat().st_mtime_ns == mtime


def test_blend_voices_are_checked_part_by_part(tmp_path: Path) -> None:
    engine = TeraTTSEngine.__new__(TeraTTSEngine)
    engine.path = _release(tmp_path)
    engine._check_voice("ru_f1+ru_f2")
    with pytest.raises(ValueError, match="ru_x"):
        engine._check_voice("ru_f1+ru_x")
