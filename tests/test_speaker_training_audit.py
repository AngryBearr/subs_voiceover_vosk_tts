"""Conservative acceptance checks for saved speaker-training data."""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from scripts.speaker_training.audit import (
    clip_matches, episode_audit, identity_aliases, overlap_conflicts, row_match, saved_row_reasons,
    unidentified_classes, valid_span,
)


def test_invalid_times_and_packed_audio_identity() -> None:
    assert not valid_span(float("nan"), 2, 3)
    assert not valid_span(-1, 2, 3)
    assert not valid_span(1, 4, 3)
    assert valid_span(1, 2, 3)
    audio = np.arange(5000, dtype=np.int16)
    assert clip_matches(audio, audio[101:500], 100 / 16000)
    assert not clip_matches(audio, audio[120:500], 100 / 16000)
    bad = audio[101:500].copy()
    bad[-1] += 1
    assert not clip_matches(audio, bad, 100 / 16000)


def test_label_conflicts_and_saved_row_relocation_rejected() -> None:
    rows = [{"start": 1., "end": 3., "person": "A", "named": True},
            {"start": 2., "end": 4., "person": "B", "named": True},
            {"start": 5., "end": 6., "person": "B", "named": True}]
    assert overlap_conflicts(rows, "person") == {0, 1}
    assert row_match({"start": 1., "end": 3.}, "A", rows) is rows[0]
    assert row_match({"start": 10., "end": 12.}, "A", rows) is None
    assert row_match({"start": 1., "end": 3.}, "C", rows) is None


def test_saved_text_and_competing_voice_must_be_checked() -> None:
    saved = {"text": "The original actual phrase", "other_voice_s": .2}
    match = {"text": "A different matching phrase", "reasons": ["low_local_coverage"]}
    assert set(saved_row_reasons(saved, match)) == {
        "saved_text_differs_from_subtitle", "competing_voice_in_saved_row", "low_local_coverage"}
    assert saved_row_reasons(saved, None) == ["competing_voice_in_saved_row", "no_matching_subtitle_row_at_saved_time"]
    assert identity_aliases({"S01:A": {"who": "One Person"}, "S01:B": {"who": "One Person"},
                             "S02:A": {"who": "One Person"}, "S01:C": {"who": "Other Person"}}) == {
                                 "S01:one person": ["S01:A", "S01:B"]}
    assert unidentified_classes({"ST07:DRIVER": {"who": "Unknown driver", "gender": "unknown"},
                                 "S41:EVVIE": {"who": "Evvie Jagoda", "gender": "unknown"},
                                 "_HOST": "male"}) == {"ST07:DRIVER"}


def test_good_episode_average_cannot_admit_an_isolated_bad_region(tmp_path: Path) -> None:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"source identity fixture")
    (tmp_path / "source.json").write_text(json.dumps({"media": str(media), "size": media.stat().st_size, "start": 0.0}))
    with wave.open(str(tmp_path / "audio.wav"), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(bytes(360 * 16000 * 2))
    segments = []
    subtitles = []
    for i in range(90):
        start = i * 4
        text = " ".join(f"word{i}letter{letter}" for letter in "abcde")
        heard = text if i < 60 or i == 82 else " ".join(f"other{i}letter{letter}" for letter in "abcde")
        segments.append({"start": start, "end": start + 2, "text": heard,
                         "words": [{"start": start + j * .3 + .1, "end": start + j * .3 + .3, "text": token}
                                   for j, token in enumerate(heard.split())]})
        a, b = start, start + 2
        subtitles.append(f"{i + 1}\n00:{a // 60:02}:{a % 60:02},000 --> 00:{b // 60:02}:{b % 60:02},000\nALICE: {text}\n")
    srt = tmp_path / "subs.srt"
    srt.write_text("\n".join(subtitles))
    (tmp_path / "asr.json").write_text(json.dumps({"segments": segments}))
    report, rows = episode_audit({"folder": tmp_path, "srt": srt, "media": str(media), "season": "S99", "ep": "S99E01"})
    assert report["named_match"] > .6
    island = next(r for r in rows if r["piece_id"] == 82)
    assert island["match"] == 1.0
    assert "low_local_coverage" in island["reasons"]
    assert island["status"] == "excluded"
