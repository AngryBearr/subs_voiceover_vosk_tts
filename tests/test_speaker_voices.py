from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import soundfile as sf

from stt.run import Episode

POOL = {"female": ["teratts:ru_f1+ru_f2", "teratts:ru_f2"], "male": ["teratts:ru_m5", "teratts:ru_m1"]}


def _episode(tmp_path: Path, name: str, cards: list[dict[str, Any]], pool: dict[str, list[str]] | None) -> Episode:
    episode = Episode.__new__(Episode)
    episode.media, episode.out = Path(f"{name}.mkv"), tmp_path / name
    episode.voice_bank = tmp_path / "Season 1.voices.json"  # absent: no refresh / export
    episode.config = {"voiceover": {"engines": {"f5": {"standin_pool": pool} if pool else {}}}}
    episode.out.mkdir(exist_ok=True)
    (episode.out / "speakers.json").write_text(json.dumps(cards), encoding="utf-8")
    (episode.out / "subs.en.json").write_text(
        json.dumps([{"speaker": c["id"], "text": ["x"]} for c in cards if c.get("cues", 1)]), encoding="utf-8")
    return episode


def _cards() -> list[dict[str, Any]]:
    return [{"id": "S1", "name": "Amy Adams", "identity": None, "gender": "female"},
            {"id": "S2", "name": None, "identity": "V03", "gender": "female"},
            {"id": "S3", "name": None, "identity": None, "gender": "female"},
            {"id": "S4", "name": "Bob Brown", "identity": None, "gender": "male"}]


def _voices(episode: Episode) -> dict[str, str]:
    return json.loads(episode._speaker_voices().read_text(encoding="utf-8"))


def test_people_without_a_clip_get_stable_pool_voices(tmp_path: Path) -> None:
    first = _voices(_episode(tmp_path, "e1", _cards(), POOL))
    assert first == {"S1": "teratts:ru_f1+ru_f2", "S2": "teratts:ru_f2", "S4": "teratts:ru_m5"}  # S3: no key
    stored = json.loads((tmp_path / "Season 1.references" / "standins.json").read_text(encoding="utf-8"))
    assert stored["V03"] == {"voice": "teratts:ru_f2", "gender": "female", "episode": "e1"}
    again = _voices(_episode(tmp_path, "e1", _cards(), POOL))
    other = _voices(_episode(tmp_path, "e2", [{**c, "id": "X" + c["id"]} for c in reversed(_cards())], POOL))
    assert again == first
    assert {k[1:]: v for k, v in other.items()} == first  # a second episode: the same voices per person


def test_a_clip_wins_and_no_pool_means_no_stand_ins(tmp_path: Path) -> None:
    refs = tmp_path / "Season 1.references"
    refs.mkdir()
    sf.write(str(refs / "Amy.wav"), np.zeros(24000, dtype=np.float32), 24000)
    (refs / "references.json").write_text(json.dumps([{"name": "Amy Adams", "identity": None, "wav": "Amy.wav"}]),
                                          encoding="utf-8")
    voices = _voices(_episode(tmp_path, "e1", _cards(), POOL))
    assert voices["S1"].startswith(str((refs / "Amy.wav").resolve()) + "#") and voices["S2"] == "teratts:ru_f1+ru_f2"
    assert "S3" not in voices
    assert set(_voices(_episode(tmp_path, "e2", _cards(), None))) == {"S1"}
    assert not (refs / "standins.json").exists() or "e2" not in (refs / "standins.json").read_text(encoding="utf-8")


def _bare_episode(tmp_path: Path, own_voice: dict[str, Any] | None) -> Episode:
    cards = [{"id": "S1", "name": "Amy Adams", "identity": None, "gender": "female"},
             {"id": "S2", "name": None, "identity": None, "gender": "male"},
             {"id": "S3", "name": None, "identity": None, "gender": "female"},
             {"id": "S4", "name": None, "identity": None, "gender": "unknown"}]
    episode = _episode(tmp_path, "e1", cards, None)  # no pool: everybody without a clip is generic
    episode.config["voiceover"]["engines"]["f5"] = {"own_voice": own_voice} if own_voice is not None else {}
    episode._engine = lambda: {"voices": {"female": "teratts:ru_f1+ru_f2", "male": "teratts:ru_m5"}}  # type: ignore[method-assign]
    return episode


def test_own_voice_maps_generic_speakers_to_the_voice_of_their_russian_sample(tmp_path: Path) -> None:
    from voiceover.engine_f5 import voice_id

    sample = tmp_path / "e1" / "own_voice" / "S2.ru.wav"
    sample.parent.mkdir(parents=True)
    sf.write(str(sample), np.zeros(24000, dtype=np.float32), 24000)
    sample.with_suffix(".txt").write_text("Привет.\n", encoding="utf-8")

    def voices_of(folder: Path, speakers: list[str], config: dict[str, Any], reasons: dict[str, str] | None = None,
                  **_: Any) -> dict[str, Path]:
        assert speakers == ["S1", "S2", "S3"]  # S4 has no gender: it is not offered
        assert reasons is not None
        reasons["S3"] = "no Russian text"
        return {"S2": sample}

    with mock.patch("stt.own_voice.build_voices", side_effect=voices_of):
        voices = _voices(_bare_episode(tmp_path, {"enabled": True}))
    assert voices == {"S2": voice_id(sample)}  # a plain clip voice; S1, S3 (no sample) and S4 stay generic


def test_own_voice_off_leaves_the_mapping_as_it_was(tmp_path: Path) -> None:
    with mock.patch("stt.own_voice.build_voices") as build:
        assert _voices(_bare_episode(tmp_path, None)) == {}
        assert _voices(_bare_episode(tmp_path, {"enabled": False})) == {}
    build.assert_not_called()


def test_vosk_stand_in_is_cloned_from_the_fallback_sentence(tmp_path: Path) -> None:
    pool = {"female": ["vosk:7"], "male": ["vosk:8"]}
    with mock.patch("voiceover.engine_f5.standin_voice", return_value="/w.wav#abc") as stand:
        assert _voices(_episode(tmp_path, "e1", _cards()[:1], pool)) == {"S1": "/w.wav#abc"}
    assert stand.call_args.args[0] == "vosk:7"
