from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from stt.run import REPO_ROOT, Episode, apply_show, load_config, show_of

CONFIG = json.loads((REPO_ROOT / "stt_config.json").read_text(encoding="utf-8"))
RACE = Path("/v/The.Amazing.Race.S01E01 - Go.avi")
SURVIVOR = Path("/v/Survivor.S20E01.mkv")


def _args() -> argparse.Namespace:
    return argparse.Namespace(season_glossary=None, force=None, engine=None, show=None)


def _episode(tmp_path: Path, media: Path) -> Episode:
    return Episode(media, tmp_path, _args(), copy.deepcopy(CONFIG))


def test_show_of() -> None:
    assert show_of(CONFIG, RACE) == "amazing race"
    assert show_of(CONFIG, Path("/v/The Amazing Race Season 1 - 30/x.avi")) == "amazing race"
    assert show_of(CONFIG, SURVIVOR) is None
    assert show_of(CONFIG, RACE, "none") is None
    assert show_of(CONFIG, SURVIVOR, "amazing race") == "amazing race"
    with pytest.raises(SystemExit):
        show_of(CONFIG, RACE, "unknown show")


def test_apply_show_merges_a_copy() -> None:
    before = copy.deepcopy(CONFIG)
    merged = apply_show(CONFIG, "amazing race")
    assert merged["voice_bank"] == {"enabled": True, "identities": True, "stitch": True}
    assert merged["voiceover"]["engines"]["f5"]["standin_pool"]["male"][0] == "teratts:ru_m5"
    assert merged["voiceover"]["engines"]["f5"]["fallback"] == "teratts"
    assert CONFIG == before
    assert apply_show(CONFIG, None) is CONFIG


def test_base_config_is_survivor() -> None:
    config = load_config(REPO_ROOT / "stt_config.json")
    assert config["voice_bank"]["identities"] is False and config["voice_bank"]["stitch"] is False
    assert "standin_pool" not in config["voiceover"]["engines"]["f5"]
    assert "settings" in config["shows"]["amazing race"]


def test_episode_takes_the_profile_of_its_path(tmp_path: Path) -> None:
    race, survivor = _episode(tmp_path, RACE), _episode(tmp_path, SURVIVOR)
    assert race.show == "amazing race" and race.config["voice_bank"]["identities"] is True
    assert survivor.show is None and survivor.config["voice_bank"]["identities"] is False
    assert "standin_pool" not in survivor.config["voiceover"]["engines"]["f5"]


def _speakers_argv(episode: Episode) -> list[str]:
    with mock.patch("stt.speakers.main") as main, mock.patch("stt.season_voices.restore_first_pass", return_value=False), \
            mock.patch.object(Episode, "_module"), mock.patch.object(Episode, "_find_cast", return_value=False), \
            mock.patch.object(Episode, "_host_speakers"), mock.patch.object(Episode, "llm", return_value="m"):
        try:
            episode.speakers()
        except Exception:  # noqa: BLE001 - only the argv of the naming call matters
            pass
    return main.call_args.args[0]


def test_speakers_identities_flag_follows_the_profile(tmp_path: Path) -> None:
    assert "--no-identities" in _speakers_argv(_episode(tmp_path, SURVIVOR))
    assert "--no-identities" not in _speakers_argv(_episode(tmp_path, RACE))


@pytest.mark.parametrize("stitch", [False, True])
def test_export_stitch_flag(tmp_path: Path, stitch: bool) -> None:
    episode = _episode(tmp_path, SURVIVOR)
    episode.config["voice_bank"]["stitch"] = stitch
    episode.voice_bank.parent.mkdir(parents=True, exist_ok=True)
    episode.voice_bank.write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []

    def module(name: str, mod: str, args: list[str]) -> None:
        calls.append(args)

    with mock.patch.object(Episode, "_module", side_effect=module):
        try:
            episode._speaker_voices()
        except Exception:  # noqa: BLE001 - only the export call matters
            pass
    export: list[Any] = next(a for a in calls if "--export" in a)
    assert ("--no-stitch" in export) is (not stitch)
