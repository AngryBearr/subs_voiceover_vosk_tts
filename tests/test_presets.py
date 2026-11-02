"""Sets of settings, the main pipeline's defaults and what happens to episodes that are already translated."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from stt import refine_speakers, run, season_voices
from stt.subtitles import imdb_of
from ui import dub_app

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "stt_config.json").read_text(encoding="utf-8"))


def test_main_pipeline_is_the_default() -> None:
    assert CONFIG["season_voices"]["enabled"] is True
    assert CONFIG["season_voices"]["model"] == "resnet293-ft"
    assert CONFIG["subtitles"]["enabled"] is True
    assert CONFIG["voiceover"]["engine"] == "f5,teratts,vosk"
    assert CONFIG["mix"]["video"]["compress"] is False
    assert run.apply_preset(CONFIG, "main") == CONFIG


def test_presets_change_what_they_name_only() -> None:
    upload = run.apply_preset(CONFIG, "upload")
    assert upload["mix"]["video"]["compress"] is True
    assert upload["mix"]["video"]["codec"] == CONFIG["mix"]["video"]["codec"]
    assert upload["season_voices"] == CONFIG["season_voices"]
    fast = run.apply_preset(CONFIG, "fast")
    assert (fast["mode"], fast["voiceover"]["engine"], fast["season_voices"]["enabled"]) == ("fast", "vosk", False)
    assert fast["voiceover"]["engines"] == CONFIG["voiceover"]["engines"]
    previous = run.apply_preset(CONFIG, "previous")
    assert (previous["mix"]["rest_duck_db"], previous["mix"]["voice_gain_db"], previous["mix"]["loudness_lufs"]) == (3.0, 0.0, None)
    assert CONFIG["mix"]["voice_gain_db"] == 3.0  # the config itself stays


def test_preset_names() -> None:
    assert run.apply_preset(CONFIG, None) is CONFIG
    with pytest.raises(SystemExit, match="nope"):
        run.apply_preset(CONFIG, "nope")


def test_episode_engine_is_one_of_several() -> None:
    args = argparse.Namespace(season_glossary=None, engine=None, mode="fast", start=None, duration=None, force=None, until="mix")
    episode = run.Episode(Path("/video/s/ep.mkv"), Path("/tmp/none"), args, {**CONFIG, "paths": {}})
    assert episode.engine == "f5"


def test_form_follows_the_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dub_app, "base_config", lambda: CONFIG)
    main, fast = dub_app.preset_form("main"), dub_app.preset_form("fast")
    assert sorted(main[0]) == sorted(dub_app.ENGINES)
    assert dub_app.MODES[main[1]] == "accurate"
    assert main[4] is True  # stress check
    assert fast[0] == ["Vosk"]
    assert dub_app.MODES[fast[1]] == "fast"
    assert fast[4] is False


def test_start_passes_the_preset_and_the_form_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dub_app, "base_config", lambda: json.loads(json.dumps(CONFIG)))
    seen: dict[str, Any] = {}
    monkeypatch.setattr(dub_app, "start_job", lambda *values, **kwargs: seen.update(values=values, **kwargs) or "ok")
    assert dub_app.start_with_preset("folder", ["ep.mkv"], "fast") == "ok"
    assert seen == {"values": ("folder", ["ep.mkv"]), "preset": "fast"}
    config = dub_app.job_config(list(dub_app.MODES)[0], list(dub_app.MODELS)[0], True, True, True, True, preset="upload")
    assert config["mix"]["video"]["compress"] is True
    assert config["mode"] == "accurate"
    config = dub_app.job_config(list(dub_app.MODES)[0], list(dub_app.MODELS)[0], True, True, True, True, preset="fast")
    assert config["mode"] == "accurate"  # a field the person set on the form
    assert config["season_voices"]["enabled"] is False


def test_trained_model_falls_back_when_its_weights_are_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(refine_speakers, "model_available", lambda name: name != "resnet293-ft")
    assert run.season_voices_model(CONFIG) == "resnet293"
    monkeypatch.setattr(refine_speakers, "model_available", lambda name: True)
    assert run.season_voices_model(CONFIG) == "resnet293-ft"
    assert refine_speakers.model_available("resnet293") is True  # downloaded when needed, never "missing"


def test_imdb_of_the_show() -> None:
    table = CONFIG["subtitles"]["imdb"]
    assert imdb_of(Path("/m/Survivor/Season 20/S20E01.mkv"), table) == "tt0239195"
    assert imdb_of(Path("/m/The Amazing Race Season 1/x.mkv"), table) == "tt0285335"
    assert imdb_of(Path("/m/Survivor/Australia S10 1080p/Australian.Survivor.S10E01.mkv"), table) is None
    assert imdb_of(Path("/m/Other Show/x.mkv"), table) is None
    assert imdb_of(Path("/m/x.mkv"), "tt1") == "tt1"


class FakeEpisode:
    def __init__(self, media: Path, out_root: Path, args: argparse.Namespace, config: dict[str, Any], lock: Any = None) -> None:
        self.media, self.out = media, out_root / media.stem
        self.season_voices = out_root / "season.npz"

    def run(self) -> None:
        (self.out / "speakers.json").touch()

    def stamp(self, stage: str) -> Path:
        return self.out / ".stages" / stage


@pytest.mark.parametrize("force,rewritten", [(None, ["new"]), ("speakers", ["done", "new"])])
def test_translated_episodes_stay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, force: str | None, rewritten: list[str]) -> None:
    season = tmp_path / "season"
    season.mkdir()
    out = tmp_path / "work"
    files = []
    for name in ("done", "new"):
        (season / f"{name}.mkv").touch()
        (out / name).mkdir(parents=True)
        files.append(season / f"{name}.mkv")
    (out / "done" / "subs.ru.json").touch()
    seen: dict[str, Any] = {}

    def apply(every: list[Path], targets: list[Path], voices: Path, **_: Any) -> dict[str, Any]:
        seen.update(every=[p.name for p in every], targets=[p.name for p in targets])
        return {p.name: season_voices.Result("unchanged", "") for p in targets}

    monkeypatch.setattr(run, "Episode", FakeEpisode)
    monkeypatch.setattr(season_voices, "apply_season", apply)
    monkeypatch.setattr(run, "season_voices_model", lambda _: "resnet293")
    args = argparse.Namespace(until="mix", force=force, engine="f5")
    run.run_season_voices(files, out, args, {"season_voices": {}}, ["f5"])
    assert seen == {"every": ["done", "new"], "targets": rewritten}


def test_a_season_all_translated_needs_no_voices(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "season").mkdir()
    media = tmp_path / "season" / "done.mkv"
    media.touch()
    (tmp_path / "work" / "done").mkdir(parents=True)
    (tmp_path / "work" / "done" / "subs.ru.json").touch()

    def never(*_: Any, **__: Any) -> None:
        raise AssertionError("no embeddings for a finished season")

    monkeypatch.setattr(run, "Episode", FakeEpisode)
    monkeypatch.setattr(season_voices, "apply_season", never)
    monkeypatch.setattr(run, "season_voices_model", lambda _: "resnet293")
    run.run_season_voices([media], tmp_path / "work", argparse.Namespace(until="mix", force="mix", engine="f5"), {"season_voices": {}}, ["f5"])
