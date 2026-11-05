"""Sets of settings, the main pipeline's defaults and what happens to episodes that are already translated."""

from __future__ import annotations

import time
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
    assert dub_app.start_with_preset("folder", ["ep.mkv"], False, "fast") == "ok"
    assert seen == {"values": ("folder", ["ep.mkv"]), "download_voice_model": False, "preset": "fast"}
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

    def named_after_translation(self) -> bool:
        return False


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


def test_voice_model_is_downloaded_once_and_a_failure_does_not_stop_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(refine_speakers, "download_model", lambda name: calls.append(name) or Path("weights"))
    monkeypatch.setattr(refine_speakers, "model_available", lambda name: True)
    run.fetch_voice_model(CONFIG)
    assert calls == []  # already here
    monkeypatch.setattr(refine_speakers, "model_available", lambda name: False)
    run.fetch_voice_model(CONFIG)
    assert calls == ["resnet293-ft"]

    def broken(name: str) -> Path:
        raise OSError("no network")

    monkeypatch.setattr(refine_speakers, "download_model", broken)
    run.fetch_voice_model(CONFIG)  # only a warning: the plain model takes over


def test_recount_of_season_voices_keeps_translated_episodes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    videos, work = tmp_path / "season", tmp_path / "work"
    videos.mkdir()
    for name in ("E1", "E2", "E3"):
        (videos / f"{name}.mkv").touch()
    started: list[list[str]] = []
    monkeypatch.setattr(dub_app, "JOB", None)
    monkeypatch.setattr(dub_app, "_run", lambda job: started.append(job.extra + [p.stem for p in job.files]))
    assert "сначала запустите обработку" in dub_app.recount_season_voices(str(videos), str(work))
    for name in ("E1", "E2"):
        (work / name).mkdir(parents=True)
        (work / name / "speakers.json").write_text("{}", encoding="utf-8")
        (work / name / "subs.ru.json").write_text("[]", encoding="utf-8")
    assert "пересчитывать нечего" in dub_app.recount_season_voices(str(videos), str(work))
    assert started == []
    (work / "E2" / "subs.ru.json").unlink()
    assert "реплики изменятся у 1" in dub_app.recount_season_voices(str(videos), str(work), True)
    for _ in range(100):
        if started:
            break
        time.sleep(0.01)
    assert started == [["--season-voices-only", "--download-voice-model", "E1", "E2"]]


def _prepared_season(tmp_path: Path) -> tuple[Path, Path]:
    source, work = tmp_path / "Season_48", tmp_path / "STT_Season_48"
    source.mkdir()
    for name in ("S48E01.mkv", "S48E02.mkv", "S48E03.mkv"):
        (source / name).write_text("")
    for name, speakers in (("S48E01", [{"id": "S01", "name": "Anna", "cues": 30}]),
                           ("S48E02", [{"id": "S01", "name": "Anna", "cues": 30}, {"id": "S02", "cues": 41}])):
        (work / name).mkdir(parents=True)
        (work / name / "speakers.json").write_text(json.dumps(speakers), encoding="utf-8")
        (work / name / "subs.en.json").write_text("[]", encoding="utf-8")
    (work / "S48E01" / "subs.ru.json").write_text("[]", encoding="utf-8")
    return source, work


def test_prepare_sets_until_and_rejects_another_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video = tmp_path / "S48E01.mkv"
    video.write_text("")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(run, "check_subscription", lambda config: True)
    monkeypatch.setattr(run, "run_lanes", lambda files, out_root, args, config, engines: seen.update(until=args.until))
    monkeypatch.setattr(run, "prepare_summary", lambda files, out_root: [])
    config = tmp_path / "config.json"
    config.write_text(json.dumps(CONFIG), encoding="utf-8")
    base = [str(video), "--out-root", str(tmp_path / "R"), "--config", str(config), "--no-season-voices"]
    run.main([*base, "--prepare"])
    assert seen["until"] == "speakers"
    run.main([*base, "--prepare", "--until", "speakers"])
    with pytest.raises(SystemExit):
        run.main([*base, "--prepare", "--until", "translate"])
    run.main(base)
    assert seen["until"] == "mix"


def test_prepare_summary(tmp_path: Path) -> None:
    source, work = _prepared_season(tmp_path)
    text = "\n".join(run.prepare_summary([source / "S48E01.mkv", source / "S48E02.mkv"], work))
    assert "2 of 3 episodes have speakers.json" in text
    assert "already translated, kept as they are: S48E01" in text
    assert "S48E02: unnamed S02 41 lines" in text
    assert "S48E01: unnamed" not in text
    assert "«Кто это?»" in text


def test_prepare_season_button_takes_all_videos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source, work = _prepared_season(tmp_path)
    launched: list[dub_app.Job] = []
    monkeypatch.setattr(dub_app, "JOB", None)
    monkeypatch.setattr(dub_app, "_launch", launched.append)
    monkeypatch.setattr(dub_app, "base_config", lambda: json.loads(json.dumps(CONFIG)))
    mode, model = list(dub_app.MODES)[0], list(dub_app.MODELS)[0]
    form = (str(source), ["S48E01.mkv"], str(work), [], mode, model, True, True, True, True, True, 10, 2, False, None, False, "main")
    answer = dub_app.prepare_season(*form)
    assert answer.startswith("⏳ Запущено: Подготовка сезона") and "3 серии: распознавание, без перевода" in answer
    job = launched[-1]
    assert job.title == dub_app.PREPARE_TITLE and job.out_root == work  # the trial piece is ignored
    assert job.files == [source / n for n in ("S48E01.mkv", "S48E02.mkv", "S48E03.mkv")]
    assert job.extra == ["--prepare"] and job.engines == [dub_app.ENGINES[list(dub_app.ENGINES)[0]]]
