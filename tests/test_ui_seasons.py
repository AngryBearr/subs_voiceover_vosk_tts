"""Seasons of the UI: suggestion, remembering, discovery, episode states, the config of a job."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stt.run import finished_folder
from ui import dub_app, seasons, settings
from ui import state as ui_state


def _videos(folder: Path, *names: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_text("")
    return folder


def test_results_are_suggested_next_to_the_source() -> None:
    assert seasons.suggest_results("/media/d/Survivor/Season_48") == Path("/media/d/Survivor/STT_Season_48")
    assert seasons.suggest_results("/media/d/Survivor/Season_48/") == Path("/media/d/Survivor/STT_Season_48")


def test_plural() -> None:
    assert [seasons.episodes_word(n) for n in (1, 2, 5, 11, 21, 24)] == [
        "1 серия", "2 серии", "5 серий", "11 серий", "21 серия", "24 серии"]


def test_remembering_keeps_the_other_keys(tmp_path: Path) -> None:
    state = tmp_path / "ui.local.json"
    state.write_text(json.dumps({"global_settings": "/x/g.json"}), encoding="utf-8")
    one = seasons.Season(tmp_path / "S1", tmp_path / "STT_S1")
    two = seasons.Season(tmp_path / "S2", tmp_path / "STT_S2")
    seasons.remember(one, {}, state)
    seasons.remember(two, {}, state)
    seasons.remember(one, {}, state)  # again: moved to the end, not doubled
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["global_settings"] == "/x/g.json"
    assert saved["seasons"] == [{"source": str(two.source), "results": str(two.results)},
                                {"source": str(one.source), "results": str(one.results)}]
    assert saved["current_season"] == str(one.source)
    assert seasons.remembered(state) == [two, one]
    assert seasons.pick_current(seasons.remembered(state), state) == one
    assert not list(tmp_path.glob(".*.tmp"))


def test_seasons_move_from_the_old_config_file_without_touching_it(tmp_path: Path) -> None:
    old = {"llm": {"model": "x"}, "ui": {"engines": ["vosk"], "current_season": "/m/S2",
                                         "seasons": [{"source": "/m/S1", "results": "/m/STT_S1"},
                                                     {"source": "/m/S2", "results": "/m/STT_S2"}]}}
    settings.LOCAL.write_text(json.dumps(old), encoding="utf-8")
    state = tmp_path / "ui.local.json"
    assert [s.name for s in seasons.remembered(state)] == ["S1", "S2"] and seasons.current_source(state) == "/m/S2"
    assert not state.exists()  # reading does not write
    seasons.remember(seasons.Season(Path("/m/S3"), Path("/m/STT_S3")), {}, state)
    assert [s.name for s in seasons.remembered(state)] == ["S1", "S2", "S3"]
    assert json.loads(settings.LOCAL.read_text(encoding="utf-8")) == old  # the old file is never changed
    # once the state has seasons of its own, the old file is not looked at any more
    settings.LOCAL.write_text(json.dumps({"ui": {"seasons": [{"source": "/z/Z", "results": "/z/STT_Z"}]}}), encoding="utf-8")
    assert [s.name for s in seasons.remembered(state)] == ["S1", "S2", "S3"]


def test_settings_reset_keeps_the_seasons(tmp_path: Path) -> None:
    state = tmp_path / "ui.local.json"
    local = tmp_path / "stt_config.local.json"
    seasons.remember(seasons.Season(tmp_path / "S1", tmp_path / "STT_S1"), {}, state)
    local.write_text(json.dumps({"mode": "fast", "llm": {"opencode": ["openai/*"]}}), encoding="utf-8")
    settings.reset(local)
    assert len(seasons.remembered(state)) == 1 and json.loads(local.read_text(encoding="utf-8")) == {
        "llm": {"opencode": ["openai/*"]}}


def test_discovery_finds_pairs_next_to_the_finished_folder(tmp_path: Path) -> None:
    drive = tmp_path / "Survivor"
    _videos(drive / "Season_48", "E01.mkv")
    (drive / "STT_Season_48").mkdir()
    _videos(drive / "Season_49", "E01.mkv")  # no results folder yet: not a pair
    (drive / "STT_Season_50").mkdir()  # no source
    (drive / "Season_51").mkdir()  # source without videos
    (drive / "STT_Season_51").mkdir()
    config = {"paths": {"finished": str(drive / "STT_Season_48")}}
    found = seasons.discover(config, media_glob="/nonexistent/*/Survivor")
    assert found == [seasons.Season(drive / "Season_48", drive / "STT_Season_48")]
    assert seasons.available(config, tmp_path / "none.json") == found


def test_discovery_in_the_media_folders(tmp_path: Path) -> None:
    drive = tmp_path / "user" / "disk" / "Survivor"
    _videos(drive / "Season_51", "a.mp4")
    (drive / "STT_Season_51").mkdir()
    found = seasons.discover({}, media_glob=f"{tmp_path}/*/*/Survivor")
    assert [s.name for s in found] == ["Season_51"]


def test_choosing_a_found_season_remembers_all_of_them(tmp_path: Path) -> None:
    drive = tmp_path / "Survivor"
    for number in (48, 51):
        _videos(drive / f"Season_{number}", "a.mkv")
        (drive / f"STT_Season_{number}").mkdir()
    config = {"paths": {"finished": str(drive / "STT_Season_48")}}
    local = tmp_path / "ui.local.json"
    chosen = seasons.choose(str(drive / "Season_51"), config, local)
    assert chosen is not None and chosen.name == "Season_51"
    assert [s.name for s in seasons.remembered(local)] == ["Season_48", "Season_51"]
    assert seasons.current_source(local) == str(drive / "Season_51")
    assert seasons.choose(str(drive / "nothing"), config, local) is None


def test_validation_messages(tmp_path: Path) -> None:
    source = _videos(tmp_path / "Season_1", "e1.mkv", "e2.mkv", "notes.txt")
    assert seasons.describe_source("") == (False, "Впишите путь к папке с оригинальными сериями.")
    assert "нет" in seasons.describe_source(str(tmp_path / "missing"))[1]
    assert not seasons.describe_source(str(tmp_path))[0]  # no videos
    assert seasons.describe_source(str(source)) == (True, "✅ Найдено видео: 2.")
    results = tmp_path / "STT_Season_1"
    assert "будет создана" in seasons.describe_results(str(source), str(results))[1]
    season, problem = seasons.validate(str(source), str(results))
    assert season == seasons.Season(source, results) and problem == ""
    (tmp_path / "file").write_text("")
    assert seasons.validate(str(source), str(tmp_path / "file"))[0] is None
    assert seasons.validate(str(source), str(source))[0] is None
    assert seasons.validate(str(source), str(tmp_path / "file" / "x"))[0] is None


def _mark(folder: Path, *names: str) -> None:
    (folder / ".stages").mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / ".stages" / name).write_text("")


def test_episode_states_from_the_results_folder(tmp_path: Path) -> None:
    source = _videos(tmp_path / "S", "S01E01.mkv", "S01E02.mkv", "S01E03.mkv", "S01E04.mkv", "S01E05.mkv")
    results = tmp_path / "STT_S"
    _mark(results / "S01E02", "transcribe", "glossary", "speakers")
    _mark(results / "S01E03", "transcribe", "dub.vosk", "voiceover.vosk")
    (results / "S01E03" / ".progress").mkdir()
    (results / "S01E03" / ".progress" / "check.vosk.json").write_text(json.dumps(
        {"stage": "check", "engine": "vosk", "state": "running", "pid": 2 ** 22 + 12345, "boot": "gone",
         "started": 1.0, "updated": 1.0}), encoding="utf-8")
    _mark(results / "S01E04", "transcribe")
    (results / "S01E04.ru.mkv").write_text("")
    (results / "S01E01.ru.small.mkv").write_text("")  # a compressed copy is not the finished video
    rows = {r.stem: r for r in seasons.episode_rows(seasons.Season(source, results))}
    assert rows["S01E01"].state == seasons.STATE_NEW and rows["S01E01"].finished is None
    assert rows["S01E02"].state == "3 из 9 этапов"
    assert rows["S01E03"].state.startswith("⏸ прервано") and "Проверка речи" in rows["S01E03"].state
    assert rows["S01E04"].state == "✅ готово" and rows["S01E04"].finished == results / "S01E04.ru.mkv"
    assert rows["S01E05"].state == seasons.STATE_NEW
    table = seasons.table(seasons.Season(source, results))
    assert table[3] == ["S01E04", "✅ готово", "S01E04.ru.mkv"] and table[0][1] == seasons.STATE_NEW
    assert seasons.table(None) == []


def test_ready_video_per_engine_and_older_layout(tmp_path: Path) -> None:
    source = _videos(tmp_path / "S", "a.mkv", "b.mkv")
    results = tmp_path / "R"
    (results / "b").mkdir(parents=True)
    (results / "a.ru.f5.mkv").write_text("")
    (results / "b" / "b.ru.mkv").write_text("")
    rows = {r.stem: r.finished for r in seasons.episode_rows(seasons.Season(source, results))}
    assert rows == {"a": results / "a.ru.f5.mkv", "b": results / "b" / "b.ru.mkv"}


def test_season_config_puts_everything_in_the_results_folder(tmp_path: Path) -> None:
    base = {"paths": {"work": "output/stt", "season": None, "finished": "/old", "finished_folder": "STT_{season}"},
            "mode": "fast"}
    season = seasons.Season(tmp_path / "Season_48", tmp_path / "STT_Season_48")
    config = seasons.season_config(base, season)
    assert base["paths"]["work"] == "output/stt"  # the input is not changed
    results = str(season.results)
    assert config["paths"] == {"work": results, "season": results, "finished": results, "finished_folder": "."}
    assert config["mode"] == "fast"
    assert finished_folder(config, season.source / "Season_48.E01.mkv") == season.results


def test_start_job_runs_in_the_season(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _videos(tmp_path / "Season_9", "e1.mkv")
    results = tmp_path / "STT_Season_9"
    launched: list[dub_app.Job] = []
    monkeypatch.setattr(dub_app, "JOB", None)
    monkeypatch.setattr(dub_app, "_launch", launched.append)
    monkeypatch.setattr(dub_app, "base_config", lambda: {"paths": {"finished": "/elsewhere"}, "mix": {}, "voiceover": {}})
    args: tuple[Any, ...] = (["Vosk"], "Точный: Cohere + Whisper + LLM", "Sol: лучше перевод, дороже", True, True, True, True)
    assert dub_app.start_job("", ["e1.mkv"], "", *args, False, 0, 0) == dub_app.NO_SEASON
    assert dub_app.start_job(str(source), ["e1.mkv"], str(results), *args, False, 0, 0).startswith("⏳ Запущено: Обработка серий")
    job = launched[-1]
    assert job.out_root == results and job.files == [source / "e1.mkv"]
    saved = json.loads(job.config_path.read_text(encoding="utf-8"))
    assert saved["paths"]["finished"] == str(results) and saved["paths"]["finished_folder"] == "."
    dub_app.start_job(str(source), ["e1.mkv"], str(results), *args, True, 10, 2)
    preview = launched[-1]
    assert preview.out_root == results / "_проба" and "--start" in preview.extra
    assert json.loads(preview.config_path.read_text(encoding="utf-8"))["paths"]["finished"] is None


def test_season_handlers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    local = tmp_path / "ui.local.json"
    monkeypatch.setattr(ui_state, "PATH", local)
    monkeypatch.setattr(dub_app, "base_config", lambda: {})
    source = _videos(tmp_path / "Survivor" / "Season_7", "e1.mkv", "e2.mkv")
    note, results = dub_app.source_changed(str(source))
    assert "Найдено видео: 2" in note and results["value"] == str(tmp_path / "Survivor" / "STT_Season_7")
    assert "❌" in dub_app.source_changed(str(tmp_path / "nope"))[0]
    message, picker, form = dub_app.save_season(str(source), results["value"])
    assert message.startswith("✅") and picker["value"] == str(source) and form["visible"] is False
    assert (tmp_path / "Survivor" / "STT_Season_7").is_dir()
    assert dub_app.save_season("", "")[0].startswith("Впишите")
    got = dub_app.select_season(str(source))
    assert got[0] == str(source) and got[1] == results["value"] and [value for _, value in got[3]["choices"]] == ["e1.mkv", "e2.mkv"]
    assert got[5] == [["e1", "не начата", ""], ["e2", "не начата", ""]]
    assert dub_app.select_season(None)[0] == ""
    assert dub_app.season_table("", "") == []
    assert dub_app.season_picker()["choices"][0][1] == str(source)
    assert seasons.current_source(local) == str(source)


def test_settings_page_has_no_folder_fields() -> None:
    assert not [s for s in settings.SETTINGS if s.path[0] == "paths"]


def test_the_season_bar_drives_every_tab() -> None:
    app = dub_app.build()
    chain = [fn for fn in app.fns.values() if getattr(fn, "trigger_after", None) is not None]
    names = {fn.fn.__name__ for fn in chain}
    assert {"on_season", "load_episodes", "mix_episodes", "finished_videos"} <= names
    labels = {str(getattr(b, "label", "")) for b in app.blocks.values()}
    assert "Папка с сериями" not in labels and "Рабочая папка (готовые видео: вкладка «Настройки»)" not in labels
    assert "Сезон" in labels  # the picker


def test_episode_labels_are_short_and_keep_the_file_as_value() -> None:
    from ui.seasons import episode_choices, episode_label

    assert episode_label("Survivor.S48E01.The.Get.to.Know.You.Game.720p.AMZN.WEB-DL.DDP5.1.H.264-RAWR"
                         "(The Get to Know You Game).mkv") == "S48E01 · The Get to Know You Game"
    assert episode_label("Survivor.S48E03.Committing.to.the.Bit.720p.AMZN.WEB-DL.DDP5.1.H.264-RAWR.mkv") == \
        "S48E03 · Committing to the Bit"
    assert episode_label("Survivor.S51E02.Weaponized-Honesty.1080p.AMZN.WEB-DL.DDP2.0.H.264-RAWR.mkv") == \
        "S51E02 · Weaponized-Honesty"
    assert episode_label("Test.S99E01.mkv") == "S99E01" and episode_label("movie.mkv") == "movie"
    assert episode_choices(["Test.S99E01.mkv"]) == [("S99E01", "Test.S99E01.mkv")]
