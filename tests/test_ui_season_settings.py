"""Settings per season: the layers, the season's file, the global file, the question when the season changes, the notes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ui import dub_app, season_settings, seasons, settings
from ui import state as ui_state

MODE = ("mode",)


def _form(config: dict[str, Any], **changes: Any) -> list[Any]:
    """The form as the page sends it: the values of ``config`` with a few ``changes`` by dotted path."""
    values = settings.form_values(config)
    index = {".".join(s.path): n for n, s in enumerate(settings.SETTINGS)}
    for key, value in changes.items():
        values[index[key.replace("__", ".")]] = value
    return values


def _standard() -> dict[str, Any]:
    return settings.read_json(settings.DEFAULTS)


def test_layers_standard_then_global_then_season(tmp_path: Path) -> None:
    other = tmp_path / "global.json"
    other.write_text(json.dumps({"mode": "fast", "mix": {"duck_db": 20}, "llm": {"opencode": ["openai/*"]}}), encoding="utf-8")
    ui_state.update({"global_settings": str(other)})
    base = dub_app.base_config()
    assert base["mode"] == "fast" and base["mix"]["duck_db"] == 20 and base["llm"]["opencode"] == ["openai/*"]
    assert base["mix"]["subtitles"] == _standard()["mix"]["subtitles"]  # untouched keys come from stt_config.json
    results = tmp_path / "STT_S"
    results.mkdir()
    (results / "season_settings.json").write_text(json.dumps(
        {"mode": "accurate", "mix": {"duck_db": 9}, "paths": {"work": "/ignored"}, "unknown": 1}), encoding="utf-8")
    config = dub_app.season_base(str(results))
    assert config["mode"] == "accurate" and config["mix"]["duck_db"] == 9 and config["mix"]["subtitles"] is not None
    assert config["paths"] == base["paths"] and "unknown" not in config  # only the keys of the settings tab count
    assert config["llm"]["opencode"] == ["openai/*"]
    config["mix"]["duck_db"] = 1  # the result is a copy
    assert dub_app.base_config()["mix"]["duck_db"] == 20
    assert dub_app.season_base("")["mode"] == "fast"  # no season: global


def test_a_job_and_the_form_use_the_season_settings(tmp_path: Path) -> None:
    results = tmp_path / "STT_S"
    season_settings.save_own(_form(_standard(), mode="Быстрый: только Whisper", mix__subtitles=False), results, _standard())
    config = dub_app.job_config(list(dub_app.MODES)[0], list(dub_app.MODELS)[0], True, True, True, True,
                                base=dub_app.season_base(str(results)))
    assert config["mode"] == "accurate"  # the options of the form come after the season settings
    assert dub_app.form_defaults(dub_app.season_base(str(results)))[1] == "Быстрый: только Whisper"
    assert dub_app.form_defaults(dub_app.season_base(str(tmp_path / "other")))[1] != "Быстрый: только Whisper"
    assert dub_app.preset_form(None, str(results))[1] == "Быстрый: только Whisper"
    assert dub_app.mix_form_defaults(str(results)) == dub_app.mix_form_defaults(str(results))


def test_season_file_is_a_snapshot_of_the_tab_keys(tmp_path: Path) -> None:
    results = tmp_path / "R"  # does not exist yet: made on save
    path = season_settings.save_own(_form(_standard(), mix__duck_db=17), results, _standard())
    assert path == results / "season_settings.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["mix"]["duck_db"] == 17 and "mode" in saved and "paths" not in saved
    assert {tuple(k for k in s.path) for s in settings.SETTINGS if season_settings._lookup(saved, s.path) is
            season_settings._MISSING} <= {s.path for s in settings.SETTINGS if s.nullable}
    assert not list(results.glob(".*.tmp"))
    season_settings.save_own([30], results, _standard(), only=[next(s for s in settings.SETTINGS if s.path == ("mix", "duck_db"))])
    assert season_settings.read_own(results)["mix"]["duck_db"] == 30 and season_settings.read_own(results)["mode"] == saved["mode"]
    with pytest.raises(settings.SettingsError):
        season_settings.save_own([None], results, _standard(), only=[next(s for s in settings.SETTINGS if s.path == ("mix", "duck_db"))])
    assert season_settings.read_own(results)["mix"]["duck_db"] == 30  # nothing written on a bad value


def test_save_and_reset_and_remove_handlers(tmp_path: Path) -> None:
    source = tmp_path / "Season_5"
    source.mkdir()
    results = tmp_path / "STT_Season_5"
    glob = season_settings.global_path()
    values = _form(_standard(), mix__duck_db=14)
    text = dub_app.save_settings(str(source), str(results), dub_app.TARGET_GLOBAL, *values)
    assert str(glob) in text and json.loads(glob.read_text(encoding="utf-8"))["mix"] == {"duck_db": 14.0}
    assert not (results / "season_settings.json").exists()
    text = dub_app.save_settings(str(source), str(results), dub_app.TARGET_SEASON, *_form(_standard(), mix__duck_db=11))
    assert "настройки сезона Season_5" in text and str(results / "season_settings.json") in text
    assert dub_app.season_base(str(results))["mix"]["duck_db"] == 11
    text = dub_app.save_settings(str(source), str(results), dub_app.TARGET_GLOBAL, *values)
    assert "не влияет" in text  # the season keeps its own file
    assert "Не сохранено" in dub_app.save_settings(str(source), str(results), dub_app.TARGET_SEASON,
                                                    *_form(_standard(), mix__duck_db=None))
    assert "сброшены" in dub_app.reset_settings(str(source), str(results), dub_app.TARGET_SEASON)
    assert dub_app.season_base(str(results))["mix"]["duck_db"] == _standard()["mix"]["duck_db"]
    message = dub_app.remove_season_settings(str(source), str(results))
    assert "удалён" in message and not (results / "season_settings.json").exists()
    assert dub_app.season_base(str(results))["mix"]["duck_db"] == 14.0  # global again
    assert "нет своих" in dub_app.remove_season_settings(str(source), str(results))
    dub_app.reset_settings(str(source), str(results), dub_app.TARGET_GLOBAL)
    assert "mix" not in json.loads(glob.read_text(encoding="utf-8"))


def test_sound_tab_saves_where_the_season_settings_live(tmp_path: Path) -> None:
    results = tmp_path / "STT_S"
    values = dub_app.mix_form_defaults(str(results))
    values[2] = 13
    assert "общие настройки" in dub_app.save_mix_settings(str(results), *values)
    assert json.loads(season_settings.global_path().read_text(encoding="utf-8"))["mix"]["duck_db"] == 13.0
    season_settings.save_own(_form(_standard()), results, _standard())
    values[2] = 8
    assert "настройки сезона" in dub_app.save_mix_settings(str(results), *values)
    assert dub_app.season_base(str(results))["mix"]["duck_db"] == 8
    assert json.loads(season_settings.global_path().read_text(encoding="utf-8"))["mix"]["duck_db"] == 13.0


def test_choosing_another_global_file(tmp_path: Path) -> None:
    glob = season_settings.global_path()
    glob.write_text(json.dumps({"mode": "fast", "llm": {"opencode": ["openai/*"]}}), encoding="utf-8")
    new = tmp_path / "sub" / "mine.json"
    status, message = season_settings.choose_global(str(new))
    assert status == "ok" and "копия" in message
    assert json.loads(new.read_text(encoding="utf-8")) == {"mode": "fast", "llm": {"opencode": ["openai/*"]}}
    assert season_settings.global_path() == new and json.loads(ui_state.PATH.read_text(encoding="utf-8"))["global_settings"] == str(new)
    assert season_settings.choose_global(str(new))[0] == "same"
    assert dub_app.apply_global(str(new)) == {"__type__": "update"}  # nothing to say
    assert dub_app.base_config()["mode"] == "fast"
    new.write_text(json.dumps({"mode": "accurate"}), encoding="utf-8")
    assert dub_app.base_config()["mode"] == "accurate"  # the chosen file now rules
    assert season_settings.choose_global(str(tmp_path))[0] == "error"
    assert season_settings.choose_global(str(tmp_path / "x.txt"))[0] == "error"
    assert season_settings.choose_global("")[0] == "error"
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    assert season_settings.choose_global(str(bad))[0] == "error"
    assert season_settings.choose_global(str(glob))[0] == "ok"  # back to the default file: the pointer goes away
    assert "global_settings" not in json.loads(ui_state.PATH.read_text(encoding="utf-8"))
    assert season_settings.global_path() == settings.LOCAL


def test_question_when_a_season_without_settings_follows_one_with_them(tmp_path: Path) -> None:
    base = _standard()
    old, new = tmp_path / "STT_Old", tmp_path / "STT_New"
    assert not season_settings.should_ask(old, new, base)  # the previous one has no file
    season_settings.save_own(_form(base), old, base)
    assert not season_settings.should_ask(old, new, base)  # a snapshot equal to the global settings
    season_settings.save_own(_form(base, mix__duck_db=3), old, base)
    assert season_settings.should_ask(old, new, base)
    assert not season_settings.should_ask(None, new, base) and not season_settings.should_ask(old, old, base)
    season_settings.copy_own(old, new)
    assert not season_settings.should_ask(old, new, base)  # the new one has its own now
    assert season_settings.read_own(new) == season_settings.read_own(old)


def test_season_change_asks_and_the_answers(tmp_path: Path) -> None:
    drive = tmp_path / "Survivor"
    for number in (1, 2):
        (drive / f"Season_{number}").mkdir(parents=True)
        (drive / f"Season_{number}" / "e1.mkv").write_text("")
        seasons.remember(seasons.Season(drive / f"Season_{number}", drive / f"STT_Season_{number}"), {})
    base = dub_app.base_config()
    one, two = drive / "STT_Season_1", drive / "STT_Season_2"
    season_settings.save_own(_form(base, mix__duck_db=5), one, base)
    # the first start: no previous season, no question
    first = dub_app.select_season(str(drive / "Season_1"), "")
    assert first[6]["visible"] is False and "действуют: настройки сезона" in first[2]
    asked = dub_app.select_season(str(drive / "Season_2"), str(one))
    assert asked[6]["visible"] is True and "Season_2 нет своих настроек" in asked[7]
    assert asked[8]["value"] == "Настройки сезона Season_1" and asked[9] == [str(one), str(two)]
    assert "действуют: общие настройки" in asked[2]
    # a season with its own settings is not asked, and nothing is asked when the previous one has none
    assert dub_app.select_season(str(drive / "Season_1"), str(two))[6]["visible"] is False
    assert dub_app.select_season(str(drive / "Season_2"), "")[6]["visible"] is False
    panel, message = dub_app.answer_settings_question(asked[9], False)
    assert panel["visible"] is False and "общие настройки" in message and not (two / "season_settings.json").exists()
    panel, message = dub_app.answer_settings_question(asked[9], True)
    assert panel["visible"] is False and "скопированные" in message and "Season_1" in message
    assert season_settings.read_own(two) == season_settings.read_own(one)
    assert dub_app.answer_settings_question(None, True)[1] == ""


def test_the_view_follows_the_season(tmp_path: Path) -> None:
    source, results = tmp_path / "Season_3", tmp_path / "STT_Season_3"
    source.mkdir()
    n = len(settings.SETTINGS)
    view = dub_app.settings_view(str(source), str(results), None)
    effect, radio, remove, line, global_file = view[n + 8 + 7:n + 8 + 7 + 5]
    assert effect == "Сейчас для сезона Season_3 действуют: общие настройки." and radio["value"] == "global"
    assert remove["visible"] is False and "общие настройки" in line and global_file == str(season_settings.global_path())
    assert [v for _, v in radio["choices"]] == ["global", "season"]
    season_settings.save_own(_form(_standard()), results, _standard())
    view = dub_app.settings_view(str(source), str(results), None)
    effect, radio, remove = view[n + 15:n + 18]
    assert "настройки сезона" in effect and str(results / "season_settings.json") in effect
    assert radio["value"] == "season" and remove["visible"] is True
    assert dub_app.settings_view(str(source), str(results), "global")[n + 16]["value"] == "global"
    nothing = dub_app.settings_view("", "", None)
    assert nothing[n + 15] == "Сезон не выбран: действуют общие настройки." and [v for _, v in nothing[n + 16]["choices"]] == ["global"]
    assert len(view) == n + 8 + 7 + 5 + len(seasons.NOTE_TABS)


def test_each_tab_says_what_it_writes_where(tmp_path: Path) -> None:
    season = seasons.Season(tmp_path / "Season_48", tmp_path / "STT_Season_48")
    notes = seasons.tab_notes(season, "/home/x/g.json")
    r = str(season.results)
    assert set(notes) == set(seasons.NOTE_TABS)
    assert f"`{r}`" in notes["process"] and "*.ru.mkv" in notes["process"] and "_проба" in notes["process"]
    assert f"`{r}/Season_48.references/`" in notes["voices"] and "Season_48.voices.json" in notes["voices"]
    assert f"`{r}/<серия>/corrections.json`" in notes["edits"]
    assert f"`{r}`" in notes["sound"]
    assert f"`{r}/season_settings.json`" in notes["settings"] and "`/home/x/g.json`" in notes["settings"]
    assert set(seasons.tab_notes(None).values()) == {seasons.NO_SEASON_NOTE}


def test_the_names_in_the_notes_are_the_names_stt_run_writes(tmp_path: Path) -> None:
    import argparse

    from stt import voice_bank
    from stt.run import Episode

    args = argparse.Namespace(season_glossary=None, force=None, engine="vosk")
    media = tmp_path / "Season_48" / "e1.mkv"
    config = {"paths": {"season": str(tmp_path / "STT_Season_48")}, "voiceover": {}}
    episode = Episode(media, tmp_path / "STT_Season_48", args, config)
    assert episode.out == tmp_path / "STT_Season_48" / "e1"
    assert episode.voice_bank.name == "Season_48.voices.json" and episode.season.name == "Season_48.glossary.json"
    assert voice_bank.references_dir(episode.voice_bank).name == "Season_48.references"
