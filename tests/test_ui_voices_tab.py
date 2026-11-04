"""Tab "Голоса сезона": the people table, cutting a clip by hand (subprocess), saving / removing it, the re-voice banner,
the names check, and the buttons of the tab answering at once."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from stt import voice_bank
from stt.voice_bank import VoiceBank
from ui import dub_app, voices


def _person(bank: VoiceBank, name: str, label: str, gender: str | None, speech: float, episode: str = "S48E01.ep") -> None:
    bank.prints.append({"episode": episode, "speaker": "S1", "name": name, "label_ru": label, "gender": gender,
                        "speech_s": speech, "embedding": [1.0, 0.0], "samples": []})


@pytest.fixture
def season(tmp_path: Path) -> tuple[Path, Path, VoiceBank]:
    """(source folder, results folder, bank) of a small season: three people, one of them with a clip."""
    source, results = tmp_path / "Season_48", tmp_path / "STT_Season_48"
    source.mkdir()
    (source / "S48E01.ep.mkv").write_text("")
    results.mkdir()
    bank = VoiceBank(results / "Season_48.voices.json")
    _person(bank, "Ann Lee", "Энн", "female", 600.0)
    _person(bank, "Bob Ray", "Боб", "male", 1200.0)
    _person(bank, "Cy Dunn", "Саймон", None, 60.0)
    bank.save()
    refs = voice_bank.references_dir(bank.path)
    refs.mkdir()
    sf.write(str(refs / "Ann_Lee.wav"), np.zeros(24_000, np.float32), 24_000)
    (refs / "Ann_Lee.txt").write_text("hello", encoding="utf-8")
    (refs / "references.json").write_text(json.dumps([{
        "name": "Ann Lee", "label_ru": "Энн", "gender": "female", "wav": "Ann_Lee.wav", "text": "hello",
        "episode": "Survivor.S48E03.x", "start": 750.0, "end": 760.0, "score": 0.87, "music_db": -8.0}]), encoding="utf-8")
    (refs / "missing.json").write_text(json.dumps({"Bob Ray": {"episodes": 1, "stretches": 3, "other_voices": 3}}), encoding="utf-8")
    return source, results, bank


# ---------------------------------------------------------------- the people table

def test_table_rows_are_sorted_and_labelled(season: tuple[Path, Path, VoiceBank]) -> None:
    _, _, bank = season
    rows = voices.table_rows(voices.read_states(bank.path))
    assert [r[0] for r in rows] == ["Bob Ray", "Cy Dunn", "Ann Lee"]  # no clip first, most speech first, then the clips
    assert rows[0] == ["Bob Ray", "Боб", "муж.", 20.0, "❌ нет", "3 кандидата: в 3 рядом звучат другие голоса"]
    assert rows[1][2:5] == ["—", 1.0, "❌ нет"] and "нажмите «Найти другой автоматически»" in rows[1][5]
    assert rows[2] == ["Ann Lee", "Энн", "жен.", 10.0, "✅ найден сам", "S48E03 · 12:30 · сходство 0.87"]


def test_a_hand_made_clip_has_its_own_label_and_no_similarity_when_unknown() -> None:
    manual = {"name": "A", "state": "manual", "episode": "Survivor.S48E01.x", "start": 118.4, "score": 0.0}
    assert voices.detail_text(manual) == "S48E01 · 1:58.4"
    assert voices.table_rows([{**manual, "speech_s": 60.0, "gender": "male"}])[0][4] == "✋ указан вручную"
    assert voices.table_rows([]) == [] and voices.read_states(Path("/nonexistent/S.voices.json")) == []


def test_the_card_text_says_what_is_known(season: tuple[Path, Path, VoiceBank]) -> None:
    _, _, bank = season
    states = voices.read_states(bank.path)
    text = voices.card_text(voices.find_state(states, "Cy Dunn"))  # type: ignore[arg-type]
    assert "Саймон" in text and "пол не известен" in text and "❌ нет" in text
    assert voices.find_state(states, "Nobody") is None and voices.find_state(states, None) is None


# ---------------------------------------------------------------- cutting a clip

def test_range_errors_are_russian_and_come_first(season: tuple[Path, Path, VoiceBank]) -> None:
    source, results, bank = season
    states = voices.read_states(bank.path)
    ann, cy = voices.find_state(states, "Ann Lee"), voices.find_state(states, "Cy Dunn")
    episode = "S48E01.ep.mkv"
    assert voices.check_cut(source, None, episode, "1:00-1:05", None, None) == "❌ Сначала выберите человека в таблице."
    assert voices.check_cut(source, "Ann Lee", episode, "", ann, None).startswith("❌ Не указан ни один отрезок")
    assert voices.check_cut(source, "Ann Lee", episode, "12:30", ann, None).startswith("❌ Не понял отрезок «12:30»")
    assert "конец не позже начала" in voices.check_cut(source, "Ann Lee", episode, "2:00-1:00", ann, None)  # type: ignore[operator]
    assert "длиннее 60 секунд" in voices.check_cut(source, "Ann Lee", episode, "0:00-3:00", ann, None)  # type: ignore[operator]
    assert voices.check_cut(source, "Ann Lee", None, "1:00-1:05", ann, None) == "❌ Выберите серию, из которой берутся отрезки."
    assert "Нет файла с видео" in voices.check_cut(source, "Ann Lee", "gone.mkv", "1:00-1:05", ann, None)  # type: ignore[operator]
    assert "не знает пол" in voices.check_cut(source, "Cy Dunn", episode, "1:00-1:05", cy, None)  # type: ignore[operator]
    assert voices.check_cut(source, "Cy Dunn", episode, "1:00-1:05", cy, "male") is None
    assert voices.check_cut(source, "Ann Lee", episode, "1:58.4-2:03.0\n2:03.5-2:05.7", ann, None) is None


def test_the_cut_runs_in_a_subprocess_and_reads_the_json_line(season: tuple[Path, Path, VoiceBank], tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    source, results, bank = season
    seen: dict[str, Any] = {}
    answer = {"wav": "x.wav", "duration": 7.1, "text": "hi", "warnings": ["w"], "similarity": 0.68, "music_db": 0.6,
              "pieces": [[118.4, 123.0], [123.5, 125.7]]}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(cmd=cmd, kwargs=kwargs)
        return subprocess.CompletedProcess(cmd, 0, "some library noise\n" + json.dumps(answer, ensure_ascii=False) + "\n", "log")

    monkeypatch.setattr(voices.subprocess, "run", fake_run)
    found = voices.cut_clip(bank.path, "Cy Dunn", source / "S48E01.ep.mkv", "1:58.4-2:03.0\n2:03.5-2:05.7", "cpu", "male",
                            preview_dir=tmp_path / "prev")
    assert found == answer
    cmd = seen["cmd"]
    assert cmd[1:4] == ["-m", "stt.voice_bank", str(bank.path)] and "--manual-clip" in cmd and "--json" in cmd
    assert cmd[cmd.index("--person") + 1] == "Cy Dunn" and cmd[cmd.index("--device") + 1] == "cpu"
    assert [cmd[i + 1] for i, x in enumerate(cmd) if x == "--range"] == ["1:58.4-2:03", "2:03.5-2:05.7"]
    assert cmd[cmd.index("--gender") + 1] == "male" and cmd[cmd.index("--preview") + 1].startswith(str(tmp_path / "prev"))
    assert seen["kwargs"]["capture_output"] is True  # nothing runs in this process


def test_cut_failures_are_short_russian_messages(season: tuple[Path, Path, VoiceBank], tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    source, _, bank = season
    media = source / "S48E01.ep.mkv"

    def cut() -> dict[str, Any]:
        return voices.cut_clip(bank.path, "Ann Lee", media, "1:00-1:05", "cuda", preview_dir=tmp_path / "prev")

    monkeypatch.setattr(voices.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 2, json.dumps({"error": "Ни в одном из отрезков не нашлось голоса."}, ensure_ascii=False) + "\n", ""))
    assert cut() == {"error": "Ни в одном из отрезков не нашлось голоса."}
    trace = "Traceback (most recent call last):\n  File \"x\", line 1\nRuntimeError: CUDA out of memory\n"
    monkeypatch.setattr(voices.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", trace))
    error = cut()["error"]
    assert error.startswith("Не получилось вырезать образец (RuntimeError: CUDA out of memory)") and "Traceback" not in error

    def timeout(cmd: list[str], **kw: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(voices.subprocess, "run", timeout)
    assert "слишком долго" in cut()["error"]


def test_the_info_under_the_player_lists_the_warnings() -> None:
    text = voices.clip_info({"duration": 7.14, "similarity": 0.68, "warnings": ["Музыка громкая.", "Мало речи."]})
    assert "**7.1 с**" in text and "**0.68**" in text and "⚠ Музыка громкая." in text and "⚠ Мало речи." in text
    assert voices.clip_info({"duration": 4.0, "similarity": None, "warnings": []}) == \
        "Речи в клипе: **4.0 с**\n\nПредупреждений нет."


# ---------------------------------------------------------------- saving and removing

def _cut_state(tmp_path: Path, person: str = "Cy Dunn") -> dict[str, Any]:
    wav = tmp_path / "cut.wav"
    sf.write(str(wav), np.concatenate([np.full(24_000 * 7, 0.2, np.float32), np.zeros(24_000, np.float32)]), 24_000)
    return {"wav": str(wav), "duration": 7.0, "text": "whisper", "warnings": [], "similarity": 0.7, "music_db": -3.0,
            "pieces": [[118.4, 123.0], [123.5, 125.7]], "person": person, "episode": "S48E01.ep.mkv"}


def test_save_makes_a_hand_made_clip_and_remove_returns_the_automatic_one(season: tuple[Path, Path, VoiceBank],
                                                                           tmp_path: Path) -> None:
    source, results, bank = season
    refs = voice_bank.references_dir(bank.path)
    message = voices.save_clip(results, "Season_48", source, "Ann Lee", _cut_state(tmp_path, "Ann Lee"), " my text ", None)
    assert message.startswith("✅ Образец сохранён: Ann Lee — ✋ указан вручную")
    row = voices.find_state(voices.read_states(bank.path), "Ann Lee")
    assert row is not None and row["state"] == "manual" and row["text"] == "my text"
    assert (refs / "Ann_Lee.manual.wav").is_file() and (refs / "Ann_Lee.manual.txt").read_text(encoding="utf-8") == "my text\n"
    manual = json.loads((refs / "manual.json").read_text(encoding="utf-8"))
    assert manual[0]["ranges"] == [[118.4, 123.0], [123.5, 125.7]] and manual[0]["episode"] == "S48E01.ep"
    assert "✋ указан вручную" in voices.table_rows([row])[0]
    assert voices.remove_clip(results, "Season_48", "Ann Lee") == "✅ Ручной образец убран: у Ann Lee снова автоматический образец."
    row = voices.find_state(voices.read_states(bank.path), "Ann Lee")
    assert row is not None and row["state"] == "auto" and not (refs / "Ann_Lee.manual.wav").exists()
    assert voices.remove_clip(results, "Season_48", "Ann Lee") == "У Ann Lee нет ручного образца."


def test_save_errors_are_messages_not_crashes(season: tuple[Path, Path, VoiceBank], tmp_path: Path) -> None:
    source, results, bank = season
    state = _cut_state(tmp_path)
    assert voices.save_clip(results, "Season_48", source, "Cy Dunn", None, "x", "male").startswith("❌ Сначала нажмите «Вырезать")
    assert voices.save_clip(results, "Season_48", source, "Ann Lee", state, "x", None).startswith("❌ Сначала нажмите")  # other person
    assert voices.save_clip(results, "Season_48", source, "Cy Dunn", state, "  ", "male") == \
        "❌ Не сохранено: Нужен текст клипа: впишите, что в нём говорят."
    assert "укажите male или female" in voices.save_clip(results, "Season_48", source, "Cy Dunn", state, "text", None)
    assert not (voice_bank.references_dir(bank.path) / "manual.json").exists()
    assert voices.save_clip(results, "Season_48", source, "Cy Dunn", state, "text", "male").startswith("✅")
    row = voices.find_state(voices.read_states(bank.path), "Cy Dunn")
    assert row is not None and row["state"] == "manual"
    assert "нажмите «Найти другой автоматически»" in voices.remove_clip(results, "Season_48", "Cy Dunn")


def test_commands_of_the_automatic_search(season: tuple[Path, Path, VoiceBank]) -> None:
    _, _, bank = season
    refs = str(voice_bank.references_dir(bank.path))
    assert voices.another_command(bank.path, "Ann Lee")[1:] == ["-m", "stt.voice_bank", str(bank.path), "--export", refs,
                                                                 "--person", "Ann Lee", "--reject"]
    assert voices.rebuild_command(bank.path)[1:] == ["-m", "stt.voice_bank", str(bank.path), "--export", refs]


# ---------------------------------------------------------------- the re-voice banner (from files)

def _voiced(results: Path, source: Path, refs: Path, tmp_path: Path) -> Path:
    """The ESpeech episode of the season, scripted with Ann's clip as it is now."""
    from voiceover.engine_f5 import voice_id

    folder = results / "S48E01.ep"
    (folder / ".stages").mkdir(parents=True)
    for mark in ("voiceover.f5", "dub.f5"):
        (folder / ".stages" / mark).write_text("")
    (folder / "speakers.json").write_text(json.dumps([{"id": "S01", "name": "Ann Lee"}, {"id": "S02", "name": "Bob Ray"}]),
                                          encoding="utf-8")
    (folder / "speaker_voices.f5.json").write_text(json.dumps({"S01": voice_id(refs / "Ann_Lee.wav")}), encoding="utf-8")
    return folder


def test_no_banner_while_the_clips_are_the_ones_the_episodes_were_scripted_with(season: tuple[Path, Path, VoiceBank],
                                                                                 tmp_path: Path) -> None:
    source, results, bank = season
    refs = voice_bank.references_dir(bank.path)
    _voiced(results, source, refs, tmp_path)
    assert voices.banner(results, source, "Season_48") == ("", [])
    # a moved folder is not a change either: only the clip and its text count
    other = tmp_path / "moved"
    other.mkdir()
    (other / "Ann_Lee.wav").write_bytes((refs / "Ann_Lee.wav").read_bytes())
    (other / "Ann_Lee.txt").write_text("hello", encoding="utf-8")
    from voiceover.engine_f5 import voice_id

    (results / "S48E01.ep" / "speaker_voices.f5.json").write_text(json.dumps({"S01": voice_id(other / "Ann_Lee.wav")}), encoding="utf-8")
    assert voices.banner(results, source, "Season_48")[0] == ""
    assert voices.esp_folders(results, tmp_path / "nowhere") == []


def test_the_banner_names_the_people_whose_clip_changed_and_counts_episodes(season: tuple[Path, Path, VoiceBank],
                                                                            tmp_path: Path) -> None:
    source, results, bank = season
    refs = voice_bank.references_dir(bank.path)
    _voiced(results, source, refs, tmp_path)
    # a hand-made clip for Bob (new voice) and another one for Ann (replaced voice)
    assert voices.save_clip(results, "Season_48", source, "Bob Ray", _cut_state(tmp_path, "Bob Ray"), "bob", None).startswith("✅")
    assert voices.save_clip(results, "Season_48", source, "Ann Lee", _cut_state(tmp_path, "Ann Lee"), "ann", None).startswith("✅")
    text, todo = voices.banner(results, source, "Season_48")
    assert text.startswith("Образцы изменились: Ann Lee, Bob Ray. Чтобы новые голоса попали в озвучку ESpeech, переозвучьте серии.")
    assert "Заново озвучатся только реплики этих людей" in text and todo == ["S48E01.ep.mkv"]
    assert voices.revoice_label(1) == "Переозвучить ESpeech (1 серия)" and voices.revoice_label(5) == "Переозвучить ESpeech (5 серий)"
    # back to the automatic clip: Ann is as she was scripted; Bob had no clip when it was scripted and has none again
    voices.remove_clip(results, "Season_48", "Ann Lee")
    voices.remove_clip(results, "Season_48", "Bob Ray")
    assert voices.banner(results, source, "Season_48") == ("", [])


def test_an_episode_without_the_record_is_behind_when_the_index_is_newer(season: tuple[Path, Path, VoiceBank], tmp_path: Path) -> None:
    source, results, bank = season
    refs = voice_bank.references_dir(bank.path)
    folder = _voiced(results, source, refs, tmp_path)
    (folder / "speaker_voices.f5.json").unlink()
    old = time.time() - 1000
    for mark in (folder / ".stages").iterdir():
        os.utime(mark, (old, old))
    text, todo = voices.banner(results, source, "Season_48")
    assert todo == ["S48E01.ep.mkv"] and text.startswith("Образцы изменились после последней озвучки.")
    os.utime(refs / "references.json", (old - 100, old - 100))
    assert voices.banner(results, source, "Season_48") == ("", [])


# ---------------------------------------------------------------- names check

def test_recheck_messages_and_writing_the_fixes(season: tuple[Path, Path, VoiceBank], monkeypatch: pytest.MonkeyPatch) -> None:
    _, results, bank = season
    assert voices.recheck_result(bank.path) == ([], "✅ Расхождений нет: имена людей согласованы по всему сезону.")
    assert voices.recheck_result(results / "none.voices.json")[1] == voices.NO_BANK
    suggestion = {"episode": "S48E01.ep", "speaker": "S01", "name": "Ann", "suggested": "Ann Lee", "label_ru": "Энн",
                  "support": 3, "own_support": 0, "score": 0.8}
    monkeypatch.setattr(voice_bank, "recheck", lambda b: [suggestion])
    found, message = voices.recheck_result(bank.path)
    assert found == [suggestion] and message.startswith("Найдено расхождений: 1.")
    assert voices.recheck_rows(found) == [["S48E01", "S01: Ann", "Ann Lee", "3 (за прежнее имя: 0)"]]
    (results / "S48E01.ep").mkdir()
    _, done = voices.write_fixes(bank.path)
    assert done.startswith("✅ Исправления записаны для серий: S48E01") and "вкладка «Обработка»" in done
    written = json.loads((results / "S48E01.ep" / "speakers.override.json").read_text(encoding="utf-8"))
    assert written["S01"]["name"] == "Ann Lee"


# ---------------------------------------------------------------- the buttons of the tab

@pytest.fixture(scope="module")
def app() -> Any:
    return dub_app.build()


def _fn(app: Any, name: str) -> Any:
    return next(fn.fn for fn in app.fns.values() if fn.fn.__name__ == name)


def test_the_tab_replaces_the_voice_bank_and_the_processing_tab_lost_the_recount_button(app: Any) -> None:
    labels = [str(getattr(b, "label", "")) for b in app.blocks.values()]
    values = [str(getattr(b, "value", "")) for b in app.blocks.values()]
    assert "Голоса сезона" in labels and "Банк голосов" not in labels
    assert "Пересчитать голоса сезона" not in values
    assert {"Пересобрать все образцы", "Уточнить, кто говорит", "Проверить имена", "Записать исправления имён",
            "Вырезать и послушать", "Найти другой автоматически", "Вернуть автоматический выбор"} <= set(values)


def test_cut_button_validates_at_once_then_waits(app: Any, season: tuple[Path, Path, VoiceBank], monkeypatch: pytest.MonkeyPatch) -> None:
    source, results, _ = season
    monkeypatch.setattr(dub_app, "JOB", None)
    started = _fn(app, "cut_started")
    args = (str(source), str(results), "Ann Lee", "S48E01.ep.mkv")
    button, message = started(*args, "12:30", None)
    assert button["interactive"] is True and message.startswith("❌ Не понял отрезок «12:30»")  # nothing waits
    button, message = started(*args, "12:30-12:41", None)
    assert button["interactive"] is False and button["value"] == "⏳ Вырезаю…"
    assert message == "⏳ Вырезаю, очищаю от музыки и распознаю текст — около 15 секунд."
    busy = dub_app.Job(files=[], out_root=results, engines=[], config_path=Path(), log_path=results / "t.log",
                               title="Сбор образцов голосов", cmd=["x"], state="идёт")
    monkeypatch.setattr(dub_app, "JOB", busy)
    monkeypatch.setattr(dub_app, "job_alive", lambda job: True)
    _, message = started(*args, "12:30-12:41", None)
    assert message == "⏳ Сейчас идёт «Сбор образцов голосов»: вырезаю на процессоре, это медленнее — около минуты."


def test_cut_runs_on_the_processor_while_a_heavy_task_runs(app: Any, season: tuple[Path, Path, VoiceBank],
                                                           monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source, results, bank = season
    seen: list[str] = []
    state = _cut_state(tmp_path, "Ann Lee")
    monkeypatch.setattr(voices, "cut_clip", lambda bank_, name, media, ranges, device, gender=None, **kw:
                        seen.append(device) or {**state, "wav": state["wav"], "warnings": ["Музыка громкая."]})
    done = _fn(app, "cut_done")
    args = (str(source), str(results), "Ann Lee", "S48E01.ep.mkv", "1:00-1:05", None)
    monkeypatch.setattr(dub_app, "JOB", None)
    result = done(*args)
    assert seen == ["cuda"] and result[0]["interactive"] is True and result[1].startswith("✅ Готово")
    assert result[2]["value"] == state["wav"] and "Музыка громкая." in result[3] and result[4]["value"] == "whisper"
    assert result[5]["value"] == "Сохранить как образец Ann Lee" and result[6]["person"] == "Ann Lee"
    monkeypatch.setattr(dub_app, "_busy", lambda: "Сейчас идёт «X» (1 мин). Дождитесь или остановите её.")
    done(*args)
    assert seen == ["cuda", "cpu"]
    # a bad range at the second step leaves the message of the first step in place
    result = done(str(source), str(results), "Ann Lee", "S48E01.ep.mkv", "oops", None)
    assert result[0]["interactive"] is True and seen == ["cuda", "cpu"] and result[1] == {"__type__": "update"}


def test_a_failed_cut_shows_the_error_and_no_save_button(app: Any, season: tuple[Path, Path, VoiceBank],
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    source, results, _ = season
    monkeypatch.setattr(dub_app, "JOB", None)
    monkeypatch.setattr(voices, "cut_clip", lambda *a, **k: {"error": "Нет файла с видео."})
    result = _fn(app, "cut_done")(str(source), str(results), "Ann Lee", "S48E01.ep.mkv", "1:00-1:05", None)
    assert result[1] == "❌ Нет файла с видео." and result[2]["visible"] is False and result[5]["visible"] is False


def test_the_other_clip_button_starts_a_task_and_refuses_for_a_manual_clip(app: Any, season: tuple[Path, Path, VoiceBank],
                                                                             monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source, results, bank = season
    started: list[tuple[str, list[str], Path]] = []
    monkeypatch.setattr(dub_app, "start_task", lambda title, cmd, root, result="": started.append((title, cmd, root))
                        or dub_app.started_text(title))
    other = _fn(app, "run_another")
    assert other(str(source), str(results), None).startswith("❌ Сначала выберите человека")
    assert other(str(source), str(results), "Bob Ray") == "⏳ Запущено: Другой образец: Bob Ray. Ход — в строке состояния вверху."
    assert started[0][0] == "Другой образец: Bob Ray" and "--reject" in started[0][1] and started[0][2] == results
    voices.save_clip(results, "Season_48", source, "Ann Lee", _cut_state(tmp_path, "Ann Lee"), "ann", None)
    assert "ручной образец" in other(str(source), str(results), "Ann Lee") and len(started) == 1


def test_season_buttons_start_tasks_with_titles(app: Any, season: tuple[Path, Path, VoiceBank], monkeypatch: pytest.MonkeyPatch) -> None:
    source, results, bank = season
    started: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(dub_app, "start_task", lambda title, cmd, root, result="": started.append((title, cmd))
                        or dub_app.started_text(title))
    assert _fn(app, "run_rebuild")(str(source), str(results)).startswith("⏳ Запущено: Сбор образцов голосов.")
    assert started[0][1][-2:] == ["--export", str(voice_bank.references_dir(bank.path))]
    assert _fn(app, "run_rebuild")(str(source), str(results / "none")) == voices.NO_BANK
    seen: list[tuple[str, str, bool]] = []
    monkeypatch.setattr(dub_app, "recount_season_voices", lambda folder, root, download=False: seen.append((folder, root, download)) or "ok")
    assert _fn(app, "run_recount")(str(source), str(results), True) == "ok" and seen == [(str(source), str(results), True)]


def test_the_revoice_button_runs_stt_run_for_the_episodes_behind(app: Any, season: tuple[Path, Path, VoiceBank], tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    source, results, bank = season
    refs = voice_bank.references_dir(bank.path)
    revoice = _fn(app, "do_revoice")
    assert revoice(str(source), str(results)).startswith("Все серии уже озвучены")
    _voiced(results, source, refs, tmp_path)
    voices.save_clip(results, "Season_48", source, "Ann Lee", _cut_state(tmp_path, "Ann Lee"), "ann", None)
    seen: dict[str, Any] = {}

    def fake(folder: str, names: list[str], out_root: str, engines: list[str], *values: Any, **kw: Any) -> str:
        seen.update(folder=folder, names=names, out_root=out_root, engines=engines, kw=kw)
        return dub_app.started_text(kw["title"])

    monkeypatch.setattr(dub_app, "start_job", fake)
    assert revoice(str(source), str(results)) == "⏳ Запущено: Переозвучка ESpeech. Ход — в строке состояния вверху."
    assert seen["names"] == ["S48E01.ep.mkv"] and seen["engines"] == ["ESpeech (голоса героев)"] and seen["out_root"] == str(results)
    assert seen["kw"]["stt_args"] == ["--no-season-voices"]


def test_the_names_buttons_wait_then_answer(app: Any, season: tuple[Path, Path, VoiceBank]) -> None:
    source, results, bank = season
    button, message = _fn(app, "check_started")()
    assert button["interactive"] is False and message.startswith("⏳ Сравниваю голоса")
    button, message, table = _fn(app, "check_done")(str(source), str(results))
    assert button["interactive"] is True and message.startswith("✅ Расхождений нет") and table["visible"] is False
    assert _fn(app, "check_done")("", "")[1] == dub_app.NO_SEASON
    button, message = _fn(app, "fixes_started")()
    assert button["interactive"] is False and message.startswith("⏳")


def test_the_table_click_picks_the_person_and_the_card_follows(app: Any, season: tuple[Path, Path, VoiceBank]) -> None:
    source, results, _ = season
    card = _fn(app, "card_view")(str(source), str(results), "Ann Lee")
    assert "Ann Lee" in card[0] and card[1]["visible"] is True and card[1]["value"].endswith("Ann_Lee.wav")
    assert "hello" in card[2] and card[3]["visible"] is True and card[4]["visible"] is False  # "another" yes, "revert" no
    missing = _fn(app, "card_view")(str(source), str(results), "Cy Dunn")
    assert missing[1]["visible"] is False and missing[5]["open"] is True and missing[6]["visible"] is True  # gender asked
    nobody = _fn(app, "card_view")(str(source), str(results), None)
    assert nobody[0] == "" and nobody[5]["visible"] is False
    shown = _fn(app, "on_season")(str(source), str(results))
    assert [r[0] for r in shown[0]["value"]] == ["Bob Ray", "Cy Dunn", "Ann Lee"] and shown[1]["choices"] == ["Bob Ray", "Cy Dunn", "Ann Lee"]
    assert _fn(app, "on_season")("", "")[0]["value"] == []
    empty = _fn(app, "on_season")(str(source), str(results / "none"))
    assert empty[2]["visible"] is True and empty[2]["value"] == voices.NO_BANK


def test_revert_and_save_refresh_the_tab(app: Any, season: tuple[Path, Path, VoiceBank], tmp_path: Path) -> None:
    source, results, _ = season
    save = _fn(app, "do_save")
    assert "Сначала нажмите" in save(str(source), str(results), "Ann Lee", "t", None, None)[-2]
    out = save(str(source), str(results), "Ann Lee", "my words", None, _cut_state(tmp_path, "Ann Lee"))
    assert out[-2].startswith("✅ Образец сохранён")
    rows = out[0]["value"]
    assert next(r for r in rows if r[0] == "Ann Lee")[4] == "✋ указан вручную"
    revert = _fn(app, "run_revert")(str(source), str(results), "Ann Lee")
    assert revert[-1].startswith("✅ Ручной образец убран") and next(r for r in revert[0]["value"] if r[0] == "Ann Lee")[4] == "✅ найден сам"
    assert _fn(app, "run_revert")(str(source), str(results), None)[-1].startswith("❌ Сначала выберите")
