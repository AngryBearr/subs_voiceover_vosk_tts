from __future__ import annotations

import json
from pathlib import Path

from stt import subtitle_edit as se


def _episode(tmp_path: Path) -> Path:
    folder = tmp_path / "ep"
    folder.mkdir()
    lines = [
        {"index": 1, "start": 1000, "end": 3000, "text": ["Привет всем."], "gender": "male", "speaker": "S01",
         "speaker_name": "Джефф Пробст", "text_en": ["Hi all."], "window_ms": 2500},
        {"index": 2, "start": 4000, "end": 5000, "text": ["Я здесь."], "gender": "male", "speaker": "S01",
         "speaker_name": "Джефф Пробст", "text_en": ["I'm here."], "window_ms": 1200},
        {"index": 3, "start": 6000, "end": 7000, "text": ["Ага."], "gender": "unknown", "speaker": "S03",
         "speaker_name": "S03", "text_en": ["Yeah."], "window_ms": 1000},
    ]
    speakers = [{"id": "S01", "name": "Jeff Probst", "label_ru": "Джефф Пробст", "gender": "male", "cues": 2},
                {"id": "S02", "name": "Jelly Loblack", "label_ru": "Джелли Лоблак", "gender": "female", "cues": 1},
                {"id": "S03", "gender": "unknown", "cues": 1}]
    (folder / "subs.ru.json").write_text(json.dumps(lines, ensure_ascii=False))
    (folder / "speakers.json").write_text(json.dumps(speakers, ensure_ascii=False))
    (folder / "review.json").write_text(json.dumps([{"line": 2, "heard": "female (0.96)",
                                                     "sounds_like": "Джелли Лоблак (0.7)"}], ensure_ascii=False))
    return folder


def test_ass_round_trip_turns_edits_into_corrections(tmp_path: Path) -> None:
    folder = _episode(tmp_path)
    path = se.export_ass(folder, folder / "edit" / "ep.ass")
    text = path.read_text(encoding="utf-8-sig")
    assert "Dialogue: 0,0:00:01.00,0:00:03.00,Мужской,Джефф Пробст,0,0,0,#1,Привет всем." in text
    assert se.changes(folder, se.read_edited(path)) == ([], [])  # nothing edited: nothing to do
    edited = (text.replace("Мужской,Джефф Пробст,0,0,0,#2,Я здесь.", "Мужской,Джелли Лоблак,0,0,0,#2,Я тут, ребята!")
              .replace("Мужской,S03,0,0,0,#3,Ага.", "Не озвучивать,S03,0,0,0,#3,Ага.")
              .replace("#1,Привет всем.", "#1,Привет всем!"))
    path.write_text(edited, encoding="utf-8-sig")
    found, warnings = se.changes(folder, se.read_edited(path))
    assert warnings == []
    fixes = {c.index: c.fix for c in found}
    # the new speaker's own gender (female) wins over the style left at "Мужской"
    assert fixes == {1: {"text": "Привет всем!"}, 2: {"text": "Я тут, ребята!", "speaker": "S02"}, 3: {"skip": True}}
    se.apply(folder, found)
    lines = {ln.index: ln for ln in se.current_lines(folder)}
    assert (lines[2].speaker, lines[2].gender, lines[3].skip) == ("S02", "female", True)
    # back to the original: the correction goes
    found, _ = se.changes(folder, se.read_edited(se.export_ass(folder, folder / "edit" / "again.ass")))
    assert found == []
    se.apply(folder, se.changes(folder, [se.Edited(1000, 3000, "Привет всем.", None, None, False, 1)])[0])
    assert "1" not in json.loads((folder / "corrections.json").read_text())


def test_review_export_has_notes_that_do_not_come_back(tmp_path: Path) -> None:
    folder = _episode(tmp_path)
    path = se.export_ass(folder, folder / "r.ass", review=True)
    text = path.read_text(encoding="utf-8-sig")
    assert text.count("Dialogue:") == 1 and "{слышно female (0.96); похоже на Джелли Лоблак (0.7)}Я здесь." in text
    assert se.read_edited(path)[0].text == "Я здесь."
    path.write_text(text.replace(",Мужской,", ",Женский,"), encoding="utf-8-sig")
    found, _ = se.changes(folder, se.read_edited(path))
    assert [(c.index, c.fix) for c in found] == [(2, {"gender": "female"})]


def test_srt_by_time_speaker_prefix_and_warnings(tmp_path: Path) -> None:
    folder = _episode(tmp_path)
    path = se.export_srt(folder, folder / "ep.srt")
    assert "1\n00:00:01,000 --> 00:00:03,000\n[Джефф Пробст] Привет всем.\n" in path.read_text()
    edited = ("7\n00:00:04,050 --> 00:00:05,000\n[Jelly] <i>Я здесь.</i>\n\n"
              "8\n00:00:06,000 --> 00:00:07,000\n[Кто-то] Ага.\n\n"
              "9\n00:01:00,000 --> 00:01:01,000\nНовая реплика\n")
    path.write_text(edited)
    found, warnings = se.changes(folder, se.read_edited(path))
    assert [(c.index, c.fix) for c in found] == [(2, {"speaker": "S02"})]  # renumbered, found by time
    assert any("Кто-то" in w for w in warnings) and any("нет такой реплики" in w for w in warnings)


def test_fit_verdicts() -> None:
    policy = {"base_syl_per_sec": 5.0, "comfort_syl_per_sec": 6.3, "ceiling_syl_per_sec": 6.7}
    assert se.fit("Привет всем.", 2000, policy).verdict == "ok"  # 4 syllables in 2 s
    over = se.fit("Мы очень долго продумываем все правила этой игры.", 1500, policy)
    assert over.verdict == "over" and over.excess == 8 and "уберите примерно 8" in over.text  # 18 syl., 10 fit
    assert se.fit("Зам+ок", 1000, policy).syllables == 2  # a stress mark is not a letter


def test_set_line_and_reset(tmp_path: Path) -> None:
    folder = _episode(tmp_path)
    se.set_line(folder, 2, text="Я тут.", gender="female")
    assert json.loads((folder / "corrections.json").read_text()) == {"2": {"text": "Я тут.", "gender": "female"}}
    se.reset_line(folder, 2)
    assert json.loads((folder / "corrections.json").read_text()) == {}
    import pytest

    with pytest.raises(ValueError, match="нет в серии"):
        se.set_line(folder, 2, speaker="Никто")
