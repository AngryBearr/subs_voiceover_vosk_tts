"""«Кто это?»: unnamed groups, snippets, the override file and the next-step text."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from stt import who
from ui import who as who_ui

RATE = 16_000


def _rec(i: int, start: int, end: int, speaker: str, **extra: object) -> dict[str, object]:
    return {"index": i, "start": start, "end": end, "duration": end - start, "text": [f"line {i}"],
            "speaker": speaker, "overlap": False, "ocr": False, **extra}


def _episode(results: Path, name: str = "S01E01.x") -> Path:
    folder = results / name
    folder.mkdir(parents=True)
    speakers = [{"id": "S01", "name": "Jeff Probst", "cues": 30}, {"id": "S02", "name": None, "cues": 8, "gender": "male"},
                {"id": "S03", "name": None, "cues": 2}, {"id": "S04", "name": None, "cues": 6}]
    records = []
    t = 0
    for i in range(8):  # S02: one cue is overlapped, one is too short, one too long
        length = [3000, 3500, 1000, 9000, 4000, 2500, 5000, 3000][i]
        records.append(_rec(i + 1, t, t + length, "S02", overlap=(i == 1)))
        t += length + 20_000
    records.append(_rec(20, 0, 1500, "S01"))
    records += [_rec(30 + i, 500_000 + i * 4000, 500_000 + i * 4000 + 3000, "S04") for i in range(6)]
    (folder / "speakers.json").write_text(json.dumps(speakers), encoding="utf-8")
    (folder / "subs.en.json").write_text(json.dumps(records), encoding="utf-8")
    (folder / "speakers.llm.json").write_text(json.dumps({"cards": [
        {"id": "S02", "role": "contestant", "gender_evidence": "male voice"}]}), encoding="utf-8")
    seconds = 700
    sf.write(str(folder / "audio.wav"), (np.random.default_rng(0).random(seconds * RATE) - 0.5).astype("float32"), RATE)
    return folder


def test_unnamed_groups_skip_named_and_tiny(tmp_path: Path) -> None:
    folder = _episode(tmp_path)
    groups = who.unnamed_groups(folder)
    assert sorted(g.speaker for g in groups) == ["S02", "S04"]  # S01 is named, S03 has 2 cues
    s02 = next(g for g in groups if g.speaker == "S02")
    assert s02.cues == 8 and s02.role == "contestant" and s02.gender_evidence == "male voice"
    assert s02.seconds == 3.0 + 3.5 + 1.0 + 9.0 + 4.0 + 2.5 + 5.0 + 3.0


def test_snippets_are_clean_and_short(tmp_path: Path) -> None:
    folder = _episode(tmp_path)
    s02 = next(g for g in who.unnamed_groups(folder) if g.speaker == "S02")
    spans = [round(s.end - s.start, 1) for s in s02.snippets]
    assert len(s02.snippets) == 5 and all(2.0 <= x <= 6.0 for x in spans)
    assert 20.0 not in [s.start for s in s02.snippets] and all(s.text.startswith("line") for s in s02.snippets)


def test_cut_snippets_writes_wavs(tmp_path: Path) -> None:
    folder = _episode(tmp_path)
    group = next(g for g in who.unnamed_groups(folder) if g.speaker == "S02")
    who.cut_snippets(group, folder, tmp_path / "cache")
    info = [sf.info(str(s.wav)) for s in group.snippets]
    assert all(i.samplerate == RATE and 1.9 < i.duration < 6.1 for i in info)


def test_save_answer_keeps_other_entries_and_hides_the_group(tmp_path: Path) -> None:
    folder = _episode(tmp_path)
    who.save_answer(folder, "S02", name="Sonja Christopher", gender="female")
    who.save_answer(folder, "S04", mixed=True)
    data = json.loads((folder / "speakers.override.json").read_text(encoding="utf-8"))
    assert data == {"S02": {"name": "Sonja Christopher", "gender": "female"}, "S04": {"mixed": True}}
    assert who.unnamed_groups(folder) == []
    try:
        who.save_answer(folder, "S02")
    except ValueError:
        pass
    else:
        raise AssertionError("empty answer must be refused")


def test_cast_names_host_first(tmp_path: Path) -> None:
    path = tmp_path / "Season 1.cast.json"
    path.write_text(json.dumps({"cast": [
        {"name": "Ann", "gender": "female", "role": "Contestant"},
        {"name": "Jeff Probst", "gender": "male", "role": "Host"}]}), encoding="utf-8")
    assert who.cast_names(path) == ["Jeff Probst", "Ann"]
    assert who.cast_gender(path, "Ann") == "female"
    assert who.cast_names(tmp_path / "none.json") == []


def test_stage_and_command(tmp_path: Path) -> None:
    folder = _episode(tmp_path)
    assert who.episode_stage(tmp_path, folder) == "early"
    (folder / "subs.ru.json").write_text("[]")
    assert who.episode_stage(tmp_path, folder) == "translated"
    (folder / "S01E01.x.ru.vosk.m4a").write_text("")
    assert who.episode_stage(tmp_path, folder) == "voiced"
    assert "--force speakers" in who.redo_command(Path("/v/S01E01.x.mkv"), tmp_path)


def _season(tmp_path: Path) -> tuple[Path, Path, Path]:
    source, results = tmp_path / "Season 1", tmp_path / "STT_Season 1"
    source.mkdir()
    (source / "S01E01.x.mkv").write_text("")
    folder = _episode(results)
    (results / "Season 1.cast.json").write_text(json.dumps({"cast": [
        {"name": "Ann", "gender": "female", "role": "Contestant"}]}), encoding="utf-8")
    return source, results, folder


def test_ui_save_messages(tmp_path: Path) -> None:
    source, results, folder = _season(tmp_path)
    key = "S01E01.x|S02"
    assert "выберите сезон" in who_ui.save("", "", key, "Ann", None)
    assert "Нечего сохранять" in who_ui.save(str(source), str(results), None, "Ann", None)
    assert "Ничего не сохранено" in who_ui.save(str(source), str(results), key, who_ui.SKIP_CHOICE, None)
    assert not (folder / "speakers.override.json").exists()
    text = who_ui.save(str(source), str(results), key, "Ann", None)
    assert text.startswith("✅") and "ещё не переведена" in text
    assert json.loads((folder / "speakers.override.json").read_text(encoding="utf-8"))["S02"] == {
        "name": "Ann", "gender": "female"}  # gender from the cast
    (folder / "subs.ru.json").write_text("[]")
    (folder / "S01E01.x.ru.f5.m4a").write_text("")
    text = who_ui.save(str(source), str(results), "S01E01.x|S04", who_ui.MIXED_CHOICE, None)
    assert "смешанная" in text and "--force speakers" in text and "S01E01.x.mkv" in text
    assert "Не найдена" in who_ui.save(str(source), str(results), "nope|S02", "Ann", None)


def test_ui_choices_and_page(tmp_path: Path) -> None:
    source, results, _ = _season(tmp_path)
    assert who_ui.name_choices(results, "Season 1") == ["Ann", who_ui.MIXED_CHOICE, who_ui.SKIP_CHOICE]
    assert sorted(who_ui.group_key(g) for g in who_ui.groups_for(results, None)) == ["S01E01.x|S02", "S01E01.x|S04"]
    assert who_ui.groups_for(results, ["other"]) == []
    from ui import dub_app

    dub_app.build()  # the tab is part of the page
