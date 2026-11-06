"""Another cut of the episode: the base's alternates are tried when the first file barely fits the audio
(Episode._try_other_cuts), the subtitle settings of the page and the "Субтитры" column of the season table."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any, Callable

import pytest

import stt.subtitles as subs
from stt import run, subtitle_base
from tests.test_subtitle_base import LABELLED, TURNS, make_archive, row, srt_text
from ui import seasons, settings

FIRST, SECOND, THIRD = srt_text(LABELLED, n=40), srt_text(LABELLED, n=30), srt_text(TURNS, n=30)


@pytest.fixture(autouse=True)
def no_embedded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subs, "embedded_candidates", lambda m: [])


class Setup:
    def __init__(self, tmp_path: Path, shares: dict[str, float | None], min_pinned: float = 0.3) -> None:
        files = {"a.srt": FIRST, "b.srt": SECOND, "c.srt": THIRD}
        archive = make_archive(tmp_path, files, [row(n, 5, 2) for n in files])
        self.base = tmp_path / "base"
        subtitle_base.build(archive, self.base, "survivor", "tt0239195")
        media = tmp_path / "video" / "Survivor.S05E02.mkv"
        media.parent.mkdir()
        media.write_bytes(b"")
        args = argparse.Namespace(season_glossary=None, force=None, until="mix", engine="vosk", start=None, duration=None,
                                  planned_engines=["vosk"])
        cfg = {"enabled": True, "online": False, "imdb": "tt0239195", "archive": None, "base": str(self.base),
               "min_pinned": min_pinned}
        self.episode = run.Episode(media, tmp_path / "out", args, {"subtitles": cfg}, lock=threading.Lock())
        self.episode.out.mkdir(parents=True)
        self.shares = shares
        self.runs: list[str] = []

    def assemble(self) -> None:
        """The fake transcribe module: the pinned share depends on the file now in subs.source.en.srt."""
        text = (self.episode.out / subs.SOURCE_SRT).read_text(encoding="utf-8")
        self.runs.append(text)
        share = self.shares[text]
        report = {"kind": "labels", "pinned_lines": 0, "not_confirmed": 0}
        if share is not None:
            report.update(pinned_lines=round(share * 100), not_confirmed=100 - round(share * 100))
        (self.episode.out / "subtitle_hints.json").write_text(json.dumps(report), encoding="utf-8")

    def go(self) -> dict[str, Any]:
        self.episode.subtitle_source()  # the lookup of the transcribe stage
        self.assemble()  # the first assembly
        self.runs.clear()
        self.episode._try_other_cuts(self.assemble)
        return json.loads((self.episode.out / subs.SOURCE_JSON).read_text(encoding="utf-8"))


def test_a_poor_first_file_tries_the_alternates_and_keeps_the_best(tmp_path: Path) -> None:
    setup = Setup(tmp_path, {FIRST: 0.1, SECOND: 0.2, THIRD: 0.25})
    data = setup.go()
    assert setup.runs == [SECOND, THIRD]  # the best is the last one tried: no extra assembly
    assert (setup.episode.out / subs.SOURCE_SRT).read_text(encoding="utf-8") == THIRD
    assert data["origin"] == "base" and data["kind"] == "turns" and data["path_or_url"].endswith("c.srt")
    assert [r["pinned_share"] for r in data["rejected"]] == [0.1, 0.2]
    assert data["rejected"][0]["path"].endswith("S05E02.en.srt")  # the base's own file
    assert data["searched"]["base"] == str(setup.base)
    assert setup.episode.subtitle_key() == f"turns:{data['sha1']}:0.3"
    # the next run finds the cache and the same key
    again = subs.find_subtitles(setup.episode.media, setup.episode.out, setup.episode.config["subtitles"])
    assert again is not None and again["sha1"] == data["sha1"]


def test_the_first_alternate_that_fits_ends_the_search(tmp_path: Path) -> None:
    setup = Setup(tmp_path, {FIRST: 0.1, SECOND: 0.8, THIRD: 0.9})
    data = setup.go()
    assert setup.runs == [SECOND]
    assert data["path_or_url"].endswith("b.srt") and len(data["rejected"]) == 1


def test_when_none_fits_the_best_is_kept_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    setup = Setup(tmp_path, {FIRST: 0.2, SECOND: 0.1, THIRD: None})
    with caplog.at_level("WARNING"):
        data = setup.go()
    assert setup.runs == [SECOND, THIRD, FIRST]  # the first file is assembled again
    assert (setup.episode.out / subs.SOURCE_SRT).read_text(encoding="utf-8") == FIRST
    assert data["path_or_url"].endswith("S05E02.en.srt")
    assert [r["pinned_share"] for r in data["rejected"]] == [0.1, 0.0]
    assert "no file of the base fits" in caplog.text


def test_a_good_first_file_tries_nothing(tmp_path: Path) -> None:
    setup = Setup(tmp_path, {FIRST: 0.5, SECOND: 0.9, THIRD: 0.9})
    data = setup.go()
    assert setup.runs == [] and "rejected" not in data


def test_a_file_not_from_the_base_tries_nothing(tmp_path: Path) -> None:
    setup = Setup(tmp_path, {FIRST: 0.0, SECOND: 0.9, THIRD: 0.9})
    media = setup.episode.media
    (media.parent / "Survivor.S05E02.srt").write_text(FIRST, encoding="utf-8")
    setup.episode.config["subtitles"]["base"] = None
    data = setup.go()
    assert data["origin"] == "sidecar" and setup.runs == []


def test_the_threshold_is_a_setting(tmp_path: Path) -> None:
    setup = Setup(tmp_path, {FIRST: 0.4, SECOND: 0.9, THIRD: 0.9}, min_pinned=0.5)
    assert setup.go()["path_or_url"].endswith("b.srt")
    assert json.loads((Path(run.REPO_ROOT) / "stt_config.json").read_text(encoding="utf-8"))["subtitles"]["min_pinned"] == 0.3


# ---- the page

def test_subtitle_settings_of_the_page() -> None:
    by_path = {s.path: s for s in settings.SETTINGS}
    kinds = {("subtitles", "enabled"): "bool", ("subtitles", "online"): "bool", ("subtitles", "base"): "folder",
             ("subtitles", "folder"): "folder"}
    for path, kind in kinds.items():
        assert by_path[path].kind == kind and by_path[path].group == "Субтитры"
    assert ("mix", "subtitles") in by_path and by_path[("mix", "subtitles")].group != "Субтитры"
    standard = settings.read_json(settings.DEFAULTS)
    assert settings.get(standard, ("subtitles", "enabled")) is True and settings.get(standard, ("subtitles", "base")) is None
    assert settings.get({}, ("subtitles", "online")) is True  # the fallback when a file lacks the key


@pytest.mark.parametrize(("data", "word"), [
    ({"origin": "base", "kind": "labels"}, "имена, база"),
    ({"origin": "season folder", "kind": "turns"}, "смены говорящих, своя папка"),
    ({"origin": "embedded", "kind": "plain"}, "только текст, в видео"),
    ({"origin": "sidecar", "kind": "labels"}, "имена, рядом с видео"),
    ({"origin": "archive", "kind": "turns"}, "смены говорящих, архив"),
    ({"origin": "online:stremio", "kind": "labels"}, "имена, интернет"),
    ({"origin": "online:gestdown", "kind": "labels"}, "имена, интернет"),
    ({"origin": None, "kind": "none"}, "нет"),
])
def test_table_column_words(tmp_path: Path, data: dict[str, Any], word: str) -> None:
    (tmp_path / "subs.source.json").write_text(json.dumps(data), encoding="utf-8")
    assert seasons.subtitles_note(tmp_path) == word


def test_table_has_the_column_and_a_dash_without_a_work_folder(tmp_path: Path) -> None:
    source, results = tmp_path / "S", tmp_path / "R"
    source.mkdir()
    for name in ("S01E01.mkv", "S01E02.mkv"):
        (source / name).write_bytes(b"")
    (results / "S01E02").mkdir(parents=True)
    (results / "S01E02" / "subs.source.json").write_text(json.dumps({"origin": "base", "kind": "turns"}), encoding="utf-8")
    table = seasons.table(seasons.Season(source, results))
    assert [row[2] for row in table] == ["—", "смены говорящих, база"]
    assert seasons.subtitles_note(results / "S01E01") == "—"
