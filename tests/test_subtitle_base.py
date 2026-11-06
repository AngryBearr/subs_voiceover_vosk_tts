"""Subtitle base builder (stt.subtitle_base), base lookup in find_subtitles, subtitle key handling in Episode.needed."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any

import pytest

import stt.subtitles as subs
from stt import run, subtitle_base

LABELLED = ["NICK: We are going to win this and that is it", "SANDRA: You know that it is not the plan we had",
            "LIZ: I think we should talk"]
TURNS = [">> WE ARE GOING TO WIN THIS AND THAT IS IT", ">> YOU KNOW THAT IT IS NOT THE PLAN WE HAD"]
PLAIN = ["We are going to win this and that is it", "You know that it is not the plan we had"]


def srt_text(lines: list[str], n: int = 30) -> str:
    out = []
    for k in range(n):
        out.append(f"{k + 1}\n00:{k // 60:02d}:{k % 60:02d},000 --> 00:{k // 60:02d}:{k % 60:02d},800\n{lines[k % len(lines)]}\n")
    return "\n".join(out)


def make_archive(tmp_path: Path, files: dict[str, str], rows: list[dict[str, Any]]) -> Path:
    archive = tmp_path / "hunt"
    archive.mkdir()
    for name, text in files.items():
        (archive / name).write_text(text, encoding="utf-8")
    (archive / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return archive


def row(path: str, season: int = 1, episode: int = 1, **extra: Any) -> dict[str, Any]:
    return {"season": season, "episode": episode, "path": path, "language": "en", **extra}


# ---- build

def test_build_picks_labels_over_turns_over_plain_and_writes_files_and_index(tmp_path: Path) -> None:
    archive = make_archive(tmp_path, {"p.srt": srt_text(PLAIN), "t.srt": srt_text(TURNS), "l.srt": srt_text(LABELLED)},
                           [row("p.srt"), row("t.srt"), row("l.srt")])
    out = tmp_path / "base"
    summary = subtitle_base.build(archive, out, "survivor", "tt0239195")
    entry = json.loads((out / "survivor" / "index.json").read_text(encoding="utf-8"))["episodes"]["S01E01"]
    assert entry["kind"] == "labels" and entry["named_lines"] == 30 and entry["names"] == 3 and entry["cues"] == 30
    assert entry["file"] == "survivor/S01/S01E01.en.srt"
    assert entry["source"].endswith("l.srt")
    assert [a["kind"] for a in entry["alternates"]] == ["turns", "plain"]
    assert (out / entry["file"]).read_text(encoding="utf-8") == srt_text(LABELLED)
    index = json.loads((out / "survivor" / "index.json").read_text(encoding="utf-8"))
    assert (index["show"], index["imdb"], index["archive"]) == ("survivor", "tt0239195", str(archive))
    assert summary["after"] == {"labels": 1} and summary["episodes"] == 1


def test_build_ignores_the_manifest_counts_and_rereads_the_files(tmp_path: Path) -> None:
    archive = make_archive(tmp_path, {"liar.srt": srt_text(PLAIN), "real.srt": srt_text(TURNS)},
                           [row("liar.srt", n_label_lines=999, kind="labels"), row("real.srt", n_label_lines=0)])
    out = tmp_path / "base"
    subtitle_base.build(archive, out)
    entry = json.loads((out / "survivor" / "index.json").read_text(encoding="utf-8"))["episodes"]["S01E01"]
    assert entry["kind"] == "turns" and entry["source"].endswith("real.srt")


def test_build_ranks_by_named_lines_within_the_same_kind(tmp_path: Path) -> None:
    few = srt_text(LABELLED, n=21)
    many = srt_text(LABELLED, n=40)
    archive = make_archive(tmp_path, {"few.srt": few, "many.srt": many}, [row("few.srt"), row("many.srt")])
    subtitle_base.build(archive, tmp_path / "base")
    entry = json.loads((tmp_path / "base" / "survivor" / "index.json").read_text(encoding="utf-8"))["episodes"]["S01E01"]
    assert entry["source"].endswith("many.srt") and entry["named_lines"] == 40


def test_build_ignores_non_english_rows_and_unusable_files(tmp_path: Path) -> None:
    archive = make_archive(tmp_path, {"ru.srt": srt_text(LABELLED), "bad.srt": "not subtitles", "ok.srt": srt_text(PLAIN)},
                           [row("ru.srt", language="ru"), row("bad.srt"), row("ok.srt"), row("missing.srt")])
    subtitle_base.build(archive, tmp_path / "base")
    entry = json.loads((tmp_path / "base" / "survivor" / "index.json").read_text(encoding="utf-8"))["episodes"]["S01E01"]
    assert entry["source"].endswith("ok.srt") and entry["alternates"] == []


def test_build_accepts_absolute_and_relative_paths_and_several_episodes(tmp_path: Path) -> None:
    archive = make_archive(tmp_path, {"a.srt": srt_text(LABELLED)}, [])
    outside = tmp_path / "elsewhere.srt"
    outside.write_text(srt_text(TURNS), encoding="utf-8")
    (archive / "manifest.jsonl").write_text(
        json.dumps(row("a.srt", 2, 3)) + "\n" + json.dumps(row(str(outside), 4, 5)) + "\n", encoding="utf-8")
    subtitle_base.build(archive, tmp_path / "base")
    episodes = json.loads((tmp_path / "base" / "survivor" / "index.json").read_text(encoding="utf-8"))["episodes"]
    assert set(episodes) == {"S02E03", "S04E05"}
    assert episodes["S02E03"]["kind"] == "labels" and episodes["S04E05"]["kind"] == "turns"
    assert (tmp_path / "base" / "survivor" / "S04" / "S04E05.en.srt").is_file()


def test_build_counts_before_and_after_per_kind(tmp_path: Path) -> None:
    archive = make_archive(tmp_path, {"p.srt": srt_text(PLAIN), "l.srt": srt_text(LABELLED)},
                           [row("p.srt", kind="labels"), row("l.srt", 1, 2, kind="plain")])
    summary = subtitle_base.build(archive, tmp_path / "base", dry_run=True)
    assert summary["after"] == {"plain": 1, "labels": 1}
    assert sum(summary["before"].values()) == 2


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    archive = make_archive(tmp_path, {"l.srt": srt_text(LABELLED)}, [row("l.srt")])
    out = tmp_path / "base"
    summary = subtitle_base.build(archive, out, dry_run=True)
    assert not out.exists()
    assert summary["after"] == {"labels": 1}


def test_build_counts_names_in_any_case_that_the_file_repeats(tmp_path: Path) -> None:
    mixed = srt_text(["Nick: We are going to win this and that is it", "Sandra: You know that it is not the plan"])
    archive = make_archive(tmp_path, {"m.srt": mixed}, [row("m.srt")])
    summary = subtitle_base.build(archive, tmp_path / "base", dry_run=True)
    entry = summary["index"]["episodes"]["S01E01"]
    assert entry["kind"] == "labels" and entry["names"] == 2


def test_cli_errors_without_a_manifest_and_dry_run_prints(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        subtitle_base.main(["build", "--archive", str(empty), "--out", str(tmp_path / "o"), "--dry-run"])
    archive = make_archive(tmp_path, {"l.srt": srt_text(LABELLED)}, [row("l.srt")])
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"subtitles": {"imdb": {"survivor": "tt0239195"}}}), encoding="utf-8")
    code = subtitle_base.main(["build", "--archive", str(archive), "--out", str(tmp_path / "o"), "--config", str(cfg), "--dry-run"])
    assert code == 0 and not (tmp_path / "o").exists()
    assert "dry run" in capsys.readouterr().out


# ---- lookup

def make_base(tmp_path: Path, show: str = "survivor", imdb: str = "tt0239195", lines: list[str] = LABELLED) -> Path:
    archive = make_archive(tmp_path, {"l.srt": srt_text(lines)}, [row("l.srt", 5, 2)])
    base = tmp_path / "base"
    subtitle_base.build(archive, base, show, imdb)
    return base


def cfg_of(**extra: Any) -> dict[str, Any]:
    return {"online": False, "imdb": "tt0239195", "archive": None, **extra}


def media_file(tmp_path: Path, name: str = "Survivor.S05E02.mkv") -> Path:
    media = tmp_path / "video" / name
    media.parent.mkdir(exist_ok=True)
    media.write_bytes(b"")
    return media


@pytest.fixture(autouse=True)
def no_embedded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subs, "embedded_candidates", lambda m: [])


def test_base_gives_subtitles_and_records_its_origin(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    out = tmp_path / "out"
    result = subs.find_subtitles(media_file(tmp_path), out, cfg_of(base=str(base)))
    assert result is not None and result["origin"] == "base" and result["kind"] == "labels"
    assert json.loads((out / subs.SOURCE_JSON).read_text(encoding="utf-8"))["origin"] == "base"
    assert (out / subs.SOURCE_SRT).read_text(encoding="utf-8") == srt_text(LABELLED)


def test_season_folder_comes_before_the_base_and_ends_the_search_with_labels(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    folder = tmp_path / "season"
    folder.mkdir()
    (folder / "Survivor.S05E02.srt").write_text(srt_text(LABELLED[:2]), encoding="utf-8")
    result = subs.find_subtitles(media_file(tmp_path), tmp_path / "out", cfg_of(base=str(base), folder=str(folder)))
    assert result is not None and result["origin"] == "season folder"


def test_base_is_skipped_once_a_better_source_has_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media = media_file(tmp_path)
    (media.parent / "Survivor.S05E02.srt").write_text(srt_text(LABELLED), encoding="utf-8")
    monkeypatch.setattr(subs, "base_candidates", lambda *a: pytest.fail("labels already found"))
    result = subs.find_subtitles(media, tmp_path / "out", cfg_of(base=str(tmp_path / "base")))
    assert result is not None and result["origin"] != "base"


def test_a_better_kind_from_the_base_beats_plain_next_to_the_video(tmp_path: Path) -> None:
    base = make_base(tmp_path, lines=TURNS)
    media = media_file(tmp_path)
    (media.parent / "Survivor.S05E02.srt").write_text(srt_text(PLAIN), encoding="utf-8")
    result = subs.find_subtitles(media, tmp_path / "out", cfg_of(base=str(base)))
    assert result is not None and result["origin"] == "base" and result["kind"] == "turns"


def test_base_is_ignored_for_another_show(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    media = media_file(tmp_path, "The Amazing Race S05E02.mkv")
    cfg = cfg_of(base=str(base), imdb={"amazing race": "tt0285331"})
    assert subs.find_subtitles(media, tmp_path / "out", cfg) is None


def test_archive_is_for_survivor_us_only(tmp_path: Path) -> None:
    archive = make_archive(tmp_path, {"l.srt": srt_text(LABELLED)}, [row("l.srt", 5, 2)])
    race = media_file(tmp_path, "The Amazing Race S05E02.mkv")
    imdb = {"amazing race": "tt0285331"}
    assert not subs.show_allows(race, imdb, "tt0239195")
    assert subs.find_subtitles(race, tmp_path / "o1", cfg_of(archive=str(archive), imdb=imdb)) is None
    survivor = media_file(tmp_path, "Survivor.S05E02.mkv")
    result = subs.find_subtitles(survivor, tmp_path / "o2", cfg_of(archive=str(archive), imdb={"survivor": "tt0239195"}))
    assert result is not None and result["origin"] == "archive"


def test_missing_base_or_episode_is_not_an_error(tmp_path: Path) -> None:
    base = make_base(tmp_path)
    other = media_file(tmp_path, "Survivor.S09E09.mkv")
    assert subs.find_subtitles(other, tmp_path / "o1", cfg_of(base=str(base))) is None
    assert subs.find_subtitles(media_file(tmp_path), tmp_path / "o2", cfg_of(base=str(tmp_path / "nope"))) is None


def test_setting_base_makes_a_new_search_but_a_config_without_it_keeps_the_old_cache(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media = media_file(tmp_path)
    out = tmp_path / "out"
    assert subs.find_subtitles(media, out, cfg_of()) is None  # cached "nothing"
    base = make_base(tmp_path)
    result = subs.find_subtitles(media, out, cfg_of(base=str(base)))
    assert result is not None and result["origin"] == "base"
    # the same settings again: cached
    monkeypatch.setattr(subs, "base_candidates", lambda *a: pytest.fail("cached"))
    assert subs.find_subtitles(media, out, cfg_of(base=str(base))) == result


def test_cache_made_without_base_or_folder_stays_valid_when_the_keys_are_absent(tmp_path: Path) -> None:
    media = media_file(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / subs.SOURCE_JSON).write_text(json.dumps(
        {"origin": None, "kind": "none", "searched": {"online": False, "imdb": "tt0239195", "archive": ""}}), encoding="utf-8")
    base = make_base(tmp_path)
    # an older cache (no folder/base keys) answers for a config that does not set them, even though a base exists on disk
    assert subs.find_subtitles(media, out, cfg_of()) is None
    assert base.exists()


def test_setting_folder_makes_a_new_search(tmp_path: Path) -> None:
    media = media_file(tmp_path)
    out = tmp_path / "out"
    assert subs.find_subtitles(media, out, cfg_of()) is None
    folder = tmp_path / "season"
    folder.mkdir()
    (folder / "S05E02.srt").write_text(srt_text(LABELLED), encoding="utf-8")
    result = subs.find_subtitles(media, out, cfg_of(folder=str(folder)))
    assert result is not None and result["origin"] == "season folder"


# ---- Episode.needed with a changed subtitle key

def make_episode(tmp_path: Path, force: str | None = None) -> run.Episode:
    args = argparse.Namespace(season_glossary=None, force=force, until="mix", engine="vosk", start=None, duration=None,
                              planned_engines=["vosk"])
    episode = run.Episode(tmp_path / "S" / "ep01.mkv", tmp_path / "out", args, {}, lock=threading.Lock())
    episode.out.mkdir(parents=True, exist_ok=True)
    return episode


def prepare(episode: run.Episode, monkeypatch: pytest.MonkeyPatch, translated: bool) -> Path:
    output = episode.out / "subs.en.json"
    output.write_text("[]", encoding="utf-8")
    episode.subtitle_stamp.parent.mkdir(parents=True, exist_ok=True)
    episode.subtitle_stamp.write_text("turns:abc:0.3\n", encoding="utf-8")
    if translated:
        (episode.out / "subs.ru.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(episode, "subtitle_key", lambda: "labels:def:0.3")
    return output


def test_changed_subtitles_assemble_an_untranslated_episode_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    episode = make_episode(tmp_path)
    output = prepare(episode, monkeypatch, translated=False)
    assert episode.needed("transcribe", [output]) is True


def test_changed_subtitles_keep_the_transcript_of_a_translated_episode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    episode = make_episode(tmp_path)
    output = prepare(episode, monkeypatch, translated=True)
    assert episode.needed("transcribe", [output]) is False


def test_force_transcribe_takes_new_subtitles_of_a_translated_episode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    episode = make_episode(tmp_path, force="transcribe")
    output = prepare(episode, monkeypatch, translated=True)
    assert episode.needed("transcribe", [output]) is True


def test_unchanged_subtitle_key_does_not_trigger_assembly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    episode = make_episode(tmp_path)
    output = prepare(episode, monkeypatch, translated=False)
    monkeypatch.setattr(episode, "subtitle_key", lambda: "turns:abc:0.3")
    assert episode.needed("transcribe", [output]) is False
