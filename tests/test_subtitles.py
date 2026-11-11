"""Subtitle pieces: who speaks, pinned to the recognized words."""

from __future__ import annotations

import io
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest

import stt.subtitles as subs
from stt.subtitles import (
    HOST,
    Piece,
    join_continued,
    labelled,
    names_person,
    parse_cues,
    pieces_of,
    pin_pieces,
    read_pieces,
    subtitle_kind,
    turn_starts,
)
from stt.types import Word

SAMPLE_STREAMS = """Input #0, matroska,webm, from 'Survivor.S51E01.mkv':
  Stream #0:0: Video: h264 (High), yuv420p, 1920x1080, 23.98 fps
  Stream #0:1(eng): Audio: eac3, 48000 Hz, 5.1(side), fltp
    Metadata:
      title           : English
  Stream #0:2(eng): Subtitle: subrip (default)
    Metadata:
      title           : English
  Stream #0:3(eng): Subtitle: subrip (hearing impaired)
    Metadata:
      title           : English SDH
  Stream #0:4(fre): Subtitle: subrip
  Stream #0:5(eng): Subtitle: hdmv_pgs_subtitle (pgssub), 1920x1080
  Stream #0:6: Subtitle: ass
    Metadata:
      title           : Signs and songs
  Stream #0:7(und): Subtitle: webvtt
"""


def _words(text: str, start: float = 10.0, step: float = 0.5) -> list[Word]:
    return [Word(start + i * step, start + i * step + 0.4, w) for i, w in enumerate(text.split())]


def _srt(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "ep.srt"
    path.write_text(body, encoding="utf-8")
    return path


def test_label_alone_on_a_line_and_two_voices_in_one_cue() -> None:
    pieces = pieces_of((1.0, 3.0, ["SANDRA:", "I am queen", "- NICK: No you are not", "- Yes I am"]))
    assert [(p.person, p.text.strip()) for p in pieces] == [("SANDRA", "I am queen"), ("NICK", "No you are not"), ("", "Yes I am")]
    assert all(p.opens_turn for p in pieces)


def test_turn_marks_and_all_caps_captions() -> None:
    pieces = pieces_of((1.0, 3.0, [">> WE GOT IT", ">> I THINK: YES WE DID", "AND THEN SOME"]))
    assert [p.person for p in pieces] == ["", ""]
    assert [p.opens_turn for p in pieces] == [True, True]
    assert pieces[1].text.endswith("AND THEN SOME")
    assert pieces_of((1.0, 3.0, [">> SANDRA: Nice one"]))[0].person == "SANDRA"


def test_host_aliases_and_generic_labels() -> None:
    assert pieces_of((0.0, 1.0, ["JEFF PROBST: Come on in"]))[0].person == HOST
    assert not names_person("WOMAN")
    assert not names_person("MAN 2")
    assert names_person("SANDRA")


def test_sound_notes_are_dropped() -> None:
    piece = pieces_of((0.0, 1.0, ["[LAUGHS] NICK: (quietly) No way"]))[0]
    assert piece.person == "NICK"
    assert piece.tokens == ["no", "way"]


def test_a_sentence_running_into_the_next_cue_keeps_its_person() -> None:
    pieces = [Piece("LIZ", "After Tribal Council,", 1.0, 2.0), Piece("", "we all knew.", 2.1, 3.0, tokens=["we", "all", "knew"]),
              Piece("", "Something else", 3.1, 4.0, tokens=["something", "else"])]
    pieces[0].tokens = ["after", "tribal", "council"]
    joined = join_continued(pieces)
    assert len(joined) == 2
    assert joined[0].person == "LIZ" and joined[0].cue_end == 3.0
    assert joined[0].tokens[-1] == "knew"


def test_a_finished_sentence_or_a_dash_does_not_continue() -> None:
    first = Piece("LIZ", "We lost.", 1.0, 2.0, tokens=["we", "lost"])
    assert len(join_continued([first, Piece("", "we all knew", 2.1, 3.0, tokens=["we", "all", "knew"])])) == 2
    open_line = Piece("LIZ", "We lost", 1.0, 2.0, tokens=["we", "lost"])
    assert len(join_continued([open_line, Piece("", "No", 2.1, 3.0, tokens=["no"], opens_turn=True)])) == 2


def test_parse_tolerates_junk_and_kind(tmp_path: Path) -> None:
    body = "1\n00:00:01,000 --> 00:00:02,000\nNICK: Hello\n\nnot a cue\n\n2\n00:00:03.5 --> 00:00:04.5\n- Bye\n"
    path = _srt(tmp_path, body)
    cues = parse_cues(path)
    assert cues[0][:2] == (1.0, 2.0)
    assert cues[1][:2] == (3.5, 4.5)
    assert subtitle_kind(read_pieces(path)) == "plain"
    many = "".join(f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},900\nNICK: Hello there\n\n" for i in range(1, 30))
    assert subtitle_kind(read_pieces(_srt(tmp_path, many))) == "labels"
    turns = "".join(f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},900\n>> HELLO THERE\n\n" for i in range(1, 30))
    assert subtitle_kind(read_pieces(_srt(tmp_path, turns))) == "turns"


def test_pinning_gets_the_words_of_each_voice() -> None:
    words = _words("we should vote out nick no way that is crazy")
    pieces = pieces_of((10.0, 12.5, ["SANDRA: We should vote out Nick."])) + pieces_of((12.5, 15.0, ["- NICK: No way, that is crazy."]))
    pinned, offset = pin_pieces(pieces, words)
    assert offset == 0.0
    assert [(p.piece.person, p.first, p.last) for p in pinned] == [("SANDRA", 0, 4), ("NICK", 5, 9)]
    assert [p.piece.person for p in labelled(pinned)] == ["SANDRA", "NICK"]
    assert turn_starts(pinned) == [0, 5]


def test_pinned_piece_keeps_the_matched_blocks_and_the_offset() -> None:
    from stt.subtitles import WordIndex, pin_piece

    words = _words("noise get on the mat", 10.9)
    index = WordIndex(words)
    pin = pin_piece(pieces_of((10.0, 12.0, ["Come on get on the mat"]))[0], index, 0.5)
    assert pin is not None and pin.offset == 0.5 and pin.blocks == [(2, 1, 4)]
    short = pin_piece(Piece("", "color", 10.9, 11.5, tokens=["color"]), WordIndex(_words("colour", 10.9)))
    assert short is not None and short.blocks == [(0, 0, 1)]


def test_text_the_audio_does_not_confirm_is_dropped() -> None:
    words = _words("we should vote out nick")
    pinned, _ = pin_pieces(pieces_of((10.0, 12.0, ["SANDRA: Completely different sentence here today."])), words)
    assert pinned == []


def test_a_shifted_file_is_found() -> None:
    words = _words("we should all vote out nick right now tonight", start=30.0)
    pieces = pieces_of((10.0, 14.0, ["SANDRA: We should all vote out Nick right now tonight."]))
    pinned, offset = pin_pieces(pieces, words)
    assert abs(offset - 20.0) < 1.0
    assert pinned[0].first == 0 and pinned[0].last == 8


def test_each_act_of_another_cut_has_its_own_shift() -> None:
    # 40 lines of five unique words, the second act 90 s later in the audio than in the subtitles
    def line(k: int) -> str:
        return " ".join(f"w{k}x{j}" for j in range(5))

    words: list[Word] = []
    pieces = []
    for k in range(40):
        cue = 10.0 + k * 3.0
        at = cue + (10.0 if k < 20 else 100.0)
        words += _words(line(k), start=at)
        pieces += pieces_of((cue, cue + 2.5, [f"NICK: {line(k)}."]))
    pinned, _ = pin_pieces(pieces, words)
    assert len(pinned) == 40
    assert [p.first for p in pinned] == [5 * k for k in range(40)]


# ---- finding

def srt_text(lines: list[str], n: int = 30) -> str:
    """A believable English subtitle file: ``n`` cues, each with the next of ``lines`` (cycled)."""
    out = []
    for k in range(n):
        text = lines[k % len(lines)]
        out.append(f"{k + 1}\n00:{k // 60:02d}:{k % 60:02d},000 --> 00:{k // 60:02d}:{k % 60:02d},800\n{text}\n")
    return "\n".join(out)


LABELLED = ["NICK: We are going to win this and that is it", "SANDRA: You know that it is not the plan we had", "LIZ: I think we should talk"]
TURNS = [">> WE ARE GOING TO WIN THIS AND THAT IS IT", ">> YOU KNOW THAT IT IS NOT THE PLAN WE HAD"]
PLAIN = ["We are going to win this and that is it", "You know that it is not the plan we had"]


def test_episode_codes() -> None:
    assert subs.episode_code("Survivor.S51E01.Permanent.Uncertainty.1080p.AMZN.WEB-DL.DDP5.1.H.264-RAWR.mkv") == (51, 1)
    assert subs.episode_code("Australian Survivor S01E03.mkv") == (1, 3)
    assert subs.episode_code("S40_E02.srt") == (40, 2)
    assert subs.episode_code("Survivor 10x05 dvdrip.avi") == (10, 5)
    assert subs.episode_code("audio.wav") is None
    assert subs.episode_code("movie.1920x1080.x264.mkv") is None


def test_subtitle_streams_of_a_video() -> None:
    streams = subs.parse_subtitle_streams(SAMPLE_STREAMS)
    assert [(s["index"], s["lang"], s["codec"]) for s in streams] == [
        ("0:2", "eng", "subrip"), ("0:3", "eng", "subrip"), ("0:4", "fre", "subrip"), ("0:5", "eng", "hdmv_pgs_subtitle"),
        ("0:6", "", "ass"), ("0:7", "und", "webvtt")]
    # English text codecs only, the hearing-impaired one first; "Signs and songs" has a title that is not English
    assert [s["index"] for s in subs.english_text_streams(streams)] == ["0:3", "0:2", "0:7"]


def test_embedded_streams_are_extracted_in_one_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "-map" in command:
            for path in command[1:]:
                if path.endswith(".srt"):
                    Path(path).write_text(srt_text(LABELLED if "0.srt" in path else PLAIN), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr=SAMPLE_STREAMS)

    monkeypatch.setattr(subs.subprocess, "run", fake_run)
    found = subs.embedded_candidates(tmp_path / "ep.mkv")
    assert [c.info["kind"] for c in found] == ["labels", "plain", "plain"]
    assert found[0].where.endswith("#0:3")  # the SDH stream
    assert len(calls) == 2 and calls[1].count("-map") == 3
    assert subs.embedded_candidates(tmp_path / "ep.wav") == []  # audio files have no streams to ask


def test_a_missing_ffmpeg_means_no_subtitles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def broken(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subs.subprocess, "run", broken)
    assert subs.embedded_candidates(tmp_path / "ep.mkv") == []


def test_sidecar_filter_on_realistic_names(tmp_path: Path) -> None:
    season = tmp_path / "Season 40 - Winners at War 1080p"
    (season / "Subs" / "Survivor.S40E05.The.Buddy.System.720p.AMZN").mkdir(parents=True)
    (season / "subs").mkdir(exist_ok=True)
    media = season / "Survivor.S40E05.The.Buddy.System.on.Steroids.1080p.mkv"
    media.write_bytes(b"")
    good = srt_text(LABELLED)
    mine = ["S40E05.srt", "Survivor.S40E05.The.Buddy.System.on.Steroids.1080p.en.srt", "Survivor.S40E05.English.WEB-KiNGS.srt",
            "subs/S40E05.vtt", "Subs/Survivor.S40E05.The.Buddy.System.720p.AMZN/2_English.srt"]
    skipped = ["S40E04.srt", "S40_E05_ru_gpt.srt", "Survivor.S40E05.ru.srt", "S40E05.translated.srt", "S40E05_whisperx.srt",
               "S40E05_edit.srt", "S40E05_synced.srt", "S40E05.txt", "Subs/S40E06.srt", "subs.en.srt", "ru_S40E05.srt"]
    for name in mine + skipped:
        (season / name).write_text(good, encoding="utf-8")
    assert sorted(p.relative_to(season).as_posix() for p in subs.sidecar_paths(media)) == sorted(mine)
    # a file by that name that is Russian (or not English at all) is read but not used
    (season / "S40E05.srt").write_text(srt_text(["Привет, как дела у тебя сегодня вечером?"]), encoding="utf-8")
    (season / "Survivor.S40E05.English.WEB-KiNGS.srt").write_text(srt_text(["Das ist ein Satz auf Deutsch und nicht mehr"]), encoding="utf-8")
    assert {Path(c.where).name for c in subs.sidecar_candidates(media)} == {
        "Survivor.S40E05.The.Buddy.System.on.Steroids.1080p.en.srt", "S40E05.vtt", "2_English.srt"}


def test_a_video_without_an_episode_code_uses_only_files_of_its_own_name(tmp_path: Path) -> None:
    media = tmp_path / "Pilot.mkv"
    media.write_bytes(b"")
    for name in ("Pilot.srt", "Pilot.en.srt", "Other.srt", "S01E01.srt"):
        (tmp_path / name).write_text(srt_text(LABELLED), encoding="utf-8")
    assert sorted(p.name for p in subs.sidecar_paths(media)) == ["Pilot.en.srt", "Pilot.srt"]


def test_web_vtt_with_short_times_is_read() -> None:
    vtt = "WEBVTT\n\n" + "\n".join(f"00:{k:02d}.000 --> 00:{k:02d}.800\n{LABELLED[k % 3]}\n" for k in range(30))
    info = subs.describe_text(vtt)
    assert (info["kind"], info["cues"]) == ("labels", 30)


def test_archive_picks_the_file_with_most_named_lines(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    for name, text in (("a.srt", srt_text(PLAIN)), ("b.srt", srt_text(LABELLED)), ("c.srt", srt_text(LABELLED, 24))):
        (archive / name).write_text(text, encoding="utf-8")
    rows = [{"season": 5, "episode": 2, "path": str(archive / "a.srt"), "language": "en", "n_label_lines": 0, "n_turn_marks": 0, "n_cues": 30},
            {"season": 5, "episode": 2, "path": "b.srt", "language": "en", "n_label_lines": 30, "n_turn_marks": 0, "n_cues": 30},
            {"season": 5, "episode": 2, "path": str(archive / "c.srt"), "language": "en", "n_label_lines": 24, "n_turn_marks": 0, "n_cues": 24},
            {"season": 5, "episode": 3, "path": str(archive / "a.srt"), "language": "en", "n_label_lines": 99},
            {"season": 5, "episode": 2, "path": str(archive / "a.srt"), "language": "ru", "n_label_lines": 99}]
    (archive / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    found = subs.archive_candidates(archive, 5, 2)
    assert [(c.info["kind"], c.info["labels"]) for c in found] == [("labels", 30), ("labels", 24), ("plain", 0)]
    assert subs.archive_candidates(archive, 9, 9) == []
    assert subs.archive_candidates(tmp_path / "none", 5, 2) == []


def test_zip_members_are_read_in_memory_only() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../evil.srt", srt_text(LABELLED))
        archive.writestr("run.exe", b"MZ")
        archive.writestr("big.txt", b"x")
    assert [len(m) for m in subs.unzip_texts(buffer.getvalue())] == [len(srt_text(LABELLED))]
    assert subs.unzip_texts(b"not a zip") == []


class Net:
    """A fake ``_fetch``: answers by url, counts the requests."""

    def __init__(self, answers: dict[str, bytes | None]) -> None:
        self.answers = answers
        self.urls: list[str] = []

    def __call__(self, url: str, timeout_s: float = 20.0) -> bytes | None:
        self.urls.append(url)
        return next((v for k, v in self.answers.items() if url.startswith(k)), None)


def stremio(imdb: str, ids: list[str]) -> bytes:
    items = [{"id": i, "lang": "eng", "url": f"https://dl.example/{i}"} for i in ids] + [{"id": "x", "lang": "rus", "url": "https://dl.example/x"}]
    return json.dumps({"subtitles": items}).encode()


def test_online_stops_at_named_lines_and_stays_polite(monkeypatch: pytest.MonkeyPatch) -> None:
    net = Net({"https://opensubtitles-v3.strem.io/subtitles/series/tt1:3:4.json": stremio("tt1", ["1", "2", "3", "4"]),
               "https://dl.example/1": srt_text(PLAIN).encode(), "https://dl.example/2": srt_text(LABELLED).encode()})
    monkeypatch.setattr(subs, "_fetch", net)
    found = subs.online_candidates("tt1", 3, 4)
    assert [c.info["kind"] for c in found] == ["plain", "labels"] + ["plain"] * 0
    assert net.urls[0].endswith("tt1:3:4.json") and len(net.urls) <= 4  # list + at most three downloads
    assert not any("gestdown" in u for u in net.urls) or len(net.urls) <= 8


def test_online_falls_to_gestdown_and_prefers_hearing_impaired(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = json.dumps({"matchingSubtitles": [
        {"downloadUri": "/subtitles/download/aaa", "hearingImpaired": False, "downloadCount": 900},
        {"downloadUri": "/subtitles/download/bbb", "hearingImpaired": True, "downloadCount": 5}]}).encode()
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.writestr("ep.srt", srt_text(LABELLED))
    net = Net({"https://opensubtitles-v3.strem.io": json.dumps({"subtitles": []}).encode(),
               "https://api.gestdown.info/subtitles/get/5rxCcEsDlBbYRBuNnzkGXf/12/3/English": listing,
               "https://api.gestdown.info/subtitles/download/bbb": zipped.getvalue(),
               "https://api.gestdown.info/subtitles/download/aaa": b"<!DOCTYPE html><html>blocked</html>"})
    monkeypatch.setattr(subs, "_fetch", net)
    found = subs.online_candidates("tt0239195", 12, 3)
    assert [c.origin for c in found] == ["online:gestdown"] and found[0].info["kind"] == "labels"
    assert net.urls[2].endswith("/download/bbb")  # the hearing-impaired one first
    assert len(net.urls) <= 8


def test_a_network_failure_is_a_warning_not_a_crash(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import requests

    def down(*args: Any, **kwargs: Any) -> None:
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(requests, "get", down)
    monkeypatch.setattr(subs, "_last_request", {})
    monkeypatch.setattr(subs, "MIN_REQUEST_INTERVAL_S", 0.0)
    assert subs.online_candidates("tt0239195", 1, 1) == []
    assert "not reachable" in caplog.text


def _search_cfg(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    return {"online": False, "imdb": "tt0239195", "archive": None, **extra}


def test_find_subtitles_order_early_stop_and_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    media = tmp_path / "video" / "Survivor.S05.E02.mkv"
    media.parent.mkdir()
    media.write_bytes(b"")
    (media.parent / "Survivor.S05.E02.srt").write_text(srt_text(TURNS), encoding="utf-8")  # a code like S05.E02 is not matched
    (media.parent / "Survivor.S05E02.srt").write_text(srt_text(TURNS), encoding="utf-8")
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "x.srt").write_text(srt_text(LABELLED), encoding="utf-8")
    (archive / "manifest.jsonl").write_text(json.dumps({"season": 5, "episode": 2, "path": "x.srt", "language": "en"}) + "\n")
    order: list[str] = []
    real_embedded, real_sidecar, real_archive = subs.embedded_candidates, subs.sidecar_candidates, subs.archive_candidates
    monkeypatch.setattr(subs, "embedded_candidates", lambda m: order.append("embedded") or [])
    monkeypatch.setattr(subs, "sidecar_candidates", lambda m: order.append("sidecar") or real_sidecar(m))
    monkeypatch.setattr(subs, "archive_candidates", lambda *a: order.append("archive") or real_archive(*a))
    monkeypatch.setattr(subs, "online_candidates", lambda *a, **k: pytest.fail("named lines were found: no need to go online"))
    out = tmp_path / "out"
    cfg = _search_cfg(tmp_path, archive=str(archive), online=True)
    result = subs.find_subtitles(media, out, cfg)
    assert order == ["embedded", "sidecar", "archive"]
    assert result is not None and result["origin"] == "archive" and result["kind"] == "labels"
    assert result["labels"] == 30 and result["turns"] == 0 and result["cues"] == 30
    assert (out / subs.SOURCE_SRT).read_text(encoding="utf-8") == srt_text(LABELLED)
    assert json.loads((out / subs.SOURCE_JSON).read_text(encoding="utf-8"))["origin"] == "archive"
    # cached: no search again
    monkeypatch.setattr(subs, "embedded_candidates", lambda m: pytest.fail("cached"))
    assert subs.find_subtitles(media, out, cfg) == result
    assert real_embedded is not None


def test_a_better_kind_wins_over_an_earlier_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    media = tmp_path / "Survivor.S05E02.mkv"
    media.write_bytes(b"")
    (tmp_path / "Survivor.S05E02.srt").write_text(srt_text(PLAIN), encoding="utf-8")
    (tmp_path / "S05E02.vtt").write_text("WEBVTT\n\n" + "\n".join(f"00:{k:02d}.000 --> 00:{k:02d}.800\n{TURNS[k % 2]}\n" for k in range(30)), encoding="utf-8")
    monkeypatch.setattr(subs, "embedded_candidates", lambda m: [])
    result = subs.find_subtitles(media, tmp_path / "out", _search_cfg(tmp_path))
    assert result is not None and (result["kind"], result["turns"]) == ("turns", 30)
    assert result["path_or_url"].endswith("S05E02.vtt")


def test_nothing_found_is_cached_too_until_the_search_settings_change(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    media = tmp_path / "Survivor.S05E02.mkv"
    media.write_bytes(b"")
    monkeypatch.setattr(subs, "embedded_candidates", lambda m: [])
    out = tmp_path / "out"
    assert subs.find_subtitles(media, out, _search_cfg(tmp_path)) is None
    assert json.loads((out / subs.SOURCE_JSON).read_text(encoding="utf-8"))["kind"] == "none"
    monkeypatch.setattr(subs, "sidecar_candidates", lambda m: pytest.fail("cached"))
    assert subs.find_subtitles(media, out, _search_cfg(tmp_path)) is None
    # online switched on: look again, now with the network (faked)
    net = Net({"https://opensubtitles-v3.strem.io/subtitles/series/tt0239195:5:2.json": stremio("x", ["7"]),
               "https://dl.example/7": srt_text(LABELLED).encode()})
    monkeypatch.setattr(subs, "sidecar_candidates", lambda m: [])
    monkeypatch.setattr(subs, "_fetch", net)
    result = subs.find_subtitles(media, out, _search_cfg(tmp_path, online=True))
    assert result is not None and result["origin"] == "online:stremio" and result["path_or_url"] == "https://dl.example/7"
    assert len(net.urls) == 2


def test_online_off_or_no_imdb_never_touches_the_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    media = tmp_path / "Survivor.S05E02.mkv"
    media.write_bytes(b"")
    monkeypatch.setattr(subs, "embedded_candidates", lambda m: [])
    monkeypatch.setattr(subs, "_fetch", lambda *a, **k: pytest.fail("network"))
    assert subs.find_subtitles(media, tmp_path / "a", _search_cfg(tmp_path)) is None
    assert subs.find_subtitles(media, tmp_path / "b", _search_cfg(tmp_path, online=True, imdb=None)) is None
    no_code = tmp_path / "video.mkv"
    no_code.write_bytes(b"")
    assert subs.find_subtitles(no_code, tmp_path / "c", _search_cfg(tmp_path, online=True)) is None


def test_english_check_takes_sdh_text_and_refuses_other_languages() -> None:
    from stt.subtitles import looks_english

    sdh = "JEFF PROBST: You know that it is time to vote. SANDRA: I think we go to the beach and we talk about it. Go, go! " * 5
    assert looks_english(sdh)
    assert not looks_english("Vamos a la playa con mis amigos porque hace mucho sol hoy en esta ciudad tan bonita. " * 5)
    assert not looks_english("Wir gehen heute zusammen nach Hause und essen dort etwas Warmes mit der ganzen Familie. " * 5)


# ---- speaker labels: ALL-CAPS names and soft (mixed or lower case) names

def _cue(*lines: str, t: float = 0.0) -> tuple[float, float, list[str]]:
    return (t, t + 0.8, list(lines))


def test_all_caps_names_are_labels_with_or_without_a_turn_mark() -> None:
    assert subs.label_of("JEFF PROBST: We are in Ha Tai Yao,") is not None
    found = subs.label_of(">> PROBST: it is still")
    assert found is not None and found[0] == "PROBST" and found[1] == "it is still"
    assert subs.label_of("- NICK: Hello there friend") is not None


def test_mixed_and_lower_case_names_are_soft_labels() -> None:
    for line in ("Jeff: we go now", ">> jeff: we go now", "Jeff Probst: we go now"):
        found = subs.label_of(line)
        assert found is not None and found[2] is True, line
    caps = subs.label_of("NICK: we go now")
    assert caps is not None and caps[2] is False


@pytest.mark.parametrize("line", ["Note: this is a thing", "Destination: Tarutao Island", "One two three four: text",
                                  "Player 2: hello"])
def test_soft_labels_refuse_generic_words_long_names_and_digits(line: str) -> None:
    assert subs.label_of(line) is None


def test_a_soft_name_counts_only_when_used_at_least_three_times() -> None:
    twice = [_cue("Nick: one thing", t=i) for i in range(2)]
    thrice = [_cue("Nick: one thing", t=i) for i in range(3)]
    assert subs.soft_label_names(twice) == set()
    assert subs.soft_label_names(thrice) == {"NICK"}


def test_a_soft_label_used_once_in_a_sentence_is_not_a_label_in_the_file() -> None:
    cues = [_cue("Their destination: Tarutao Island,", t=1.0)] + [_cue("Plain words here", t=i + 2.0) for i in range(5)]
    pieces = subs.file_pieces(cues)
    assert all(p.person == "" for p in pieces)
    assert "Their destination" in " ".join(p.text for p in pieces)


def test_a_soft_name_used_often_becomes_a_label_in_every_line() -> None:
    cues = [_cue("Sandra: we vote tonight", t=i) for i in range(4)]
    pieces = subs.file_pieces(cues)
    assert [p.person for p in pieces] == ["SANDRA"] * 4
    assert pieces[0].text == "we vote tonight"


def test_host_names_in_any_case_map_to_the_host_marker() -> None:
    for raw in ("JEFF", "Jeff", "jeff probst", "PROBST", "Jeff Probst"):
        assert subs.clean_person(raw) == HOST


def test_file_of_host_labels_and_bare_turn_marks_is_labels_and_keeps_turn_marks() -> None:
    cues = [_cue(">> jeff: Come on in, guys", t=i * 2) for i in range(25)]
    cues += [_cue(">> Where do we go from here friend", t=100 + i * 2) for i in range(25)]
    info = subs.describe_text(_cues_text(cues))
    assert info["kind"] == "labels"
    assert info["names"] == 1
    assert info["turn_marks"] >= 50


def test_describe_text_counts_distinct_names_and_kinds() -> None:
    names = ["NICK", "SANDRA", "LIZ"]
    cues = [_cue(f"{names[i % 3]}: we are going to win this one", t=i) for i in range(30)]
    info = subs.describe_text(_cues_text(cues))
    assert (info["kind"], info["names"], info["cues"]) == ("labels", 3, 30)
    turns = subs.describe_text(_cues_text([_cue(">> we are going to win this one", t=i) for i in range(30)]))
    assert turns["kind"] == "turns" and turns["turn_marks"] == 30
    plain = subs.describe_text(_cues_text([_cue("we are going to win this one", t=i) for i in range(30)]))
    assert plain["kind"] == "plain" and plain["names"] == 0


def test_nineteen_named_lines_are_still_plain_and_twenty_are_labels() -> None:
    def kind(n: int) -> str:
        return subs.describe_text(_cues_text([_cue("NICK: we are going to win this one", t=i) for i in range(n)]))["kind"]

    assert kind(19) == "plain" and kind(20) == "labels"


def test_pieces_of_with_explicit_soft_names_accepts_only_those() -> None:
    cue = _cue("Jeff Probst: hello there", "Nick: hi")
    pieces = subs.pieces_of(cue, {"NICK"})
    assert pieces[0].person == "" and "Jeff Probst" in pieces[0].text
    assert any(p.person == "NICK" for p in pieces)


def _cues_text(cues: list[tuple[float, float, list[str]]]) -> str:
    out = []
    for n, (start, end, lines) in enumerate(cues, 1):
        def stamp(x: float) -> str:
            return f"{int(x) // 3600:02d}:{int(x) // 60 % 60:02d}:{int(x) % 60:02d},{int(x % 1 * 1000):03d}"

        out.append(f"{n}\n{stamp(start)} --> {stamp(end)}\n" + "\n".join(lines) + "\n")
    return "\n".join(out)


def test_the_show_is_found_in_a_dotted_file_name() -> None:
    from stt import subtitles as s

    imdb = {"amazing race": "tt0285335", "australia": None, "survivor": "tt0239195"}
    assert s.imdb_of(Path("/v/The.Amazing.Race.S01E01.mkv"), imdb) == "tt0285335"
    assert s.imdb_of(Path("/v/Australian_Survivor-S10E01.mkv"), imdb) is None
    assert s.imdb_of(Path("/v/Survivor.S48E01.720p.mkv"), imdb) == "tt0239195"
    assert not s.show_allows(Path("/v/The.Amazing.Race.S01E01.mkv"), imdb, "tt0239195")
    assert s.show_allows(Path("/v/Survivor.S48E01.mkv"), imdb, "tt0239195")


def test_only_folder_keeps_the_labels_next_to_the_video_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Experiments hide labels in a folder's file: the sidecar (or embedded) labels must not come back through the search."""
    monkeypatch.setattr(subs, "embedded_candidates", lambda media: [])
    videos, folder, out = tmp_path / "videos", tmp_path / "folder", tmp_path / "out"
    videos.mkdir(), folder.mkdir()
    media = videos / "Survivor.S51E02.mkv"
    media.write_bytes(b"")
    (videos / "Survivor.S51E02.srt").write_text(srt_text(LABELLED), encoding="utf-8")
    (folder / "S51E02.en.srt").write_text(srt_text(TURNS), encoding="utf-8")
    found = subs.find_subtitles(media, out, {"folder": str(folder)})
    assert found is not None and found["kind"] == "labels"  # the sidecar's labels win over the folder's marks
    found = subs.find_subtitles(media, out / "only", {"folder": str(folder), "only_folder": True})
    assert found is not None and (found["kind"], found["origin"]) == ("turns", "season folder")
