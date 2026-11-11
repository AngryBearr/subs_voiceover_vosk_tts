"""The fill pass: words for stretches with speech but no words (fakes only: no models, no GPU)."""

from __future__ import annotations

import argparse
import dataclasses
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stt import fill, run, transcribe
from stt.assemble import CueOptions, build_cues, subtitle_records
from stt.runner import load_json, save_json
from stt.types import Segment, Word


def _w(start: float, end: float, text: str, score: float | None = 0.9, **kw: Any) -> Word:
    return Word(start=start, end=end, text=text, score=score, **kw)


def _seg(*words: Word) -> Segment:
    return Segment(words[0].start, words[-1].end, " ".join(w.text for w in words), list(words))


def _audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * fill.SAMPLE_RATE), dtype=np.float32)


# ---- free_stretches

def test_free_stretches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fill, "speech_regions", lambda audio, **k: [(0.0, 10.0), (20.0, 20.8), (30.0, 100.0)])
    words = [_w(0.0, 3.0, "a"), _w(7.0, 10.0, "b")]
    got = fill.free_stretches(_audio(110.0), words, min_s=1.0, pad_s=0.3, context_s=0.25, max_s=28.0)
    # (3.3, 6.7) is widened; (20, 20.8) is too short; (30, 100) is split into three equal parts
    assert got[0] == (3.05, 6.95)
    third = (100.25 - 29.75) / 3
    assert got[1:] == [(round(29.75 + i * third, 3), round(29.75 + (i + 1) * third, 3)) for i in range(3)]


def test_free_stretches_nothing_when_words_cover_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fill, "speech_regions", lambda audio, **k: [(1.0, 5.0)])
    assert fill.free_stretches(_audio(6.0), [_w(0.5, 2.5, "a"), _w(2.6, 5.5, "b")]) == []


def test_free_stretches_clip_to_the_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fill, "speech_regions", lambda audio, **k: [(0.1, 2.0)])
    assert fill.free_stretches(_audio(2.1), []) == [(0.0, 2.1)]


# ---- accept

STRETCH = (10.0, 13.0)


def test_accept_keeps_the_words_and_marks_them() -> None:
    seg = _seg(_w(10.2, 10.6, "Go"), _w(10.6, 10.6, "team", 0.7), _w(11.0, 12.0, "go"))
    words, reason = fill.accept(seg, STRETCH, [])
    assert reason is None
    assert [w.source for w in words] == ["fill"] * 3 and [w.score for w in words] == [0.9, 0.7, 0.9]
    assert words[1].end == pytest.approx(10.65)  # a zero-length word gets 0.05 s


def test_accept_outside() -> None:
    assert fill.accept(_seg(_w(20.0, 21.0, "far")), STRETCH, []) == ([], "outside")
    assert fill.accept(Segment(10.0, 11.0, "x"), STRETCH, []) == ([], "outside")  # no words at all


def test_accept_drops_only_the_words_outside() -> None:
    words, reason = fill.accept(_seg(_w(8.0, 8.4, "early"), _w(10.5, 11.0, "inside")), STRETCH, [])
    assert reason is None and [w.text for w in words] == ["inside"]


def test_accept_empty() -> None:
    assert fill.accept(_seg(_w(10.5, 11.0, "...")), STRETCH, []) == ([], "empty")


def test_accept_score() -> None:
    assert fill.accept(_seg(_w(10.5, 11.0, "maybe", 0.2), _w(11.0, 11.5, "words", 0.3)), STRETCH, [])[1] == "score"
    assert fill.accept(_seg(_w(10.5, 11.0, "fine", None)), STRETCH, [])[1] is None  # None counts as 1.0


def test_accept_stock() -> None:
    assert fill.accept(_seg(_w(10.5, 11.0, "Thank"), _w(11.0, 11.5, "you.")), STRETCH, []) == ([], "stock")


def test_accept_echo_of_the_neighbours() -> None:
    near = [_w(8.0, 8.5, "we"), _w(8.5, 9.0, "have"), _w(9.0, 9.9, "to")]
    seg = _seg(_w(10.2, 10.6, "We"), _w(10.6, 11.0, "have"), _w(11.0, 11.4, "won"), _w(11.4, 12.0, "it"))
    assert fill.accept(seg, STRETCH, near) == ([], "echo")
    far = [_w(1.0, 1.5, "we"), _w(1.5, 2.0, "have")]  # too far from the stretch: not a neighbour
    assert fill.accept(seg, STRETCH, far)[1] is None
    scattered = [_w(8.0, 8.5, "we"), _w(8.5, 9.0, "won"), _w(9.0, 9.9, "it")]  # a run of two of four words is enough
    assert fill.accept(seg, STRETCH, scattered)[1] == "echo"
    one = [_w(9.0, 9.9, "we")]
    assert fill.accept(seg, STRETCH, one)[1] is None  # one of four words in common: not an echo


# ---- run

class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, name: str, env: str, args: list[str], out_path: Path, log_dir: Path, force: bool) -> dict[str, Any]:
        self.calls.append(args)
        data = {"model": "m", "transcribe_s": 7.5, "segments": [
            _seg(_w(10.5, 11.5, "Run", 0.9), _w(11.5, 12.0, "faster", 0.8)).to_dict(),
            _seg(_w(14.5, 14.9, "Thank", 0.9), _w(14.9, 15.3, "you.", 0.9)).to_dict(),
            _seg(_w(20.5, 20.9, "we", 0.9), _w(20.9, 21.3, "must", 0.9)).to_dict(),
        ]}
        save_json(out_path, data)
        return data


@pytest.fixture()
def regions(monkeypatch: pytest.MonkeyPatch) -> list[tuple[float, float]]:
    current = [(10.0, 12.5), (14.0, 15.5), (20.0, 22.0)]
    monkeypatch.setattr(fill, "speech_regions", lambda audio, **k: list(current))
    return current


def _run(tmp_path: Path, runner: FakeRunner, words: list[Word], *, force: bool = False) -> tuple[list[Word], dict[str, Any]]:
    return fill.run(tmp_path, _audio(30.0), words, model="whisper-large-v3", device="cpu", logs=tmp_path / "logs",
                    force=force, min_s=1.0, min_score=0.4, runner=runner)


def test_run_filters_caches_and_reruns(tmp_path: Path, regions: list[tuple[float, float]]) -> None:
    runner = FakeRunner()
    known = [_w(18.0, 19.0, "we"), _w(19.0, 19.6, "must")]  # next to the third stretch: an echo
    words, stats = _run(tmp_path, runner, known)
    assert [w.text for w in words] == ["Run", "faster"] and {w.source for w in words} == {"fill"}
    assert stats == {"stretches": 3, "stretch_s": 7.5, "segments": 3, "accepted_segments": 1,
                     "words": 2, "rejected": {"stock": 1, "echo": 1}, "fill_s": 7.5}
    chunks = load_json(tmp_path / "fill.chunks.json")
    assert len(chunks) == 3 and chunks[0] == [9.75, 12.75]
    report = load_json(tmp_path / "fill.json")
    assert report["lines"] == [{"start": 10.5, "end": 12.0, "text": "Run faster", "score": 0.85}]
    assert {r["reason"] for r in report["rejected"]} == {"stock", "echo"}
    args = runner.calls[0]
    assert args[args.index("--chunks") + 1] == str(tmp_path / "fill.chunks.json")
    assert args[args.index("--model") + 1] == "whisper-large-v3" and args[0] == "asr"

    _run(tmp_path, runner, known)  # same stretches: the worker output is reused
    assert len(runner.calls) == 1
    _run(tmp_path, runner, known, force=True)
    assert len(runner.calls) == 2
    regions[:] = [(10.0, 12.5)]  # other stretches: new recognition
    _run(tmp_path, runner, known)
    assert len(runner.calls) == 3


def test_run_without_stretches_does_not_call_the_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fill, "speech_regions", lambda audio, **k: [])
    (tmp_path / "asr.fill.json").write_text("{}")
    runner = FakeRunner()
    words, stats = _run(tmp_path, runner, [])
    assert words == [] and stats["stretches"] == 0 and runner.calls == []
    assert not (tmp_path / "asr.fill.json").exists()


# ---- cues

def test_fill_words_make_their_own_cue() -> None:
    words = [_w(0.0, 0.5, "Hello", speaker="A"), _w(0.5, 1.0, "there.", speaker="A"),
             _w(1.1, 1.5, "Go", speaker="A", source="fill"), _w(1.5, 1.9, "go!", speaker="A", source="fill"),
             _w(2.0, 2.4, "Yes.", speaker="A")]
    cues = build_cues(words, CueOptions())
    assert [c.text for c in cues] == ["Hello there.", "Go go!", "Yes."]
    records = subtitle_records(cues, {"A": "S1"}, {"S1": "male"})
    assert [r.get("fill") for r in records] == [None, True, None]
    assert "fill" not in records[0]


# ---- wiring

def test_fill_flags_parse() -> None:
    seen: list[argparse.Namespace] = []
    real = transcribe.transcribe
    transcribe.transcribe = lambda a: seen.append(a) or Path(".")  # type: ignore[assignment]
    try:
        transcribe.main(["x.mkv", "--out", "o"])
        transcribe.main(["x.mkv", "--out", "o", "--fill", "whisper-large-v3", "--fill-min-s", "2", "--force", "fill",
                         "--fill-edges"])
    finally:
        transcribe.transcribe = real  # type: ignore[assignment]
    assert seen[0].fill == "none" and seen[0].fill_min_s == 1.0 and seen[0].fill_min_score == 0.7
    assert seen[0].fill_edges is False
    assert seen[1].fill == "whisper-large-v3" and seen[1].fill_min_s == 2.0 and seen[1].force == ["fill"]
    assert seen[1].fill_edges is True


@pytest.mark.parametrize("fill_config, expected", [
    ({"enabled": True, "model": "whisper-large-v3", "min_s": 1.5, "min_score": 0.5},
     ["--fill", "whisper-large-v3", "--fill-min-s", "1.5", "--fill-min-score", "0.5"]),
    ({"enabled": False}, None),
])
def test_run_passes_the_fill_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fill_config: dict[str, Any],
                                   expected: list[str] | None) -> None:
    args = argparse.Namespace(season_glossary=None, force=None, until="mix", engine="f5", start=None, duration=None,
                              planned_engines=["f5"], mode="fast")
    config = {"mix": {}, "voiceover": {"engines": {"f5": {}}}, "gpu": {"parallel": "off"}, "ocr": {"enabled": False},
              "modes": {"fast": {"asr": "w", "fuse": []}}, "diarizer": "x", "fill": fill_config}
    episode = run.Episode(tmp_path / "S" / "ep01.mkv", tmp_path / "out", args, config, lock=threading.Lock())
    monkeypatch.setattr(episode, "subtitle_source", lambda: None)
    seen: list[list[str]] = []
    monkeypatch.setattr(transcribe, "main", lambda argv: seen.append(argv))
    episode.transcribe()
    if expected is None:
        assert "--fill" not in seen[0]
    else:
        i = seen[0].index("--fill")
        assert seen[0][i:i + 6] == expected


@pytest.mark.parametrize("fill_config, expected", [
    ({"enabled": False, "subtitles": True}, True), ({"enabled": False, "subtitles": True, "edges": False}, False),
    ({"enabled": False, "subtitles": False}, False),
])
def test_run_passes_fill_edges_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                          fill_config: dict[str, Any], expected: bool) -> None:
    args = argparse.Namespace(season_glossary=None, force=None, until="mix", engine="f5", start=None, duration=None,
                              planned_engines=["f5"], mode="fast")
    config = {"mix": {}, "voiceover": {"engines": {"f5": {}}}, "gpu": {"parallel": "off"}, "ocr": {"enabled": False},
              "modes": {"fast": {"asr": "w", "fuse": []}}, "diarizer": "x", "fill": fill_config}
    episode = run.Episode(tmp_path / "S" / "ep01.mkv", tmp_path / "out", args, config, lock=threading.Lock())
    monkeypatch.setattr(episode, "subtitle_source", lambda: None)
    seen: list[list[str]] = []
    monkeypatch.setattr(transcribe, "main", lambda argv: seen.append(argv))
    episode.transcribe()
    assert ("--fill-edges" in seen[0]) is expected


def test_accept_drops_laughter_and_subtitle_credits() -> None:
    from stt.fill import accept
    from stt.types import Segment, Word

    def seg(text: str, score: float = 0.9) -> Segment:
        words = text.split()
        return Segment(start=10.0, end=12.0, text=text,
                       words=[Word(start=10.0 + i * 0.3, end=10.2 + i * 0.3, text=w, score=score) for i, w in enumerate(words)])

    assert accept(seg("Ha ha ha ha"), (10.0, 12.0), [])[1] == "empty"
    assert accept(seg("Oh! Oh!"), (10.0, 12.0), [])[1] == "empty"
    assert accept(seg("Subtitles by Subtitle Workshop"), (10.0, 12.0), [])[1] == "stock"
    assert accept(seg("Subtitles by the Amara.org community"), (10.0, 12.0), [])[1] == "stock"
    assert accept(seg("Oh, we got it!"), (10.0, 12.0), [])[1] is None
    assert accept(seg("Come on, paddle!", score=0.65), (10.0, 12.0), [])[1] == "score"  # under the 0.7 default


# ---- lines from subtitles in sync

def _srt(cues: list[tuple[float, float, str]]) -> str:
    def stamp(t: float) -> str:
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"

    return "\n".join(f"{i}\n{stamp(a)} --> {stamp(b)}\n{text}\n" for i, (a, b, text) in enumerate(cues, start=1))


def _spoken(text: str, start: float, step: float = 0.3, speaker: str | None = "S01"):
    from stt.types import Word

    return [Word(start=round(start + i * step, 3), end=round(start + (i + 1) * step - 0.05, 3), text=w, speaker=speaker)
            for i, w in enumerate(text.split())]


def test_from_subtitles_inserts_free_unconfirmed_lines_and_keeps_the_label(tmp_path) -> None:
    from stt.fill import from_subtitles

    # 14 of 16 lines are said by the words (in sync, shifted by 1.5 s), two are not and their time is free
    words = []
    cues = []
    for i in range(14):
        text = f"spoken line number {i} with several words here"
        words += _spoken(text, 10 + i * 5)
        cues.append((8.5 + i * 5, 11 + i * 5, text))
    cues.append((80.0, 81.0, "Listen up!"))  # free time between the spoken lines, no label
    cues.append((80.0, 81.0, "RUDY: Grab your backpacks!"))  # the same cue, a labelled voice
    cues.append((300.0, 302.0, "Captioning sponsored by CBS"))
    cues.append((310.0, 311.0, "Whoo!"))
    path = tmp_path / "s.srt"
    path.write_text(_srt(cues), encoding="utf-8")
    got, stats = from_subtitles(path, words, end_s=400.0)
    assert stats["lines"] == 18 and stats["pinned"] == 14 and stats["share"] > 0.7
    assert from_subtitles(path, words, end_s=200.0)[1]["lines"] == 16  # the lines past the audio's end are not counted
    # a test piece: the file's lines outside the piece (its words start at the cut) do not count against the sync
    piece_words = [dataclasses.replace(w, start=w.start - 70.0, end=w.end - 70.0) for w in words if w.start >= 70.0]
    piece_stats = from_subtitles(path, piece_words, start_s=70.0, end_s=30.0)[1]
    assert piece_stats["lines"] == 4 and piece_stats["pinned"] == 2 and piece_stats["inserted"] == 2
    assert stats["inserted"] == 2 and stats["skipped"] == {"credits": 1, "cries": 1}
    texts = [(w.text, w.source) for w in got]
    assert texts == [("Listen", "fill"), ("up!", "fill"), ("Grab", "subtitle"), ("your", "subtitle"), ("backpacks!", "subtitle")]
    assert 81.4 < got[0].start < 81.6 and got[1].end <= got[2].start  # shifted by the offset, the cue's time shared


def test_from_subtitles_takes_nothing_from_a_file_out_of_sync_or_over_spoken_words(tmp_path) -> None:
    from stt.fill import from_subtitles

    words = _spoken("one two three four five six seven eight nine ten", 10.0)
    cues = [(200.0, 201.0, "another cut entirely"), (205.0, 206.0, "nothing of this is said"),
            (210.0, 211.0, "not these words either")]
    path = tmp_path / "s.srt"
    path.write_text(_srt(cues), encoding="utf-8")
    got, stats = from_subtitles(path, words, end_s=400.0)
    assert not got and stats["skipped"] == {"not in sync": 3}
    cues = [(10.0, 13.0, "one two three four five six seven eight nine ten"), (11.0, 12.0, "a different reading")]
    path.write_text(_srt(cues), encoding="utf-8")
    got, stats = from_subtitles(path, words, min_pinned_share=0.4)
    assert not got and stats["skipped"] == {"words there": 1}


# ---- missing words at the edges of a partly recognised line

def _edge_file(tmp_path: Path, cue: tuple[float, float, str], spoken: list, *, extra: int = 4):
    """A file whose first ``extra`` lines are pinned in full (so it is in sync) and one test cue."""
    from stt.fill import from_subtitles

    words = []
    cues = []
    for i in range(extra):
        text = f"spoken line number {i} with several words here"
        words += _spoken(text, 30 + i * 5)
        cues.append((30 + i * 5, 32 + i * 5, text))
    cues.append(cue)
    path = tmp_path / "s.srt"
    path.write_text(_srt(sorted(cues)), encoding="utf-8")
    return lambda **kw: from_subtitles(path, sorted(words + spoken, key=lambda w: w.start), end_s=200.0, **kw)


def _edges(got: list) -> list:
    return [w for w in got if w.source == "edge"]


def test_edges_complete_the_start_of_a_shouted_line(tmp_path: Path) -> None:
    got, stats = _edge_file(tmp_path, (10.0, 12.0, "Come on, Jake, get on the mat right now!"), _spoken("get on the mat right now", 10.9))()
    edge = _edges(got)
    assert [w.text for w in edge] == ["Come", "on,", "Jake,"] and stats["edges"] == 1 and stats["edge_words"] == 3
    assert all(a.end <= b.start for a, b in zip(edge, edge[1:])) and edge[-1].end <= 10.9 - 0.02 + 1e-9
    assert edge[0].start >= 9.0 and all(0.15 - 1e-9 <= w.end - w.start <= 0.5 + 1e-9 for w in edge)
    assert stats["inserted"] == 0


def test_edges_take_a_line_that_matches_only_in_part_with_a_solid_run(tmp_path: Path) -> None:
    got, stats = _edge_file(tmp_path, (10.0, 12.0, "Come on, Jake, get on the mat!"), _spoken("get on the mat", 10.9))()
    edge = _edges(got)
    assert [w.text for w in edge] == ["Come", "on,", "Jake,"] and edge[-1].end <= 10.9 - 0.02 + 1e-9
    assert stats["edge_candidates"] == 1 and stats["edges"] == 1
    assert stats["pinned"] == 4 and stats["lines"] == 5  # the in-sync count is untouched


def test_edges_leave_alone_a_line_with_only_a_short_matched_run(tmp_path: Path) -> None:
    spoken = _spoken("we got", 11.0) + _spoken("completely different words entirely", 10.0, step=0.1, speaker="S02")
    got, stats = _edge_file(tmp_path, (10.0, 12.0, "Let's go guys we got this one"), spoken)()
    assert not _edges(got) and stats["edge_candidates"] == 0


def test_edges_complete_the_end_of_a_line(tmp_path: Path) -> None:
    spoken = _spoken("get on the mat", 10.0) + _spoken("next one", 12.0)
    got, stats = _edge_file(tmp_path, (10.0, 12.0, "Get on the mat, you guys!"), spoken)()
    edge = _edges(got)
    assert [w.text for w in edge] == ["you", "guys!"] and stats["edges"] == 1
    assert edge[0].start >= 10.9 + 0.02 - 1e-9 and edge[-1].end <= 12.0 - 0.02 + 1e-9


def test_edges_leave_a_line_alone_when_others_speak_where_it_belongs(tmp_path: Path) -> None:
    spoken = _spoken("get on the mat right now", 10.9) + _spoken("stop right there everybody now", 10.0, step=0.17, speaker="S02")
    got, stats = _edge_file(tmp_path, (10.0, 12.0, "Come on, Jake, get on the mat right now!"), spoken)()
    assert not _edges(got) and stats["skipped"] == {"edge no room": 1}


def test_edges_do_not_treat_a_substituted_word_as_a_gap(tmp_path: Path) -> None:
    got, stats = _edge_file(tmp_path, (10.0, 12.0, "we have to do it now"), _spoken("we have to do that now", 10.0))()
    assert not _edges(got) and stats["edges"] == 0


def test_edges_fill_a_gap_in_the_middle(tmp_path: Path) -> None:
    spoken = _spoken("we have to", 10.0) + _spoken("now", 12.0)
    got, stats = _edge_file(tmp_path, (10.0, 12.5, "we have to do it now"), spoken)()
    edge = _edges(got)
    assert [w.text for w in edge] == ["do", "it"] and stats["edges"] == 1
    assert 10.85 <= edge[0].start and edge[-1].end <= 12.0


def test_edges_skip_cries_and_can_be_turned_off(tmp_path: Path) -> None:
    run_file = _edge_file(tmp_path, (10.0, 12.0, "Whoo! Let's go."), _spoken("let's go", 11.0))
    got, stats = run_file()
    assert not _edges(got) and stats["skipped"] == {"edge cries": 1}
    run_file = _edge_file(tmp_path, (10.0, 12.0, "s. Get on the mat right now!"), _spoken("get on the mat right now", 10.9))
    got, stats = run_file()
    assert not _edges(got) and stats["skipped"] == {"edge noise": 1}  # a cut word is not voiced
    run_file = _edge_file(tmp_path, (10.0, 12.0, "Come on, Jake, get on the mat right now!"),
                          _spoken("get on the mat right now", 10.9))
    got, stats = run_file(edges=False)
    assert not got and stats["edges"] == 0 and stats["edge_words"] == 0


def test_is_credits() -> None:
    from stt.fill import is_credits

    assert is_credits("Captioning sponsored by CBS and SURVIVOR PRODUCTIONS")
    assert is_credits("Captioned by the National Captioning Institute --www.ncicap.org--")
    assert not is_credits("Grab your backpacks!")
