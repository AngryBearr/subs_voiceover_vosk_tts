"""Existing subtitles as speaker hints: fixed words, own states, turn marks (fake embedder, no GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stt.assemble import CueOptions, smooth_speakers
from stt.boundaries import OWN_STATE, RelabelParams, WordHints, plan_states, relabel_words, restore_fixed
from stt.speakers import apply_subtitle_names
from stt.subtitle_hints import SubtitleInfo, build_hints, evidence, report, subtitle_persons
from stt.types import Word
from tests.test_boundaries import PERIOD, FakeEmbedder, labels, scene


def _labels(words: list[Word]) -> list[str | None]:
    return [w.speaker for w in words]


def _srt(path: Path, lines: list[tuple[float, float, str]]) -> Path:
    def stamp(t: float) -> str:
        return f"00:{int(t // 60):02d}:{int(t) % 60:02d},{int(round((t % 1) * 1000)):03d}"

    path.write_text("\n".join(f"{k}\n{stamp(a)} --> {stamp(b)}\n{text}\n" for k, (a, b, text) in enumerate(lines, 1)),
                    encoding="utf-8")
    return path


def _episode(n_lines: int = 30, per_line: int = 4) -> list[Word]:
    """Words w0 w1 ... (unique tokens), ``per_line`` per subtitle line, 0.6 s apart, from second 10."""
    return [Word(10 + i * PERIOD, 10 + i * PERIOD + 0.3, f"w{i}") for i in range(n_lines * per_line)]


def _lines(words: list[Word], who: list[str], per_line: int = 4, mark: str = "{}: ", broken: set[int] | None = None) -> list[tuple[float, float, str]]:
    out = []
    for k, person in enumerate(who):
        chunk = words[k * per_line:(k + 1) * per_line]
        text = " ".join(f"zz{k}q{j}" if k in (broken or set()) else w.text for j, w in enumerate(chunk))
        out.append((chunk[0].start - 0.05, chunk[-1].end + 0.05, mark.format(person) + text))
    return out


def test_fixed_words_survive_smoothing() -> None:
    # one 8-word sentence: four words of A's line, four of B's line; the majority vote would give all eight to one speaker
    words = [Word(i * PERIOD, i * PERIOD + 0.3, f"w{i}" + ("." if i == 7 else ""), speaker="A" if i < 4 else "B") for i in range(8)]
    fixed = {i: "A" if i < 4 else "B" for i in range(8)}
    smoothed = smooth_speakers(words, CueOptions())
    assert len(set(_labels(smoothed))) == 1  # the problem
    assert "".join(w.speaker or "-" for w in restore_fixed(smoothed, fixed)) == "AAAABBBB"
    assert labels(words) == "AAAABBBB"  # inputs are not modified


def test_a_cluster_held_by_one_person_is_kept_and_merged_people_get_own_states() -> None:
    # cluster A holds Nick (words 0-19, 40-59) and Sandra (20-39); cluster C is Liz
    words, audio = scene("A" * 60 + "C" * 20, "A" * 20 + "B" * 20 + "A" * 20 + "C" * 20)
    hints = WordHints(persons={"NICK": list(range(0, 20)), "SANDRA": list(range(20, 40)), "LIZ": list(range(60, 80))})
    out, stats = relabel_words(words, audio, FakeEmbedder(), RelabelParams(), hints=hints)
    assert hints.state_of == {"LIZ": "C", "NICK": OWN_STATE + "NICK", "SANDRA": OWN_STATE + "SANDRA"}
    assert stats["own_states"] == 2
    got = _labels(out)
    assert got[:20] == ["P:NICK"] * 20 and got[20:40] == ["P:SANDRA"] * 20  # fixed
    assert got[40:60] == ["P:NICK"] * 20  # not hinted: the voice decides
    assert got[60:] == ["C"] * 20


def test_named_line_wins_against_the_audio_and_the_diarizer() -> None:
    # the audio of words 8-11 is B's voice, the diarizer says B, the subtitles say Sandra (cluster A's person)
    words, audio = scene("A" * 8 + "B" * 4 + "A" * 8 + "B" * 20, "A" * 8 + "B" * 4 + "A" * 8 + "B" * 20)
    free, _ = relabel_words(words, audio, FakeEmbedder(), RelabelParams())
    assert _labels(free)[8:12] == ["B"] * 4
    hints = WordHints(persons={"SANDRA": list(range(0, 12)) + list(range(12, 20)), "NICK": list(range(20, 40))})
    out, _ = relabel_words(words, audio, FakeEmbedder(), RelabelParams(), hints=hints)
    assert _labels(out)[:20] == ["A"] * 20
    assert hints.state_of["SANDRA"] == "A" or hints.state_of["SANDRA"].startswith(OWN_STATE)


def test_a_person_with_few_words_in_a_shared_cluster_takes_the_main_cluster() -> None:
    words, _ = scene("A" * 40, "A" * 40)
    own = [w.speaker for w in words]
    hints = WordHints(persons={"NICK": list(range(0, 25)), "BOB": [30, 31, 32]})
    plan_states(words, own, hints, {"A"})
    assert hints.state_of == {"NICK": "A", "BOB": "A"}  # Nick holds 89 % of the hinted seconds; Bob has 3 words
    assert all(hints.fixed[i] == "A" for i in (30, 31, 32))
    hints = WordHints(persons={"NICK": list(range(0, 15)), "BOB": [30, 31, 32, 33, 34, 35, 36, 37, 38]})
    plan_states(words, own, hints, {"A"})  # 15 : 9 words: nobody holds 80 %
    assert hints.state_of == {"NICK": "P:NICK", "BOB": "A"}


def test_without_viterbi_own_states_need_no_centroids() -> None:
    words, _ = scene("A" * 40, "A" * 40)
    hints = WordHints(persons={"NICK": list(range(0, 15)), "SANDRA": list(range(15, 30))})
    extra = plan_states(words, [w.speaker for w in words], hints, {"A"}, emb=None)
    assert extra == {}
    assert hints.state_of == {"NICK": "P:NICK", "SANDRA": "P:SANDRA"}
    assert len(hints.fixed) == 30


def test_nothing_changes_without_hints() -> None:
    words, audio = scene("AAAAAA" + "BBBBBB", "AABAAA" + "BBBBBB")
    base, base_stats = relabel_words(words, audio, FakeEmbedder(), RelabelParams())
    for hints in (None, WordHints()):
        out, stats = relabel_words(words, audio, FakeEmbedder(), RelabelParams(), hints=hints)
        assert _labels(out) == _labels(base)
        assert {k: v for k, v in stats.items() if k not in ("own_states", "embed_s")} == {k: v for k, v in base_stats.items() if k != "embed_s"}


def test_turn_marks_make_a_change_cheaper() -> None:
    # two speakers with close voices: without a mark the switch penalty keeps the middle word with the first speaker
    words, audio = scene("A" * 6 + "B" * 6, "A" * 6 + "B" * 6)
    emb = np.zeros((12, 8), dtype=np.float32)
    emb[:6, 0] = 1.0
    emb[6:, 0], emb[6:, 1] = 0.55, 0.835
    emb[6, 0], emb[6, 1] = 0.75, 0.66  # the first word of B is in between
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    class Fixed:
        def batch(self, windows: np.ndarray) -> np.ndarray:
            return emb

    own_a_all = [Word(w.start, w.end, w.text, speaker="A" if i < 7 else "B") for i, w in enumerate(words)]  # diarizer is late by one word
    plain, _ = relabel_words(own_a_all, audio, Fixed(), RelabelParams())
    marked, _ = relabel_words(own_a_all, audio, Fixed(), RelabelParams(), hints=WordHints(turns={6}))
    assert "".join(w.speaker or "-" for w in plain) == "AAAAAAABBBBB"
    assert "".join(w.speaker or "-" for w in marked) == "AAAAAABBBBBB"


def test_build_hints_from_named_lines_and_not_confirmed_lines(tmp_path: Path) -> None:
    words = _episode()
    who = ["SANDRA"] * 10 + ["NICK"] * 10 + ["LIZ"] * 10
    path = _srt(tmp_path / "ep.srt", _lines(words, who, broken={3, 12}))
    hints, info = build_hints(path, words)
    assert info.kind == "labels"
    assert (info.pinned_lines, info.not_confirmed) == (28, 2)
    assert hints.turns == set()
    assert sorted(hints.persons) == ["LIZ", "NICK", "SANDRA"]
    assert len(hints.persons["SANDRA"]) == 36 and len(hints.persons["NICK"]) == 36 and len(hints.persons["LIZ"]) == 40
    assert 12 not in {i // 4 for i in hints.persons["NICK"]}  # the line the audio does not confirm gives no words
    assert info.line_persons.count("SANDRA") == 9


def test_build_hints_shifts_by_the_start_of_the_piece(tmp_path: Path) -> None:
    words = _episode()
    lines = [(a + 100.0, b + 100.0, text) for a, b, text in _lines(words, ["NICK"] * 30)]
    hints, info = build_hints(_srt(tmp_path / "ep.srt", lines), words, start_s=100.0)
    assert info.pinned_lines == 30 and len(hints.persons["NICK"]) == 120


def test_build_hints_from_turn_marks(tmp_path: Path) -> None:
    words = _episode()
    path = _srt(tmp_path / "ep.srt", _lines(words, [""] * 30, mark=">> {}"))
    hints, info = build_hints(path, words)
    assert info.kind == "turns" and hints.persons == {}
    assert hints.turns == {4 * k for k in range(30)}


def test_plain_subtitles_give_no_hints(tmp_path: Path) -> None:
    words = _episode()
    hints, info = build_hints(_srt(tmp_path / "ep.srt", _lines(words, [""] * 30, mark="{}")), words)
    assert info.kind == "plain" and hints.persons == {} and hints.turns == set()


def test_evidence_names_the_person_and_marks_the_host() -> None:
    hints = WordHints(state_of={"SANDRA": "P:SANDRA", "NICK": "A", "BOB": "A", "_HOST": "H", "ONCE": "Z"})
    info = SubtitleInfo("labels", 0.0, 20, 0, ["SANDRA"] * 5 + ["NICK"] * 4 + ["BOB"] * 4 + ["_HOST"] * 3 + ["ONCE"])
    assert evidence(hints, info) == {"P:SANDRA": ["caption label: SANDRA x5"], "H": ["host per subtitles x3"]}
    assert subtitle_persons(hints, info) == {"P:SANDRA": "SANDRA"}  # A is shared by two people, Z has one line, H is the host


def test_report_counts_relabelled_fixed_words() -> None:
    words, _ = scene("AAAA", "AAAA")
    final = [Word(w.start, w.end, w.text, speaker="P:NICK" if i < 2 else "A") for i, w in enumerate(words)]
    hints = WordHints(persons={"NICK": [0, 1]}, state_of={"NICK": "P:NICK"}, fixed={0: "P:NICK", 1: "P:NICK"})
    info = SubtitleInfo("labels", 0.3, 20, 2, ["NICK"] * 20)
    assert report(hints, info, final, [w.speaker for w in words]) == {
        "kind": "labels", "offset_s": 0.3, "pinned_lines": 20, "not_confirmed": 2, "fixed_words": 2, "turns": 0,
        "persons": {"NICK": {"label": "P:NICK", "words": 2}}, "relabelled": 2}


def test_subtitle_name_fills_only_an_open_name() -> None:
    speakers = [{"id": "S01", "subtitle_person": "O'NEIL", "name": None}, {"id": "S02", "subtitle_person": "LIZ", "name": "Elizabeth"},
                {"id": "S03", "name": None}]
    apply_subtitle_names(speakers)
    assert [s["name"] for s in speakers] == ["O'Neil", "Elizabeth", None]
    assert speakers[0]["name_source"] == "subtitles" and "name_source" not in speakers[2]


def _episode_run(tmp_path: Path, config: dict, **flags: bool):  # type: ignore[no-untyped-def]
    import argparse

    from stt.run import Episode

    args = argparse.Namespace(season_glossary=None, force=None, until="dub", **flags)
    return Episode(tmp_path / "Season 1" / "Survivor.S01E02.mkv", tmp_path / "out", args, config)


def test_run_assembles_again_only_when_the_subtitle_source_changed(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import stt.subtitles as subs

    found = {"origin": "sidecar", "kind": "labels", "sha1": "abc", "labels": 80}
    calls: list[Path] = []
    monkeypatch.setattr(subs, "find_subtitles", lambda media, out, cfg: calls.append(out) or found)
    config = {"subtitles": {"enabled": True, "turn_bonus": 0.3}}
    done = tmp_path / "out" / "Survivor.S01E02" / "subs.en.json"
    done.parent.mkdir(parents=True)
    done.write_text("[]")

    def fresh(cfg: dict, **flags: bool):  # type: ignore[no-untyped-def]
        return _episode_run(tmp_path, cfg, **flags)

    ep = fresh({})  # no subtitles section: as before, the cached transcript stays
    assert ep.subtitle_key() == "none" and not ep.needed("transcribe", [done])
    assert calls == []
    ep = fresh(config)
    assert ep.subtitle_key() == "labels:abc:0.3"
    assert not ep.needed("transcribe", [done])  # made before this rule (no stamp): stays, later LLM stages would rerun
    ep.subtitle_stamp.parent.mkdir(parents=True, exist_ok=True)
    ep.subtitle_stamp.write_text(ep.subtitle_key() + "\n")
    assert not ep.needed("transcribe", [done])
    assert fresh(config, no_subtitles=True).needed("transcribe", [done])  # switched off: back to the plain assembly
    found["sha1"] = "def"  # another file
    assert fresh(config).needed("transcribe", [done])
    found["kind"] = "plain"  # a plain file gives no hints: same as none
    assert fresh({"subtitles": {"enabled": False}}).subtitle_key() == "none"
    assert fresh(config).subtitle_key() == "none"


def test_a_failing_search_does_not_stop_the_run(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import stt.subtitles as subs

    def boom(*args: object) -> None:
        raise RuntimeError("disk")

    monkeypatch.setattr(subs, "find_subtitles", boom)
    assert _episode_run(tmp_path, {"subtitles": {"enabled": True}}).subtitle_key() == "none"


def test_captions_naming_the_host_still_give_their_turn_marks(tmp_path: Path) -> None:
    from stt.subtitle_hints import build_hints

    body = ""
    words = []
    for k in range(30):
        start = 10.0 + k * 2.0
        text = f"line{k} says hello"
        words += [Word(start + j * 0.5, start + j * 0.5 + 0.4, t) for j, t in enumerate(text.split())]
        prefix = ">> PROBST: " if k % 2 == 0 else ">> "
        body += f"{k + 1}\n00:00:{int(start):02d},000 --> 00:00:{int(start) + 1:02d},900\n{prefix}{text}\n\n"
    path = tmp_path / "ep.srt"
    path.write_text(body, encoding="utf-8")
    hints, info = build_hints(path, words)
    assert info.kind == "turns"
    assert len(hints.turns) == 30


def test_captions_with_many_host_lines_are_labels_and_keep_their_marks(tmp_path: Path) -> None:
    from stt.subtitle_hints import build_hints

    body = ""
    words = []
    for k in range(50):
        start = 10.0 + k * 2.0
        text = f"line{k} says hello"
        words += [Word(start + j * 0.5, start + j * 0.5 + 0.4, t) for j, t in enumerate(text.split())]
        prefix = ">> PROBST: " if k % 2 == 0 else ">> "
        stamp = f"00:{int(start) // 60:02d}:{int(start) % 60:02d},000 --> 00:{int(start) // 60:02d}:{int(start) % 60:02d},900"
        body += f"{k + 1}\n{stamp}\n{prefix}{text}\n\n"
    path = tmp_path / "ep.srt"
    path.write_text(body, encoding="utf-8")
    hints, info = build_hints(path, words)
    assert info.kind == "labels"
    assert len(hints.persons) == 1
    assert len(hints.turns) == 50
