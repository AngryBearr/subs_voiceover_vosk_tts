"""Season voices: build (fake embeddings, no audio, no models) and apply on small episode folders."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stt import season_voices as sv
from stt.boundaries import WordHints
from stt.runner import load_json, save_json

MODEL = "fake"
DIM = 8
AXIS = {"A": 0, "B": 1, "C": 2, "D": 3}  # true person -> embedding axis
Run = tuple[str, str, int]  # (diarizer cluster, true person, words)


def _unit(person: str, rng: np.random.Generator) -> np.ndarray:
    v = np.zeros(DIM)
    v[AXIS[person]] = 1.0
    v += 0.05 * rng.standard_normal(DIM)
    return v / np.linalg.norm(v)


def make_episode(root: Path, name: str, runs: list[Run], names: dict[str, str | None], ocr_at: set[int] | None = None,
                 seed: int = 0) -> Path:
    """A finished first pass: every run is one sentence of one diarizer cluster; a cluster is one card (``names``)."""
    rng = np.random.default_rng(seed)
    folder = root / name
    folder.mkdir(parents=True)
    clusters = list(dict.fromkeys(c for c, _, _ in runs))
    ids = {c: f"S{k:02d}" for k, c in enumerate(clusters, 1)}
    words: list[dict[str, Any]] = []
    emb: list[np.ndarray] = []
    turns: list[dict[str, Any]] = []
    t = 1.0
    for cluster, person, n in runs:
        t0 = t
        for k in range(n):
            last = k == n - 1
            words.append({"start": round(t, 3), "end": round(t + 0.3, 3), "text": f"w{len(words)}" + ("." if last else ""),
                          "score": 0.9, "speaker": ids[cluster], "source": "ocr" if len(words) in (ocr_at or set()) else None})
            emb.append(_unit(person, rng))
            t += 0.35
        turns.append({"start": round(t0, 3), "end": round(t, 3), "speaker": cluster})
        t += 0.8
    cards = [{"id": ids[c], "diarizer_label": c, "name": names.get(c), "gender": "male" if c < "m" else "female",
              "role": "contestant", "label_ru": f"Голос {c}", "same_as": "S01", "on_screen": [], "embedding": [1.0, 0.0, 0.0],
              "speech_s": 10.0, "cues": sum(1 for r in runs if r[0] == c),
              "words": sum(n for cl, _, n in runs if cl == c)}
             for c in clusters]
    records = [{"index": k, "start": int(w["start"] * 1000), "end": int(w["end"] * 1000), "duration": 300,
                "text": [w["text"]], "gender": "unknown", "speaker": w["speaker"], "overlap": False, "ocr": False}
               for k, w in enumerate(words, 1)]
    save_json(folder / "words.json", words)
    save_json(folder / "speakers.json", cards)
    save_json(folder / "subs.en.json", records)
    save_json(folder / "diarization.fake.refined.fake.json", {"turns": turns, "exclusive": [], "embeddings": {}})
    np.savez(folder / f"words.emb.{MODEL}.npz", starts=[w["start"] for w in words], ends=[w["end"] for w in words],
             emb=np.stack(emb).astype(np.float32), window_s=np.float64(1.0))
    return folder


def spoken(folder: Path, speakers_file: str = "speakers.json", words_file: str = "words.json") -> list[str | None]:
    """The name (or cluster label) of the speaker of every word."""
    cards = {c["id"]: c for c in load_json(folder / speakers_file)}
    out: list[str | None] = []
    for w in load_json(folder / words_file):
        card = cards.get(w["speaker"])
        out.append(None if card is None else card.get("name") or card["diarizer_label"])
    return out


@pytest.fixture()
def season(tmp_path: Path) -> list[Path]:
    """E1: cluster X is mixed (A and B) and named Alice; Y is B alone, Z is C alone and has no name.
    E2/E3: clean clusters that give the season enough words of A, B."""
    root = tmp_path / "work"
    e1 = make_episode(root, "e1", [("X", "A", 12), ("X", "B", 8), ("X", "A", 10), ("Y", "B", 12), ("Z", "C", 15)],
                      {"X": "Alice Smith", "Y": "Bob Jones", "Z": None}, seed=1)
    e2 = make_episode(root, "e2", [("P", "A", 40), ("Q", "B", 40), ("P", "A", 20)], {"P": "Alice Smith", "Q": "Bob Jones"}, seed=2)
    e3 = make_episode(root, "e3", [("P", "A", 30), ("Q", "B", 30)], {"P": "Alice", "Q": "Bob"}, seed=3)
    return [e1, e2, e3]


# ---------------------------------------------------------------- names and build
def test_group_names_merges_spellings_of_one_person() -> None:
    keys = sv.group_names({"Jeff Probst": 100, "Jeff": 30, "Ori Jean-Charles": 50, "Ori": 5, "Natalie Anderson": 10,
                           "Nadiya Anderson": 12, "Jon Misch": 3, "John Misch": 9})
    assert keys["Jeff"] == keys["Jeff Probst"] == "Jeff Probst"
    assert keys["Ori"] == "Ori Jean-Charles"
    assert keys["Natalie Anderson"] != keys["Nadiya Anderson"]  # relatives are two people
    assert keys["Jon Misch"] == keys["John Misch"] == "John Misch"  # the spelling with more words


def test_build_sums_the_season_and_applies_min_words(tmp_path: Path, season: list[Path]) -> None:
    # D: 12 words in e2 and 12 in e3 (24 >= 20); E: 12 words only
    e4 = make_episode(tmp_path / "work", "e4", [("D1", "D", 12), ("T", "C", 12)], {"D1": "Dan Dunn", "T": "Eve Eng"}, seed=4)
    e5 = make_episode(tmp_path / "work", "e5", [("D1", "D", 12)], {"D1": "Dan"}, seed=5)
    voices = sv.build_voices([*season, e4, e5], MODEL, min_words=20)
    assert set(voices.names) == {"Alice Smith", "Bob Jones", "Dan Dunn"}  # "Alice"/"Bob"/"Dan" are spellings; Eve has 12
    assert voices.counts[voices.names.index("Dan Dunn")] == 24
    assert voices.person_of("Alice") == "Alice Smith" and voices.person_of("Eve Eng") is None
    assert np.allclose(np.linalg.norm(voices.vectors, axis=1), 1.0)
    bob = voices.vectors[voices.names.index("Bob Jones")]
    assert bob[AXIS["B"]] > 0.95  # the mixed cluster X is named Alice: its B words add a little to Alice, not to Bob
    assert voices.cards["Alice Smith"]["gender"] == "male"
    assert voices.episodes == ["e1", "e2", "e3", "e4", "e5"]


def test_build_counts_named_subtitle_words_for_the_person(season: list[Path], monkeypatch: pytest.MonkeyPatch) -> None:
    hints = WordHints(persons={"BOB": [12, 13, 14]})  # three words of e1 that the first pass gave to Alice's cluster X
    monkeypatch.setattr(sv, "subtitle_hints", lambda episode: hints if episode.folder.name == "e1" else None)
    named = sv.build_voices(season, MODEL, 20)
    assert named.counts[named.names.index("Bob Jones")] == 12 + 40 + 30 + 3
    assert named.counts[named.names.index("Alice Smith")] == 30 + 60 + 30 - 3  # moved from Alice to Bob


def test_voices_roundtrip_and_reuse(tmp_path: Path, season: list[Path], monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "s.season_voices.npz"
    first = sv.ensure_voices(season, path, MODEL, 20)
    loaded = sv.SeasonVoices.load(path)
    assert loaded is not None and loaded.names == first.names and loaded.fingerprint == first.fingerprint
    assert np.allclose(loaded.vectors, first.vectors, atol=1e-6) and loaded.cards == first.cards
    calls: list[int] = []
    real = sv.build_voices
    monkeypatch.setattr(sv, "build_voices", lambda *a, **k: calls.append(1) or real(*a, **k))
    sv.ensure_voices(season, path, MODEL, 20)
    assert calls == []  # same inputs: reused
    sv.ensure_voices(season[:2], path, MODEL, 20)
    assert calls == [1]  # another set of episodes: built again


# ---------------------------------------------------------------- apply
def test_mixed_cluster_goes_to_the_right_people(season: list[Path]) -> None:
    voices = sv.build_voices(season, MODEL, 20)
    before = spoken(season[0])
    assert before[12:20] == ["Alice Smith"] * 8  # the first pass: B's words sit in Alice's cluster
    result = sv.apply_episode(season[0], voices, model=MODEL)
    assert result.status == "applied" and result.report["words_moved"] >= 7
    after = spoken(season[0])
    assert after[:12] == ["Alice Smith"] * 12 and after[12:20] == ["Bob Jones"] * 8 and after[20:30] == ["Alice Smith"] * 10
    assert after[30:42] == ["Bob Jones"] * 12  # Y is Bob's cluster, which is the person now
    assert after[42:] == ["Z"] * 15  # the cluster without a name stays itself
    cards = load_json(season[0] / "speakers.json")
    assert [c["id"] for c in cards] == ["S01", "S02", "S03"]  # one speaker per person, unique ids
    by_name = {c["name"] or c["diarizer_label"]: c for c in cards}
    assert by_name["Alice Smith"]["diarizer_label"] == "V:Alice Smith" and by_name["Alice Smith"]["diarizer_labels"] == ["X"]
    assert by_name["Bob Jones"]["diarizer_labels"] == ["Y"] and by_name["Bob Jones"]["gender"] == "male"
    assert by_name["Z"]["name"] is None and by_name["Z"]["diarizer_label"] == "Z"
    assert all(c["cues"] > 0 and c["same_as"] is None for c in cards)
    assert by_name["Bob Jones"]["words"] == 20 and by_name["Alice Smith"]["words"] == 22
    subs = load_json(season[0] / "subs.en.json")
    assert {r["speaker_name"] for r in subs} == {"Голос X", "Голос Y", "Голос Z"} and len(subs) == result.report["cues_after"]
    assert (season[0] / "subs.en.srt").exists() and "[S01]" in (season[0] / "subs.en.speakers.srt").read_text("utf-8")
    report = load_json(season[0] / "season_voices.json")
    assert report["per_person"]["Bob Jones"] == {"before": 12, "after": 20}
    assert report["per_person"]["Alice Smith"] == {"before": 30, "after": 22}
    for name in sv.FILES:  # the first pass is kept
        assert (season[0] / f"{name}{sv.BEFORE}.json").exists()


def test_fixed_words_stay_with_their_person(season: list[Path], monkeypatch: pytest.MonkeyPatch) -> None:
    voices = sv.build_voices(season, MODEL, 20)
    # words 0-2 sound like Alice, but a named subtitle line says BOB (first name match)
    monkeypatch.setattr(sv, "subtitle_hints", lambda episode: WordHints(persons={"BOB": [0, 1, 2]}))
    sv.apply_episode(season[0], voices, model=MODEL)
    after = spoken(season[0])
    assert after[:3] == ["Bob Jones"] * 3 and after[3:12] == ["Alice Smith"] * 9


def test_ocr_words_keep_their_speaker(tmp_path: Path, season: list[Path]) -> None:
    # a caption sentence (3 on-screen words) inside Alice's cluster, with B's voice around it
    folder = make_episode(tmp_path / "ocr", "e1", [("X", "A", 12), ("X", "B", 3), ("X", "B", 8), ("X", "A", 10), ("Y", "B", 12)],
                          {"X": "Alice Smith", "Y": "Bob Jones"}, ocr_at={12, 13, 14}, seed=1)
    voices = sv.build_voices(season, MODEL, 20)
    sv.apply_episode(folder, voices, model=MODEL)
    after = spoken(folder)
    assert after[12:15] == ["Alice Smith"] * 3  # a caption has no voice: it stays with the speaker it had
    assert after[:12] == ["Alice Smith"] * 12 and after[15:23] == ["Bob Jones"] * 8 and after[23:33] == ["Alice Smith"] * 10
    assert load_json(folder / "subs.en.json")[0]["ocr"] is False and any(r["ocr"] for r in load_json(folder / "subs.en.json"))


def test_rerun_is_idempotent_and_starts_from_the_first_pass(tmp_path: Path, season: list[Path]) -> None:
    pristine = tmp_path / "pristine"
    shutil.copytree(season[0], pristine / "e1")
    first_pass = load_json(season[0] / "words.json")
    voices = sv.build_voices(season, MODEL, 20)
    first = sv.apply_episode(season[0], voices, model=MODEL)
    snapshot = {n: (season[0] / f"{n}.json").read_bytes() for n in sv.FILES}
    again = sv.apply_episode(season[0], voices, model=MODEL)
    assert first.status == "applied" and again.status == "unchanged"
    assert snapshot == {n: (season[0] / f"{n}.json").read_bytes() for n in sv.FILES}
    # a season rebuilt with another episode is applied from the first pass: the same as on a pristine copy
    e4 = make_episode(tmp_path / "work", "e4", [("P", "A", 40), ("Q", "B", 40)], {"P": "Alice Smith", "Q": "Bob Jones"}, seed=9)
    bigger = sv.build_voices([*season, e4], MODEL, 20)
    sv.apply_episode(season[0], bigger, model=MODEL)
    sv.apply_episode(pristine / "e1", bigger, model=MODEL)
    for name in sv.FILES:
        assert load_json(season[0] / f"{name}.json") == load_json(pristine / "e1" / f"{name}.json")
    assert load_json(season[0] / "words.before_voices.json") == first_pass  # never overwritten by a rerun


def test_a_new_first_pass_replaces_the_saved_one(season: list[Path]) -> None:
    voices = sv.build_voices(season, MODEL, 20)
    sv.apply_episode(season[0], voices, model=MODEL)
    fresh = load_json(season[0] / "words.before_voices.json")
    fresh[0]["text"] = "again"  # the transcript was made again: words.json is a new first pass
    save_json(season[0] / "words.json", fresh)
    sv.apply_episode(season[0], voices, model=MODEL)
    assert load_json(season[0] / "words.before_voices.json")[0]["text"] == "again"


def test_nothing_changes_without_names(tmp_path: Path) -> None:
    folder = make_episode(tmp_path / "work", "e1", [("X", "A", 30), ("Y", "B", 30)], {"X": None, "Y": None})
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    voices = sv.build_voices([folder], MODEL, 20)
    assert len(voices) == 0
    results = sv.apply_season([folder], [folder], tmp_path / "v.npz", model=MODEL)
    assert results["e1"].status == "skipped"
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before  # also no *.before_voices.json


def test_dry_run_writes_nothing(tmp_path: Path, season: list[Path]) -> None:
    before = {str(p): p.read_bytes() for e in season for p in e.iterdir()}
    results = sv.apply_season(season, season, tmp_path / "v.npz", model=MODEL, dry_run=True)
    assert results["e1"].status == "would-apply" and results["e1"].report["words_moved"] > 0
    assert {str(p): p.read_bytes() for e in season for p in e.iterdir()} == before and not (tmp_path / "v.npz").exists()


def test_episode_without_embeddings_is_skipped(season: list[Path]) -> None:
    voices = sv.build_voices(season, MODEL, 20)
    (season[1] / f"words.emb.{MODEL}.npz").unlink()
    assert sv.apply_episode(season[1], voices, model=MODEL).status == "skipped"
    other = np.load(season[0] / f"words.emb.{MODEL}.npz")
    np.savez(season[1] / f"words.emb.{MODEL}.npz", starts=other["starts"], ends=other["ends"], emb=other["emb"])
    assert sv.apply_episode(season[1], voices, model=MODEL).status == "skipped"  # not this episode's words


def test_person_from_another_episode_gets_the_card_of_that_episode(tmp_path: Path, season: list[Path]) -> None:
    # e5 has Carol's voice inside Alice's cluster, with no card of hers; Carol is named in e6 only
    e6 = make_episode(tmp_path / "work", "e6", [("K", "C", 50)], {"K": "Carol Cole"}, seed=6)
    e5 = make_episode(tmp_path / "work", "e5", [("N", "A", 20), ("N", "C", 20)], {"N": "Alice Smith"}, seed=7)
    voices = sv.build_voices([*season, e5, e6], MODEL, 20)
    sv.apply_episode(e5, voices, model=MODEL)
    assert spoken(e5) == ["Alice Smith"] * 20 + ["Carol Cole"] * 20
    card = next(c for c in load_json(e5 / "speakers.json") if c["name"] == "Carol Cole")
    assert card["gender"] == "male" and card["label_ru"] == "Голос K"  # copied from e6's card
    assert card["diarizer_label"] == "V:Carol Cole" and card["diarizer_labels"] == [] and card["words"] == 20


def test_cluster_cli_main_dry_run(season: list[Path], capsys: pytest.CaptureFixture[str]) -> None:
    sv.main([str(season[0].parent), "--dry-run", "--model", MODEL, "--episodes", "e1"])
    out = capsys.readouterr().out
    assert "e1: would-apply" in out and "e2" not in out and "nothing written" in out
    assert json.loads((season[0] / "speakers.json").read_text("utf-8"))[0]["diarizer_label"] == "X"


# ---------------------------------------------------------------- stt.run wiring
def _season_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, dict[str, Any], list[Path], list[str]]:
    from stt import run

    video = tmp_path / "Season 1"
    video.mkdir()
    files = [video / f"{n}.mkv" for n in ("e1", "e2", "e3")]
    for f in files:
        f.write_bytes(b"")
    out = tmp_path / "work"
    runs = [[("X", "A", 12), ("X", "B", 8), ("X", "A", 10), ("Y", "B", 12)], [("P", "A", 40), ("Q", "B", 40)],
            [("P", "A", 30), ("Q", "B", 30)]]
    names = [{"X": "Alice Smith", "Y": "Bob Jones"}, {"P": "Alice Smith", "Q": "Bob Jones"}, {"P": "Alice", "Q": "Bob"}]
    for k, f in enumerate(files):
        make_episode(out, f.stem, runs[k], names[k], seed=k)
        for stage in ("transcribe", "glossary", "speakers"):
            run.Episode(f, out, argparse.Namespace(season_glossary=None, force=None, until="mix"), {}).stamp(stage)
    ran: list[str] = []
    monkeypatch.setattr(run.Episode, "run", lambda self: ran.append(f"{self.media.stem}:{self.args.until}"))
    args = argparse.Namespace(season_glossary=None, force="glossary", until="mix", no_season_voices=False)
    config = {"season_voices": {"enabled": True, "min_words": 20}, "word_speakers": {"model": MODEL}}
    return run, args, config, files, ran


def test_run_applies_the_stage_after_the_first_pass_and_marks_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, args, config, files, ran = _season_run(tmp_path, monkeypatch)
    out = tmp_path / "work"
    assert run.season_voices_enabled(config, args)
    run.run_season_voices(files, out, args, config, ["vosk"])
    assert ran == ["e1:speakers", "e2:speakers", "e3:speakers"]  # every episode up to speakers first
    assert args.force is None  # the forced stages ran in that first pass
    mark = out / "e1" / ".stages" / "speakers"
    assert mark.exists()  # e1 changed: the stages after speakers see it
    assert (out / "e1" / "words.before_voices.json").exists()
    assert (tmp_path / "work" / "Season 1.season_voices.npz").exists()  # beside the voice bank
    mark.unlink()
    run.run_season_voices(files, out, args, config, ["vosk"])  # nothing new: no new mark
    assert not mark.exists()
    args.no_season_voices = True
    assert not run.season_voices_enabled(config, args)
    assert not run.season_voices_enabled({"season_voices": {"enabled": False}}, args)
    assert not run.season_voices_enabled({}, args)


def test_run_stage_waits_for_speakers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, args, config, files, ran = _season_run(tmp_path, monkeypatch)
    args.until = "glossary"
    run.run_season_voices(files, tmp_path / "work", args, config, ["vosk"])
    assert ran == [] and not (tmp_path / "work" / "e1" / "words.before_voices.json").exists()


# ---------------------------------------------------------------- the assembly shared with stt.transcribe
def test_assemble_cues_ids_genders_and_files(tmp_path: Path) -> None:
    from stt.assemble import CueOptions
    from stt.diarize import Diarization, Turn
    from stt.transcribe import assemble_cues, write_transcript
    from stt.types import Word

    words = [Word(1.0, 1.3, "Hello", speaker="b"), Word(1.4, 1.7, "there.", speaker="b"), Word(5.0, 5.3, "Hi.", speaker="a")]
    diarization = Diarization(turns=[Turn(0.5, 2.0, "b"), Turn(4.5, 6.0, "a"), Turn(7.0, 8.0, "silent")])
    cues, ids, records = assemble_cues(words, diarization, CueOptions(), offset=10.0,
                                       gender_of=lambda label: {"a": "female"}.get(label, "male"),
                                       extra_labels=diarization.speakers)
    assert ids == {"b": "S01", "a": "S02", "silent": "S03"}  # by first appearance, speakers without words last
    assert [(r["speaker"], r["gender"], r["start"], r["text"]) for r in records] == [
        ("S01", "male", 11000, ["Hello there."]), ("S02", "female", 15000, ["Hi."])]
    write_transcript(tmp_path, records, words, ids, 10.0)
    assert load_json(tmp_path / "words.json")[2] == {"start": 15.0, "end": 15.3, "text": "Hi.", "score": None,
                                                     "speaker": "S02", "source": None}
    assert "[S01] Hello there." in (tmp_path / "subs.en.speakers.srt").read_text("utf-8")
    assert (tmp_path / "subs.en.srt").read_text("utf-8").startswith("1\n00:00:11,000 --> 00:00:11,700\nHello there.")


def test_own_state_cards_of_subtitle_hints_do_not_hide_the_diarization(season: list[Path]) -> None:
    """A first pass with subtitles has cards labelled "P:NAME", which no diarization knows."""
    folder = season[0]
    cards = json.loads((folder / "speakers.json").read_text(encoding="utf-8"))
    cards.append({**cards[0], "id": "S99", "diarizer_label": "P:SOMEONE", "name": "Someone", "cues": 3})
    (folder / "speakers.json").write_text(json.dumps(cards), encoding="utf-8")
    assert sv.find_diarization(folder, cards) is not None


def test_subtitle_surnames_and_nicknames_find_the_person() -> None:
    voices = sv.SeasonVoices(["Danny Kilby", "Angelica Loblack", "Jeff Probst"], np.eye(3), [50, 50, 50], MODEL,
                             aliases={"jelly": "Angelica Loblack"})
    assert voices.by_first_name("KILBY") == "Danny Kilby"
    assert voices.by_first_name("JELLY") == "Angelica Loblack"
    assert voices.by_first_name("DANNY") == "Danny Kilby"
    assert voices.by_first_name("NOBODY") is None


def test_aliases_come_from_the_cast_list_beside_the_voices(tmp_path: Path) -> None:
    path = tmp_path / "Season_51.season_voices.npz"
    (tmp_path / "Season_51.cast.json").write_text(json.dumps({"cast": [{"name": "Angelica Loblack", "aliases": ["Jelly"]}]}), encoding="utf-8")
    assert sv.cast_aliases(path) == {"jelly": "Angelica Loblack"}
    assert sv.cast_aliases(tmp_path / "Other.season_voices.npz") == {}


def test_prepare_embeddings_recomputes_embeddings_of_an_earlier_transcript(tmp_path, monkeypatch) -> None:
    """A rebuilt transcript (other words) makes the stored word embeddings stale: they are computed again, not skipped."""
    import json as _json

    import numpy as _np

    from stt import audio as stt_audio, boundaries, refine_speakers, season_voices

    folder = tmp_path / "ep"
    folder.mkdir()
    words = [{"start": 1.0, "end": 1.4, "text": "hi", "speaker": "S01"},
             {"start": 2.0, "end": 2.5, "text": "there", "speaker": "S01"}]
    (folder / "words.json").write_text(_json.dumps(words), encoding="utf-8")
    (folder / "speakers.json").write_text("[]", encoding="utf-8")
    (folder / "subs.en.json").write_text("[]", encoding="utf-8")
    (folder / "audio.wav").write_bytes(b"")
    path = season_voices.embedding_path(folder, "resnet293")
    _np.savez(path, starts=_np.array([0.5]), ends=_np.array([0.9]), emb=_np.zeros((1, 4), _np.float32),
              window_s=_np.float64(1.0))  # one word of an older transcript
    calls: list[int] = []
    monkeypatch.setattr(refine_speakers, "make_embedder", lambda model, device: object())
    monkeypatch.setattr(stt_audio, "read_wav", lambda p: _np.zeros(16000 * 3, _np.float32))
    monkeypatch.setattr(boundaries, "word_windows", lambda w, audio, window: _np.zeros((len(w), 16000), _np.float32))
    monkeypatch.setattr(boundaries, "embed_windows", lambda windows, embedder: (calls.append(len(windows)),
                                                                                 _np.ones((len(windows), 4)))[1])
    season_voices.prepare_embeddings([folder], "resnet293")
    assert calls == [2]
    with _np.load(path) as data:
        assert list(data["starts"]) == [1.0, 2.0] and data["emb"].shape == (2, 4)
    season_voices.prepare_embeddings([folder], "resnet293")  # up to date now: nothing computed
    assert calls == [2]


CAST = {"cast": [
    {"name": "Alice Smith", "aliases": ["Ali"], "gender": "female", "role": "Contestant"},
    {"name": "Bob Jones", "aliases": [], "gender": "male", "role": "Contestant"},
    {"name": "Carol Wu", "aliases": ["Caz", "C.W."], "gender": "female", "role": "Contestant"},
    {"name": "Jeff Probst", "aliases": ["Jeff"], "gender": "male", "role": "Host"},
]}


def test_subtitle_labels_resolve_to_the_cast(season: list[Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """Initials and nicknames find the season person; spellings of one new person share one speaker named as the cast does."""
    voices = sv.build_voices(season, MODEL, 20, cast=CAST)
    assert voices.by_first_name("ALI") == "Alice Smith" and voices.by_first_name("_HOST") is None  # no host voice yet
    hints = WordHints(persons={"ALI": [0, 1, 2], "CAZ": list(range(42, 47)), "C.W": list(range(47, 52)),
                               "_HOST": list(range(52, 57))})
    monkeypatch.setattr(sv, "subtitle_hints", lambda episode: hints)
    sv.apply_episode(season[0], voices, model=MODEL)
    cards = load_json(season[0] / "speakers.json")
    names = [c["name"] for c in cards if c.get("cues")]
    assert names.count("Carol Wu") == 1  # "CAZ" and "C.W" are one person (each alone is under the 10 words a state needs)
    carol = next(c for c in cards if c["name"] == "Carol Wu")
    assert (carol["gender"], carol["gender_source"], carol["name_source"]) == ("female", "cast", "subtitles")
    assert spoken(season[0])[:3] == ["Alice Smith"] * 3
    assert not any(n and n.startswith(("Caz", "C.W", "_")) for n in names)


def test_host_label_names_the_cast_host(season: list[Path], monkeypatch: pytest.MonkeyPatch) -> None:
    voices = sv.build_voices(season, MODEL, 20, cast=CAST)
    monkeypatch.setattr(sv, "subtitle_hints", lambda episode: WordHints(persons={"_HOST": list(range(42, 56))}))
    sv.apply_episode(season[0], voices, model=MODEL)
    host = next(c for c in load_json(season[0] / "speakers.json") if c["name"] == "Jeff Probst")
    assert host["gender"] == "male" and host["diarizer_label"] == "P:Jeff Probst"


def test_label_of_a_season_person_does_not_make_a_second_speaker(season: list[Path], monkeypatch: pytest.MonkeyPatch) -> None:
    voices = sv.build_voices(season, MODEL, 20, cast=CAST)
    monkeypatch.setattr(sv, "subtitle_hints", lambda episode: WordHints(persons={"BOB": [0, 1], "BOB JONES": [2, 3]}))
    sv.apply_episode(season[0], voices, model=MODEL)
    names = [c["name"] for c in load_json(season[0] / "speakers.json") if c.get("cues")]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------- naming again after the stage
def _after_stage(folder: Path, current: list[dict[str, Any]], first: list[dict[str, Any]]) -> None:
    """Files as the stage leaves them: its output in place, the first pass saved, the report naming its words.json."""
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "words.json", [{"start": 1.0, "end": 1.3, "text": "w", "speaker": current[0]["id"]}])
    save_json(folder / "speakers.json", current)
    save_json(folder / "subs.en.json", [])
    save_json(sv.before_path(folder, "words"), [{"start": 1.0, "end": 1.3, "text": "w", "speaker": first[0]["id"]}])
    save_json(sv.before_path(folder, "speakers"), first)
    save_json(sv.before_path(folder, "subs.en"), [])
    save_json(folder / sv.REPORT, {"written": {"words": sv._sha(folder / "words.json")}})


def test_first_pass_ids_follow_the_diarizer_label(tmp_path: Path) -> None:
    folder = tmp_path / "e1"
    current = [{"id": "S01", "diarizer_label": "V:Alice Smith", "name": "Alice Smith"},
               {"id": "S02", "diarizer_label": "2", "name": None}, {"id": "S03", "diarizer_label": "9", "name": None}]
    first = [{"id": "S01", "diarizer_label": "P:_HOST"}, {"id": "S02", "diarizer_label": "0"},
             {"id": "S03", "diarizer_label": "2"}]
    _after_stage(folder, current, first)
    assert sv.first_pass_ids(folder) == {"S02": "S03"}  # a person and a label unknown to the first pass have none
    save_json(folder / sv.REPORT, {"written": {"words": "another"}})  # the files are a first pass again
    assert sv.first_pass_ids(folder) == {"S01": "S01", "S02": "S02", "S03": "S03"}


def test_restore_first_pass_moves_the_saved_files_back(tmp_path: Path) -> None:
    folder = tmp_path / "e1"
    first = [{"id": "S01", "diarizer_label": "0", "name": None}]
    _after_stage(folder, [{"id": "S07", "diarizer_label": "V:Bob", "name": "Bob"}], first)
    assert sv.restore_first_pass(folder)
    assert load_json(folder / "speakers.json") == first and load_json(folder / "words.json")[0]["speaker"] == "S01"
    assert not any(sv.before_path(folder, n).exists() for n in sv.FILES)
    assert not sv.restore_first_pass(folder)  # nothing saved any more
    assert not sv.written_by_stage(folder)


def test_renaming_reaches_the_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A naming run after the stage starts from the first pass, and the stage then starts from the new names."""
    run, args, config, files, _ = _season_run(tmp_path, monkeypatch)
    out = tmp_path / "work"
    args.force = None
    run.run_season_voices(files, out, args, config, ["vosk"])
    e1 = out / "e1"
    assert sv.written_by_stage(e1)
    assert sv.restore_first_pass(e1)  # what Episode.speakers does before naming
    cards = load_json(e1 / "speakers.json")
    cards[1]["name"] = "Carol White"  # the naming gives a new name
    save_json(e1 / "speakers.json", cards)
    run.run_season_voices(files, out, args, config, ["vosk"])
    assert load_json(sv.before_path(e1, "speakers"))[1]["name"] == "Carol White"  # not the old saved names


def test_translated_episode_named_again_gets_the_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    run, args, config, files, _ = _season_run(tmp_path, monkeypatch)
    out = tmp_path / "work"
    args.force = None
    e1 = out / "e1"
    (e1 / "subs.ru.json").write_text("[]", encoding="utf-8")
    episode = run.Episode(files[0], out, argparse.Namespace(season_glossary=None, force=None, until="mix"), {})
    named, translated = episode.stamp("speakers"), episode.stamp("translate")
    named.parent.mkdir(parents=True, exist_ok=True)
    named.touch()
    translated.touch()
    os.utime(named, (1_000, 1_000))  # named before the translation: stays as it is
    assert not episode.named_after_translation()
    run.run_season_voices(files, out, args, config, ["vosk"])
    assert not sv.before_path(e1, "words").exists()
    os.utime(named, (translated.stat().st_mtime + 10,) * 2)  # named again since («Кто это?»): translated again anyway
    assert episode.named_after_translation()
    run.run_season_voices(files, out, args, config, ["vosk"])
    assert sv.before_path(e1, "words").exists()
