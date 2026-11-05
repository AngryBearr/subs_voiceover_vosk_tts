from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stt import voice_bank
from stt.speakers import build_prompt
from stt.voice_bank import VoiceBank, attach_matches, reference_samples, resolve, same_person


def _vec(seed: int, noise: float = 0.0, base: int | None = None) -> list[float]:
    rng = np.random.default_rng(seed)
    v = np.random.default_rng(base if base is not None else seed).standard_normal(256)
    return list(v + noise * rng.standard_normal(256))


def _speaker(sid: str, name: str | None, emb: list[float], **extra: Any) -> dict[str, Any]:
    return {"id": sid, "name": name, "label_ru": name and name.split()[0], "gender": "female",
            "speech_s": 60.0, "llm_confidence": 0.9, "embedding": emb, "cues": 10, **extra}


def _cue(start: float, end: float, speaker: str, text: str = "words", **extra: Any) -> dict[str, Any]:
    return {"start": int(start * 1000), "end": int(end * 1000), "speaker": speaker, "text": [text], **extra}


def test_same_person_spellings() -> None:
    assert same_person("Jon Misch", "John Misch") and same_person("Kelley Wentworth", "Kelly Wentworth")
    assert same_person("Baylor", "Baylor Wilson")
    assert not same_person("Nadiya Anderson", "Natalie Anderson")  # twins
    assert not same_person("Keith Nale", "Wes Nale")


def test_reference_samples_take_solo_stretches() -> None:
    records = [_cue(0, 3, "S01"), _cue(3.2, 7, "S01"), _cue(7.1, 9, "S01"),  # 9 s solo
               _cue(10, 14, "S01", overlap=True), _cue(20, 23, "S01"), _cue(30, 31, "S02")]
    samples = reference_samples("S01", records)
    assert samples == [{"start": 0.0, "end": 9.0, "text": "words words words", "cues": 3}]


def test_bank_names_an_unnamed_cluster_and_flags_a_contradiction(tmp_path: Path) -> None:
    bank = VoiceBank(tmp_path / "s.voices.json")
    ep1 = [_speaker("S01", "Val Collins", _vec(1)), _speaker("S02", "Julie McGee", _vec(2)),
           _speaker("S03", "Reed Kelly", _vec(3), speech_s=5.0)]  # too little speech for a print
    assert bank.add_episode("ep1", ep1, []) == 2
    bank.save()
    bank = VoiceBank.load(tmp_path / "s.voices.json")

    ep2 = [_speaker("S01", None, _vec(11, 0.3, base=1)),  # Val's voice, unnamed by the LLM
           _speaker("S02", "Missy Payne", _vec(12, 0.3, base=2)),  # Julie's voice, named otherwise
           _speaker("S03", None, _vec(13))]  # a new voice
    attach_matches(ep2, bank, "ep2")
    assert ep2[0]["voice_bank"][0]["name"] == "Val Collins" and "voice_bank" not in ep2[2]
    assert "Val Collins (0.9" in build_prompt(ep2, [])
    notes = resolve(ep2)
    assert ep2[0]["name"] == "Val Collins" and ep2[0]["name_source"] == "voice_bank"
    assert ep2[1]["name"] == "Missy Payne" and ep2[1]["voice_bank_conflict"]["bank_name"] == "Julie McGee"
    assert ep2[2]["name"] is None and len(notes) == 2

    assert bank.add_episode("ep2", ep2, []) == 2
    assert [c["bank_name"] for c in bank.data["conflicts"]] == ["Julie McGee"]
    assert bank.add_episode("ep2", ep2, []) == 2  # a re-run replaces the episode's prints
    assert len(bank.prints) == 4 and len(bank.data["conflicts"]) == 1
    assert [p["name"] for p in bank.people()][:1] == ["Val Collins"]


def test_unclear_match_names_nobody(tmp_path: Path) -> None:
    bank = VoiceBank(tmp_path / "s.voices.json")
    twin = _vec(5)
    bank.add_episode("ep1", [_speaker("S01", "Nadiya Anderson", twin), _speaker("S02", "Natalie Anderson", twin)], [])
    ep2 = [_speaker("S01", None, _vec(6, 0.2, base=5))]
    attach_matches(ep2, bank, "ep2")
    assert len(ep2[0]["voice_bank"]) == 2
    assert resolve(ep2) == [] and ep2[0]["name"] is None
    assert voice_bank.MARGIN > 0


def test_reference_transcript_from_aligned_words(tmp_path: Path) -> None:
    import json

    words = [{"start": 0.9, "end": 1.2, "text": "Before"}, {"start": 1.0, "end": 1.4, "text": "I'm"},
             {"start": 1.5, "end": 2.0, "text": "dating"}, {"start": 7.9, "end": 8.4, "text": "after"}]
    (tmp_path / "words.json").write_text(json.dumps(words), encoding="utf-8")
    assert voice_bank._words(tmp_path, 1.1, 8.0) == "I'm dating"
    assert voice_bank._words(tmp_path / "missing", 0, 1) is None
    assert voice_bank._safe_name("Jon Misch / host") == "Jon_Misch_host"


def test_recheck_renames_a_voice_other_episodes_know_by_another_name(tmp_path: Path) -> None:
    import json

    bank = VoiceBank(tmp_path / "s.voices.json")
    for ep in ("ep2", "ep3", "ep4"):
        (tmp_path / ep).mkdir()
    (tmp_path / "ep1").mkdir()
    bank.add_episode("ep1", [_speaker("S06", "Jaclyn Schultz", _vec(21, 0.2, base=7))], [])
    for i, ep in enumerate(("ep2", "ep3", "ep4")):
        bank.add_episode(ep, [_speaker("S02", "Baylor Wilson", _vec(30 + i, 0.2, base=7))], [])
    suggestions = voice_bank.recheck(bank)
    assert [(s["episode"], s["suggested"], s["support"]) for s in suggestions] == [("ep1", "Baylor Wilson", 3)]
    (tmp_path / "ep1" / "speakers.override.json").write_text(json.dumps({"S01": {"name": "Jeff"}}))
    voice_bank.write_overrides(bank, suggestions)
    overrides = json.loads((tmp_path / "ep1" / "speakers.override.json").read_text())
    assert overrides["S01"] == {"name": "Jeff"} and overrides["S06"]["name"] == "Baylor Wilson"


def test_recheck_needs_two_other_episodes(tmp_path: Path) -> None:
    bank = VoiceBank(tmp_path / "s.voices.json")
    bank.add_episode("ep1", [_speaker("S06", "Jaclyn Schultz", _vec(21, 0.2, base=7))], [])
    bank.add_episode("ep2", [_speaker("S02", "Baylor Wilson", _vec(30, 0.2, base=7))], [])
    assert voice_bank.recheck(bank) == []  # one episode against one: no majority


def test_recheck_leaves_relatives_alone(tmp_path: Path) -> None:
    bank = VoiceBank(tmp_path / "s.voices.json")
    bank.add_episode("ep1", [_speaker("S02", "Nadiya Anderson", _vec(41, 0.2, base=9))], [])
    for i, ep in enumerate(("ep2", "ep3", "ep4")):
        bank.add_episode(ep, [_speaker("S05", "Natalie Anderson", _vec(50 + i, 0.2, base=9))], [])
    assert voice_bank.relatives("Nadiya Anderson", "Natalie Anderson")
    assert not voice_bank.relatives("Jon Misch", "John Misch")
    assert voice_bank.recheck(bank) == []


def test_reject_reference_remembers_the_clip(tmp_path: Path) -> None:
    import json

    index = [{"name": "Julie McGee", "episode": "ep1", "start": 1976.6, "end": 1985.1}]
    (tmp_path / "references.json").write_text(json.dumps(index))
    assert voice_bank.reject_reference(tmp_path, "Julie McGee")["start"] == 1976.6
    assert voice_bank.reject_reference(tmp_path, "Nobody") is None
    assert voice_bank.load_rejected(tmp_path) == [{"name": "Julie McGee", "episode": "ep1", "start": 1976.6}]


def test_unchanged_clip_index_keeps_its_time(tmp_path: Path) -> None:
    # a rewritten but equal references.json made every F5 episode look stale and voiced again
    import os

    index = tmp_path / "references.json"
    assert voice_bank.write_if_changed(index, "[]\n")
    os.utime(index, (1000, 1000))
    assert not voice_bank.write_if_changed(index, "[]\n")
    assert index.stat().st_mtime == 1000
    assert voice_bank.write_if_changed(index, '[{"name": "Jeff"}]\n')


def _episode(tmp_path: Path, renumbered: bool) -> Path:
    import json

    folder = tmp_path / "ep1"
    folder.mkdir()
    first = [{"id": "S01", "name": "Aaliyah Puglia", "diarizer_label": "3"},
             {"id": "S02", "name": "Devin Way", "diarizer_label": "10"}]
    (folder / "diarization.test.json").write_text(json.dumps({"turns": [
        {"speaker": "3", "start": 0.0, "end": 9.0}, {"speaker": "10", "start": 9.0, "end": 18.0}]}))
    cues = [_cue(0, 9, "S01", "aaliyah"), _cue(9, 18, "S02", "devin")]
    if renumbered:  # season voices swapped the ids and keep the first pass aside
        (folder / "speakers.before_voices.json").write_text(json.dumps(first))
        (folder / "subs.en.before_voices.json").write_text(json.dumps(cues))
        now = [{"id": "S01", "name": "Devin Way", "diarizer_label": "V:Devin", "diarizer_labels": ["10"]},
               {"id": "S02", "name": "Aaliyah Puglia", "diarizer_label": "V:Aaliyah", "diarizer_labels": ["3"]}]
        cues = [_cue(0, 9, "S02", "aaliyah"), _cue(9, 18, "S01", "devin")]
    else:
        now = first
    (folder / "speakers.json").write_text(json.dumps(now))
    (folder / "subs.en.json").write_text(json.dumps(cues))
    return folder


def test_person_is_found_by_name_after_season_voices(tmp_path: Path) -> None:
    import json

    folder = _episode(tmp_path, renumbered=True)
    # the print still says S01 (first pass), but S01 is Devin now
    ids, subs, speakers = voice_bank.locate_person(folder, "Aaliyah Puglia", "S01")
    assert (ids, subs, speakers) == ({"S02"}, "subs.en.json", "speakers.json")
    records = json.loads((folder / subs).read_text())
    assert reference_samples(ids, records)[0]["text"] == "aaliyah"
    assert voice_bank.other_voice_seconds(folder, ids, 0.0, 9.0, speakers) == 0.0
    assert voice_bank.other_voice_seconds(folder, ids, 5.0, 12.0, speakers) == 3.0


def test_person_without_a_current_card_uses_the_first_pass(tmp_path: Path) -> None:
    import json

    folder = _episode(tmp_path, renumbered=True)
    ids, subs, speakers = voice_bank.locate_person(folder, "Kristin Flickinger", "S02")
    assert (ids, subs, speakers) == ({"S02"}, "subs.en.before_voices.json", "speakers.before_voices.json")
    records = json.loads((folder / subs).read_text())
    assert reference_samples(ids, records)[0]["text"] == "devin"
    assert voice_bank.other_voice_seconds(folder, ids, 9.0, 18.0, speakers) == 0.0


def test_unrenumbered_episode_keeps_the_print_id(tmp_path: Path) -> None:
    folder = _episode(tmp_path, renumbered=False)
    assert voice_bank.locate_person(folder, "Aaliyah Puglia", "S01")[0] == {"S01"}
    assert voice_bank.locate_person(folder, "Nobody Else", "S02") == ({"S02"}, "subs.en.json", "speakers.json")


def _final_episode(tmp_path: Path) -> Path:
    """An episode after the season voices stage: S01 Richard (a season person with a print), S02 Sean (named by the
    subtitles, no print yet, 30 s of cues), S03 Dirk (subtitles, 8 s), S04 an unnamed cluster, S05 an LLM-only
    name with low confidence."""
    import json

    import soundfile as sf

    folder = tmp_path / "S01E01"
    folder.mkdir()
    cards = [
        _speaker("S01", "Richard Hatch", _vec(1), season_voice=True, name_source="llm", llm_confidence=0.2),
        {"id": "S02", "name": "Sean Kenniff", "name_source": "subtitles", "gender": "male", "speech_s": None,
         "embedding": None, "cues": 6, "season_voice": False},
        {"id": "S03", "name": "Dirk Been", "name_source": "subtitles", "gender": "male", "speech_s": None,
         "embedding": None, "cues": 2},
        _speaker("S04", None, _vec(4)),
        _speaker("S05", "Maybe Person", _vec(5), name_source="llm", llm_confidence=0.3),
    ]
    cues = [_cue(2 * k, 2 * k + 1.9, "S01") for k in range(15)]  # 28.5 s
    cues += [_cue(40 + 6 * k, 45 + 6 * k, "S02") for k in range(6)]  # 30 s
    cues += [_cue(80 + 4 * k, 84 + 4 * k, "S03") for k in range(2)]  # 8 s
    (folder / "speakers.json").write_text(json.dumps(cards))
    (folder / "subs.en.json").write_text(json.dumps(cues))
    sf.write(str(folder / "audio.wav"), np.zeros(16_000 * 100, dtype=np.float32), 16_000)
    return folder


def test_prints_for_people_named_by_subtitles_or_season_voices(tmp_path: Path) -> None:
    folder = _final_episode(tmp_path)
    calls: list[int] = []

    def embed(audio: np.ndarray, rate: int) -> np.ndarray:
        calls.append(len(audio))
        return np.ones(8) / np.sqrt(8)

    bank_path = tmp_path / "Season 1.voices.json"
    assert voice_bank.refresh_episode(bank_path, folder, embed=embed) == 2
    prints = {p["name"]: p for p in VoiceBank.load(bank_path).prints}
    assert set(prints) == {"Richard Hatch", "Sean Kenniff"}  # not Dirk (8 s), nobody unnamed, no unsure LLM name
    assert prints["Sean Kenniff"]["speaker"] == "S02" and prints["Sean Kenniff"]["source"] == "subtitles"
    assert prints["Sean Kenniff"]["speech_s"] == 30.0 and len(prints["Sean Kenniff"]["embedding"]) == 8
    assert prints["Richard Hatch"]["embedding"] == [float(x) for x in _vec(1)]  # the card's own print is kept
    assert len(calls) == 6  # one embedding per long cue of Sean only
    # speakers.json is not touched: the season voices stage compares it with what it would write
    import json
    assert json.loads((folder / "speakers.json").read_text())[1]["embedding"] is None


def test_refresh_twice_does_not_double_prints_nor_embed_again(tmp_path: Path) -> None:
    folder = _final_episode(tmp_path)
    calls: list[int] = []

    def embed(audio: np.ndarray, rate: int) -> np.ndarray:
        calls.append(1)
        return np.ones(8) / np.sqrt(8)

    bank_path = tmp_path / "Season 1.voices.json"
    bank = VoiceBank(bank_path)  # the first pass left prints with ids of the first numbering
    bank.add_episode(folder.name, [_speaker("S09", "Richard Hatch", _vec(1))],
                     [_cue(0, 30, "S09")])
    bank.add_episode("other", [_speaker("S01", "Rudy Boesch", _vec(7))], [])
    bank.save()
    voice_bank.refresh_episode(bank_path, folder, embed=embed)
    first = calls[:]
    voice_bank.refresh_episode(bank_path, folder, embed=embed)
    assert calls == first  # the earlier print of Sean (same id, name and cues) is reused
    prints = VoiceBank.load(bank_path).prints
    mine = [(p["speaker"], p["name"]) for p in prints if p["episode"] == folder.name]
    assert sorted(mine) == [("S01", "Richard Hatch"), ("S02", "Sean Kenniff")]  # no S09, no duplicates
    assert [p["name"] for p in prints if p["episode"] == "other"] == ["Rudy Boesch"]  # other episodes untouched


def test_add_episode_keeps_the_llm_confidence_rule_for_llm_only_names() -> None:
    bank = VoiceBank(Path("unused.json"))
    cues = [_cue(0, 40, "S01")]
    unsure = _speaker("S01", "Ann Lee", _vec(1), name_source="llm", llm_confidence=0.5)
    assert bank.add_episode("e", [unsure], cues) == 0
    assert bank.add_episode("e", [{**unsure, "llm_confidence": 0.9}], cues) == 1
    assert bank.add_episode("e", [{**unsure, "name_source": "subtitles"}], cues) == 1
