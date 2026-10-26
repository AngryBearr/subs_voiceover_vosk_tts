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
