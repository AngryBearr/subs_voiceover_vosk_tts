from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stt import speakers as sp

CAST: dict[str, Any] = {"cast": [
    {"name": "Jeff Probst", "aliases": ["Jeff"], "gender": "male", "role": "Host"},
    {"name": "Bryant Gumbel", "aliases": [], "gender": "male", "role": "Reunion host"},
    {"name": "Rudy Boesch", "aliases": [], "gender": "male", "role": "contestant"},
]}


def card(sid: str, role: str = "contestant", name: str | None = None, gender: str = "male",
         same_as: str | None = None, **extra: Any) -> dict[str, Any]:
    return {"id": sid, "role": role, "name": name, "gender": gender, "same_as": same_as, "cues": 10, **extra}


def by_id(speakers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in speakers}


def test_host_role_gets_cast_host() -> None:
    speakers = [card("S01", "host and narrator", gender="unknown"), card("S02", "narrator of the show"),
                card("S03", "contestant"), card("S04", "merged contestant voices, with occasional host dialogue",
                                                gender="unknown"),
                card("S05", "host", name="Someone Else")]
    sp.apply_cast_host(speakers, CAST)
    s = by_id(speakers)
    assert s["S01"]["name"] == "Jeff Probst" and s["S01"]["name_source"] == "cast_host"
    assert s["S01"]["gender"] == "male" and s["S01"]["gender_source"] == "cast"
    assert s["S02"]["name"] == "Jeff Probst"
    assert s["S03"]["name"] is None and s["S04"]["name"] is None
    assert s["S05"]["name"] == "Someone Else"  # a name already there stays


def test_no_host_without_one_host_in_cast() -> None:
    speakers = [card("S01", "host")]
    sp.apply_cast_host(speakers, {"cast": [{"name": "A", "role": "host"}, {"name": "B", "role": "host"}]})
    sp.apply_cast_host(speakers, None)
    assert speakers[0]["name"] is None


def test_protected_host_card_is_not_named() -> None:
    speakers = [card("S01", "host")]
    sp.apply_cast_host(speakers, CAST, frozenset({"S01"}))
    assert speakers[0]["name"] is None


def test_same_as_merges_into_named_card() -> None:
    speakers = [card("S01", name="Jeff Probst", name_source="subtitles", gender_source="cast"),
                card("S02", "host and narrator", same_as="S01", gender="unknown")]
    sp.apply_same_as(speakers)
    s02 = by_id(speakers)["S02"]
    assert s02["name"] == "Jeff Probst" and s02["name_source"] == "same_as" and s02["gender"] == "male"


def test_same_as_follows_chain_and_stops_at_cycle() -> None:
    chain = [card("S01", name="Rudy Boesch"), card("S02", same_as="S01"), card("S03", same_as="S02")]
    sp.apply_same_as(chain)
    assert by_id(chain)["S03"]["name"] == "Rudy Boesch" and by_id(chain)["S02"]["name"] == "Rudy Boesch"
    loop = [card("S01", same_as="S02"), card("S02", same_as="S01"), card("S03", same_as="S03")]
    sp.apply_same_as(loop)
    assert all(s["name"] is None for s in loop)


def test_same_as_ignores_mixed_groups_other_names_and_gender_conflicts() -> None:
    speakers = [card("S01", "merged contestant voices", gender="unknown", name="Voices"),
                card("S02", same_as="S01"),
                card("S03", name="Rudy Boesch"), card("S04", name="Other Person", same_as="S03"),
                card("S05", same_as="S03", gender="female"),
                card("S06", same_as="S99")]
    sp.apply_same_as(speakers)
    s = by_id(speakers)
    assert s["S02"]["name"] is None
    assert s["S04"]["name"] == "Other Person"
    assert s["S05"]["name"] is None
    assert s["S06"]["name"] is None
    mixed_source = [card("S01", name="Rudy Boesch"), card("S02", "mixed voices", gender="unknown", same_as="S01")]
    sp.apply_same_as(mixed_source)
    assert mixed_source[1]["name"] is None


def test_name_speakers_human_override_wins(tmp_path: Path) -> None:
    speakers = [card("S01", "host and narrator"), card("S02", "host and narrator", same_as="S01"),
                card("S03", "host and narrator", same_as="S01")]
    records = [{"index": 1, "speaker": "S01", "start": 0, "end": 1000, "text": ["x"]}]
    (tmp_path / "speakers.json").write_text(json.dumps(speakers), encoding="utf-8")
    (tmp_path / "subs.en.json").write_text(json.dumps(records), encoding="utf-8")
    (tmp_path / "speakers.llm.json").write_text(json.dumps({"cards": [
        {"id": "S01", "role": "host and narrator", "gender": "male", "name": None},
        {"id": "S02", "role": "host and narrator", "gender": "male", "name": None, "same_as": "S01"},
        {"id": "S03", "role": "host and narrator", "gender": "male", "name": None, "same_as": "S01"}]}),
        encoding="utf-8")
    (tmp_path / "speakers.override.json").write_text(json.dumps({"S01": {"name": "Narrator Nick"}}), encoding="utf-8")
    cast_path = tmp_path / "cast.json"
    cast_path.write_text(json.dumps(CAST), encoding="utf-8")
    result = by_id(sp.name_speakers(tmp_path, "m", use_llm=False, cast_path=cast_path, saved_llm=True))
    assert result["S01"]["name"] == "Narrator Nick"  # the person's name beats the cast host
    assert result["S02"]["name"] == "Jeff Probst"  # unprotected host cards follow the cast
