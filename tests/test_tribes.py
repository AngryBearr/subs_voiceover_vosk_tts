"""Tribe camps as a soft prior for who can speak (stt.tribes)."""

from __future__ import annotations

import numpy as np

from stt.boundaries import RelabelParams, _viterbi
from stt.tribes import camp_spans, card_tribe, cast_tribes, tribe_penalties
from stt.types import Word

CARDS = [
    {"start": 38.0, "end": 40.0, "kind": "name_card", "text": "B.B."},
    {"start": 300.0, "end": 303.0, "kind": "name_card", "text": "TAGI TRIBE"},
    {"start": 360.0, "end": 363.0, "kind": "name_card", "text": "PAGONG TRIBE"},
    {"start": 400.0, "end": 403.0, "kind": "name_card", "text": "MITCH", "descriptor": "CIVA TRIBE"},
    {"start": 700.0, "end": 703.0, "kind": "name_card", "text": "PAGONG", "descriptor": "DAY4"},
    {"start": 760.0, "end": 763.0, "kind": "name_card", "text": "TRIBAL COUNCIL"},
]
CAST = {"cast": [
    {"name": "Jeff Probst", "role": "Host"},
    {"name": "Sonja Christopher", "role": "Contestant; Tagi tribe; first voted out"},
    {"name": "Gretchen Cordy", "role": "Contestant; Pagong and Rattana tribes; seventh voted out"},
    {"name": "Mystery Guest", "role": "Contestant"},
]}


def test_tribes_come_from_the_roles() -> None:
    cast = {"cast": [*CAST["cast"], {"name": "Jan Gentry", "role": "Contestant; Chuay Gahn tribe; runner-up"},
                     {"name": "Clay Jordan", "role": "Contestant; Sook Jai, then Chuay Gahn tribes; voted out in episode 13"}]}
    assert cast_tribes(cast) == {"Sonja Christopher": {"tagi"}, "Gretchen Cordy": {"pagong", "rattana"},
                                 "Jan Gentry": {"chuay gahn"}, "Clay Jordan": {"sook jai", "chuay gahn"}}


def test_place_cards_name_one_tribe_despite_ocr_slips() -> None:
    tribes = {"tagi", "pagong", "kucha", "chuay gahn", "sook jai", "vula"}
    texts = ["TAGI TRIBE", "TAGITRIBE", "KUGHA TRIBE", "CHUAY GAHNTRIBE", "SOOK JAI TRIBE", "B.B.", "TRIBAL COUNCIL"]
    assert [card_tribe({"text": t}, tribes) for t in texts] == ["tagi", "tagi", "kucha", "chuay gahn", "sook jai", None, None]
    assert card_tribe({"text": "VULA", "descriptor": "DAY4"}, tribes) == "vula"
    assert card_tribe({"text": "MITCH", "descriptor": "CIVA TRIBE"}, tribes) is None  # a person's card


def test_camp_lasts_until_the_next_place_card_or_the_hold() -> None:
    spans = camp_spans(CARDS, {"tagi", "pagong", "rattana"}, hold_s=180.0)
    assert spans == [(300.0, 360.0, "tagi"), (360.0, 540.0, "pagong"), (700.0, 760.0, "pagong")]


def test_penalties_hit_only_people_of_other_tribes_at_a_camp() -> None:
    words = [Word(t, t + 0.3, "w") for t in (100.0, 310.0, 370.0, 710.0, 800.0)]
    out = tribe_penalties(words, CARDS, CAST, ["Jeff Probst", "Sonja Christopher", "Gretchen Cordy", "Mystery Guest"], 0.1)
    assert set(out) == {"Sonja Christopher", "Gretchen Cordy"}  # the host and people without a tribe: no prior
    assert list(out["Sonja Christopher"]) == [0.0, 0.0, -0.1, -0.1, 0.0]
    assert list(out["Gretchen Cordy"]) == [0.0, -0.1, 0.0, 0.0, 0.0]
    assert tribe_penalties(words, CARDS, CAST, ["Sonja Christopher"], 0.0) == {}


def test_bias_moves_a_close_call() -> None:
    words = [Word(i * 0.4, i * 0.4 + 0.3, f"w{i}") for i in range(4)]
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    emb = np.stack([0.52 * a + 0.48 * b] * 4)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    params = RelabelParams(switch_penalty=0.3)
    assert _viterbi(words, [None] * 4, emb, {"A": a, "B": b}, params) == ["A"] * 4
    assert _viterbi(words, [None] * 4, emb, {"A": a, "B": b}, params, bias={"A": np.full(4, -0.1)}) == ["B"] * 4
