"""Which tribe's camp a scene is at, and which tribes a cast member belongs to: a soft prior for who can speak.

Before the merge a scene at one tribe's camp has only that tribe's people in it. Survivor shows where a scene is with a
card ("TAGI TRIBE", "VULA (DAY 4)", a lower third "PAGONG TRIBE"); the season's cast (``<season>.cast.json``, its
``role`` text: "Contestant; Pagong and Rattana tribes; seventh voted out") says which tribes there are and who was in
which. A camp
lasts from its card until another place card or ``HOLD_S``. ``stt.season_voices`` then makes the people of other
tribes a little less likely there (``tribe_penalties``); nobody is ruled out: challenges, Tribal Council and the
"previously on" recap mix the tribes.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

import numpy as np

from stt.types import Word

HOLD_S = 180.0  # a camp scene lasts at most this long after its card (the show cuts on without a new card)


def _squash(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def cast_tribes(cast: dict[str, Any]) -> dict[str, set[str]]:
    """Cast name -> its tribes from the role text: "Contestant; Pagong and Rattana tribes; ..." -> {pagong, rattana},
    "Chuay Gahn tribe" -> {chuay gahn}. People whose role names no tribe are left out."""
    out: dict[str, set[str]] = {}
    for person in cast.get("cast") or []:
        if not isinstance(person, dict) or not person.get("name"):
            continue
        found: set[str] = set()
        for clause in re.split(r"[;()]", str(person.get("role") or "")):
            match = re.search(r"([A-Za-z][A-Za-z' ,-]*?)\s+tribes?\b", clause, re.I)
            if not match:
                continue
            for name in re.split(r"\s*(?:,|\band\b|\bthen\b|\blater\b|\bto\b)\s*", match.group(1), flags=re.I):
                name = re.sub(r"^(?:the|original|starting|merged|swapped)\s+", "", name.strip(), flags=re.I).lower()
                if name and name not in ("contestant", "member"):
                    found.add(name)
        if found:
            out[str(person["name"])] = found
    return out


def card_tribe(card: dict[str, Any], tribes: set[str]) -> str | None:
    """The one tribe a place card names ("TAGI TRIBE", "CHUAY GAHNTRIBE", "KUGHA TRIBE" for Kucha, "VULA" + "DAY4");
    None for a person's card ("MITCH" + "CIVA TRIBE": the person's tribe, not a scene), for two tribes or for none.
    The text is compared without spaces and with room for an OCR slip (``tribes``: names from ``cast_tribes``)."""
    text = _squash(re.sub(r"\b(?:tribe|days?|nights?)\b|\d+", " ", str(card.get("text") or ""), flags=re.I))
    text = re.sub(r"tribe$", "", text)
    if len(text) < 3:
        return None
    hits = [t for t in tribes if _squash(t) == text
            or (len(text) >= 4 and difflib.SequenceMatcher(None, _squash(t), text).ratio() >= 0.8)]
    return hits[0] if len(hits) == 1 else None


def camp_spans(cards: list[dict[str, Any]], tribes: set[str], hold_s: float = HOLD_S) -> list[tuple[float, float, str]]:
    """(start, end, tribe) of the camp scenes: from a tribe's place card to the next place card or ``hold_s`` later."""
    from stt.assemble import place_card
    from stt.onscreen import _tribe_words

    seen = _tribe_words([{"text": f"{c.get('text') or ''} {c.get('descriptor') or ''}"} for c in cards])
    places = sorted((c for c in cards if place_card(c, seen) or card_tribe(c, tribes)), key=lambda c: c["start"])
    spans: list[tuple[float, float, str]] = []
    for k, card in enumerate(places):
        tribe = card_tribe(card, tribes)
        if tribe is None:
            continue
        end = card["start"] + hold_s
        later = [c["start"] for c in places[k + 1:] if c["start"] > card["end"] and card_tribe(c, tribes) != tribe]
        if later:
            end = min(end, later[0])
        if spans and spans[-1][2] == tribe and card["start"] <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end), tribe)  # the same camp shown again: one scene
        else:
            spans.append((float(card["start"]), float(end), tribe))
    return spans


def tribe_penalties(words: list[Word], cards: list[dict[str, Any]], cast: dict[str, Any], people: list[str],
                    penalty: float, hold_s: float = HOLD_S) -> dict[str, np.ndarray]:
    """Person -> per-word score change: ``-penalty`` on the words of a camp scene of a tribe the person was never in.
    ``people``: the cast names of the states (``stt.season_voices`` persons). Empty when the episode shows no camp."""
    if penalty <= 0 or not cast:
        return {}
    of = cast_tribes(cast)
    tribes = set().union(*of.values()) if of else set()
    spans = camp_spans(cards, tribes, hold_s)
    if not spans:
        return {}
    middles = np.array([(w.start + w.end) / 2 for w in words])
    camp: list[str | None] = [None] * len(words)
    for start, end, tribe in spans:
        for i in np.nonzero((middles >= start) & (middles < end))[0]:
            camp[int(i)] = tribe
    out: dict[str, np.ndarray] = {}
    for name in people:
        own = of.get(name)
        if not own:
            continue
        vec = np.array([-penalty if t is not None and t not in own else 0.0 for t in camp])
        if vec.any():
            out[name] = vec
    return out
