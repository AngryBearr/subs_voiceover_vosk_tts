"""Stress check: suspects, answer validation, voting and sentence fixes."""

from __future__ import annotations

from types import SimpleNamespace

from voiceover.engine import cache_key
from voiceover.stress_check import Suspect, apply_sentence_fixes, collect, valid_stress, vote


def _accentizer() -> SimpleNamespace:
    return SimpleNamespace(custom_dict={"вавау": "вав+ау"}, omographs={"голоса": ["г+олоса", "голос+а"]},
                           accents={"голоса": "голос+а", "пляж": "пл+яж", "вавау": "вав+ау"})


def test_collect_takes_guessed_words_and_homographs_only() -> None:
    found = collect(["Два голос+а за Над+ию на пл+яже Вав+ау.", "Кое-ч+ем обязан."], _accentizer())
    assert {w: s.source for w, s in found.items()} == {"голоса": "homograph", "надию": "guess", "пляже": "guess",
                                                      "обязан": "guess"}
    assert found["голоса"].variants == ["г+олоса", "голос+а"]


def test_valid_stress_and_vote() -> None:
    suspect = Suspect("голоса", "голос+а", "homograph", ["г+олоса", "голос+а"])
    assert valid_stress(suspect, "г+олоса") and not valid_stress(suspect, "голо+са") and not valid_stress(suspect, "гол+оса")
    assert vote(suspect, ["г+олоса", "г+олоса", None]) == "г+олоса"
    assert vote(suspect, ["г+олоса", "голос+а", None]) is None


def test_sentence_fixes_touch_only_their_sentence() -> None:
    fixes = {"Два голоса за Надию.": {"голоса": "г+олоса"}}
    assert apply_sentence_fixes("Два голос+а за Над+ию.", fixes, "Два голоса за Надию.") == "Два г+олоса за Над+ию."
    assert apply_sentence_fixes("Нужны голос+а.", fixes, "Нужны голоса.") == "Нужны голос+а."


def test_cache_key_keeps_default_voice_keys() -> None:
    assert cache_key("Прив+ет.") == cache_key("Прив+ет.", None) != cache_key("Прив+ет.", 49)
