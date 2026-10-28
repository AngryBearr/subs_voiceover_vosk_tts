"""Stress heard in takes: which words are listened to and how marks map to vowels."""

from __future__ import annotations

from voiceover.stress_listen import _plain_with_stress, fold, listen_words, words_in


def test_listen_words_from_a_stress_check_result() -> None:
    result = {"verdicts": [{"word": "Натали"}, {"word": "кит"}], "dictionary": {"алек": "+алек"},
              "sentences": {"Два голоса.": {"голоса": "г+олоса"}}}
    assert listen_words(result) == {"натали", "алек", "голоса"}  # one vowel: nothing to hear


def test_words_in_an_engine_text() -> None:
    words = {"натали", "голоса"}
    assert words_in("Натал+и, дв+а г+олоса за Натал+и.", words) == ["натали", "голоса", "натали"]
    assert fold("Всё+") == "все"


def test_marks_to_vowel_positions() -> None:
    plain, stressed = _plain_with_stress("Г+олоса  за  Натал+и")
    assert plain == "Голоса за Натали"
    assert sorted(plain[i] for i in stressed) == ["и", "о"]
    assert plain.index("о") in stressed
