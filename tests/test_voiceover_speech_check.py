"""Speech check: text normalisation, error rate, word differences and flags."""

from __future__ import annotations

from voiceover.speech_check import char_error_rate, check_clip, plain_words, word_diffs


def test_plain_words_drop_stress_punctuation_and_yo() -> None:
    assert plain_words("Ещё р+аз, на вост+оке США!") == ["еще", "раз", "на", "востоке", "сша"]
    assert plain_words("Вэл и Дж+ереми") == plain_words("велл и джерреми")


def test_char_error_rate_and_word_diffs() -> None:
    assert char_error_rate("abc", "abc") == 0.0
    assert abs(char_error_rate("abcd", "abxd") - 0.25) < 1e-9
    assert word_diffs(["на", "востоке", "сша"], ["на", "востоке", "ш"]) == [{"said": "сша", "heard": "ш"}]


def test_check_clip_flags_mispronounced_and_drawn_out() -> None:
    clip = {"unit_id": "u1", "start_ms": 0, "tempo": 1.0, "tts_text": "Мы на вост+оке СШ+А.", "path": "x.wav"}
    words = [("мы", 0.0, 0.1), ("на", 0.2, 0.3), ("востоке", 0.4, 1.1), ("ш", 1.3, 1.4)]
    result = check_clip(clip, "мы на востоке ш", words, 1.6)
    assert "pronunciation" in result["flags"] and result["diffs"] == [{"said": "сша", "heard": "ш"}]
    slow = check_clip({**clip, "tts_text": "Мы н+а-а-а вост+оке."}, "мы на востоке",
                      [("мы", 0.0, 0.1), ("на", 0.2, 1.5), ("востоке", 1.6, 2.3)], 2.4)
    assert "drawn_out" in slow["flags"] and slow["stretched_word"] == "на"


def test_spelled_abbreviation_matches_asr_spelling() -> None:
    clip = {"unit_id": "u2", "start_ms": 0, "tempo": 1.0, "tts_text": "Ведомства Сэ-Ш+э-А.", "path": "x.wav"}
    result = check_clip(clip, "ведомства сша", [("ведомства", 0.0, 0.6), ("сша", 0.7, 1.0)], 1.1)
    assert result["diffs"] == [] and "pronunciation" not in result["flags"]


def test_leading_extra_finds_a_leaked_reference_end() -> None:
    from voiceover.speech_check import leading_extra

    words = [("ай", 0.0, 0.2), ("долт", 0.2, 0.5), ("фор", 0.5, 0.7), ("хим", 0.7, 0.9),
             ("нам", 1.0, 1.2), ("нужно", 1.2, 1.5), ("четыре", 1.5, 1.9), ("голоса", 1.9, 2.3)]
    assert leading_extra("Нам н+ужно чет+ыре г+олоса.", words) == 1.0
    assert leading_extra("Нам н+ужно чет+ыре г+олоса.", words[4:]) is None
    assert leading_extra("Привет.", []) is None
