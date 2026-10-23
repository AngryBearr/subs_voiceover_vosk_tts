from __future__ import annotations

import pytest

from voiceover.normalize import expand, guard, space_after_ellipsis


def test_expand_numbers_and_abbreviations() -> None:
    assert expand("Это 15-й день на острове.").text == "Это пятнадцатый день на острове."
    assert expand("Он выиграл $500.").text == "Он выиграл пятьсот долларов."
    assert expand("т.е. это так").text == "то есть это так"
    assert expand("5 кг риса").text == "пять килограммов риса"


def test_expand_spells_letter_abbreviations_with_vowels() -> None:
    assert expand("Армия США и ФБР.").text == "Армия Сэ-Шэ-А и эф бэ эр."
    assert expand("Сшалось.").text == "Сшалось."


def test_expand_leaves_plain_russian_alone() -> None:
    for text in ["Я не нашла стену - в этом-то и проблема.", "Дженна-Луиза и Сэм у клетки.", "О-о-ох Сью.",
                 "Это Ро.", "Кэт, что ты по этому поводу чувствуешь?", "Э-э... я подслушал... ...разговор..."]:
        result = expand(text)
        assert result.text == text
        assert not result.changed
        assert result.uncertain == ()


def test_expand_reports_uncertain_spans_but_still_expands() -> None:
    result = expand("Нас осталось 10, 1 иммунитет.")
    assert result.text == "Нас осталось десять, один иммунитет."
    assert result.changed
    assert result.uncertain  # bare numbers: grammatical case undetermined


def test_guard_catches_changed_or_dropped_words() -> None:
    with pytest.raises(ValueError):
        guard("Теперь Сэм у клетки.", "Теперь сам у клетки.")
    with pytest.raises(ValueError):
        guard("у тебя нет времени", "у меня нет времени")
    with pytest.raises(ValueError):
        guard("Как НЕ отправлять кого-либо домой", "Как э отправлять кого-либо домой")
    with pytest.raises(ValueError):
        guard("Дженна-Луиза сейчас", "дженнаЭлуиза сейчас")


def test_guard_allows_yo_restoration_expansion_and_abbreviations() -> None:
    guard("Вот почему Эван ушел домой.", "Вот почему Эван ушёл домой.")
    guard("У нас 3 дня.", "У нас три дня.")
    guard("т.е. это так", "то есть это так")
    guard("5 кг риса", "пять килограммов риса")


def test_space_after_glued_ellipsis() -> None:
    assert space_after_ellipsis("хотят…хотят меня") == "хотят… хотят меня"
    assert space_after_ellipsis("Но... но племя") == "Но... но племя"
    assert space_after_ellipsis("...разговор...") == "... разговор..."


def test_keep_yo_spelling_of_homographs() -> None:
    from voiceover.normalize import keep_yo_spelling

    yo = {"все": "всё", "узнаем": "узнаём"}
    accents = {"все": "вс+е", "всё": "вс+ё"}
    accented = "Вс+ё хвал+или. Гд+е вс+ё? Всё хорош+о, ещё раз."
    assert keep_yo_spelling("Все хвалили. Где все? Всё хорошо, ещё раз.", accented, yo, accents) == \
        "Вс+е хвал+или. Гд+е вс+е? Всё хорош+о, ещё раз."
    assert keep_yo_spelling("Мы узнаем", "Мы узна+ём", yo, accents) == "Мы узна+ем"
