from __future__ import annotations

from typing import Any

from utils.text_normalizer import unstress_function_words, accent_cue, fill_dictionary_stress, ruaccent_pass


class _Accentizer:
    """Marks the first vowel of every word it is given and records what it got."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def process_all(self, text: str, skip_regex: Any = None) -> str:
        self.calls.append((text, skip_regex))
        import re

        vowel = "аеиоуыэюяАЕИОУЫЭЮЯ"
        return re.sub(rf"\b([^\W\d_{vowel}]*)([{vowel}])", r"\1+\2", text) if not skip_regex else text


def test_accent_cue_reads_the_pieces_between_ellipses_alone() -> None:
    fake = _Accentizer()
    assert accent_cue(fake, "Это... страшно!") == "+Это... стр+ашно!"
    assert [c[0] for c in fake.calls] == ["Это", " страшно!"]
    fake.calls.clear()
    assert accent_cue(fake, "…Так") == "…Т+ак"
    assert [c[0] for c in fake.calls] == ["Так"]


def test_ruaccent_pass_per_cue_sends_each_cue_alone() -> None:
    fake = _Accentizer()
    rows = [{"index": 1, "_normalized_text": "Это мы."}, {"index": 2, "_normalized_text": "Вот она."}]
    out = ruaccent_pass(rows, fake, batch_size=10, per_cue=True)  # type: ignore[arg-type]
    assert [r["text"] for r in out] == [["+Это м+ы."], ["В+от +она."]]
    assert fake.calls == [("Это мы.", None), ("Вот она.", None)]  # no joined list string
    fake.calls.clear()
    ruaccent_pass(rows, fake, batch_size=10)  # type: ignore[arg-type]
    assert len(fake.calls) == 1 and fake.calls[0][0].startswith("['")  # the default stays the joined batch


def test_fill_dictionary_stress() -> None:
    accents = {"это": "+это", "сразу": "ср+азу", "было": "б+ыло", "льюис": "ль+юис", "замок": "з+амок"}
    omographs = {"замок": ["з+амок", "зам+ок"]}
    assert fill_dictionary_stress("Это мы сразу, Льюис.", accents, omographs) == "+Это мы ср+азу, Ль+юис."
    assert fill_dictionary_stress("н+е было сез+она", accents, omographs) == "н+е было сез+она"  # stress moved
    assert fill_dictionary_stress("в+от замок", accents, omographs) == "в+от замок"  # a homograph
    assert fill_dictionary_stress("ср+азу всё", accents, omographs) == "ср+азу всё"  # marked / ё
    assert fill_dictionary_stress("Род-Айленд", accents, omographs) == "Род-Айленд"  # not in the dictionary


def test_user_stress_marks_win_and_leave_the_display() -> None:
    from utils.text_normalizer import keep_user_stress, strip_user_stress

    assert strip_user_stress("Это зам+ок, а не 2+2. +Эрик!") == "Это замок, а не 2+2. Эрик!"
    assert keep_user_stress("+Это з+амок на двер+и.", "Это зам+ок на двери.") == "+Это зам+ок на двер+и."
    assert keep_user_stress("Т+ы из +Айовы, д+а?", "Ты из Ай+овы, да?") == "Т+ы из Ай+овы, д+а?"
    # number expansion adds words: the marked one is still found in order
    assert keep_user_stress("В дв+е т+ысячи гор+оде зам+ок", "В 2000 город+е з+амок") == "В дв+е т+ысячи город+е з+амок"
    assert keep_user_stress("+Это всё", "Это всё") == "+Это всё"


def test_unstress_function_words() -> None:
    assert unstress_function_words("Он+и б+ыли н+а моём плот+у.") == "Он+и б+ыли на моём плот+у."
    assert unstress_function_words("Н+е хоч+у д+о конц+а, н+о б+ез неё.") == "Не хоч+у до конц+а, но без неё."
    assert unstress_function_words("н+е было, н+а берег, н+и за что, н+е за что") == "н+е было, н+а берег, н+и за что, н+е за что"
    assert unstress_function_words("Д+а, з+амок на д+ому") == "Д+а, з+амок на д+ому"  # other words stay
    assert unstress_function_words("Н+а") == "На"  # nothing after it


def test_unstress_hyphen_particles() -> None:
    assert unstress_function_words("Кт+о-ниб+удь и ког+о-либ+о.") == "Кт+о-нибудь и ког+о-либо."
    assert unstress_function_words("Кт+о-т+о, скаж+и-к+а, так-т+аки.") == "Кт+о-то, скаж+и-ка, так-таки."
    assert unstress_function_words("К+ое-чт+о и к+ой-как.") == "Кое-чт+о и кой-как."
    assert unstress_function_words("Т+о есть, к+а и ниб+удь") == "Т+о есть, к+а и ниб+удь"  # no hyphen: untouched
