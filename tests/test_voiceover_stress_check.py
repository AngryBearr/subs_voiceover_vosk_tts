"""Stress check: suspects, answer validation, voting and sentence fixes."""

from __future__ import annotations

from types import SimpleNamespace

from voiceover.engine import cache_key
from voiceover.stress_check import (Suspect, apply_sentence_fixes, collect, disagreements, glossary_names,
                                    name_of, valid_stress, vote)


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
    assert valid_stress(suspect, "г+олоса") and not valid_stress(suspect, "голо+са")
    assert valid_stress(suspect, "гол+оса")  # outside the dictionary's variants is allowed (понял+а)
    assert not valid_stress(suspect, "г+олос+а") and not valid_stress(suspect, "г+олосу")
    assert vote(suspect, ["г+олоса", "г+олоса", None]) == "г+олоса"
    assert vote(suspect, ["г+олоса", "голос+а", None]) is None


def test_sentence_fixes_touch_only_their_sentence() -> None:
    fixes = {"Два голоса за Надию.": {"голоса": "г+олоса"}}
    assert apply_sentence_fixes("Два голос+а за Над+ию.", fixes, "Два голоса за Надию.") == "Два г+олоса за Над+ию."
    assert apply_sentence_fixes("Нужны голос+а.", fixes, "Нужны голоса.") == "Нужны голос+а."


def test_cache_key_keeps_default_voice_keys() -> None:
    assert cache_key("Прив+ет.") == cache_key("Прив+ет.", None) != cache_key("Прив+ет.", 49)


def test_names_of_the_show_are_checked_in_every_case_form() -> None:
    names = glossary_names([{"en": "Keith Nale", "ru": "Кит Нейл", "type": "person"},
                            {"en": "Jaclyn Schultz", "ru": "Жаклин Шульц", "type": "person"},
                            {"en": "Nadiya Anderson", "ru": "Надия Андерсон", "type": "person"},
                            {"en": "Hunahpu", "ru": "Хунапу", "type": "tribe"}])
    assert names == {"кит": "Keith", "нейл": "Nale", "жаклин": "Jaclyn", "шульц": "Schultz", "надия": "Nadiya",
                     "андерсон": "Anderson"}
    assert name_of("Кита", names) == name_of("Китом", names) == "Keith"
    assert name_of("Надию", names) == name_of("Надия", names) == "Nadiya"
    assert name_of("кита", names) is None and name_of("Китай", names) is None  # a whale, a country
    accentizer = SimpleNamespace(custom_dict={}, omographs={}, accents={"кита": "кит+а", "жаклин": "жакл+ин",
                                                                        "видела": "в+идела", "план": "пл+ан"})
    found = collect(["Ж+аклин видела кит+а.", "У Кит+а и Жакл+ин план."], accentizer, names=names)
    assert {w: (s.source, s.name_en) for w, s in found.items()} == {
        "жаклин": ("name", "Jaclyn"), "кита": ("name", "Keith")}
    assert found["кита"].contexts == ["У Кита и Жаклин план."]  # the whale's sentence is not a name context


def test_second_placer_disagreements_mark_their_sentences() -> None:
    assert disagreements("Посл+е всег+о.", "П+осле всег+о.") == {0}
    assert disagreements("Всё равн+о.", "Вс+ё р+авно.") == {1}  # ё words and one-vowel words are left out
    assert disagreements("Посл+е всег+о.", "П+осле.") == set()  # not aligned: nothing
    accentizer = SimpleNamespace(custom_dict={}, omographs={"после": ["п+осле", "посл+е"]},
                                 accents={"после": "посл+е", "всего": "всег+о", "пора": "п+ора", "идти": "идт+и"})
    found = collect(["Посл+е всег+о.", "Посл+е всег+о.", "П+ора идт+и."], accentizer,
                    others=["П+осле всег+о.", "Посл+е всег+о.", "Пор+а идт+и."])
    assert found["после"].source == "homograph" and found["после"].disputed == ["После всего."]
    assert found["пора"].source == "disputed" and found["пора"].contexts == ["Пора идти."]
    assert "всего" not in found and "идти" not in found


def test_season_words_are_decided_once(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from voiceover import stress_check

    assert stress_check.decided(["н+атали", "н+атали", None]) == "н+атали"
    assert stress_check.decided(["н+атали", "натал+и", None]) is None  # no majority: asked again next episode

    class Engine:
        def prepare(self, items):  # noqa: ANN001
            return [SimpleNamespace(tts_text=t) for t in ("Натал+и и Ал+ек.", "Натал+и тут.")]

    engine = Engine()
    engine._accentizer = SimpleNamespace(custom_dict={}, omographs={}, accents={"натали": "натал+и", "алек": "ал+ек"})
    monkeypatch.setattr(stress_check, "build_engine", lambda config: engine)
    asked: list[list[str]] = []

    def ask(pairs, client):  # noqa: ANN001
        asked.append([s.word for s, _ in pairs])
        return [{"натали": "н+атали", "алек": "+алек"}[s.word] for s, _ in pairs]

    monkeypatch.setattr(stress_check, "ask", ask)
    monkeypatch.setattr(stress_check, "_client", lambda *a, **k: SimpleNamespace(stats=lambda: {}))
    names = {"натали": "Natalie", "алек": "Alec"}
    lines = [{"text": "Натали и Алек."}, {"text": "Натали тут."}]
    season = {"натали": {"stressed": "н+атали", "source": "name"}}  # decided in an earlier episode
    result = stress_check.check(lines, {}, {}, ["a", "b", "c"], names=names, second=False, season=season)
    assert asked == [["алек"]] * 3  # only the new name goes to the three models
    assert result["dictionary"] == {"алек": "+алек", "натали": "н+атали"}
    assert set(result["season"]["decided"]) == {"алек"}
    path = Path(tmp_path) / "S.stress.json"
    assert stress_check.save_season(path, {"натали": {"stressed": "натал+и"}, **result["season"]["decided"]}, "E02") == 2
    assert stress_check.save_season(path, {"натали": {"stressed": "н+атали"}}, "E03") == 0  # never replaced
    assert stress_check.load_season(path)["натали"]["stressed"] == "натал+и"
