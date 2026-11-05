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


# --- proofreading -------------------------------------------------------------------------------

import json  # noqa: E402

from voiceover import stress_check  # noqa: E402
from voiceover.stress_check import Proposal, proofread, validate_proposal  # noqa: E402

TEXTS = ["Сам+оа тут.", "Это Само+а и Само+а.", "Д+ома, в моём царств+е.", "Подс+адим к+ого-либо."]


def test_proposal_validation_drops_the_unusable() -> None:
    ok = {"id": 1, "word": "Само+а", "stressed": "Сам+оа"}
    assert validate_proposal(ok, TEXTS) == Proposal(1, "само+а", "сам+оа")
    assert validate_proposal({**ok, "id": 9}, TEXTS) is None  # unknown id
    assert validate_proposal({**ok, "id": True}, TEXTS) is None
    assert validate_proposal({**ok, "id": 2}, TEXTS) is None  # word not in that line
    assert validate_proposal({**ok, "stressed": "Сам+оо"}, TEXTS) is None  # other letters
    assert validate_proposal({**ok, "stressed": "С+ам+оа"}, TEXTS) is None  # two "+"
    assert validate_proposal({**ok, "stressed": "Самоа"}, TEXTS) is None  # no "+"
    assert validate_proposal({"id": 0, "word": "Сам+оа", "stressed": "сам+оа"}, TEXTS) is None  # same stress
    assert validate_proposal("junk", TEXTS) is None


class _Result:
    def __init__(self, content: dict) -> None:
        self.content = content


class FakeClient:
    """Answers the proofreading prompt with `proposals` for the lines it gets; the vote prompt per `votes`."""

    def __init__(self, proposals=None, votes=None, fail=False, only=None) -> None:  # noqa: ANN001
        self.proposals, self.votes, self.fail, self.only = proposals or [], votes or {}, fail, only
        self.config = SimpleNamespace(model="fake")
        self.seen: list[str] = []
        self.seen_sentences: set[str] = set()

    def complete_json(self, system, user, prompt_version, tag):  # noqa: ANN001
        if self.fail:
            raise RuntimeError("not reachable")
        payload = json.loads(user)
        if "lines" in payload:
            self.seen += [x["text"] for x in payload["lines"]]
            ids = {x["id"] for x in payload["lines"]}
            return _Result({"items": [p for p in self.proposals if p["id"] in ids]})
        self.seen_sentences |= {x["sentence"] for x in payload["items"]}
        return _Result({"items": [{"id": x["id"], "stressed": self.votes.get(x["word"], x["word"])
                                   if self.only in (None, x["sentence"]) else x["word"]}
                                  for x in payload["items"]]})

    def stats(self) -> dict:
        return {}


def test_proofread_returns_valid_proposals_and_survives_failure() -> None:
    client = FakeClient([{"id": 0, "word": "Сам+оа", "stressed": "сам+оа"}, {"id": 1, "word": "Само+а", "stressed": "сам+оа"},
                         {"id": 3, "word": "x", "stressed": "x"}, {"id": 99, "word": "y", "stressed": "y"}])
    report: dict[str, int] = {}
    assert proofread(TEXTS, client, batch=2, report=report) == [Proposal(1, "само+а", "сам+оа")]
    assert report == {"proposals": 3} and client.seen == TEXTS
    assert proofread(TEXTS, FakeClient(fail=True)) == []


def _check(monkeypatch, texts, voters, proofreader_client, omographs=None, season=None, accents=None, passes=1):  # noqa: ANN001
    class Engine:
        def prepare(self, items):  # noqa: ANN001
            return [SimpleNamespace(tts_text=t) for t in texts]

    engine = Engine()
    engine._accentizer = SimpleNamespace(custom_dict={}, omographs=omographs or {},
                                         accents=accents or {"самоа": "сам+оа", "дома": "дом+а"})
    monkeypatch.setattr(stress_check, "build_engine", lambda config: engine)
    clients = iter(voters)
    monkeypatch.setattr(stress_check, "_client", lambda cfg, model, env, **k: proofreader_client if model == "sol"
                        else next(clients))
    lines = [{"text": t.replace("+", "")} for t in texts]
    return stress_check.check(lines, {}, {}, ["a", "b", "c"], second=False, season=season, proofreader="sol",
                             proofread_passes=passes)


def test_proposals_are_voted_and_a_majority_applies(monkeypatch) -> None:
    texts = ["Само+а тут.", "Это Само+а."]
    proposals = [{"id": 0, "word": "Само+а", "stressed": "сам+оа"}, {"id": 1, "word": "Само+а", "stressed": "сам+оа"}]
    voters = [FakeClient(votes={"само+а": "сам+оа"}), FakeClient(votes={"само+а": "сам+оа"}), FakeClient()]
    result = _check(monkeypatch, texts, voters, FakeClient(proposals))
    # two of three: fixed where it was voted, but not for the whole season
    assert result["dictionary"] == {} and result["season"]["decided"] == {}
    assert result["sentences"] == {"Самоа тут.": {"самоа": "сам+оа"}, "Это Самоа.": {"самоа": "сам+оа"}}
    assert result["proofread"] == {"model": "sol", "proposals": 2, "valid": 2, "corrected": 2}
    # all three: one stress for the season
    voters = [FakeClient(votes={"само+а": "сам+оа"}) for _ in range(3)]
    result = _check(monkeypatch, texts, voters, FakeClient(proposals))
    assert result["dictionary"] == {"самоа": "сам+оа"} and result["sentences"] == {}
    assert result["season"]["decided"]["самоа"]["source"] == "proofread"
    assert result["season"]["decided"]["самоа"]["votes"] == {"a": "сам+оа", "b": "сам+оа", "c": "сам+оа"}
    assert next(v for v in result["verdicts"] if v["word"] == "самоа")["contexts"][0]["proposal"] == "сам+оа"
    # one of three agrees: nothing changes
    voters = [FakeClient(votes={"само+а": "сам+оа"}), FakeClient(), FakeClient()]
    result = _check(monkeypatch, texts, voters, FakeClient(proposals))
    assert result["dictionary"] == {} and result["season"]["decided"] == {}
    assert result["proofread"]["corrected"] == 0


def test_homograph_correction_stays_in_its_sentence(monkeypatch) -> None:
    texts = ["Дом+а, в моём царств+е.", "Мы дом+а."]
    proposals = [{"id": 0, "word": "Дом+а", "stressed": "д+ома"}]
    voters = [FakeClient(votes={"дом+а": "д+ома"}, only="Дома, в моём царстве.") for _ in range(3)]
    omographs = {"дома": ["д+ома", "дом+а"]}
    result = _check(monkeypatch, texts, voters, FakeClient(proposals), omographs=omographs)
    assert result["sentences"] == {"Дома, в моём царстве.": {"дома": "д+ома"}}
    assert result["dictionary"] == {} and "дома" not in result["season"]["decided"]
    # not a homograph for RUAccent, but stressed two ways in the episode: also per sentence
    result = _check(monkeypatch, texts, [FakeClient(votes={"дом+а": "д+ома"}, only="Дома, в моём царстве.") for _ in range(3)],
                    FakeClient(proposals))
    assert result["sentences"] == {"Дома, в моём царстве.": {"дома": "д+ома"}} and result["dictionary"] == {}


def test_season_words_are_applied_before_proofreading(monkeypatch) -> None:
    texts = ["Само+а тут.", "Д+ома тут."]
    proofreader = FakeClient([{"id": 0, "word": "Сам+оа", "stressed": "само+а"}])
    season = {"самоа": {"stressed": "сам+оа", "source": "proofread"}, "дома": {"stressed": "дом+а"}}
    result = _check(monkeypatch, texts, [FakeClient(), FakeClient(), FakeClient()], proofreader,
                    omographs={"дома": ["д+ома", "дом+а"]}, season=season)
    assert proofreader.seen == ["Сам+оа тут.", "Д+ома тут."]  # the homograph is not forced
    assert result["dictionary"] == {"самоа": "сам+оа"}
    assert result["proofread"]["valid"] == 1 and result["proofread"]["corrected"] == 0  # the season's word: not asked
    assert result["season"]["decided"] == {}


def test_proofreader_failing_keeps_the_old_behaviour(monkeypatch) -> None:
    result = _check(monkeypatch, ["Само+а тут."], [FakeClient(), FakeClient(), FakeClient()], FakeClient(fail=True))
    assert result["dictionary"] == {} and result["proofread"]["proposals"] == 0


def test_plain_word_corrected_in_every_sentence_goes_to_the_whole_season(monkeypatch) -> None:
    texts = ["На Само+а.", "И снова Само+а, и хватит."]
    proposals = [{"id": 0, "word": "Само+а", "stressed": "сам+оа"}]  # the proofreader points at one sentence only
    voters = [FakeClient(votes={"само+а": "сам+оа"}) for _ in range(3)]
    result = _check(monkeypatch, texts, voters, FakeClient(proposals))
    assert result["dictionary"] == {"самоа": "сам+оа"} and result["sentences"] == {}
    assert result["season"]["decided"]["самоа"]["source"] == "proofread"
    assert all(v.seen_sentences >= {"На Самоа.", "И снова Самоа, и хватит."} for v in voters)  # the other one is voted on too


def test_two_readings_are_merged(monkeypatch) -> None:
    class Alternating(FakeClient):  # each reading finds a different error
        def complete_json(self, system, user, prompt_version, tag):  # noqa: ANN001
            self.proposals = [self.queue.pop(0)] if "lines" in json.loads(user) else self.proposals
            return super().complete_json(system, user, prompt_version, tag)

    reader = Alternating()
    reader.queue = [{"id": 0, "word": "Само+а", "stressed": "сам+оа"}, {"id": 1, "word": "Дом+а", "stressed": "д+ома"}]
    texts = ["На Само+а.", "Дом+а, в моём царстве."]
    voters = [FakeClient(votes={"само+а": "сам+оа", "дом+а": "д+ома"}) for _ in range(3)]
    result = _check(monkeypatch, texts, voters, reader, omographs={"дома": ["д+ома", "дом+а"]}, passes=2)
    assert result["proofread"]["proposals"] == 2 and result["proofread"]["valid"] == 2
    assert result["dictionary"] == {"самоа": "сам+оа"}
    assert result["sentences"] == {"Дома, в моём царстве.": {"дома": "д+ома"}}


def test_each_sentence_is_voted_against_its_own_stress(monkeypatch) -> None:
    """A homograph stressed two ways: the vote for the second sentence compares with ITS stress, not the first one's."""
    texts = ["Я д+ома.", "Дом+а, в моём царстве."]
    voters = [FakeClient(votes={"д+ома": "д+ома", "дом+а": "д+ома"}) for _ in range(3)]
    proposals = [{"id": 1, "word": "Дом+а", "stressed": "д+ома"}]
    result = _check(monkeypatch, texts, voters, FakeClient(proposals), omographs={"дома": ["д+ома", "дом+а"]})
    assert result["sentences"] == {"Дома, в моём царстве.": {"дома": "д+ома"}}


def test_screener_runs_while_the_text_is_proofread_with_the_same_questions(monkeypatch) -> None:
    import threading

    asked_by_screener = threading.Event()
    texts = ["Дом+а, в моём царств+е.", "Мы д+ома.", "Это Само+а."]
    omographs = {"дома": ["д+ома", "дом+а"]}

    class Screener(FakeClient):
        def complete_json(self, system, user, prompt_version, tag):  # noqa: ANN001
            asked_by_screener.set()
            return super().complete_json(system, user, prompt_version, tag)

    class Reader(FakeClient):
        def complete_json(self, system, user, prompt_version, tag):  # noqa: ANN001
            # the proofreading waits for the screener: it fails to see it when the two run one after the other
            self.overlap = getattr(self, "overlap", True) and asked_by_screener.wait(timeout=5)
            return super().complete_json(system, user, prompt_version, tag)

    def run(use_reader: bool) -> tuple[dict, Screener, Reader]:
        class Engine:
            def prepare(self, items):  # noqa: ANN001
                return [SimpleNamespace(tts_text=t) for t in texts]

        engine = Engine()
        engine._accentizer = SimpleNamespace(custom_dict={}, omographs=omographs, accents={"самоа": "сам+оа"})
        monkeypatch.setattr(stress_check, "build_engine", lambda config: engine)
        screener = Screener(votes={"дом+а": "д+ома"}, only="Мы дома.")
        reader = Reader([{"id": 2, "word": "Само+а", "stressed": "сам+оа"}])
        voters = iter([FakeClient(votes={"само+а": "сам+оа", "д+ома": "дом+а"}) for _ in range(3)])
        monkeypatch.setattr(stress_check, "_client", lambda cfg, model, env, **k:
                            screener if model == "scr" else reader if model == "sol" else next(voters))
        lines = [{"text": t.replace("+", "")} for t in texts]
        result = stress_check.check(lines, {}, {}, ["a", "b", "c"], second=False, screen={"model": "scr"},
                                    proofreader="sol" if use_reader else None)
        return result, screener, reader

    result, screener, reader = run(True)
    assert reader.overlap  # the screener was asked before the proofreading finished
    plain, plain_screener, _ = run(False)
    # the screener got the same questions (the base ones) with and without the proofreader
    assert screener.seen_sentences == plain_screener.seen_sentences == {"Дома, в моём царстве.", "Мы дома.", "Это Самоа."}
    screens = {c["sentence"]: c["screen"] for v in result["verdicts"] if v["word"] == "дома" for c in v["contexts"]}
    plain_screens = {c["sentence"]: c["screen"] for v in plain["verdicts"] if v["word"] == "дома" for c in v["contexts"]}
    assert screens == plain_screens and screens["Мы дома."] == "д+ома"
    assert result["proofread"]["valid"] == 1 and "screen:scr" in result["llm"]
