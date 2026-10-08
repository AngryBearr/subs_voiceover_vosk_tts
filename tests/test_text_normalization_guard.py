"""Pure guard checks with fake normalizers; no models, downloads or TTS."""
from __future__ import annotations

import copy
import re
from typing import Any, cast

import pytest

from utils import text_normalizer as tn


class FakeRUNorm:
    def __init__(self, result: str) -> None:
        self.result = result

    def norm(self, text: str) -> str:
        return self.result


class FakeRUAccent:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls = 0

    def process_all(self, text: str, **kwargs: Any) -> str:
        self.calls += 1
        return self.result


def test_runorm_kat_mutation_fails_before_accent_or_tts_and_preserves_source() -> None:
    source = [{"index": 1266, "text": ["Верю, проголосуют за Кэт."], "start": 1, "end": 2}]
    original = copy.deepcopy(source)
    accent = FakeRUAccent("['Верю, проголосуют за к+ати.']")
    with pytest.raises(ValueError, match=r"RUNorm lexical change at cue index 1266:.*кэт.*кати"):
        tn.normalize_and_accent_subtitles(source, cast(Any, FakeRUNorm("Верю, проголосуют за кати.")), cast(Any, accent))
    assert accent.calls == 0 and source == original
    with pytest.raises(ValueError, match="RUNorm lexical change"):
        tn.normalize_subtitles_runorm_only(source, cast(Any, FakeRUNorm("Верю, проголосуют за кати.")))
    assert source == original


@pytest.mark.parametrize("before,normalized,accented", [
    ("Ёлка... КЭТ!", "Елка.. кэт!", "Е+лка.. кэ\u0301т!"),
    ("У меня 2,5 яблока.", "У меня два с половиной яблока.", "У мен+я дв+а с полов+иной +яблока."),
    ("Это Survivor.", "Это сурвайвор.", "+Это сурв+айвор."),
])
def test_safe_case_punctuation_accents_numbers_and_transliteration(
    before: str, normalized: str, accented: str,
) -> None:
    source = [{"index": 7, "text": [before], "start": 100, "end": 200}]
    original = copy.deepcopy(source)
    out = tn.normalize_and_accent_subtitles(source, cast(Any, FakeRUNorm(normalized)),
                                           cast(Any, FakeRUAccent(repr([accented]))))
    assert out[0]["text"] == [tn.restore_ellipsis(accented)] and source == original
    assert out[0]["index"] == 7 and out[0]["start"] == 100 and out[0]["end"] == 200


@pytest.mark.parametrize("before,after", [("Кэт", "кати"), ("2 apple", "3 apple"), ("Кэт и Дэз", "Дэз и Кэт")])
def test_ruaccent_preserves_all_letters_and_token_order(before: str, after: str) -> None:
    source = [{"index": 9, "text": [before], "_normalized_text": before}]
    original = copy.deepcopy(source)
    with pytest.raises(ValueError, match="RUAccent lexical change at cue index 9"):
        tn.ruaccent_pass(source, cast(Any, FakeRUAccent(repr([after]))))
    assert source == original


@pytest.mark.parametrize("result", ["[]", "['Кэт', 'Дэз']", "'Кэт'", "[None]", "['Дэз', 'Кэт']"])
def test_malformed_or_reordered_accent_batch_is_rejected(result: str) -> None:
    rows = [{"index": 1, "text": ["Кэт"], "_normalized_text": "Кэт"}]
    if result == "['Дэз', 'Кэт']":
        rows.append({"index": 2, "text": ["Дэз"], "_normalized_text": "Дэз"})
    original = copy.deepcopy(rows)
    with pytest.raises(ValueError, match="RUAccent"):
        tn.ruaccent_pass(rows, cast(Any, FakeRUAccent(result)))
    assert rows == original


def test_missing_normalized_binding_rejected() -> None:
    with pytest.raises(ValueError, match="bound RUNorm text"):
        tn.ruaccent_pass([{"index": 1, "text": ["Кэт"]}], cast(Any, FakeRUAccent("[]")))


def test_batch_runorm_word_swap_is_rejected() -> None:
    source = [{"index": 1, "text": ["Кэт"]}, {"index": 2, "text": ["Дэз"]}]
    with pytest.raises(ValueError, match="RUNorm lexical change at cue index 1"):
        tn.normalize_subtitles_runorm_only(source, cast(Any, FakeRUNorm(tn.DEFAULT_SEPARATOR.join(["Дэз", "Кэт"]))))


@pytest.mark.parametrize("before,after", [
    ("Катя и Кэт.", "Катя и кэт."),
    ("КЭТ, Катя, Кэт!", "Кэт, катя, кэт!"),
    ("Кэ\u0301т и Дэз", "к+эт и д+эз"),
])
def test_name_identity_case_stress_and_punctuation(before: str, after: str) -> None:
    tn._check_lexical_invariant(before, after, stage="RUNorm", cue_index=1266)
    tn._check_lexical_invariant(before, after, stage="RUAccent", cue_index=1266)


@pytest.mark.parametrize("before,after", [
    ("Катя и Кэт.", "Кэт и Катя."),
    ("Катя и Кэт.", "Катя и Катя."),
    ("Кэт и Кэт.", "Кэт."),
    ("Верю, проголосуют за Кэт.", "Знаю, проголосуют за Кэт."),
    ("Верю, проголосуют за Кэт.", "Верю, проголосуют против Кэт."),
])
@pytest.mark.parametrize("stage", ["RUNorm", "RUAccent"])
def test_unexpected_name_order_belief_and_voting_mutations_fail(
    before: str, after: str, stage: str,
) -> None:
    with pytest.raises(ValueError, match=f"{stage} lexical change"):
        tn._check_lexical_invariant(before, after, stage=stage, cue_index=1266)


def tagged_normalizer(tags: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> tn.ProtectedNameRUNorm:
    normalizer = tn.ProtectedNameRUNorm()
    monkeypatch.setattr(normalizer, "tagger", lambda text: tags, raising=False)
    return normalizer


@pytest.mark.parametrize("text", ["Кэт.", "кэт!", "КЭТ?", "«Кэт», Кэт.", "Катя, Кэт и Кати."])
def test_adapter_protects_whole_name_case_punctuation_and_repeated_occurrences(
    text: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tags = [{"start": m.start(), "end": m.end(), "word": m.group().casefold(),
             "entity_group": "ABBREVIATION" if m.group().rstrip(".,!?").casefold() == "кэт" else "PLAIN"}
            for m in re.finditer(r"\w+[.,!?]?", text)]
    original = copy.deepcopy(tags)
    normalizer = tagged_normalizer(tags, monkeypatch)
    monkeypatch.setattr(normalizer, "predict_abbr", lambda prompt: pytest.fail("Protected name must not request expansion"))
    assert normalizer.proccess_abbr(text) == text
    assert tags == original


def test_adapter_protects_split_name_tags_and_punctuation(monkeypatch: pytest.MonkeyPatch) -> None:
    tags = [{"start": a, "end": b, "word": word, "entity_group": "ABBREVIATION"}
            for a, b, word in [(0, 1, "к"), (1, 3, "##эт"), (3, 4, ".")]]
    assert tagged_normalizer(tags, monkeypatch).proccess_abbr("Кэт.") == "Кэт."


@pytest.mark.parametrize("text", ["Кэтрин", "Катя", "Кати", "ракэт", "Кэт2", "Кэт_два"])
def test_adapter_does_not_protect_other_words(text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    normalizer = tagged_normalizer([{"start": 0, "end": len(text), "word": text, "entity_group": "ABBREVIATION"}], monkeypatch)
    monkeypatch.setattr(normalizer, "predict_abbr", lambda prompt: "<extra_id_0>expanded</s>")
    assert normalizer.proccess_abbr(text) == "expanded"


def test_adapter_keeps_other_abbreviations_numbers_and_time_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Кэт РФ и 2 в 12:30"
    tags = [{"start": a, "end": b, "word": text[a:b], "entity_group": group}
            for a, b, group in [(0, 3, "ABBREVIATION"), (4, 6, "ABBREVIATION"),
                                (7, 8, "PLAIN"), (9, 10, "CARDINAL"),
                                (11, 12, "PLAIN"), (13, 18, "TIME")]]
    normalizer = tagged_normalizer(tags, monkeypatch)
    prompts: list[str] = []

    def predict(prompt: str) -> str:
        prompts.append(prompt)
        return "<extra_id_0>эр эф<extra_id_1>два</s>"

    monkeypatch.setattr(normalizer, "predict_abbr", predict)
    assert normalizer.proccess_abbr(text) == "Кэт эр эф и два в 12:30"
    assert prompts == ["Кэт [РФ]<extra_id_0> и [2]<extra_id_1> в 12:30"]


@pytest.mark.parametrize("tag", [
    {"start": -1, "end": 3, "word": "Кэт"},
    {"start": 0, "end": 99, "word": "Кэт"},
    {"start": True, "end": 3, "word": "Кэт"},
    {"start": 0, "end": 3, "word": "Катя"},
    {"start": 4, "end": 8, "word": "Кэт"},
    {"start": 0, "end": 8, "word": "Кэт Катя"},
])
def test_adapter_rejects_bad_offsets_foreign_words_and_crossing_spans(
    tag: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="source-bound|crosses protected"):
        tagged_normalizer([{**tag, "entity_group": "ABBREVIATION"}], monkeypatch).proccess_abbr("Кэт Катя")


def test_shared_loader_returns_adapter_without_changing_load_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(tn.ProtectedNameRUNorm, "load", lambda self, **kwargs: calls.append(kwargs))
    assert isinstance(tn._load_runorm("cpu"), tn.ProtectedNameRUNorm)
    assert calls == [{"workdir": "./runorm_cache", "model_size": "big", "device": "cpu"}]


@pytest.mark.parametrize("text,canonical", [
    ("...Кэт", "...Кэт"), ("Кэт... Дэз", "Кэт... Дэз"), ("Если это будет...", "Если это будет..."),
    ("…Кэт... Дэз…", "...Кэт... Дэз..."), ("..Кэт..", "..Кэт.."),
    ("... … ..", "... ... .."), ("3.5, т.д. и Кэт.", "3.5, т.д. и Кэт."),
])
def test_ellipsis_helpers_are_idempotent_and_keep_legacy_dots(text: str, canonical: str) -> None:
    for helper in (tn.replace_ellipsis, tn.restore_ellipsis):
        assert helper(text) == canonical and helper(helper(text)) == canonical
        assert tn._word_tokens(text) == tn._word_tokens(canonical)


@pytest.mark.parametrize("text", [
    "...Кэт", "Кэт... Дэз", "Если это будет...", "…Кэт... Кэт…", "..Кэт..",
    "... … ..", "3.5, т.д. и Кэт.",
])
def test_adapter_isolates_ellipsis_without_dropping_text_or_tagging_punctuation(
    text: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalizer = tn.ProtectedNameRUNorm()
    calls: list[str] = []

    def tagger(fragment: str) -> list[dict[str, Any]]:
        calls.append(fragment)
        assert not re.search(r"\.{2,}|…", fragment) and fragment.strip()
        return [{"start": m.start(), "end": m.end(), "word": m.group(),
                 "entity_group": "ABBREVIATION" if m.group().casefold() == "кэт" else "PLAIN"}
                for m in re.finditer(r"\w+", fragment)]

    monkeypatch.setattr(normalizer, "tagger", tagger, raising=False)
    monkeypatch.setattr(normalizer, "predict_abbr", lambda prompt: pytest.fail("No expansion expected"))
    assert normalizer.proccess_abbr(text) == text
    expected = [p for i, p in enumerate(re.split(r"(\.{2,}|…)", text)) if not i % 2 and p.strip()]
    assert calls == expected


def test_invented_word_after_ellipsis_still_fails_before_accent() -> None:
    accent = FakeRUAccent("['Если это б+удеттак...']")
    with pytest.raises(ValueError, match="RUNorm lexical change.*будеттак"):
        tn.normalize_and_accent_subtitles([{"index": 1262, "text": ["Если это будет..."]}],
            cast(Any, FakeRUNorm("Если это будеттак...")), cast(Any, accent))
    assert accent.calls == 0


def test_shared_paths_preserve_metadata_ellipsis_and_exact_tts_boundary() -> None:
    from utils.vosk_text import normalize_vosk_text

    class EchoRUNorm:
        def __init__(self) -> None:
            self.inputs: list[str] = []

        def norm(self, text: str) -> str:
            self.inputs.append(text)
            return text

    class EchoAccent:
        def process_all(self, text: str, **kwargs: Any) -> str:
            assert kwargs == {"skip_regex": r'[\[\],\"]'}
            return text.replace("будет", "б+удет")

    source = [{"index": 1262, "start": 100, "end": 200, "unknown": {"keep": True},
               "text": ["Тебя это устраивает? Если это будет…"]}]
    original = copy.deepcopy(source)
    normalizer = EchoRUNorm()
    out = tn.normalize_and_accent_subtitles(source, cast(Any, normalizer), cast(Any, EchoAccent()))
    assert normalizer.inputs == ["Тебя это устраивает? Если это будет..."]
    assert out[0]["text"] == ["Тебя это устраивает? Если это б+удет..."]
    spoken = normalize_vosk_text(out[0]["text"][0])
    assert spoken == "Тебя это устраивает? Если это б+удет..."
    tn._check_lexical_invariant(source[0]["text"][0], spoken, stage="RUAccent", cue_index=1262)
    only = tn.normalize_subtitles_runorm_only(source, cast(Any, EchoRUNorm()))
    assert only[0]["text"] == source[0]["text"] and source == original
    for row in (out[0], only[0]):
        assert {k: v for k, v in row.items() if k != "text"} == {k: v for k, v in source[0].items() if k != "text"}
