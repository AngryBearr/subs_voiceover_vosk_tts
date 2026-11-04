"""Deterministic OCR text repair: stray dots, glued words, damaged job lines."""

from __future__ import annotations

from stt.ocr import TextEvent, reclassify
from stt.ocr_repair import Vocabulary, fix_punctuation, proper_names, repair_text, split_token

ASR = "What's your name brother I feel really good about honoring what I said need this Jenna Rob".split()


def _vocab(*ocr: str, names: tuple[str, ...] = ()) -> Vocabulary:
    return Vocabulary.build(ASR, ocr, names)


def test_punctuation_stray_dots_and_spaces() -> None:
    assert fix_punctuation("Ifeel really good about..") == "Ifeel really good about..."
    assert fix_punctuation(".going Jenna.") == "going Jenna."
    assert fix_punctuation("...deciding who's going home.") == "...deciding who's going home."
    assert fix_punctuation("Hello , world !") == "Hello, world!"
    assert fix_punctuation("RESIDENT,OBGYN") == "RESIDENT, OBGYN"
    assert fix_punctuation("TOKA·TRIBE") == "TOKA TRIBE"
    assert fix_punctuation("A.B. Britten") == "A.B. Britten"


def test_glued_words_are_split_when_both_parts_are_known() -> None:
    vocab = _vocab()
    assert repair_text("What's yourname, brother?", vocab) == "What's your name, brother?"
    assert repair_text("Ifeel really good", vocab) == "I feel really good"
    assert repair_text("I'm honoringwhat I said.", vocab) == "I'm honoring what I said."
    assert repair_text("Ineed this.", vocab) == "I need this."


def test_known_words_and_names_are_never_split() -> None:
    vocab = _vocab("SHARONDA", names=("Sharonda",))
    for word in ("something", "inside", "Jenna", "Sharonda", "really", "brother"):
        assert split_token(word, vocab) == [word]
    assert repair_text("Sharonda, nice to meet you.", vocab) == "Sharonda, nice to meet you."


def test_capitals_need_long_parts_seen_in_the_ocr() -> None:
    vocab = _vocab("TOKA TRIBE", "TOKA TRIBE", "TOKATRIBE", "ANDY CASTOR", "VOICE ACTRESS", "SAVU TRIBE")
    assert repair_text("TOKATRIBE", vocab) == "TOKA TRIBE"
    assert repair_text("VOIGEACTRESS", vocab) == "VOICE ACTRESS"  # a damaged letter, one away from a seen word
    assert repair_text("ANDY CASTOR", vocab) == "ANDY CASTOR"  # "CAST" + "OR" is not trusted


def test_without_a_vocabulary_only_punctuation_changes() -> None:
    assert repair_text("Ifeel good..") == "Ifeel good..."


def test_proper_names_are_words_never_written_in_lower_case() -> None:
    names = proper_names(["Jenna", "said", "Said", "Brady,", "brady", "I'm"])
    assert "jenna" in names and "said" not in names and "brady" not in names


def test_reclassify_repairs_shown_texts_not_the_classification() -> None:
    def event(text: str, box: tuple[float, float, float, float], start: float = 1.0) -> dict:
        return TextEvent(start, start + 1.5, text, "scene", box, 0.99).to_dict()

    data = {"events": [event("Ifeel really good about..", (0.3, 0.8, 0.7, 0.9)),
                       event("DEVIN", (0.14, 0.74, 0.32, 0.85), 10.0),
                       event("SOCIOLOGYPROFESSOR", (0.14, 0.83, 0.5, 0.91), 10.0),
                       event("SOCIOLOGY PROFESSOR", (0.14, 0.83, 0.5, 0.91), 20.0),
                       event("SOCIOLOGY PROFESSOR", (0.14, 0.83, 0.5, 0.91), 30.0)]}
    out = reclassify(data, ASR)
    assert [c["text"] for c in out["captions"]] == ["I feel really good about..."]
    descriptors = [e["text"] for e in out["events"] if e["kind"] == "name_card_descriptor"]
    assert descriptors == ["SOCIOLOGY PROFESSOR"]
    assert reclassify(data)["captions"][0]["text"] == "Ifeel really good about.."  # no words: untouched
