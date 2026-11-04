"""On-screen captions: caption-wins merging, captioned cues, display items, translation, neutral voice."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stt import onscreen
from stt.assemble import CueOptions, build_cues, merge_captions, subtitle_records
from stt.dub import plan
from stt.types import Word

VOICES = {"female": "f_generic", "male": "m_generic", "unknown": "m_generic"}
REAL_OCR = Path(__file__).resolve().parents[1] / "output/speaker_ab_20261104/S51E01/common/ocr.json"


def _words(text: str, start: float, step: float = 0.3) -> list[Word]:
    return [Word(round(start + step * i, 2), round(start + step * i + 0.25, 2), t) for i, t in enumerate(text.split())]


# ---- merge_captions

def test_kept_caption_replaces_the_text_of_its_own_words_and_keeps_timing() -> None:
    words = _words("Let's go to the beach, Brad.", 5.0) + _words("Next up.", 9.0)
    caps = [{"start": 4.9, "end": 7.2, "text": "Let's go to the beach, Red."}]
    decisions: list[dict[str, Any]] = []
    merged, stats = merge_captions(words, caps, decisions=decisions)
    assert stats["kept_asr"] == 1 and stats["replaced"] == 0
    assert " ".join(w.text for w in merged) == "Let's go to the beach, Red. Next up."
    captioned = merged[:6]
    assert all(w.caption for w in captioned) and not any(w.caption for w in merged[6:])
    assert captioned[0].start == pytest.approx(5.0) and captioned[-1].end == pytest.approx(6.75, abs=0.01)
    assert decisions[0]["action"] == "kept_asr" and decisions[0]["changed"] and decisions[0]["voiced"]
    assert "Brad." in decisions[0]["asr"]


def test_kept_caption_with_the_same_words_only_marks_them() -> None:
    words = _words("I love your tattoo.", 1.0) + _words("Unrelated speech here.", 8.0)
    merged, _ = merge_captions(words, [{"start": 0.9, "end": 2.5, "text": "I love your tattoo"}])
    assert [w.text for w in merged[:4]] == ["I", "love", "your", "tattoo."]  # the ASR punctuation stays
    assert [w.caption for w in merged] == [True] * 4 + [False] * 3
    assert {w.source for w in merged} == {None}


def test_speech_that_is_not_the_caption_is_never_deleted() -> None:
    # a loud other person talks under/after the caption: those words stay as they are
    words = _words("You good", 10.0) + _words("Yeah, I'm good. Let's do it.", 11.0)
    caps = [{"start": 10.0, "end": 11.0, "text": "You good?"}, {"start": 12.0, "end": 12.8, "text": "Someone swap over."}]
    decisions: list[dict[str, Any]] = []
    merged, stats = merge_captions(words, caps, decisions=decisions)
    assert "Yeah, I'm good. Let's do it." in " ".join(w.text for w in merged)
    assert [d["action"] for d in decisions] in (["kept_asr", "rejected"], ["kept_asr", "under_speech"])
    assert not any(w.caption for w in merged if w.text in ("Let's", "do", "it."))
    assert stats["rejected"] + stats["under_speech"] == 1
    assert [d["voiced"] for d in decisions] == [True, False]


def test_decisions_follow_the_input_order_and_cover_replaced_and_added() -> None:
    words = [Word(1.0, 1.3, "I"), Word(1.3, 1.6, "love"), Word(1.6, 1.9, "your"), Word(1.9, 2.4, "tattoo."),
             Word(5.0, 5.3, "Theirs"), Word(5.3, 5.6, "all"), Word(5.6, 5.8, "ready"), Word(5.8, 6.0, "free"),
             Word(6.0, 6.2, "of"), Word(6.2, 6.4, "us.")]
    caps = [{"start": 9.0, "end": 9.8, "text": "Is it me?"},
            {"start": 5.0, "end": 6.2, "text": "There's already three of us."},
            {"start": 0.9, "end": 2.5, "text": "I love your tattoo."}]
    decisions: list[dict[str, Any]] = []
    merged, stats = merge_captions(words, caps, decisions=decisions)
    assert [d["action"] for d in decisions] == ["added", "replaced", "kept_asr"]
    assert [d["index"] for d in decisions] == [0, 1, 2] and all(d["voiced"] for d in decisions)
    assert stats["added"] == stats["replaced"] == stats["kept_asr"] == 1
    assert all(w.caption for w in merged)
    assert [w.source for w in merged][-5:] == ["ocr"] * 5


def test_merge_captions_without_decisions_is_unchanged() -> None:
    merged, stats = merge_captions(_words("hello there", 1.0), [{"start": 1.0, "end": 2.0, "text": "hello there"}])
    assert stats["kept_asr"] == 1 and len(merged) == 2


# ---- cues

def test_captioned_run_is_its_own_cue_with_on_screen_flag() -> None:
    words = [Word(0.0, 0.3, "Well"), Word(0.3, 0.6, "then,", speaker="A", caption=False),
             Word(0.7, 1.0, "I", speaker="A", caption=True), Word(1.0, 1.3, "see.", speaker="A", caption=True),
             Word(1.4, 1.7, "Go.", speaker="A")]
    words[0] = Word(0.0, 0.3, "Well", speaker="A")
    cues = build_cues(words, CueOptions())
    assert [c.text for c in cues] == ["Well then,", "I see.", "Go."]
    records = subtitle_records(cues, {"A": "S01"}, {"S01": "male"})
    assert [r.get("on_screen", False) for r in records] == [False, True, False]
    assert records[1]["ocr"] is False


# ---- collect

def _event(start: float, end: float, text: str, kind: str, box: list[float], variants: list[str] | None = None) -> dict:
    return {"start": start, "end": end, "text": text, "kind": kind, "box": box, "score": 0.99, "frames": 2,
            "variants": variants or []}


def _ocr() -> dict[str, Any]:
    return {
        "captions": [
            {"start": 70.0, "end": 72.0, "text": "I love your tattoo, George.", "label": "Shonee"},
            {"start": 80.0, "end": 81.0, "text": "There's a boat.", "label": None},
            {"start": 90.0, "end": 90.2, "text": "Too short.", "label": None},
        ],
        "cards": [],
        "events": [
            _event(70.0, 72.0, "Shonee: I love your", "caption", [0.30, 0.80, 0.70, 0.85]),
            _event(70.0, 72.0, "tattoo, George.", "caption", [0.35, 0.85, 0.65, 0.91]),
            _event(80.0, 81.0, "There's a boat.", "caption", [0.38, 0.82, 0.62, 0.90]),
            _event(90.0, 90.2, "Too short.", "caption", [0.38, 0.82, 0.62, 0.90]),
            _event(20.0, 22.0, "DEVIN", "name_card", [0.05, 0.70, 0.20, 0.80], ["descriptor: ACTOR"]),
            _event(20.0, 22.0, "ACTOR", "name_card_descriptor", [0.05, 0.80, 0.18, 0.85]),
            _event(30.0, 30.5, "SPIRIT OF THE PACIFIC", "lower_third", [0.39, 0.63, 0.71, 0.72]),
            _event(30.5, 31.5, "SPIRIT OF THE PACIFIC", "lower_third", [0.30, 0.71, 0.69, 0.80]),
            _event(31.5, 32.0, "OF THE PACIFIC", "lower_third", [0.0, 0.88, 0.47, 1.0]),
            _event(60.0, 62.0, "DAY 3", "lower_third", [0.4, 0.7, 0.6, 0.8]),
            _event(40.0, 41.0, "FISHING", "scene", [0.6, 0.8, 0.66, 0.87]),
            _event(45.0, 47.0, "..", "lower_third", [0.4, 0.7, 0.6, 0.8]),
            _event(50.0, 50.2, "BRIEF", "lower_third", [0.4, 0.7, 0.6, 0.8]),
        ],
    }


def test_collect_builds_items_with_boxes_merged_fragments_and_voiced_flag() -> None:
    decisions = [{"index": 0, "voiced": True}, {"index": 1, "voiced": False}, {"index": 2, "voiced": True}]
    items = onscreen.collect(_ocr(), decisions, offset=10.0)
    assert [i["id"] for i in items] == list(range(1, len(items) + 1))
    by_kind = {k: [i for i in items if i["kind"] == k] for k in ("caption", "name_card", "lower_third")}
    assert [c["text"] for c in by_kind["caption"]] == ["I love your tattoo, George.", "There's a boat."]
    first = by_kind["caption"][0]
    assert first["box"] == [0.3, 0.8, 0.7, 0.91] and first["voiced"] is True and first["label"] == "Shonee"
    assert first["start"] == 80.0  # the audio offset is added
    assert by_kind["caption"][1]["voiced"] is False
    card = by_kind["name_card"][0]
    assert (card["text"], card["descriptor"], card["box"], card["voiced"]) == (
        "DEVIN", "ACTOR", [0.05, 0.7, 0.2, 0.8], False)
    thirds = by_kind["lower_third"]
    assert [t["text"] for t in thirds] == ["SPIRIT OF THE PACIFIC", "DAY 3"]  # fragments merged; "..", short, scene dropped
    assert (thirds[0]["start"], thirds[0]["end"]) == (40.0, 42.0) and thirds[0]["voiced"] is False
    assert {i["kind"] for i in items} <= {"caption", "name_card", "lower_third"}


def test_collect_without_decisions_voices_nothing_and_skips_missing_boxes() -> None:
    ocr = {"captions": [{"start": 1.0, "end": 2.0, "text": "Hello there.", "label": None}], "cards": [], "events": []}
    items = onscreen.collect(ocr)
    assert items == [{"id": 1, "kind": "caption", "start": 1.0, "end": 2.0, "text": "Hello there.",
                      "descriptor": None, "box": None, "voiced": False}]


def test_name_card_with_lower_third_descriptor_consumes_it() -> None:
    ocr = {"captions": [], "cards": [], "events": [
        _event(5.0, 7.0, "KILBY", "name_card", [0.05, 0.70, 0.20, 0.80]),
        _event(5.0, 7.0, "GAME DESIGNER", "lower_third", [0.05, 0.80, 0.30, 0.86]),
    ]}
    items = onscreen.collect(ocr)
    assert [(i["kind"], i["text"], i["descriptor"]) for i in items] == [("name_card", "KILBY", "GAME DESIGNER")]


@pytest.mark.skipif(not REAL_OCR.exists(), reason="no S51E01 scan")
def test_collect_on_the_real_scan() -> None:
    ocr = json.loads(REAL_OCR.read_text(encoding="utf-8"))
    items = onscreen.collect(ocr, [])
    kinds = {i["kind"] for i in items}
    assert kinds == {"caption", "name_card", "lower_third"}
    assert sum(i["kind"] == "caption" for i in items) <= len(ocr["captions"])
    assert all(i["end"] - i["start"] >= 0.3 and not i["voiced"] for i in items)
    assert not any(i["text"] == "OF THE PACIFIC" for i in items if i["kind"] == "lower_third")


# ---- translation

class FakeClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat_json(self, system: str, user: str, **_: Any) -> dict[str, Any]:
        self.prompts.append(user)
        rows = [json.loads(line) for line in user.split("Lines:\n", 1)[1].splitlines()]
        return {"items": [{"i": r["i"], "ru": "RU " + r["en"], **({"descriptor_ru": "RU " + r["descriptor"]}
                                                                    if r.get("descriptor") else {})} for r in rows]}


def _items() -> list[dict[str, Any]]:
    return onscreen.collect(_ocr(), [{"index": 0, "voiced": True}])


def test_translate_items_adds_text_and_descriptor_and_passes_cast_and_glossary() -> None:
    client = FakeClient()
    speakers = [{"id": "S01", "name": "Devin Walker", "label_ru": "Девин", "gender": "male", "cues": 3}]
    out = onscreen.translate_items(_items(), speakers, client, glossary="Devin -> Девин")
    card = next(i for i in out if i["kind"] == "name_card")
    assert (card["text_ru"], card["descriptor_ru"]) == ("RU DEVIN", "RU ACTOR")
    assert all(i["text_ru"] for i in out)
    assert all("descriptor_ru" not in i for i in out if i["kind"] != "name_card")
    assert "Девин" in client.prompts[0] and "Devin -> Девин" in client.prompts[0]


def test_translate_folder_caches_unchanged_items(tmp_path: Path) -> None:
    items = _items()
    (tmp_path / "onscreen.json").write_text(json.dumps(items), encoding="utf-8")
    (tmp_path / "speakers.json").write_text("[]", encoding="utf-8")
    first = FakeClient()
    onscreen.translate_folder(tmp_path, first)
    second = FakeClient()
    result = onscreen.translate_folder(tmp_path, second)
    assert second.prompts == [] and [i["text_ru"] for i in result] == ["RU " + i["text"] for i in items]
    assert json.loads((tmp_path / "onscreen.ru.json").read_text(encoding="utf-8"))[0]["id"] == 1


def test_translate_retries_missing_items_then_fails() -> None:
    class Lossy:
        def chat_json(self, system: str, user: str, **_: Any) -> dict[str, Any]:
            return {"items": []}

    with pytest.raises(RuntimeError):
        onscreen.translate_items(_items(), [], Lossy())


# ---- speaker gender in the translation

CUES = [{"index": 1, "start": 70500, "end": 72000, "text": ["I was waiting."], "gender": "female", "speaker": "S01",
         "on_screen": True, "speaker_name": "Ана Сани"},
        {"index": 2, "start": 80000, "end": 81000, "text": ["Hm."], "gender": "male", "speaker": "S02",
         "speaker_name": "Девин Уэй"}]
SPEAKERS = [{"id": "S01", "name": "Ana Sani", "label_ru": "Ана Сани", "gender": "female", "cues": 3,
             "on_screen": ["name_card: ANA (VOICE ACTRESS)"]},
            {"id": "S02", "name": "Devin Way", "label_ru": "Девин Уэй", "gender": "male", "cues": 2,
             "on_screen": ["name_card: DEVIN (ACTOR)"]},
            {"id": "S03", "name": "Ana Lee", "label_ru": "Ана Ли", "gender": "female", "cues": 2}]


def _item(id: int, kind: str, text: str, start: float, end: float, **extra: Any) -> dict[str, Any]:
    return {"id": id, "kind": kind, "start": start, "end": end, "text": text, "descriptor": None, "box": None,
            "voiced": False, **extra}


def test_captions_get_the_gender_and_name_of_the_cue_that_overlaps_them() -> None:
    items = [_item(1, "caption", "I was waiting for tribes.", 70.7, 72.0, voiced=True),
             _item(2, "caption", "Hm.", 80.1, 80.9),
             _item(3, "caption", "Nobody here.", 200.0, 201.0),
             _item(4, "name_card", "ANA", 10.0, 12.0, descriptor="ACTRESS"),
             _item(5, "name_card", "KILBY", 10.0, 12.0), _item(6, "lower_third", "DAY 3", 5.0, 6.0)]
    onscreen.add_speaker_genders(items, CUES, SPEAKERS)
    assert [(i.get("gender"), i.get("speaker_ru")) for i in items] == [
        ("female", "Ана Сани"), ("male", "Девин Уэй"), (None, None), ("female", None), (None, None), (None, None)]


def test_a_caption_label_names_the_speaker() -> None:
    item = _item(1, "caption", "Hm.", 200.0, 201.0, label="Devin")
    onscreen.add_speaker_genders([item], CUES, SPEAKERS)
    assert (item["gender"], item["speaker_ru"]) == ("male", "Девин Уэй")


def test_gender_reaches_the_prompt_and_the_output_and_changes_the_cache_key() -> None:
    client = FakeClient()
    items = [_item(1, "caption", "I was waiting for tribes.", 70.7, 72.0, voiced=True)]
    first = onscreen.translate_items(items, SPEAKERS, client, cues=CUES)
    line = json.loads(client.prompts[0].split("Lines:\n", 1)[1])
    assert (line["gender"], line["speaker"]) == ("female", "Ана Сани") and first[0]["gender"] == "female"
    again = FakeClient()
    onscreen.translate_items(items, SPEAKERS, again, cues=CUES, cached=first)
    assert again.prompts == []  # the same input: cached
    cues_male = [{**CUES[0], "gender": "male"}, CUES[1]]
    changed = FakeClient()
    onscreen.translate_items(items, [{**SPEAKERS[0], "gender": "male"}, *SPEAKERS[1:]], changed, cues=cues_male,
                             cached=first)
    assert len(changed.prompts) == 1  # another speaker gender: translated again
    old = [{k: v for k, v in first[0].items() if k not in ("gender", "speaker_ru")}]
    assert len(_prompts(items, old)) == 1  # a cache from before the gender existed does not match


def _prompts(items: list[dict[str, Any]], cached: list[dict[str, Any]]) -> list[str]:
    client = FakeClient()
    onscreen.translate_items(items, SPEAKERS, client, cues=CUES, cached=cached)
    return client.prompts


# ---- merged card lines and truncated frames

def _card_ocr() -> dict[str, Any]:
    return {"captions": [], "cards": [], "events": [
        _event(10.0, 12.0, "DEVIN", "name_card", [0.14, 0.74, 0.32, 0.85]),
        _event(10.0, 12.0, "ACTOR", "name_card_descriptor", [0.14, 0.82, 0.28, 0.91]),
        _event(20.0, 20.5, "DEVIN ACTOR", "name_card", [0.14, 0.74, 0.46, 0.85]),
        _event(20.5, 22.0, "DEVIN", "name_card", [0.14, 0.74, 0.32, 0.85]),
        _event(20.5, 22.0, "TOKA TRIBE", "name_card_descriptor", [0.14, 0.83, 0.36, 0.91]),
        _event(30.0, 31.0, "JENNA VE", "name_card", [0.14, 0.73, 0.4, 0.86]),
        _event(31.0, 33.0, "JENNA", "name_card", [0.14, 0.73, 0.33, 0.85]),
        _event(31.0, 33.0, "WEDDING PHOTOGRAPHER", "lower_third", [0.33, 0.76, 0.76, 0.85]),
        _event(40.0, 41.0, "WELCOME TO", "lower_third", [0.6, 0.64, 0.87, 0.81]),
        _event(42.0, 42.5, "ELCOME TO", "lower_third", [0.62, 0.66, 0.85, 0.8]),
        _event(50.0, 51.0, "SOCIOLOGY PROFESSOR", "lower_third", [0.2, 0.7, 0.6, 0.8]),
        _event(100.0, 101.0, "ELCOME TO", "lower_third", [0.62, 0.66, 0.85, 0.8]),
    ]}


def test_card_lines_read_as_one_are_split_into_name_and_descriptor() -> None:
    items = onscreen.collect(_card_ocr())
    cards = [(i["text"], i["descriptor"]) for i in items if i["kind"] == "name_card"]
    assert cards == [("DEVIN", "ACTOR"), ("DEVIN", "ACTOR\nTOKA TRIBE"), ("JENNA", "WEDDING PHOTOGRAPHER")]
    assert onscreen.collect(_card_ocr(), names=["Devin"])[1]["text"] == "DEVIN"


def test_truncated_frames_fold_into_the_full_text_only_when_close_in_time_and_place() -> None:
    thirds = [(i["text"], i["start"], i["end"]) for i in onscreen.collect(_card_ocr()) if i["kind"] == "lower_third"]
    assert ("WELCOME TO", 40.0, 42.5) in thirds  # the cut frame 1 s later is part of it
    assert ("ELCOME TO", 100.0, 101.0) in thirds  # far away: nothing to fold into
    assert len([t for t in thirds if t[0] == "ELCOME TO"]) == 1


def test_a_graphic_that_repeats_the_job_line_of_a_card_is_dropped() -> None:
    ocr = _card_ocr()
    ocr["events"] += [_event(10.0, 11.0, "ACTOR", "lower_third", [0.14, 0.82, 0.28, 0.91]),
                      _event(11.0, 12.0, "CTOR", "lower_third", [0.14, 0.82, 0.28, 0.91])]
    assert not [i for i in onscreen.collect(ocr) if i["kind"] == "lower_third" and i["text"] in ("ACTOR", "CTOR")]


# ---- dub: neutral voice

def _rec(index: int, speaker: str = "S01", gender: str = "female", **extra: Any) -> dict[str, Any]:
    return {"index": index, "start": index * 5000, "end": index * 5000 + 2000, "text": ["Привет."],
            "text_en": ["Hello."], "speaker": speaker, "gender": gender, **extra}


def test_on_screen_lines_get_the_neutral_voice_not_the_speakers() -> None:
    records = [_rec(1), _rec(2, on_screen=True), _rec(3, speaker="S02", gender="male", on_screen=True)]
    voiced, _ = plan(records, VOICES, speaker_voices={"S01": "/clips/s01.wav", "S02": "/clips/jeff.wav"})
    assert [r["voice"] for r in voiced] == ["/clips/s01.wav", "m_generic", "m_generic"]
    female, _ = plan([_rec(1, on_screen=True)], VOICES, speaker_voices={"S01": "/clips/s01.wav"},
                     neutral_gender="female")
    assert female[0]["voice"] == "f_generic"


def test_hero_option_keeps_the_speakers_voice() -> None:
    voiced, _ = plan([_rec(1, on_screen=True)], VOICES, speaker_voices={"S01": "/clips/s01.wav"},
                     on_screen_voice="hero")
    assert voiced[0]["voice"] == "/clips/s01.wav"


def test_manual_correction_wins_over_the_neutral_voice() -> None:
    record = _rec(1, on_screen=True, gender_changed=True)
    voiced, _ = plan([record], VOICES, corrected={1})
    assert voiced[0]["voice"] == "f_generic"
    voiced, _ = plan([_rec(2, on_screen=True)], VOICES, corrected={2}, speaker_voices={"S01": "/clips/s01.wav"})
    assert voiced[0]["voice"] == "/clips/s01.wav"


def test_dub_main_reads_corrections_and_cli_options(tmp_path: Path) -> None:
    from stt import dub

    records = [_rec(1, on_screen=True), _rec(2, on_screen=True)]
    (tmp_path / "subs.ru.json").write_text(json.dumps(records), encoding="utf-8")
    (tmp_path / "speakers.json").write_text(json.dumps([
        {"id": "S01", "gender": "female", "name": "Ana"}, {"id": "S02", "gender": "male", "name": "Rob"}]), encoding="utf-8")
    (tmp_path / "corrections.json").write_text(json.dumps({"2": {"gender": "male"}}), encoding="utf-8")
    dub.main([str(tmp_path), "--voices", "female=f_generic,male=m_generic", "--neutral-gender", "female"])
    out = json.loads((tmp_path / "dub.ru.json").read_text(encoding="utf-8"))
    assert [r["voice"] for r in out] == ["f_generic", "m_generic"]  # line 2: its corrected gender, not the neutral one
