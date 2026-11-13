"""Shouts in the voice-over: interjection modes of stt.dub, the fill lines' voice, the neutral mode's caption rule,
and the shout layer (stt.crowd) that says the left-out interjections."""
from __future__ import annotations

from typing import Any

from stt import crowd
from stt.dub import plan

VOICES: dict[str, int | str] = {"female": "f_generic", "male": "m_generic", "unknown": "m_generic"}


def _rec(index: int, start: int, end: int, text_en: str, speaker: str | None = "S01", gender: str = "female",
         **extra: Any) -> dict[str, Any]:
    return {"index": index, "start": start, "end": end, "text": ["Ру."], "text_en": [text_en], "speaker": speaker,
            "gender": gender, **extra}


# ---- interjection modes

def _exchange() -> list[dict[str, Any]]:
    return [_rec(1, 0, 2000, "We are going in."),
            _rec(2, 2200, 2600, "Yeah!", speaker="S02", gender="male"),  # rapid exchange, not on anyone
            _rec(3, 2700, 4000, "Come on, let's go!", overlap=True),  # on top of another speaker
            _rec(4, 9000, 11000, "Wow.")]  # alone


def test_overlap_mode_voices_rapid_exchange_interjections_and_skips_only_overlapping_ones() -> None:
    voiced, skipped = plan(_exchange(), VOICES)  # "overlap" is the default
    assert [r["source_index"] for r in voiced] == [1, 2, 4]
    assert [(s["index"], s["reason"]) for s in skipped] == [(3, "interjection over another speaker")]


def test_dense_mode_also_skips_interjections_in_a_rapid_exchange() -> None:
    voiced, skipped = plan(_exchange(), VOICES, interjections="dense")
    assert [r["source_index"] for r in voiced] == [1, 4]
    assert {s["index"]: s["reason"] for s in skipped} == {2: "interjection in a rapid exchange",
                                                          3: "interjection over another speaker"}


def test_keep_and_all_modes() -> None:
    voiced, skipped = plan(_exchange(), VOICES, interjections="keep")
    assert len(voiced) == 4 and not skipped
    voiced, skipped = plan(_exchange(), VOICES, interjections="all")
    assert [r["source_index"] for r in voiced] == [1] and len(skipped) == 3


# ---- fill lines (second recognition pass)

def test_fill_lines_get_the_neutral_voice_by_default_and_their_own_voice_on_request() -> None:
    records = [_rec(1, 0, 2000, "Go, go, go, get it up!", fill=True), _rec(2, 3000, 5000, "I am fine.")]
    voiced, _ = plan(records, VOICES, speaker_voices={"S01": "/clips/s01.wav"})
    assert [r["voice"] for r in voiced] == ["m_generic", "/clips/s01.wav"]
    voiced, _ = plan(records, VOICES, speaker_voices={"S01": "/clips/s01.wav"}, neutral_gender="female")
    assert voiced[0]["voice"] == "f_generic"
    voiced, _ = plan(records, VOICES, speaker_voices={"S01": "/clips/s01.wav"}, fill_voice="speaker")
    assert voiced[0]["voice"] == "/clips/s01.wav"
    voiced, _ = plan([_rec(1, 0, 2000, "Go!", fill=True, speaker=None, gender="unknown")], VOICES, fill_voice="speaker")
    assert voiced[0]["voice"] == "m_generic"  # no speaker: the unknown voice


def test_a_fill_line_with_a_manual_fix_keeps_the_fix() -> None:
    voiced, _ = plan([_rec(1, 0, 2000, "Go!", fill=True)], VOICES, corrected={1},
                     speaker_voices={"S01": "/clips/s01.wav"})
    assert voiced[0]["voice"] == "/clips/s01.wav"


# ---- neutral mode: the ASR's misheard copy under a captioned line is not voiced twice

def test_neutral_mode_skips_a_line_mostly_under_a_voiced_captioned_line() -> None:
    caps = [{"start": 1.0, "end": 3.0, "text": "I really needed this so badly", "text_ru": "Мне это было так нужно."}]
    records = [_rec(1, 1000, 3000, "I really needed this so badly", on_screen=True),
               _rec(2, 1200, 2800, "I really need this so bad.", speaker="S02"),  # the misheard copy
               _rec(3, 2900, 9000, "A long confessional that only touches the caption", speaker="S03"),
               _rec(4, 2000, 2600, "I", speaker="S04")]  # a fragment under it
    voiced, skipped = plan(records, VOICES, on_screen_voice="neutral", caption_texts=caps)
    assert [r["source_index"] for r in voiced] == [1, 3]
    assert voiced[0]["voice"] == "m_generic"
    assert [(s["index"], s["reason"]) for s in skipped] == [(2, "under an on-screen caption: the caption's own line is voiced"),
                                                            (4, "under an on-screen caption: the caption's own line is voiced")]
    voiced, skipped = plan(records[:3], VOICES, on_screen_voice="neutral")  # spans unknown: nothing to compare with
    assert [r["source_index"] for r in voiced] == [1, 2, 3] and not skipped


def test_a_line_under_a_caption_known_only_by_its_times_is_voiced_in_both_modes() -> None:
    # without the caption's text nothing shows the line is the caption's speech: it is voiced as heard
    records = [_rec(1, 1000, 3000, "We should talk about it later.")]
    for mode in ("neutral", "text"):
        voiced, skipped = plan(records, VOICES, on_screen_voice=mode, captions=[(1.0, 3.0)])
        assert len(voiced) == 1 and not skipped


def test_neutral_mode_keeps_a_corrected_line_under_a_caption() -> None:
    records = [_rec(1, 1000, 3000, "Caption", on_screen=True), _rec(2, 1200, 2800, "Fixed by hand", speaker="S02")]
    voiced, _ = plan(records, VOICES, on_screen_voice="neutral", corrected={2})
    assert [r["source_index"] for r in voiced] == [1, 2]


# ---- shout layer

def _skipped(index: int, start: int, end: int, text: str, reason: str, speaker: str | None = "S01",
             gender: str = "male") -> dict[str, Any]:
    return {"index": index, "start": start, "end": end, "speaker": speaker, "gender": gender, "text_en": text,
            "text": [text], "reason": reason}


def test_groups_take_only_interjections_and_mark_crowds() -> None:
    skipped = [
        _skipped(1, 1000, 1400, "Да!", "interjection over another speaker"),
        _skipped(2, 1800, 2600, "Давай, давай!", "interjection over another speaker", speaker="S02", gender="female"),
        _skipped(3, 9000, 9500, "Ого.", "interjection in a rapid exchange"),
        _skipped(4, 20000, 23000, "Надпись", "on-screen caption: shown as Russian text"),
        _skipped(5, 30000, 30800, "Вперёд, вперёд!", "interjection over another speaker"),
        _skipped(6, 40000, 40800, "", "empty"),
    ]
    groups = crowd.groups(skipped)
    assert [(g["start"], g["end"], g["text"], g["crowd"]) for g in groups] == [
        (1000, 2600, "Давай, давай!", True),  # two speakers
        (9000, 9500, "Ого.", False),  # one person, no chant
        (30000, 30800, "Вперёд, вперёд!", True),  # a chant
    ]
    assert groups[0]["genders"] == ["male", "female"] and groups[0]["speakers"] == ["S01", "S02"]


def test_voice_pool_starts_with_the_dubs_generic_voice_for_every_engine() -> None:
    assert crowd.voice_pool("vosk", {"female": 49, "male": 56, "unknown": 56}) == {"female": [49, 38, 15],
                                                                                   "male": [56, 12, 27]}
    assert crowd.voice_pool("f5", {"female": "teratts:ru_f1+ru_f2", "male": "teratts:ru_m5"}) == {
        "female": ["teratts:ru_f1+ru_f2", "teratts:ru_f2"], "male": ["teratts:ru_m5", "teratts:ru_m1"]}
    assert crowd.voice_pool("teratts", {"female": "ru_f2", "male": "ru_m5"})["female"] == ["ru_f2"]  # no duplicate
    try:
        crowd.voice_pool("vosk", {})
    except ValueError:
        pass
    else:
        raise AssertionError("no voices must be an error")


def test_backchannels_are_not_shouts() -> None:
    assert crowd.is_backchannel("Да.") and crowd.is_backchannel("Ага, ага") and crowd.is_backchannel("Мм...")
    assert not crowd.is_backchannel("Да!") and not crowd.is_backchannel("Давай!") and not crowd.is_backchannel("Да, пошли.")
    assert not crowd.is_backchannel("")


def test_pick_voices_prefers_the_genders_heard_then_the_neutral_gender() -> None:
    pool = {"female": ["f1", "f2"], "male": ["m1", "m2"]}
    assert crowd.pick_voices(["female", "female", "male"], pool, 3) == ["f1", "m1", "f2"]
    assert crowd.pick_voices([], pool, 1) == ["m1"]
    assert crowd.pick_voices([], pool, 1, neutral_gender="female") == ["f1"]
    assert crowd.pick_voices(["female"], pool, 1, neutral_gender="male") == ["f1"]


def test_build_says_every_group_in_solo_mode_and_only_crowds_otherwise(tmp_path, monkeypatch) -> None:
    import json

    import numpy as np

    report = {"voices": {"female": 49, "male": 56, "unknown": 56}, "skipped": [
        _skipped(1, 1000, 1400, "Да!", "interjection over another speaker"),
        _skipped(2, 9000, 9500, "Вперёд, вперёд!", "interjection over another speaker"),
        _skipped(3, 20000, 23000, "Надпись", "on-screen caption: shown as Russian text"),
        _skipped(4, 25000, 25400, "Да.", "interjection over another speaker"),  # a backchannel, not a shout
        _skipped(5, 27000, 27400, "Ага, ага.", "interjection over another speaker")]}
    (tmp_path / "dub.report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "voiceover.vosk.config.json").write_text(json.dumps({"engine": {}, "export": {"sample_rate": 100}}),
                                                       encoding="utf-8")
    (tmp_path / "voiceover.vosk").mkdir()
    (tmp_path / "voiceover.vosk" / "calibration.json").write_text(json.dumps({"calibration": {"rate": 1.0}}),
                                                                  encoding="utf-8")

    class Engine:
        def prepare(self, items):
            return items

        def synthesize_many(self, prepared, cache, workers):
            return [type("R", (), {"error": None, "path": str(tmp_path / f"{i['cue_id'].replace('|', '_')}.wav")})()
                    for i in prepared]

    monkeypatch.setattr("voiceover.engine.build_engine", lambda cfg: Engine())
    monkeypatch.setattr("voiceover.render.stretch_clip", lambda path, rate, sr: np.ones(50, dtype=np.float32))
    monkeypatch.setattr(crowd.sf, "info", lambda path: type("I", (), {"duration": 30.0})())

    groups, track, sr = crowd.build(tmp_path, "vosk", 1, solo=True)
    assert [(g["text"], g["voices"]) for g in groups] == [("Да!", [56]), ("Вперёд, вперёд!", [56])]
    assert sr == 100 and track[100] > 0 and track[900] > 0  # both groups on the track, at their time
    groups, track, _ = crowd.build(tmp_path, "vosk", 3, solo=False)
    assert [(g["text"], g["voices"]) for g in groups] == [("Вперёд, вперёд!", [56, 49, 12])]  # genders interleaved
    assert track[100] == 0 and track[900] > 0


# ---- neutral mode with the captions' texts (onscreen.ru.json)

def _cap(start: float, end: float, text: str, text_ru: str | None = "Перевод титра.") -> dict[str, Any]:
    return {"start": start, "end": end, "text": text, "text_ru": text_ru}


def test_neutral_mode_says_the_captions_text_for_its_misheard_copy_without_a_line_of_its_own() -> None:
    # S48E01 22:41: the caption was never merged in as a line, the ASR kept its misheard version
    records = [_rec(1, 1200, 2800, "Should we do like being an angel?", speaker="S02")]
    caps = [_cap(1.0, 3.0, "Should we do, like, name and age and...", "Может, назовём имя, возраст и...")]
    voiced, skipped = plan(records, VOICES, on_screen_voice="neutral", caption_texts=caps,
                           speaker_voices={"S02": "/clips/s02.wav"})
    assert not skipped and len(voiced) == 1
    line = voiced[0]
    assert line["text"] == ["Может, назовём имя, возраст и..."]
    assert line["text_en"] == ["Should we do, like, name and age and..."]
    assert line["caption_text"] and line["on_screen"] and line["voice"] == "m_generic"  # the neutral voice, not S02's
    assert (line["start"], line["end"]) == (1200, 2800)  # at the time the ASR heard it
    voiced, skipped = plan(records, VOICES, caption_texts=caps)  # text mode: the caption's Russian text shows it
    assert not voiced and skipped[0]["reason"] == "under an on-screen caption: shown as Russian text"


def test_a_line_that_shares_almost_nothing_with_the_caption_is_voiced_as_heard() -> None:
    # S48E01 1:21: a misheard whisper and an answer next to a caption look the same by text: both are voiced
    records = [_rec(1, 1200, 2800, "I don't like being in the snow bathroom", speaker="S02")]
    caps = [_cap(1.0, 3.0, "I really needed this so badly", "Мне это было так нужно.")]
    for mode in ("neutral", "text"):
        voiced, skipped = plan(records, VOICES, on_screen_voice=mode, caption_texts=caps)
        assert not skipped and voiced[0]["text_en"] == ["I don't like being in the snow bathroom"]


def test_neutral_mode_keeps_a_long_line_over_a_quiet_caption_and_one_without_a_translation() -> None:
    long = _rec(1, 1000, 6000, "the environment we find ourselves in to say that we are all in this together now")
    voiced, _ = plan([long], VOICES, on_screen_voice="neutral", caption_texts=[_cap(1.5, 2.5, "Okay.")])
    assert voiced[0]["text_en"] == long["text_en"] and "caption_text" not in voiced[0]
    short = _rec(2, 1200, 2800, "Good next.")
    voiced, _ = plan([short], VOICES, on_screen_voice="neutral", caption_texts=[_cap(1.0, 3.0, "Kevin, you all right?", None)])
    assert voiced[0]["text_en"] == ["Good next."]  # no Russian caption text: as heard


def test_neutral_mode_drops_the_copy_only_when_the_captions_own_line_looks_alike() -> None:
    caps = [_cap(1.0, 3.0, "I really needed this so badly")]
    own = _rec(1, 1000, 3000, "I really needed this so badly", on_screen=True)
    copy = _rec(2, 2500, 3900, "I really need this so bad.", speaker="S02")
    voiced, skipped = plan([own, copy], VOICES, on_screen_voice="neutral", caption_texts=caps)
    assert [r["source_index"] for r in voiced] == [1]
    assert skipped[0]["reason"] == "under an on-screen caption: the caption's own line is voiced"
    half = _rec(2, 2500, 3900, "Should we do like being an angel?", speaker="S02")  # 0.76 alike: an own line exists
    voiced, _ = plan([_rec(1, 1000, 3000, "Should we do, like, name and age and...", on_screen=True), half], VOICES,
                     on_screen_voice="neutral", caption_texts=[_cap(1.0, 3.0, "Should we do, like, name and age and...")])
    assert [r["source_index"] for r in voiced] == [1, 2]
    answer = _rec(2, 2500, 3900, "I'm half Guyanese.", speaker="S02")
    voiced, skipped = plan([_rec(1, 1000, 3000, "You're Guyanese?", on_screen=True), answer], VOICES,
                           on_screen_voice="neutral", caption_texts=[_cap(1.0, 3.0, "You're Guyanese?")])
    assert [r["source_index"] for r in voiced] == [1, 2] and not skipped  # other speech next to the caption


def test_a_captions_text_is_said_once_under_it() -> None:
    caps = [_cap(1.0, 3.0, "You got this, Kev!", "Ты справишься, Кев!")]
    records = [_rec(1, 1000, 2000, "You got it, Kevin!"), _rec(2, 2100, 2900, "got this", speaker="S02")]
    voiced, skipped = plan(records, VOICES, on_screen_voice="neutral", caption_texts=caps)
    assert [(r["source_index"], r["text_en"][0]) for r in voiced] == [(1, "You got this, Kev!")]
    assert skipped[0]["reason"] == "under an on-screen caption: the caption's text is said by the line before"


def test_caption_items_prefer_the_translated_file(tmp_path) -> None:
    import json

    from stt.dub import caption_items, caption_spans

    (tmp_path / "onscreen.json").write_text(json.dumps([
        {"id": 1, "kind": "caption", "start": 1.0, "end": 2.0, "text": "Ow!"},
        {"id": 2, "kind": "name_card", "start": 5.0, "end": 6.0, "text": "JOE"}]), encoding="utf-8")
    assert caption_items(tmp_path) == [{"start": 1.0, "end": 2.0, "text": "Ow!", "text_ru": None}]
    (tmp_path / "onscreen.ru.json").write_text(json.dumps([
        {"id": 1, "kind": "caption", "start": 1.0, "end": 2.0, "text": "Ow!", "text_ru": "Ай!"}]), encoding="utf-8")
    assert caption_items(tmp_path)[0]["text_ru"] == "Ай!" and caption_spans(tmp_path) == [(1.0, 2.0)]
    assert caption_items(tmp_path / "nowhere") == []
