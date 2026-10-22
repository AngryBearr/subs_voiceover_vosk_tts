"""Pure-logic tests for the local GPU STT package (no models, no GPU).

Run: subs_env/bin/python -m unittest tests.test_stt_local -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from stt.align import attach_times
from stt.assemble import (
    CueOptions,
    assign_speakers,
    build_cues,
    mark_overlaps,
    merge_captions,
    on_screen_hints,
    sentence_majority,
    snap_speaker_boundaries,
    speaker_ids,
    subtitle_records,
)
from stt.diarize import Diarization, Turn
from stt.dub import is_interjection, plan
from stt.fusion import chunk_texts, fuse
from stt.mix import mix, speech_envelope
from stt.ocr import TextEvent, captions, classify
from stt.run import STAGES, Episode, dotenv
from stt.reference import Cue, clean_caption_line, load_ass_cues, snap_window, window_reference
from stt.textfix import collapse_repeats, repeat_keep_indices
from stt.translate import DurationModel, cue_windows, gender_mismatches, over_budget, split_parts
from stt.types import Segment, Word
from stt.vad import merge_regions
from stt.voice import clean_windows


class AlignTest(unittest.TestCase):
    def test_attach_times_keeps_punctuation_and_order(self) -> None:
        tokens = "Hello, world. It's me!".split()
        aligned = [("Hello", 0.1, 0.4), ("world", 0.5, 0.9), ("It's", 1.2, 1.4), ("me", 1.4, 1.6)]
        words = attach_times(tokens, aligned, offset=10.0)
        self.assertEqual([w.text for w in words], tokens)
        self.assertEqual((words[1].start, words[1].end), (10.5, 10.9))
        self.assertTrue(all(a.start <= b.start for a, b in zip(words, words[1:])))

    def test_attach_times_fills_unmatched_tokens(self) -> None:
        words = attach_times(["Mr.", "Quilter", "-", "yes"], [("mister", 0.0, 0.3), ("quilter", 0.3, 0.8),
                                                             ("yes", 1.0, 1.2)], offset=0.0)
        self.assertEqual(len(words), 4)
        self.assertEqual((words[0].start, words[0].end), (0.0, 0.3))  # replace of equal length
        self.assertEqual(words[3].start, 1.0)


class DubTest(unittest.TestCase):
    def test_interjections(self) -> None:
        self.assertTrue(is_interjection("Go, go, go!"))
        self.assertTrue(is_interjection("Oh my God."))
        self.assertFalse(is_interjection("Yes."))
        self.assertFalse(is_interjection("Come on in, guys!"))

    def test_plan_skips_dense_interjections_and_assigns_voices(self) -> None:
        records = [
            {"index": 1, "start": 0, "end": 2000, "text": ["Привет."], "text_en": ["Hello there."], "gender": "female"},
            {"index": 2, "start": 2300, "end": 2600, "text": ["Ага."], "text_en": ["Yeah."], "gender": "male"},
            {"index": 3, "start": 3000, "end": 5000, "text": ["Пошли."], "text_en": ["Let's head out."],
             "gender": "male"},
            {"index": 4, "start": 9000, "end": 9500, "text": ["Ух ты."], "text_en": ["Wow."], "gender": "unknown"},
        ]
        voiced, skipped = plan(records, {"female": 49, "male": 56, "unknown": 56})
        self.assertEqual([r["source_index"] for r in voiced], [1, 3, 4])  # isolated "Wow." is kept
        self.assertEqual([r["voice"] for r in voiced], [49, 56, 56])
        self.assertEqual([r["index"] for r in voiced], [1, 2, 3])
        self.assertEqual(skipped[0]["reason"], "interjection in a rapid exchange")


class MixTest(unittest.TestCase):
    def test_original_is_ducked_only_under_speech(self) -> None:
        sr = 1000
        original = np.full((3 * sr, 2), 0.5, np.float32)
        voice = np.zeros(3 * sr, np.float32)
        voice[sr : 2 * sr] = 0.1  # one second of "speech" in the middle
        out = mix(original, voice, sr, duck_db=12.0)
        self.assertAlmostEqual(float(out[100, 0]), 0.5, places=3)  # before speech: untouched
        self.assertLess(float(out[1600, 0]) - 0.1, 0.5 * 0.3)  # under speech: about -12 dB
        self.assertEqual(speech_envelope(voice, sr).shape, voice.shape)


class OcrTest(unittest.TestCase):
    def _event(self, start: float, text: str, box: tuple[float, float, float, float], end: float | None = None
               ) -> TextEvent:
        return TextEvent(start, end if end is not None else start + 2.0, text, "", box, 0.99, 4, [text])

    def test_classify_and_join_caption_lines(self) -> None:
        events = [
            self._event(10.0, "Shonee: I love your", (0.31, 0.79, 0.69, 0.85)),
            self._event(10.0, "tattoo, George.", (0.35, 0.84, 0.65, 0.91)),
            self._event(20.0, "GEORGE", (0.09, 0.73, 0.2, 0.86)),
            self._event(20.0, "SEASON 6 FINALIST  VILLAINS", (0.09, 0.84, 0.46, 0.9)),
            self._event(30.0, "THE LEGEND.", (0.38, 0.82, 0.51, 0.89)),
            self._event(40.0, "Enter at 10play.com.au/win", (0.24, 0.84, 0.76, 0.95)),
            self._event(50.0, "SHAUN", (0.07, 0.46, 0.39, 0.71), end=52.7),
        ]
        kinds = [e.kind for e in classify(events, 600.0)]
        self.assertEqual(kinds, ["caption", "caption", "name_card", "name_card_descriptor", "scene", "scene", "title"])
        self.assertIn("descriptor: SEASON 6 FINALIST  VILLAINS", events[2].variants)
        caps = captions(events)
        self.assertEqual([(c.label, c.text) for c in caps], [("Shonee", "I love your tattoo, George.")])

    def test_credits_roll_is_not_dialogue(self) -> None:
        box = (0.3, 0.8, 0.7, 0.86)
        events = [self._event(3819.3 + i, text, box, end=3820.3 + i) for i, text in enumerate(
            ["Wardrobe Supervisor", "Best Boy Grip", "I hope Natalie wins it all.", "Standby Props Art Department Buyer",
             "Dive Supervisor", "Editorial olutions by"])]
        events.append(self._event(100.0, "Back in the day", box))
        kinds = [e.kind for e in classify(events, 3900.0)]
        self.assertEqual(kinds, ["credits"] * 6 + ["caption"])

    def test_merge_captions_keeps_good_asr_and_replaces_missed(self) -> None:
        words = [Word(1.0, 1.3, "I"), Word(1.3, 1.6, "love"), Word(1.6, 1.9, "your"), Word(1.9, 2.4, "tattoo."),
                 Word(5.0, 5.4, "Soul."), Word(5.4, 5.8, "He's"), Word(5.8, 6.0, "there.")]
        caps = [{"start": 0.9, "end": 2.5, "text": "I love your tattoo."},
                {"start": 5.0, "end": 6.2, "text": "There's already three of us."},
                {"start": 9.0, "end": 9.8, "text": "Is it me?"}]
        merged, stats = merge_captions(words, caps)
        self.assertEqual(stats, {"captions": 3, "kept_asr": 1, "replaced": 1, "added": 1, "under_speech": 0})
        texts = " ".join(w.text for w in merged)
        self.assertEqual(texts, "I love your tattoo. There's already three of us. Is it me?")
        self.assertEqual([w.source for w in merged][-3:], ["ocr", "ocr", "ocr"])

    def test_merge_captions_aligns_shifted_back_to_back_captions(self) -> None:
        def timed(text: str, start: float) -> list[Word]:
            return [Word(round(start + 0.3 * i, 2), round(start + 0.3 * i + 0.25, 2), t)
                    for i, t in enumerate(text.split())]

        words = (timed("Does that hurt as you do that?", 46.0)
                 + timed("I'm excited you're here, Gerry.", 60.0)
                 + timed("I'm here to see a frickin' thing.", 61.6)
                 + timed("And I think you're probably gonna take on some of those big boys.", 63.8))
        caps = [{"start": 47.0, "end": 49.0, "text": "Does that hurt once you do that?"},
                {"start": 60.1, "end": 61.7, "text": "I'm excited you're here Gerry."},
                {"start": 61.7, "end": 63.8, "text": "I think you're going to be a friggin' hoot..."},
                {"start": 63.9, "end": 67.5, "text": "and I think you're probably going to take on some of those big boys"}]
        merged, stats = merge_captions(words, caps)
        self.assertEqual(stats, {"captions": 4, "kept_asr": 3, "replaced": 1, "added": 0, "under_speech": 0})
        self.assertEqual(" ".join(w.text for w in merged),
                         "Does that hurt as you do that? I'm excited you're here, Gerry. "
                         "I think you're going to be a friggin' hoot... "
                         "And I think you're probably gonna take on some of those big boys.")

    def test_merge_captions_skips_caption_under_other_speech(self) -> None:
        words = [Word(10.0 + 0.8 * i, 10.6 + 0.8 * i, t)
                 for i, t in enumerate("Sam hands it off to Shaun, who climbs up.".split())]
        merged, stats = merge_captions(words, [{"start": 12.8, "end": 13.8, "text": "Someone swap over."}])
        self.assertEqual(stats["under_speech"], 1)
        self.assertEqual(merged, words)

    def test_on_screen_hints(self) -> None:
        diarization = Diarization(turns=[Turn(0.0, 5.0, "A"), Turn(5.0, 9.0, "B")])
        words = [Word(6.0, 6.5, "Hi", speaker="B"), Word(6.5, 7.0, "there.", speaker="B")]
        ocr = {"cards": [{"start": 1.0, "end": 3.0, "kind": "name_card", "text": "GEORGE", "descriptor": "VILLAINS"}],
               "captions": [{"start": 6.0, "end": 7.0, "text": "Hi there.", "label": "Stevie"}]}
        self.assertEqual(on_screen_hints(ocr, words, diarization),
                         {"A": ["name_card: GEORGE (VILLAINS)"], "B": ["caption label: Stevie"]})


class RunTest(unittest.TestCase):
    def _episode(self, tmp: str, force: str | None = None) -> Episode:
        import argparse

        args = argparse.Namespace(season_glossary=None, force=force, until="dub")
        return Episode(Path(tmp) / "Season 1" / "ep01.mkv", Path(tmp) / "out", args, {})

    def test_stages_skip_done_outputs_and_force_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            done = Path(tmp) / "done.json"
            done.write_text("{}")
            ep = self._episode(tmp)
            self.assertFalse(ep.needed("translate", [done]))
            self.assertTrue(ep.needed("translate", [done, Path(tmp) / "missing.json"]))
            forced = self._episode(tmp, force="speakers")
            self.assertFalse(forced.needed("glossary", [done]))
            self.assertTrue(forced.needed("translate", [done]))
            self.assertEqual(ep.season.name, "Season 1.glossary.json")
        self.assertEqual(STAGES[-1], "mix")

    def test_dotenv_parses_without_executing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text("# comment\nexport A_KEY='x1'\nB=\"y 2\"\nnot a line $(rm -rf /)\n")
            self.assertEqual(dotenv(env), {"A_KEY": "x1", "B": "y 2"})


class _FakeClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat_json(self, system: str, user: str) -> dict:
        self.prompts.append(user)
        return {"chunks": [{"id": 0, "text": "Hello, Jordie."}]}  # chunk 1 missing -> fallback


class FusionTest(unittest.TestCase):
    def test_chunk_texts_by_midpoint(self) -> None:
        segs = [Segment(0.0, 2.0, "a b"), Segment(2.5, 4.0, "c"), Segment(10.0, 12.0, "d")]
        self.assertEqual(chunk_texts(segs, [(0.0, 5.0), (9.0, 13.0)]), ["a b c", "d"])

    def test_fuse_uses_llm_and_falls_back_to_first_model(self) -> None:
        chunks = [(0.0, 5.0), (9.0, 13.0)]
        hyps = {"m1": [Segment(0.0, 5.0, "hello jordy"), Segment(9.0, 13.0, "bye")],
                "m2": [Segment(0.0, 5.0, "hello jordie"), Segment(9.0, 13.0, "by")]}
        client = _FakeClient()
        fused = fuse(chunks, hyps, client, glossary="Jordie -> Джорди")  # type: ignore[arg-type]
        self.assertEqual([(s.start, s.text) for s in fused], [(0.0, "Hello, Jordie."), (9.0, "bye")])
        self.assertIn('"asr2": "hello jordie"', client.prompts[0])
        self.assertIn("Jordie -> Джорди", client.prompts[0])


class ReferenceTest(unittest.TestCase):
    def test_clean_caption_line(self) -> None:
        self.assertEqual(clean_caption_line("<i>JEFF: Come on in, guys!</i>"), "Come on in, guys!")
        self.assertEqual(clean_caption_line("- [laughs] Yeah. (sighs)"), "Yeah.")
        self.assertEqual(clean_caption_line("♪ la la la ♪"), "")

    def test_snap_window_keeps_whole_cues(self) -> None:
        cues = [Cue(0.0, 2.0, "a"), Cue(3.0, 5.0, "b"), Cue(6.0, 8.0, "c"), Cue(9.0, 11.0, "d")]
        start, end = snap_window(cues, start=2.5, duration=6.0)  # requested 2.5..8.5
        self.assertEqual((start, end), (2.7, 8.3))
        ref = window_reference(cues, start, end)
        self.assertEqual(ref["text"], "b c")


    def test_load_ass_cues(self) -> None:
        content = (
            "[Script Info]\nTitle: x\n\n[V4+ Styles]\nFormat: Name, Fontname\nStyle: Subs,Arial\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:00:19.08,0:00:22.22,Subs,,0,0,0,,ДЖОНАТАН: Семь\\Nсезонов, да...\n"
            "Dialogue: 0,0:00:22.25,0:00:24.98,Subs,,0,0,0,,{\\i1}(МУЗЫКА){\\i0}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.ass"
            path.write_text(content, encoding="utf-8")
            raw = load_ass_cues(path)
            clean = load_ass_cues(path, clean=True)
        self.assertEqual([c.text for c in raw], ["ДЖОНАТАН: Семь сезонов, да...", "(МУЗЫКА)"])
        self.assertEqual(raw[0].start, 19.08)
        self.assertEqual([c.text for c in clean], ["Семь сезонов, да..."])


class TextFixTest(unittest.TestCase):
    def test_collapses_loops_keeps_normal_repeats(self) -> None:
        tokens = "Push! Go, go, go, go, go, go, go. Be driven".split()
        self.assertEqual(" ".join(tokens[i] for i in repeat_keep_indices(tokens)), "Push! Go, go, go, go, Be driven")
        normal = "yes yes yes no".split()
        self.assertEqual(repeat_keep_indices(normal), [0, 1, 2, 3])

    def test_multiword_loop_on_segments(self) -> None:
        text = " ".join(["Oh my god!"] * 9) + " Done."
        cleaned, removed = collapse_repeats([Segment(0.0, 5.0, text)])
        self.assertEqual(cleaned[0].text, " ".join(["Oh my god!"] * 4) + " Done.")
        self.assertEqual(removed, 15)


class TranslateBudgetTest(unittest.TestCase):
    def test_windows_extend_into_pauses_only(self) -> None:
        records = [
            {"index": 1, "start": 0, "end": 2000},
            {"index": 2, "start": 2300, "end": 4000},  # 0.3 s pause before it
            {"index": 3, "start": 3500, "end": 5000},  # overlaps the previous cue
            {"index": 4, "start": 9000, "end": 10000},
        ]
        self.assertEqual(cue_windows(records, extend_s=0.6), [2.3, 1.7, 2.1, 1.6])

    def test_gender_mismatches(self) -> None:
        self.assertEqual(gender_mismatches("Так я и решил.", "female"), ["я и решил"])
        self.assertEqual(gender_mismatches("Я так устала, но победила!", "female"), [])
        self.assertEqual(gender_mismatches("Я устала", "male"), ["Я устала"])
        self.assertEqual(gender_mismatches("Я устал", "unknown"), [])

    def test_split_parts_at_longest_pauses(self) -> None:
        records, t = [], 0
        for i in range(400):
            gap = 30000 if i in (100, 300) else 500  # two long scene pauses
            t += gap
            records.append({"index": i + 1, "start": t, "end": t + 1000})
            t += 1000
        parts = split_parts(records, 3)
        self.assertEqual([p[0]["index"] for p in parts], [1, 101, 301])
        self.assertEqual(sum(len(p) for p in parts), 400)
        self.assertEqual(len(split_parts(records[:50], 4)), 1)

    def test_duration_model_budgets(self) -> None:
        self.assertEqual(DurationModel.from_cps(14).max_chars(2.0), 28)
        vosk = DurationModel(intercept=0.5, slope=0.05)
        self.assertEqual(vosk.max_chars(3.0), 50)
        self.assertEqual(vosk.max_chars(0.5), 8)  # floor for interjections
        self.assertEqual(over_budget({1: "x" * 30, 2: "y" * 10}, {1: 20, 2: 20}, 1.15), [1])


class VadTest(unittest.TestCase):
    def test_merge_regions_respects_limits(self) -> None:
        regions = [(0.0, 5.0), (5.5, 20.0), (21.0, 27.0), (40.0, 45.0)]
        self.assertEqual(merge_regions(regions, max_chunk_s=28.0, max_gap_s=2.0), [(0.0, 27.0), (40.0, 45.0)])
        self.assertEqual(merge_regions(regions, max_chunk_s=20.0), [(0.0, 20.0), (21.0, 27.0), (40.0, 45.0)])


def _words(spec: list[tuple[float, float, str]]) -> list[Word]:
    return [Word(start=s, end=e, text=t) for s, e, t in spec]


class AssembleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.diarization = Diarization(
            turns=[Turn(0.0, 3.0, "A"), Turn(2.8, 6.0, "B"), Turn(6.5, 9.0, "A")],
            exclusive=[Turn(0.0, 2.9, "A"), Turn(2.9, 6.0, "B"), Turn(6.5, 9.0, "A")],
        )
        self.options = CueOptions()

    def test_speaker_assignment_and_cues(self) -> None:
        words = _words(
            [(0.1, 0.5, "Hi"), (0.6, 1.0, "there."), (3.1, 3.5, "Hello"), (3.6, 4.0, "back."),
             (6.6, 7.0, "Bye"), (7.1, 7.4, "now.")]
        )
        labelled = assign_speakers(words, self.diarization, self.options)
        self.assertEqual([w.speaker for w in labelled], ["A", "A", "B", "B", "A", "A"])
        cues = build_cues(labelled, self.options)
        self.assertEqual(
            [(c.speaker, c.text) for c in cues], [("A", "Hi there."), ("B", "Hello back."), ("A", "Bye now.")]
        )
        ids = speaker_ids(cues)
        self.assertEqual(ids, {"A": "S01", "B": "S02"})
        records = subtitle_records(cues, ids, {"S01": "female"}, offset_s=100.0)
        self.assertEqual(records[0]["start"], 100100)
        self.assertEqual(records[0]["text"], ["Hi there."])
        self.assertEqual(records[0]["gender"], "female")
        self.assertEqual(records[1]["gender"], "unknown")
        self.assertEqual([r["index"] for r in records], [1, 2, 3])

    def test_long_runs_split_at_sentence_end(self) -> None:
        spec = []
        t = 0.0
        for i in range(30):
            text = f"w{i}." if i == 14 else f"w{i}"
            spec.append((t, t + 0.25, text))
            t += 0.3
        words = [Word(s, e, x, speaker="A") for s, e, x in spec]
        cues = build_cues(words, CueOptions(max_duration_s=7.0, sentence_split_s=100.0))
        self.assertTrue(cues[0].text.endswith("w14."))
        self.assertTrue(all(c.end - c.start <= 7.0 for c in cues))

    def test_snap_moves_mid_sentence_change_to_sentence_end(self) -> None:
        words = [Word(i * 0.4, i * 0.4 + 0.3, t, speaker=sp) for i, (t, sp) in enumerate(
            [("I", "A"), ("really", "A"), ("think", "A"), ("so.", "B"), ("Yes.", "B"), ("No.", "A")])]
        snapped = snap_speaker_boundaries(words)
        self.assertEqual([w.speaker for w in snapped], ["A", "A", "A", "A", "B", "A"])

    def test_snap_moves_change_back_to_sentence_start(self) -> None:
        words = [Word(i * 0.4, i * 0.4 + 0.3, t, speaker=sp) for i, (t, sp) in enumerate(
            [("Done.", "A"), ("Well", "A"), ("I", "B"), ("want", "B"), ("more", "B"), ("food", "B")])]
        snapped = snap_speaker_boundaries(words)
        self.assertEqual([w.speaker for w in snapped], ["A", "B", "B", "B", "B", "B"])

    def test_sentence_majority_fixes_flicker(self) -> None:
        spec = [("intelligent.", "A"), ("What's", "B"), ("your", "C"), ("game", "C"), ("plan,", "B"), ("then?", "C"),
                ("Yeah.", "A")]
        words = [Word(i * 0.3, i * 0.3 + 0.2, t, speaker=sp) for i, (t, sp) in enumerate(spec)]
        self.assertEqual([w.speaker for w in sentence_majority(words)], ["A", "C", "C", "C", "C", "C", "A"])

    def test_pause_inside_sentence_does_not_split(self) -> None:
        spec = [(0.0, 0.3, "I"), (0.4, 0.8, "went"), (2.0, 2.4, "home."), (3.5, 3.8, "Bye.")]
        words = [Word(s, e, t, speaker="A") for s, e, t in spec]
        cues = build_cues(words, CueOptions())
        self.assertEqual([c.text for c in cues], ["I went home.", "Bye."])

    def test_overlap_flag(self) -> None:
        words = assign_speakers(_words([(2.85, 3.0, "Wait"), (3.1, 3.3, "what")]), self.diarization, self.options)
        cues = build_cues(words, self.options)
        mark_overlaps(cues, self.diarization)
        self.assertTrue(any(c.overlap for c in cues))


class VoiceWindowsTest(unittest.TestCase):
    def test_clean_windows_exclude_other_speakers(self) -> None:
        turns = [Turn(0.0, 10.0, "A"), Turn(4.0, 5.0, "B")]
        windows = clean_windows(turns, "A", min_s=1.0, max_s=6.0, edge_s=0.0)
        self.assertEqual(windows, [(0.0, 4.0), (5.0, 10.0)])
        self.assertEqual(clean_windows(turns, "B", min_s=1.5, edge_s=0.0), [])


if __name__ == "__main__":
    unittest.main()
