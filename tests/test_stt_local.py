"""Pure-logic tests for the local GPU STT package (no models, no GPU).

Run: subs_env/bin/python -m unittest tests.test_stt_local -v
"""

from __future__ import annotations

import os
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
            self._event(60.0, "CURRENTLY IN 7TH PLACE", (0.3, 0.8, 0.7, 0.86)),
        ]
        kinds = [e.kind for e in classify(events, 600.0)]
        self.assertEqual(kinds, ["caption", "caption", "name_card", "name_card_descriptor", "lower_third", "scene",
                                 "title", "lower_third"])
        self.assertIn("descriptor: SEASON 6 FINALIST  VILLAINS", events[2].variants)
        caps = captions(events)
        self.assertEqual([(c.label, c.text) for c in caps], [("Shonee", "I love your tattoo, George.")])

    def test_damaged_video_packet_is_skipped(self) -> None:
        import av

        from stt.ocr import _decoded_frames

        class Packet:
            def __init__(self, pts: int, bad: bool = False) -> None:
                self.pts, self.bad = pts, bad

            def decode(self) -> list[str]:
                if self.bad:
                    raise av.error.InvalidDataError(1094995529, "Invalid data found when processing input")
                return [f"frame{self.pts}"]

        class Container:
            def demux(self, stream: object) -> list[Packet]:
                return [Packet(0), Packet(1000, bad=True), Packet(2000)]

        class Stream:
            time_base = 0.001

        stats: dict = {}
        self.assertEqual(list(_decoded_frames(Container(), Stream(), stats)), ["frame0", "frame2000"])
        self.assertEqual(stats, {"bad_packets": 1, "bad_packet_times": [1.0]})

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
            import os

            self.assertFalse(ep.needed("translate", [done], "speakers"))  # no finish marks yet
            for stage, age in (("speakers", 10), ("translate", 20)):
                ep.stamp(stage).parent.mkdir(parents=True, exist_ok=True)
                ep.stamp(stage).touch()
                os.utime(ep.stamp(stage), (ep.stamp(stage).stat().st_mtime - age,) * 2)
            self.assertTrue(ep.needed("translate", [done], "speakers"))  # speakers was redone since
            ep.stamp("translate").touch()
            self.assertFalse(ep.needed("translate", [done], "speakers"))
            forced = self._episode(tmp, force="speakers")
            self.assertFalse(forced.needed("glossary", [done]))
            self.assertTrue(forced.needed("translate", [done]))
            self.assertEqual(ep.season.name, "Season 1.glossary.json")
        self.assertEqual(STAGES[-1], "mix")

    def test_cast_lookup_once_per_season(self) -> None:
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            ep = self._episode(tmp)
            self.assertEqual(ep.cast.name, "Season 1.cast.json")
            ep.cast.parent.mkdir(parents=True)
            calls: list[list[str]] = []

            def fake_main(argv: list[str]) -> int:
                calls.append(argv)
                Path(argv[argv.index("--cast") + 1]).write_text('{"cast": []}')
                return 0

            with mock.patch("stt.cast.main", side_effect=fake_main):
                self.assertTrue(ep._find_cast())
                self.assertTrue(ep._find_cast())  # the file is there: no second lookup
            self.assertEqual(len(calls), 1)
            self.assertIn("openrouter:openai/gpt-6-luna:online,openrouter:anthropic/claude-haiku-5.5:online", calls[0])
            ep.cast.unlink()
            with mock.patch("stt.cast.main", side_effect=RuntimeError("no web")):
                self.assertFalse(ep._find_cast())  # a failed lookup is only a warning
            ep.config = {"cast": {"enabled": False}}
            self.assertFalse(ep._find_cast())

    def test_output_folders_and_one_video_with_all_voiceovers(self) -> None:
        import argparse
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(season_glossary=None, force=None, until="mix", engine="vosk", start=None,
                                      duration=None)
            config = {"paths": {"season": f"{tmp}/voices", "finished": f"{tmp}/done", "finished_folder": "STT_{season}"},
                      "voiceover": {"engines": {"vosk": {}, "teratts": {}, "f5": {}}},
                      "mix": {"tracks": "combined", "track_order": ["f5", "teratts", "vosk"],
                              "track_titles": {"f5": "Голоса героев"}, "subtitles": False}}
            ep = Episode(Path(tmp) / "Season 1" / "ep01.mkv", Path(tmp) / "out", args, config)
            self.assertEqual(ep.voice_bank, Path(tmp) / "voices" / "Season 1.voices.json")
            self.assertEqual(ep.references.parent, Path(tmp) / "voices")
            self.assertEqual(ep.out, Path(tmp) / "out" / "ep01")
            ep.out.mkdir(parents=True)
            for engine in ("vosk", "f5"):  # TeraTTS not voiced yet
                ep.track(engine).write_bytes(b"aac")
            with mock.patch("stt.mix.mux_tracks") as mux:
                ep.mux()
            video, tracks, out, srt, *_ = mux.call_args.args
            self.assertEqual([(p.name, title) for p, title in tracks],
                             [("ep01.ru.f5.m4a", "Голоса героев"), ("ep01.ru.vosk.m4a", "Русский (vosk)")])
            self.assertEqual(out, Path(tmp) / "done" / "STT_Season 1" / "ep01.ru.mkv")
            self.assertIsNone(srt)
            ep.config["mix"]["subtitles"] = True  # the voiced lines of the default track's engine
            voiced = ep.out / "voiceover.f5" / "export" / "display.srt"
            voiced.parent.mkdir(parents=True)
            voiced.write_text("")
            with mock.patch("stt.mix.mux_tracks") as mux:
                ep.mux()
            self.assertEqual(mux.call_args.args[3], voiced)
            ep.config["mix"]["tracks"] = "separate"
            with mock.patch("stt.mix.mux_tracks") as mux:
                ep.mux()
            self.assertEqual([p.name for p, _ in mux.call_args.args[1]], ["ep01.ru.vosk.m4a"])
            self.assertEqual(mux.call_args.args[2].name, "ep01.ru.vosk.mkv")
            args.start, args.duration = 600.0, 60.0  # a test piece stays in the episode folder
            self.assertEqual(ep.finished_video().parent, ep.out)

    def test_llm_model_per_stage(self) -> None:
        import argparse

        args = argparse.Namespace(season_glossary=None, force=None, until="dub")
        one = Episode(Path("ep.mkv"), Path("out"), args, {"llm": {"model": "openai/gpt-6.1-sol"}})
        self.assertEqual((one.llm("translate"), one.llm("stress")),
                         ("openrouter:openai/gpt-6.1-sol", "openrouter:openai/gpt-6.1-sol"))
        mixed = Episode(Path("ep.mkv"), Path("out"), args,
                        {"llm": {"model": "openai/gpt-6-luna", "stages": {"stress": ["a", "b", "c"]}}})
        self.assertEqual((mixed.llm("fuse"), mixed.llm("stress")),
                         ("openrouter:openai/gpt-6-luna", ["openrouter:a", "openrouter:b", "openrouter:c"]))
        legacy = Episode(Path("ep.mkv"), Path("out"), args, {"llm_model": "x/y"})
        self.assertEqual(legacy.llm("glossary"), "openrouter:x/y")
        routed = Episode(Path("ep.mkv"), Path("out"), args, {"llm": {
            "model": "openai/gpt-6-luna", "opencode": ["openai/*"], "router": "orcarouter",
            "stages": {"stress": ["openai/gpt-6.1-sol", "anthropic/claude-sonnet-5.5", "openrouter:google/g"]}}})
        self.assertEqual(routed.llm("translate"), "opencode+orcarouter:openai/gpt-6-luna")
        self.assertEqual(routed.llm("stress"), ["opencode+orcarouter:openai/gpt-6.1-sol",
                                                "orcarouter:anthropic/claude-sonnet-5.5", "openrouter:google/g"])
        chain = Episode(Path("ep.mkv"), Path("out"), args, {"llm": {
            "model": "openai/gpt-6-luna", "opencode": ["openai/*"], "router": ["orcarouter", "openrouter"]}})
        self.assertEqual(chain.llm("fuse"), "opencode+orcarouter+openrouter:openai/gpt-6-luna")
        self.assertEqual(chain.llm("stress"), "opencode+orcarouter+openrouter:openai/gpt-6-luna")
        no_fallback = Episode(Path("ep.mkv"), Path("out"), args, {"llm": {
            "model": "openai/gpt-6-luna", "opencode": ["openai/*"], "fallback": False}})
        self.assertEqual(no_fallback.llm("fuse"), "opencode:openai/gpt-6-luna")

    def test_opencode_falls_back_to_the_paid_router(self) -> None:
        from unittest import mock

        import stt.llm

        class _Down:
            def generate(self, *args: object, **kwargs: object) -> str:
                raise stt.llm.OpenCodeError("HTTP 401: login expired")

        posted: list[tuple[str, str]] = []

        def post(self: stt.llm.OpenRouterClient, body: dict, router: str = "openrouter") -> dict:
            posted.append((router, body["model"]))
            return {"model": "gpt-6.1-sol", "choices": [{"message": {"content": '{"ok": 1}'}}], "usage": {}}

        with mock.patch.object(stt.llm, "shared_server", return_value=_Down()), \
                mock.patch.object(stt.llm.OpenRouterClient, "_post", post), \
                mock.patch.object(stt.llm, "load_api_key", return_value="k"), \
                mock.patch.object(stt.llm.time, "sleep"):
            client = stt.llm.OpenRouterClient("opencode+orcarouter:openai/gpt-6.1-sol")
            self.assertEqual(client.chat_json("s", "u"), {"ok": 1})
            self.assertEqual(client.chat_json("s", "u"), {"ok": 1})
        self.assertEqual(posted, [("orcarouter", "openai/gpt-6.1-sol")] * 2)
        self.assertEqual(client.transport, "orcarouter")

    def test_opencode_model_needs_no_api_key(self) -> None:
        from unittest import mock

        import stt.llm

        class _Server:
            prompts: list[tuple[str, str, str | None]] = []

            def generate(self, prompt: str, model: str, variant: str | None = None, timeout: float = 0) -> str:
                self.prompts.append((prompt, model, variant))
                return '```json\n{"items": [1]}\n```'

        server = _Server()
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}), \
                mock.patch.object(stt.llm, "load_api_key", side_effect=AssertionError("no key needed")), \
                mock.patch.object(stt.llm, "shared_server", return_value=server):
            client = stt.llm.OpenRouterClient("opencode:openai/gpt-6.1-sol", reasoning="high")
            self.assertEqual(client.chat_json("SYSTEM", "USER"), {"items": [1]})
        self.assertEqual(server.prompts, [("SYSTEM\n\nUSER", "openai/gpt-6.1-sol", "high")])
        self.assertEqual((client.usage.calls, client.usage.cost), (1, 0.0))

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

    def test_short_pause_inside_sentence_does_not_split(self) -> None:
        spec = [(0.0, 0.3, "I"), (0.4, 0.8, "went"), (1.6, 2.0, "home."), (3.5, 3.8, "Bye.")]
        words = [Word(s, e, t, speaker="A") for s, e, t in spec]
        cues = build_cues(words, CueOptions())
        self.assertEqual([c.text for c in cues], ["I went home.", "Bye."])

    def test_long_pause_inside_sentence_splits(self) -> None:
        # the Russian line of one cue would be said in the pause, ahead of the English second half
        spec = [(0.0, 0.3, "I"), (0.4, 0.6, "went"), (0.7, 0.8, "back"), (2.0, 2.2, "to"), (2.3, 2.5, "the"),
                (2.6, 2.9, "camp."), (4.0, 4.3, "Bye.")]
        words = [Word(s, e, t, speaker="A") for s, e, t in spec]
        cues = build_cues(words, CueOptions())
        self.assertEqual([c.text for c in cues], ["I went back", "to the camp.", "Bye."])

    def test_pause_does_not_split_off_a_fragment(self) -> None:
        # "Google | it." voiced as two cues became "Проверь" ... "в Гугле" seconds apart
        spec = [(0.0, 0.4, "Google"), (3.1, 3.3, "it."), (5.0, 5.4, "Colby"), (5.5, 5.8, "ready"), (5.9, 6.0, "to"),
                (6.1, 6.3, "make"), (7.6, 7.7, "a"), (11.1, 11.5, "move.")]
        words = [Word(s, e, t, speaker="A") for s, e, t in spec]
        cues = build_cues(words, CueOptions())
        self.assertEqual([c.text for c in cues], ["Google it.", "Colby ready to make a move."])
        self.assertEqual([c.text for c in build_cues(words, CueOptions(fragment_words=0))],
                         ["Google", "it.", "Colby ready to make", "a", "move."])

    def test_pause_after_a_comma_still_splits(self) -> None:
        # the host's pause before the name of the vote stays in the voice-over
        spec = [(0.0, 0.3, "First"), (0.4, 0.8, "vote,"), (2.5, 3.0, "Sugar.")]
        words = [Word(s, e, t, speaker="A") for s, e, t in spec]
        self.assertEqual([c.text for c in build_cues(words, CueOptions())], ["First vote,", "Sugar."])

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


class CrowdTest(unittest.TestCase):
    def test_groups_voices_and_render(self) -> None:
        import numpy as np

        from stt.crowd import groups, pick_voices, render

        skipped = [{"start": 1000, "end": 1500, "text": ["Давай!"], "gender": "male", "speaker": "S01"},
                   {"start": 2500, "end": 3200, "text": ["Давай, давай!"], "gender": "female", "speaker": "S02"},
                   {"start": 9000, "end": 9400, "text": ["Боже мой!"], "gender": "male", "speaker": "S03"}]
        found = groups(skipped)
        self.assertEqual([(g["start"], g["end"], g["text"], g["crowd"]) for g in found],
                         [(1000, 3200, "Давай, давай!", True), (9000, 9400, "Боже мой!", False)])
        from stt.crowd import is_chant

        self.assertTrue(is_chant("Давай, давай!") and is_chant("Вперёд! Вперёд!"))
        self.assertFalse(is_chant("Да, да, да.") or is_chant("Ага, ага.") or is_chant("Боже мой!"))
        pool = {"female": [49, 38], "male": [56, 12]}
        self.assertEqual(pick_voices(["male", "female", "male"], pool), [56, 49, 12])
        found[0]["voices"], found[1]["voices"] = [56, 49], [56]
        clip = np.ones(100, dtype=np.float32)
        track = render(found, {(0, 56): clip, (0, 49): clip, (1, 56): clip}, 1000, 10.0, offset_ms=50, gain_db=0.0)
        self.assertEqual((track[1000], track[1060], track[9000]), (1.0, 2.0, 1.0))


def test_local_config_on_top(tmp_path: Path) -> None:
    import json

    from stt.run import load_config

    (tmp_path / "c.json").write_text(json.dumps({"llm": {"model": "luna", "stages": {"a": 1}}, "mode": "fast"}))
    assert load_config(tmp_path / "c.json")["llm"]["model"] == "luna"
    (tmp_path / "c.local.json").write_text(json.dumps({"llm": {"model": "sol", "opencode": ["openai/*"]}}))
    assert load_config(tmp_path / "c.json") == {"llm": {"model": "sol", "stages": {"a": 1}, "opencode": ["openai/*"]},
                                                 "mode": "fast"}


class CombineTest(unittest.TestCase):
    def test_streams_of_a_combined_video(self) -> None:
        from unittest import mock

        from stt.mix import streams

        info = """Input #0, matroska,webm, from 'ep.ru.mkv':
  Stream #0:0: Video: h264 (Main), yuv420p, 960x720 (default)
  Stream #0:1(rus): Audio: aac (LC), 48000 Hz, stereo, fltp (default)
      Metadata:
        title           : Многоголосый закадровый
  Stream #0:2: Audio: aac (LC), 48000 Hz, stereo, fltp
      Metadata:
        title           : Original
  Stream #0:3(rus): Subtitle: subrip (srt)
"""
        with mock.patch("subprocess.run", return_value=mock.Mock(stderr=info)):
            self.assertEqual(streams(Path("ep.ru.mkv")), [("video", ""), ("audio", "Многоголосый закадровый"),
                                                          ("audio", "Original"), ("subtitle", "")])


class WorkerRetryTest(unittest.TestCase):
    def test_out_of_memory_stage_is_retried_with_a_smaller_batch(self) -> None:
        from unittest import mock

        from stt import worker

        with tempfile.TemporaryDirectory() as tmp:
            out, log_path = Path(tmp) / "asr.json", Path(tmp) / "asr.log"
            seen: list[list[str]] = []

            def fake_run(env: str, module: str, args: list[str], path: Path) -> tuple[int, str]:
                seen.append(list(args))
                if len(seen) == 1:
                    path.write_text("RuntimeError: CUDA failed with error out of memory\n", encoding="utf-8")
                    return 1, "out of memory"
                out.write_text('{"segments": []}', encoding="utf-8")
                return 0, ""

            with mock.patch.object(worker, "run_module", fake_run), mock.patch.object(worker.time, "sleep"):
                data = worker.call("subs_env", ["asr", "--model", "whisper-large-v3", "--out", str(out)], out,
                                   log_path)
            self.assertEqual(data, {"segments": []})
            self.assertEqual(seen[1][-2:], ["--batch-size", "8"])  # half of Whisper's 16
            self.assertTrue(log_path.with_suffix(".oom1.log").exists())

    def test_other_failures_are_not_retried(self) -> None:
        from unittest import mock

        from stt import worker

        with tempfile.TemporaryDirectory() as tmp:
            out, log_path = Path(tmp) / "ocr.json", Path(tmp) / "ocr.log"
            calls = mock.Mock(side_effect=lambda *a: (log_path.write_text("ValueError: bad frame"), (1, "bad"))[1])
            with mock.patch.object(worker, "run_module", calls), self.assertRaises(RuntimeError):
                worker.call("subs_env", ["ocr", "--out", str(out)], out, log_path)
            self.assertEqual(calls.call_count, 1)

    def test_voiceover_module_is_started_again_after_out_of_memory(self) -> None:
        from unittest import mock

        from stt import run

        with tempfile.TemporaryDirectory() as tmp:
            episode = mock.Mock(out=Path(tmp), media=Path("ep.mkv"))
            codes = iter([1, 0])

            def fake_run(cmd: list[str], **kwargs: object) -> mock.Mock:
                kwargs["stdout"].write("torch.OutOfMemoryError: CUDA out of memory.\n")  # type: ignore[attr-defined]
                return mock.Mock(returncode=next(codes))

            with mock.patch.object(run.subprocess, "run", fake_run), mock.patch.object(run.time, "sleep"), \
                    mock.patch.object(run, "dotenv", return_value={}):
                run.Episode._module(episode, "voiceover.f5", "voiceover.pipeline", [])
            self.assertTrue((Path(tmp) / "logs" / "voiceover.f5.oom1.log").exists())


class SubscriptionCheckTest(unittest.TestCase):
    CONFIG = {"llm": {"model": "openai/gpt-6.1-sol", "opencode": ["openai/*"], "router": ["orcarouter", "openrouter"],
                      "stages": {"stress": ["openai/gpt-6.1-sol", "anthropic/claude-sonnet-5.5",
                                            "openrouter:google/gemini-4-argon"], "fuse": "opencode:openai/gpt-6-luna"}}}

    def test_models_through_the_subscription(self) -> None:
        from stt.run import subscription_models

        self.assertEqual(subscription_models(self.CONFIG), ["openai/gpt-6.1-sol", "openai/gpt-6-luna"])
        self.assertEqual(subscription_models({"llm": {"model": "openai/gpt-6-luna"}}), [])  # no opencode patterns

    def test_failed_login_warns_or_stops(self) -> None:
        from unittest import mock

        from stt import run

        with mock.patch("stt.llm.subscription_error", return_value=None):
            self.assertTrue(run.check_subscription(self.CONFIG))
        with mock.patch("stt.llm.subscription_error", return_value="token is expired"), \
                self.assertLogs("stt.run", "WARNING") as logs:
            self.assertFalse(run.check_subscription(self.CONFIG))
        self.assertIn("opencode auth login", logs.output[0])
        strict = {"llm": {**self.CONFIG["llm"], "fallback": False}}
        with mock.patch("stt.llm.subscription_error", return_value="token is expired"), self.assertRaises(SystemExit):
            run.check_subscription(strict)


class TurnEvalTest(unittest.TestCase):
    SRT = """1
00:00:01,000 --> 00:00:02,000
>> jeff: Welcome back to the
game everybody.

2
00:00:02,000 --> 00:00:03,000
Game everybody.
>> thank you jeff.

3
00:00:03,000 --> 00:00:04,000
>> thank you jeff.
>> good luck.
"""

    def test_roll_up_lines_are_dropped_and_turns_scored(self) -> None:
        from stt.turn_eval import caption_lines, evaluate

        with tempfile.TemporaryDirectory() as tmp:
            srt = Path(tmp) / "ep.en.srt"
            srt.write_text(self.SRT, encoding="utf-8")
            self.assertEqual(caption_lines(srt), [">> jeff: Welcome back to the", "game everybody.",
                                                  ">> thank you jeff.", ">> good luck."])

            def words(speakers: str) -> list[dict[str, str]]:
                text = "Welcome back to the game everybody. Thank you, Jeff. Good luck.".split()
                return [{"text": t, "speaker": s} for t, s in zip(text, speakers.split())]

            right = evaluate(srt, words("A A A A A A B B B A A"))
            self.assertEqual((right["caption_turns"], right["recall"], right["precision"]), (2, 1.0, 1.0))
            merged = evaluate(srt, words("A A A A A A A A A A A"))
            self.assertEqual((merged["recall"], merged["our_changes"]), (0.0, 0))
            wrong = evaluate(srt, words("A A B B B B B B B B B"))
            self.assertEqual((wrong["recall"], wrong["precision"]), (0.0, 0.0))


class DiarizenTest(unittest.TestCase):
    def test_solo_stretches_leave_out_other_voices(self) -> None:
        from stt.voice_bank import solo_stretches

        turns = [{"start": 0.0, "end": 10.0, "speaker": "A"}, {"start": 4.0, "end": 5.0, "speaker": "B"},
                 {"start": 9.0, "end": 12.0, "speaker": "B"}]
        self.assertEqual(solo_stretches(turns), [("A", 0.0, 4.0), ("A", 5.0, 9.0), ("B", 10.0, 12.0)])

    def test_cluster_prints_are_unit_means_of_solo_stretches(self) -> None:
        from stt.voice_bank import cluster_prints

        audio = np.zeros(16_000 * 20, dtype=np.float32)
        audio[16_000 * 10:] = 1.0  # B's stretch is loud, A's silent: the fake embedder tells them apart
        embed = lambda piece, rate: np.array([1.0, 0.0]) if piece.mean() < 0.5 else np.array([0.0, 1.0])
        prints = cluster_prints([{"start": 0.0, "end": 8.0, "speaker": "A"},
                                 {"start": 10.0, "end": 20.0, "speaker": "B"}], audio, embed)
        self.assertEqual(prints, {"A": [1.0, 0.0], "B": [0.0, 1.0]})

    def test_diarizen_backend_returns_turns(self) -> None:
        from unittest import mock

        from stt.diarize import DIARIZERS, DiarizenDiarizer

        segment = mock.Mock(start=1.0, end=2.5)
        annotation = mock.Mock(itertracks=lambda yield_label: [(segment, None, 0)])
        diarizer = DiarizenDiarizer(DIARIZERS["diarizen-v2"])
        diarizer._pipeline = mock.Mock(return_value=annotation)
        result = diarizer.diarize(np.zeros(16_000, dtype=np.float32), max_speakers=25)
        self.assertEqual([(t.start, t.end, t.speaker) for t in result.turns], [(1.0, 2.5, "0")])
        self.assertEqual((result.exclusive, result.embeddings, diarizer._pipeline.max_speakers), ([], {}, 25))


class DiarizerFallbackTest(unittest.TestCase):
    def test_missing_diarizen_env_falls_back_to_pyannote(self) -> None:
        from unittest import mock

        from stt import run, transcribe

        with tempfile.TemporaryDirectory() as tmp:
            episode = mock.Mock(media=Path("ep.mkv"), out=Path(tmp), season=Path(tmp) / "s.glossary.json",
                                args=mock.Mock(mode="fast", start=None, duration=None),
                                config={"diarizer": "diarizen-v2", "modes": {"fast": {"asr": "whisper-large-v3"}},
                                        "ocr": {"enabled": False}})
            with mock.patch.object(run, "REPO_ROOT", Path(tmp)), mock.patch.object(transcribe, "main") as main, \
                    self.assertLogs("stt.run", "WARNING"):
                run.Episode.transcribe(episode)
            argv = main.call_args[0][0]
            self.assertEqual(argv[argv.index("--diarizer") + 1], "pyannote-community-1")


class LineGenderTest(unittest.TestCase):
    def test_unknown_speakers_get_a_vote_then_lines_their_own_voice(self) -> None:
        from stt.line_gender import apply, decide

        speakers = [{"id": "S01", "gender": "male"}, {"id": "S02", "gender": "unknown"},
                    {"id": "S03", "gender": "unknown"}]
        cues = [{"index": i, "speaker": sid, "gender": g} for i, (sid, g) in enumerate(
            [("S01", "male"), ("S02", "unknown"), ("S02", "unknown"), ("S02", "unknown"), ("S02", "unknown"),
             ("S03", "unknown"), ("S03", "unknown")])]
        p_female = {0: 0.99, 1: 0.97, 2: 0.95, 3: 0.92, 4: 0.5, 5: 0.03, 6: 0.98}
        voted, lines = decide(cues, speakers, p_female)
        self.assertEqual(voted, {"S02": "female"})  # 3 confident lines, all female
        self.assertEqual(lines, {1: "female", 2: "female", 3: "female", 5: "male", 6: "female"})
        self.assertEqual(apply(cues, speakers, voted, lines), 6)  # line 4 (unclear) takes S02's vote
        self.assertEqual([c["gender"] for c in cues], ["male", "female", "female", "female", "female", "male", "female"])
        self.assertEqual(speakers[1]["gender_source"], "line vote")  # S01 stays male although its line sounds female
