"""STT pure-logic checks plus opt-in, local-media runtime smoke checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from utils import stt_pipeline as stt
from utils import stt_refiner as refiner
from utils.srt_to_json import parse_srt_file


class FakeAnnotation:
    def labels(self):
        return [0, 1, 2]

    def itertracks(self, yield_label=False):
        return iter([(NS(start=1., end=4.), 0, 2), (NS(start=2., end=5.), 1, 0),
                     (NS(start=2., end=5.), 2, 1)])


class STTTests(unittest.TestCase):
    def test_triple_matching_permutation_collision_ambiguity_and_duplicates(self):
        similarity = [[1., .1, .1], [.1, 1., .1], [.1, .1, 1.]]
        order, margin = refiner.match_sources([[.1, .1, .9], [.9, .1, .1], [.1, .9, .1]], similarity)
        self.assertEqual(order, (2, 0, 1))
        self.assertGreater(margin, .08)
        self.assertEqual(refiner.match_sources([[.9, .8, .1], [.92, .2, .1], [.1, .1, .95]], similarity)[0], (1, 0, 2))
        for scores, sims in [([[.6] * 3] * 3, similarity),
                             ([[.3, .1, .1], [.1, .3, .1], [.1, .1, .3]], similarity),
                             ([[.9, .1, .1], [.1, .9, .1], [.1, .1, .9]], [[1.] * 3] * 3),
                             ([[float("nan")] * 3] * 3, similarity)]:
            with self.assertRaisesRegex(ValueError, "needs review"):
                refiner.match_sources(scores, sims)

    def test_triple_sweep_bounded_windows_and_other_voice_context(self):
        disjoint = [stt.Turn(i * 5, i * 5 + 4, f"SPEAKER_{i:02d}") for i in range(3)]
        self.assertEqual(refiner.triple_windows(disjoint, 20), [])
        turns = [stt.Turn(2, 29, f"SPEAKER_{i:02d}") for i in range(3)]
        turns += [stt.Turn(0, 1.5, "SPEAKER_03"), stt.Turn(29.5, 31, "SPEAKER_04")]
        windows = refiner.triple_windows(turns, 31)
        self.assertEqual(windows[0]["write_start"], 1.5)
        self.assertEqual(windows[-1]["write_end"], 29.5)
        for window in windows:
            self.assertLessEqual(window["end"] - window["start"], 10)
            self.assertLessEqual(window["core_end"] - window["core_start"], 8)
            self.assertGreaterEqual(window["core_end"] - window["core_start"], 2)
            self.assertEqual(window["speakers"], ("SPEAKER_00", "SPEAKER_01", "SPEAKER_02"))
        self.assertTrue(all(a["core_end"] == b["core_start"] for a, b in zip(windows, windows[1:])))
        with self.assertRaisesRegex(ValueError, "more than three"):
            refiner.triple_windows(turns + [stt.Turn(10, 12, "SPEAKER_05")], 31)
        interrupted = [stt.Turn(0, 3, s) for s in ("A", "B")] + [stt.Turn(3.2, 6.2, s) for s in ("A", "B")]
        interrupted += [stt.Turn(0, 6.2, "C"), stt.Turn(3, 3.2, "D")]
        windows = refiner.triple_windows(interrupted, 6.2)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["end"], 3)
        self.assertEqual(windows[1]["start"], 3.2)

    def test_one_two_sources_no_refiner_download_or_embedding(self):
        for count in (1, 2):
            turns = [stt.Turn(0, 3, f"SPEAKER_{i:02d}") for i in range(count)]
            sources, mapping = NS(data="unchanged"), {t.speaker: {"source_channel": i} for i, t in enumerate(turns)}
            with patch.object(refiner, "load_sepformer", side_effect=AssertionError("no model download")):
                refiner.refine_triples([0.] * 48000, turns, sources, mapping, None, self.root)
            self.assertEqual(sources.data, "unchanged")
            self.assertTrue(all("refinement_status" not in x for x in mapping.values()))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.media = self.root / "input.wav"
        self.media.write_bytes(b"preserve")

    def test_source_channels_follow_labels_not_chronological_ids(self):
        sources = NS(data=NS(shape=(80000, 3)), sliding_window=NS(start=0, step=1 / 16000, duration=2 / 16000))
        turns, mapping = stt.map_sources(FakeAnnotation(), sources, 5)
        self.assertEqual(mapping["SPEAKER_00"]["source_channel"], 2)
        self.assertEqual(mapping["SPEAKER_01"]["source_channel"], 0)
        self.assertEqual(mapping["SPEAKER_02"]["source_channel"], 1)
        self.assertEqual(len(turns), 3)
        for shape, step in [((80000, 2), 1 / 16000), ((100, 3), 1 / 16000), ((80000, 3), 1 / 8000)]:
            sources.data.shape, sources.sliding_window.step = shape, step
            with self.assertRaises(ValueError):
                stt.map_sources(FakeAnnotation(), sources, 5)

    def test_twelve_clusters_string_order_and_sparse_columns(self):
        labels = sorted(range(12), key=str)
        self.assertEqual(labels, [0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9])
        annotation = NS(labels=lambda: labels, itertracks=lambda **kw: iter(
            (NS(start=i, end=i + .5), i, label) for i, label in enumerate([10, 0, 1, 11, 2, 3, 4, 5, 6, 7, 8, 9])
        ))
        sources = NS(data=NS(shape=(192000, 12)), sliding_window=NS(start=0, step=1 / 16000, duration=2 / 16000))
        _, mapping = stt.map_sources(annotation, sources, 12)
        self.assertEqual(mapping["SPEAKER_00"]["source_channel"], 10)
        self.assertEqual(mapping["SPEAKER_00"]["pipeline_label"], "SPEAKER_02")
        self.assertEqual(mapping["SPEAKER_01"]["source_channel"], 0)
        sparse = NS(labels=lambda: [0, 10, 2], itertracks=lambda **kw: iter(
            [(NS(start=0, end=1), 0, 10), (NS(start=1, end=2), 1, 2), (NS(start=2, end=3), 2, 0)]
        ))
        _, mapping = stt.map_sources(sparse, sources, 12)
        self.assertEqual([m["source_channel"] for m in mapping.values()], [10, 2, 0])
        sources.data.shape = (192000, 3)
        with self.assertRaisesRegex(ValueError, "3 active labels, 3 source channels.*--num-speakers"):
            stt.map_sources(sparse, sources, 12)
        overcounted = NS(labels=lambda: [0, 1, 2, 3], itertracks=lambda **kw: iter([]))
        with self.assertRaisesRegex(ValueError, "4 active labels, 3 source channels"):
            stt.map_sources(overcounted, sources, 12)

    def test_bounded_parakeet_and_whisper_boundary_timing(self):
        tokens = stt.token_words([" Last", " word"], [.95, 1.079999999], 1, 60, "SPEAKER_00")
        self.assertEqual(tokens[-1], stt.Word(61, 61, "word", "SPEAKER_00"))
        raw = [NS(start=.9, end=1.01, word=" last"), NS(start=1.02, end=1.07, word=" word")]
        words = stt.whisper_words(raw, 1, 60, "SPEAKER_00")
        self.assertEqual(words[-1], stt.Word(61, 61, "word", "SPEAKER_00"))
        self.assertEqual(words[0].end, 61)
        self.assertEqual(stt.token_words([" Equal"], [1], 1)[0].start, 1)
        for times in [[1.081], [float("inf")], [1.05, 1.02]]:
            with self.assertRaises(ValueError):
                stt.token_words([" word"] * len(times), times, 1)
        for raw in [[NS(start=1.081, end=1.09, word="bad")],
                    [NS(start=.7, end=.6, word="bad")],
                    [NS(start=.8, end=.9, word="a"), NS(start=.7, end=.8, word="b")]]:
            with self.assertRaises(ValueError):
                stt.whisper_words(raw, 1, 0, "SPEAKER_00")

    def test_three_overlapping_cues_keep_lineage_and_identical_phrases(self):
        turns = [stt.Turn(0, 5, f"SPEAKER_{i:02d}") for i in range(3)]
        words = [stt.Word(1, 2, "Hello!", t.speaker) for t in turns]
        metadata = {t.speaker: {"gender": g, "separation_status": "estimated"}
                    for t, g in zip(turns, ["male", "female", "unknown"])}
        records = stt.subtitle_records(words, turns, metadata, 60)
        self.assertEqual(len(records), 3)
        self.assertEqual([r["start"] for r in records], [61000] * 3)
        self.assertEqual([r["end"] for r in records], [62000] * 3)
        self.assertEqual([r["gender"] for r in records], ["male", "female", "unknown"])
        self.assertTrue(all(r["overlap"] and r["needs_review"] for r in records))
        prefix = self.root / "triple"
        stt.write_outputs(prefix, records, metadata)
        parsed = parse_srt_file(str(prefix.with_suffix(".srt")))
        self.assertEqual([r["start"] for r in parsed], [61000] * 3)

    def test_bpe_grouping_and_approximate_timing(self):
        words = stt.token_words([" Hel", "lo", ",", "▁world", " !"], [.1, .2, .3, .5, .7], 1, 2, "SPEAKER_00")
        self.assertEqual([w.text for w in words], ["Hello,", "world!"])
        self.assertAlmostEqual(words[0].start, 2.1)
        self.assertAlmostEqual(words[0].end, 2.5)
        self.assertEqual(words[1].end, 3)
        self.assertEqual(words[1].speaker, "SPEAKER_00")
        quoted = stt.token_words([" He", " said", ' "', " hello", '"', "."], [.1, .2, .3, .4, .5, .6], 1)
        self.assertEqual([w.text for w in quoted], ["He", "said", '"hello".'])
        for tokens, times in [(["x"], []), (["x"], [float("nan")]), (["x"], [-1]),
                              (["x", "y"], [.5, .2]), (["x"], [2])]:
            with self.assertRaises(ValueError):
                stt.token_words(tokens, times, 1)

    def test_chunk_union_and_subtitle_limits(self):
        self.assertEqual(stt.speech_chunks([stt.Turn(0, 50, "SPEAKER_00")], 50), [(0, 28), (28, 50)])
        words = [stt.Word(i * .3, i * .3 + .2, "abcdefgh", "SPEAKER_00") for i in range(50)]
        records = stt.subtitle_records(words, [], {})
        self.assertEqual(" ".join(" ".join(r["text"]) for r in records), " ".join(w.text for w in words))
        self.assertTrue(all(r["duration"] <= 6000 and len(r["text"]) <= 2 for r in records))
        self.assertEqual([r["index"] for r in records], list(range(1, len(records) + 1)))

    def test_gender_conflicts_short_weak_manual_and_clean_evidence(self):
        self.assertEqual(stt.aggregate_gender([(.9, .1)], [3], .8)[0], "male")
        self.assertEqual(stt.aggregate_gender([(.1, .9)], [3], .8)[0], "female")
        for scores, durations in [([(.7, .3)], [3]), ([(.99, .01), (.1, .9)], [10, 3]),
                                  ([(.99, .01)], [2.9]), ([], [])]:
            self.assertEqual(stt.aggregate_gender(scores, durations, .8)[0], "unknown")
        with self.assertRaises(ValueError):
            stt.aggregate_gender([(.9, .1)], [-1], .8)
        turns = [stt.Turn(0, 20, "SPEAKER_00"), stt.Turn(3, 8, "SPEAKER_01")]
        clean = stt.clean_samples(turns, "SPEAKER_00")
        self.assertTrue(all(b <= 3 or a >= 8 for a, b in clean))
        self.assertLessEqual(sum(b - a for a, b in clean), 15)
        with patch.object(stt, "download", side_effect=AssertionError("must not download")):
            meta = stt.estimate_genders(None, turns, self.root, 1, .8,
                                        {"SPEAKER_00": "unknown", "SPEAKER_01": "female"})
        self.assertEqual(meta["SPEAKER_00"]["gender"], "unknown")
        self.assertEqual(meta["SPEAKER_00"]["gender_source"], "manual")
        with self.assertRaisesRegex(ValueError, "available"):
            stt.estimate_genders(None, turns, self.root, 1, .8, {"SPEAKER_09": "male"})
        short = stt.estimate_genders(None, [stt.Turn(0, 1, "SPEAKER_00")], self.root, 1, .8, {})
        self.assertEqual(short["SPEAKER_00"]["gender"], "unknown")

    def test_validation_before_inference_and_source_collisions(self):
        defaults = dict(start=0, duration=None, threads=4, num_speakers=None, gender_threshold=.8)
        for options in [dict(start=-1), dict(start=float("nan")), dict(duration=0), dict(duration=float("inf")),
                        dict(threads=0), dict(threads=33), dict(num_speakers=0), dict(gender_threshold=.5)]:
            with self.assertRaises(ValueError):
                stt.validate_options(self.media, self.root / "out", **(defaults | options))
        with patch.object(stt, "decode_media", side_effect=AssertionError("validate first")):
            with self.assertRaises(ValueError):
                stt.run_pipeline(self.media, self.root / "out", duration=-1)
        source = self.root / "out.json"
        source.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            stt.validate_options(source, self.root / "out", **defaults)
        mapping = self.root / "mapping.json"
        for obj in [[], {"SPEAKER_BAD": "male"}, {"SPEAKER_00": "man"}, {"SPEAKER_00": None}]:
            mapping.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):
                stt.load_overrides(mapping)
        mapping.write_text('{"SPEAKER_00":"unknown"}')
        self.assertEqual(stt.load_overrides(mapping), {"SPEAKER_00": "unknown"})

    def test_unknown_json_srt_and_empty_outputs(self):
        records = stt.subtitle_records([stt.Word(.25, 1.25, "Hello!")], [], {}, 60)
        prefix = self.root / "out"
        stt.write_outputs(prefix, records, {})
        self.assertEqual(records[0]["gender"], "unknown")
        self.assertEqual(records[0]["duration"], 1000)
        self.assertIs(type(records[0]["start"]), int)
        parsed = parse_srt_file(str(prefix.with_suffix(".srt")))
        self.assertEqual(parsed[0]["start"], 60250)
        self.assertEqual(parsed[0]["text"], ["Hello!"])
        stt.write_outputs(prefix, [], {})
        self.assertEqual(json.loads(prefix.with_suffix(".json").read_text()), [])
        self.assertEqual(prefix.with_suffix(".srt").read_text(), "")

    def test_help_without_optional_imports_and_failed_atomic_download(self):
        process = subprocess.run([sys.executable, "-S", "-m", "utils.stt_pipeline", "--help"], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        source, target = self.root / "source", self.root / "target"
        source.write_bytes(b"wrong checksum")
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            stt.download(source.as_uri(), target, sha256="0" * 64)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.glob(".download-*")), [])

    def test_missing_local_auth_is_actionable_without_secret_output(self):
        with patch.dict(sys.modules, {"huggingface_hub": NS(get_token=lambda: None)}):
            with self.assertRaisesRegex(ValueError, "speech-separation-ami-1.0"):
                stt.require_hf_token()

    def test_empty_separation_and_speaker_changes(self):
        empty = NS(labels=lambda: [])
        self.assertEqual(stt.map_sources(empty, None, 3), ([], {}))
        words = [stt.Word(0, 1, "First.", "SPEAKER_00"), stt.Word(1, 2, "Second.", "SPEAKER_01")]
        records = stt.subtitle_records(words, [], {}, 60)
        self.assertEqual([r["speaker_id"] for r in records], ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual([r["start"] for r in records], [60000, 61000])
        with self.assertRaises(ValueError):
            stt.subtitle_records([stt.Word(2, 1, "Bad")], [], {})


@unittest.skipUnless(os.environ.get("STT_RUNTIME_CHECK"), "opt-in installed runtime checks")
class RuntimeChecks(unittest.TestCase):
    def test_refiner_resampling_preserves_odd_length_and_source_origin(self):
        import numpy as np
        import torch
        count = 10001
        time = np.arange(count) / stt.SR
        references = np.stack([np.sin(2 * np.pi * f * time) for f in (100, 300, 500)], axis=1).astype(np.float32)

        class FakeModel:
            def separate_batch(self, mix):
                self.shape = tuple(mix.shape)
                time8 = torch.arange(mix.shape[1]) / 8000
                return torch.stack([torch.sin(2 * torch.pi * f * time8) for f in (100, 300, 500)], dim=-1)[None, :]

        model = FakeModel()
        result = refiner.separated_window(model, references.sum(axis=1))
        self.assertEqual(model.shape, (1, 5001))
        self.assertEqual(result.shape, (count, 3))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(all(np.corrcoef(references[:, i], result[:, i])[0, 1] > .99 for i in range(3)))
        with self.assertRaisesRegex(ValueError, "invalid source shape"):
            refiner.separated_window(NS(separate_batch=lambda mix: torch.zeros(1, mix.shape[1], 2)), references.sum(axis=1))

    def test_refinement_missing_clean_profile_and_invalid_embedding_fail_before_download(self):
        import numpy as np
        turns = [stt.Turn(0, 3, f"SPEAKER_{i:02d}") for i in range(3)]
        sources = NS(data=np.zeros((48000, 3)), sliding_window=NS(start=0))
        with patch.object(refiner, "load_sepformer", side_effect=AssertionError("missing profiles must fail first")):
            with self.assertRaisesRegex(ValueError, "lacks 3s clean original"):
                refiner.refine_triples(np.ones(48000, dtype=np.float32), turns, sources, {}, NS(sample_rate=stt.SR), Path("models/stt"))
        with self.assertRaisesRegex(ValueError, "invalid ECAPA"):
            refiner.embed_profile(lambda _: np.array([[float("nan"), 1.]]), [np.ones(48000, dtype=np.float32)])
        for samples in (np.zeros(48000, dtype=np.float32), np.ones(100, dtype=np.float32)):
            with self.assertRaisesRegex(ValueError, "short/silent"):
                refiner.embed_profile(lambda _: self.fail("no embedding for short/silent audio"), [samples])

    def test_refinement_asr_context_union_no_duplicates_and_sample_offsets(self):
        import numpy as np
        from unittest.mock import Mock
        recognizer = Mock()
        recognizer.with_timestamps.return_value = recognizer
        recognizer.recognize.return_value = NS(text="Same!", tokens=[" Same!"], timestamps=[.2])
        windows = {"SPEAKER_00": [{"start": 5., "end": 6.}] * 2}
        with patch("huggingface_hub.snapshot_download", return_value="unused"), \
             patch("onnx_asr.load_model", return_value=recognizer):
            words = stt.transcribe({"SPEAKER_00": np.ones(10 * stt.SR, dtype=np.float32)},
                                   [stt.Turn(1, 2, "SPEAKER_00")], Path("models/stt"), 4, "fast", "large-v3-turbo",
                                   refinement_windows=windows)
        self.assertEqual(recognizer.recognize.call_count, 2)
        self.assertEqual([w.start for w in words], [1., 5.])
        records = stt.subtitle_records(words, [], {}, offset=60)
        self.assertEqual([r["start"] for r in records], [61000, 65000])
        with patch("huggingface_hub.snapshot_download", return_value="unused"), \
             patch("onnx_asr.load_model", return_value=recognizer):
            recognizer.recognize.reset_mock()
            words = stt.transcribe({"SPEAKER_00": np.ones(22 * stt.SR, dtype=np.float32)},
                                   [stt.Turn(10, 14.312, "SPEAKER_00"), stt.Turn(15.088, 20.136, "SPEAKER_00")],
                                   Path("models/stt"), 4, "fast", "large-v3-turbo",
                                   refinement_windows={"SPEAKER_00": [{"start": 14.392, "end": 20.088}]})
        self.assertEqual(recognizer.recognize.call_count, 2)
        self.assertEqual([round(w.start, 3) for w in words], [10., 14.392])

    def test_refinement_replaces_not_sums_with_sample_origin_and_same_phrase_lineage(self):
        import numpy as np
        data = np.full((48000, 3), 100., dtype=np.float32)
        replacement = np.tile(np.array([[1., 2., 3.]], dtype=np.float32), (16000, 1))
        refiner.replace_window(data, replacement, [2, 0, 1], (2, 0, 1), origin=-1, begin=16000,
                               write_first=16000, write_last=32000, last=True)
        self.assertTrue(np.all(data[:16001] == 100))
        self.assertTrue(np.all(data[16642:31360, 2] == 3))
        self.assertTrue(np.all(data[16642:31360, 0] == 1))
        self.assertTrue(np.all(data[16642:31360, 1] == 2))
        self.assertTrue(np.all(data[32001:] == 100))
        with self.assertRaisesRegex(ValueError, "no valid"):
            refiner.replace_window(data, replacement, [0, 1, 2], (0, 1, 2), origin=0, begin=0,
                                   write_first=60000, write_last=65000, last=True)
        records = stt.subtitle_records([stt.Word(1, 2, "Same!", f"SPEAKER_{i:02d}") for i in range(3)],
                                      [stt.Turn(0, 3, f"SPEAKER_{i:02d}") for i in range(3)], {}, offset=60)
        self.assertEqual(len(records), 3)
        self.assertEqual([r["start"] for r in records], [61000] * 3)
        self.assertTrue(all(r["gender"] == "unknown" and r["overlap"] for r in records))

    def test_local_gate_presence_without_printing_credentials(self):
        try:
            stt.require_hf_token()
        except ValueError:
            print("PixIT gate check: local HF authentication absent; no gated downloads attempted")
        else:
            print("PixIT gate check: local HF authentication present; repository access still needs verification")

    def test_real_numeric_annotation_conversion_before_rename(self):
        import numpy as np
        from pyannote.core import SlidingWindow, SlidingWindowFeature
        from pyannote.audio.pipelines.speech_separation import SpeechSeparation
        activity = np.zeros((36, 12), dtype=np.float32)
        order = [10, 0, 1, 11, 2, 3, 4, 5, 6, 7, 8, 9]
        for i, cluster in enumerate(order):
            activity[i * 3:i * 3 + 3, cluster] = 1
        discrete = SlidingWindowFeature(activity, SlidingWindow(start=0, step=.1, duration=.1))
        raw = SpeechSeparation.to_annotation(discrete, min_duration_on=0, min_duration_off=0)
        self.assertEqual(raw.labels(), sorted(range(12), key=str))
        renamed = raw.rename_labels({label: f"SPEAKER_{i:02d}" for i, label in enumerate(raw.labels())})
        self.assertEqual(next(renamed.itertracks(yield_label=True))[2], "SPEAKER_02")
        sources = NS(data=NS(shape=(57600, 12)), sliding_window=NS(start=0, step=1 / 16000, duration=2 / 16000))
        _, mapping = stt.map_sources(raw, sources, 3.6)
        self.assertEqual(mapping["SPEAKER_00"]["source_channel"], 10)

    def test_local_pyannote_checkpoint_load_with_aligned_lightning(self):
        import torch
        import lightning
        import lightning_fabric
        import pytorch_lightning as pl
        from pyannote.audio import Model
        from pyannote.audio.core.task import Specifications, Problem, Resolution
        from pyannote.audio.models.segmentation.debug import SimpleSegmentationModel
        self.assertEqual((lightning.__version__, lightning_fabric.__version__, pl.__version__), ("2.4.0",) * 3)
        model = SimpleSegmentationModel()
        model.specifications = Specifications(problem=Problem.MULTI_LABEL_CLASSIFICATION,
                                              resolution=Resolution.FRAME, duration=1, classes=["a", "b", "c"])
        model.setup()
        model.eval()
        checkpoint = {"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                      "pytorch-lightning_version": pl.__version__}
        model.on_save_checkpoint(checkpoint)
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as temp:
            path = Path(temp) / "local-debug.ckpt"
            torch.save(checkpoint, path)
            loaded = Model.from_pretrained(path).eval()
            samples = torch.randn(1, 1, 1600)
            with torch.inference_mode():
                self.assertTrue(torch.allclose(model(samples), loaded(samples)))

    def test_complete_pipeline_binding_with_three_fake_sources(self):
        import numpy as np
        turns = [stt.Turn(0, 3, f"SPEAKER_{i:02d}") for i in range(3)]
        source_data = np.tile(np.array([[1., 2., 3.]], dtype=np.float32), (48000, 1))
        mapping = {t.speaker: {"source_channel": channel, "pipeline_label": label,
                              "source_start": 0, "separation_status": "estimated"}
                   for t, channel, label in zip(turns, [2, 0, 1], ["C", "A", "B"])}

        def fake_asr(tracks, *args, **kwargs):
            self.assertEqual([tracks[t.speaker][0] for t in turns], [3, 1, 2])
            return [stt.Word(1, 2, "Hi!", t.speaker) for t in turns]

        with tempfile.TemporaryDirectory() as temp:
            media, prefix = Path(temp) / "input.wav", Path(temp) / "triple"
            media.write_bytes(b"fake decode")
            with patch.object(stt, "decode_media", return_value=np.ones(48000, dtype=np.float32)), \
                 patch.object(stt, "separate", return_value=(turns, NS(data=source_data), mapping)), \
                 patch.object(stt, "transcribe", side_effect=fake_asr):
                records = stt.run_pipeline(media, prefix, start=60)
            self.assertEqual(len(records), 3)
            self.assertEqual([r["start"] for r in records], [61000] * 3)
            self.assertTrue(all(r["gender"] == "unknown" for r in records))
            self.assertTrue(all(r["overlap"] for r in records))

    def test_installed_imports(self):
        import inspect
        import speechbrain
        from speechbrain.inference.interfaces import Pretrained
        import torch
        import torchaudio
        import onnx_asr
        import onnxruntime
        from pyannote.audio import Pipeline
        from pyannote.audio.models.separation.ToTaToNet import ToTaToNet
        self.assertIsNotNone(Pipeline)
        self.assertIsNotNone(ToTaToNet)
        self.assertIsNotNone(onnx_asr)
        self.assertIsNotNone(onnxruntime)
        self.assertEqual(torch.version.cuda, None)
        self.assertEqual(torchaudio.__version__, "2.5.1+cpu")
        self.assertEqual(speechbrain.__version__, "1.0.3")
        self.assertIn("use_auth_token", inspect.signature(Pretrained.from_hparams).parameters)

    def test_local_wavlm_legacy_weight_norm_load(self):
        import torch
        from transformers import WavLMConfig, WavLMModel
        config = WavLMConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                             intermediate_size=16, conv_dim=(8,), conv_stride=(2,), conv_kernel=(3,),
                             num_conv_pos_embedding_groups=2, num_conv_pos_embeddings=4)
        model = WavLMModel(config)
        weights = model.state_dict()
        root = "encoder.pos_conv_embed.conv."
        old_g, old_v = root + "weight_g", root + "weight_v"
        new_g, new_v = root + "parametrizations.weight.original0", root + "parametrizations.weight.original1"
        weights[old_g] = weights.pop(new_g)
        weights[old_v] = weights.pop(new_v)
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as temp:
            config.save_pretrained(temp)
            torch.save(weights, Path(temp) / "pytorch_model.bin")
            loaded, info = WavLMModel.from_pretrained(temp, output_loading_info=True)
        self.assertEqual(info["missing_keys"], [])
        self.assertEqual(info["unexpected_keys"], [])
        self.assertTrue(torch.equal(loaded.state_dict()[new_g], weights[old_g]))
        self.assertTrue(torch.equal(loaded.state_dict()[new_v], weights[old_v]))

    def test_real_silence_outputs(self):
        import wave
        with tempfile.TemporaryDirectory() as temp:
            media, prefix = Path(temp) / "silence.wav", Path(temp) / "empty"
            with wave.open(str(media), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(16000)
                out.writeframes(b"\0\0" * 32000)
            with patch.object(stt, "separate", side_effect=AssertionError("silence must not load models")):
                self.assertEqual(stt.run_pipeline(media, prefix, start=1, duration=1), [])
            self.assertEqual(json.loads(prefix.with_suffix(".json").read_text()), [])
            self.assertEqual(prefix.with_suffix(".srt").read_text(), "")


@unittest.skipUnless(os.environ.get("STT_PUBLIC_SMOKE"), "opt-in public ASR/gender smoke; no separation claims")
class PublicSmoke(unittest.TestCase):
    def test_public_asr_and_gender(self):
        media = Path(os.environ["STT_PUBLIC_SMOKE"])
        preset = os.environ.get("STT_SMOKE_PRESET", "fast")
        duration = 30 if preset == "fast" else 10
        audio = stt.decode_media(media, 60, duration)
        turns = [stt.Turn(0, duration, stt.UNKNOWN)]
        before = time.monotonic()
        words = stt.transcribe({stt.UNKNOWN: audio}, turns, Path("models/stt"), 4, preset, "large-v3-turbo")
        self.assertTrue(words, "Public model returned no speech")
        meta = stt.estimate_genders(audio, turns, Path("models/stt"), 4, .8, {})
        # This excerpt is a mix, not a diarized clean speaker: keep execution scores
        # separate from any person-level estimate and leave exported gender unknown.
        evidence = meta[stt.UNKNOWN]
        meta = {stt.UNKNOWN: {"gender": "unknown", "gender_source": "unknown", "gender_confidence": None,
                             "gender_runtime_check_only": evidence, "separation_status": "not_run"}}
        records = stt.subtitle_records(words, turns, meta, 60)
        self.assertTrue(all(60000 <= r["start"] <= r["end"] <= (60 + duration) * 1000 for r in records))
        prefix = Path(f"/tmp/opencode/stt_smoke/public_{preset}")
        stt.write_outputs(prefix, records, meta)
        self.assertEqual(len(parse_srt_file(str(prefix.with_suffix(".srt")))), len(records))
        print(f"PUBLIC ONLY {preset}: {duration}s media, {time.monotonic() - before:.2f}s incl downloads, "
              f"{len(words)} words, {len(records)} cues; no diarization/separation tested; outputs={prefix}")


if __name__ == "__main__":
    unittest.main()
