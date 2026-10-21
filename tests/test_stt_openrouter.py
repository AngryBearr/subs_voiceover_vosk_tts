"""Offline HTTP/credential fixtures only; no real network or user credential reads."""
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import wave

import numpy as np

from utils import stt_openrouter as api
from utils import stt_pipeline as stt

KEY = "sk-or-offline-fixture-only"


class Response:
    def __init__(self, value, generation=None):
        self.value = json.dumps(value).encode()
        self.headers = {"X-Generation-Id": generation} if generation else {}

    def read(self, count):
        return self.value[:count]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class OpenRouterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/tmp/opencode")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.network = patch("urllib.request.urlopen", side_effect=AssertionError("real network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.audio = np.ones(16000, dtype=np.float32) * .1
        self.wav = api.pcm_wav(self.audio)

    def test_key_environment_priority_safe_synthetic_dotenv_only(self):
        env = self.root / ".env"
        env.write_text("OTHER_SECRET=$(touch bad)\nexport OPENROUTER_API_KEY='fixture-file-key' # safe\n")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": KEY}, clear=True):
            self.assertEqual(api.load_key(env), KEY)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(api.load_key(env), "fixture-file-key")
            env.write_text('OPENROUTER_API_KEY="fixture-double" # comment\n')
            self.assertEqual(api.load_key(env), "fixture-double")
            env.write_text('OPENROUTER_API_KEY=$(echo dangerous)\n')
            with self.assertRaises(ValueError) as caught:
                api.load_key(env)
            self.assertNotIn("dangerous", str(caught.exception))
            env.write_text("OPENROUTER_API_KEY=one\nOPENROUTER_API_KEY=two\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                api.load_key(env)
            with self.assertRaisesRegex(ValueError, "Set OPENROUTER"):
                api.load_key(self.root / "absent")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=True):
            with self.assertRaises(ValueError):
                api.load_key(env)

    def test_pcm_complete_wav_and_model_specific_payload(self):
        with wave.open(io.BytesIO(self.wav)) as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()), (1, 2, 16000, 16000))
        for model in (api.MAI, api.MUSE):
            body = api.request_body(model, self.wav)
            self.assertEqual(base64.b64decode(body["input_audio"]["data"]), self.wav)
            self.assertFalse(body["input_audio"]["data"].startswith("data:"))
        mai = api.request_body(api.MAI, self.wav)
        self.assertEqual(mai["timestamp_granularities"], ["segment", "word"])
        self.assertTrue(mai["provider"]["options"]["azure"]["diarization"]["enabled"])
        muse = api.request_body(api.MUSE, self.wav)
        self.assertNotIn("timestamp_granularities", muse)
        self.assertEqual(muse["provider"]["options"]["meta"]["mode"], "DIARIZATION")
        with self.assertRaises(ValueError):
            api.pcm_wav(np.array([float("nan")]))
        with self.assertRaises(ValueError):
            api.pcm_wav(np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            api.pcm_wav(np.zeros(181 * 16000))
        with self.assertRaises(ValueError):
            api.wav_seconds(self.wav[:-10])
        with patch.object(api, "MAX_BYTES", 10):
            with self.assertRaises(ValueError):
                api.request_body(api.MAI, self.wav)

    def test_mai_native_segments_offsets_overlap_unknown_gender(self):
        raw = {"text": "Same! Same!", "segments": [
            {"start": .1, "end": .8, "text": "Same!", "speaker": 5},
            {"start": .2, "end": .9, "text": "Same!", "speaker": 0}],
            "words": [{"start": .1, "end": .8, "word": "Same!", "speaker": 5}]}
        parsed = api.parse_reply(raw, 1)
        records, speakers = api.records_for(parsed, "request_0001", offset=60)
        self.assertEqual([r["start"] for r in records], [60100, 60200])
        self.assertEqual([r["end"] for r in records], [60800, 60900])
        self.assertEqual([r["speaker_id"] for r in records], ["SPEAKER_00", "SPEAKER_01"])
        self.assertTrue(all(r["overlap"] and r["needs_review"] for r in records))
        self.assertTrue(all(r["gender"] == "unknown" and isinstance(r["text"], list) for r in records))
        self.assertEqual(speakers["SPEAKER_00"]["api_speaker_label"], 5)
        self.assertEqual(speakers["SPEAKER_00"]["api_request_scope"], "request_0001")
        self.assertEqual(parsed.timing_origin, "native_segments_seconds")

    def test_four_experimental_bodies_complete_wav_and_exact_options(self):
        for model, options in api.EXPERIMENT_OPTIONS.items():
            with self.subTest(model=model):
                body = api.request_body(model, self.wav)
                self.assertEqual(base64.b64decode(body["input_audio"]["data"]), self.wav)
                self.assertEqual(body, {"model": model,
                    "input_audio": {"data": base64.b64encode(self.wav).decode(), "format": "wav"},
                    "response_format": "verbose_json", "timestamp_granularities": ["segment", "word"], **options})
                self.assertNotIn(model, api.RATES_PER_HOUR)
                if model == "deepgram/nova-3":
                    self.assertNotIn("diarize", body)
                    self.assertEqual(body["provider"], {"options": {"deepgram": {"diarize": True}}})
                else:
                    self.assertTrue(body["diarize"])
                    self.assertNotIn("provider", body)
        too_long = api.pcm_wav(np.zeros(30 * 16000 + 1))
        for model in api.EXPERIMENT_OPTIONS:
            with self.assertRaisesRegex(ValueError, "<=30"):
                api.request_body(model, too_long)

    def test_experimental_fixed_reserve_budget_short_clip_and_unknown_guards(self):
        for model in api.EXPERIMENT_OPTIONS:
            ledger = api.BudgetLedger(1)
            self.assertEqual(ledger.check(model, 7.855), .05)
            self.assertEqual(ledger.check(model, 30), .05)
            for seconds in (0, 30.0001, float("nan"), True):
                with self.assertRaises(ValueError):
                    ledger.check(model, seconds)
            with self.assertRaises(ValueError):
                api.BudgetLedger(1.01).check(model, 1)
            entry = ledger.reserve(model, 1)
            self.assertEqual(entry["preflight_cost_kind"], "fixed_experiment_reservation_not_provider_rate")
            self.assertEqual(entry["preflight_cost_usd"], .05)
            with self.assertRaisesRegex(ValueError, "unknown billing"):
                ledger.check(model, 1)
            entry["actual_cost_usd"] = .96
            with self.assertRaisesRegex(ValueError, "limit"):
                ledger.check(model, 1)
        core = api.BudgetLedger(1)
        self.assertEqual(core.max_requests, 18)
        self.assertAlmostEqual(core.check(api.MAI, 30), 30 * .1 / 3600 * 10)
        self.assertNotIn("preflight_cost_kind", core.reserve(api.MAI, 1))

    def test_four_experimental_client_wire_bodies_without_claimed_diarization(self):
        requests = []
        def opener(request, timeout):
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.full_url, api.ENDPOINT)
            requests.append(json.loads(request.data))
            return Response({"text": "Synthetic response; no requested labels or timing returned", "usage": {"cost": .001}})
        ledger = api.BudgetLedger(1)
        client = api.Client(KEY, ledger, opener)
        for model in api.EXPERIMENT_OPTIONS:
            raw, entry = client.call(model, self.wav)
            self.assertEqual(requests[-1], api.request_body(model, self.wav))
            self.assertEqual(api.records_for(api.parse_reply(raw, 1), entry["request_scope"]), ([], {}))
            self.assertEqual(entry["preflight_cost_kind"], "fixed_experiment_reservation_not_provider_rate")
        self.assertEqual(len(requests), 4)
        self.assertEqual(ledger.total, .004)

    def test_scribe_word_text_speaker_id_and_non_speech_skip_before_validation(self):
        raw = {"text": "Hello [laughter] world", "words": [
            {"type": "word", "text": "Hello", "start": .1, "end": .4, "speaker_id": "speaker_0"},
            {"type": "spacing", "text": " ", "start": "bad", "speaker_id": {}},
            {"type": "audio_event", "text": "[laughter]", "end": float("nan"), "speaker_id": {}},
            {"type": "word", "text": "world", "start": .5, "end": .9, "speaker_id": 0}]}
        before = json.dumps(raw)
        parsed = api.parse_reply(raw, 1)
        rows, speakers = api.records_for(parsed, "request_0073")
        self.assertEqual(json.dumps(raw), before)
        self.assertEqual(parsed.text, raw["text"])
        self.assertEqual(parsed.diagnostics["skipped_non_speech_items"], 2)
        self.assertEqual([r["text"] for r in rows], [["Hello"], ["world"]])
        self.assertEqual([m["api_speaker_label"] for m in speakers.values()], ["speaker_0", 0])
        self.assertTrue(all(r["gender"] == "unknown" for r in rows))
        for kind in ("unrecognized", None, True):
            with self.assertRaisesRegex(ValueError, "word item type"):
                api.parse_reply({"words": [{"type": kind, "text": "spoken", "start": .1, "end": .4}]}, 1)

    def test_grok_word_text_labels_zero_and_existing_field_priority(self):
        parsed = api.parse_reply({"words": [
            {"text": "zero", "start": .1, "end": .4, "speaker": 0, "speaker_label": "other", "speaker_id": "ignored"},
            {"word": "preferred", "text": "ignored", "start": .5, "end": .9, "speaker_label": "B", "speaker_id": "other"}]}, 1)
        self.assertEqual([(w["text"], w["label"]) for w in parsed.words], [("zero", 0), ("preferred", "B")])
        parsed = api.parse_reply({"text": "Gemini synthetic text-only fixture; not claimed live response"}, 1)
        self.assertEqual(api.records_for(parsed, "request_0079"), ([], {}))
        self.assertEqual(parsed.timing_origin, "unavailable")

    def test_native_meta_candidate_milliseconds_and_words_only(self):
        # Contract fixture, NOT a claimed observed live Meta response.
        parsed = api.parse_reply({"turns": [{"startMs": 200, "endMs": 800,
            "transcript": "Hello", "speaker": "speakerA"}]}, 1)
        rows, meta = api.records_for(parsed, "request_0002", offset=14)
        self.assertEqual((rows[0]["start"], rows[0]["end"]), (14200, 14800))
        self.assertEqual(rows[0]["timestamp_origin"], "native_turns_ms")
        self.assertEqual(meta["SPEAKER_00"]["api_speaker_label"], "speakerA")
        parsed = api.parse_reply({"words": [{"start": .2, "end": .8, "word": "hello"}]}, 1)
        rows, meta = api.records_for(parsed, "request_0003")
        self.assertEqual(rows[0]["speaker_id"], stt.UNKNOWN)
        self.assertEqual(rows[0]["start"], 200)

    def test_documented_string_speaker_label_and_zero_numeric_priority(self):
        parsed = api.parse_reply({"segments": [
            {"start": .1, "end": .4, "text": "one", "speaker_label": "A"},
            {"start": .5, "end": .8, "text": "two", "speaker": 0, "speaker_label": "other"}]}, 1)
        rows, meta = api.records_for(parsed, "request_0001")
        self.assertEqual(meta[rows[0]["speaker_id"]]["api_speaker_label"], "A")
        self.assertEqual(meta[rows[1]["speaker_id"]]["api_speaker_label"], 0)
        ledger = api.BudgetLedger(1)
        api.Client(KEY, ledger, lambda *a, **k: Response({"text": "x", "usage": {"cost": .001, "costUSD": .002}})).call(api.MAI, self.wav)
        self.assertEqual(ledger.total, .001)

    def test_selected_collection_only_and_native_turns_precedence(self):
        segment = {"start": .1, "end": .8, "text": "native segment", "speaker": 0}
        parsed = api.parse_reply({"segments": [segment], "words": [{"word": "bad", "start": "invalid"}]}, 1)
        self.assertEqual(parsed.words, [])
        self.assertEqual(parsed.diagnostics["selected_collection"], "segments")
        self.assertEqual(parsed.diagnostics["ignored_collections"], ["words"])
        raw = {"turns": [{"startMs": 200, "endMs": 700, "transcript": "native turn", "speaker": "A"}],
               "segments": "unused malformed representation", "words": [{"word": 42}]}
        parsed = api.parse_reply(raw, 1)
        rows, _ = api.records_for(parsed, "request_0001", offset=14)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["start"], rows[0]["end"], rows[0]["text"]), (14200, 14700, ["native turn"]))
        self.assertEqual(parsed.diagnostics["ignored_collections"], ["segments", "words"])
        with self.assertRaises(ValueError):
            api.parse_reply({"turns": [{"transcript": "nonblank", "startMs": "bad"}], "segments": [segment]}, 1)

    def test_blank_items_skip_invalid_times_labels_but_nonstring_text_fails(self):
        for name, text_key in (("segments", "text"), ("turns", "transcript"), ("words", "word")):
            raw = {name: [{text_key: " \t\n", "start": "bad", "startMs": float("nan"), "speaker": {}}]}
            parsed = api.parse_reply(raw, 1)
            self.assertEqual((parsed.segments, parsed.words, parsed.timing_origin), ([], [], "unavailable"))
            self.assertEqual(parsed.diagnostics["skipped_blank_items"], 1)
            with self.assertRaisesRegex(ValueError, "text must be a string"):
                api.parse_reply({name: [{text_key: None}]}, 1)
        parsed = api.parse_reply({"segments": [{"text": " "},
            {"start": .2, "end": .8, "text": "Kept", "speaker": 0}]}, 1)
        self.assertEqual(len(parsed.segments), 1)
        self.assertEqual(parsed.text, "Kept")

    def test_bounded_api_endpoint_clipping_preserves_raw_and_annotates(self):
        raw = {"segments": [{"start": .2, "end": 1.2, "text": "hello", "speaker": 1.0}]}
        before = json.dumps(raw)
        parsed = api.parse_reply(raw, 1)
        rows, meta = api.records_for(parsed, "request_0001", offset=60)
        self.assertEqual(json.dumps(raw), before)
        self.assertEqual((rows[0]["start"], rows[0]["end"]), (60200, 61000))
        self.assertTrue(rows[0]["timestamp_clipped"])
        self.assertEqual(meta["SPEAKER_00"]["api_speaker_label"], 1)
        self.assertIs(type(meta["SPEAKER_00"]["api_speaker_label"]), int)
        self.assertEqual(parsed.diagnostics["clipped_endpoints"][0]["raw_end_seconds"], 1.2)
        self.assertAlmostEqual(parsed.diagnostics["clipped_endpoints"][0]["overshoot_seconds"], .2)
        self.assertEqual(api.parse_reply({"segments": [{"start": .2, "end": 1.25, "text": "x"}]}, 1).segments[0]["end"], 1)
        for start, end in ((.2, 1.251), (1.01, 1.05), (1, 1), (0, 0), (-.001, .5),
                           (.2, -.1), (.8, .2), (float("nan"), .9), (.2, float("inf")), (10**400, 10**400)):
            with self.assertRaises(ValueError):
                api.parse_reply({"segments": [{"start": start, "end": end, "text": "x"}]}, 1)
        parsed = api.parse_reply({"turns": [{"startMs": 100, "endMs": 1200, "transcript": "x"}]}, 1)
        self.assertEqual(parsed.segments[0]["end"], 1)
        reply = {**raw, "usage": {"costUSD": .001}, "words": [{"word": "unused", "start": "bad"}]}
        with patch.object(api, "load_key", return_value=KEY), \
             patch.object(stt, "decode_media", return_value=self.audio), \
             patch("urllib.request.urlopen", return_value=Response(reply)):
            rows = self.pipeline(self.root / "clipped")
        report = json.loads((self.root / "clipped.api.json").read_text())
        self.assertEqual(report["requests"][0]["response"]["segments"][0]["end"], 1.2)
        self.assertEqual(report["requests"][0]["timing_normalization"]["ignored_collections"], ["words"])
        self.assertEqual(report["requests"][0]["timing_normalization"]["clipped_endpoints"][0]["raw_end_seconds"], 1.2)
        self.assertTrue(rows[0]["timestamp_clipped"])
        self.assertEqual(rows[0]["end"], 61000)

    def test_speaker_canonicalization_and_cross_speaker_overlap_only(self):
        for labels in ((1, 1.0), (" A ", "A"), (None, None)):
            parsed = api.parse_reply({"segments": [
                {"start": .1, "end": .8, "text": "x", "speaker": labels[0]},
                {"start": .2, "end": .9, "text": "y", "speaker": labels[1]}]}, 1)
            rows, _ = api.records_for(parsed, "request_0001")
            self.assertTrue(all(not row["overlap"] for row in rows))
            self.assertEqual(rows[0]["speaker_id"], rows[1]["speaker_id"])
        for label in (1.5, -1., float("nan"), float("inf"), True, "  "):
            with self.assertRaisesRegex(ValueError, "speaker label"):
                api.parse_reply({"segments": [{"start": .1, "end": .8, "text": "x", "speaker": label}]}, 1)
        parsed = api.parse_reply({"segments": [
            {"start": .1, "end": .8, "text": "Same", "speaker": 1},
            {"start": .2, "end": .9, "text": "Same", "speaker": 2}]}, 1)
        self.assertTrue(all(r["overlap"] for r in api.records_for(parsed, "request_0001")[0]))

    def test_nested_sensitive_names_and_case_variant_fixture_echoes(self):
        names = ("secret", "access_token", "api-key", "apikey", "OPENROUTER_API_KEY", "Authorization",
                 "input_audio", "audio-data", "client_secret", "refreshToken")
        raw = {"nested": [{name: "different fixture sensitive value" for name in names}],
               "text": "Bearer " + KEY.upper() + " / " + KEY.swapcase(), "nonsecret": "preserved"}
        result = api.sanitized(raw, KEY)
        self.assertTrue(all(v == "[redacted]" for v in result["nested"][0].values()))
        self.assertEqual(result["text"], "Bearer [redacted] / [redacted]")
        self.assertEqual(result["nonsecret"], "preserved")

    def test_malformed_times_labels_and_text_only_do_not_fabricate_alignment(self):
        for row in ({"start": "0", "end": 1, "text": "x"},
                    {"start": True, "end": 1, "text": "x"},
                    {"start": .9, "end": .2, "text": "x"},
                    {"start": 0, "end": float("nan"), "text": "x"},
                    {"start": 0, "end": 2, "text": "x"},
                    {"start": 0, "end": 1, "text": "x", "speaker": True}):
            with self.assertRaises(ValueError):
                api.parse_reply({"segments": [row]}, 1)
        parsed = api.parse_reply({"text": "There are three voices"}, 1)
        self.assertEqual(parsed.timing_origin, "unavailable")
        self.assertEqual(api.records_for(parsed, "request_0001"), ([], {}))

    def test_client_auth_endpoint_usage_cost_and_secret_redaction(self):
        calls = []

        def opener(request, timeout):
            calls.append(request)
            self.assertEqual(timeout, 70)
            self.assertEqual(request.full_url, api.ENDPOINT)
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.get_header("Authorization"), "Bearer " + KEY)
            return Response({"text": "echo " + KEY, "usage": {"seconds": 1, "costUSD": .002},
                             "input_audio": {"data": "never store this"},
                             "echo": base64.b64encode(self.wav).decode()}, "gen-test")

        ledger = api.BudgetLedger(1, self.root / "ledger.json")
        raw, entry = api.Client(KEY, ledger, opener).call(api.MAI, self.wav)
        self.assertEqual(len(calls), 1)
        self.assertEqual(ledger.total, .002)
        self.assertEqual(entry["generation_id"], "gen-test")
        self.assertNotIn(KEY, json.dumps(raw))
        self.assertNotIn(KEY, (self.root / "ledger.json").read_text())
        self.assertNotIn("never store this", json.dumps(raw))
        self.assertNotIn(base64.b64encode(self.wav).decode(), json.dumps(raw))

    def test_generation_read_only_lookup_and_unknown_billing_blocks_more_posts(self):
        methods = []

        def opener(request, timeout):
            methods.append(request.get_method())
            return Response({"text": "x"}, "gen-id") if request.get_method() == "POST" else Response(
                {"data": {"total_cost": .003, "api_type": "stt", "provider_name": "fixture"}})

        ledger = api.BudgetLedger(1)
        api.Client(KEY, ledger, opener).call(api.MUSE, self.wav)
        self.assertEqual(methods, ["POST", "GET"])
        self.assertEqual(ledger.total, .003)
        count = []
        ledger = api.BudgetLedger(1)
        client = api.Client(KEY, ledger, lambda *a, **k: count.append(1) or Response({"text": "x"}))
        client.call(api.MAI, self.wav)
        self.assertIsNone(ledger.total)
        with self.assertRaisesRegex(ValueError, "unknown billing"):
            client.call(api.MUSE, self.wav)
        self.assertEqual(count, [1])

    def test_http_errors_timeout_and_malformed_reply_never_retry_post(self):
        for error in [urllib.error.HTTPError(api.ENDPOINT, code, "fixture", {}, io.BytesIO(KEY.encode()))
                      for code in (400, 401, 429)] + [TimeoutError(KEY), urllib.error.URLError(KEY)]:
            with self.subTest(error=type(error).__name__):
                calls = []

                def opener(*args, **kwargs):
                    calls.append(1)
                    raise error

                ledger = api.BudgetLedger(1)
                with self.assertRaises(ValueError) as caught:
                    api.Client(KEY, ledger, opener).call(api.MAI, self.wav)
                self.assertEqual(len(calls), 1)
                self.assertNotIn(KEY, str(caught.exception))
                self.assertIsNone(ledger.total)
        response = Response({})
        response.value = b"not-json"
        ledger = api.BudgetLedger(1)
        with self.assertRaisesRegex(ValueError, "Invalid OpenRouter reply"):
            api.Client(KEY, ledger, lambda *a, **k: response).call(api.MAI, self.wav)
        self.assertIsNone(ledger.total)

    def test_http_error_evidence_redacted_and_generation_cost_counted_once(self):
        methods = []

        def opener(request, timeout):
            methods.append(request.get_method())
            if request.get_method() == "POST":
                error = {"error": {"message": KEY.upper(), "api-key": KEY,
                                  "input_audio": {"data": "fixture inline audio"}}}
                raise urllib.error.HTTPError(api.ENDPOINT, 400, "fixture", {"X-Generation-Id": "error-gen"},
                                             io.BytesIO(json.dumps(error).encode()))
            return Response({"data": {"total_cost": .001}})

        ledger = api.BudgetLedger(1, self.root / "ledger.json")
        with self.assertRaisesRegex(ValueError, "HTTP 400.*known"):
            api.Client(KEY, ledger, opener).call(api.MUSE, self.wav)
        self.assertEqual(methods, ["POST", "GET"])
        self.assertEqual(ledger.total, .001)
        self.assertEqual(ledger.entries[0]["status"], "http_error_cost_known")
        self.assertEqual(ledger.entries[0]["response"]["error"]["message"], "[redacted]")
        self.assertEqual(ledger.entries[0]["response"]["error"]["input_audio"], "[redacted]")
        self.assertNotIn(KEY, (self.root / "ledger.json").read_text())
        self.assertEqual(api.BudgetLedger(1, self.root / "ledger.json").total, .001)

    def test_budget_shared_models_persistence_pending_and_actual_limit(self):
        ledger = api.BudgetLedger(.005, self.root / "ledger.json")
        self.assertAlmostEqual(ledger.check(api.MUSE, 1), .001)
        entry = ledger.reserve(api.MAI, 1)
        entry["actual_cost_usd"] = .005
        ledger.save()
        shared = api.BudgetLedger(.005, self.root / "ledger.json")
        with self.assertRaisesRegex(ValueError, "limit"):
            shared.reserve(api.MUSE, 1)
        ledger = api.BudgetLedger(1, self.root / "pending.json")
        ledger.reserve(api.MAI, 1)
        with self.assertRaisesRegex(ValueError, "unknown"):
            api.BudgetLedger(1, self.root / "pending.json").check(api.MUSE, 1)
        with self.assertRaises(ValueError):
            api.BudgetLedger(float("nan"))
        first = api.BudgetLedger(.005, self.root / "stale.json")
        stale = api.BudgetLedger(.005, self.root / "stale.json")
        entry = first.reserve(api.MAI, 1)
        with self.assertRaisesRegex(ValueError, "unknown"):
            stale.reserve(api.MUSE, 1)
        entry["actual_cost_usd"] = .005
        first.save()
        with self.assertRaisesRegex(ValueError, "limit"):
            stale.reserve(api.MUSE, 1)

    def test_ledger_atomic_updates_keep_private_mode_and_refuse_read_only_snapshot(self):
        path = self.root / "private.json"
        ledger = api.BudgetLedger(1, path)
        ledger.save()
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        entry = ledger.reserve(api.MAI, 1)
        entry["actual_cost_usd"] = .001
        ledger.save()
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        original = path.read_bytes()
        path.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "Read-only"):
            ledger.save()
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(path.with_suffix(".pending.json").exists())

    def test_budget_get_retry_only_and_unknown_cost_persists(self):
        methods = []

        def opener(request, timeout):
            methods.append(request.get_method())
            if request.get_method() == "POST":
                return Response({"text": "x"}, "unresolved")
            raise TimeoutError("synthetic metadata unavailable")

        ledger = api.BudgetLedger(1, self.root / "unknown.json")
        api.Client(KEY, ledger, opener).call(api.MUSE, self.wav)
        self.assertEqual(methods, ["POST", "GET", "GET"])
        self.assertIsNone(ledger.total)
        self.assertIsNone(json.loads((self.root / "unknown.json").read_text())["actual_total_usd"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            api.BudgetLedger(1, self.root / "unknown.json").reserve(api.MAI, 1)
        ledger = api.BudgetLedger(1)
        ledger.entries = [{"actual_cost_usd": 0}] * 18
        with self.assertRaisesRegex(ValueError, "limit"):
            ledger.reserve(api.MAI, 1)

    def test_request_cap_default_and_explicit_boundary_preserve_history(self):
        for cap, explicit in ((18, False), (72, True)):
            with self.subTest(cap=cap):
                path = self.root / f"cap_{cap}.json"
                options = {"max_requests": cap} if explicit else {}
                ledger = api.BudgetLedger(1, path, **options)
                self.assertEqual(ledger.max_requests, cap)
                ledger.entries = [{"actual_cost_usd": .0001, "status": "received"}
                                  for _ in range(cap - 1)]
                ledger.save()
                history = json.loads(json.dumps(ledger.entries))
                final = ledger.reserve(api.MAI, 1)
                self.assertEqual(len(ledger.entries), cap)
                self.assertEqual(final["request_scope"], f"request_{cap:04d}")
                final["actual_cost_usd"] = .0001
                ledger.save()
                with self.assertRaisesRegex(ValueError, "limit"):
                    api.BudgetLedger(1, path, **options).reserve(api.MAI, 1)
                self.assertEqual(json.loads(path.read_text())["requests"][:-1], history)

    def test_request_cap_invalid_values_fail_closed(self):
        for value in (True, False, 0, -1, 18.0, "72", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                api.BudgetLedger(1, max_requests=value)

    def test_high_request_cap_does_not_bypass_unknown_billing_or_usd_budget(self):
        path = self.root / "high_cap.json"
        ledger = api.BudgetLedger(1, path, max_requests=72)
        entry = ledger.reserve(api.MAI, 1)
        with self.assertRaisesRegex(ValueError, "unknown billing"):
            api.BudgetLedger(1, path, max_requests=72).reserve(api.MAI, 1)
        entry["actual_cost_usd"] = 1
        ledger.save()
        with self.assertRaisesRegex(ValueError, "limit"):
            api.BudgetLedger(1, path, max_requests=72).reserve(api.MAI, 1)

    def pipeline(self, prefix, **kwargs):
        input_path = self.root / "input.wav"
        input_path.write_bytes(self.wav)
        return stt.run_pipeline(input_path, prefix, preset="api", duration=1, start=60, **kwargs)

    def test_direct_core_dispatch_does_not_load_pixit_and_exports_real_times(self):
        with patch.object(api, "load_key", return_value=KEY), \
             patch.object(stt, "decode_media", return_value=self.audio), \
             patch.object(stt, "separate", side_effect=AssertionError("no local model")), \
             patch("urllib.request.urlopen", return_value=Response({"text": "hello", "segments": [
                 {"start": .2, "end": .8, "text": "hello", "speaker": 0}], "usage": {"costUSD": .001}})):
            rows = self.pipeline(self.root / "direct")
        self.assertEqual(rows[0]["start"], 60200)
        self.assertTrue((self.root / "direct.srt").exists())
        self.assertEqual(json.loads((self.root / "direct.api.json").read_text())["actual_total_usd"], .001)

    def test_plain_text_persists_transcript_and_raw_but_no_fake_srt(self):
        with patch.object(api, "load_key", return_value=KEY), \
             patch.object(stt, "decode_media", return_value=self.audio), \
             patch("urllib.request.urlopen", return_value=Response({"text": "fixture transcript", "usage": {"costUSD": .001}})):
            with self.assertRaisesRegex(ValueError, "JSON/SRT unavailable"):
                self.pipeline(self.root / "plain", api_model=api.MUSE)
        self.assertEqual((self.root / "plain.transcript.txt").read_text(), "fixture transcript\n")
        self.assertTrue((self.root / "plain.api.json").exists())
        self.assertFalse((self.root / "plain.srt").exists())
        self.assertFalse((self.root / "plain.json").exists())

    def test_direct_parse_failure_preserves_paid_text_raw_and_known_cost(self):
        reply = {"text": "Returned paid text", "segments": [{"start": .1, "end": 2, "text": "invalid timing"}],
                 "usage": {"costUSD": .002}}
        with patch.object(api, "load_key", return_value=KEY), \
             patch.object(stt, "decode_media", return_value=self.audio), \
             patch("urllib.request.urlopen", return_value=Response(reply)) as network:
            with self.assertRaisesRegex(ValueError, "250ms"):
                self.pipeline(self.root / "invalid-direct")
        self.assertEqual(network.call_count, 1)
        self.assertEqual((self.root / "invalid-direct.transcript.txt").read_text(), "Returned paid text\n")
        report = json.loads((self.root / "invalid-direct.api.json").read_text())
        self.assertEqual(report["requests"][0]["response"], reply)
        self.assertIn("250ms", report["requests"][0]["timing_error"])
        self.assertEqual(report["actual_total_usd"], .002)
        self.assertFalse(report["timing_export_available"])
        self.assertEqual(json.loads((self.root / ".stt_api_budget.json").read_text())["actual_total_usd"], .002)
        self.assertFalse((self.root / "invalid-direct.json").exists())
        self.assertFalse((self.root / "invalid-direct.srt").exists())

    def test_hybrid_parse_failure_preserves_completed_source_text_without_partial_export(self):
        from types import SimpleNamespace
        mapping = {f"SPEAKER_{i:02d}": {"source_channel": i, "source_start": 0, "separation_status": "estimated"}
                   for i in range(2)}
        replies = [Response({"turns": [{"startMs": 100, "endMs": 800, "transcript": "Completed source"}],
                             "usage": {"costUSD": .001}}),
                   Response({"text": "Second paid source", "segments": [{"start": .2, "end": 2, "text": "bad"}],
                             "usage": {"costUSD": .001}})]
        with patch.object(api, "load_key", return_value=KEY), \
             patch.object(stt, "decode_media", return_value=self.audio), \
             patch.object(stt, "separate", return_value=([stt.Turn(0, 1, s) for s in mapping],
                 SimpleNamespace(data=np.column_stack([self.audio, self.audio])), mapping)), \
             patch.object(stt, "estimate_genders", return_value={s: {"gender": "unknown"} for s in mapping}), \
             patch("urllib.request.urlopen", side_effect=replies) as network:
            with self.assertRaisesRegex(ValueError, "250ms"):
                self.pipeline(self.root / "invalid-hybrid", api_mode="hybrid")
        self.assertEqual(network.call_count, 2)
        transcript = (self.root / "invalid-hybrid.transcript.txt").read_text().splitlines()
        self.assertEqual(transcript, ["[SPEAKER_00 60.000s] Completed source", "[SPEAKER_01 60.000s] Second paid source"])
        report = json.loads((self.root / "invalid-hybrid.api.json").read_text())
        self.assertEqual(report["actual_total_usd"], .002)
        self.assertEqual(report["requests"][0]["timing_normalization"]["selected_collection"], "turns")
        self.assertIn("250ms", report["requests"][1]["timing_error"])
        self.assertFalse(report["timing_export_available"])
        self.assertFalse((self.root / "invalid-hybrid.srt").exists())
        self.assertFalse((self.root / "invalid-hybrid.json").exists())

    def test_hybrid_keeps_real_local_source_identity_and_offset(self):
        from types import SimpleNamespace
        sources = SimpleNamespace(data=(self.audio * 100)[:, None])
        mapping = {"SPEAKER_02": {"source_channel": 0, "source_start": 0, "separation_status": "estimated"}}
        metadata = {"SPEAKER_02": {"gender": "unknown", "gender_confidence": None, "gender_source": "unknown"}}
        with patch.object(api, "load_key", return_value=KEY), \
             patch.object(stt, "decode_media", return_value=self.audio), \
             patch.object(stt, "separate", return_value=([stt.Turn(0, 1, "SPEAKER_02")], sources, mapping)), \
             patch.object(stt, "estimate_genders", return_value=metadata), \
             patch("urllib.request.urlopen", return_value=Response({"segments": [
                 {"start": .1, "end": .9, "text": "hello", "speaker": "unrelated-cloud-label"}], "usage": {"costUSD": .001}})):
            rows = self.pipeline(self.root / "hybrid", api_mode="hybrid")
        self.assertEqual((rows[0]["speaker_id"], rows[0]["start"]), ("SPEAKER_02", 60100))
        meta = json.loads((self.root / "hybrid.speakers.json").read_text())
        self.assertEqual(meta["SPEAKER_02"]["identity_origin"], "local_pixit_source")
        evidence = json.loads((self.root / "hybrid.api.json").read_text())["requests"][0]
        self.assertAlmostEqual(evidence["pcm_transport_gain"], .1)
        self.assertEqual(evidence["local_source_speaker"], "SPEAKER_02")

    def test_hybrid_text_only_and_unknown_cost_stop_before_second_upload(self):
        from types import SimpleNamespace
        sources = SimpleNamespace(data=np.column_stack([self.audio, self.audio]))
        mapping = {f"SPEAKER_{i:02d}": {"source_channel": i, "source_start": 0, "separation_status": "estimated"}
                   for i in range(2)}
        meta = {s: {"gender": "unknown"} for s in mapping}
        with patch.object(api, "load_key", return_value=KEY), \
             patch.object(stt, "decode_media", return_value=self.audio), \
             patch.object(stt, "separate", return_value=([stt.Turn(0, 1, s) for s in mapping], sources, mapping)), \
             patch.object(stt, "estimate_genders", return_value=meta), \
             patch("urllib.request.urlopen", return_value=Response({"text": "x"})) as network:
            with self.assertRaisesRegex(ValueError, "unknown billing"):
                self.pipeline(self.root / "stopped", api_mode="hybrid")
        self.assertEqual(network.call_count, 1)
        self.assertFalse((self.root / "stopped.srt").exists())
        self.assertIn("x", (self.root / "stopped.transcript.txt").read_text())
        report = json.loads((self.root / "stopped.api.json").read_text())
        self.assertIsNone(report["actual_total_usd"])
        self.assertFalse(report["timing_export_available"])

    def test_missing_key_and_invalid_options_collisions_fail_before_decode(self):
        with patch.object(api, "load_key", side_effect=ValueError("Set OPENROUTER_API_KEY")), \
             patch.object(stt, "decode_media", side_effect=AssertionError("no decode/upload")):
            with self.assertRaisesRegex(ValueError, "Set OPENROUTER"):
                self.pipeline(self.root / "missing")
        input_path = self.root / "audio.wav"
        input_path.write_bytes(self.wav)
        for duration, model, mode, budget in [(None, api.MAI, "direct", 1), (181, api.MAI, "direct", 1),
            (1, "unknown", "direct", 1), (1, api.MAI, "wrong", 1), (1, api.MAI, "direct", 0)]:
            with self.assertRaises(ValueError):
                api.validate_api(input_path, self.root / "out", duration, model, mode, budget, None)
        with self.assertRaisesRegex(ValueError, "collision"):
            api.validate_api(input_path, self.root / ".env", 1, api.MAI, "direct", 1, None)
        (self.root / "exists.srt").write_text("preserved")
        with self.assertRaisesRegex(ValueError, "existing"):
            api.validate_api(input_path, self.root / "exists", 1, api.MAI, "direct", 1, None)
        self.assertEqual((self.root / "exists.srt").read_text(), "preserved")
        (self.root / "symlink.api.json").symlink_to(self.root / "untouched")
        with self.assertRaisesRegex(ValueError, "collision"):
            api.validate_api(input_path, self.root / "symlink", 1, api.MAI, "direct", 1, None)

    def test_cli_flags_thread_into_core_without_network(self):
        with patch.object(sys, "argv", ["stt", "input.wav", "-o", "result", "--preset", "api",
            "--duration", "30", "--api-model", api.MUSE, "--api-mode", "hybrid", "--api-budget", "1"]), \
             patch.object(stt, "run_pipeline", return_value=[]) as run:
            stt.main()
        options = run.call_args.kwargs
        self.assertEqual((options["preset"], options["api_model"], options["api_mode"], options["api_budget"]),
                         ("api", api.MUSE, "hybrid", 1.))

    def test_default_local_preset_never_dispatches_api_and_help(self):
        with patch.object(api, "run_api", side_effect=AssertionError("no implicit upload")), \
             patch.object(stt, "decode_media", return_value=np.zeros(16000)):
            input_path = self.root / "silent.wav"
            input_path.write_bytes(self.wav)
            self.assertEqual(stt.run_pipeline(input_path, self.root / "local"), [])
        result = subprocess.run([sys.executable, "-m", "utils.stt_pipeline", "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        for flag in ("--api-model", "--api-mode", "--api-budget"):
            self.assertIn(flag, result.stdout)


if __name__ == "__main__":
    unittest.main()
