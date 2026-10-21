"""Offline episode launcher checks: synthetic media/HTTP only, no real credentials or network."""
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

from utils import stt_episode as ep
from utils import stt_openrouter as api


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/tmp/opencode")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.out, self.path = self.root / "input.mp4", self.root / "episode", self.root / "active.json"
        self.source.write_bytes(b"offline-media-fixture")
        api.BudgetLedger(1, self.path).save()
        self.frames = ep.RATE * 61 + 3
        self.commands = []
        for target, options in (
            ("urllib.request.urlopen", {"side_effect": AssertionError("Network forbidden")}),
            ("utils.stt_episode.api.load_key", {"return_value": "offline-fixture-key"}),
            ("utils.stt_episode.subprocess.run", {"side_effect": self.media}),
            ("utils.stt_episode.api.Client.call", {"autospec": True, "side_effect": self.reply}),
        ):
            mocker = patch(target, **options)
            mocked = mocker.start()
            self.addCleanup(mocker.stop)
            if target.endswith("load_key"):
                self.key = mocked
            if target.endswith("Client.call"):
                self.calls = mocked

    def media(self, command, **kwargs):
        self.commands.append(command)
        if command[0] == "ffprobe":
            value = {"streams": [{"index": 0, "codec_type": "video"}, {"index": 1, "codec_type": "audio",
                     "codec_name": "aac", "sample_rate": "44100", "channels": 2}],
                     "format": {"duration": str(self.frames / ep.RATE)}}
            return subprocess.CompletedProcess(command, 0, json.dumps(value))
        self.assertEqual(command[command.index("-map") + 1], "0:1")
        with wave.open(command[-1], "wb") as audio:
            audio.setparams((1, 2, ep.RATE, 0, "NONE", "not compressed"))
            audio.writeframes(b"\x01\x00" * self.frames)
        return subprocess.CompletedProcess(command, 0)

    def reply(self, client, model, data):
        seconds = api.wav_seconds(data)
        entry = client.ledger.reserve(model, seconds)
        raw = {"text": "Same", "segments": [{"start": .1, "end": min(.8, seconds), "text": "Same", "speaker": 0}],
               "usage": {"seconds": math.ceil(seconds), "cost": math.ceil(seconds) * .1 / 3600}}
        entry.update(status="received", wav_sha256=ep.digest(data), response=raw,
                     generation_id=f"offline_{entry['request_scope']}", actual_cost_usd=raw["usage"]["cost"])
        client.ledger.save()
        return raw, entry

    def run_episode(self, **options):
        return ep.run_episode(self.source, self.out, budget_ledger=self.path, **options)

    def snapshot(self):
        return {str(p.relative_to(self.root)): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in self.root.rglob("*") if p.is_file()}

    def prepared(self):
        self.out.mkdir()
        ledger = api.BudgetLedger(1, self.path, max_requests=72)
        info = ep.probe(self.source)
        return ep.prepare(self.source, self.out, info, ledger), ledger

    def test_plan_only_probes_without_key_decode_network_or_writes(self):
        before = self.snapshot()
        result = self.run_episode()
        self.assertEqual(result["status"], "plan_only")
        self.assertEqual(result["total_chunks"], 2)
        self.assertEqual(result["source"]["selected_audio_stream"]["sample_rate"], "44100")
        self.assertEqual([c[0] for c in self.commands], ["ffprobe"])
        self.key.assert_not_called()
        self.calls.assert_not_called()
        self.assertEqual(self.snapshot(), before)

    def test_upload_requires_existing_external_shared_ledger(self):
        with self.assertRaisesRegex(ValueError, "budget-ledger"):
            ep.run_episode(self.source, self.out, allow_upload=True)
        for ledger in (self.root / "absent.json", self.source, self.out / "active.json"):
            with self.subTest(ledger=ledger), self.assertRaises(ValueError):
                ep.run_episode(self.source, self.out, allow_upload=True, budget_ledger=ledger)
        self.calls.assert_not_called()
        self.key.assert_not_called()

    def test_contiguous_60s_tail_native_export_and_repeat_noop(self):
        result = self.run_episode(allow_upload=True)
        self.assertEqual(result["completed_chunks"], 2)
        plan = ep.read(self.out / "run_plan.json")
        self.assertEqual([(c["start_sample"], c["end_sample"]) for c in plan["chunks"]],
                         [(0, ep.CHUNK_FRAMES), (ep.CHUNK_FRAMES, self.frames)])
        rows = ep.read(self.out / "episode.json")
        self.assertEqual([r["start"] for r in rows], [100, 60100])
        self.assertEqual([r["text"] for r in rows], [["Same"], ["Same"]])
        self.assertEqual([r["speaker_id"] for r in rows], ["CHUNK_0001_SPEAKER_00", "CHUNK_0002_SPEAKER_00"])
        self.assertTrue(all(r["needs_review"] and type(r["end"]) is int and r["gender"] == "unknown" for r in rows))
        before = self.snapshot()
        self.key.reset_mock()
        self.calls.reset_mock()
        self.commands.clear()
        for allow in (False, True):
            self.assertEqual(self.run_episode(allow_upload=allow)["status"], "complete")
        self.key.assert_not_called()
        self.calls.assert_not_called()
        self.assertEqual(self.commands, [])
        self.assertEqual(self.snapshot(), before)

    def test_false_complete_and_corrupt_export_cache_model_are_rejected(self):
        self.run_episode(allow_upload=True)
        files = [self.out / "episode.json", self.out / "chunks/0001/response.json", self.out / "run_plan.json"]
        for path in files:
            original = path.read_bytes()
            data = json.loads(original)
            if path.name == "episode.json":
                data[0]["start"] = 123
            elif path.name == "response.json":
                data["entry"]["wav_sha256"] = "wrong"
            else:
                data["model"] = api.MUSE
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                self.run_episode(allow_upload=True)
            path.write_bytes(original)
        (self.out / "chunks/0002/response.json").unlink()
        with self.assertRaisesRegex(ValueError, "complete flag"):
            self.run_episode(allow_upload=True)
        self.assertEqual(self.calls.call_count, 2)

    def test_source_content_hash_rejects_same_size_and_mtime_replacement(self):
        self.run_episode(allow_upload=True)
        stat = self.source.stat()
        self.source.write_bytes(b"x" * stat.st_size)
        os.utime(self.source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "Source hash"):
            self.run_episode()

    def test_wav_tampering_and_scope_mismatch_block_resume(self):
        plan, _ = self.prepared()
        path = Path(plan["chunks"][0]["wav_path"])
        original = path.read_bytes()
        path.write_bytes(original[:-2] + b"xx")
        with self.assertRaisesRegex(ValueError, "WAV hash"):
            self.run_episode(allow_upload=True)
        path.write_bytes(original)
        plan["chunks"][0]["expected_request_scope"] = "request_0020"
        ep.save(self.out / "run_plan.json", plan)
        with self.assertRaisesRegex(ValueError, "scope"):
            self.run_episode(allow_upload=True)
        self.calls.assert_not_called()

    def test_symlink_and_credential_hardlink_output_collisions(self):
        self.out.mkdir()
        (self.out / "manifest.json").symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.run_episode()
        (self.out / "manifest.json").unlink()
        os.link(self.source, self.out / "episode.json")
        with self.assertRaisesRegex(ValueError, "collision"):
            self.run_episode()
        with self.assertRaisesRegex(ValueError, "Credential"):
            ep.run_episode(self.root / ".env", self.root / "other")
        self.calls.assert_not_called()

    def test_unknown_cost_pending_ledger_and_attempt_marker_never_retry(self):
        plan, ledger = self.prepared()
        ep.save(self.out / "chunks/0001/attempt.json", {"status": "POST_started"})
        with self.assertRaisesRegex(ValueError, "Prior attempt"):
            self.run_episode(allow_upload=True)
        (self.out / "chunks/0001/attempt.json").unlink()
        ledger.reserve(api.MAI, plan["chunks"][0]["seconds"])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.run_episode(allow_upload=True)
        self.path.with_suffix(".pending.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Pending"):
            self.run_episode(allow_upload=True)
        self.calls.assert_not_called()

    def test_recover_returned_ledger_response_without_key_or_post(self):
        plan, ledger = self.prepared()
        client = api.Client("offline-fixture-key", ledger)
        for chunk in plan["chunks"]:
            self.reply(client, api.MAI, Path(chunk["wav_path"]).read_bytes())
            ep.save(ep.chunk_path(self.out, chunk, "attempt.json"), {"status": "POST_started", "chunk": chunk})
        self.assertEqual(self.run_episode(allow_upload=True)["status"], "complete")
        self.key.assert_not_called()
        self.calls.assert_not_called()

    def test_partial_plan_cannot_start_paid_request_without_explicit_ledger(self):
        self.prepared()
        with self.assertRaisesRegex(ValueError, "Explicit --budget-ledger"):
            ep.run_episode(self.source, self.out, allow_upload=True)
        self.key.assert_not_called()
        self.calls.assert_not_called()

    def test_known_cost_lost_reply_is_not_retryable(self):
        plan, ledger = self.prepared()
        chunk = plan["chunks"][0]
        entry = ledger.reserve(api.MAI, chunk["seconds"])
        entry.update(actual_cost_usd=.001, status="response_lost_cost_reconciled", wav_sha256=chunk["wav_sha256"])
        ledger.save()
        with self.assertRaisesRegex(ValueError, "Lost reply"):
            self.run_episode(allow_upload=True)
        self.calls.assert_not_called()

    def test_final_collision_blocks_before_any_paid_call(self):
        self.prepared()
        (self.out / "episode.txt").write_text("user work")
        with self.assertRaisesRegex(ValueError, "collision"):
            self.run_episode(allow_upload=True)
        self.calls.assert_not_called()

    def test_completed_schema_does_not_accept_boolean_or_float_integer_aliases(self):
        self.run_episode(allow_upload=True)
        path = self.out / "episode.json"
        original = ep.read(path)
        for value in (True, 1.0):
            altered = json.loads(json.dumps(original))
            altered[0]["index"] = value
            path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "Completed JSON"):
                self.run_episode(allow_upload=True)
        self.assertEqual(self.calls.call_count, 2)

    def test_nonzero_stream_start_is_rejected_before_decode_or_key(self):
        original = self.media
        def shifted(command, **kwargs):
            result = original(command, **kwargs)
            value = json.loads(result.stdout)
            value["streams"][1]["start_time"] = "1.0"
            return subprocess.CompletedProcess(command, 0, json.dumps(value))
        with patch("utils.stt_episode.subprocess.run", side_effect=shifted):
            with self.assertRaisesRegex(ValueError, "Nonzero audio"):
                self.run_episode(allow_upload=True)
        self.key.assert_not_called()
        self.calls.assert_not_called()

    def test_actual_last_reply_over_budget_and_usage_mismatch_stop(self):
        self.frames = ep.RATE
        original = self.reply
        def over_budget(client, model, data):
            raw, entry = original(client, model, data)
            raw["usage"] = {}
            entry["actual_cost_usd"] = 1.1
            client.ledger.save()
            return raw, entry
        self.calls.side_effect = over_budget
        with self.assertRaisesRegex(ValueError, "Episode stopped"):
            self.run_episode(allow_upload=True)
        self.assertFalse((self.out / "episode.json").exists())
        cache = ep.read(self.out / "chunks/0001/response.json")
        self.assertEqual(cache["entry"]["actual_cost_usd"], 1.1)
        chunk = cache["chunk"]
        entry = cache["entry"]
        entry.update(actual_cost_usd=.5, response={"usage": {"seconds": 1}})
        ledger = api.BudgetLedger(1)
        ledger.entries = [entry]
        with self.assertRaisesRegex(ValueError, "cost mismatch"):
            ep.bound_entry(ledger, {"historical_reservations": 0}, chunk)

    def test_budget_and_explicit_total_cap_fail_before_new_decode_or_key(self):
        for options in ({"budget": 1.01}, {"budget": float("nan")}, {"budget": True}, {"max_requests": True},
                        {"max_requests": 0}, {"max_requests": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_episode(allow_upload=True, **options)
        self.key.assert_not_called()
        self.calls.assert_not_called()
        self.assertFalse(any(c[0] == "ffmpeg" for c in self.commands))
        result = self.run_episode(allow_upload=True, max_requests=2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(api.BudgetLedger(1).max_requests, 18)

    def test_history_is_preserved_and_total_cap_is_not_automatically_raised(self):
        ledger = api.BudgetLedger(1, self.path)
        historical = {"actual_cost_usd": .001, "status": "received", "request_scope": "request_0001"}
        ledger.entries = [historical]
        ledger.save()
        result = self.run_episode(max_requests=2)
        self.assertEqual(result["required_total_request_cap"], 3)
        with self.assertRaisesRegex(ValueError, "max-requests insufficient"):
            self.run_episode(allow_upload=True, max_requests=2)
        self.key.assert_not_called()
        self.run_episode(allow_upload=True, max_requests=3)
        self.assertEqual(api.BudgetLedger(1, self.path).entries[0], historical)
        self.assertEqual(ep.read(self.out / "run_plan.json")["historical_reservations"], 1)
        self.assertEqual(ep.read(self.out / "manifest.json")["max_requests"], 3)

    def test_empty_native_chunk_is_valid_but_untimed_text_stops_with_raw_text(self):
        self.frames = ep.RATE
        original = self.reply
        def reply(client, model, data):
            raw, entry = original(client, model, data)
            raw.update(text="", segments=[])
            client.ledger.save()
            return raw, entry
        self.calls.side_effect = reply
        self.assertEqual(self.run_episode(allow_upload=True)["native_cues"], 0)
        self.assertEqual(ep.read(self.out / "episode.json"), [])
        self.out = self.root / "untimed"
        def untimed(client, model, data):
            raw, entry = original(client, model, data)
            raw.update(text="Text without timing", segments=[])
            client.ledger.save()
            return raw, entry
        self.calls.side_effect = untimed
        with self.assertRaisesRegex(ValueError, "Episode stopped"):
            self.run_episode(allow_upload=True)
        self.assertEqual((self.out / "chunks/0001/transcript.txt").read_text(), "Text without timing\n")
        self.assertFalse((self.out / "episode.srt").exists())
        self.assertEqual(ep.read(self.out / "manifest.json")["status"], "blocked")
        calls = self.calls.call_count
        with self.assertRaises(ValueError):
            self.run_episode(allow_upload=True)
        self.assertEqual(self.calls.call_count, calls)

    def test_failed_transport_preserves_attempt_and_never_retries(self):
        def failure(client, model, data):
            entry = client.ledger.reserve(model, api.wav_seconds(data))
            entry.update(wav_sha256=ep.digest(data), status="transport_error_billing_unknown")
            client.ledger.save()
            raise ValueError("offline timeout")
        self.calls.side_effect = failure
        with self.assertRaisesRegex(ValueError, "Episode stopped"):
            self.run_episode(allow_upload=True)
        self.assertTrue((self.out / "chunks/0001/response.json").is_file())
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.run_episode(allow_upload=True)
        self.assertEqual(self.calls.call_count, 1)

    def test_cli_defaults_plan_and_nonempty_unrecognized_directory_is_preserved(self):
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            ep.main([str(self.source), "--output-dir", str(self.out), "--budget-ledger", str(self.path)])
        self.assertEqual(json.loads(stdout.getvalue())["status"], "plan_only")
        self.out.mkdir()
        (self.out / "user.txt").write_text("preserve")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Nonempty"):
            self.run_episode(allow_upload=True)
        self.assertEqual(self.snapshot(), before)
        self.calls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
