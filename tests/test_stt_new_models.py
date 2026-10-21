"""Synthetic offline runner checks only; no real credentials, requests or local model runs."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from temp import stt_new_models_run as runner
from utils import stt_episode as ep
from utils import stt_openrouter as api


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/tmp/opencode")
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.out, self.ledger = root / "experiment", root / "active.json"
        self.out.mkdir()
        (self.out / "inputs").mkdir()
        (self.out / "results").mkdir()
        fixture_list, baseline = [], []
        for index, name in enumerate(("gold_three", "season46_tv")):
            data = ep.wav(bytes([index + 1, 0]) * (16000 * (index + 1)))
            path = self.out / "inputs" / (name + ".wav")
            path.write_bytes(data)
            old = []
            for model in (api.MAI, api.MUSE):
                entry = {"request_scope": f"request_{len(baseline) + 1:04d}", "model": model,
                         "seconds": index + 1, "wav_sha256": ep.digest(data), "actual_cost_usd": .0001, "status": "received"}
                baseline.append(entry)
                old.append(entry)
            fixture_list.append({"id": name, "prepared_path": str(path), "wav_sha256": ep.digest(data),
                                 "seconds": index + 1, "matched_direct_baselines": old})
        baseline += [{"request_scope": f"request_{i:04d}", "actual_cost_usd": .0001, "status": "received"}
                     for i in range(5, 73)]
        ledger = api.BudgetLedger(1, self.ledger, max_requests=80)
        ledger.entries = baseline
        ledger.save()
        snapshot = root / "genuine-synthetic-backup.json"
        snapshot.write_bytes(self.ledger.read_bytes())
        self.snapshot_path = snapshot
        ep.save(self.out / "ledger_recovery.json", {"status": "restored_exact_genuine_history", "source": str(snapshot),
            "target": str(self.ledger), "source_sha256": ep.file_hash(snapshot), "expected_sha256": ep.file_hash(snapshot),
            "restored_sha256": ep.file_hash(snapshot)})
        slots = [{"slot": i * 2 + j + 1, "model": model, "fixture": f["id"], "wav_sha256": f["wav_sha256"],
                  "request_scope_if_verified_72_entry_ledger": f"request_{73 + i * 2 + j:04d}",
                  "conservative_reservation_usd": .05} for i, model in enumerate(api.EXPERIMENT_OPTIONS)
                 for j, f in enumerate(fixture_list)]
        ep.save(self.out / "plan.json", {"models": [{"id": m} for m in api.EXPERIMENT_OPTIONS],
            "fixtures": fixture_list, "slots": slots, "budget": {"new_max_POSTs": 8,
            "shared_ceiling_usd": 1, "explicit_total_cap_if_history_still_72": 80}})
        self.plan = ep.read(self.out / "plan.json")
        for target, args in (("urllib.request.urlopen", {"side_effect": AssertionError("Network forbidden")}),
                             ("utils.stt_openrouter.load_key", {"return_value": "offline-key"}),
                             ("utils.stt_openrouter.Client.call", {"autospec": True, "side_effect": self.reply})):
            mocker = patch(target, **args)
            mocked = mocker.start()
            self.addCleanup(mocker.stop)
            if target.endswith("load_key"):
                self.key = mocked
            if target.endswith("Client.call"):
                self.calls = mocked

    def reply(self, client, model, data):
        entry = client.ledger.reserve(model, api.wav_seconds(data))
        raw = {"text": "Synthetic text only", "usage": {"cost": .001}}
        entry.update(wav_sha256=ep.digest(data), status="received", actual_cost_usd=.001, response=raw)
        client.ledger.save()
        return raw, entry

    def run_job(self, allow=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.run(self.out, self.ledger, allow_upload=allow)

    def files(self):
        return {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.out.parent.rglob("*") if p.is_file()}

    def test_plan_only_zero_calls_key_writes_and_fixed_eight_slots(self):
        before = self.files()
        result = self.run_job()
        self.assertEqual(result["status"], "plan_only")
        self.assertEqual(result["remaining_POSTs"], 8)
        self.assertEqual(result["conservative_remaining_reservation_usd"], .4)
        self.assertEqual(result["total_request_cap"], 80)
        self.calls.assert_not_called()
        self.key.assert_not_called()
        self.assertEqual(self.files(), before)
        self.assertNotIn("Authorization", json.dumps(result))

    def test_eight_text_only_outcomes_no_fake_exports_and_repeat_no_posts_or_writes(self):
        result = self.run_job(True)
        self.assertEqual(result["new_POSTs"], 8)
        self.assertTrue(all(r["quality_status"] == "text_only_or_empty" for r in result["results"]))
        self.assertFalse(list((self.out / "results").rglob("*.srt")))
        self.assertEqual(len(api.BudgetLedger(1, self.ledger).entries), 80)
        before = self.files()
        self.calls.reset_mock()
        self.key.reset_mock()
        self.assertEqual(self.run_job(True)["new_POSTs"], 0)
        self.assertEqual(self.files(), before)
        self.calls.assert_not_called()
        self.key.assert_not_called()

    def test_returned_ledger_reply_recovery_without_new_call_for_that_slot(self):
        ledger = api.BudgetLedger(1, self.ledger, max_requests=80)
        self.reply(api.Client("offline-key", ledger), self.plan["slots"][0]["model"],
                   Path(self.plan["fixtures"][0]["prepared_path"]).read_bytes())
        self.assertEqual(self.run_job(True)["new_POSTs"], 7)
        self.assertEqual(self.calls.call_count, 7)

    def test_attempt_without_reply_blocks_without_key_or_paid_retry(self):
        slot = self.plan["slots"][0]
        path = self.out / "results/slot_01"
        path.mkdir()
        data = Path(self.plan["fixtures"][0]["prepared_path"]).read_bytes()
        ep.save(path / "attempt.json", {"slot": slot, "request_body_schema": runner.body_schema(slot["model"], data)})
        with self.assertRaisesRegex(ValueError, "Attempt without returned"):
            self.run_job(True)
        self.calls.assert_not_called()
        self.key.assert_not_called()

    def test_all_returned_ledger_replies_recover_caches_without_key_or_post(self):
        ledger = api.BudgetLedger(1, self.ledger, max_requests=80)
        client = api.Client("offline-key", ledger)
        fixtures = {f["id"]: f for f in self.plan["fixtures"]}
        for slot in self.plan["slots"]:
            self.reply(client, slot["model"], Path(fixtures[slot["fixture"]]["prepared_path"]).read_bytes())
        self.assertEqual(self.run_job(True)["new_POSTs"], 0)
        self.calls.assert_not_called()
        self.key.assert_not_called()

    def test_over_budget_reply_durable_before_stop_and_no_next_post(self):
        def expensive(client, model, data):
            raw, entry = self.reply(client, model, data)
            entry["actual_cost_usd"] = 1.01
            client.ledger.save()
            return raw, entry
        self.calls.side_effect = expensive
        with self.assertRaisesRegex(ValueError, "Unknown/exceeded"):
            self.run_job(True)
        self.assertEqual(ep.read(self.out / "results/slot_01/response.json")["entry"]["actual_cost_usd"], 1.01)
        self.assertEqual(self.calls.call_count, 1)

    def test_symlink_ledger_and_foreign_result_artifacts_rejected(self):
        foreign = self.out / "results/user_files"
        foreign.mkdir()
        (foreign / "keep.txt").write_text("user work")
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            self.run_job(True)
        (foreign / "keep.txt").unlink()
        foreign.rmdir()
        alias = self.out.parent / "ledger_alias.json"
        alias.symlink_to(self.ledger)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            runner.run(self.out, alias, allow_upload=True)
        self.key.assert_not_called()
        self.calls.assert_not_called()

    def test_unknown_billing_saved_raw_then_blocks_all_further_calls(self):
        def unknown(client, model, data):
            raw, entry = self.reply(client, model, data)
            entry.update(actual_cost_usd=None, status="billing_unknown")
            client.ledger.save()
            return raw, entry
        self.calls.side_effect = unknown
        with self.assertRaisesRegex(ValueError, "Unknown/exceeded"):
            self.run_job(True)
        cache = ep.read(self.out / "results/slot_01/response.json")
        self.assertEqual(cache["entry"]["response"]["text"], "Synthetic text only")
        with self.assertRaisesRegex(ValueError, "Unknown/exceeded"):
            self.run_job(True)
        self.assertEqual(self.calls.call_count, 1)

    def test_known_http_rejection_and_bad_timing_are_terminal_outcomes_not_retried(self):
        def varied(client, model, data):
            raw, entry = self.reply(client, model, data)
            if len(client.ledger.entries) == 73:
                entry.update(status="http_error_cost_known", response={"error": {"message": "Synthetic 400"}})
                client.ledger.save()
                raise ValueError("Synthetic HTTP 400; known cost")
            if len(client.ledger.entries) == 74:
                raw["segments"] = [{"text": "Invalid", "start": 0, "end": 99}]
                client.ledger.save()
            return raw, entry
        self.calls.side_effect = varied
        result = self.run_job(True)
        self.assertEqual(result["new_POSTs"], 8)
        self.assertEqual(result["results"][0]["quality_status"], "request_rejected_cost_known_no_format_retry")
        self.assertEqual(result["results"][1]["quality_status"], "invalid_native_timing_or_shape")
        self.assertFalse((self.out / "results/slot_02/native.srt").exists())
        self.assertEqual(self.run_job(True)["new_POSTs"], 0)

    def test_native_word_export_and_unknown_gender_from_synthetic_scribe(self):
        def native(client, model, data):
            raw, entry = self.reply(client, model, data)
            raw["words"] = [{"type": "word", "text": "Native", "start": .1, "end": .8, "speaker_id": 0},
                            {"type": "audio_event", "text": "[noise]"}]
            client.ledger.save()
            return raw, entry
        self.calls.side_effect = native
        result = self.run_job(True)
        self.assertEqual(result["results"][0]["timing_diagnostics"]["skipped_non_speech_items"], 1)
        rows = ep.read(self.out / "results/slot_01/native.json")
        self.assertEqual((rows[0]["start"], rows[0]["end"], rows[0]["text"], rows[0]["gender"]), (100, 800, ["Native"], "unknown"))

    def test_corrupt_history_cache_fixture_and_extra_scope_fail_before_paid_call(self):
        self.run_job(True)
        self.calls.reset_mock()
        self.key.reset_mock()
        for path in (self.ledger, self.out / "results/slot_01/response.json", self.out / "inputs/gold_three.wav"):
            original = path.read_bytes()
            if path == self.ledger:
                data = json.loads(original)
                data["requests"][0]["actual_cost_usd"] = .002
                path.write_text(json.dumps(data))
            elif path.name == "response.json":
                data = json.loads(original)
                data["entry"]["wav_sha256"] = "bad"
                path.write_text(json.dumps(data))
            else:
                path.write_bytes(original[:-2] + b"xx")
            with self.assertRaises(ValueError):
                self.run_job(True)
            path.write_bytes(original)
        original = self.ledger.read_bytes()
        data = json.loads(original)
        data["requests"].append({"actual_cost_usd": .001, "request_scope": "request_0081"})
        self.ledger.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "historical ledger"):
            self.run_job(True)
        self.ledger.write_bytes(original)
        self.calls.assert_not_called()
        self.key.assert_not_called()


if __name__ == "__main__":
    unittest.main()
