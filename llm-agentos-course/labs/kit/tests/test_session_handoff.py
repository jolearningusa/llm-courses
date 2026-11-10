import json
import tempfile
import unittest
from pathlib import Path

from session_handoff_eval import BUDGET, CUTS, OP, ReceiptTool, context_text, decode_state, resume, simulate


class SessionHandoffControls(unittest.TestCase):
    def test_handoff_preserves_goal_progress_and_one_effect_at_all_cuts(self):
        for cut in CUTS:
            with self.subTest(cut=cut):
                row = simulate("handoff", cut)
                self.assertEqual(row["worker_exit_code"], 0)
                self.assertTrue(row["goal_preserved"])
                self.assertTrue(row["completed_progress_preserved"])
                self.assertTrue(row["pending_resolved"])
                self.assertEqual(row["result"]["state"]["completed"], ["prepare", "publish"])
                self.assertEqual(row["result"]["simulated_effects"], 1)

    def test_ambiguous_receipt_reuses_key_and_saved_receipt_needs_no_retry(self):
        lost = simulate("handoff", "after_effect_before_receipt")
        self.assertEqual(lost["effects_before_restart"], 1)
        self.assertEqual((lost["result"]["tool_attempts"], lost["result"]["receipt_replayed"]), (1, True))
        acked = simulate("handoff", "after_receipt")
        self.assertEqual(acked["result"]["tool_attempts"], 0)

    def test_noisy_tail_blocks_without_guessing_missing_state(self):
        for cut, effects in zip(CUTS, (0, 1, 1)):
            row = simulate("tail", cut)
            self.assertEqual(row["result"]["status"], "blocked")
            self.assertFalse(row["goal_preserved"])
            self.assertEqual(row["result"]["tool_attempts"], 0)
            self.assertEqual(row["result"]["simulated_effects"], effects)

    def test_short_tail_negative_control_uses_the_same_reader_successfully(self):
        for cut in CUTS:
            row = simulate("tail", cut, noise_events=0)
            self.assertTrue(row["goal_preserved"] and row["pending_resolved"])
            self.assertEqual(row["result"]["simulated_effects"], 1)

    def test_contexts_use_equal_exact_byte_budget_and_valid_whole_events(self):
        for mode in ("handoff", "tail"):
            row = simulate(mode, "before_tool")
            self.assertEqual(row["context_bytes"], BUDGET)
            self.assertEqual(len(row["context"].encode()), 512)
            self.assertIsInstance(json.loads(row["context"]), dict if mode == "handoff" else list)

    def test_new_retry_key_negative_control_duplicates_a_simulated_effect(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "receipts.json"
            ReceiptTool(path).apply(OP["key"], OP["payload"])
            restarted = ReceiptTool(path)
            restarted.apply("a-new-retry-key", OP["payload"])
            self.assertEqual(len(restarted.data["effects"]), 2)

    def test_same_key_different_payload_is_rejected_without_modifying_disk(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "receipts.json"
            ReceiptTool(path).apply(OP["key"], OP["payload"])
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "different payload"):
                ReceiptTool(path).apply(OP["key"], "different intent")
            self.assertEqual(path.read_bytes(), before)

    def test_same_key_and_payload_returns_the_existing_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "receipts.json"
            first, replayed = ReceiptTool(path).apply(OP["key"], OP["payload"])
            second, retried = ReceiptTool(path).apply(OP["key"], OP["payload"])
            self.assertFalse(replayed)
            self.assertTrue(retried)
            self.assertEqual(first, second)
            self.assertEqual(len(ReceiptTool(path).data["effects"]), 1)

    def test_missing_key_or_wrong_task_blocks_before_tool_attempt(self):
        row = simulate("handoff", "before_tool")
        for mutate in (lambda s: s["pending"].pop("key"), lambda s: s.update(task_id="another-task")):
            state = json.loads(row["context"])
            mutate(state)
            with tempfile.TemporaryDirectory() as d:
                context, receipts = Path(d) / "context.json", Path(d) / "receipts.json"
                context.write_text(context_text("handoff", state, []))
                result = resume(context, receipts)
                self.assertEqual((result["status"], result["tool_attempts"]), ("blocked", 0))
                self.assertFalse(receipts.exists())

    def test_budget_and_schema_failures_are_explicit(self):
        with self.assertRaises(ValueError):
            context_text("handoff", {"goal": "x" * 600}, [])
        for text in ("not json", "{}", "[]", " " * 513):
            with self.assertRaises(ValueError):
                decode_state(text)

    def test_missing_checkpoint_does_not_create_tool_state(self):
        with tempfile.TemporaryDirectory() as d:
            receipts = Path(d) / "receipts.json"
            result = resume(Path(d) / "missing.json", receipts)
            self.assertEqual((result["status"], result["tool_attempts"]), ("blocked", 0))
            self.assertFalse(receipts.exists())

    def test_no_pending_operation_is_not_itself_a_completion_proof(self):
        state = json.loads(simulate("handoff", "before_tool")["context"])
        state["pending"] = None
        with tempfile.TemporaryDirectory() as d:
            context, receipts = Path(d) / "context.json", Path(d) / "receipts.json"
            context.write_text(context_text("handoff", state, []))
            result = resume(context, receipts)
            self.assertEqual((result["status"], result["tool_attempts"]), ("incomplete", 0))


if __name__ == "__main__":
    unittest.main()
