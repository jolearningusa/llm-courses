"""Lab 3 autograder tests: nanoos/context.py (ThresholdCompaction, ToolResultClearing)."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from models.scripted import ScriptedModel
from nanoos.context import (CLEARED, PIN_TAG, SUMMARY_PREFIX, ContextManagedModel, ThresholdCompaction,
                            ToolResultClearing, count_tokens)
from run_suite import main as run_suite


def conversation(n_tools: int, size: int = 800, pin_at: int | None = None) -> list[dict]:
    """system, task, then n_tools x (assistant tool call, tool result of `size` characters)."""
    msgs = [{"role": "system", "content": "You are careful."}, {"role": "user", "content": "Do the task."}]
    for i in range(n_tools):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": f"f{i}"})}}]})
        body = f"file {i}: " + "x" * size
        if pin_at == i:
            body = f"{PIN_TAG} the code word is HERON\n" + body
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": body})
    return msgs


class CountingSummarizer:
    def __init__(self):
        self.calls = []

    def __call__(self, previous, chunk):
        self.calls.append((previous, len(chunk)))
        return f"summary #{len(self.calls)} of {len(chunk)} messages"


def assert_well_formed(test, msgs):
    """Every tool message must follow the assistant message that made its call."""
    open_ids = set()
    for m in msgs:
        if m["role"] == "assistant":
            open_ids = {c["id"] for c in m.get("tool_calls") or []}
        elif m["role"] == "tool":
            test.assertIn(m["tool_call_id"], open_ids, "a tool result lost its assistant tool call")


class TestThreshold(unittest.TestCase):
    def test_under_budget_is_unchanged(self):
        msgs = conversation(3)
        eng = ThresholdCompaction(budget_tokens=100_000, summarizer=CountingSummarizer())
        self.assertEqual(eng.prepare(msgs), msgs)
        self.assertEqual(eng.compactions, 0)

    def test_over_budget_compacts_and_keeps_head_and_tail(self):
        msgs = conversation(20)
        s = CountingSummarizer()
        eng = ThresholdCompaction(budget_tokens=2000, summarizer=s, keep_last=6)
        out = eng.prepare(msgs)
        self.assertEqual(out[:2], msgs[:2], "system prompt and task must stay first")
        self.assertTrue(out[2]["content"].startswith(SUMMARY_PREFIX))
        self.assertEqual(out[3:], msgs[len(msgs) - 6:], "the last keep_last messages stay as they are")
        self.assertLess(count_tokens(out), count_tokens(msgs))
        self.assertEqual((eng.compactions, len(s.calls)), (1, 1))
        assert_well_formed(self, out)

    def test_never_splits_a_tool_call_from_its_result(self):
        msgs = conversation(20)
        eng = ThresholdCompaction(budget_tokens=2000, summarizer=CountingSummarizer(), keep_last=5)  # odd: lands on a tool
        out = eng.prepare(msgs)
        assert_well_formed(self, out)
        self.assertEqual(out[3]["role"], "assistant")

    def test_summary_is_reused_until_budget_is_hit_again(self):
        s = CountingSummarizer()
        eng = ThresholdCompaction(budget_tokens=3000, summarizer=s, keep_last=6)
        msgs = conversation(20)
        eng.prepare(msgs)
        more = msgs + conversation(21)[-2:]  # one more call and result
        out = eng.prepare(more)
        self.assertEqual(len(s.calls), 1, "no new summary while the compacted view fits the budget")
        self.assertEqual(out[-2:], more[-2:])
        self.assertTrue(out[2]["content"].startswith(SUMMARY_PREFIX))

    def test_second_compaction_rolls_the_previous_summary_in(self):
        s = CountingSummarizer()
        eng = ThresholdCompaction(budget_tokens=2000, summarizer=s, keep_last=4)
        eng.prepare(conversation(10))
        out = eng.prepare(conversation(30))
        self.assertEqual(eng.compactions, 2)
        self.assertIsNotNone(s.calls[1][0], "the second summary sees the first one")
        self.assertIn("summary #2", out[2]["content"])
        assert_well_formed(self, out)

    def test_pinned_lines_survive_every_compaction(self):
        eng = ThresholdCompaction(budget_tokens=2000, summarizer=CountingSummarizer(), keep_last=4)
        eng.prepare(conversation(10, pin_at=1))
        out = eng.prepare(conversation(30, pin_at=1))
        self.assertEqual(eng.compactions, 2)
        self.assertIn(f"{PIN_TAG} the code word is HERON", out[2]["content"])

    def test_input_is_not_modified(self):
        msgs = conversation(20)
        before = copy.deepcopy(msgs)
        ThresholdCompaction(budget_tokens=2000, summarizer=CountingSummarizer()).prepare(msgs)
        self.assertEqual(msgs, before)


class TestClearing(unittest.TestCase):
    def test_old_results_cleared_recent_kept(self):
        msgs = conversation(8)
        out = ToolResultClearing(keep_last_results=3).prepare(msgs)
        tools_out = [m for m in out if m["role"] == "tool"]
        tools_in = [m for m in msgs if m["role"] == "tool"]
        for m in tools_out[:5]:
            self.assertEqual(m["content"], CLEARED.format(chars=len(tools_in[0]["content"])))
        self.assertEqual(tools_out[5:], tools_in[5:])
        self.assertEqual([m["tool_call_id"] for m in tools_out], [m["tool_call_id"] for m in tools_in])
        self.assertEqual(len(out), len(msgs))
        self.assertLess(count_tokens(out), count_tokens(msgs))

    def test_short_results_are_left_alone(self):
        msgs = conversation(6, size=10)
        self.assertEqual(ToolResultClearing(keep_last_results=1, min_chars=200).prepare(msgs), msgs)

    def test_input_is_not_modified(self):
        msgs = conversation(8)
        before = copy.deepcopy(msgs)
        ToolResultClearing().prepare(msgs)
        self.assertEqual(msgs, before)


class TestIntegration(unittest.TestCase):
    def test_model_sees_the_prepared_context(self):
        inner = ScriptedModel([{"content": "ok"}])
        m = ContextManagedModel(inner, ToolResultClearing(keep_last_results=1))
        m.complete(conversation(4), [])
        sent = inner.calls[0]["messages"]
        self.assertEqual(sum(1 for x in sent if x["role"] == "tool" and x["content"].startswith("[tool result cleared")), 3)

    def test_long_tier_demo_with_each_engine(self):
        for engine in ("threshold", "clearing"):
            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as d, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                out = Path(d) / "r"
                run_suite(["--model", "demo", "--tier", "long", "--context", engine, "--context-budget", "4000",
                           "--max-steps", "60", "--out", str(out)])
                summary = json.loads((out / "summary.json").read_text())
                self.assertEqual(summary["success_rate"], 1.0)
                self.assertGreater(summary["compactions"], 0)
