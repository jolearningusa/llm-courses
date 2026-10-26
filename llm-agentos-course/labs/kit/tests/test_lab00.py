"""Lab 0 autograder tests: nanoos/trace.py (JsonlTracer.event, summarize_trace, token_breakdown)."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path

from nanoos.trace import JsonlTracer, read_trace, summarize_trace, token_breakdown
from nanoos.types import estimate_tokens as est
from run_suite import main as run_suite


class TestJsonlTracer(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "t.jsonl"

    def tearDown(self):
        self.dir.cleanup()

    def test_record_has_required_keys_and_fields(self):
        t = JsonlTracer(self.path, "run-7")
        before = time.time()
        t.event("tool_call", name="read_file", is_error=False)
        t.close()
        (rec,) = read_trace(self.path)
        self.assertEqual((rec["seq"], rec["run_id"], rec["kind"]), (0, "run-7", "tool_call"))
        self.assertEqual((rec["name"], rec["is_error"]), ("read_file", False))
        self.assertGreaterEqual(rec["ts"], before)
        self.assertLessEqual(rec["ts"], time.time())

    def test_seq_counts_up_and_ts_does_not_go_back(self):
        t = JsonlTracer(self.path, "r")
        for i in range(5):
            t.event("model_call", step=i)
        t.close()
        recs = read_trace(self.path)
        self.assertEqual([r["seq"] for r in recs], [0, 1, 2, 3, 4])
        self.assertEqual([r["step"] for r in recs], [0, 1, 2, 3, 4])
        self.assertEqual([r["ts"] for r in recs], sorted(r["ts"] for r in recs))

    def test_each_line_is_flushed_before_close(self):
        t = JsonlTracer(self.path, "r")
        t.event("model_call", step=0)
        self.assertEqual(len(read_trace(self.path)), 1)
        t.close()

    def test_values_json_cannot_encode_do_not_crash(self):
        t = JsonlTracer(self.path, "r")
        t.event("odd", where=Path("a/b"))
        t.close()
        self.assertEqual(read_trace(self.path)[0]["where"], "a/b")


class TestSummarize(unittest.TestCase):
    EVENTS = [
        {"seq": 0, "ts": 100.0, "kind": "run_start", "task": "x"},
        {"seq": 1, "ts": 100.5, "kind": "model_call", "input_tokens": 500, "cached_input_tokens": 0, "output_tokens": 20},
        {"seq": 2, "ts": 100.6, "kind": "tool_call", "name": "read_file", "is_error": False},
        {"seq": 3, "ts": 100.7, "kind": "tool_call", "name": "search", "is_error": True},
        {"seq": 4, "ts": 101.2, "kind": "model_call", "input_tokens": 700, "cached_input_tokens": 480, "output_tokens": 30},
        {"seq": 5, "ts": 101.3, "kind": "tool_call", "name": "read_file", "is_error": False},
        {"seq": 6, "ts": 102.0, "kind": "model_call", "input_tokens": 900, "cached_input_tokens": 680, "output_tokens": 15},
        {"seq": 7, "ts": 102.5, "kind": "run_end", "stopped_by": "answer"},
    ]

    def test_totals(self):
        s = summarize_trace(self.EVENTS)
        self.assertEqual((s["model_calls"], s["tool_calls"], s["tool_errors"]), (3, 3, 1))
        self.assertEqual((s["input_tokens"], s["cached_input_tokens"], s["output_tokens"]), (2100, 1160, 65))
        self.assertEqual(s["tools_used"], {"read_file": 2, "search": 1})
        self.assertAlmostEqual(s["wall_s"], 2.5)

    def test_exact_keys(self):
        self.assertEqual(set(summarize_trace(self.EVENTS)), {
            "model_calls", "tool_calls", "tool_errors", "input_tokens", "cached_input_tokens", "output_tokens",
            "tools_used", "wall_s"})

    def test_empty_and_single_event(self):
        s = summarize_trace([])
        self.assertEqual((s["model_calls"], s["input_tokens"], s["tools_used"], s["wall_s"]), (0, 0, {}, 0.0))
        self.assertEqual(summarize_trace(self.EVENTS[:1])["wall_s"], 0.0)


class TestTokenBreakdown(unittest.TestCase):
    def test_categories(self):
        calls = [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"a\"}"}}]
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
        messages = [
            {"role": "system", "content": "You are careful. " * 10},
            {"role": "user", "content": "What is in a?"},
            {"role": "assistant", "content": None, "tool_calls": calls},
            {"role": "tool", "tool_call_id": "c1", "content": "x" * 400},
            {"role": "assistant", "content": "It holds 400 x characters."},
        ]
        b = token_breakdown(messages, tools)
        self.assertEqual(set(b), {"system", "tools", "tool_results", "history"})
        self.assertEqual(b["system"], est("You are careful. " * 10))
        self.assertEqual(b["tools"], est(json.dumps(tools)))
        self.assertEqual(b["tool_results"], 100)
        self.assertEqual(b["history"], est("What is in a?") + est(json.dumps(calls)) + est("It holds 400 x characters."))

    def test_empty_prompt(self):
        self.assertEqual(token_breakdown([], []), {"system": 0, "tools": est("[]"), "tool_results": 0, "history": 0})


class TestEndToEnd(unittest.TestCase):
    def test_suite_writes_a_trace_you_can_summarize(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                run_suite(["--model", "demo", "--tasks", "b07", "--out", str(out)])
            events = read_trace(out / "traces" / "b07_two_hop__0.jsonl")
            self.assertEqual([e["kind"] for e in events][0], "run_start")
            self.assertEqual(events[-1]["kind"], "run_end")
            s = summarize_trace(events)
            self.assertEqual((s["model_calls"], s["tool_calls"], s["tool_errors"]), (3, 2, 0))
            self.assertEqual(s["tools_used"], {"search": 2})
