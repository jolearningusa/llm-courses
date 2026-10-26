"""Lab 9 autograder tests: nanoos/delegate.py (child isolation, limits, failures, fan-out)."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path

from models.scripted import ScriptedModel
from multiagent_eval import main as multiagent_eval
from nanoos import baseline
from nanoos.delegate import Delegator
from nanoos.tools_basic import WorkspaceTools
from tasks.basic import call, say

SECRET = "PARENT-ONLY-7731"


def ws_with_files() -> tempfile.TemporaryDirectory:
    d = tempfile.TemporaryDirectory()
    for i in range(5):
        (Path(d.name) / f"f{i}.txt").write_text(f"file {i}\n")
    return d


class Factory:
    """Scripted workers keyed by brief; records every model it hands out."""

    def __init__(self, scripts, default=None):
        self.scripts, self.default, self.models = scripts, default, []

    def __call__(self, brief, depth):
        m = ScriptedModel(self.scripts.get(brief, self.default or [say(f"done: {brief}")]))
        m.depth = depth
        self.models.append(m)
        return m


def delegate(brief, **kw):
    return {"tool_calls": [{"name": "delegate", "arguments": {"brief": brief, **kw}}]}


class TestIsolation(unittest.TestCase):
    def setUp(self):
        self.dir = ws_with_files()
        self.ws = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_child_sees_only_its_brief(self):
        f = Factory({"Count the files.": [call("list_files"), say("5 files")]})
        d = Delegator(WorkspaceTools(self.ws), f)
        parent = ScriptedModel([call("read_file", path="f0.txt"), delegate("Count the files."), say("Five.")])
        r = baseline.run(parent, f"Remember {SECRET}. How many files?", d)
        self.assertEqual(r.answer, "Five.")
        first = f.models[0].calls[0]["messages"]
        self.assertEqual([m["role"] for m in first], ["system", "user"])
        self.assertEqual(first[1]["content"], "Count the files.")
        self.assertNotIn(SECRET, json.dumps([c["messages"] for c in f.models[0].calls]))
        self.assertIn("5 files", parent.calls[-1]["messages"][-1]["content"], "only the summary comes back")
        rec = d.records[0]
        self.assertEqual((rec.id, rec.depth, rec.status, rec.answer, rec.steps), (1, 1, "ok", "5 files", 2))
        self.assertGreater(rec.usage.input_tokens, 0)
        self.assertEqual(d.children_usage().input_tokens, rec.usage.input_tokens)

    def test_restricted_tools(self):
        f = Factory({"b": [call("write_file", path="x.txt", content="x"), say("tried")]})
        d = Delegator(WorkspaceTools(self.ws), f)
        out, err = d.call("delegate", {"brief": "b", "tools": ["read_file", "list_files"]})
        self.assertFalse(err)
        child_tools = [t["function"]["name"] for t in f.models[0].calls[0]["tools"]]
        self.assertEqual(sorted(child_tools), ["list_files", "read_file"], "no delegation at max depth 1")
        self.assertIn("not available", f.models[0].calls[1]["messages"][-1]["content"])
        self.assertFalse((self.ws / "x.txt").exists())
        out, err = d.call("delegate", {"brief": "b", "tools": ["read_file", "rm_rf"]})
        self.assertTrue(err)
        self.assertIn("rm_rf", out)
        self.assertIn("list_files", out, "list what is available")
        self.assertEqual(len(d.records), 1, "a rejected request starts no child")

    def test_depth_limit(self):
        f = Factory({"outer": [delegate("inner"), say("outer done")], "inner": [say("inner done")]})
        d1 = Delegator(WorkspaceTools(self.ws), f, max_depth=1)
        self.assertIn("delegate", [s["function"]["name"] for s in d1.schemas()])
        d2 = Delegator(WorkspaceTools(self.ws), f, max_depth=2)
        self.assertEqual(d2.call("delegate", {"brief": "outer"}), ("outer done", False))
        self.assertEqual([(r.brief, r.depth, r.status) for r in d2.records],
                         [("outer", 1, "ok"), ("inner", 2, "ok")])
        self.assertIn("delegate", [t["function"]["name"] for t in f.models[0].calls[0]["tools"]])
        self.assertNotIn("delegate", [t["function"]["name"] for t in f.models[1].calls[0]["tools"]])
        deep = Delegator(WorkspaceTools(self.ws), f, depth=1, max_depth=1)
        out, err = deep.call("delegate", {"brief": "inner"})
        self.assertTrue(err)
        self.assertIn("depth", out)
        self.assertEqual(deep.records, [])


class TestLimitsAndFailures(unittest.TestCase):
    def setUp(self):
        self.dir = ws_with_files()
        self.tools = WorkspaceTools(Path(self.dir.name))

    def tearDown(self):
        self.dir.cleanup()

    def test_children_budget(self):
        d = Delegator(self.tools, Factory({}), max_children=2)
        self.assertFalse(d.call("delegate", {"brief": "a"})[1])
        self.assertFalse(d.call("delegate", {"brief": "b"})[1])
        out, err = d.call("delegate", {"brief": "c"})
        self.assertTrue(err)
        self.assertIn("2", out)
        self.assertEqual(len(d.records), 2)
        out, err = d.call("delegate_many", {"briefs": ["d", "e"]})
        self.assertTrue(err, "every child was refused")

    def test_failed_child_returns_an_error_summary(self):
        f = Factory({"boom": [{"raise": "rate limited"}], "loop": [call("list_files")] * 10})
        d = Delegator(self.tools, f, child_max_steps=3)
        out, err = d.call("delegate", {"brief": "boom"})
        self.assertTrue(err)
        self.assertTrue(out.startswith("ERROR: worker 1 failed: ModelError"), out)
        self.assertIn("rate limited", out)
        out, err = d.call("delegate", {"brief": "loop"})
        self.assertTrue(err)
        self.assertIn("without an answer", out)
        self.assertEqual([r.status for r in d.records], ["error", "error"])
        self.assertEqual(d.records[1].steps, 3)

    def test_a_hung_child_times_out(self):
        d = Delegator(self.tools, Factory({"slow": [{"delay_s": 3, "content": "late"}]}), child_timeout_s=0.3)
        t0 = time.monotonic()
        out, err = d.call("delegate", {"brief": "slow"})
        self.assertLess(time.monotonic() - t0, 2)
        self.assertTrue(err)
        self.assertIn("timed out", out)
        self.assertEqual(d.records[0].status, "timeout")
        self.assertEqual(d.shared.active, 0)

    def test_long_summaries_are_cut(self):
        d = Delegator(self.tools, Factory({"long": [say("x" * 500)]}), max_summary_chars=100)
        out, err = d.call("delegate", {"brief": "long"})
        self.assertFalse(err)
        self.assertEqual(out, "x" * 100 + " [summary truncated]")


class TestFanOut(unittest.TestCase):
    def test_parallel_in_order_and_capped(self):
        with tempfile.TemporaryDirectory() as dd:
            scripts = {f"q{i}": [{"delay_s": 0.2, "content": f"a{i}"}] for i in range(6)}
            scripts["q3"] = [{"raise": "down"}]
            d = Delegator(WorkspaceTools(Path(dd)), Factory(scripts), max_concurrency=3)
            t0 = time.monotonic()
            out, err = d.call("delegate_many", {"briefs": [f"q{i}" for i in range(6)]})
            wall = time.monotonic() - t0
        self.assertFalse(err, "one failed child is not a failed fan-out")
        blocks = out.split("\n\n")
        self.assertEqual(len(blocks), 6)
        self.assertEqual(blocks[0], "[1] a0")
        self.assertEqual(blocks[5], "[6] a5")
        self.assertTrue(blocks[3].startswith("[4] ERROR: worker"), blocks[3])
        self.assertEqual(d.peak_concurrency, 3)
        self.assertLess(wall, 0.9, "children run in parallel (sequential would take 1.0s)")
        self.assertGreater(wall, 0.35, "at most three at a time")
        self.assertEqual(sorted(r.id for r in d.records), list(range(1, 7)))


class TestMultiagentEval(unittest.TestCase):
    def test_when_workers_pay(self):
        s = {}
        for arch in ("single", "orchestrator"):
            with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
                multiagent_eval(["--arch", arch, "--out", str(Path(d) / "r")])
                s[arch] = json.loads((Path(d) / "r" / "summary.json").read_text())
        self.assertEqual((s["single"]["accuracy"], s["orchestrator"]["accuracy"]), (1.0, 1.0))
        single, orch = s["single"]["tasks"], s["orchestrator"]["tasks"]
        survey, chain = "r1_vendor_survey", "r2_dependency"
        self.assertLess(orch[survey]["parent_context_tokens"] * 10, single[survey]["parent_context_tokens"])
        self.assertLess(orch[survey]["total_tokens"], single[survey]["total_tokens"])
        self.assertEqual(orch[survey]["workers"], 5)
        self.assertGreater(orch[chain]["total_tokens"], single[chain]["total_tokens"], "hand-offs cost tokens")


if __name__ == "__main__":
    unittest.main()
