"""Lab 12 autograder tests: nanoos/ports.py (Hermes ContextEngine and MemoryProvider, OpenClaw context engine).

The tests use stand-in engines and a stand-in memory store, so this lab's grade does not depend on Labs 3 and 4.
"""

from __future__ import annotations

import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from models.scripted import ScriptedModel
from nanoos.context import count_tokens
from nanoos.hosts import BridgeClient, HermesHost, OpenClawHost
from nanoos.ports import HermesContextEngine, HermesMemoryProvider, OpenClawEngine, fit_budget
from tasks.basic import call, say

SUMMARY = {"role": "user", "content": "[summary of earlier work]"}


class Squash:
    """A view engine: over budget, keep the first two messages, a summary and the last two."""

    def __init__(self, budget: int, log: list | None = None):
        self.budget, self.compactions, self.seen = budget, 0, []
        if log is not None:
            log.append(budget)

    def prepare(self, messages):
        self.seen.append(len(messages))
        if count_tokens(messages) <= self.budget or len(messages) <= 5:
            return messages
        self.compactions += 1
        return messages[:2] + [SUMMARY] + messages[-2:]


class Clear:
    """A view engine that shortens every tool result but the last."""

    compactions = 0

    def __init__(self, budget: int = 0):
        self.calls = 0

    def prepare(self, messages):
        self.calls += 1
        last = max((i for i, m in enumerate(messages) if m["role"] == "tool"), default=-1)
        out = [dict(m, content="[cleared]") if m["role"] == "tool" and i != last else m for i, m in enumerate(messages)]
        return messages if out == messages else out


def history(n: int, size: int = 400) -> list[dict]:
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(n):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * size})
    return msgs


class TestHermesContextEngine(unittest.TestCase):
    def test_should_compress(self):
        e = HermesContextEngine(lambda b: Squash(b), "compress", context_length=1000, threshold_ratio=0.5)
        self.assertEqual(e.threshold_tokens, 500)
        self.assertFalse(e.should_compress(499))
        self.assertTrue(e.should_compress(500))
        e.update_from_response({"prompt_tokens": 800, "completion_tokens": 5})
        self.assertTrue(e.should_compress(), "defaults to the last prompt tokens")
        s = HermesContextEngine(lambda b: Squash(b), "select", context_length=1000)
        self.assertFalse(s.should_compress(10 ** 6), "select mode never asks Hermes to compress")

    def test_compress_uses_a_fresh_engine_at_the_target(self):
        budgets: list[int] = []
        e = HermesContextEngine(lambda b: Squash(b, budgets), "compress", context_length=2000, threshold_ratio=0.5)
        budgets.clear()
        msgs = history(10)
        before = copy.deepcopy(msgs)
        out = e.compress(msgs, 900)
        self.assertEqual(budgets, [500], "a fresh engine with target_tokens (half the threshold)")
        self.assertEqual(out, msgs[:2] + [SUMMARY] + msgs[-2:])
        self.assertEqual(msgs, before, "the snapshot is not changed")
        self.assertEqual(e.compression_count, 1)
        e.compress(out, 900)
        self.assertEqual(len(budgets), 2, "a new engine each time, since the transcript was rewritten")
        short = history(1)
        self.assertIs(e.compress(short), short, "nothing to compact: the same list back")
        self.assertEqual(e.compression_count, 1)

    def test_select_context(self):
        made: list[int] = []
        e = HermesContextEngine(lambda b: made.append(b) or Clear(b), "select", context_length=1000)
        msgs = history(3)
        view = e.select_context(msgs)
        self.assertEqual([m["content"] for m in view if m["role"] == "tool"], ["[cleared]", "[cleared]", "x" * 400])
        self.assertEqual(msgs, history(3), "the transcript is not touched")
        self.assertIsNone(e.select_context(history(1)), "None when the view is unchanged")
        self.assertEqual(len(made), 1, "one engine for the session")
        self.assertEqual(e.engine.calls, 2)
        c = HermesContextEngine(lambda b: Clear(b), "compress", context_length=1000)
        self.assertIsNone(c.select_context(msgs))
        clone = e.clone_for_agent()
        self.assertIsNot(clone.engine, e.engine)
        self.assertEqual((clone.mode, clone.threshold_tokens), ("select", e.threshold_tokens))


class FakeStore:
    """A stand-in for your Lab 4 MemoryStore: ingest asks the extractor about user messages only."""

    instances: list["FakeStore"] = []

    def __init__(self, embedder=None):
        self.facts: dict[str, list[str]] = {}
        self.ingested: list[list[dict]] = []
        FakeStore.instances.append(self)

    @classmethod
    def load(cls, path, embedder=None):
        s = cls(embedder)
        s.facts = json.loads(Path(path).read_text())
        s.loaded_from = str(path)
        return s

    def save(self, path):
        Path(path).write_text(json.dumps(self.facts))

    def ingest(self, user_id, messages, extractor, trusted_tools=()):
        self.ingested.append(messages)
        for m in messages:
            if m["role"] == "user" or (m["role"] == "tool" and "tool" in trusted_tools):
                for op in extractor(user_id, m["content"], []):
                    self.facts.setdefault(user_id, []).append(op["text"])

    def search(self, user_id, query, k=5):
        words = set(query.lower().split())
        hits = [t for t in self.facts.get(user_id, []) if words & set(t.lower().split())]
        return [type("M", (), {"text": t}) for t in hits[:k]]


def extractor_factory(gate: threading.Event | None = None):
    def factory():
        def extract(user_id, text, existing):
            if gate is not None:
                gate.wait(5)
            return [{"op": "ADD", "text": text}] if "remember" in text.lower() else []
        return extract
    return factory


class TestHermesMemoryProvider(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()

    def provider(self, gate=None, **kw):
        return HermesMemoryProvider(extractor_factory(gate), store_cls=FakeStore, **kw)

    def test_initialize_scopes_and_loads(self):
        p = self.provider()
        p.initialize("s1", hermes_home=self.home)
        self.assertEqual(p.user_id, "default")
        self.assertEqual(p.path, Path(self.home) / "nanoos-memory" / "memory.json")
        self.assertTrue(p.path.parent.is_dir())
        Path(p.path).write_text(json.dumps({"alice": ["remember: alice lives in Lisbon"]}))
        q = self.provider()
        q.initialize("s2", hermes_home=self.home, user_id="alice", platform="telegram")
        self.assertEqual(q.store.loaded_from, str(p.path))
        self.assertEqual((q.session_id, q.user_id), ("s2", "alice"))
        self.assertIn("Lisbon", q.prefetch("where does alice live? Lisbon"))
        bob = self.provider()
        bob.initialize("s3", hermes_home=self.home, user_id="bob")
        self.assertEqual(bob.prefetch("Lisbon"), "", "another user sees nothing")

    def test_prefetch_uses_the_warm_result_once(self):
        p = self.provider()
        p.initialize("s1", hermes_home=self.home, user_id="u")
        p.store.facts["u"] = ["remember the blue mug"]
        p.warm["the mug"] = "WARM"
        self.assertEqual(p.prefetch("the mug"), "WARM")
        self.assertNotIn("the mug", p.warm)
        self.assertIn("blue mug", p.prefetch("the mug"))

    def test_sync_turn_does_not_block_and_session_end_flushes(self):
        gate = threading.Event()
        p = self.provider(gate)
        p.initialize("s1", hermes_home=self.home, user_id="u")
        t0 = time.monotonic()
        p.sync_turn("Please remember my badge is 4411", "Noted.")
        self.assertLess(time.monotonic() - t0, 1.0, "sync_turn must not wait for the write")
        self.assertFalse(p.path.exists())
        gate.set()
        p.on_session_end([])
        self.assertEqual(json.loads(p.path.read_text()), {"u": ["Please remember my badge is 4411"]})
        self.assertEqual([m["role"] for m in p.store.ingested[0]], ["user", "assistant"])

    def test_on_pre_compress_writes_now_with_the_trust_gate(self):
        p = self.provider(trusted_tools=("tool",))
        p.initialize("s1", hermes_home=self.home, user_id="u")
        msgs = [{"role": "user", "content": "remember the deadline is Friday"}]
        p.on_pre_compress(msgs)
        self.assertEqual(p.store.facts["u"], ["remember the deadline is Friday"], "written before it returns")
        self.assertEqual(p.store.ingested, [msgs])

    def test_host_hook_order(self):
        p = self.provider()
        e = HermesContextEngine(lambda b: Squash(b), "compress", context_length=1200, threshold_ratio=0.5)
        script = [call("read_file", path=f"f{i}") for i in range(6)] + [say("done")]

        class Files:
            def schemas(self):
                return [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]

            def call(self, name, args):
                return "y" * 600, False

        r = HermesHost(ScriptedModel(script), Files(), engine=e, memory=p, hermes_home=self.home).run(
            ["remember to read every file"])
        self.assertEqual(r.answers, ["done"])
        self.assertGreaterEqual(r.compressions, 1)
        i = r.hooks.index("engine.compress")
        self.assertEqual(r.hooks[i - 1], "memory.on_pre_compress")
        self.assertEqual(r.hooks[:2], ["memory.initialize", "engine.on_session_start"])
        self.assertEqual(r.hooks[-2:], ["memory.on_session_end", "memory.shutdown"])
        self.assertIn("remember to read every file", json.loads(p.path.read_text())["default"])


class TestOpenClawEngine(unittest.TestCase):
    def test_assemble(self):
        budgets: list[int] = []
        e = OpenClawEngine(lambda b: Squash(b, budgets))
        for m in history(8):
            self.assertEqual(e.ingest({"sessionId": "a", "message": m}), {"ingested": True})
        self.assertEqual(e.ingest({"sessionId": "a", "message": {"role": "user"}, "isHeartbeat": True}),
                         {"ingested": False})
        out = e.assemble({"sessionId": "a", "messages": [], "tokenBudget": 600})
        self.assertEqual(out["messages"], fit_budget(history(8)[:2] + [SUMMARY] + history(8)[-2:], 600))
        self.assertEqual(out["estimatedTokens"], count_tokens(out["messages"]))
        self.assertNotIn("contextProjection", out)
        e.assemble({"sessionId": "a", "messages": [], "tokenBudget": 600})
        self.assertEqual(budgets, [600], "one engine per session, built with the token budget")
        host = history(2)
        self.assertEqual(e.assemble({"sessionId": "b", "messages": host})["messages"], host,
                         "the host's messages win over the log; no budget means no cut")
        tight = e.assemble({"sessionId": "a", "messages": [], "tokenBudget": 120})
        self.assertLessEqual(tight["estimatedTokens"], 120, "the budget is respected even after the view")

    def test_compact(self):
        budgets: list[int] = []
        off = OpenClawEngine(lambda b: Squash(b, budgets), owns_compaction=False)
        off.ingest({"sessionId": "a", "message": history(1)[0]})
        self.assertEqual(off.compact({"sessionId": "a"}), {"ok": True, "compacted": False})
        e = OpenClawEngine(lambda b: Squash(b, budgets))
        for m in history(8):
            e.ingest({"sessionId": "a", "message": m})
        e.assemble({"sessionId": "a", "messages": [], "tokenBudget": 10 ** 6})
        budgets.clear()
        self.assertEqual(e.compact({"sessionId": "a", "tokenBudget": 1000}), {"ok": True, "compacted": True})
        self.assertEqual(budgets, [500], "a fresh engine at target_ratio of the budget")
        self.assertEqual(e.sessions["a"].log, history(8)[:2] + [SUMMARY] + history(8)[-2:])
        self.assertEqual(e.sessions["a"].epoch, 1)
        out = e.assemble({"sessionId": "a", "messages": [], "tokenBudget": 10 ** 6})
        self.assertEqual(budgets[-1], 10 ** 6, "the assemble engine is rebuilt after the log changed")
        self.assertEqual(out["contextProjection"], {"mode": "thread_bootstrap", "epoch": 1})
        self.assertEqual(e.compact({"sessionId": "a", "tokenBudget": 10 ** 6}), {"ok": True, "compacted": False})
        self.assertEqual(budgets[-1], int(10 ** 6 * 0.5))
        e.compact({"sessionId": "a", "force": True})
        self.assertEqual(budgets[-1], 0, "/compact forces a compaction")

    def test_bridge_and_quarantine(self):
        bridge = BridgeClient(["--engine", "none", "--id", "t-engine"])
        self.addCleanup(bridge.close)
        self.assertEqual(bridge.call("info")["id"], "t-engine")
        bridge.call("ingest", {"sessionId": "s", "message": {"role": "user", "content": "hi"}})
        out = bridge.call("assemble", {"sessionId": "s", "messages": [], "tokenBudget": 100})
        self.assertEqual(out["messages"], [{"role": "user", "content": "hi"}])
        with self.assertRaises(RuntimeError):
            bridge.call("assemble", {})

        class Broken(OpenClawEngine):
            def assemble(self, params):
                raise ValueError("boom")

        host = OpenClawHost(ScriptedModel([say("ok")]), type("T", (), {"schemas": lambda s: [],
                                                                       "call": lambda s, n, a: ("", False)})(),
                            Broken(lambda b: Clear(b)), token_budget=500)
        r = host.run(["hello"])
        self.assertEqual(r.answers, ["ok"], "the reply survives a failing engine")
        self.assertIn("assemble: boom", host.quarantined)


if __name__ == "__main__":
    unittest.main()
