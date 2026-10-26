"""Lab 5 autograder tests: nanoos/prompt.py (block hashes, a simulated prefix cache, break diagnosis)."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

from cache_experiment import main as cache_experiment
from models.scripted import ScriptedModel
from nanoos.memory import MEMORY_HEADER
from nanoos.prompt import CacheAuditModel, PrefixCache, Prompt, block_hashes, cache_tokens, diagnose, render

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
SYSTEM = "You are careful. " * 40


def prompt(*user_turns, system=SYSTEM, tools=TOOLS):
    msgs = [{"role": "system", "content": system}]
    for u in user_turns:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": f"ok, {u}"}]
    return Prompt(msgs[:-1], tools)


class TestBlockHashes(unittest.TestCase):
    def test_full_blocks_only_and_the_exact_chain(self):
        toks = [f"t{i}" for i in range(37)]
        hs = block_hashes(toks, block_size=16)
        self.assertEqual(len(hs), 2)
        h0 = hashlib.sha256(("" + "\x00" + "\x00".join(toks[:16])).encode()).hexdigest()
        h1 = hashlib.sha256((h0 + "\x00" + "\x00".join(toks[16:32])).encode()).hexdigest()
        self.assertEqual(hs, [h0, h1])

    def test_a_change_invalidates_its_block_and_every_later_one(self):
        toks = [f"t{i}" for i in range(64)]
        changed = list(toks)
        changed[20] = "X"
        a, b = block_hashes(toks), block_hashes(changed)
        self.assertEqual(a[0], b[0])
        self.assertTrue(all(x != y for x, y in zip(a[1:], b[1:])))

    def test_salt_changes_every_block(self):
        toks = [f"t{i}" for i in range(48)]
        self.assertTrue(all(x != y for x, y in zip(block_hashes(toks), block_hashes(toks, salt="tenant-b"))))


class TestPrefixCache(unittest.TestCase):
    def test_hits_are_the_leading_cached_blocks(self):
        c = PrefixCache(100)
        self.assertEqual(c.access(["a", "b", "c"]), 0)
        self.assertEqual(c.access(["a", "b", "c", "d"]), 3)
        self.assertEqual(c.access(["x", "b", "c"]), 0, "a miss at block 0 stops the walk")

    def test_lru_eviction_takes_the_tail_first(self):
        c = PrefixCache(4)
        c.access(["a", "b", "c"])
        c.access(["x", "y"])  # 5 blocks: one must go, and it is the tail of the older request
        self.assertEqual(c.evictions, 1)
        self.assertNotIn("c", c.blocks)
        self.assertEqual(c.access(["a", "b", "c"]), 2)

    def test_recently_used_blocks_survive(self):
        c = PrefixCache(4)
        c.access(["a", "b"])
        c.access(["x", "y"])
        c.access(["a", "b"])  # a and b are now the most recent
        c.access(["p"])
        self.assertIn("a", c.blocks)
        self.assertNotIn("y", c.blocks)


class TestDiagnose(unittest.TestCase):
    def test_first_and_append_only(self):
        self.assertEqual(diagnose(None, prompt("one")).cause, "first")
        b = diagnose(prompt("one"), prompt("one", "two"))
        self.assertEqual((b.cause, b.block, b.segment), ("none", None, None))

    def test_tool_list_change(self):
        more = TOOLS + [{"type": "function", "function": {"name": "web_fetch", "parameters": {"type": "object"}}}]
        b = diagnose(prompt("one"), prompt("one", "two", tools=more))
        self.assertEqual((b.cause, b.segment), ("tools", "tools"))

    def test_timestamp_in_the_system_prompt(self):
        b = diagnose(prompt("one", system="Time: 09:01\n" + SYSTEM), prompt("one", "two", system="Time: 09:02\n" + SYSTEM))
        self.assertEqual((b.cause, b.segment), ("timestamp", "system"))

    def test_memory_edit_and_other_system_edits(self):
        mem = lambda fact: SYSTEM + MEMORY_HEADER + "- The user lives in " + fact  # noqa: E731
        b = diagnose(prompt("one", system=mem("Seattle.")), prompt("one", "two", system=mem("Denver.")))
        self.assertEqual(b.cause, "memory")
        b = diagnose(prompt("one"), prompt("one", "two", system="Be brief. " + SYSTEM))
        self.assertEqual(b.cause, "system")

    def test_history_rewrite_and_the_block_index(self):
        prev = prompt("one", "two", "three")
        cur = prompt("ONE", "two", "three", "four")
        b = diagnose(prev, cur)
        self.assertEqual((b.cause, b.segment), ("history", "message:1:user"))
        a, c = cache_tokens(render(prev)), cache_tokens(render(cur))
        first = next(i for i in range(len(a)) if a[i] != c[i])
        self.assertEqual(b.block, first // 16)


class TestAudit(unittest.TestCase):
    def test_audit_model_records_hits_per_call(self):
        m = CacheAuditModel(ScriptedModel([{"content": "a"}, {"content": "b"}]))
        p1, p2 = prompt("one"), prompt("one", "two")
        m.complete(p1.messages, p1.tools)
        m.complete(p2.messages, p2.tools)
        self.assertEqual([r["cause"] for r in m.rows], ["first", "none"])
        self.assertEqual(m.rows[0]["hit_tokens"], 0)
        self.assertEqual(m.rows[1]["hit_tokens"], (len(cache_tokens(render(p1))) // 16) * 16)


class TestExperiment(unittest.TestCase):
    def run_rows(self, *args):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            cache_experiment(["--quiet", "--csv", str(Path(d) / "r.csv"), *args])
            with open(Path(d) / "r.csv", encoding="utf-8") as f:
                return list(csv.DictReader(f))

    def test_each_variant_breaks_where_it_should(self):
        rows = self.run_rows()
        breaks = {}
        for r in rows:
            if r["cause"] not in ("first", "none"):
                breaks.setdefault(r["variant"], []).append((int(r["turn"]), r["cause"]))
        self.assertNotIn("baseline", breaks)
        self.assertNotIn("masked_tool_10", breaks, "a masked tool keeps the prefix")
        self.assertEqual(breaks["tool_at_10"], [(10, "tools")])
        self.assertEqual(breaks["memory_at_10"], [(10, "memory")])
        self.assertEqual(breaks["compaction_15"], [(15, "history")])
        self.assertEqual(breaks["timestamp"], [(t, "timestamp") for t in range(1, 20)])

        def rate(v):
            rs = [r for r in rows if r["variant"] == v]
            return sum(int(r["hit_tokens"]) for r in rs) / sum(int(r["prompt_tokens"]) for r in rs)
        self.assertGreater(rate("baseline"), 0.85)
        self.assertLess(rate("timestamp"), 0.2)

    def test_too_little_kv_evicts_across_sessions(self):
        def rate(rows):
            return sum(int(r["hit_tokens"]) for r in rows) / sum(int(r["prompt_tokens"]) for r in rows)
        roomy = rate(self.run_rows("--variants", "baseline", "--sessions", "6", "--cache-blocks", "100000"))
        tight = rate(self.run_rows("--variants", "baseline", "--sessions", "6", "--cache-blocks", "2000"))
        self.assertGreater(roomy, 0.9)
        self.assertLess(tight, roomy - 0.2)
