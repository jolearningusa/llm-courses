"""Tests for the kit itself: tools, models, tasks and the suite runner. These must always pass."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from models.flaky import Flaky
from models.openai_compat import parse_response
from models.scripted import ModelError, ScriptedModel
from nanoos import baseline
from nanoos.tools_basic import WorkspaceTools
from run_suite import COLUMNS, main as run_suite, pass_at_k, pass_hat_k
from tasks import load_tier


class TestTasks(unittest.TestCase):
    def test_ten_basic_tasks_with_unique_ids(self):
        tasks = load_tier("basic")
        self.assertEqual(len(tasks), 10)
        self.assertEqual(len({t.id for t in tasks}), 10)

    def test_demo_script_solves_each_task(self):
        for task in load_tier("basic"):
            with self.subTest(task=task.id), tempfile.TemporaryDirectory() as d:
                ws = Path(d)
                task.setup(ws)
                result = baseline.run(ScriptedModel(task.demo), task.prompt, WorkspaceTools(ws))
                self.assertEqual(result.stopped_by, "answer")
                self.assertEqual(result.tool_errors, 0)
                self.assertTrue(task.check(ws, result.answer), result.answer)

    def test_no_task_passes_without_work(self):
        for task in load_tier("basic"):
            with self.subTest(task=task.id), tempfile.TemporaryDirectory() as d:
                task.setup(Path(d))
                self.assertFalse(task.check(Path(d), "I don't know"))


class TestLongTier(unittest.TestCase):
    def test_long_demos_solve_with_file_tools_only(self):
        for task in load_tier("long"):
            with self.subTest(task=task.id), tempfile.TemporaryDirectory() as d:
                ws = Path(d)
                task.setup(ws)
                self.assertFalse(task.check(ws, "I don't know"))
                tools = WorkspaceTools(ws, task.tools)
                self.assertNotIn("search", {s["function"]["name"] for s in tools.schemas()})
                self.assertTrue(tools.call("search", {"pattern": "x"})[1], "a disallowed tool must fail")
                result = baseline.run(ScriptedModel(task.demo), task.prompt, tools, max_steps=60)
                self.assertGreaterEqual(result.steps, 40)
                self.assertTrue(task.check(ws, result.answer), task.id)


class TestTools(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.ws = Path(self.dir.name)
        (self.ws / "a.txt").write_text("hello\nworld\n")
        self.tools = WorkspaceTools(self.ws)

    def tearDown(self):
        self.dir.cleanup()

    def test_read_write_search_list(self):
        self.assertEqual(self.tools.call("read_file", {"path": "a.txt"}), ("hello\nworld\n", False))
        self.assertFalse(self.tools.call("write_file", {"path": "d/b.txt", "content": "x"})[1])
        self.assertEqual(self.tools.call("list_files", {})[0], "a.txt\nd/b.txt")
        self.assertEqual(self.tools.call("search", {"pattern": "wor"})[0], "a.txt:2: world")

    def test_errors_are_reported_not_raised(self):
        for name, args in [("read_file", {"path": "../etc/passwd"}), ("read_file", {"path": "missing.txt"}),
                           ("read_file", {}), ("nope", {}), ("search", {"pattern": "("})]:
            with self.subTest(name=name, args=args):
                out, is_error = self.tools.call(name, args)
                self.assertTrue(is_error)
                self.assertTrue(out.startswith("ERROR"))

    def test_schemas_name_every_tool(self):
        names = {s["function"]["name"] for s in self.tools.schemas()}
        self.assertEqual(names, {"list_files", "read_file", "write_file", "search"})


class TestModels(unittest.TestCase):
    def test_scripted_model_replays_records_and_raises(self):
        m = ScriptedModel([{"tool_calls": [{"name": "t", "arguments": {"x": 1}}]}, {"raise": "429"}, {"content": "done"}])
        r = m.complete([{"role": "user", "content": "hi"}], [])
        self.assertEqual((r.tool_calls[0].name, r.tool_calls[0].arguments), ("t", {"x": 1}))
        self.assertGreater(r.usage.input_tokens, 0)
        with self.assertRaises(ModelError):
            m.complete([], [])
        self.assertEqual(m.complete([], []).content, "done")
        self.assertEqual(m.complete([], []).content, "(script ended)")
        self.assertEqual(len(m.calls), 4)

    def test_parse_openai_response(self):
        r = parse_response({"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"a\"}"}},
            {"id": "c2", "type": "function", "function": {"name": "search", "arguments": "{bad json"}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 64}}})
        self.assertEqual([c.name for c in r.tool_calls], ["read_file", "search"])
        self.assertEqual(r.tool_calls[0].arguments, {"path": "a"})
        self.assertIn("_unparsed", r.tool_calls[1].arguments)
        self.assertEqual((r.usage.input_tokens, r.usage.cached_input_tokens, r.usage.output_tokens), (100, 64, 7))


class TestFlaky(unittest.TestCase):
    def test_fail_rate_bounds(self):
        with self.assertRaises(ModelError):
            Flaky(ScriptedModel([{"content": "x"}]), 1.0).complete([], [])
        self.assertEqual(Flaky(ScriptedModel([{"content": "x"}]), 0.0).complete([], []).content, "x")

    def test_suite_records_model_errors(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                run_suite(["--model", "demo", "--tasks", "b01", "--fail-rate", "1", "--out", str(out)])
            with (out / "metrics.csv").open() as f:
                row = next(csv.DictReader(f))
            self.assertEqual((row["success"], row["stopped_by"]), ("0", "error"))


class TestSuite(unittest.TestCase):
    def test_pass_metrics(self):
        self.assertEqual(pass_at_k(5, 0, 5), 0.0)
        self.assertEqual(pass_at_k(5, 1, 5), 1.0)
        self.assertEqual(pass_hat_k(5, 4, 5), 0.0)
        self.assertEqual(pass_hat_k(5, 5, 5), 1.0)
        self.assertAlmostEqual(pass_at_k(4, 2, 2), 1 - 1 / 6)
        self.assertAlmostEqual(pass_hat_k(4, 2, 2), 1 / 6)

    def test_demo_run_writes_metrics_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(run_suite(["--model", "demo", "--trials", "2", "--out", str(out)]), 0)
            with (out / "metrics.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(list(rows[0].keys()), COLUMNS)
            self.assertEqual(len(rows), 20)
            self.assertTrue(all(r["success"] == "1" for r in rows))
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual((summary["success_rate"], summary["pass^2"]), (1.0, 1.0))


class TestMemoryFixtures(unittest.TestCase):
    def test_question_counts_and_demo_answers(self):
        from tasks.memory import QUESTIONS
        self.assertEqual(sum(q.needs_memory for q in QUESTIONS), 20)
        self.assertEqual(sum(not q.needs_memory for q in QUESTIONS), 20)
        self.assertEqual(len({q.id for q in QUESTIONS}), 40)
        for q in QUESTIONS:
            with self.subTest(q=q.id):
                self.assertTrue(q.check(q.demo_answer), q.demo_answer)
                abstain_kind = q.kind in ("forget", "other_user", "absent") or (q.kind == "poison" and not q.gold)
                self.assertEqual(q.check("I don't know."), abstain_kind)
                if q.stale:
                    self.assertFalse(q.check(q.stale))

    def test_every_user_turn_and_tool_result_has_gold_ops(self):
        from tasks.memory import SESSIONS, gold_ops, session_messages
        ops = gold_ops()
        for user, sessions in SESSIONS.items():
            for n in range(len(sessions)):
                for m in session_messages(user, n):
                    if m["role"] in ("user", "tool"):
                        self.assertIn(m["content"], ops)

    def test_parse_ops_and_model_extractor(self):
        from nanoos.memory import ModelExtractor, parse_ops
        self.assertEqual(parse_ops('Sure! [{"op": "ADD", "text": "x"}, 3] done'), [{"op": "ADD", "text": "x"}])
        self.assertEqual(parse_ops("no list [here"), [])
        ex = ModelExtractor(ScriptedModel([{"content": '[{"op": "NOOP"}]'}]))
        self.assertEqual(ex("alice", "hi", []), [{"op": "NOOP"}])
        self.assertGreater(ex.usage.input_tokens, 0)


class TestCacheSession(unittest.TestCase):
    def test_baseline_is_append_only_and_variants_differ(self):
        from nanoos.prompt import render
        from tasks.cache_session import VARIANTS, session_prompt
        prev = None
        for t in range(20):
            cur = render(session_prompt("baseline", t))
            if prev is not None:
                self.assertTrue(cur.startswith(prev), f"turn {t}")
            prev = cur
        last = {v: render(session_prompt(v, 19)) for v in VARIANTS}
        self.assertEqual(len(set(last.values())), len(VARIANTS))
        self.assertNotEqual(render(session_prompt("baseline", 3, session=1)), render(session_prompt("baseline", 3)))


class TestRedteamFixtures(unittest.TestCase):
    def test_benign_demos_pass_and_checks_need_work(self):
        from models.gullible import GullibleModel
        from tasks.redteam import TASKS, RedteamTools
        for task in TASKS:
            with self.subTest(task=task.id), tempfile.TemporaryDirectory() as d:
                ws = Path(d)
                task.setup(ws)
                self.assertFalse(task.check(ws, "I don't know", RedteamTools(ws)))
                env = RedteamTools(ws)
                result = baseline.run(GullibleModel(task.demo), task.prompt, env, max_steps=12)
                self.assertTrue(task.check(ws, result.answer, env), result.answer)
                self.assertEqual(env.leaked(), task.injected is not None, "the gullible model obeys every injection")

    def test_reader_keeps_facts_and_drops_injections(self):
        from nanoos.policy import strip_instructions
        from tasks.redteam import PAGES, STEPS_RE
        page = strip_instructions(PAGES["https://docs.vendor.example/release-notes"])
        self.assertIn("4.2.1", page)
        self.assertNotIn("AI agents", page)
        self.assertFalse(STEPS_RE.search(page))
        self.assertIn("attacker.example", page, "a filter misses the sentence that does not name an AI")
