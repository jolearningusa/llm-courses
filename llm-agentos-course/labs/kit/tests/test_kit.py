"""Tests for the kit itself: tools, models, tasks and the suite runner. These must always pass."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import shutil
import subprocess
import sys
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


class TestTicketFixtures(unittest.TestCase):
    """The Lab 7 tools, called directly (the registry that builds their schemas is a Lab 7 TODO)."""

    class Collect:
        def __init__(self):
            self.fns = {}

        def register(self, fn):
            self.fns[fn.__name__] = fn
            return fn

    def test_demos_answer_and_checks_need_work(self):
        from tasks.tickets import TASKS, setup, v1_tools, v2_tools
        with tempfile.TemporaryDirectory() as d:
            ws = Path(d)
            setup(ws)
            v1, v2 = self.Collect(), self.Collect()
            v1_tools(v1, ws)
            v2_tools(v2, ws)
            for task in TASKS:
                with self.subTest(task=task.id):
                    self.assertFalse(task.check("I don't know"))
                    for demo, fns in ((task.demo_v1, v1.fns), (task.demo_v2, v2.fns)):
                        self.assertTrue(task.check(demo[-1]["content"]), demo[-1])
                        for step in demo[:-1]:
                            fns[step["tool_calls"][0]["name"]](**step["tool_calls"][0]["arguments"])
            self.assertGreater(len(v1.fns["search"](q="login")), 8000, "v1 returns everything")
            self.assertLess(len(v2.fns["search_tickets"](query="login")), 1000)


class TestSkillFixtures(unittest.TestCase):
    def test_install_and_demo_reports(self):
        from tasks.skills import TASKS, install
        with tempfile.TemporaryDirectory() as d:
            ws = Path(d)
            root = install(ws, "all")
            self.assertEqual(len(list(root.rglob("SKILL.md"))), 50)
            for task in TASKS:
                with self.subTest(task=task.id):
                    self.assertFalse(task.check(ws, "I don't know"))
                    for demo, ok in ((task.demo_without, not task.needs_skill), (task.demo_with, True)):
                        for step in demo[:-1]:
                            c = step["tool_calls"][0]
                            if c["name"] == "write_file":
                                p = ws / c["arguments"]["path"]
                                p.write_text(c["arguments"]["content"])
                        self.assertEqual(task.check(ws, demo[-1]["content"]), ok)


class TestResearchFixtures(unittest.TestCase):
    def test_answer_keys_and_worker_scripts(self):
        from tasks.research import TASKS
        for task in TASKS:
            with self.subTest(task=task.id):
                self.assertFalse(task.check("I don't know"))
                self.assertTrue(task.check(task.demo_single[-1]["content"]))
                self.assertTrue(task.check(task.demo_parent[-1]["content"]))
                briefs = []
                for step in task.demo_parent:
                    for c in step.get("tool_calls", []):
                        briefs += c["arguments"].get("briefs", []) + [c["arguments"].get("brief")]
                self.assertEqual({b for b in briefs if b}, set(task.demo_workers))
                for script in task.demo_workers.values():
                    for step in script[:-1]:
                        path = step["tool_calls"][0]["arguments"].get("path")
                        self.assertTrue(path in task.files or any(f.startswith(path + "/") for f in task.files))
                        self.assertNotIn("..", path)


class TestNotesServer(unittest.TestCase):
    """The Lab 11 reference MCP server checks requests the way the 2026-07-28 spec asks."""

    def post(self, url, body, headers=None, method="POST"):
        import urllib.error
        import urllib.request
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        h.update(headers or {})
        req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers=h,
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                data = r.read()
                return r.status, json.loads(data) if data else None
        except urllib.error.HTTPError as e:
            data = e.read()
            return e.code, json.loads(data) if data else None

    def test_http_validation(self):
        from tasks.mcp_notes import M, NotesServer, start_http
        server = NotesServer()
        httpd, url = start_http(server)
        self.addCleanup(httpd.shutdown)

        def call(name, args, version="2026-07-28", caps=None, extra=None):
            meta = {M + "protocolVersion": version, M + "clientInfo": {"name": "t", "version": "1"},
                    M + "clientCapabilities": {"elicitation": {}} if caps is None else caps}
            return {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": name, "arguments": args, "_meta": meta, **(extra or {})}}

        good = {"MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/call", "Mcp-Name": "read_note"}
        status, body = self.post(url, call("read_note", {"id": "n03"}), good)
        self.assertEqual(status, 200)
        self.assertIn("Chen Smith", body["result"]["content"][0]["text"])
        for headers in ({**good, "Mcp-Name": "send_note"}, {k: v for k, v in good.items() if k != "Mcp-Method"}):
            status, body = self.post(url, call("read_note", {"id": "n03"}), headers)
            self.assertEqual((status, body["error"]["code"]), (400, -32020))
        status, body = self.post(url, call("search_notes", {"query": "", "folder": "ops"}),
                                 {**good, "Mcp-Name": "search_notes"})
        self.assertEqual((status, body["error"]["code"]), (400, -32020), "Mcp-Param-Folder is required")
        status, body = self.post(url, call("read_note", {"id": "n03"}, "2027-01-01"),
                                 {**good, "MCP-Protocol-Version": "2027-01-01"})
        self.assertEqual((status, body["error"]["code"], body["error"]["data"]["supported"]),
                         (400, -32022, ["2026-07-28"]))
        status, body = self.post(url, {"jsonrpc": "2.0", "id": 2, "method": "nope/x", "params": {
            "_meta": {M + "protocolVersion": "2026-07-28"}}}, {**good, "Mcp-Method": "nope/x"})
        self.assertEqual((status, body["error"]["code"]), (404, -32601))
        self.assertEqual(self.post(url, {"jsonrpc": "2.0", "method": "notifications/cancelled"})[0], 202)
        self.assertEqual(self.post(url, None, method="GET")[0], 405)
        status, _ = self.post(url, call("read_note", {"id": "n03"}), {**good, "Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        send = {**good, "Mcp-Name": "send_note"}
        args = {"id": "n01", "to": "a@example.org"}
        status, body = self.post(url, call("send_note", args, caps={}), send)
        self.assertEqual((status, body["error"]["code"]), (400, -32021))
        status, body = self.post(url, call("send_note", args), send)
        r = body["result"]
        self.assertEqual(r["resultType"], "input_required")
        self.assertEqual(r["inputRequests"]["confirm"]["method"], "elicitation/create")
        accept = {"confirm": {"action": "accept", "content": {"confirm": True}}}
        forged = call("send_note", {**args, "to": "b@example.org"},
                      extra={"inputResponses": accept, "requestState": r["requestState"]})
        status, body = self.post(url, forged, send)
        self.assertEqual(body["error"]["code"], -32602, "requestState is bound to the call's arguments")
        status, body = self.post(url, call("send_note", args, extra={"inputResponses": accept,
                                                                     "requestState": r["requestState"]}), send)
        self.assertFalse(body["result"]["isError"])
        self.assertEqual(server.sent, [args])

    def test_tasks_need_the_tools(self):
        from tasks.mcp_notes import TASKS
        for task in TASKS:
            with self.subTest(task=task.id):
                self.assertFalse(task.check("I don't know", [{"id": "n99", "to": "x@example.org"}]))
                self.assertTrue(task.demo[-1]["content"])


class TestPortsProvided(unittest.TestCase):
    """Lab 12's provided pieces: the budget cut, and the plugin files that load the kit into Hermes and OpenClaw."""

    PLUGINS = Path(__file__).resolve().parent.parent.parent / "lab1_memory_plugins"

    def test_fit_budget(self):
        from nanoos.context import count_tokens
        from nanoos.ports import fit_budget
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
        for i in range(6):
            msgs += [{"role": "assistant", "content": None, "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 400}]
        kept = fit_budget(msgs, 450)
        self.assertEqual(kept[0]["role"], "system")
        self.assertNotEqual(kept[1]["role"], "tool", "never starts the tail with an orphaned tool result")
        self.assertLessEqual(count_tokens(kept), 450)
        self.assertEqual(fit_budget(msgs, None), msgs)

    def test_hermes_plugin_registers_both(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("hermes_nanoos_plugin", self.PLUGINS / "hermes_custom_memory.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        names = []
        ctx = type("Ctx", (), {"register_memory_provider": lambda s, p: names.append(p.name),
                               "register_context_engine": lambda s, e: names.append(e.name)})()
        mod.register(ctx)
        self.assertEqual(names, ["nanoos-memory", "nanoos-context"])

    def test_openclaw_shell_forwards_to_python(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")
        major, minor = (int(x) for x in subprocess.run([node, "--version"], capture_output=True, text=True)
                        .stdout.strip().lstrip("v").split(".")[:2])
        if (major, minor) < (22, 6):
            self.skipTest("node 22.6 or newer is needed to run TypeScript directly")
        kit = Path(__file__).resolve().parent.parent
        driver = """
const [ts, kit] = process.argv.slice(1);
const mod = await import(ts);
let factory;
mod.default({ registerContextEngine: (id, f) => { factory = f; } });
const eng = factory({ config: { kitDir: kit, python: process.env.PY } });
const out = { info: eng.info };
out.ingest = await eng.ingest({ sessionId: "s", message: { role: "user", content: "hi" } });
try { await eng.assemble({}); } catch (e) { out.error = e.message; }
await eng.dispose();
console.log(JSON.stringify(out));
"""
        proc = subprocess.run([node, "--experimental-strip-types", "--no-warnings", "--input-type=module", "-e", driver,
                               str(self.PLUGINS / "openclaw_context_engine.ts"), str(kit)], capture_output=True,
                              text=True, timeout=60, env={**os.environ, "PY": sys.executable})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout)
        self.assertEqual((out["info"]["id"], out["info"]["ownsCompaction"]), ("nanoos-engine", True))
        self.assertEqual(out["ingest"], {"ingested": True})
        self.assertIn("nanoos engine:", out["error"], "a Python error rejects the call")


class TestObservabilityProvided(unittest.TestCase):
    """Lab 13's provided pieces: OTLP encoding and export, the offline judge, the calibration set, obs_eval.py."""

    def spans(self):
        from nanoos.otel import Span
        root = Span("invoke_agent nanoos", "a" * 32, "1" * 16, None, 1_000, 9_000, attributes={"n": 3, "ok": True})
        tool = Span("execute_tool read_file", "a" * 32, "2" * 16, "1" * 16, 2_000, 5_000, status="ERROR",
                    attributes={"x": 0.5, "names": ["a", "b"]}, events=[{"name": "retry", "time_ns": 3_000,
                                                                          "attributes": {"attempt": 1}}])
        return [root, tool]

    def test_otlp_round_trip_and_tree(self):
        from nanoos.otel import from_otlp, span_tree, to_otlp
        body = to_otlp(self.spans(), "svc")
        rs = body["resourceSpans"][0]
        self.assertIn({"key": "service.name", "value": {"stringValue": "svc"}}, rs["resource"]["attributes"])
        raw = rs["scopeSpans"][0]["spans"]
        self.assertEqual(raw[1]["parentSpanId"], "1" * 16)
        self.assertNotIn("parentSpanId", raw[0])
        self.assertEqual(raw[1]["status"]["code"], 2)
        back = from_otlp(json.loads(json.dumps(body)))
        self.assertEqual(back[0].attributes, {"n": 3, "ok": True})
        self.assertEqual(back[1].attributes, {"x": 0.5, "names": ["a", "b"]})
        self.assertEqual(back[1].events[0]["attributes"], {"attempt": 1})
        self.assertEqual(span_tree(back).splitlines(),
                         ["invoke_agent nanoos  0.0 ms", "  execute_tool read_file  0.0 ms  ERROR"])

    def test_http_exporter_posts_json_and_fails_open(self):
        import http.server
        import threading
        from nanoos.otel import OtlpHttpExporter
        got = []

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                got.append((self.path, self.headers["Content-Type"],
                            json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            OtlpHttpExporter(f"http://127.0.0.1:{srv.server_port}").export(self.spans())
        finally:
            srv.shutdown()
        self.assertEqual(got[0][:2], ("/v1/traces", "application/json"))
        self.assertEqual(len(got[0][2]["resourceSpans"][0]["scopeSpans"][0]["spans"]), 2)
        dead = OtlpHttpExporter(f"http://127.0.0.1:{srv.server_port}", timeout_s=0.5)
        with contextlib.redirect_stdout(io.StringIO()):
            dead.export(self.spans())
        self.assertEqual(dead.failures, 1)

    def test_calibration_set_and_offline_judge(self):
        from models.substring_judge import SubstringJudge
        from nanoos.graders import judge_messages, load_cases
        cases = load_cases()
        self.assertEqual(len(cases), 30)
        self.assertEqual(len({c["id"] for c in cases}), 30)
        self.assertEqual({c["label"] for c in cases}, {"pass", "fail"})
        self.assertTrue(all(c["question"] and c["reference"] and c["answer"] and c["why"] for c in cases))
        j = SubstringJudge()
        self.assertTrue(j.complete(judge_messages("q", "4210", "West totals 4,210."), []).content.endswith("PASS"))
        self.assertTrue(j.complete(judge_messages("q", "30", "set to 300"), []).content.endswith("FAIL"))

    def test_obs_eval_runs_at_any_stage(self):
        import obs_eval
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = obs_eval.main(["--config", "base=loop=baseline", "--tasks", "b01,b07", "--trials", "2",
                                  "--judge", "none", "--out", tmp])
            self.assertEqual(code, 0)
            rows = list(csv.DictReader(open(Path(tmp) / "base" / "metrics.csv", encoding="utf-8")))
            self.assertEqual([r["success"] for r in rows], ["1"] * 4)
            self.assertIn("pass@1", (Path(tmp) / "dashboard.html").read_text(encoding="utf-8"))


class TestImproveProvided(unittest.TestCase):
    """Lab 14's provided pieces: the simulated agent, edits, the harness runner and the scripted proposer."""

    def test_habits_follow_the_harness(self):
        from nanoos.improve import READ_FIRST, WRITE_STRICT
        from tasks.improve_sim import active_habits
        tools = WorkspaceTools(Path(tempfile.mkdtemp())).schemas()
        sys_msg = [{"role": "system", "content": "You are an assistant."}]
        self.assertEqual(active_habits("b01_count_lines", sys_msg, tools), ["guess"])
        self.assertEqual(active_habits("b01_count_lines", [{"role": "system", "content": READ_FIRST}], tools), [])
        self.assertEqual(active_habits("b03_fix_typo", sys_msg, tools), [])
        strict = [dict(t, function=dict(t["function"], description=WRITE_STRICT))
                  if t["function"]["name"] == "write_file" else t for t in tools]
        self.assertEqual(active_habits("b03_fix_typo", sys_msg, strict), ["no_overwrite"])
        self.assertEqual(active_habits("b08_collect_todos", sys_msg, strict), [])

    def test_edits_runner_and_scores(self):
        from nanoos.improve import (Edit, Evaluator, HarnessConfig, ScriptedProposer, apply_edit, conflicts,
                                    failure_report, success)
        from tasks.basic import TASKS
        from tasks.improve_sim import DEV, HELDOUT, HabitModel
        cfg = HarnessConfig()
        new = apply_edit(cfg, Edit("tool:write_file", "set", "Paths are relative to the workspace root."))
        self.assertEqual(cfg.tool_descriptions, {})
        self.assertTrue(conflicts(Edit("setting:max_steps", "set", 8), Edit("setting:max_steps", "set", 9)))
        self.assertFalse(conflicts(Edit("system_prompt", "append", "a"), Edit("system_prompt", "append", "b")))
        ev = Evaluator(TASKS, HabitModel, Path(tempfile.mkdtemp()))
        dev = ev.run(cfg, DEV)
        self.assertAlmostEqual(success(dev), 2 / 6)
        self.assertAlmostEqual(success(ev.run(cfg, HELDOUT)), 3 / 4)
        self.assertEqual(sorted(t for t, r in ev.run(new, ["b08_collect_todos"]).items() if r[0].success),
                         ["b08_collect_todos"])
        report = failure_report(dev, {t.id: t.prompt for t in TASKS})
        self.assertEqual({r["task_id"]: r["signature"] for r in report},
                         {"b01_count_lines": "answered_without_tools", "b04_csv_sum": "answered_without_tools",
                          "b08_collect_todos": "bad_write", "b09_urgent_refund": "wrong_answer"})
        p = ScriptedProposer()
        first = p.propose(cfg, report, [])
        self.assertEqual(len(first), 3)
        again = p.propose(cfg, report, [e.describe() for e in first])
        self.assertTrue(all(e.describe() not in {x.describe() for x in first} for e in again))

    def test_improve_eval_runs_at_any_stage(self):
        import improve_eval
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = improve_eval.main(["--rounds", "1", "--out", tmp])
            self.assertIn(code, (0, 1))
            self.assertTrue((Path(tmp) / "config_before.json").is_file())
