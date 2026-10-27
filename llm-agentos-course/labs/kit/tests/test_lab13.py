"""Lab 13 autograder tests: nanoos/otel.py (redaction, the span tree) and nanoos/graders.py.

The span tests drive the provided baseline loop with scripted models, so this lab's grade does not depend on
Lab 1. The last test reports the offline judge's agreement with the 30 human-labeled cases.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.scripted import ScriptedModel
from models.substring_judge import SubstringJudge
from nanoos import baseline
from nanoos.graders import (LLMJudge, agreement, convergence, exact_match, load_cases, normalize, parse_verdict,
                            trajectory)
from nanoos.otel import InMemoryExporter, OtelTracer, OtlpFileExporter, Redactor
from nanoos.tools_basic import WorkspaceTools
from tasks.basic import TASKS, call, say

FAKE_KEY = "sk-nanoos-FAKE0123456789abcdef"  # not a real key: the fixture a trace must not leak
ENV_FILE = f"# service settings\nAPI_KEY={FAKE_KEY}\nREGION=eu-west-1\n"


class Clock:
    """A clock the test moves by hand."""

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def run_traced(script, files=None, task_prompt="task", capture="redacted", exporter=None, redactor=None):
    exporter = exporter or InMemoryExporter()
    tmp = Path(tempfile.mkdtemp())
    for rel, content in (files or {}).items():
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / rel).write_text(content, encoding="utf-8")
    tracer = OtelTracer(exporter, run_id="r1", capture_content=capture, redactor=redactor)
    tracer.event("run_start", task="t", trial=0, model="demo")
    result = baseline.run(ScriptedModel(script, name="demo"), task_prompt, WorkspaceTools(tmp), tracer)
    tracer.close()
    return result, exporter


def by_name(spans, prefix):
    return [s for s in spans if s.name.startswith(prefix)]


def children(spans, parent):
    """A span's children in the order they were opened (the order close() exports them in)."""
    return [s for s in spans if s.parent_id == parent.span_id]


class TestRedactor(unittest.TestCase):
    def test_patterns_and_known_secrets(self):
        r = Redactor(secrets=["hunter2-very-secret"])
        text = (f"key {FAKE_KEY} aws AKIAIOSFODNN7EXAMPLE mail ana.lopez1@example.org "
                "pw hunter2-very-secret auth: Bearer abcdefghijklmnop1234")
        out = r.redact(text)
        for leaked in (FAKE_KEY, "AKIAIOSFODNN7EXAMPLE", "ana.lopez1@example.org", "hunter2-very-secret",
                       "abcdefghijklmnop1234"):
            self.assertNotIn(leaked, out)
        for label in ("[REDACTED:api_key]", "[REDACTED:aws_key]", "[REDACTED:email]", "[REDACTED:secret]",
                      "[REDACTED:bearer]"):
            self.assertIn(label, out)
        self.assertIn("key ", out)  # the text around a secret stays

    def test_longest_secret_first_truncation_and_nesting(self):
        r = Redactor(secrets=["abcd1234", "abcd1234efgh5678"], patterns=[], max_chars=0)
        self.assertEqual(r.redact("x abcd1234efgh5678 y"), "x [REDACTED:secret] y")
        r = Redactor(secrets=[], patterns=[], max_chars=10)
        self.assertEqual(r.redact("0123456789abc"), "0123456789...[truncated 3 chars]")
        self.assertEqual(r.redact("short"), "short")
        r = Redactor(secrets=["s3cret-value"], patterns=[])
        self.assertEqual(r.redact_value({"a": ["s3cret-value", 3], "b": {"c": "ok s3cret-value"}}),
                         {"a": ["[REDACTED:secret]", 3], "b": {"c": "ok [REDACTED:secret]"}})


class TestSpanTree(unittest.TestCase):
    def test_tree_shape_on_a_scripted_run(self):
        task = next(t for t in TASKS if t.id == "b07_two_hop")
        result, exp = run_traced(task.demo, task.files, task.prompt)
        spans = exp.spans
        roots = [s for s in spans if s.parent_id is None]
        self.assertEqual(len(roots), 1, "one root span per run")
        root = roots[0]
        self.assertEqual(root.name, "invoke_agent nanoos")
        self.assertEqual(root.attributes.get("gen_ai.operation.name"), "invoke_agent")
        self.assertEqual(root.attributes.get("gen_ai.conversation.id"), "r1")
        self.assertEqual(root.attributes.get("gen_ai.request.model"), "demo")
        self.assertEqual(root.attributes.get("nanoos.task"), "t")
        self.assertEqual(root.attributes.get("nanoos.stopped_by"), "answer")
        self.assertEqual(root.attributes.get("nanoos.steps"), 3)
        self.assertEqual({s.trace_id for s in spans}, {root.trace_id})
        self.assertEqual(len({s.span_id for s in spans}), len(spans), "span ids are unique")
        turns = children(spans, root)
        self.assertEqual([t.name for t in turns], ["turn 0", "turn 1", "turn 2"])
        shape = [[c.name for c in children(spans, t)] for t in turns]
        self.assertEqual(shape, [["chat demo", "execute_tool search"], ["chat demo", "execute_tool search"],
                                 ["chat demo"]])
        chats = by_name(spans, "chat ")
        self.assertTrue(all(c.kind == "CLIENT" for c in chats))
        self.assertEqual(chats[0].attributes.get("gen_ai.response.finish_reasons"), ["tool_calls"])
        self.assertEqual(chats[-1].attributes.get("gen_ai.response.finish_reasons"), ["stop"])
        self.assertEqual(sum(c.attributes["gen_ai.usage.input_tokens"] for c in chats), result.usage.input_tokens)
        self.assertEqual(sum(c.attributes["gen_ai.usage.output_tokens"] for c in chats), result.usage.output_tokens)
        tool = by_name(spans, "execute_tool ")[0]
        self.assertEqual(tool.attributes.get("gen_ai.tool.name"), "search")
        self.assertEqual(tool.attributes.get("gen_ai.operation.name"), "execute_tool")
        ids = {s.span_id: s for s in spans}
        for s in spans:
            self.assertIsNotNone(s.end_ns, f"{s.name} was never ended")
            if s.parent_id:
                p = ids[s.parent_id]
                self.assertTrue(p.start_ns <= s.start_ns and s.end_ns <= p.end_ns, f"{s.name} sticks out of {p.name}")

    def test_tool_error_sets_status(self):
        _, exp = run_traced([call("read_file", path="missing.txt"), say("done")])
        tool = by_name(exp.spans, "execute_tool read_file")[0]
        self.assertEqual(tool.status, "ERROR")
        self.assertEqual(tool.attributes.get("error.type"), "tool_error")
        self.assertNotEqual(by_name(exp.spans, "invoke_agent")[0].status, "ERROR", "a tool error is not a failed run")

    def test_model_timing_retrieval_and_other_events(self):
        clock, exp = Clock(), InMemoryExporter()
        tr = OtelTracer(exp, run_id="r2", clock=clock)
        tr.event("run_start", model="m")
        clock.t += 2.0
        tr.event("model_call", step=0, input_tokens=10, cached_input_tokens=4, output_tokens=2, latency_s=0.5,
                 n_tool_calls=1)
        clock.t += 0.25
        tr.event("retrieval", step=0, source="memory", query="where do I live", n_results=3, latency_s=0.25)
        clock.t += 1.0
        tr.event("retry", step=1, model="m", attempt=1, error="429")
        tr.event("model_call", step=1, input_tokens=12, output_tokens=3, latency_s=0.0, n_tool_calls=0)
        tr.event("run_end", stopped_by="answer", steps=2)
        tr.close()
        spans = exp.spans
        root = by_name(spans, "invoke_agent")[0]
        t0, t1 = by_name(spans, "turn 0")[0], by_name(spans, "turn 1")[0]
        chat0 = children(spans, t0)[0]
        self.assertEqual(chat0.name, "chat m")
        self.assertEqual(chat0.end_ns - chat0.start_ns, 500_000_000, "a model span covers latency_s before the event")
        self.assertEqual(chat0.attributes.get("gen_ai.usage.cache_read.input_tokens"), 4)
        self.assertLessEqual(t0.start_ns, chat0.start_ns)
        ret = by_name(spans, "retrieval")[0]
        self.assertEqual(ret.name, "retrieval memory")
        self.assertEqual(ret.parent_id, t0.span_id)
        self.assertEqual(ret.attributes.get("gen_ai.data_source.id"), "memory")
        self.assertEqual(ret.attributes.get("gen_ai.operation.name"), "retrieval")
        self.assertEqual(ret.attributes.get("gen_ai.retrieval.query.text"), "where do I live")
        self.assertEqual(ret.attributes.get("nanoos.n_results"), 3)
        self.assertEqual([e["name"] for e in t1.events], ["retry"])
        self.assertEqual(t1.events[0]["attributes"].get("attempt"), 1)
        self.assertLessEqual(t0.end_ns, t1.start_ns, "a turn ends when the next one starts")
        self.assertEqual(root.end_ns, int(clock.t * 1e9))

    def test_failed_run_and_close_once(self):
        class Counting(InMemoryExporter):
            calls = 0

            def export(self, spans):
                self.calls += 1
                super().export(spans)

        exp = Counting()
        tr = OtelTracer(exp, run_id="r3")
        tr.event("run_start", model="m")
        tr.event("model_call", step=0, input_tokens=1, output_tokens=1, latency_s=0, n_tool_calls=1)
        tr.event("run_end", stopped_by="error", error="model failed")
        tr.close()
        tr.close()
        self.assertEqual(exp.calls, 1, "close() exports once")
        self.assertEqual(by_name(exp.spans, "invoke_agent")[0].status, "ERROR")
        self.assertTrue(all(s.end_ns is not None for s in exp.spans))


class TestContentCapture(unittest.TestCase):
    script = [call("read_file", path="service.env"), say("The region is eu-west-1.")]

    def test_redaction_is_on_by_default(self):
        out = Path(tempfile.mkdtemp()) / "spans.json"
        run_traced(self.script, {"service.env": ENV_FILE}, exporter=OtlpFileExporter(out))
        raw = out.read_text(encoding="utf-8")
        self.assertNotIn(FAKE_KEY, raw, "the fake secret reached the exported trace")
        self.assertIn("[REDACTED:api_key]", raw)
        body = json.loads(raw)
        spans = body["resourceSpans"][0]["scopeSpans"][0]["spans"]
        tool = next(s for s in spans if s["name"] == "execute_tool read_file")
        keys = {a["key"] for a in tool["attributes"]}
        self.assertIn("gen_ai.tool.call.arguments", keys)
        self.assertIn("gen_ai.tool.call.result", keys, "the tool result is captured, redacted")
        result = next(a for a in tool["attributes"] if a["key"] == "gen_ai.tool.call.result")
        self.assertIn("REGION=eu-west-1", result["value"]["stringValue"])

    def test_off_and_full(self):
        _, exp = run_traced(self.script, {"service.env": ENV_FILE}, capture="off")
        tool = by_name(exp.spans, "execute_tool")[0]
        self.assertNotIn("gen_ai.tool.call.arguments", tool.attributes)
        self.assertNotIn("gen_ai.tool.call.result", tool.attributes)
        _, exp = run_traced(self.script, {"service.env": ENV_FILE}, capture="full")
        tool = by_name(exp.spans, "execute_tool")[0]
        self.assertIn(FAKE_KEY, tool.attributes.get("gen_ai.tool.call.result", ""))


class TestGraders(unittest.TestCase):
    def test_normalize_and_exact_match(self):
        self.assertEqual(normalize("  West totals 4,210.  "), "west totals 4210")
        self.assertEqual(normalize("E204 (9 occurrences)!"), "e204 9 occurrences")
        self.assertTrue(exact_match("4,210.", "4210").passed)
        self.assertTrue(exact_match("Priya  Natarajan", "priya natarajan").passed)
        self.assertFalse(exact_match("42100", "4210").passed)
        self.assertFalse(exact_match("Priya Natarajan", "priya natarajan", normalized=False).passed)
        g = exact_match("T-1006.", "t-1006")
        self.assertEqual((g.score, g.detail["answer"]), (1.0, "t-1006"))

    def test_trajectory(self):
        exp = ["search", "read_file"]
        self.assertTrue(trajectory(["search", "read_file"], exp, "exact").passed)
        g = trajectory(["list_files", "search", "read_file"], exp, "exact")
        self.assertFalse(g.passed)
        self.assertEqual(g.detail, {"missing": [], "extra": ["list_files"]})
        self.assertTrue(trajectory(["list_files", "search", "read_file"], exp, "in_order").passed)
        g = trajectory(["read_file", "search"], exp, "in_order")
        self.assertFalse(g.passed)
        self.assertEqual(g.detail["missing"], ["read_file"])
        self.assertTrue(trajectory(["read_file", "search"], exp, "any_order").passed)
        g = trajectory(["search", "search"], ["search", "read_file", "read_file"], "any_order")
        self.assertEqual(g.detail, {"missing": ["read_file", "read_file"], "extra": ["search"]})
        self.assertEqual(g.score, 0.0)
        with self.assertRaises(ValueError):
            trajectory([], exp, "fuzzy")

    def test_convergence(self):
        self.assertAlmostEqual(convergence([3, 3, 6]), (1 + 1 + 0.5) / 3)
        self.assertAlmostEqual(convergence([4, 8], optimal=2), (0.5 + 0.25) / 2)
        self.assertAlmostEqual(convergence([2, 0]), 1.0)
        self.assertEqual(convergence([]), 0.0)

    def test_parse_verdict(self):
        self.assertIs(parse_verdict("Looks right.\nVERDICT: PASS"), True)
        self.assertIs(parse_verdict("verdict:fail"), False)
        self.assertIs(parse_verdict("**Verdict:** **PASS**"), True)
        self.assertIs(parse_verdict('{"reason": "wrong total", "verdict": "fail"}'), False)
        self.assertIs(parse_verdict("VERDICT: PASS\nOn reflection, VERDICT: FAIL"), False)
        self.assertIsNone(parse_verdict("I think it passes."))
        self.assertIsNone(parse_verdict(None))

    def test_judge_votes(self):
        m = ScriptedModel([say("ok\nVERDICT: PASS"), say("VERDICT: FAIL"), say("VERDICT: PASS")])
        g = LLMJudge(m, samples=3).grade("q?", "42", "42")
        self.assertTrue(g.passed)
        self.assertEqual(g.detail["votes"], [True, False, True])
        self.assertEqual(len(m.calls), 3)
        self.assertEqual(m.calls[0]["tools"], [], "the judge gets no tools")
        self.assertIn("Reference answer:\n42", m.calls[0]["messages"][-1]["content"])
        g = LLMJudge(ScriptedModel([say("VERDICT: PASS"), say("no idea")]), samples=2).grade("q", "a", "a")
        self.assertFalse(g.passed, "a tie fails")
        self.assertEqual(g.detail["unparsed"], 1)

    def test_agreement(self):
        judge = [True, True, False, False, True, True]
        human = [True, False, False, True, True, True]
        a = agreement(judge, human)
        self.assertEqual((a["tp"], a["fp"], a["tn"], a["fn"], a["n"]), (3, 1, 1, 1, 6))
        self.assertAlmostEqual(a["accuracy"], 4 / 6)
        self.assertAlmostEqual(a["tpr"], 0.75)
        self.assertAlmostEqual(a["tnr"], 0.5)
        p_e = (4 / 6) * (4 / 6) + (2 / 6) * (2 / 6)
        self.assertAlmostEqual(a["kappa"], (4 / 6 - p_e) / (1 - p_e))
        self.assertEqual(agreement([True, True], [True, True])["kappa"], 1.0)
        self.assertIsNone(agreement([True], [True])["tnr"])
        with self.assertRaises(ValueError):
            agreement([True], [True, False])

    def test_offline_judge_calibration_is_reported(self):
        cases = load_cases()
        self.assertEqual(len(cases), 30)
        judge = LLMJudge(SubstringJudge())
        verdicts = [judge.grade(c["question"], c["reference"], c["answer"]).passed for c in cases]
        a = agreement(verdicts, [c["label"] == "pass" for c in cases])
        print(f"\n  substring judge vs. 30 human labels: accuracy {a['accuracy']:.0%}, kappa {a['kappa']:.2f}, "
              f"TPR {a['tpr']:.0%}, TNR {a['tnr']:.0%}")
        self.assertEqual((a["tp"], a["fp"], a["tn"], a["fn"]), (12, 8, 6, 4))


if __name__ == "__main__":
    unittest.main()
