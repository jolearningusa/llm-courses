"""Lab 1 autograder tests: nanoos/loop.py (budgets, tools, retries, interrupts, steering, repetition, hooks)."""

from __future__ import annotations

import contextlib
import io
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path

from models.scripted import ScriptedModel
from nanoos import loop
from nanoos.loop import REPEAT_MSG, WRAP_UP_MSG, LoopConfig
from run_suite import main as run_suite


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def turn(*calls):
    return {"tool_calls": list(calls)}


class FakeTools:
    """Tools that echo their call; `s` sleeps that long first. Records the order calls start in."""

    def __init__(self):
        self.started: list[str] = []
        self.lock = threading.Lock()

    def schemas(self):
        return [{"type": "function", "function": {"name": "echo", "parameters": {"type": "object"}}}]

    def call(self, name, arguments):
        with self.lock:
            self.started.append(arguments.get("x", ""))
        time.sleep(arguments.get("s", 0))
        if name != "echo":
            return f"ERROR: unknown tool {name!r}", True
        return f"echo {arguments.get('x', '')}", False


def cfg(**kw):
    kw.setdefault("backoff_s", 0)
    return LoopConfig(**kw)


def tool_messages(messages):
    return [m["content"] for m in messages if m["role"] == "tool"]


def user_texts(messages):
    return [m["content"] for m in messages if m["role"] == "user"]


class TestBasics(unittest.TestCase):
    def test_answers_without_tools(self):
        r = loop.run(ScriptedModel([{"content": "hi"}]), "say hi", FakeTools(), config=cfg())
        self.assertEqual((r.answer, r.steps, r.stopped_by), ("hi", 1, "answer"))

    def test_tool_result_is_linked_to_its_call(self):
        m = ScriptedModel([turn(call("echo", x="a")), {"content": "done"}])
        r = loop.run(m, "go", FakeTools(), config=cfg())
        tool_msg = next(x for x in r.messages if x["role"] == "tool")
        call_id = next(x for x in r.messages if x.get("tool_calls"))["tool_calls"][0]["id"]
        self.assertEqual((tool_msg["tool_call_id"], tool_msg["content"]), (call_id, "echo a"))
        self.assertEqual(m.calls[1]["messages"][-1]["role"], "tool")

    def test_works_with_run_suite(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_suite(["--model", "demo", "--loop", "nanoos", "--out", str(Path(d) / "r")]), 0)
            self.assertIn('"success_rate": 1.0', (Path(d) / "r" / "summary.json").read_text())


class TestBudget(unittest.TestCase):
    def test_budget_stops_the_loop(self):
        m = ScriptedModel([turn(call("echo", x=str(i))) for i in range(10)])
        r = loop.run(m, "go", FakeTools(), config=cfg(max_steps=4, wrap_up_at=0, repeat_limit=0))
        self.assertEqual((r.stopped_by, r.steps, r.answer, len(m.calls)), ("budget", 4, "", 4))

    def test_wrap_up_message_once_at_the_right_step(self):
        m = ScriptedModel([turn(call("echo", x=str(i))) for i in range(10)])
        loop.run(m, "go", FakeTools(), config=cfg(max_steps=5, wrap_up_at=3, repeat_limit=0))
        wrap = WRAP_UP_MSG.format(left=3)
        self.assertNotIn(wrap, user_texts(m.calls[1]["messages"]))
        self.assertEqual(user_texts(m.calls[2]["messages"])[-1], wrap)
        self.assertEqual(user_texts(m.calls[4]["messages"]).count(wrap), 1)

    def test_max_steps_argument_is_used_without_config(self):
        m = ScriptedModel([turn(call("echo", x=str(i))) for i in range(10)])
        self.assertEqual(loop.run(m, "go", FakeTools(), max_steps=2).steps, 2)


class TestTools(unittest.TestCase):
    def test_parallel_calls_keep_call_order(self):
        m = ScriptedModel([turn(call("echo", x="a", s=0.3), call("echo", x="b", s=0.1), call("echo", x="c", s=0.2)),
                           {"content": "done"}])
        t0 = time.time()
        r = loop.run(m, "go", FakeTools(), config=cfg(parallel_tools=True))
        self.assertLess(time.time() - t0, 0.5, "the three calls should run at the same time")
        self.assertEqual(tool_messages(r.messages), ["echo a", "echo b", "echo c"])

    def test_sequential_when_parallel_is_off(self):
        tools = FakeTools()
        m = ScriptedModel([turn(call("echo", x="a", s=0.05), call("echo", x="b")), {"content": "done"}])
        loop.run(m, "go", tools, config=cfg(parallel_tools=False))
        self.assertEqual(tools.started, ["a", "b"])

    def test_tool_errors_are_counted_and_returned_to_the_model(self):
        m = ScriptedModel([turn(call("nope", x="a")), {"content": "done"}])
        r = loop.run(m, "go", FakeTools(), config=cfg())
        self.assertEqual(r.tool_errors, 1)
        self.assertTrue(tool_messages(r.messages)[0].startswith("ERROR"))


class TestRetries(unittest.TestCase):
    def test_retries_then_succeeds(self):
        m = ScriptedModel([{"raise": "429"}, {"raise": "503"}, {"content": "ok"}])
        r = loop.run(m, "go", FakeTools(), config=cfg(max_retries=2))
        self.assertEqual((r.answer, r.stopped_by, r.steps), ("ok", "answer", 1))

    def test_fallback_model_after_retries(self):
        primary = ScriptedModel([{"raise": "429"}] * 3)
        backup = ScriptedModel([{"content": "from backup"}], name="backup")
        r = loop.run(primary, "go", FakeTools(), config=cfg(max_retries=2, fallback=backup))
        self.assertEqual((r.answer, len(primary.calls), len(backup.calls)), ("from backup", 3, 1))

    def test_error_when_everything_fails(self):
        r = loop.run(ScriptedModel([{"raise": "500"}] * 5), "go", FakeTools(), config=cfg(max_retries=1))
        self.assertEqual((r.stopped_by, r.answer), ("error", ""))

    def test_backoff_doubles(self):
        m = ScriptedModel([{"raise": "429"}, {"raise": "429"}, {"content": "ok"}])
        t0 = time.time()
        loop.run(m, "go", FakeTools(), config=LoopConfig(max_retries=2, backoff_s=0.1))
        self.assertGreaterEqual(time.time() - t0, 0.29)  # 0.1 + 0.2


class TestInterruptAndSteering(unittest.TestCase):
    def test_interrupt_during_a_slow_model_call(self):
        stop = threading.Event()
        m = ScriptedModel([{"content": "too late", "delay_s": 2}])
        threading.Timer(0.1, stop.set).start()
        t0 = time.time()
        r = loop.run(m, "go", FakeTools(), config=cfg(interrupt=stop))
        self.assertEqual(r.stopped_by, "interrupted")
        self.assertLess(time.time() - t0, 1.0, "the loop must not wait for the model call to finish")

    def test_steering_message_arrives_between_steps(self):
        q: queue.Queue[str] = queue.Queue()

        def arrive_mid_turn(c):
            q.put("Also count the vowels.")
            return None

        m = ScriptedModel([turn(call("echo", x="a")), {"content": "done"}])
        loop.run(m, "go", FakeTools(), config=cfg(steering=q, pre_tool_call=[arrive_mid_turn]))
        second = m.calls[1]["messages"]
        self.assertEqual(second[-1], {"role": "user", "content": "Also count the vowels."})
        self.assertEqual(second[-2]["role"], "tool", "steering goes after the tool results")
        self.assertTrue(q.empty())


class TestRepetition(unittest.TestCase):
    def test_nudge_after_repeated_identical_calls(self):
        m = ScriptedModel([turn(call("echo", x="same"))] * 3 + [{"content": "ok"}])
        loop.run(m, "go", FakeTools(), config=cfg(repeat_limit=3))
        nudge = REPEAT_MSG.format(n=3)
        self.assertNotIn(nudge, user_texts(m.calls[2]["messages"]))
        self.assertEqual(user_texts(m.calls[3]["messages"])[-1], nudge)

    def test_no_nudge_when_arguments_differ(self):
        m = ScriptedModel([turn(call("echo", x=str(i))) for i in range(3)] + [{"content": "ok"}])
        loop.run(m, "go", FakeTools(), config=cfg(repeat_limit=3))
        self.assertNotIn(REPEAT_MSG.format(n=3), user_texts(m.calls[3]["messages"]))


class TestHooks(unittest.TestCase):
    def test_pre_hook_blocks_and_post_hook_rewrites(self):
        order = []

        def deny_b(c):
            order.append(("pre", c.arguments["x"]))
            return "b is not allowed" if c.arguments["x"] == "b" else None

        def shout(c, out, is_error):
            order.append(("post", c.arguments["x"]))
            return out.upper()

        tools = FakeTools()
        m = ScriptedModel([turn(call("echo", x="a"), call("echo", x="b")), {"content": "done"}])
        r = loop.run(m, "go", tools, config=cfg(parallel_tools=False, pre_tool_call=[deny_b], post_tool_call=[shout]))
        self.assertEqual(tool_messages(r.messages), ["ECHO A", "BLOCKED: b is not allowed"])
        self.assertEqual(tools.started, ["a"], "a blocked call must not run")
        self.assertEqual(order, [("pre", "a"), ("post", "a"), ("pre", "b")])
        self.assertEqual(r.tool_errors, 1)

    def test_hooks_run_in_list_order(self):
        seen = []
        hooks = [lambda c: seen.append(1), lambda c: seen.append(2)]
        m = ScriptedModel([turn(call("echo", x="a")), {"content": "done"}])
        loop.run(m, "go", FakeTools(), config=cfg(pre_tool_call=hooks))
        self.assertEqual(seen, [1, 2])
