"""Lab 7 autograder tests: nanoos/tools.py (schemas, validation, truncation) and nanoos/sandbox.py."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Literal, Optional

from nanoos.sandbox import DockerSandbox, HostSandbox, NamespaceSandbox, shell_tool
from nanoos.tools import ToolError, ToolRegistry, json_type, schema_from_function
from tools_eval import main as tools_eval


def search(query: str, status: Literal["open", "closed"] = "open", limit: int = 10,
           tags: Optional[list[str]] = None, exact: bool = False, score: float = 0.5) -> str:
    """Search the things.

    More text that is not part of the description.

    Args:
        query: words to look for,
            across two lines.
        status: only items with this status.
    """
    return f"{query}|{status}|{limit}"


class TestSchemas(unittest.TestCase):
    def test_json_types(self):
        self.assertEqual(json_type(str), {"type": "string"})
        self.assertEqual(json_type(int), {"type": "integer"})
        self.assertEqual(json_type(float), {"type": "number"})
        self.assertEqual(json_type(bool), {"type": "boolean"})
        self.assertEqual(json_type(list[int]), {"type": "array", "items": {"type": "integer"}})
        self.assertEqual(json_type(Literal["a", "b"]), {"type": "string", "enum": ["a", "b"]})
        self.assertEqual(json_type(Literal[1, 2]), {"type": "integer", "enum": [1, 2]})
        self.assertEqual(json_type(Optional[str]), {"type": "string"})
        self.assertEqual(json_type(int | None), {"type": "integer"})
        with self.assertRaises(TypeError):
            json_type(dict)

    def test_schema_from_signature_and_docstring(self):
        s = schema_from_function(search)
        self.assertEqual(s["type"], "function")
        f = s["function"]
        self.assertEqual((f["name"], f["description"]), ("search", "Search the things."))
        p = f["parameters"]
        self.assertEqual((p["type"], p["required"], p["additionalProperties"]), ("object", ["query"], False))
        self.assertEqual(list(p["properties"]), ["query", "status", "limit", "tags", "exact", "score"])
        self.assertEqual(p["properties"]["query"],
                         {"type": "string", "description": "words to look for, across two lines."})
        self.assertEqual(p["properties"]["status"], {"type": "string", "enum": ["open", "closed"],
                                                     "description": "only items with this status.", "default": "open"})
        self.assertEqual(p["properties"]["tags"], {"type": "array", "items": {"type": "string"}, "default": None})
        self.assertEqual(p["properties"]["score"], {"type": "number", "default": 0.5})
        self.assertEqual(schema_from_function(search, "find")["function"]["name"], "find")

    def test_missing_hint_is_an_error(self):
        def f(x, y: int) -> str:
            return ""
        with self.assertRaises(TypeError):
            schema_from_function(f)

    def test_registry_uses_the_schema(self):
        with tempfile.TemporaryDirectory() as d:
            reg = ToolRegistry(d)
            reg.register(search)
            reg.register(name="find")(search)
            self.assertEqual([s["function"]["name"] for s in reg.schemas()], ["search", "find"])
            self.assertEqual(reg.call("find", {"query": "x"}), ("x|open|10", False))


class TestCall(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.ws = Path(self.dir.name)
        self.reg = ToolRegistry(self.ws, max_chars=100)
        self.reg.register(search)

        @self.reg.register
        def fail(kind: str) -> str:
            """Fail on purpose.

            Args:
                kind: "tool" or "bug".
            """
            if kind == "tool":
                raise ToolError("no such ticket. Ticket ids look like T-1001.")
            raise KeyError("x")

        @self.reg.register
        def big(n: int) -> str:
            """Return n characters."""
            return "".join(str(i % 10) for i in range(n))

    def tearDown(self):
        self.dir.cleanup()

    def test_success(self):
        self.assertEqual(self.reg.call("search", {"query": "login", "limit": 3}), ("login|open|3", False))
        self.assertEqual(self.reg.errors, 0)

    def test_unknown_tool_lists_the_tools(self):
        out, err = self.reg.call("serch", {})
        self.assertTrue(err)
        self.assertTrue(out.startswith("ERROR: "))
        self.assertIn("serch", out)
        for name in ("search", "fail", "big"):
            self.assertIn(name, out)

    def test_bad_arguments_object(self):
        for args in ("query=x", ["x"], {"_unparsed": "{query: x"}):
            out, err = self.reg.call("search", args)
            self.assertTrue(err, args)
            self.assertIn("JSON object", out)

    def test_unknown_argument_lists_valid_ones(self):
        out, err = self.reg.call("search", {"query": "x", "stat": "open"})
        self.assertTrue(err)
        self.assertIn("stat", out)
        self.assertIn("status", out)
        self.assertIn("limit", out)

    def test_missing_required_argument_is_described(self):
        out, err = self.reg.call("search", {"status": "open"})
        self.assertTrue(err)
        self.assertIn("query", out)
        self.assertIn("words to look for", out)

    def test_wrong_type_and_enum(self):
        out, err = self.reg.call("search", {"query": "x", "limit": "3"})
        self.assertTrue(err)
        self.assertIn("limit", out)
        self.assertIn("integer", out)
        self.assertIn("'3'", out)
        out, err = self.reg.call("search", {"query": "x", "status": "pending"})
        self.assertTrue(err)
        self.assertIn("closed", out, "an enum error lists the allowed values")
        self.assertIn("pending", out)
        self.assertTrue(self.reg.call("search", {"query": "x", "exact": 1})[1], "1 is not a boolean")
        self.assertTrue(self.reg.call("search", {"query": "x", "limit": True})[1], "True is not an integer")
        self.assertFalse(self.reg.call("search", {"query": "x", "score": 1})[1], "an integer is a number")
        self.assertEqual(self.reg.errors, 4)

    def test_tool_errors_and_crashes_do_not_raise(self):
        self.assertEqual(self.reg.call("fail", {"kind": "tool"}),
                         ("ERROR: no such ticket. Ticket ids look like T-1001.", True))
        out, err = self.reg.call("fail", {"kind": "bug"})
        self.assertTrue(err)
        self.assertTrue(out.startswith("ERROR: KeyError"), out)
        self.assertEqual(self.reg.errors, 2)

    def test_truncation_saves_the_full_output(self):
        self.assertEqual(self.reg.call("big", {"n": 100}), ("0123456789" * 10, False))
        self.assertEqual(self.reg.truncations, 0)
        out, err = self.reg.call("big", {"n": 250})
        self.assertFalse(err)
        head, _, note = out.partition("\n")
        self.assertEqual(head, ("0123456789" * 10))
        self.assertTrue(note.startswith("[truncated:"), note)
        self.assertIn("250", note)
        self.assertIn(".tool_outputs/big_1.txt", note)
        self.assertEqual((self.ws / ".tool_outputs" / "big_1.txt").read_text(), ("0123456789" * 25))
        self.reg.call("big", {"n": 300})
        self.assertTrue((self.ws / ".tool_outputs" / "big_2.txt").exists())
        self.assertEqual(self.reg.truncations, 2)


class TestSandbox(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.ws = Path(self.dir.name)
        (self.ws / "data.txt").write_text("hello\n")

    def tearDown(self):
        self.dir.cleanup()

    def test_docker_command_line(self):
        argv = DockerSandbox(self.ws, memory_mb=256, cpus=0.5, pids=64).argv("ls -l")
        self.assertEqual(argv[:4], ["docker", "run", "--rm", "-i"])
        joined = " ".join(argv)
        for part in ("--network none", "--read-only", "--tmpfs /tmp:rw,size=64m", "--memory 256m", "--cpus 0.5",
                     "--pids-limit 64", "--cap-drop ALL", "--security-opt no-new-privileges",
                     f"-v {self.ws.resolve()}:/workspace:ro", "-w /workspace"):
            self.assertIn(part, joined)
        self.assertEqual(argv[-4:], ["python:3.12-slim", "sh", "-c", "ls -l"])
        self.assertIn(f"{self.ws.resolve()}:/workspace:rw", DockerSandbox(self.ws, writable=True).argv("x"))

    def test_run_returns_output_and_exit_code(self):
        r = HostSandbox(self.ws).run("cat data.txt; echo oops >&2; exit 3")
        self.assertEqual((r.exit_code, r.stdout, r.stderr, r.timed_out, r.truncated),
                         (3, "hello\n", "oops\n", False, False))
        self.assertGreaterEqual(r.duration_s, 0)

    def test_environment_is_scrubbed(self):
        os.environ["LAB7_FAKE_TOKEN"] = "secret"
        try:
            r = HostSandbox(self.ws).run("env")
        finally:
            del os.environ["LAB7_FAKE_TOKEN"]
        self.assertNotIn("LAB7_FAKE_TOKEN", r.stdout)

    def test_timeout_is_a_result(self):
        r = HostSandbox(self.ws, timeout_s=0.5).run("echo started; sleep 5; echo never")
        self.assertTrue(r.timed_out)
        self.assertIsNone(r.exit_code)
        self.assertIn("started", r.stdout)
        self.assertLess(r.duration_s, 4, "the whole process group is killed")

    def test_output_is_capped(self):
        r = HostSandbox(self.ws, max_output=50).run("yes | head -c 1000")
        self.assertEqual(len(r.stdout), 50)
        self.assertTrue(r.truncated)

    def test_shell_tool_formats_results(self):
        reg = ToolRegistry(self.ws)
        reg.register(shell_tool(HostSandbox(self.ws, timeout_s=0.5, max_output=5)))
        self.assertEqual(reg.call("run_shell", {"command": "echo hi"}), ("exit code: 0\nhi\n", False))
        out, err = reg.call("run_shell", {"command": "echo bad >&2; exit 2"})
        self.assertFalse(err, "a failing command is output, not a tool error")
        self.assertEqual(out, "exit code: 2\n\n[stderr]\nbad\n")
        self.assertTrue(reg.call("run_shell", {"command": "echo 1234567890"})[0].endswith("\n[output truncated]"))
        out, err = reg.call("run_shell", {"command": "sleep 5"})
        self.assertTrue(err)
        self.assertIn("timed out after 0.5s", out)


@unittest.skipUnless(sys.platform == "linux" and NamespaceSandbox.available(), "needs unprivileged user namespaces")
class TestNamespaceSandbox(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.ws = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_no_network(self):
        code = "import socket; s=socket.socket(); s.settimeout(2); s.connect(('1.1.1.1', 53))"
        r = NamespaceSandbox(self.ws).run(f'python3 -c "{code}"')
        self.assertNotEqual(r.exit_code, 0)
        self.assertIn("unreachable", r.stderr.lower())

    def test_only_the_workspace_is_writable(self):
        outside = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(outside))
        r = NamespaceSandbox(self.ws, writable=True).run(f"echo x > out.txt && echo x > {outside}/escape.txt")
        self.assertTrue((self.ws / "out.txt").exists())
        self.assertFalse((Path(outside) / "escape.txt").exists())
        self.assertIn("read-only", r.stderr.lower())
        r = NamespaceSandbox(self.ws).run("echo x > again.txt")
        self.assertNotEqual(r.exit_code, 0, "writable=False makes the workspace read-only too")
        self.assertFalse((self.ws / "again.txt").exists())


class TestToolsEval(unittest.TestCase):
    def test_designed_tools_cost_fewer_tokens(self):
        s = {}
        for v in ("v1", "v2"):
            with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
                tools_eval(["--tools", v, "--sandbox", "host", "--out", str(Path(d) / "r")])
                s[v] = json.loads((Path(d) / "r" / "summary.json").read_text())
        self.assertEqual((s["v1"]["success_rate"], s["v2"]["success_rate"]), (1.0, 1.0))
        self.assertGreater(s["v1"]["truncations"], 0)
        self.assertEqual(s["v2"]["truncations"], 0)
        self.assertLess(s["v2"]["input_tokens_per_task"], 0.6 * s["v1"]["input_tokens_per_task"])


if __name__ == "__main__":
    unittest.main()
