"""Lab 10 autograder tests: nanoos/plugins.py (config validation, loading, slots, ordered hooks, crashes)."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from nanoos.plugins import HookedTools, PluginConfigError, PluginError, PluginHost, validate_config
from plugins_eval import main as plugins_eval

KIT_PLUGINS = Path(__file__).resolve().parent.parent / "plugins"
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["path"],
          "properties": {"path": {"type": "string"}, "limit": {"type": "integer", "default": 3},
                         "mode": {"type": "string", "enum": ["fast", "safe"], "default": "safe"},
                         "tags": {"type": "array", "items": {"type": "string"}}}}


def make_plugin(root: Path, pid: str, body: str, kind: str | None = None, schema: dict | None = None) -> Path:
    d = root / pid
    d.mkdir(parents=True)
    (d / "plugin.json").write_text(json.dumps({"id": pid, "version": "1.0.0", "kind": kind,
                                               "configSchema": schema or {"type": "object", "properties": {}}}))
    (d / "plugin.py").write_text(textwrap.dedent(body))
    return d


class Echo:
    def schemas(self):
        return []

    def call(self, name, arguments):
        return f"ran {name} {json.dumps(arguments, sort_keys=True)}", False


class TestValidateConfig(unittest.TestCase):
    def test_defaults_and_valid(self):
        given = {"path": "x", "tags": ["a"]}
        cfg, problems = validate_config(SCHEMA, given)
        self.assertEqual(problems, [])
        self.assertEqual(cfg, {"path": "x", "tags": ["a"], "limit": 3, "mode": "safe"})
        self.assertEqual(given, {"path": "x", "tags": ["a"]}, "the given dict is not modified")
        self.assertEqual(validate_config({"type": "object", "properties": {"a": {"type": "integer", "default": 1}}},
                                         None), ({"a": 1}, []))

    def test_problems_name_the_key(self):
        _, p = validate_config(SCHEMA, {"path": "x", "limt": 5})
        self.assertEqual(len(p), 1)
        self.assertIn("limt", p[0])
        _, p = validate_config(SCHEMA, {})
        self.assertEqual(len(p), 1)
        self.assertIn("path", p[0])
        _, p = validate_config(SCHEMA, {"path": "x", "limit": "3", "mode": "turbo", "tags": [1]})
        self.assertEqual(len(p), 3)
        self.assertTrue(any("limit" in x and "integer" in x for x in p))
        self.assertTrue(any("mode" in x and "safe" in x for x in p), "an enum problem lists the allowed values")
        self.assertEqual(len(validate_config(SCHEMA, ["path"])[1]), 1)
        loose = {"type": "object", "properties": {}}
        self.assertEqual(validate_config(loose, {"anything": 1}), ({"anything": 1}, []))


class TestLoad(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.marker = self.root / "imported.txt"

    def tearDown(self):
        self.dir.cleanup()

    def test_bad_config_fails_before_plugin_code_runs(self):
        make_plugin(self.root, "p", f"""
            from pathlib import Path
            Path({str(self.marker)!r}).write_text("x")
            def register(api):
                pass
            """, schema=SCHEMA)
        host = PluginHost([self.root])
        with self.assertRaises(PluginConfigError) as cm:
            host.load({"enabled": ["p"], "entries": {"p": {"path": 7}}})
        self.assertIn("p", str(cm.exception))
        self.assertIn("path", str(cm.exception))
        self.assertFalse(self.marker.exists(), "config is checked from the manifest, before import")
        self.assertEqual(host.loaded, [])
        ok = PluginHost([self.root])
        ok.load({"enabled": ["p"], "entries": {"p": {"path": "a"}}})
        self.assertEqual(ok.loaded, ["p"])
        self.assertTrue(self.marker.exists())

    def test_config_reaches_the_plugin(self):
        make_plugin(self.root, "p", """
            SEEN = []
            def register(api):
                SEEN.append(dict(api.config))
                api.on("before_tool_call", lambda ev: {"block": True, "reason": str(api.config["limit"])})
            """, schema=SCHEMA)
        host = PluginHost([self.root])
        host.load({"enabled": ["p"], "entries": {"p": {"path": "a"}}})
        self.assertEqual(HookedTools(Echo(), host.bus).call("t", {}), ("ERROR: blocked by policy (p): 3", True))

    def test_unknown_denied_and_crashing_plugins(self):
        make_plugin(self.root, "good", "def register(api):\n    pass\n")
        make_plugin(self.root, "bad", "def register(api):\n    raise ValueError('oops')\n")
        with self.assertRaises(PluginError) as cm:
            PluginHost([self.root]).load({"enabled": ["nope"]})
        self.assertIn("nope", str(cm.exception))
        self.assertIn("good", str(cm.exception), "list what was found")
        host = PluginHost([self.root])
        host.load({"enabled": ["good", "bad"], "deny": ["bad"]})
        self.assertEqual(host.loaded, ["good"])
        with self.assertRaises(PluginError) as cm:
            PluginHost([self.root]).load({"enabled": ["good", "bad"]})
        self.assertIn("bad", str(cm.exception))
        self.assertIn("oops", str(cm.exception))
        make_plugin(self.root, "typo", "def register(api):\n    api.on('before_tool', print)\n")
        with self.assertRaises(PluginError):
            PluginHost([self.root]).load({"enabled": ["typo"]})


class TestSlots(unittest.TestCase):
    def test_one_provider_per_slot(self):
        with self.assertRaises(PluginError) as cm:
            PluginHost([KIT_PLUGINS]).load({"enabled": ["threshold-compaction", "tool-result-clearing"]})
        self.assertIn("threshold-compaction", str(cm.exception))
        self.assertIn("tool-result-clearing", str(cm.exception))

    def test_the_slot_setting_chooses(self):
        host = PluginHost([KIT_PLUGINS])
        host.load({"enabled": ["threshold-compaction", "tool-result-clearing"],
                   "slots": {"context-engine": "tool-result-clearing"},
                   "entries": {"tool-result-clearing": {"keep_last_results": 2}}})
        self.assertEqual(host.loaded, ["tool-result-clearing"])
        self.assertEqual(len(host.skipped), 1)
        self.assertIn("threshold-compaction", host.skipped[0])
        engine = host.slot("context-engine")
        self.assertEqual((type(engine).__name__, engine.keep_last_results), ("ToolResultClearing", 2))
        self.assertIsNone(host.slot("memory"))
        forced = PluginHost([KIT_PLUGINS])
        forced.load({"slots": {"context-engine": "threshold-compaction"}})
        self.assertEqual(forced.loaded, ["threshold-compaction"], "selecting a plugin for a slot enables it")
        with self.assertRaises(PluginError):
            PluginHost([KIT_PLUGINS]).load({"slots": {"memory": "threshold-compaction"}})
        with self.assertRaises(PluginError):
            PluginHost([KIT_PLUGINS]).load({"slots": {"context-engine": "tool-result-clearing"},
                                            "deny": ["tool-result-clearing"]})


class TestHooks(unittest.TestCase):
    """Three plugins hook every call; each appends its id to a shared log file, so the order is visible."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.log = self.root / "order.txt"
        schema = {"type": "object", "properties": {"log": {"type": "string"}, "priority": {"type": "integer"}}}
        body = """
            def register(api):
                def before(ev):
                    with open(api.config["log"], "a") as f:
                        f.write(api.id + "\\n")
                    ev["arguments"] = "tampered"
                    if ev["name"] == "risky-" + api.id:
                        return {"block": True}
                api.on("before_tool_call", before, priority=api.config["priority"])
                api.on("tool_result", lambda ev: ev["output"] + " +" + api.id, priority=api.config["priority"])
            """
        for pid in ("low", "high", "tie"):
            make_plugin(self.root, pid, body, schema=schema)
        make_plugin(self.root, "watch", """
            def register(api):
                with open(api.config["log"], "w"):
                    pass
                def seen(ev):
                    with open(api.config["log"], "a") as f:
                        f.write(f"{ev['blocked']} {ev['output']}\\n")
                api.on("after_tool_call", seen)
            """, schema=schema)
        self.watch = self.root / "watch.txt"
        host = PluginHost([self.root])
        host.load({"enabled": ["low", "watch", "high", "tie"],
                   "entries": {"low": {"log": str(self.log), "priority": 1}, "high": {"log": str(self.log),
                               "priority": 50}, "tie": {"log": str(self.log), "priority": 1},
                               "watch": {"log": str(self.watch)}}})
        self.tools = HookedTools(Echo(), host.bus)

    def tearDown(self):
        self.dir.cleanup()

    def order(self):
        lines = self.log.read_text().split() if self.log.exists() else []
        self.log.unlink(missing_ok=True)
        return lines

    def test_priority_then_registration_order(self):
        out, err = self.tools.call("t", {"x": 1})
        self.assertFalse(err)
        self.assertEqual(self.order(), ["high", "low", "tie"])
        self.assertEqual(out, 'ran t {"x": 1} +high +low +tie', "transforms chain in hook order")
        self.assertEqual(self.watch.read_text(), 'False ran t {"x": 1} +high +low +tie\n',
                         "a handler's edits to its payload do not reach the tool or later handlers")

    def test_a_block_stops_lower_priority_handlers(self):
        out, err = self.tools.call("risky-low", {})
        self.assertTrue(err)
        self.assertIn("(low)", out)
        self.assertEqual(self.order(), ["high", "low"], "tie (same priority, registered later) never ran")
        self.tools.call("risky-high", {})
        self.assertEqual(self.order(), ["high"])
        self.assertTrue(self.watch.read_text().splitlines()[-1].startswith("True "), "observers see blocked calls")
        self.assertEqual([b[0] for b in self.tools.blocked], ["risky-low", "risky-high"])


class TestCrashes(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_crashing_observer_is_disabled_and_the_call_goes_on(self):
        make_plugin(self.root, "boom", """
            def register(api):
                api.on("after_tool_call", lambda ev: 1 / 0)
                api.on("tool_result", lambda ev: ev["output"] + " seen")
            """)
        host = PluginHost([self.root])
        host.load({"enabled": ["boom"]})
        tools = HookedTools(Echo(), host.bus)
        self.assertEqual(tools.call("t", {}), ("ran t {} seen", False))
        self.assertEqual(host.bus.disabled, {"boom"})
        self.assertEqual(len(host.bus.errors), 1)
        self.assertIn("ZeroDivisionError", host.bus.errors[0])
        self.assertEqual(tools.call("t", {}), ("ran t {}", False), "none of its hooks run again")

    def test_crashing_guard_fails_closed_once(self):
        make_plugin(self.root, "guard", """
            def register(api):
                def before(ev):
                    raise RuntimeError("bug")
                api.on("before_tool_call", before)
            """)
        host = PluginHost([self.root])
        host.load({"enabled": ["guard"]})
        tools = HookedTools(Echo(), host.bus)
        out, err = tools.call("t", {})
        self.assertTrue(err)
        self.assertIn("guard crashed", out)
        self.assertEqual(tools.call("t", {}), ("ran t {}", False), "after the crash the guard is gone")


class TestPluginsEval(unittest.TestCase):
    def run_eval(self, *extra):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            plugins_eval(["--out", str(Path(d) / "r"), *extra])
            s = json.loads((Path(d) / "r" / "summary.json").read_text())
            s["log"] = [json.loads(x) for x in (Path(d) / "r" / "shell_commands.jsonl").read_text().splitlines()] \
                if (Path(d) / "r" / "shell_commands.jsonl").exists() else []
            return s

    def test_shell_guard(self):
        s = self.run_eval()
        self.assertEqual((s["benign_success"], s["bypassed"]), (1.0, ["o5_two_step"]))
        self.assertAlmostEqual(s["risky_blocked"], 2 / 3)
        self.assertEqual([x["blocked"] for x in s["log"]], [False, False, True, True, False])
        none = self.run_eval("--enable", "none")
        self.assertEqual(none["risky_blocked"], 0.0)
        crashed = self.run_eval("--set", "shell-guard.crash_on=wc")
        self.assertEqual(crashed["disabled_after_crash"], ["shell-guard"])
        self.assertEqual(crashed["risky_blocked"], 0.0, "a disabled guard guards nothing")


if __name__ == "__main__":
    unittest.main()
