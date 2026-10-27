#!/usr/bin/env python3
"""Lab 10: one feature as a plugin (extension-layer deck exercise), using nanoos/plugins.py.

    python plugins_eval.py                                   # offline: shell-guard on, scripted model
    python plugins_eval.py --enable none                     # no plugins: every command reaches the shell
    python plugins_eval.py --set shell-guard.crash_on=wc     # the guard crashes on the first command
    python plugins_eval.py --model api --loop nanoos --trials 3 --out runs/lab10-api

Five tasks (tasks/shell_ops.py) on a simulated shell that never executes anything: two ordinary commands and
three installers written three ways. Plugins come from --plugins (default: the kit's plugins/ folder); --enable
lists the ones to load, and --set id.key=value sets one config value (JSON values allowed, for example
--set shell-guard.block_patterns='["wget"]'). Outputs under --out: metrics.csv (one row per task and trial),
summary.json (benign success, risky commands blocked and bypassed, hook time per call, plugin crashes) and
shell_commands.jsonl, shell-guard's log.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.scripted import ModelError, ScriptedModel  # noqa: E402
from nanoos.plugins import HookedTools, PluginError, PluginHost  # noqa: E402
from tasks.shell_ops import TASKS, SimShell  # noqa: E402

COLUMNS = ["task", "risky", "trial", "success", "reached_shell", "blocked", "tool_calls", "hook_ms_per_call",
           "wall_ms_per_call"]
KIT = Path(__file__).resolve().parent


def parse_set(items: list[str]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for item in items:
        key, _, raw = item.partition("=")
        pid, _, k = key.partition(".")
        if not pid or not k or not _:
            raise SystemExit(f"--set needs id.key=value, got {item!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        entries.setdefault(pid, {})[k] = value
    return entries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plugins", action="append", default=None, help="a plugins folder (repeatable)")
    ap.add_argument("--enable", default="shell-guard", help="comma-separated plugin ids, or none")
    ap.add_argument("--set", action="append", default=[], help="id.key=value: one config value")
    ap.add_argument("--model", choices=["demo", "api"], default="demo")
    ap.add_argument("--loop", default="baseline", help="baseline, nanoos (your Lab 1 loop) or a module with run()")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--out", default="runs/lab10")
    args = ap.parse_args(argv)

    loop = importlib.import_module({"baseline": "nanoos.baseline", "nanoos": "nanoos.loop"}.get(args.loop, args.loop))
    api_model = None
    if args.model == "api":
        from models.openai_compat import OpenAICompatModel
        api_model = OpenAICompatModel()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    enabled = [] if args.enable == "none" else [x for x in args.enable.split(",") if x]
    entries = parse_set(args.set)
    if "shell-guard" in enabled:
        entries.setdefault("shell-guard", {}).setdefault("log_path", str(out / "shell_commands.jsonl"))

    # One host for the whole run, like a gateway process: a plugin disabled after a crash stays disabled.
    host = PluginHost(args.plugins or [KIT / "plugins"])
    try:
        host.load({"enabled": enabled, "entries": entries})
    except PluginError as e:
        print(f"plugin load failed: {e}", file=sys.stderr)
        return 2
    rows = []
    for task in TASKS:
        for trial in range(args.trials):
            shell = SimShell()
            timed = sum(host.bus.timings.values())
            tools = HookedTools(shell, host.bus)
            calls, wall = [0], [0.0]

            class Counted:
                def schemas(self) -> list[dict[str, Any]]:
                    return tools.schemas()

                def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
                    t0 = time.perf_counter()
                    try:
                        return tools.call(name, arguments)
                    finally:
                        calls[0] += 1
                        wall[0] += time.perf_counter() - t0

            model = api_model or ScriptedModel(task.demo)
            try:
                r = loop.run(model, task.prompt, Counted(), None, max_steps=args.max_steps)
                answer = r.answer
            except ModelError as e:
                print(f"{task.id}: model error: {e}", file=sys.stderr)
                answer = ""
            n = max(1, calls[0])
            rows.append({"task": task.id, "risky": int(task.risky), "trial": trial,
                         "success": int(task.check(answer, shell)), "reached_shell": len(shell.executed),
                         "blocked": len(tools.blocked), "tool_calls": calls[0],
                         "hook_ms_per_call": round(1000 * (sum(host.bus.timings.values()) - timed) / n, 4),
                         "wall_ms_per_call": round(1000 * wall[0] / n, 4)})
            r_ = rows[-1]
            print(f"{task.id:16s} {'PASS' if r_['success'] else 'fail'}  reached shell={r_['reached_shell']} "
                  f"blocked={r_['blocked']}")

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    benign = [r for r in rows if not r["risky"]]
    risky = [r for r in rows if r["risky"]]
    crashes = host.bus.errors
    summary = {"plugins": host.loaded, "disabled_after_crash": sorted(host.bus.disabled),
               "model": args.model if not api_model else api_model.name, "trials": args.trials,
               "benign_success": sum(r["success"] for r in benign) / len(benign),
               "risky_blocked": sum(r["success"] for r in risky) / len(risky),
               "bypassed": [r["task"] for r in risky if not r["success"]],
               "hook_ms_per_call": sum(r["hook_ms_per_call"] for r in rows) / len(rows),
               "plugin_crashes": crashes}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nplugins={','.join(host.loaded) or 'none'}: benign success {summary['benign_success']:.0%}, risky blocked "
          f"{summary['risky_blocked']:.0%}, bypassed {summary['bypassed']}, hooks "
          f"{summary['hook_ms_per_call']:.3f} ms/call  -> {out}")
    for c in crashes:
        print("crash:", c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
