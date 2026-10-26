#!/usr/bin/env python3
"""Lab 7: what tool design costs and saves (tools-and-sandbox deck exercise), using nanoos/tools.py and
nanoos/sandbox.py.

    python tools_eval.py --tools v1                     # offline: the scripted solutions with the vague tool
    python tools_eval.py --tools v2                     # ... and with the designed one
    python tools_eval.py --tools v2 --model api --trials 3 --loop nanoos --out runs/lab7-v2

Five ticket questions (tasks/tickets.py). Each run gets a fresh workspace with tickets.json, the chosen search
tools, read_file and run_shell in a sandbox (--sandbox auto picks Docker, then Linux namespaces, then the host).
Outputs under --out: metrics.csv (one row per task and trial) and summary.json: success, tokens per task, tool
errors (calls the registry rejected or that failed), truncations and shell commands.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.scripted import ModelError, ScriptedModel  # noqa: E402
from nanoos.sandbox import DockerSandbox, HostSandbox, NamespaceSandbox, best_available, shell_tool  # noqa: E402
from nanoos.tools import ToolRegistry  # noqa: E402
from nanoos.tools_basic import WorkspaceTools  # noqa: E402
from tasks.tickets import TASKS, setup, v1_tools, v2_tools  # noqa: E402

COLUMNS = ["task", "tools", "trial", "success", "steps", "input_tokens", "output_tokens", "tool_errors",
           "truncations", "shell_calls"]
BACKENDS = {"docker": DockerSandbox, "namespace": NamespaceSandbox, "host": HostSandbox}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tools", choices=["v1", "v2"], default="v2")
    ap.add_argument("--model", choices=["demo", "api"], default="demo")
    ap.add_argument("--loop", default="baseline", help="baseline, nanoos (your Lab 1 loop) or a module with run()")
    ap.add_argument("--sandbox", choices=["auto", *BACKENDS], default="auto")
    ap.add_argument("--max-chars", type=int, default=8000, help="truncate tool results longer than this")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--out", default="runs/lab7")
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

    rows, backend = [], None
    for task in TASKS:
        for trial in range(args.trials):
            with tempfile.TemporaryDirectory() as d:
                ws = Path(d)
                setup(ws)
                reg = ToolRegistry(ws, max_chars=args.max_chars)
                (v1_tools if args.tools == "v1" else v2_tools)(reg, ws)
                reg.add(WorkspaceTools(ws, allowed=("read_file",)))
                box = best_available(ws) if args.sandbox == "auto" else BACKENDS[args.sandbox](ws)
                backend = box.name
                shell = shell_tool(box)
                calls = [0]

                def run_shell(command: str) -> str:
                    calls[0] += 1
                    return shell(command)
                run_shell.__doc__ = shell.__doc__
                reg.register(run_shell)
                model = api_model or ScriptedModel(task.demo_v1 if args.tools == "v1" else task.demo_v2)
                try:
                    r = loop.run(model, task.prompt, reg, None, max_steps=args.max_steps)
                    answer, steps, usage = r.answer, r.steps, r.usage
                except ModelError as e:
                    print(f"{task.id}: model error: {e}", file=sys.stderr)
                    answer, steps, usage = "", 0, None
            rows.append({"task": task.id, "tools": args.tools, "trial": trial, "success": int(task.check(answer)),
                         "steps": steps, "input_tokens": usage.input_tokens if usage else 0,
                         "output_tokens": usage.output_tokens if usage else 0, "tool_errors": reg.errors,
                         "truncations": reg.truncations, "shell_calls": calls[0]})
            r_ = rows[-1]
            print(f"{task.id:22s} {'PASS' if r_['success'] else 'fail'}  in={r_['input_tokens']:<7d} "
                  f"errors={r_['tool_errors']} truncations={r_['truncations']}")

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    summary = {"tools": args.tools, "model": args.model if not api_model else api_model.name, "sandbox": backend,
               "trials": args.trials, "success_rate": sum(r["success"] for r in rows) / n,
               "input_tokens_per_task": sum(r["input_tokens"] for r in rows) / n,
               "tool_errors_per_task": sum(r["tool_errors"] for r in rows) / n,
               "truncations": sum(r["truncations"] for r in rows), "shell_calls": sum(r["shell_calls"] for r in rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\ntools={args.tools} sandbox={backend}: success {summary['success_rate']:.0%}, "
          f"{summary['input_tokens_per_task']:.0f} input tokens/task, {summary['tool_errors_per_task']:.2f} tool "
          f"errors/task, {summary['truncations']} truncations  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
