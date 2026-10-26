#!/usr/bin/env python3
"""Lab 9: when do sub-agents pay for themselves? (multi-agent deck exercise), using nanoos/delegate.py.

    python multiagent_eval.py --arch single                    # offline: one agent reads everything
    python multiagent_eval.py --arch orchestrator              # ... a parent with workers (delegate_many)
    python multiagent_eval.py --arch orchestrator --latency-s 0.05   # simulate model latency: wall-clock time
    python multiagent_eval.py --arch orchestrator --model api --worker-model small-model --trials 5 \\
        --loop nanoos --out runs/lab9-orch

Two research tasks (tasks/research.py): a survey whose five sub-questions are independent, and a three-hop
question whose hops depend on each other. Workers get list_files and read_file. Outputs under --out:
metrics.csv (one row per task and trial) and summary.json: accuracy, pass^k, total tokens (parent and
workers), the parent's context at the end, wall-clock time, workers, worker errors and peak concurrency.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.scripted import ModelError, ScriptedModel  # noqa: E402
from nanoos.context import count_tokens  # noqa: E402
from nanoos.delegate import Delegator  # noqa: E402
from nanoos.tools_basic import WorkspaceTools  # noqa: E402
from nanoos.types import ModelResponse  # noqa: E402
from tasks.research import FILE_TOOLS, TASKS  # noqa: E402

COLUMNS = ["task", "arch", "trial", "success", "steps", "parent_tokens", "worker_tokens", "total_tokens",
           "parent_context_tokens", "wall_s", "workers", "worker_errors", "peak_concurrency"]
ORCHESTRATOR_HINT = ("\n\nYou can hand self-contained sub-tasks to worker agents with delegate (one) or delegate_many "
                     "(several in parallel). Workers see only the brief you write.")


class Slow:
    """Adds a fixed latency to every model call, to see wall-clock effects offline."""

    def __init__(self, model: Any, latency_s: float):
        self.model, self.latency_s, self.name = model, latency_s, model.name

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        time.sleep(self.latency_s)
        return self.model.complete(messages, tools)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=["single", "orchestrator"], default="orchestrator")
    ap.add_argument("--model", choices=["demo", "api"], default="demo")
    ap.add_argument("--worker-model", default=None, help="with --model api: a model name for the workers")
    ap.add_argument("--loop", default="baseline", help="baseline, nanoos (your Lab 1 loop) or a module with run()")
    ap.add_argument("--max-concurrency", type=int, default=5)
    ap.add_argument("--max-children", type=int, default=10)
    ap.add_argument("--latency-s", type=float, default=0.0, help="simulated latency per model call")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--out", default="runs/lab9")
    args = ap.parse_args(argv)

    loop = importlib.import_module({"baseline": "nanoos.baseline", "nanoos": "nanoos.loop"}.get(args.loop, args.loop))
    parent_api = worker_api = None
    if args.model == "api":
        from models.openai_compat import OpenAICompatModel
        parent_api = OpenAICompatModel()
        worker_api = OpenAICompatModel(model=args.worker_model) if args.worker_model else parent_api
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rows = []
    for task in TASKS:
        for trial in range(args.trials):
            with tempfile.TemporaryDirectory() as d:
                ws = Path(d)
                for rel, content in task.files.items():
                    (ws / rel).parent.mkdir(parents=True, exist_ok=True)
                    (ws / rel).write_text(content, encoding="utf-8")
                tools: Any = WorkspaceTools(ws, allowed=FILE_TOOLS)
                prompt, delegator = task.prompt, None
                if args.arch == "orchestrator":
                    def workers(brief: str, depth: int, task=task) -> Any:
                        m = worker_api or ScriptedModel(task.demo_workers.get(brief, [{"raise": "no script"}]))
                        return Slow(m, args.latency_s) if args.latency_s else m
                    delegator = Delegator(tools, workers, loop.run, max_children=args.max_children,
                                          max_concurrency=args.max_concurrency, child_max_steps=args.max_steps)
                    tools, prompt = delegator, prompt + ORCHESTRATOR_HINT
                script = task.demo_single if args.arch == "single" else task.demo_parent
                model: Any = parent_api or ScriptedModel(script)
                if args.latency_s:
                    model = Slow(model, args.latency_s)
                t0 = time.monotonic()
                try:
                    r = loop.run(model, prompt, tools, None, max_steps=args.max_steps)
                    answer, steps, usage, context = r.answer, r.steps, r.usage, count_tokens(r.messages)
                except ModelError as e:
                    print(f"{task.id}: model error: {e}", file=sys.stderr)
                    answer, steps, usage, context = "", 0, None, 0
                wall = time.monotonic() - t0
            parent = (usage.input_tokens + usage.output_tokens) if usage else 0
            wu = delegator.children_usage() if delegator else None
            worker = (wu.input_tokens + wu.output_tokens) if wu else 0
            recs = delegator.records if delegator else []
            rows.append({"task": task.id, "arch": args.arch, "trial": trial, "success": int(task.check(answer)),
                         "steps": steps, "parent_tokens": parent, "worker_tokens": worker,
                         "total_tokens": parent + worker, "parent_context_tokens": context, "wall_s": round(wall, 3),
                         "workers": len(recs), "worker_errors": sum(x.status != "ok" for x in recs),
                         "peak_concurrency": delegator.peak_concurrency if delegator else 0})
            r_ = rows[-1]
            print(f"{task.id:18s} {'PASS' if r_['success'] else 'fail'}  total={r_['total_tokens']:<7d} "
                  f"parent context={r_['parent_context_tokens']:<6d} workers={r_['workers']} wall={r_['wall_s']}s")

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    summary = {"arch": args.arch, "model": args.model if not parent_api else parent_api.name,
               "worker_model": worker_api.name if worker_api else args.model, "trials": args.trials,
               "accuracy": sum(r["success"] for r in rows) / len(rows),
               "pass_hat_k": sum(all(r["success"] for r in v) for v in by_task.values()) / len(by_task),
               "tasks": {t: {k: sum(r[k] for r in v) / len(v) for k in
                             ("success", "total_tokens", "parent_context_tokens", "wall_s", "workers", "worker_errors")}
                         for t, v in by_task.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\narch={args.arch}: accuracy {summary['accuracy']:.0%}, pass^{args.trials} {summary['pass_hat_k']:.0%}"
          f"  -> {out}")
    for t, s in summary["tasks"].items():
        print(f"  {t:18s} total tokens {s['total_tokens']:.0f}, parent context {s['parent_context_tokens']:.0f}, "
              f"wall {s['wall_s']:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
