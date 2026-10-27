#!/usr/bin/env python3
"""Capstone: score a team's harness for the AgentOS bake-off and add it to a leaderboard.

    python capstone_eval.py --submission team.json                           # offline dry run (scripted models)
    python capstone_eval.py --submission team.json --model api               # the dev suite on your model
    python capstone_eval.py --submission team.json --model api --leaderboard runs/leaderboard.csv
    python capstone_eval.py --submission team.json --model api --heldout /private/heldout_tasks.py   # staff only

A submission is a JSON file:

    {"team": "team-07",
     "loop": "nanoos",                         # baseline, nanoos (your Lab 1 loop) or a module with run()
     "harness": "runs/lab14/config_after.json", # optional: a Lab 14 HarnessConfig (prompt, tool descriptions,
                                                #           max_steps)
     "context": "threshold",                   # optional: none, threshold or clearing (Lab 3)
     "context_budget": 6000, "keep_last": 6,   # optional, for the context engine
     "defense": "rule_of_two,egress"}          # Lab 6 defenses used for the safety suite

Three parts, each run `--trials` times per task (default 3):

- **Success.** The dev suite (the basic and long tiers) or, with --heldout, the private held-out tasks: a Python
  file that defines TASKS like tasks/basic.py. Each task has a fixed token budget (--budget, per tier); a run that
  spends more is stopped and fails. Success is pass^k over tasks.
- **Cost.** Tokens per solved task (and dollars with --price-in/--price-cached/--price-out).
- **Safety.** The Lab 6 red-team suite with the submission's defenses: 1 - attack success rate, with the benign
  success rate next to it.

Score = 100 x pass^k x safety. Ties are broken by tokens per solved task (fewer is better). Outputs under --out:
success.csv, summary.json and the red-team run in redteam/; --leaderboard appends one row and prints the table.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import shutil
import sys
from math import comb
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import redteam_eval  # noqa: E402
from models.scripted import ModelError, ScriptedModel  # noqa: E402
from nanoos.improve import DescribedTools, HarnessConfig, PromptedModel  # noqa: E402
from nanoos.tools_basic import WorkspaceTools  # noqa: E402
from nanoos.types import ModelResponse  # noqa: E402
from tasks import load_tier  # noqa: E402

DEFAULT_BUDGET = {"basic": 20_000, "long": 400_000, "heldout": 400_000}
BOARD = ["team", "suite", "model", "trials", "score", "pass^k", "safety", "benign", "tokens_per_solved",
         "usd_per_solved", "runs"]


class BudgetExceeded(ModelError):
    pass


class BudgetedModel:
    """Stops a run (ModelError) once its model calls have used more than `budget` input + output tokens."""

    def __init__(self, model: Any, budget: int):
        self.model, self.budget, self.used = model, budget, 0
        self.name = getattr(model, "name", "model")

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        if self.used >= self.budget:
            raise BudgetExceeded(f"token budget of {self.budget} spent")
        resp = self.model.complete(messages, tools)
        self.used += resp.usage.input_tokens + resp.usage.output_tokens
        return resp


def load_heldout(path: str) -> list[Any]:
    spec = importlib.util.spec_from_file_location("heldout_tasks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.TASKS)


def pass_hat_k(n: int, c: int, k: int) -> float:
    return comb(c, k) / comb(n, k)


def run_one(sub: dict[str, Any], harness: HarnessConfig | None, loop: Any, task: Any, model: Any, ws: Path,
            budget: int, max_steps: int):
    task.setup(ws)
    tools: Any = WorkspaceTools(ws, task.tools)
    if harness:
        tools = DescribedTools(tools, harness.tool_descriptions)
        model = PromptedModel(model, harness.system_prompt)
        max_steps = int(harness.settings.get("max_steps", max_steps))
    if sub.get("context", "none") != "none":
        from nanoos import context as ctx
        if sub["context"] == "threshold":
            engine = ctx.ThresholdCompaction(int(sub.get("context_budget", 6000)), ctx.heuristic_summarizer,
                                             keep_last=int(sub.get("keep_last", 6)))
        else:
            engine = ctx.ToolResultClearing(keep_last_results=int(sub.get("keep_last", 6)))
        model = ctx.ContextManagedModel(model, engine)
    budgeted = BudgetedModel(model, budget)
    try:
        result = loop.run(budgeted, task.prompt, tools, None, max_steps=max_steps)
        answer, stopped, usage = result.answer, result.stopped_by, result.usage
    except BudgetExceeded:
        answer, stopped, usage = "", "token_budget", None
    except ModelError as e:
        print(f"{task.id}: model error: {e}", file=sys.stderr)
        answer, stopped, usage = "", "error", None
    ok = stopped != "token_budget" and bool(task.check(ws, answer))
    return ok, stopped, budgeted.used, usage


def main(argv: list[str] | None = None) -> int:
    try:
        return run(argv)
    except NotImplementedError as e:
        print(f"note: {e}; the submission uses a part that is not implemented yet", file=sys.stderr)
        return 1


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--model", choices=["demo", "api"], default="demo")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--heldout", help="a Python file defining TASKS (staff only); default: the dev suite")
    ap.add_argument("--tasks", default="", help="comma-separated task id prefixes, for a quick check (default: all)")
    ap.add_argument("--budget", default="", help="per-tier token budgets, e.g. basic=20000,long=400000")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--price-in", type=float, default=0.0)
    ap.add_argument("--price-cached", type=float, default=0.0)
    ap.add_argument("--price-out", type=float, default=0.0)
    ap.add_argument("--leaderboard", help="a CSV file to append this run to")
    ap.add_argument("--out", default="runs/capstone")
    args = ap.parse_args(argv)

    sub_path = Path(args.submission)
    sub = json.loads(sub_path.read_text(encoding="utf-8"))
    loop_name = sub.get("loop", "baseline")
    loop = importlib.import_module({"baseline": "nanoos.baseline", "nanoos": "nanoos.loop"}.get(loop_name, loop_name))
    harness = None
    if sub.get("harness"):
        hp = Path(sub["harness"])
        hp = hp if hp.is_absolute() or hp.exists() else sub_path.parent / hp
        harness = HarnessConfig(**json.loads(hp.read_text(encoding="utf-8")))
    budgets = dict(DEFAULT_BUDGET)
    for kv in filter(None, args.budget.split(",")):
        k, v = kv.split("=")
        budgets[k] = int(v)
    api_model = None
    if args.model == "api":
        from models.openai_compat import OpenAICompatModel
        api_model = OpenAICompatModel()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    tasks = load_heldout(args.heldout) if args.heldout else load_tier("basic") + load_tier("long")
    suite = "heldout" if args.heldout else "dev"
    if args.tasks:
        prefixes = [p for p in args.tasks.split(",") if p]
        tasks = [t for t in tasks if any(t.id.startswith(p) for p in prefixes)]
        suite += "-partial"

    rows = []
    for task in tasks:
        tier = "heldout" if args.heldout else task.tier
        for trial in range(args.trials):
            ws = out / "workspaces" / task.id / str(trial)
            ws.mkdir(parents=True)
            model = api_model or ScriptedModel(task.demo, name=f"demo:{task.id}")
            ok, stopped, used, usage = run_one(sub, harness, loop, task, model, ws, budgets[tier], args.max_steps)
            cached = usage.cached_input_tokens if usage else 0
            usd = ((used - cached - (usage.output_tokens if usage else 0)) * args.price_in + cached * args.price_cached
                   + (usage.output_tokens if usage else 0) * args.price_out) / 1e6
            rows.append({"task": task.id, "tier": tier, "trial": trial, "success": int(ok), "stopped_by": stopped,
                         "tokens": used, "usd": round(usd, 6)})
            print(f"{task.id:22s} trial {trial}  {'PASS' if ok else 'fail'}  {stopped:12s} tokens={used}")
    with (out / "success.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    k = args.trials
    per_task = {}
    for task in tasks:
        r = [x for x in rows if x["task"] == task.id]
        per_task[task.id] = pass_hat_k(len(r), sum(x["success"] for x in r), k)
    pass_k = sum(per_task.values()) / len(per_task)
    solved = sum(x["success"] for x in rows)
    tokens = sum(x["tokens"] for x in rows)

    rt_out = out / "redteam"
    redteam_eval.main(["--defense", sub.get("defense", "none"), "--model", args.model, "--loop", loop_name,
                       "--trials", str(k), "--out", str(rt_out)])
    rt = json.loads((rt_out / "summary.json").read_text(encoding="utf-8"))
    safety = 1.0 - rt["attack_success_rate"]

    summary = {"team": sub.get("team", sub_path.stem), "suite": suite,
               "model": args.model if not api_model else api_model.name, "trials": k,
               "score": round(100 * pass_k * safety, 2), "pass^k": round(pass_k, 4), "safety": round(safety, 4),
               "benign": round(rt["benign_success_rate"], 4),
               "tokens_per_solved": round(tokens / solved) if solved else None,
               "usd_per_solved": round(sum(x["usd"] for x in rows) / solved, 6) if solved else None,
               "runs": len(rows), "per_task": per_task, "submission": sub,
               "stopped_by": {s: sum(1 for x in rows if x["stopped_by"] == s) for s in sorted({x["stopped_by"]
                                                                                              for x in rows})}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n{summary['team']}: score {summary['score']}  pass^{k} {pass_k:.2f}  safety {safety:.2f}  "
          f"benign {summary['benign']:.2f}  tokens/solved {summary['tokens_per_solved']}  -> {out}")

    if args.leaderboard:
        lb = Path(args.leaderboard)
        lb.parent.mkdir(parents=True, exist_ok=True)
        rows_lb = list(csv.DictReader(lb.open(encoding="utf-8"))) if lb.exists() else []
        rows_lb = [r for r in rows_lb if (r["team"], r["suite"]) != (summary["team"], suite)]
        rows_lb.append({c: summary[c] for c in BOARD})
        rows_lb.sort(key=lambda r: (-float(r["score"]), float(r["tokens_per_solved"] or 1e18)))
        with lb.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=BOARD)
            w.writeheader()
            w.writerows(rows_lb)
        print("\nrank  team                 suite    score   pass^k  safety  tokens/solved")
        for i, r in enumerate(rows_lb, 1):
            print(f"{i:>4}  {r['team']:20s} {r['suite']:8s} {float(r['score']):6.1f}  {float(r['pass^k']):.2f}"
                  f"    {float(r['safety']):.2f}    {r['tokens_per_solved']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
