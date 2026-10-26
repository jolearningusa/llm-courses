#!/usr/bin/env python3
"""Run a task tier k times per task and write traces, a metrics table and a summary.

Examples:

    python run_suite.py --model demo                      # offline: each task's scripted solution
    python run_suite.py --model api --trials 3 --out runs/baseline-3
    python run_suite.py --model api --loop nanoos --tasks b07,b09

Outputs under --out:

    metrics.csv        one row per (task, trial), same columns in every lab
    summary.json       success rate, pass@k and pass^k per task and overall, tokens and cost
    traces/*.jsonl     one trace per run (once Lab 0's JsonlTracer works)
    messages/*.json    the final message list of each run
    workspaces/        the files each run left behind, for inspection

--model api reads NANOOS_BASE_URL, NANOOS_API_KEY and NANOOS_MODEL (see models/openai_compat.py).
Prices are in USD per million tokens and only feed the cost column.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import shutil
import sys
import time
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.scripted import ModelError, ScriptedModel  # noqa: E402
from nanoos.tools_basic import WorkspaceTools  # noqa: E402
from nanoos.trace import JsonlTracer, NullTracer  # noqa: E402
from tasks import load_tier  # noqa: E402

COLUMNS = ["task", "trial", "success", "steps", "stopped_by", "input_tokens", "cached_input_tokens",
           "output_tokens", "cost_usd", "wall_s", "tool_errors", "approvals"]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k trials succeeds (unbiased estimate from n trials, c successes)."""
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k: probability that all k trials succeed (unbiased estimate from n trials, c successes)."""
    return comb(c, k) / comb(n, k)


def make_tracer(path: Path, run_id: str, warned: list[bool]):
    tracer = JsonlTracer(path, run_id)
    try:
        tracer.event("probe")  # dropped below; only checks that event() is implemented
    except NotImplementedError:
        tracer.close()
        path.unlink(missing_ok=True)
        if not warned[0]:
            print("note: JsonlTracer.event is not implemented yet (Lab 0), so no traces are written", file=sys.stderr)
            warned[0] = True
        return NullTracer()
    tracer.close()  # start the file again without the probe line
    return JsonlTracer(path, run_id)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default="basic")
    ap.add_argument("--tasks", default="", help="comma-separated task id prefixes (default: all)")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--model", choices=["demo", "api"], default="demo",
                    help="demo: replay each task's scripted solution; api: an OpenAI-compatible endpoint")
    ap.add_argument("--loop", default="baseline",
                    help="baseline (provided), nanoos (your Lab 1 loop) or any module with a run() function")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--price-in", type=float, default=0.0)
    ap.add_argument("--price-cached", type=float, default=0.0)
    ap.add_argument("--price-out", type=float, default=0.0)
    args = ap.parse_args(argv)

    loop = importlib.import_module({"baseline": "nanoos.baseline", "nanoos": "nanoos.loop"}.get(args.loop, args.loop))
    tasks = load_tier(args.tier)
    if args.tasks:
        prefixes = [p.strip() for p in args.tasks.split(",") if p.strip()]
        tasks = [t for t in tasks if any(t.id.startswith(p) for p in prefixes)]
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 2

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "traces").mkdir(parents=True)
    api_model = None
    if args.model == "api":
        from models.openai_compat import OpenAICompatModel
        api_model = OpenAICompatModel()

    rows, warned = [], [False]
    for task in tasks:
        for trial in range(args.trials):
            ws = out / "workspaces" / task.id / str(trial)
            ws.mkdir(parents=True)
            task.setup(ws)
            model = api_model or ScriptedModel(task.demo, name=f"demo:{task.id}")
            run_id = f"{task.id}__{trial}"
            tracer = make_tracer(out / "traces" / f"{run_id}.jsonl", run_id, warned)
            tracer.event("run_start", task=task.id, trial=trial, model=model.name, loop=args.loop)
            t0 = time.time()
            try:
                result = loop.run(model, task.prompt, WorkspaceTools(ws), tracer, max_steps=args.max_steps)
                answer, stopped_by = result.answer, result.stopped_by
            except ModelError as e:
                tracer.event("run_end", stopped_by="error", error=str(e))
                print(f"{run_id}: model error: {e}", file=sys.stderr)
                result, answer, stopped_by = None, "", "error"
            finally:
                tracer.close()
            wall = time.time() - t0
            if result is not None:
                (out / "messages").mkdir(exist_ok=True)
                (out / "messages" / f"{run_id}.json").write_text(json.dumps(result.messages, indent=1), encoding="utf-8")
            ok = bool(task.check(ws, answer))
            u = result.usage if result else None
            uncached = (u.input_tokens - u.cached_input_tokens) if u else 0
            cost = (uncached * args.price_in + (u.cached_input_tokens if u else 0) * args.price_cached
                    + (u.output_tokens if u else 0) * args.price_out) / 1e6
            rows.append({"task": task.id, "trial": trial, "success": int(ok), "steps": result.steps if result else 0,
                         "stopped_by": stopped_by, "input_tokens": u.input_tokens if u else 0,
                         "cached_input_tokens": u.cached_input_tokens if u else 0,
                         "output_tokens": u.output_tokens if u else 0, "cost_usd": round(cost, 6),
                         "wall_s": round(wall, 3), "tool_errors": result.tool_errors if result else 0,
                         "approvals": result.approvals if result else 0})
            print(f"{run_id:28s} {'PASS' if ok else 'fail'}  steps={rows[-1]['steps']:<3d} "
                  f"in={rows[-1]['input_tokens']:<7d} out={rows[-1]['output_tokens']}")

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    k = args.trials
    per_task = {}
    for task in tasks:
        r = [x for x in rows if x["task"] == task.id]
        c = sum(x["success"] for x in r)
        per_task[task.id] = {"trials": len(r), "successes": c, f"pass@{k}": pass_at_k(len(r), c, k),
                             f"pass^{k}": pass_hat_k(len(r), c, k)}
    summary = {
        "tier": args.tier, "loop": args.loop, "model": args.model if not api_model else api_model.name,
        "trials": k, "runs": len(rows),
        "success_rate": sum(x["success"] for x in rows) / len(rows),
        f"pass@{k}": sum(t[f"pass@{k}"] for t in per_task.values()) / len(per_task),
        f"pass^{k}": sum(t[f"pass^{k}"] for t in per_task.values()) / len(per_task),
        "input_tokens": sum(x["input_tokens"] for x in rows),
        "cached_input_tokens": sum(x["cached_input_tokens"] for x in rows),
        "output_tokens": sum(x["output_tokens"] for x in rows),
        "cost_usd": round(sum(x["cost_usd"] for x in rows), 6),
        "tasks": per_task,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nsuccess {summary['success_rate']:.0%}  pass@{k} {summary[f'pass@{k}']:.2f}  "
          f"pass^{k} {summary[f'pass^{k}']:.2f}  tokens in/out {summary['input_tokens']}/{summary['output_tokens']}"
          f"  cost ${summary['cost_usd']:.4f}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
