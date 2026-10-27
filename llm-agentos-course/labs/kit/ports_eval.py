#!/usr/bin/env python3
"""Lab 12: your Lab 3 engines behind real plugin contracts, on the long tier (using nanoos/ports.py).

    python ports_eval.py --host hermes --engine none                       # no engine: the floor
    python ports_eval.py --host hermes --engine threshold --mode compress  # Hermes rewrites the transcript
    python ports_eval.py --host hermes --engine threshold --mode select    # ... or a view per request
    python ports_eval.py --host hermes --engine clearing --mode select
    python ports_eval.py --host openclaw --engine threshold                # assemble + compact, in process
    python ports_eval.py --host openclaw --engine threshold --bridge       # the same over stdio, as OpenClaw does
    python ports_eval.py --host hermes --engine threshold --model api --trials 3 --out runs/lab12-api

The three long-tier tasks (tasks/long.py) run through nanoos/hosts.py, which calls the plugins the way each
harness does. --context-length sets Hermes's window (compression starts at 75% of it) and --budget OpenClaw's
token budget; both default to small values so the scripted runs compact. Outputs under --out: metrics.csv (one
row per task and trial) and summary.json: success, prompt tokens in total and at the peak, transcript
rewrites, compactions inside views, the cacheable share (prompt tokens that repeat the previous request's
leading messages) and, for OpenClaw, whether the engine was quarantined.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.scripted import ModelError, ScriptedModel  # noqa: E402
from nanoos.context import ModelSummarizer  # noqa: E402
from nanoos.hosts import BridgeClient, HermesHost, OpenClawHost  # noqa: E402
from nanoos.ports import HermesContextEngine, OpenClawEngine, make_engine  # noqa: E402
from nanoos.tools_basic import WorkspaceTools  # noqa: E402
from tasks.long import TASKS  # noqa: E402

COLUMNS = ["task", "trial", "success", "requests", "prompt_tokens", "peak_prompt_tokens", "rewrites",
           "view_compactions", "cacheable_share", "quarantined"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", choices=["hermes", "openclaw"], default="hermes")
    ap.add_argument("--engine", choices=["none", "threshold", "clearing"], default="threshold")
    ap.add_argument("--mode", choices=["compress", "select"], default="compress", help="Hermes only")
    ap.add_argument("--context-length", type=int, default=8000, help="Hermes: the model's window in tokens")
    ap.add_argument("--budget", type=int, default=6000, help="OpenClaw: the token budget per request")
    ap.add_argument("--bridge", action="store_true", help="OpenClaw: run the engine in a subprocess over stdio")
    ap.add_argument("--keep-last", type=int, default=6)
    ap.add_argument("--model", choices=["demo", "api"], default="demo")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--out", default="runs/lab12")
    args = ap.parse_args(argv)

    api_model = None
    if args.model == "api":
        from models.openai_compat import OpenAICompatModel
        api_model = OpenAICompatModel()
    summarizer = ModelSummarizer(api_model) if api_model else None
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rows = []
    for task in TASKS:
        for trial in range(args.trials):
            ws = out / "workspaces" / task.id / str(trial)
            ws.mkdir(parents=True)
            task.setup(ws)
            model = api_model or ScriptedModel(task.demo, name=f"demo:{task.id}")
            tools = WorkspaceTools(ws, task.tools)
            factory = make_engine(args.engine, args.keep_last, summarizer)
            bridge, host = None, None
            try:
                if args.host == "hermes":
                    engine = None if args.engine == "none" else HermesContextEngine(
                        factory, args.mode, context_length=args.context_length)
                    host = HermesHost(model, tools, engine=engine, hermes_home=out / "hermes_home",
                                      max_steps=args.max_steps)
                else:
                    if args.bridge:
                        engine = bridge = BridgeClient(["--engine", args.engine, "--keep-last", str(args.keep_last)])
                    else:
                        engine = OpenClawEngine(factory)
                    host = OpenClawHost(model, tools, engine, token_budget=args.budget, max_steps=args.max_steps)
                r = host.run([task.prompt])
            except ModelError as e:
                print(f"{task.id}: model error: {e}", file=sys.stderr)
                continue
            finally:
                if bridge:
                    bridge.close()
            rows.append({"task": task.id, "trial": trial, "success": int(task.check(ws, r.answers[-1])),
                         "requests": len(r.requests), "prompt_tokens": r.total_prompt_tokens,
                         "peak_prompt_tokens": max(r.prompt_tokens, default=0), "rewrites": r.compressions,
                         "view_compactions": r.view_compactions, "cacheable_share": round(r.cacheable_share(), 3),
                         "quarantined": getattr(host, "quarantined", None) or ""})
            x = rows[-1]
            print(f"{task.id:18s} {'PASS' if x['success'] else 'fail'}  tokens={x['prompt_tokens']} "
                  f"peak={x['peak_prompt_tokens']} rewrites={x['rewrites']} cacheable={x['cacheable_share']:.0%}"
                  + (f"  QUARANTINED {x['quarantined']}" if x["quarantined"] else ""))

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    if not rows:
        return 1
    summary = {"host": args.host + (" (bridge)" if args.bridge else ""), "engine": args.engine,
               "mode": args.mode if args.host == "hermes" else None,
               "model": args.model if not api_model else api_model.name, "trials": args.trials,
               "success": sum(r["success"] for r in rows) / len(rows),
               "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
               "peak_prompt_tokens": max(r["peak_prompt_tokens"] for r in rows),
               "rewrites": sum(r["rewrites"] for r in rows),
               "view_compactions": sum(r["view_compactions"] for r in rows),
               "cacheable_share": round(sum(r["cacheable_share"] for r in rows) / len(rows), 3),
               "quarantined": sum(bool(r["quarantined"]) for r in rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n{summary['host']} {args.engine}{'/' + args.mode if args.host == 'hermes' else ''}: success "
          f"{summary['success']:.0%}, {summary['prompt_tokens']} prompt tokens (peak {summary['peak_prompt_tokens']}), "
          f"{summary['rewrites']} rewrites, cacheable {summary['cacheable_share']:.0%}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
