#!/usr/bin/env python3
"""Lab 14: run the improvement loop (nanoos/improve.py) and report what it kept and what it cost.

    python improve_eval.py                                    # offline: the simulated agent and scripted proposer
    python improve_eval.py --model api --proposer api --trials 2 --out runs/lab14-api
    python improve_eval.py --model api --proposer api --start runs/lab14-api/config_after.json --rounds 1

The DEV and HELDOUT task split comes from tasks/improve_sim.py (six and four basic-tier tasks). Offline, every
run uses HabitModel, a simulated agent whose bad habits the right edits remove; with --model api, the agent is
your model. --proposer api asks your model (or --proposer-model) for edits.

Outputs under --out: config_before.json and config_after.json, lessons.jsonl (one decision per proposed edit, in
the course's lesson schema), history.json (DEV and HELDOUT success per round) and summary.json: success before
and after, agent runs and tokens the loop spent, proposer calls, and the decisions by kind.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nanoos.improve import Evaluator, HarnessConfig, ModelProposer, ScriptedProposer, improve  # noqa: E402
from tasks.basic import TASKS  # noqa: E402
from tasks.improve_sim import DEV, HELDOUT, HabitModel  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["demo", "api"], default="demo", help="demo: the simulated agent")
    ap.add_argument("--proposer", choices=["demo", "api"], default="demo")
    ap.add_argument("--proposer-model", help="model name for --proposer api (default: NANOOS_MODEL)")
    ap.add_argument("--loop", default="baseline", help="baseline or nanoos (your Lab 1 loop)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--trials", type=int, default=1, help="trials per task; a task passes only if all pass")
    ap.add_argument("--start", help="a config_*.json to start from (default: the bare starting harness)")
    ap.add_argument("--out", default="runs/lab14")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    import importlib
    loop = importlib.import_module({"baseline": "nanoos.baseline", "nanoos": "nanoos.loop"}.get(args.loop, args.loop))
    if args.model == "api":
        from models.openai_compat import OpenAICompatModel
        agent = OpenAICompatModel()
        make_model = lambda tid: agent  # noqa: E731 - the API client keeps no state between runs
    else:
        make_model = HabitModel
    if args.proposer == "api":
        from models.openai_compat import OpenAICompatModel
        proposer = ModelProposer(OpenAICompatModel(model=args.proposer_model) if args.proposer_model
                                 else OpenAICompatModel())
    else:
        proposer = ScriptedProposer()
    config = HarnessConfig(**json.loads(Path(args.start).read_text())) if args.start else HarnessConfig()
    (out / "config_before.json").write_text(json.dumps(config.to_json(), indent=2) + "\n", encoding="utf-8")

    evaluator = Evaluator(TASKS, make_model, out / "workspaces", trials=args.trials, loop=loop)
    try:
        result = improve(config, evaluator, proposer, list(DEV), list(HELDOUT), rounds=args.rounds)
    except NotImplementedError as e:
        print(f"note: {e} (Lab 14), so the loop cannot run yet", file=sys.stderr)
        return 1

    (out / "config_after.json").write_text(json.dumps(result.config.to_json(), indent=2) + "\n", encoding="utf-8")
    with (out / "lessons.jsonl").open("w", encoding="utf-8") as f:
        for lesson in result.lessons:
            f.write(json.dumps(lesson) + "\n")
    (out / "history.json").write_text(json.dumps(result.history, indent=2) + "\n", encoding="utf-8")
    before, after = result.history[0], result.history[-1]
    decisions = Counter(x["decision"] for x in result.lessons)
    summary = {"model": args.model, "proposer": args.proposer, "trials": args.trials, "rounds": len(result.history) - 1,
               "dev_before": before["dev"], "dev_after": after["dev"], "heldout_before": before["heldout"],
               "heldout_after": after["heldout"], "agent_runs": result.runs, "agent_tokens": result.tokens,
               "proposer_calls": result.proposer_calls, "decisions": dict(decisions),
               "heldout_shown_to_proposer": sorted(set(getattr(proposer, "seen", [])) & set(HELDOUT))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for x in result.lessons:
        print(f"{x['lesson_id']:9s} {x['decision']:18s} {x['proposed_edit'][:90]}")
    print(f"\nDEV {before['dev']:.0%} -> {after['dev']:.0%}, HELDOUT {before['heldout']:.0%} -> {after['heldout']:.0%}"
          f" after {summary['rounds']} round(s); {result.runs} agent runs, {result.tokens} tokens, "
          f"{result.proposer_calls} proposer calls  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
