#!/usr/bin/env python3
"""Lab 2: compare scheduling policies on simulated agent workloads (uses your nanoos/sched_sim.py).

    python sched_experiment.py                                   # the deck's exercise settings
    python sched_experiment.py --long-frac 0.4 --tool-ms 500 3000 --kv-ttl 2000 --seeds 5
    python sched_experiment.py --csv runs/lab2/sweep.csv --sweep-long-frac 0 0.1 0.2 0.4

Prints mean and P99 job completion time (JCT), Jain's fairness index over JCT, re-prefills and GPU
utilization for each policy, averaged over --seeds workloads.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nanoos.sched_sim import (Simulator, fcfs_per_call, jain, least_attained, make_programs, mean,  # noqa: E402
                              p99, program_fcfs)

POLICIES = [fcfs_per_call, program_fcfs, least_attained]


def evaluate(args: argparse.Namespace, long_frac: float) -> list[dict]:
    rows = []
    for pol in POLICIES:
        acc = {"mean_jct_ms": [], "p99_jct_ms": [], "jain": [], "reprefills": [], "gpu_util": []}
        for seed in range(args.seeds):
            progs = make_programs(n=args.n, long_frac=long_frac, seed=seed, mean_gap_ms=args.gap_ms,
                                  tool_ms=tuple(args.tool_ms))
            sim = Simulator(gpu_slots=args.gpu_slots, kv_slots=args.kv_slots, kv_ttl_ms=args.kv_ttl,
                            reprefill_ms=lambda p: p.ctx_tokens / args.prefill_tok_per_ms)
            r = sim.run(progs, pol)
            jcts = list(r.jct_ms.values())
            acc["mean_jct_ms"].append(mean(jcts))
            acc["p99_jct_ms"].append(p99(jcts))
            acc["jain"].append(jain(jcts))
            acc["reprefills"].append(r.reprefills)
            acc["gpu_util"].append(r.gpu_busy_ms / (args.gpu_slots * r.makespan_ms) if r.makespan_ms else 0.0)
        rows.append({"policy": pol.__name__, "long_frac": long_frac, **{k: mean(v) for k, v in acc.items()}})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50, help="programs per workload")
    ap.add_argument("--long-frac", type=float, default=0.2)
    ap.add_argument("--sweep-long-frac", type=float, nargs="*", help="run several long fractions")
    ap.add_argument("--gap-ms", type=float, default=100.0, help="mean time between program arrivals")
    ap.add_argument("--tool-ms", type=float, nargs=2, default=[20.0, 200.0], metavar=("MIN", "MAX"))
    ap.add_argument("--gpu-slots", type=int, default=8)
    ap.add_argument("--kv-slots", type=int, default=32)
    ap.add_argument("--kv-ttl", type=float, default=None, help="drop KV when a tool call is longer (ms)")
    ap.add_argument("--prefill-tok-per-ms", type=float, default=4.0, help="re-prefill speed")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--csv", help="also write the rows to this CSV file")
    args = ap.parse_args(argv)

    rows = []
    for lf in args.sweep_long_frac or [args.long_frac]:
        rows += evaluate(args, lf)
    print(f"{'policy':16s} {'long':>5s} {'mean JCT':>10s} {'P99 JCT':>10s} {'Jain':>6s} {'re-prefill':>10s} {'GPU util':>8s}")
    for r in rows:
        print(f"{r['policy']:16s} {r['long_frac']:5.2f} {r['mean_jct_ms']:10.0f} {r['p99_jct_ms']:10.0f} "
              f"{r['jain']:6.3f} {r['reprefills']:10.1f} {r['gpu_util']:8.2f}")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
