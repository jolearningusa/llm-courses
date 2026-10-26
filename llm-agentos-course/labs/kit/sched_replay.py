#!/usr/bin/env python3
"""Lab 2, part C: replay a simulated agent workload against a real vLLM server.

Each program runs in its own thread: it waits for its arrival time, then for each step sends one chat
completion (its prompt grows step by step, like an agent's history) and sleeps for the step's tool time.
With --priority program, every request carries vLLM's `priority` field set to the program's arrival rank,
which makes vLLM's priority scheduler behave like program-level FCFS (lower values are served first).

    # server, in another terminal (one GPU):
    vllm serve Qwen/Qwen3-8B --scheduling-policy priority --max-num-seqs 8
    # replay, once without and once with program priorities:
    python sched_replay.py --base-url http://localhost:8000/v1 --model Qwen/Qwen3-8B --priority none
    python sched_replay.py --base-url http://localhost:8000/v1 --model Qwen/Qwen3-8B --priority program

Check the `priority` request field and `--scheduling-policy` in the vLLM docs for the version you run.
Prints mean and P99 job completion time and Jain's index, and writes per-program JCTs with --csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nanoos.sched_sim import jain, make_programs, mean, p99  # noqa: E402

FILLER = "The agent read another file and noted what it found. "  # about 12 tokens


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--priority", choices=["none", "program"], default="none")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--long-frac", type=float, default=0.25)
    ap.add_argument("--gap-ms", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens-per-step", type=int, default=400, help="prompt growth per step")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--csv")
    args = ap.parse_args(argv)

    programs = make_programs(n=args.n, long_frac=args.long_frac, seed=args.seed, mean_gap_ms=args.gap_ms)
    url = args.base_url.rstrip("/") + "/chat/completions"
    t0 = time.time() + 0.5
    jct: dict[str, float] = {}
    errors: list[str] = []

    def worker(rank: int, p) -> None:
        time.sleep(max(0.0, t0 + p.arrival_ms / 1000 - time.time()))
        start = time.time()
        history = f"Program {p.id}. " + FILLER * 20
        try:
            for i, (_, tool_ms) in enumerate(p.steps):
                history += FILLER * (args.tokens_per_step // 12)
                body = {"model": args.model, "max_tokens": args.max_tokens, "temperature": 0,
                        "messages": [{"role": "user", "content": history + f"\nStep {i}: reply with one sentence."}]}
                if args.priority == "program":
                    body["priority"] = rank
                post(url, body, args.timeout)
                time.sleep(tool_ms / 1000)
            jct[p.id] = (time.time() - start) * 1000
        except Exception as e:  # noqa: BLE001 - report and keep the other programs running
            errors.append(f"{p.id}: {e}")

    threads = [threading.Thread(target=worker, args=(rank, p)) for rank, p in enumerate(programs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    xs = list(jct.values())
    print(f"priority={args.priority} programs={len(xs)}/{len(programs)} mean JCT {mean(xs):.0f} ms  "
          f"P99 {p99(xs):.0f} ms  Jain {jain(xs):.3f}")
    for e in errors[:5]:
        print("error:", e, file=sys.stderr)
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["program", "steps", "jct_ms", "priority"])
            for p in programs:
                if p.id in jct:
                    w.writerow([p.id, len(p.steps), round(jct[p.id], 1), args.priority])
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
