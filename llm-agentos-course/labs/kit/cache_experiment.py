#!/usr/bin/env python3
"""Lab 5: break the cache on purpose (KV-cache deck exercise), using your nanoos/prompt.py.

    python cache_experiment.py                                   # offline: every variant, simulated prefix cache
    python cache_experiment.py --sessions 8 --cache-blocks 6000  # interleaved sessions competing for KV blocks
    python cache_experiment.py --model api --variants baseline,timestamp --csv runs/lab5/vllm.csv

Each variant replays the same scripted 20-turn session (tasks/cache_session.py) and differs in one change:
a timestamp, a tool added at turn 10, a memory edit at turn 10, compaction at turn 15, or a masked tool.
For every request it prints the simulated block hits, the cause of the first changed block (diagnose) and,
with --model api, what the server reported: cached prompt tokens and the request time with max_tokens=1
(about the time to first token). Point NANOOS_BASE_URL at a vLLM server started with prefix caching and
--enable-prompt-tokens-details (see the handout).

The summary gives the hit rate, prefill tokens per session (prompt tokens not served from cache) and the
projected API cost in units of one uncached input token, with the deck's multipliers: cache reads at
--read-mult (0.1) and cache writes at --write-mult (1.25).

--scrape URL prints vLLM's prefix-cache counters from URL (a /metrics endpoint) and exits; run it before and
after a session in Hermes or OpenClaw to get that harness's hit rate.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nanoos.prompt import BLOCK_SIZE, PrefixCache, block_hashes, cache_tokens, diagnose, render  # noqa: E402
from tasks.cache_session import VARIANTS, session_prompt  # noqa: E402

COLUMNS = ["variant", "session", "turn", "prompt_tokens", "hit_tokens", "cause", "block", "segment",
           "api_prompt_tokens", "api_cached_tokens", "api_latency_s"]


def scrape(url: str) -> dict[str, float]:
    """Sum vLLM's prefix-cache counters (vllm:prefix_cache_queries / _hits, any labels) from a /metrics page."""
    text = urllib.request.urlopen(url, timeout=10).read().decode()
    out = {"queries": 0.0, "hits": 0.0}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key in out:
            if line.startswith(f"vllm:prefix_cache_{key}"):
                out[key] += float(line.rsplit(" ", 1)[1])
    return out


def run_variant(variant: str, args, api_model) -> list[dict]:
    cache = PrefixCache(args.cache_blocks)
    tag = uuid.uuid4().hex[:8] if api_model else "sim"  # a fresh prefix per run, so earlier runs cannot help
    prev: dict[int, object] = {}
    rows = []
    for t in range(args.turns):
        for s in range(args.sessions):  # round-robin: sessions compete for the same cache
            prompt = session_prompt(variant, t, s, tag)
            tokens = cache_tokens(render(prompt))
            hits = cache.access(block_hashes(tokens, args.block_size))
            brk = diagnose(prev.get(s), prompt, args.block_size)
            prev[s] = prompt
            row = {"variant": variant, "session": s, "turn": t, "prompt_tokens": len(tokens),
                   "hit_tokens": hits * args.block_size, "cause": brk.cause, "block": brk.block,
                   "segment": brk.segment, "api_prompt_tokens": "", "api_cached_tokens": "", "api_latency_s": ""}
            if api_model:
                t0 = time.perf_counter()
                resp = api_model.complete(prompt.messages, prompt.tools)
                row.update(api_prompt_tokens=resp.usage.input_tokens, api_cached_tokens=resp.usage.cached_input_tokens,
                           api_latency_s=round(time.perf_counter() - t0, 4))
            rows.append(row)
    return rows


def summarize(rows: list[dict], args) -> dict:
    prompt = sum(r["prompt_tokens"] for r in rows)
    hit = sum(r["hit_tokens"] for r in rows)
    cost = sum(r["hit_tokens"] * args.read_mult + (r["prompt_tokens"] - r["hit_tokens"]) * args.write_mult for r in rows)
    out = {"hit_rate": hit / prompt if prompt else 0.0, "prefill_tokens_per_session": (prompt - hit) / args.sessions,
           "cost_units": round(cost), "cost_vs_no_cache": cost / prompt if prompt else 0.0,
           "breaks": [f"t{r['turn']}:{r['cause']}" for r in rows if r["session"] == 0 and r["cause"] not in ("first", "none")]}
    api = [r for r in rows if r["api_prompt_tokens"] != ""]
    if api:
        p = sum(r["api_prompt_tokens"] for r in api)
        out["api_hit_rate"] = sum(r["api_cached_tokens"] for r in api) / p if p else 0.0
        later = [r["api_latency_s"] for r in api if r["turn"] > 0]
        out["api_mean_latency_s"] = sum(later) / len(later) if later else 0.0
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    ap.add_argument("--cache-blocks", type=int, default=1_000_000, help="simulated KV capacity in blocks")
    ap.add_argument("--model", choices=["sim", "api"], default="sim")
    ap.add_argument("--read-mult", type=float, default=0.1)
    ap.add_argument("--write-mult", type=float, default=1.25)
    ap.add_argument("--csv", help="write every request's row to this CSV file")
    ap.add_argument("--quiet", action="store_true", help="only print the summary table")
    ap.add_argument("--scrape", metavar="URL", help="print vLLM prefix-cache counters from a /metrics URL and exit")
    args = ap.parse_args(argv)

    if args.scrape:
        c = scrape(args.scrape)
        print(f"prefix_cache_queries {c['queries']:.0f}  prefix_cache_hits {c['hits']:.0f}  "
              f"hit rate {c['hits'] / c['queries'] if c['queries'] else 0:.1%}")
        return 0
    api_model = None
    if args.model == "api":
        from models.openai_compat import OpenAICompatModel
        api_model = OpenAICompatModel(max_tokens=1, temperature=0.0)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    all_rows, summaries = [], {}
    for v in variants:
        rows = run_variant(v, args, api_model)
        all_rows += rows
        summaries[v] = summarize(rows, args)
        if not args.quiet:
            print(f"\n{v}")
            for r in rows:
                if r["session"] == 0:
                    api = (f"  api cached {r['api_cached_tokens']}/{r['api_prompt_tokens']} in {r['api_latency_s']}s"
                           if api_model else "")
                    print(f"  turn {r['turn']:2d}  tokens {r['prompt_tokens']:6d}  hits {r['hit_tokens']:6d}  "
                          f"{r['cause']:9s} {'' if r['block'] is None else 'block ' + str(r['block'])}{api}")

    print(f"\n{'variant':15s} {'hit rate':>8s} {'prefill/session':>15s} {'cost vs none':>12s}"
          + ("  api hit  latency" if api_model else "") + "  breaks")
    for v, s in summaries.items():
        api = f"  {s['api_hit_rate']:7.1%}  {s['api_mean_latency_s']:6.3f}s" if api_model else ""
        print(f"{v:15s} {s['hit_rate']:8.1%} {s['prefill_tokens_per_session']:15.0f} {s['cost_vs_no_cache']:12.2f}{api}  "
              f"{' '.join(s['breaks'][:4])}{' ...' if len(s['breaks']) > 4 else ''}")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(all_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
