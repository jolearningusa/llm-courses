"""Small coverage x reward experiment, not a reproduction of an LLM paper.

python coverage_ablation.py --output /tmp/coverage.json
No weights, downloads or API calls; fixed 80/20 prompt split; no held-out selection.
"""
import argparse
import copy
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import torch

from grpo import sft, train
from policy import Policy
from task import EOS, FILLER, MAX_LEN, decode, prompts, reward


def fixed_split():
    ps = prompts()
    random.Random(314159).shuffle(ps)
    return ps[:80], ps[80:]


def width_reward(prompt, tokens):
    """Deliberately weak shaping: exact=1; finished right-width wrong number=0.5; else=0.

    This is an answer-format proxy, not process supervision or a trusted dense reward.
    """
    if reward(prompt, tokens):
        return 1.0
    return 0.5 if EOS in tokens and len(decode(tokens)) == len(str(sum(prompt))) else 0.0


def strict_check(prompt, tokens):
    """Evaluation oracle, independent of reward() and decode(); fillers are allowed by task rules."""
    digits = []
    for token in tokens:
        if token == EOS:
            return int("".join(digits) == str(prompt[0] + prompt[1]))
        if 0 <= token <= 9:
            digits.append(str(token))
        elif token != FILLER:
            return 0
    return 0


def pass_at_k(n, c, k):
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("need 0 <= c <= n and 1 <= k <= n")
    return 1.0 - math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 1.0


@torch.no_grad()
def measure(policy, ps, seed, samples=16):
    g = torch.Generator().manual_seed(seed)
    a = torch.tensor([p[0] for p in ps]).repeat_interleave(samples)
    b = torch.tensor([p[1] for p in ps]).repeat_interleave(samples)
    tokens = policy.sample(a, b, generator=g)
    _, mask = policy.logprobs(a, b, tokens)
    counts = [sum(strict_check(p, t.tolist()) for t in group)
              for p, group in zip(ps, tokens.view(-1, samples, MAX_LEN))]
    return {"samples_per_prompt": samples, "strict_successes_per_prompt": counts,
            "pass_at_k": {str(k): sum(pass_at_k(samples, c, k) for c in counts) / len(ps)
                          for k in (1, 4, 8)},
            "mean_valid_length": float(mask.sum(1).mean()),
            "mean_width_reward": sum(width_reward((int(x), int(y)), t.tolist())
                                     for x, y, t in zip(a, b, tokens)) / len(tokens)}


def state_hash(policy):
    # Exact CPU tensor bytes without requiring NumPy.
    h = hashlib.sha256()
    for name, value in policy.state_dict().items():
        h.update(name.encode())
        h.update(bytes(value.detach().contiguous().view(torch.uint8).flatten().tolist()))
    return h.hexdigest()


def run(seeds, steps, sft_steps):
    pool, heldout = fixed_split()
    report = {"scope": "toy addition; weak width shaping; no frontier reasoning claim",
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "platform": platform.platform(), "device": "cpu", "dtype": "float32",
                              "threads": torch.get_num_threads()},
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("coverage_ablation.py", "grpo.py", "policy.py", "task.py")},
              "config": {"seeds": seeds, "steps": steps, "sft_steps": sft_steps,
                         "sft_batch": 32, "coverage_sizes": [10, 40], "G": 8,
                         "prompts_per_step": 16, "inner": 2, "lr": 0.003,
                         "max_len": MAX_LEN, "evaluation_samples": 16,
                         "algorithm": "GRPO, std and response-length normalization, no KL",
                         "checkpoint_selection": "fixed final step; no tuning/search",
                         "split_seed": 314159, "eval_seed_offset": 10000},
              "train_prompts": pool, "heldout_prompts": heldout, "runs": []}
    for seed in seeds:
        for coverage in (10, 40):
            torch.manual_seed(seed)
            base = Policy()
            start = time.perf_counter()
            sft(base, pool[:coverage], steps=sft_steps, seed=seed)
            sft_seconds = time.perf_counter() - start
            before_train = measure(base, pool, seed + 10000)
            before_test = measure(base, heldout, seed + 10000)
            base_hash = state_hash(base)
            for name, reward_fn in (("sparse", reward), ("width_proxy", width_reward)):
                policy = copy.deepcopy(base)
                trace = []
                start = time.perf_counter()
                train(policy, pool, steps=steps, seed=seed, reward_fn=reward_fn, trace=trace)
                seconds = time.perf_counter() - start
                totals = {key: sum(step[key] for step in trace) for key in trace[0]}
                report["runs"].append({"seed": seed, "sft_coverage": coverage, "reward": name,
                    "base_sha256": base_hash, "final_sha256": state_hash(policy),
                    "sft_seconds": sft_seconds, "rl_seconds": seconds,
                    "before_train": before_train, "before_heldout": before_test,
                    "after_heldout": measure(policy, heldout, seed + 10000),
                    "totals": totals, "trace": trace})
                print(f"seed={seed} coverage={coverage} reward={name} finished", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--sft-steps", type=int, default=300)
    args = ap.parse_args()
    if args.steps < 1 or args.sft_steps < 1 or len(set(args.seeds)) != len(args.seeds):
        ap.error("positive step counts and distinct seeds required")
    if args.output.exists():
        ap.error("output exists; use a new file")
    torch.set_num_threads(1)
    report = run(args.seeds, args.steps, args.sft_steps)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
