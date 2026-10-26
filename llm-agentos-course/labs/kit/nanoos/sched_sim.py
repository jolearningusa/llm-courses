"""Lab 2: a discrete-event simulator for scheduling agent programs.

The scheduler deck's point: an agent is a *program* (LLM call, tool call, LLM call, ...), but an engine like
vLLM schedules single calls. This simulator lets you compare policies that see only calls with policies that
see programs, and what keeping a program's KV cache across its tool calls buys.

The model (all times in milliseconds):

- A Program arrives at `arrival_ms` and runs its `steps` in order. Step i is (llm_ms, tool_ms): one LLM call
  that needs a GPU slot for llm_ms, then a tool call that needs no GPU for tool_ms. The program is done when
  the tool call of its last step ends. Its job completion time (JCT) is done time minus arrival time.
- When a program is ready for an LLM call it becomes a Request in the waiting set. Whenever a GPU slot is free,
  the scheduler starts the waiting request with the smallest policy key (a policy is a function from Request
  to a sortable key). Calls are not preempted.
- KV cache. After an LLM call that is not the program's last, the program keeps its KV cache during the tool
  call if keeps_kv() says so; it then occupies one of `kv_slots` until its next LLM call starts. A later LLM call
  (step > 0) whose program did not keep its KV takes reprefill_ms(program) longer. The first call (step 0)
  never pays re-prefill: its prefill is part of llm_ms.
- Time and ties. Process all events that happen at the same time before starting any call. Several LLM calls
  that finish at the same time are handled (KV decision included) in the order they were started. Requests
  with equal keys are ordered by the key's own tie-breakers (the policy functions end with the program id).

`make grade LAB=2` checks hand-computed job completion times; `python sched_experiment.py` runs the deck's
exercise on random workloads with your code. Search for TODO.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Program:
    id: str
    arrival_ms: float
    steps: list[tuple[float, float]]  # (llm_ms, tool_ms) per step
    ctx_tokens: int = 4000            # context size; the default re-prefill cost grows with it


@dataclass
class Request:
    program: Program
    step: int            # which step's LLM call this is
    ready_ms: float      # when it joined the waiting set
    attained_ms: float   # GPU time the program had received before this request (re-prefills included)


Policy = Callable[[Request], tuple]


# --- policies: smaller key runs first ------------------------------------------------------------------
def fcfs_per_call(r: Request) -> tuple:
    """vLLM's default: the earliest request first, whatever program it belongs to (ties: program id)."""
    # TODO(lab 2)
    raise NotImplementedError("Lab 2: implement fcfs_per_call")


def program_fcfs(r: Request) -> tuple:
    """Continuum-style: the program that arrived first goes first, so started programs finish (ties: id)."""
    # TODO(lab 2)
    raise NotImplementedError("Lab 2: implement program_fcfs")


def least_attained(r: Request) -> tuple:
    """Autellix-style: the program with the least GPU time so far goes first (ties: ready time, then id)."""
    # TODO(lab 2)
    raise NotImplementedError("Lab 2: implement least_attained")


# --- metrics ---------------------------------------------------------------------------------------------
def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs: list[float], p: float) -> float:
    """Nearest-rank percentile: the smallest value with at least p% of the values at or below it (0.0 if empty)."""
    # TODO(lab 2)
    raise NotImplementedError("Lab 2: implement percentile")


def p99(xs: list[float]) -> float:
    return percentile(xs, 99)


def jain(xs: list[float]) -> float:
    """Jain's fairness index (sum x)^2 / (n * sum x^2): 1.0 when all equal, 1/n when one value has it all.

    Return 1.0 for an empty list or when every value is 0.
    """
    # TODO(lab 2)
    raise NotImplementedError("Lab 2: implement jain")


# --- the simulator ----------------------------------------------------------------------------------------
@dataclass
class SimResult:
    jct_ms: dict[str, float]      # program id -> job completion time
    reprefills: int = 0           # LLM calls that paid re-prefill
    kv_kept: int = 0              # tool calls during which a program kept its KV
    gpu_busy_ms: float = 0.0      # total GPU time used, re-prefills included
    makespan_ms: float = 0.0      # time the last program finished


@dataclass
class Simulator:
    gpu_slots: int = 8
    kv_slots: int = 32
    reprefill_ms: Callable[[Program], float] = field(default=lambda p: p.ctx_tokens / 4)
    kv_ttl_ms: float | None = None  # keep KV only for tool calls of at most this long (None: any length)

    def keeps_kv(self, tool_ms: float, kv_in_use: int) -> bool:
        """Keep a program's KV during a tool call of tool_ms, given kv_in_use slots already held?

        Yes when a slot is free and (there is no TTL, or the tool call is not longer than the TTL).
        """
        # TODO(lab 2)
        raise NotImplementedError("Lab 2: implement keeps_kv")

    def run(self, programs: list[Program], policy: Policy) -> SimResult:
        """Simulate the programs under `policy` and return their job completion times and counters.

        Hint: a heap of (time, sequence number, kind, payload) events with two kinds, "ready" (a program
        wants its next LLM call) and "llm_done", is enough. The sequence number keeps same-time events in
        the order they were pushed.
        """
        # TODO(lab 2)
        raise NotImplementedError("Lab 2: implement Simulator.run")


# --- workloads ---------------------------------------------------------------------------------------------
def make_programs(n: int = 50, long_frac: float = 0.2, seed: int = 0, mean_gap_ms: float = 100.0,
                  tool_ms: tuple[float, float] = (20.0, 200.0)) -> list[Program]:
    """n programs with Poisson arrivals; a share `long_frac` has 10-20 steps, the rest 2-4."""
    rng = random.Random(seed)
    t, out = 0.0, []
    for i in range(n):
        t += rng.expovariate(1 / mean_gap_ms)
        long = rng.random() < long_frac
        k = rng.randint(10, 20) if long else rng.randint(2, 4)
        steps = [(rng.uniform(50, 150), rng.uniform(*tool_ms)) for _ in range(k)]
        steps[-1] = (steps[-1][0], 0.0)
        out.append(Program(f"p{i:03d}", round(t, 3), steps, ctx_tokens=rng.randint(2000, 16000)))
    return out
