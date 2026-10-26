# Lab 2: scheduling whole agent programs

**Lecture:** [the scheduler](../../../slides/01_scheduler_deck.html), Module 1 · **Week:** 2–3 ·
**Autograder:** `make grade LAB=2` · **Budget:** CPU only for parts A and B; one GPU for part C (can be done
together with Lab 5)

An engine like vLLM schedules LLM calls. An agent is a program of many calls with tool calls in between,
and what users wait for is the whole program. In this lab you write a small discrete-event simulator, use it
to compare a per-call policy with two program-aware ones (in the spirit of Continuum and Autellix from the
deck), and then check one result on a real vLLM server.

## Part A: build the simulator (graded)

Open `nanoos/sched_sim.py`. The module docstring defines the model: programs, steps, GPU slots, KV slots,
re-prefill and how ties are broken. Implement:

1. The three policy keys: `fcfs_per_call`, `program_fcfs`, `least_attained`.
2. `percentile` (nearest rank) and `jain` (Jain's fairness index).
3. `Simulator.keeps_kv` and `Simulator.run`.

`make grade LAB=2` runs 16 tests. Most compare job completion times with small workloads worked out by hand in
the test comments; read them when a test fails, and work the case out on paper before changing code.

## Part B: the deck's exercise (report)

`sched_experiment.py` runs all three policies on random workloads and averages over seeds:

```bash
python sched_experiment.py                                  # 8 GPU slots, 32 KV slots, light load
python sched_experiment.py --gpu-slots 2 --kv-slots 4 --gap-ms 60          # overloaded, few KV slots
python sched_experiment.py --gpu-slots 2 --kv-slots 32 --gap-ms 60 --tool-ms 500 3000 --kv-ttl 1000
python sched_experiment.py --gpu-slots 2 --kv-slots 4 --gap-ms 60 --sweep-long-frac 0 0.1 0.2 0.4 --csv runs/lab2/sweep.csv
```

Answer the deck's three questions with numbers and one plot:

1. How do mean and P99 job completion time change per policy as load grows (`--gap-ms`) and as the share of
   long programs grows (`--sweep-long-frac`)?
2. What does a KV time-to-live buy when tool calls are short vs. long? Compare `--kv-ttl` values against
   `--tool-ms` ranges, and look at the re-prefill column.
3. Which policy is fairest by Jain's index, and is the fairest policy also the one with the best P99?

Explain one result in terms of the mechanism. For example: why does per-call FCFS cause so many re-prefills when
KV slots are scarce?

## Part C: one real engine (report)

Serve a model with vLLM's priority scheduler (Lab 5 sets up the server; one GPU is enough) and replay a
simulated workload twice: once with no priorities, once with each program's arrival rank as its `priority`:

```bash
vllm serve <model> --scheduling-policy priority --max-num-seqs 8
python sched_replay.py --base-url http://localhost:8000/v1 --model <model> --priority none    --csv runs/lab2/none.csv
python sched_replay.py --base-url http://localhost:8000/v1 --model <model> --priority program --csv runs/lab2/program.csv
```

Check the `priority` field and `--scheduling-policy` in the vLLM docs for your version before you run this.
Compare mean and P99 job completion time and Jain's index, and say whether the direction matches the simulator.
If not, name one thing the simulator leaves out (prefix caching, batching, chunked prefill, ...).

Hand in one page from [`report_template.md`](../report_template.md).

## Grading

| Part | Weight |
|------|-------:|
| Autograder (`make grade LAB=2`) | 50% |
| Report: the three questions with a plot, and the vLLM comparison | 40% |
| Code quality of `sched_sim.py` | 10% |

## Stretch

- Add a fourth policy that uses what an agent harness knows: an estimate of remaining steps (shortest
  remaining program first), or KVFlow's idea of keeping KV for the program whose next call is soonest.
- Make KV eviction smarter than "first come keeps the slot": evict the program with the longest expected tool wait.
- Add preemption: let a long call be paused when a short program arrives, and charge a re-prefill for it.
