# Lab 9: multi-agent systems

**Lecture:** [multi-agent systems](../../../slides/05_multi_agent_deck.html), Module 5 · **Week:** 9 ·
**Autograder:** `make grade LAB=9` · **Budget:** Part A is free. Part B needs 2 tasks × 5 trials × 2 architectures,
plus a run with a cheaper worker model. The single-agent survey is the expensive run, because it reads 40 long
files in one growing context. Part C needs Hermes Agent and OpenClaw.

A sub-agent is a process fork for agents. The child starts from a fresh context, works on one brief, and only
its final answer comes back. Anthropic reports that a multi-agent research system beat a single agent by 90%,
at about 15 times the tokens of a chat. Cognition argues against multi-agents, because workers that cannot see
each other's decisions make conflicting ones. This lab measures both sides on tasks where you know the answer.

## Part A: a delegate tool (graded)

`nanoos/delegate.py` gives a parent two tools: `delegate(brief)` and `delegate_many(briefs)`. Implement the two
TODOs:

1. **`Delegator.spawn`.** Run one child loop and return its summary.
   - The child sees only its brief, never the parent's history.
   - It gets the tools you name, and nothing else.
   - It can delegate further only while it is under `max_depth`.
   - Enforce the limits: depth, and a children budget for the whole tree.
   - A child that raises, runs out of steps or hangs past `child_timeout_s` comes back as an error summary.
     The parent keeps going instead of crashing or waiting forever.
   - Long answers are cut to `max_summary_chars`.
2. **`Delegator.delegate_many`.** Run several children in parallel, at most `max_concurrency` at a time. Return
   their answers in the order of the briefs, not the order they finished.

`RestrictedTools`, the tool schemas, argument checks and `ChildRecord` bookkeeping are provided. The limits map
onto the harnesses' settings:

| Here | OpenClaw | Hermes Agent |
|------|----------|--------------|
| `max_depth` | `maxSpawnDepth` (default 1) | `max_spawn_depth` |
| `max_children` | `maxChildrenPerAgent` (default 5) | |
| `max_concurrency` | subagent lane `maxConcurrent` (default 8) | batch size of `delegate_task` |
| `child_timeout_s` | `runTimeoutSeconds` | |

`make grade LAB=9` runs 9 tests. One of them runs six children with a concurrency cap of three and checks the
wall-clock time.

## Part B: when do workers pay? (report)

`tasks/research.py` has two research tasks with answer keys:

- **`r1_vendor_survey`.** Five vendors, eight long documents each, and two facts per vendor buried in them. The
  five sub-questions are independent.
- **`r2_dependency`.** A three-hop question: job, then service, then team, then lead. Each hop needs the answer
  of the one before.

Offline, the scripted solutions already show the trade-off:

```bash
python multiagent_eval.py --arch single --latency-s 0.02 --out runs/lab9-demo-single
python multiagent_eval.py --arch orchestrator --latency-s 0.02 --out runs/lab9-demo-orch
```

On the survey, workers cut the total tokens about four times, cut the parent's final context from about 16,000
tokens to under 1,000, and cut the wall-clock time. On the dependency chain, the orchestrator spends about three
times the tokens of one agent and is slower. Explain both results from the traces.

Then run your model through your Lab 1 loop, 5 trials each:

```bash
python multiagent_eval.py --arch single --model api --loop nanoos --trials 5 --out runs/lab9-single
python multiagent_eval.py --arch orchestrator --model api --loop nanoos --trials 5 --out runs/lab9-orch
python multiagent_eval.py --arch orchestrator --model api --worker-model <a cheaper model> --loop nanoos \
    --trials 5 --out runs/lab9-orch-cheap
```

Report for each architecture and task:

- **Accuracy and pass^5.**
- **Total tokens** (parent plus workers) **and cost.** Use your provider's prices; workers on the cheaper model
  are priced separately.
- **Wall-clock time** and the parent's context at the end.
- **Workers and worker errors.** Did your model write self-contained briefs? Find one brief that left out a
  fact the worker needed.
- **MAST labels.** Label every failed trace with a failure mode from the deck's MAST taxonomy (system design,
  inter-agent misalignment, task verification). Report the most common category per architecture.

## Part C: the same in real harnesses (report)

Write the survey's files into a folder (`tasks/research.py`'s `TASKS[0].files`), then run the survey 5 times in
each of:

- one Hermes agent;
- Hermes with a `delegate_task` batch of workers;
- OpenClaw with `sessions_spawn` workers on a cheaper model.

Report the same measures as in Part B. Read each harness's settings before you run, because defaults differ:
OpenClaw children get `AGENTS.md` and `TOOLS.md` but not `MEMORY.md`, and leaf workers lose the spawn tools.
Did any harness return more than the final summary to the parent?

Hand in one page from [`report_template.md`](../report_template.md).

## Grading

| Part | Weight |
|------|-------:|
| Autograder (`make grade LAB=9`) | 50% |
| Report: the two tasks per architecture, cost with a cheaper worker, MAST labels | 30% |
| Report: the harness runs | 10% |
| Code quality of `delegate.py` | 10% |

## Stretch

- **Fork instead of fresh.** Add a `context="fork"` option that gives the child a copy of the parent's history,
  as OpenClaw can. Measure tokens and accuracy on both tasks against fresh children.
- **A verifier.** Add a worker that checks the others' answers against the files before the parent replies.
  Which MAST failures does it catch, and what does it cost?
- **τ²-bench.** Run one τ²-bench domain with a single agent and with an orchestrator, and compare pass^k.
