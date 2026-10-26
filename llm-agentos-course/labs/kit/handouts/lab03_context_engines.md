# Lab 3: context engines

**Lecture:** [the context manager](../../../slides/01_context_manager_deck.html), Module 1 · **Week:** 3 ·
**Autograder:** `make grade LAB=3` · **Budget:** API only; the long tier is the most token-hungry part of the
course so far (roughly 0.5–1M input tokens per engine for three trials; check your price first)

Long tasks fill the window. In this lab you build the deck's two "compress" strategies as context engines,
then run 40-step tasks with each and measure what they save and what they lose. CMU 11-768's first assignment
asks for compaction at a token threshold, and Harvard CS2680 grades an agent token-optimization assignment;
this lab does both and adds a recall check (see [`LAB_PLAN.md`](../../LAB_PLAN.md), section 2).

## Part A: build two engines (graded)

A context engine sits between the loop and the model (`ContextManagedModel` in `nanoos/context.py`), so it
works with the baseline and with your Lab 1 loop. Implement, following the class docstrings:

1. **`ThresholdCompaction`**: when the prompt passes a token budget, summarize the middle of the history
   (everything after the system prompt and task, up to the last `keep_last` messages) into one summary
   message. Never separate a tool result from the call that produced it. Reuse the summary until the budget is
   hit again, roll the old summary into the next one, and carry `[pin]` lines verbatim.
2. **`ToolResultClearing`**: the lighter cousin. Keep every message, but replace the content of old tool
   results with a short placeholder.

`make grade LAB=3` runs 12 tests, including a run of the long tier with each engine on scripted solutions.

## Part B: measure on the long tier (report)

The long tier (`tasks/long.py`) has three tasks of 40+ tool calls each, with only the file tools (no `search`):
summing 40 files, a 40-file log report whose house rule sits in a README read at the start, and a 45-file
pointer chain. Run each engine three times:

```bash
for c in none threshold clearing; do
  python run_suite.py --model api --tier long --trials 3 --max-steps 60 --loop nanoos \
      --context $c --context-budget 6000 --out runs/lab3-$c
done
```

With `--context threshold` and `--model api`, the summarizer is the same model (`ModelSummarizer`); its tokens
are added to the run's totals.

Report, per engine:

- Tokens and cost per task, and the cached share of input tokens if your endpoint reports it (a vLLM server
  with prefix caching does; see Lab 5).
- Compactions per task and peak context size (`compactions`, `peak_context_tokens` in `metrics.csv`).
- Task success and pass^3.
- **Recall after compaction.** In `l02_error_report`, was the README's rule (first line HERON) still followed
  after the README had been compacted away? Count it over the three trials.
- **One lost fact.** Find one run where an engine lost something the agent needed later, and quote the
  trace or the final messages (`runs/.../messages/`) to show it.

Then answer: tool-result clearing rewrites old messages on every call, while threshold compaction keeps the
summary byte-identical between compactions. What does each do to a prompt-prefix cache? Use your cached-token
numbers if you have them; Lab 5 measures this directly.

Hand in one page from [`report_template.md`](../report_template.md).

## Grading

| Part | Weight |
|------|-------:|
| Autograder (`make grade LAB=3`) | 50% |
| Report: the per-engine table, the recall count, one lost fact, the cache question | 40% |
| Code quality of `context.py` | 10% |

## Stretch

- **Todo recitation** (a technique Manus describes in its context-engineering notes): keep a short plan at the end of the context and update it
  every few steps, so the goal stays in the model's recent attention.
- **Hierarchical summaries with an expand tool**: keep per-chunk summaries and give the model an `expand(chunk)`
  tool that brings one chunk back.
- **Port an engine to Hermes or OpenClaw**: wrap your engine in Hermes's `ContextEngine` interface or
  OpenClaw's context-engine slot (today's `../../lab1_memory_plugins/` has templates; this is Lab 12 in the plan).
