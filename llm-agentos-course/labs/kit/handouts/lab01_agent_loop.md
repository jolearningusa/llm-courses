# Lab 1: a minimal agent loop that survives real conditions

**Lecture:** [the agent loop](../../../slides/01_agent_loop_deck.html), Module 1 · **Week:** 1–2 ·
**Autograder:** `make grade LAB=1` · **Budget:** API only; about 300 short model calls

Every later lab plugs into the loop you write here. The provided `nanoos/baseline.py` finishes easy tasks, but
it has no budget warning, runs tools one by one, crashes on the first rate-limit error, cannot be interrupted
and repeats itself forever. You will fix all of that in `nanoos/loop.py`, keeping the same `run()` signature so
the task suite can switch loops with `--loop nanoos`.

The deck's exercise slide ("Write a minimal agent loop") is the starting point. CMU 11-768 and Stanford CS329Z
start their courses with the same assignment: a harness with no framework (see [`LAB_PLAN.md`](../../LAB_PLAN.md),
section 2).

## Part A: build the loop (graded)

Open `nanoos/loop.py`. The module docstring is the specification; it lists seven behaviors:

| # | Behavior | Deck slide it comes from |
|---|----------|--------------------------|
| 1 | Step budget, stop conditions and a one-time wrap-up message | Stop conditions and budgets |
| 2 | Parallel tool calls, results kept in call order | Running tools inside the loop |
| 3 | Retries with exponential backoff, then a fallback model | Errors, retries and fallback |
| 4 | Interrupting a slow model call | Interrupts and steering |
| 5 | A steering queue drained between model calls | Interrupts and steering |
| 6 | A nudge when the model repeats the same call | What goes wrong in agent loops |
| 7 | `pre_tool_call` (can block) and `post_tool_call` (can rewrite) hooks | Hooks along the loop |

Implement `call_model`, `run_tools` and `run`. `make grade LAB=1` runs 19 tests against the scripted model,
which can raise errors (`{"raise": "429"}`) and stall (`{"delay_s": 2}`) on cue. Lab 0's tracer is used if you
finished it; nothing here depends on it.

## Part B: measure (report)

1. **Same tasks, two loops.** Run the basic tier three times with each loop on the same model:

   ```bash
   python run_suite.py --model api --trials 3 --loop baseline --out runs/lab1-baseline
   python run_suite.py --model api --trials 3 --loop nanoos   --out runs/lab1-nanoos
   ```

   Compare success, pass^3, steps per task, tokens per task and wall time. Parallel tool calls should show up
   in wall time on tasks where the model asks for several files at once (b09).

2. **Under failure.** Real APIs fail. Inject failures and measure how each loop degrades:

   ```bash
   for r in 0.1 0.2 0.3; do
     python run_suite.py --model api --trials 3 --loop baseline --fail-rate $r --out runs/lab1-base-f$r
     python run_suite.py --model api --trials 3 --loop nanoos   --fail-rate $r --out runs/lab1-nanoos-f$r
   done
   ```

   Plot pass^3 against the failure rate for both loops. (Offline, `--model demo` gives the same experiment
   with scripted solutions, which is a quick way to check your retry logic.)

3. **Against a production loop.** Run five basic tasks in Hermes Agent on the same model
   (`python export_task.py <task> <dir>`, as in Lab 0) and compare steps and tokens with your loop. Find, in
   Hermes's documentation or source, how it handles one of: the iteration budget, interrupts, or retries, and
   say how it differs from yours.

4. **Read a tiny agent.** Read `mini-swe-agent` (about 100 lines for the agent class, bash as its only tool).
   Name one thing it does with fewer lines than your loop, and one thing your loop handles that it does not.

Hand in one page from [`report_template.md`](../report_template.md).

## Grading

| Part | Weight |
|------|-------:|
| Autograder (`make grade LAB=1`) | 50% |
| Report: two loops, failure curve, Hermes comparison, mini-swe-agent note | 40% |
| Code quality of `loop.py` | 10% |

## Stretch

- A workflow mode: for one task, write the fixed sequence of steps as code (no model choosing tools), and
  compare cost and success with the agent mode. This is the deck's "workflows vs. agents" slide in numbers.
- Stream the model's reply and allow an interrupt mid-stream (needs `stream: true` in the OpenAI-compatible client).
- Emit one span per step in the format the observability deck describes; Lab 13 builds on it.
