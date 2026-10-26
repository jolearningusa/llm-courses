# Lab 0: watch a harness work

**Lecture:** [master deck](../../../slides/00_master_agentos_deck.html), Module 1 · **Week:** 1 ·
**Autograder:** `make grade LAB=0` · **Budget:** API only; about 150 short model calls

Before you build any part of an AgentOS, watch two production harnesses do real work and find the nine
components of the master deck's reference architecture (slide 5) in what they log. At the same time, set up
the lab kit and write the trace logger that every later lab uses to report its numbers.

This lab follows the first assignment of Harvard CS2680, which asks students to watch and analyze Claude Code
before writing their own loop (see [`LAB_PLAN.md`](../../LAB_PLAN.md), section 2).

## Part A: set up the kit (15 minutes)

You need Python 3.10 or newer. The kit has no dependencies.

```bash
cd llm-agentos-course/labs/kit
make test-kit      # the kit's own tests: all should pass
make demo          # runs the 10 basic tasks with scripted solutions, offline
make grade LAB=0   # your Lab 0 tests: they fail until you finish Part B
```

Then point the kit at a model with tool calling. Any OpenAI-compatible endpoint works: a hosted API or a
local vLLM server (Lab 5 sets one up).

```bash
export NANOOS_BASE_URL=...   NANOOS_API_KEY=...   NANOOS_MODEL=...
python run_suite.py --model api --tasks b01 --out runs/smoke
```

## Part B: build the trace logger (graded)

Open `nanoos/trace.py` and finish the three TODOs:

1. `JsonlTracer.event`: write one JSON record per event with `seq`, `ts`, `run_id`, `kind` and the event's
   fields, and flush after every line.
2. `summarize_trace`: totals for one run (model calls, tool calls and errors, token sums, tools used, wall time).
3. `token_breakdown`: how many tokens of one prompt go to the system prompt, the tool schemas, tool results and
   the rest of the history.

`make grade LAB=0` runs 10 tests against the scripted model. The docstrings are the specification.

## Part C: measure (report)

1. **Your baseline.** Run the basic tier three times with the provided baseline loop:

   ```bash
   python run_suite.py --model api --trials 3 --out runs/lab0-baseline
   ```

   Record success, pass^3, steps per task and tokens per task from `runs/lab0-baseline/summary.json`. For the
   run that used the most tokens, call `token_breakdown` on its final messages
   (`runs/lab0-baseline/messages/<run>.json`, with `WorkspaceTools(...).schemas()` as the tools) and report
   where the tokens went.

2. **Two production harnesses.** Pick five basic tasks. Run each in Hermes Agent and in one of Codex CLI,
   OpenClaw or Claude Code, with the same model where the harness allows it:

   ```bash
   python export_task.py b07_two_hop /tmp/lab0/b07      # writes the files and prints the prompt
   # start the harness in /tmp/lab0/b07, paste the prompt, let it finish, then:
   python export_task.py b07_two_hop /tmp/lab0/b07 --check "<the harness's final answer>"
   ```

   Turn on each harness's own logging first. The observability deck
   ([slides](../../../slides/06_observability_evaluation_deck.html), the Hermes and OpenClaw slides) shows where each
   one writes traces; Codex CLI and Claude Code keep session transcripts as JSONL files in their home directories.

3. **Label one trace.** For one task in one harness, go through its log and label every event with the
   component that produced it: loop, scheduler, context manager, memory, tools and sandbox, skills, multi-agent,
   access control, or extension layer. Note which components you could not see at all.

4. **Compare.** For the same task: steps, input tokens and wall time in your baseline vs. both harnesses, and
   how many tokens each harness spends before the first user message (system prompt plus tool definitions).

5. **Slide 19.** Answer the master deck's exercise: which hooks along one turn can change the prompt prefix,
   and so the KV-cache hit rate? Use what you saw in the logs.

Hand in one page from [`report_template.md`](../report_template.md) plus the labeled trace (a table or an
annotated excerpt).

## Grading

| Part | Weight |
|------|-------:|
| Autograder (`make grade LAB=0`) | 50% |
| Report: baseline numbers, harness comparison, labeled trace, slide-19 answer | 40% |
| Code quality of `trace.py` | 10% |

## Stretch

- Write a converter from one harness's session log to the kit's JSONL format, so `summarize_trace` works on it.
- Run the basic tier on two models and compare pass^3 and cost per solved task.
