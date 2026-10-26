# AgentOS lab kit

The shared code for the course's practice labs, as planned in [`../LAB_PLAN.md`](../LAB_PLAN.md). Students grow
one small harness, **`nanoos`**, across the term; each lab adds one module, is graded by tests against a
scripted model, and then measures the same idea in a production harness or engine.

Python 3.10 or newer, no dependencies. Start with the [Lab 0 handout](handouts/lab00_watch_a_harness.md).

```bash
make test-kit      # the kit's own tests
make demo          # the 10 basic tasks with scripted solutions, offline
make grade LAB=0   # one lab's autograder
make suite TRIALS=3 NAME=baseline   # the basic tier on your model (set NANOOS_BASE_URL, NANOOS_API_KEY, NANOOS_MODEL)
```

## What is in the kit

| Path | What it is | Who writes it |
|------|------------|---------------|
| `nanoos/types.py` | Messages (OpenAI chat format), `ToolCall`, `ModelResponse`, `Usage`, `RunResult` | Provided |
| `nanoos/tools_basic.py` | Four file tools confined to a workspace: `list_files`, `read_file`, `write_file`, `search` | Provided; replaced in Lab 7 |
| `nanoos/baseline.py` | The smallest loop that can finish a basic task | Provided; Lab 1 writes `nanoos/loop.py` |
| `nanoos/trace.py` | JSONL trace logger, `summarize_trace`, `token_breakdown` | **Lab 0** |
| `models/scripted.py` | Replays canned assistant turns (tool calls, errors, delays) and records what it was sent | Provided |
| `models/openai_compat.py` | Client for any OpenAI-compatible endpoint (hosted API or vLLM), standard library only | Provided |
| `tasks/` | The task suite: prompt, fixture files, checker and a scripted solution per task | Provided |
| `run_suite.py` | Runs a tier k times per task; writes `metrics.csv`, `summary.json`, traces and messages | Provided |
| `export_task.py` | Writes one task's files to a folder (and scores an answer) to run it in another harness | Provided |
| `grade.py`, `tests/` | The autograder: `tests/test_labNN.py` per lab, `tests/test_kit.py` for the kit | Provided |
| `report_template.md` | The one-page report every lab hands in | Provided |
| `handouts/` | One handout per lab | Provided |

Later labs add `nanoos/loop.py` (Lab 1), `sched_sim.py` (2), `context.py` (3), `memory.py` (4), `prompt.py` (5),
`policy.py` (6), `tools.py` and `sandbox.py` (7), `skills.py` (8), `delegate.py` (9), `plugins.py` (10) and
`mcp_client.py` (11), each with its tests and handout.

## How grading works

- **Autograder.** `python grade.py N` runs `tests/test_labNN.py`. Tests only use the scripted model, so they are
  free, offline and give the same result on every machine; they check the harness's behavior, not the model's.
  Tests that hit an unfinished `TODO` are listed as `TODO`.
- **Report.** Each lab's measure part ends in one page from `report_template.md`, with numbers from `run_suite.py`.
- **Check-offs.** Short oral check-offs on a few random labs, as in MIT's 6.1810 (see the plan's section 3.5).

Reference solutions are not in this repository. Instructors keep them privately.

## Metrics

`run_suite.py` writes the same columns in every lab, so reports can compare runs:

`task, trial, success, steps, stopped_by, input_tokens, cached_input_tokens, output_tokens, cost_usd, wall_s,
tool_errors, approvals`

`summary.json` adds the success rate, **pass@k** (at least one of k trials succeeds) and **pass^k** (all k
succeed), both estimated per task from n trials and averaged over tasks. Pass `--price-in`, `--price-cached`
and `--price-out` (USD per million tokens) to fill in the cost column.

## Lab names

The plan numbers labs by lecture deck, 0 to 14. The three labs that already exist keep their folder names for
now (`../lab1_memory_plugins/`, `../lab2_aios_kv_cache/`, `../lab3_google_harness2/`); in the plan they are
Labs 12, 5 and 14. New handouts use two-digit numbers (`lab00_...`) to keep the two apart.
