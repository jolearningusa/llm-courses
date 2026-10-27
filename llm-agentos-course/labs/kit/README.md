# AgentOS lab kit

The shared code for the course's practice labs, as planned in [`../LAB_PLAN.md`](../LAB_PLAN.md). Students grow
one small harness, **`nanoos`**, across the term; each lab adds one module, is graded by tests against a
scripted model, and then measures the same idea in a production harness or engine.

Python 3.10 or newer, no dependencies. Handouts: [Lab 0](handouts/lab00_watch_a_harness.md), [Lab 1](handouts/lab01_agent_loop.md), [Lab 2](handouts/lab02_scheduler.md), [Lab 3](handouts/lab03_context_engines.md), [Lab 4](handouts/lab04_long_term_memory.md), [Lab 5](handouts/lab05_kv_cache_plane.md), [Lab 6](handouts/lab06_access_control.md), [Lab 7](handouts/lab07_tools_sandbox.md), [Lab 8](handouts/lab08_skills.md), [Lab 9](handouts/lab09_multi_agent.md), [Lab 10](handouts/lab10_extension_layer.md), [Lab 11](handouts/lab11_mcp_connectors.md).

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
| `nanoos/loop.py` | The agent loop: budgets, parallel tools, retries and fallback, interrupts, steering, hooks | **Lab 1** |
| `nanoos/context.py` | Context engines: threshold compaction and tool-result clearing, behind `ContextManagedModel` | **Lab 3** |
| `nanoos/memory.py` | Long-term memory: consolidation, a trust-gated write path, hybrid BM25 + vector search, auto-recall and a `search_memory` tool | **Lab 4** |
| `nanoos/prompt.py` | Prompt prefix stability: vLLM-style block hashes, a simulated prefix cache, cache-break diagnosis, `CacheAuditModel` | **Lab 5** |
| `nanoos/policy.py` | Access control: hash-chained audit log, sender pairing, a Rule of Two policy engine, `GuardedTools`, readers | **Lab 6** |
| `nanoos/tools.py` | A tool registry: schemas from type hints and docstrings, argument checks with actionable errors, truncation with a pointer to the full output | **Lab 7** |
| `nanoos/sandbox.py` | Run shell commands in a boundary: host, Linux namespaces or Docker, with timeouts, limits and a clean environment; a `run_shell` tool | **Lab 7** |
| `nanoos/skills.py` | Skills: validation, discovery across roots, a budgeted index, `skill_view` for bodies and files on demand | **Lab 8** |
| `nanoos/delegate.py` | Sub-agents: `delegate` and `delegate_many` with fresh child contexts, restricted tools, depth, budget, concurrency and timeout limits | **Lab 9** |
| `nanoos/plugins.py` | Plugins: manifests with config schemas, an enable list, exclusive slots, ordered hooks (decide, transform, observe), `HookedTools` | **Lab 10** |
| `nanoos/mcp_client.py` | MCP 2026-07-28: stdio and Streamable HTTP transports, `_meta` on every request, version fallback, `input_required` round trips, `McpTools` with filters and pins | **Lab 11** |
| `nanoos/sched_sim.py` | Discrete-event simulator: per-call vs. program-aware policies, KV kept across tool calls | **Lab 2** |
| `models/scripted.py` | Replays canned assistant turns (tool calls, errors, delays) and records what it was sent | Provided |
| `models/gullible.py` | A scripted model that obeys every injection it reads, to test defenses offline | Provided |
| `models/flaky.py` | Wraps a model so a share of its calls fail (`run_suite.py --fail-rate`) | Provided |
| `models/openai_compat.py` | Client for any OpenAI-compatible endpoint (hosted API or vLLM), standard library only | Provided |
| `tasks/` | The task suite: prompt, fixture files, checker and a scripted solution per task (tiers `basic`, `long`), plus the Lab 4 memory fixtures, the Lab 5 session, the Lab 6 red-team office, the Lab 7 ticket tracker, the Lab 8 skills, the Lab 9 research tasks, the Lab 10 shell tasks and the Lab 11 MCP notes server | Provided |
| `run_suite.py` | Runs a tier k times per task; writes `metrics.csv`, `summary.json`, traces and messages | Provided |
| `sched_experiment.py`, `sched_replay.py` | Lab 2's policy comparison on simulated workloads, and a replay against vLLM | Provided |
| `memory_eval.py` | Lab 4's exercise: one store, read paths none / auto / tool / both, on the fixtures in `tasks/memory.py` | Provided |
| `cache_experiment.py` | Lab 5's exercise: the cache-breaking variants of a 20-turn session, simulated or on a vLLM server | Provided |
| `redteam_eval.py` | Lab 6's exercise: the red-team tasks in `tasks/redteam/` under each defense | Provided |
| `tools_eval.py` | Lab 7's exercise: five ticket questions with a vague search tool (v1) or a designed one (v2), shell in a sandbox | Provided |
| `skills_eval.py` | Lab 8's exercise: incident reports with no skills, one skill or 50 installed; pass@1, pass^k, activation | Provided |
| `multiagent_eval.py` | Lab 9's exercise: two research tasks as one agent or as an orchestrator with workers | Provided |
| `plugins/` | Example plugins for Lab 10: a shell guard, a secret redactor, and the Lab 3 engines behind the context-engine slot | Provided |
| `plugins_eval.py` | Lab 10's exercise: the shell guard on five tasks over a simulated shell, with crash and config experiments | Provided |
| `mcp_eval.py` | Lab 11's exercise: five tasks against the notes server over stdio or HTTP, with filters, pins and confirmation policies | Provided |
| `export_task.py` | Writes one task's files to a folder (and scores an answer) to run it in another harness | Provided |
| `grade.py`, `tests/` | The autograder: `tests/test_labNN.py` per lab, `tests/test_kit.py` for the kit | Provided |
| `report_template.md` | The one-page report every lab hands in | Provided |
| `handouts/` | One handout per lab | Provided |

Lab 12 will rework `../lab1_memory_plugins/` to wrap the engines you built in Labs 3 and 4.

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
tool_errors, approvals, compactions, peak_context_tokens`

`summary.json` adds the success rate, **pass@k** (at least one of k trials succeeds) and **pass^k** (all k
succeed), both estimated per task from n trials and averaged over tasks. Pass `--price-in`, `--price-cached`
and `--price-out` (USD per million tokens) to fill in the cost column.

## Lab names

The plan numbers labs by lecture deck, 0 to 14. The three labs that already exist keep their folder names for
now (`../lab1_memory_plugins/`, `../lab2_aios_kv_cache/`, `../lab3_google_harness2/`); in the plan they are
Labs 12, 5 and 14. New handouts use two-digit numbers (`lab00_...`) to keep the two apart.
