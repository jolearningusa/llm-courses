# Lab plan: one practice lab per lecture

This is the plan for the course's hands-on work. It gives every lecture deck a lab, explains how the labs fit
together, and records what similar courses and tutorials do, so the design choices can be checked against them.
The three labs that exist today (`lab1_memory_plugins/`, `lab2_aios_kv_cache/`, `lab3_google_harness2/`) are
kept and slotted into the plan; the rest is new work, listed as PRs at the end.

Status: **plan**, written 2026-10-26. Built so far: the lab kit and Labs 0–14 ([`kit/`](kit/)), plus today's
Labs 1–3. The rest is not written yet.

---

## 1. Summary

- **One codebase that grows.** Students build a small harness, `nanoos`, across the term: a loop in week 1,
  then a scheduler simulator, a context engine, memory, a policy engine, tools, skills, sub-agents, plugins,
  an MCP client and tracing. This is how MIT's xv6 labs and the `learn-claude-code` tutorial work, and it is
  what Harvard CS2680, CMU 11-768, Stanford CS329Z and Michigan EECS 498 ask for in their first assignments.
- **Build it small, then measure it in a real system.** Each lab has a *build* part in `nanoos` (graded by
  tests against a scripted model, so grading is free and deterministic) and a *measure* part in a production
  harness or engine (Hermes Agent, OpenClaw, Letta, Codex CLI, vLLM). The measure part ends in a short report
  with numbers, because "is it better?" is the question this course keeps asking.
- **The lecture exercise becomes the lab.** Every lecture deck already ends with an exercise slide. The labs
  below keep those exercises and add what a lab needs: starter code, a task set, an autograder, a report
  template and a compute budget.
- **Evaluation is a thread, not a week.** Tracing and the task-suite runner arrive in Lab 0 so that every lab
  reports the same metrics (success, pass^k, tokens, cost, latency, cache hit rate). Lab 13 then goes deep.
- **A capstone bake-off.** Teams improve their harness on a held-out task suite under a fixed token budget,
  scored on success, cost and a safety suite, inspired by the leaderboards of the Hugging Face Agents course
  and Berkeley's AgentBeats competition.

| Lab | Lecture deck | Build in `nanoos` | Measure in a real system |
|----:|--------------|-------------------|--------------------------|
| 0 | Master deck (1–21) | Task runner + trace logger | Watch two production harnesses; map events to the nine components |
| 1 | Agent loop | Loop with budgets, parallel tools, retries, interrupts, hooks | Same tasks in Hermes; steps and cost per task |
| 2 | Scheduler | Discrete-event simulator for program-aware policies | vLLM priority scheduling on the Lab 5 server |
| 3 | Context manager | Two context engines behind one interface | 40+ step tasks; recall after compaction |
| 4 | Long-term memory | Memory store with a write path and two read paths | Memory QA set; injection vs. search tool |
| 5 | KV-cache plane (includes existing Lab 2) | Cache-stable prompt assembly, prefix-hash checker | "Break the cache on purpose" on vLLM + LMCache |
| 6 | Access control and safety | Policy engine: allowlists, Rule of Two, approvals | Red-team own agent; AgentDojo; `openclaw security audit` |
| 7 | Tools and sandbox | Tool registry, result truncation, Docker sandbox runner | Injection-in-data test with and without sandbox |
| 8 | Skills | `SKILL.md` loader with progressive disclosure | Write, measure and harden one skill in Hermes and OpenClaw |
| 9 | Multi-agent systems | `delegate` tool with isolated child context | Same task, three architectures; MAST labels |
| 10 | Extension layer | Plugin manifest, discovery, hooks and slots | One feature as Hermes plugin, OpenClaw plugin and MCP server |
| 11 | MCP, plugins and connectors | MCP client (stdio + Streamable HTTP) | Build, mount, attack and harden one MCP server |
| 12 | Harness deep-dives (22–31; existing Lab 1) | Port the Lab 3/4 engines to a plugin interface | Hermes `MemoryProvider` / `ContextEngine`, OpenClaw slot, Letta shared blocks |
| 13 | Observability and evaluation | OpenTelemetry spans, graders, LLM judge | Dashboard comparing two harness configurations |
| 14 | Harness² (Module 7; existing Lab 3) | Propose → probe → compose loop over `nanoos` config | Harness² demo and one LAB task |
| — | Capstone | Team harness | Held-out suite, fixed budget, safety score |

---

## 2. What similar courses and tutorials do

The survey was done on 2026-10-26. University courses are listed with what their **assignments** ask students to
build, since that is what this plan has to decide; lecture topics overlap heavily and are not repeated here.

### University courses

| Course | Hands-on work | What we take from it |
|--------|---------------|----------------------|
| **Harvard CS2680, Modern AI Systems: Agents and System Optimizations** (Fall 2026) | Five assignments that walk down the stack: *Mads-Lens* (watch and analyze Claude Code working), *Mads-Loop* (build your own agent loop), *Mads-Opt* (optimize agent token usage), *Mads-Serve* (build and run a serving engine on course GPUs), *Mads-Stack* (optimize the full stack), then a poster project. | The closest match to this course: agents and the serving engine in one term. Lab 0 copies the "watch a real harness first" assignment; Labs 3 and 5 copy the token- and cache-optimization assignments. |
| **CMU 11-768, AI Agents** (Fall 2026, Neubig and Fried) | Assignment 1 *Harness*: a ReAct loop, parallel tool execution with error recovery, a skills system that loads `SKILL.md` on demand, compaction at 6,000 tokens, and a controlled experiment comparing tool-output designs in Modal sandboxes. Assignment 2 *Evaluation*, Assignment 3 *Training* (RL), and a team research project worth 50%. | A first assignment that builds a whole minimal harness, and "a controlled experiment" as a required part of a build assignment. |
| **Stanford CS329Z, Engineering AI Agents** (Fall 2026, Yang, Ryan, J. Yang) | HW1 *Build an Agentic Harness*: an internal company assistant with no agent frameworks, adding tools, memory, a terminal and human-in-the-loop controls. HW2 *Evaluate an Agent*: code-based graders, an LLM judge and benchmark tasks for a pre-built agent. Two closed-book quizzes on design trade-offs; a group project worth 50%. | "No frameworks" for the build part, and a separate evaluation assignment on an agent the students did not write (Lab 13 grades a provided harness as well as their own). Quizzes on trade-offs rather than recall. |
| **Stanford CS329A, Self-Improving AI Agents** (Autumn 2025) | Three homeworks on self-improvement, multi-step reasoning and tool use; a research project with API credits, midterm presentation and poster. | Model for Lab 14 and the capstone: self-improvement measured against a fixed task set, with API credits budgeted per team. |
| **Michigan EECS 498-016, Applied Agentic Software Engineering** (Fall 2026) | Analyze phase: a private copy of Aider extended toward an agent in four parts (skill loader, model-driven tool loop, shell tool with approval rules, MCP client). Create phase: a project built on an existing harness. Three timed hackathons. | Extending a real open-source harness, not only a toy one. The four Aider parts line up with our Labs 1, 6/7, 8 and 11. |
| **MIT 6.1810, Operating System Engineering** (xv6 labs) | Eight labs on one teaching kernel (system calls, page tables, traps, copy-on-write, network driver, locks, file system, mmap). Each lab's score is the output of `make grade`; check-off meetings ask about five random labs per student. | The OS-course lab mechanics: one codebase, `make grade` autograding, and short oral check-offs so that a generated solution still has to be explained. |
| **UC Berkeley Agentic AI MOOC** (CS294/194-196, Fall 2025) | Guest lectures, quizzes, and a two-phase project (a "green" evaluator agent, then a "white" agent) tied to the AgentX–AgentBeats competition. | Students build an evaluator as well as an agent; a public leaderboard as motivation. |

### Self-paced courses and tutorials

| Course | Hands-on work | What we take from it |
|--------|---------------|----------------------|
| **Hugging Face Agents Course** | Units on agent basics, three frameworks (smolagents, LlamaIndex, LangGraph), agentic RAG, a final agent scored on a student leaderboard; bonus units on function-calling fine-tuning and on observability and evaluation. Hands-on parts run in pre-configured Spaces. | Pre-configured environments so setup does not eat a week; a leaderboard for the capstone. |
| **Hugging Face MCP Course** | An end-to-end MCP application, a deployed MCP server, a pull-request agent on the Hub. | A deployed, remote MCP server as a deliverable (Lab 11). |
| **Kaggle × Google 5-Day AI Agents Intensive** (Nov 2025, re-run June 2026) | Five days, each a whitepaper plus two ADK codelabs: agents, tools and MCP (with human-in-the-loop approval), context engineering (sessions and memory), agent quality (logging, tracing, metrics, evaluation), prototype to production (A2A, deployment); a capstone. | Pairing a reading with a short codelab; quality and tracing taught as a core day, not an extra. |
| **DeepLearning.AI, *LLMs as Operating Systems: Agent Memory*** (Letta) | Build an agent with self-editing memory from scratch, then a MemGPT agent in Letta with core and archival memory, then multi-agent shared memory blocks. | The same OS framing as this course; Lab 4 follows its "from scratch first, then Letta" order. |
| **DeepLearning.AI, *Evaluating AI Agents*** (Arize) | Five code labs: build an agent, trace it, add router and skill evaluations, add trajectory evaluations (a convergence score), structure experiments; plus judge calibration and monitoring. | The step-by-step evaluation ladder used in Lab 13, including a trajectory "convergence" metric. |
| **DeepLearning.AI, *Agent Skills with Anthropic*** | Build custom skills, use them through the Claude API, Claude Code and the Claude Agent SDK, including sub-agents equipped with skills. | Skills used across several runtimes (Lab 8 runs the same skill in Hermes and OpenClaw). |
| **DeepLearning.AI, *MCP: Build Rich-Context AI Apps with Anthropic*** | Build MCP servers and clients for a chatbot. | Background for Lab 11. |
| **Microsoft, *AI Agents for Beginners*** | Ten lessons on design patterns (tool use, agentic RAG, planning, multi-agent, metacognition, production), with framework code samples. | A pattern catalog to point students to; less useful as a lab model because it is framework-led. |
| **`shareAI-lab/learn-claude-code`** (MIT) | A nano Claude Code–like harness in Python built in 17 steps: loop, tool dispatch, permissions, hooks, todo planning, sub-agents, skill loading, compaction, memory, task graph, background tasks, cron, agent teams, MCP, integrated harness, workflow runtime, goal loop. | The best existing match for the `nanoos` build track; its step order confirms that our labs can each add one mechanism to a working harness. |
| **`SWE-agent/mini-swe-agent`** | An agent class of about 100 lines that has no tool other than bash; its README reports more than 74% on SWE-bench Verified. | Evidence that Lab 1's loop can stay tiny; a reference solution to compare against. |
| **`GeeeekExplorer/nano-vllm`** (MIT) | About 1,200 lines of Python that reimplement vLLM's paged KV cache, hash-based prefix caching, CUDA graphs and tensor parallelism (Qwen3 only). | A readable engine for Lab 5's stretch part, so students can change the prefix cache instead of only measuring it. |

### Benchmarks and tools the labs reuse

- **AgentDojo** (NeurIPS 2024 datasets track): a dynamic environment for prompt-injection attacks and defenses
  on tool-using agents. Used in Lab 6.
- **τ²-bench** (`sierra-research/tau2-bench`): customer-service tasks (airline, retail, telecom and more) with a
  simulated user; the τ-bench family introduced the pass^k reliability metric. Used in Labs 9 and 13.
- **Terminal-Bench 2.0** with the **Harbor** framework: containerized terminal tasks for coding agents. An
  optional capstone suite.
- **Inspect** (UK AI Security Institute): an open-source eval framework with tasks, solvers and scorers. An
  alternative runner for Lab 13.
- **`vllm bench serve`** and **`lmcache bench`**: load generators for Labs 2 and 5.
- Tracing backends already in the observability deck (Langfuse, Arize Phoenix, Jaeger, Prometheus + Grafana).

### What the survey says about lab design

1. **Everyone starts with "build your own loop."** Harvard, CMU, Stanford and Michigan all make the first real
   assignment a frameworkless harness. Nobody starts with a framework tutorial any more.
2. **The good courses make students measure.** CMU requires a controlled experiment in the build assignment;
   Harvard grades token optimization; DeepLearning.AI's evaluation course is entirely measurement. Our decks
   already end in "Measure" boxes, so every lab gets a report.
3. **Evaluation is its own assignment** at CMU and Stanford, and a core day at Kaggle. It also has to come
   early so that later labs can use it.
4. **Agents and serving are rarely taught together.** Only Harvard CS2680 covers both. That is this course's
   distinguishing feature, so Labs 2 and 5 keep a real GPU part rather than simulating everything.
5. **Projects carry most of the grade** (50% at CMU and Stanford). The labs here are sized so the capstone can
   still be large.
6. **Autograding plus a human check.** MIT grades with `make grade` and adds random check-offs; Stanford adds
   closed-book quizzes. With coding agents in every student's editor, the explanation is what shows understanding.

---

## 3. Design of the lab track

### 3.1 The lab kit

A new directory, `labs/kit/`, holds everything the labs share. Sketch:

```
labs/kit/
├── nanoos/                 # the student harness; each lab fills in one module
│   ├── loop.py             # Lab 1
│   ├── sched_sim.py        # Lab 2
│   ├── context.py          # Lab 3
│   ├── memory.py           # Lab 4
│   ├── prompt.py           # Lab 5: cache-stable prompt assembly
│   ├── policy.py           # Lab 6
│   ├── tools.py, sandbox.py# Lab 7
│   ├── skills.py           # Lab 8
│   ├── delegate.py         # Lab 9
│   ├── plugins.py          # Lab 10
│   ├── mcp_client.py       # Lab 11
│   └── trace.py            # Lab 0 (JSONL), Lab 13 (OpenTelemetry)
├── models/
│   ├── scripted.py         # replays a scripted model from YAML: free, deterministic, used by the autograder
│   └── openai_compat.py    # any OpenAI-compatible endpoint: a hosted API, or vLLM from Lab 5
├── tasks/                  # task suite: each task = prompt + fixture files + checker function
├── run_suite.py            # run N tasks x k trials, write traces and a metrics table
├── grade.py                # `make grade LAB=n`: unit tests against the scripted model
└── report_template.md      # the one-page report every lab hands in
```

Rules for the kit:

- **Python only for the build track**, so students do not split their time across languages. TypeScript appears
  only where a real harness requires it (OpenClaw parts of Labs 10 and 12), as in today's Lab 1B.
- **A scripted model for grading.** `models/scripted.py` returns canned assistant turns (including tool calls,
  malformed calls, long outputs and timeouts). Autograder tests check harness behavior, not model quality, so
  they cost nothing and do not flake.
- **One task suite, three tiers.** About 30 small tasks with programmatic checkers (file edits, lookups over
  fixture data, multi-hop questions), a "long" tier of 40+ step tasks for Labs 3 and 9, and a "held-out" tier
  kept private for the capstone.
- **One metrics table.** `run_suite.py` always writes the same columns: task, trial, success, steps, input
  tokens, cached input tokens, output tokens, cost, wall time, tool errors, approvals. Reports compare tables.
- **Reference solutions stay out of the public repo**, in a private instructors' branch or repository; the
  public kit ships stubs with `NotImplementedError` and docstrings.

### 3.2 Shape of every lab

Each lab README follows the same template:

1. **Goal** in two sentences, and the lecture deck it belongs to.
2. **Build** (core): what to add to `nanoos`, the interface to implement, and what `make grade LAB=n` checks.
3. **Measure**: the experiment, run on the task suite or in a real harness, with the variables to change.
4. **Report**: one page from `report_template.md`: setup, table, one plot, one failure you looked at, one
   sentence on what you would change.
5. **Stretch**: optional, for students who finish early or want capstone ideas.
6. **Budget**: expected tokens and cost with the suggested model, and whether a GPU is needed.
7. **Upstream pages**: the documentation each real-harness step was checked against, with the date (as the
   current lab files already do).

### 3.3 Models and compute

- **Default model:** a small, cheap hosted model with tool calling, set once in the kit's config. Students can
  switch to a local model served by vLLM from Lab 5 onward, which also makes cache metrics visible.
- **GPU labs:** Lab 2 (measure part) and Lab 5. One GPU with 24 GB or more is enough; Lab 5's NIXL part needs
  two GPUs or two nodes and stays a stretch goal, as in today's Lab 2.
- **Budget per student:** to be calibrated on a pilot run. Labs that repeat runs for pass^k (3, 8, 9, 13, 14,
  capstone) dominate; the plan is to cap each with a fixed number of trials and a published token ceiling.

### 3.4 Safety rules for the attack labs

Labs 6, 7, 8 and 11 plant prompt injections and poisoned tools. They run only against the student's own agent,
fake secrets and local fixtures, inside the course sandbox with network egress off by default. No lab attacks
a third-party service, and the red-team fixtures are kept in `labs/kit/tasks/redteam/` so they are easy to audit.

### 3.5 Grading (suggested)

| Part | Weight | Notes |
|------|-------:|-------|
| Labs 0–14 (best 13 of 15) | 45% | Per lab: 50% autograder, 40% report, 10% code quality |
| Check-offs | 10% | Three random labs per student, 10 minutes each, MIT style |
| Two short quizzes on design trade-offs | 10% | Closed book, Stanford CS329Z style |
| Capstone | 35% | Proposal 5, midpoint demo 5, bake-off score 10, report and poster 15 |

Coding agents are allowed in the labs (the course is about them). The report and the check-off must show the
student can explain each design choice and each number.

---

## 4. The labs

Week numbers assume a 14-week term with the seven modules taught in the README's order. Each lab is due one week
after its lecture unless noted.

### Lab 0. Watch a harness work (Master deck, week 1)

- **Goal:** see the nine AgentOS components in a real system before building any of them. Same idea as Harvard
  CS2680's *Mads-Lens*.
- **Build:** install the kit; finish the JSONL trace logger in `trace.py` (one record per model call and tool
  call), plus `summarize_trace` and `token_breakdown`. Handout: [`kit/handouts/lab00_watch_a_harness.md`](kit/handouts/lab00_watch_a_harness.md).
- **Measure:** run five suite tasks in two production harnesses (Hermes Agent and one of Codex CLI, OpenClaw or
  Claude Code) with their logging turned on. Label every event in one trace with the component that produced it
  (loop, scheduler, context, memory, tools, skills, multi-agent, access control, extension).
- **Report:** the labeled trace; a table of where each harness spends tokens (system prompt, tool definitions,
  history, tool results); the master deck's exercise from slide 19 (which hooks can change the prompt prefix).
- **Budget:** API only, small.

### Lab 1. A minimal agent loop (Agent loop deck, week 1–2)

- **Goal:** write the loop that every later lab plugs into.
- **Build:** start from the deck's exercise code. Add a step budget and stop conditions, parallel tool calls that
  keep results in call order, retries with backoff and a model fallback, an interruptible model call with a
  steering queue drained between steps, a repetition detector, and `pre_tool_call` / `post_tool_call` hooks.
- **Autograder:** scripted-model tests for each feature (budget hit, malformed tool call, tool timeout, a
  steering message arriving mid-turn, result ordering under parallel calls, hook order).
- **Measure:** run the basic tier with your loop and with Hermes on the same model; compare success, steps per
  task and tokens per task. Read `mini-swe-agent` and note one thing it does with fewer lines.
- **Stretch:** a workflow mode (fixed steps) next to the agent mode for one task, to make the "workflows vs.
  agents" slide concrete.

### Lab 2. Program-aware scheduling (Scheduler deck, week 2–3)

- **Goal:** show why scheduling whole agent programs beats scheduling single calls.
- **Build:** the deck's discrete-event simulator in `sched_sim.py`: programs as lists of (LLM ms, tool ms) steps,
  a GPU with slots, a KV pool with re-prefill cost, and three policies: per-call FCFS (vLLM's default),
  program FCFS (Continuum style) and least attained service (Autellix style). Add a KV time-to-live.
- **Autograder:** hand-computed job completion times on three tiny workloads; Jain's index implementation.
- **Measure:** mean and P99 job completion time and Jain's index per policy, while varying the share of long
  programs and tool latency. Then, on the Lab 5 vLLM server, pass a per-program priority with vLLM's priority
  scheduling and compare one real workload with and without it.
- **Budget:** CPU for the simulator; the vLLM part can be done together with Lab 5 if GPUs are scarce.

### Lab 3. Context engines (Context manager deck, week 3)

- **Goal:** keep long tasks inside the window without losing what matters.
- **Build:** a `ContextEngine` interface in `context.py` with two implementations: (a) threshold summary
  compaction, and (b) tool-result clearing plus todo recitation, or hierarchical summaries with an expand tool.
  Assemble prompts in a fixed order so the prefix stays stable (preparing for Lab 5).
- **Autograder:** the engine never drops the system prompt or the latest user turn; compaction triggers at the
  threshold; compacted history still holds every fact tagged "pinned" in the fixture.
- **Measure:** the long tier (40+ steps), three trials each: tokens and cost per task, cached-token share,
  number of compactions, a recall-after-compaction quiz, and pass^3. Find one run where an engine lost a fact
  it needed.
- **Borrowed from:** CMU 11-768 Assignment 1 (compaction at a token threshold), Harvard CS2680 *Mads-Opt*.

### Lab 4. Long-term memory (Long-term memory deck, week 4)

- **Goal:** build a memory system end to end and see that the read path matters as much as the store.
- **Build:** `memory.py` with a write path (extract facts at the end of a session and before compaction),
  consolidation (update or supersede, not only append), a hybrid index (BM25 + vector; LanceDB or SQLite with an
  embedding model), and two read paths: (A) automatic injection before each turn, (B) a `search_memory` tool.
- **Autograder:** update and delete rules on a scripted conversation; per-user scoping; no memory written from
  a tool result marked untrusted (preparing for Lab 6).
- **Measure:** the deck's exercise. 20 questions that need memory and 20 that do not, across multi-session
  fixtures (a small LongMemEval-style set): accuracy, tokens per turn, and how often the model calls the tool
  in path B. Then wire the same store into Letta (core vs. archival) and compare with path A.
- **Borrowed from:** DeepLearning.AI *LLMs as Operating Systems* (self-editing memory from scratch, then Letta).

### Lab 5. The KV-cache plane (KV-cache deck, week 5; contains today's Lab 2)

- **Goal:** connect harness decisions to cache hits, latency and cost.
- **Build:** `prompt.py` checks a session's prompts for prefix stability: it hashes each prompt in vLLM-sized
  blocks and reports the first block that changed between turns and why (timestamp, tool list, memory edit,
  compaction).
- **Measure:** the deck's "break the cache on purpose" exercise. Serve a model with vLLM (prefix caching on) using
  today's `lab2_aios_kv_cache/run_vllm_lmcache.sh`, point `nanoos` and then Hermes or OpenClaw at it, and run a
  scripted 20-turn session. Repeat with one change each: a timestamp in the system prompt, a tool added at turn
  10, a memory edit at turn 10, compaction at turn 15. Repeat the baseline with `--kv-cache-dtype fp8` and with
  LMCache CPU offload. Report hit rate and TTFT per turn, prefill tokens per session, the most concurrent sessions
  before eviction, and the projected API cost with the deck's cache-price multipliers.
- **Stretch:** in `nano-vllm`, add a prefix-cache hit counter per request, or a TTL that keeps a session's blocks
  during a tool call (the Continuum idea from Lab 2), and measure it; or the NIXL prefill/decode split from
  today's `vllm_lmcache_nixl_config.py` on two GPUs.
- **Budget:** one GPU (24 GB or more) for a 7–8B model.

### Lab 6. Access control and red-teaming (Access control deck, week 6)

- **Goal:** make the agent safe by design and measure what that costs in usefulness.
- **Build:** `policy.py`: tag each tool with the Rule of Two letters (untrusted input, sensitive data, external
  action), a session state that tracks which letters were reached, an approval step whenever a session would
  reach all three, sender and tool allowlists, and an append-only audit log.
- **Autograder:** scripted sessions that should and should not trigger approval; the audit log records every
  decision; denied tools never run.
- **Measure:** the deck's exercise. Give the agent email read, web fetch and a folder with a fake secret, plant
  indirect injections, and run with no defense, then each defense (Rule of Two split, manual approvals, no-egress
  sandbox from Lab 7, a reader sub-agent). At least 20 trials each: attack success rate, benign task success,
  approvals per task, OWASP agentic category of each successful attack. Run a small AgentDojo suite against
  `nanoos` with and without the policy engine. Run `openclaw security audit --deep` on an OpenClaw setup before
  and after hardening.
- **Safety rules:** section 3.4.

### Lab 7. Tools and the sandbox (Tools and sandbox deck, week 7)

- **Goal:** write tools a model can use well, and keep them inside a boundary.
- **Build:** a tool registry in `tools.py` (schemas from type hints, actionable error messages, result truncation
  with a pointer to the full output) and `sandbox.py`, which runs shell commands in a Docker container with a
  read-only workspace mount, no network, and CPU, memory and time limits.
- **Autograder:** a command that tries to reach the network fails; a write outside the workspace fails; a long
  output is truncated with the pointer; timeouts are reported to the model as errors, not crashes.
- **Measure:** the deck's exercise. A `search_tickets` tool over a JSON file with pagination and a
  `response_format` parameter (concise or detailed), one ticket carrying a planted injection. Run from `nanoos`,
  then from Hermes with `terminal.backend: docker` or OpenClaw with `sandbox.mode: all` and read-only workspace
  access. Report tokens per task for concise vs. detailed, tool-call errors before and after rewriting the
  descriptions, whether the injection caused any network call, and approvals per task.
- **Borrowed from:** CMU 11-768's controlled experiment on tool-output design.

### Lab 8. Skills (Skills deck, week 8)

- **Goal:** add procedural knowledge without paying for it in every prompt.
- **Build:** `skills.py`: discover `.agents/skills/*/SKILL.md`, put only names and descriptions in the system
  prompt, load the body when the model calls `skill_view`, and load referenced files on demand (three levels of
  progressive disclosure). Validate skills with `skills-ref`.
- **Autograder:** the index stays under a token budget with 50 skills installed; a skill body is never in the
  prompt until it is opened; a malformed `SKILL.md` is skipped with a warning.
- **Measure:** the deck's exercise. Pick a task the agent often fails, run 10 trials without a skill, write one
  skill, run 10 trials in `nanoos`, Hermes and OpenClaw. Let Hermes patch the skill after failures with write
  approval on and review each diff. Plant an instruction in a skill reference file and check that the sandbox
  blocks it. Report pass@1 and pass^k with and without the skill, activation rate, index tokens, and hand-written
  vs. agent-patched pass rate.
- **Borrowed from:** CMU 11-768 (skills loaded on demand), DeepLearning.AI *Agent Skills with Anthropic*.

### Lab 9. Multi-agent systems (Multi-agent deck, week 9)

- **Goal:** find out when sub-agents pay for themselves.
- **Build:** a `delegate` tool in `delegate.py` that starts a child loop with a fresh context, a task brief and a
  restricted tool set, enforces a depth and concurrency limit, and returns only a summary.
- **Autograder:** the child cannot see the parent's history; limits hold under a scripted fan-out; a failed child
  returns an error summary instead of hanging the parent.
- **Measure:** the deck's exercise. A research task with 4–6 independent sub-questions and an answer key, run as
  one agent, as an orchestrator with workers (in `nanoos` and with Hermes `delegate_task`), and as OpenClaw
  sub-agents on a cheaper model. Five trials each: accuracy and pass^5, total tokens and cost, wall-clock time,
  parent context size at the end, and the most common MAST failure category per architecture.
- **Stretch:** one τ²-bench domain with a single agent vs. an orchestrator.

### Lab 10. The extension layer (Extension layer deck, week 10)

- **Goal:** compare in-process plugins with out-of-process servers for the same feature.
- **Build:** `plugins.py`: a manifest with a config schema, discovery from a plugins folder, an explicit enable
  list, ordered hooks (observe, transform, decide) and exclusive slots (exactly one context engine, one memory
  provider). Move the Lab 3 and Lab 4 engines behind slots.
- **Autograder:** a bad config fails at load time with a clear error; two hook plugins run in a defined order; a
  crashing plugin is disabled without killing the loop; a second provider for a filled slot is rejected.
- **Measure:** the deck's exercise. Build "log every shell command and block `curl | sh`" as a Hermes plugin
  (`pre_tool_call`), an OpenClaw plugin (`before_tool_call`, TypeScript) and an MCP server that wraps shell
  execution. Compare lines of code and setup steps, whether the model can bypass it, added latency per tool
  call, and what happens when the plugin crashes.

### Lab 11. MCP and connectors (MCP, plugins and connectors deck, week 11)

- **Goal:** speak the current MCP protocol from both sides and harden a server.
- **Build:** `mcp_client.py`: a client for MCP 2026-07-28 over stdio and Streamable HTTP that lists tools,
  calls them, handles `input_required` results by asking the user, and filters tools per server. Then a server
  with `search_notes` and `send_note`, where `send_note` asks for confirmation.
- **Autograder:** the client against a reference test server: tool listing, a call over each transport, an
  `input_required` round trip, a filtered tool that is never exposed to the model.
- **Measure:** the deck's exercise. Mount the server in `nanoos`, Hermes (`mcp_servers`) and Codex
  (`[mcp_servers]`). Add a second server with a poisoned tool description and see whether the agent leaks a note;
  turn on one defense at a time (tool filtering, description pinning, a confirmation hook, the Lab 6 policy).
  Report context tokens spent on tool definitions, requests per call over HTTP, attack success per defense, and
  where each defense lived (server, host or hook).
- **Stretch:** put the HTTP server behind OAuth 2.1 with a scoped token, following the deck's authorization
  slides; deploy it as a remote server, as the Hugging Face MCP course does.
- **Note:** check which SDKs support 2026-07-28 before the lab is released; the protocol changed shape in that
  version (no initialize handshake, no sessions).

### Lab 12. Porting to real harness plugin APIs (Harness deep-dives 22–31, week 12; today's Lab 1)

- **Goal:** take what was built in `nanoos` and ship it as plugins for production harnesses.
- **Build:** today's Lab 1, unchanged in scope: **1A** a Hermes `MemoryProvider` + `ContextEngine`, **1B** an
  OpenClaw context engine selected through `plugins.slots.contextEngine`, **1C** Letta supervisor and worker
  agents sharing a block. The change is that 1A and 1B wrap the student's own Lab 3 and Lab 4 engines instead
  of the template logic.
- **Measure:** rerun the Lab 3 long tier through Hermes with the student's engine vs. the built-in one; for 1C,
  show a write by one agent appearing in the other's context on the next turn.
- **Note:** this lab follows the harness deep-dive slides, so it can move earlier (after Lab 4) if the course
  teaches the companion deck in Module 5 as the README suggests today.

### Lab 13. Observability and evaluation (Observability and evaluation deck, week 13)

- **Goal:** turn the per-lab metrics tables into a proper evaluation and monitoring setup.
- **Build:** replace the JSONL logger with OpenTelemetry spans that follow the GenAI semantic conventions
  (agent → turn → model call / tool call / retrieval), with prompt redaction on by default. Add graders to the
  kit: exact match, programmatic checkers, a trajectory grader (expected tool sequence and a convergence score)
  and an LLM judge with a calibration set of 30 human-labeled cases.
- **Autograder:** span tree shape on a scripted run; redaction removes the fake secret; the judge's agreement with
  the labeled set is reported.
- **Measure:** the deck's lab idea. Send spans to a local backend (Jaeger, Phoenix or Langfuse) and vLLM metrics
  to Prometheus; build one dashboard; compare two harness configurations (for example two context engines, or
  skill on vs. off) on the basic and long tiers with pass@k and pass^k. Also evaluate one provided harness the
  students did not write, as Stanford CS329Z HW2 does.
- **Borrowed from:** DeepLearning.AI *Evaluating AI Agents*, Kaggle day 4, the Hugging Face bonus unit.

### Lab 14. A self-improving harness (Harness² / Module 7, week 14; today's Lab 3)

- **Goal:** let the harness improve itself, and keep only the changes the evidence supports.
- **Build:** a small propose → probe → compose loop over `nanoos`'s editable parts (system prompt, tool
  descriptions, skills, context-engine settings, policy rules): a proposer model suggests edits from failed
  traces, each edit is probed on the same failing tasks, adopted edits are composed and checked on a held-back
  set, and every decision is logged with the course's lesson schema (`sample_lesson_entry.jsonl`).
- **Measure:** today's Lab 3: the Harness² demo (`run_harness2_demo.sh`) and, with credentials, one LAB task end to
  end; read the run outputs and fill in lesson records. Then compare Harness²'s adopted edits with the edits the
  student's own loop adopted on `nanoos`: success before and after, cost of the improvement loop, and how many
  adopted edits held on the held-back set.
- **Borrowed from:** Stanford CS329A (self-improvement measured on fixed tasks).

### Capstone: the AgentOS bake-off (weeks 10–15, teams of 2–4)

- Teams start from their best `nanoos` (or a plugin set for Hermes or OpenClaw) and improve it on a published
  development suite. Final scoring runs on the private held-out tier with a fixed token budget per task.
- **Score:** success (pass^3), cost per solved task, and a safety score from the Lab 6 red-team suite, combined
  into one leaderboard; Terminal-Bench 2.0 or a τ²-bench domain can be added as a second track.
- **Deliverables:** proposal (week 10), midpoint demo (week 12), leaderboard submission, a four-page report with
  ablations, and a poster or demo session.
- **Research option:** as at CMU and Stanford, a team can instead propose an open question (for example a new
  scheduling policy in `nano-vllm`, or a memory-poisoning defense) with an evaluation plan.

---

## 5. How to build this (proposed PRs)

Following the repo's one-task-one-PR workflow, in this order:

1. **Lab kit skeleton** (done): `labs/kit/` with the scripted model, the OpenAI-compatible client, `run_suite.py`,
   `grade.py`, the report template and the first ten tasks. Lab 0 handout.
2. **Labs 1–3** (loop, scheduler simulator, context engines), one PR each, with stubs and autograder tests.
   Done: [`lab01`](kit/handouts/lab01_agent_loop.md), [`lab02`](kit/handouts/lab02_scheduler.md),
   [`lab03`](kit/handouts/lab03_context_engines.md).
3. **Lab 4** (memory) and the existing TODO item "a lab for the memory deck exercise".
   Done: [`lab04`](kit/handouts/lab04_long_term_memory.md).
4. **Lab 5**: move today's `lab2_aios_kv_cache/` under the new numbering (or keep the folder name and link it),
   add `prompt.py` and the exercise README. Done: [`lab05`](kit/handouts/lab05_kv_cache_plane.md), which keeps
   the folder name and links it.
5. **Labs 6–8** (policy, tools and sandbox, skills), including the red-team fixtures and their safety notes.
   Done: [`lab06`](kit/handouts/lab06_access_control.md), with the fixtures in
   [`kit/tasks/redteam/`](kit/tasks/redteam/); [`lab07`](kit/handouts/lab07_tools_sandbox.md), without the planted
   ticket injection (a direct network probe replaces it, and Lab 6's fixtures cover attacks);
   [`lab08`](kit/handouts/lab08_skills.md), without the planted instruction in a skill file (a supply-chain
   review of a real skill replaces it).
6. **Labs 9–11** (delegation, plugins, MCP client).
   Done: [`lab09`](kit/handouts/lab09_multi_agent.md), with the research tasks in
   [`kit/tasks/research.py`](kit/tasks/research.py); [`lab10`](kit/handouts/lab10_extension_layer.md), with
   example plugins in [`kit/plugins/`](kit/plugins/); [`lab11`](kit/handouts/lab11_mcp_connectors.md), with a
   reference server in [`kit/tasks/mcp_notes.py`](kit/tasks/mcp_notes.py) and without the poisoned tool
   description (a benign definition change tests pinning, and Lab 6's fixtures cover attacks).
7. **Lab 12**: rework today's `lab1_memory_plugins/` to wrap the students' engines.
   Done: [`lab12`](kit/handouts/lab12_harness_plugins.md), with the adapters in `kit/nanoos/ports.py`; the folder
   keeps its name, and 1B is now a TypeScript shell that forwards to the Python engine over stdio.
8. **Lab 13**: OpenTelemetry, graders, judge calibration set, dashboard.
   Done: [`lab13`](kit/handouts/lab13_observability_eval.md), with spans in `kit/nanoos/otel.py` (OTLP JSON, no SDK
   needed), graders in `kit/nanoos/graders.py` and 30 labeled cases in
   [`kit/tasks/judge_cases.json`](kit/tasks/judge_cases.json).
9. **Lab 14**: improvement loop; link today's `lab3_google_harness2/`.
   Done: [`lab14`](kit/handouts/lab14_self_improving_harness.md), with the loop in `kit/nanoos/improve.py` and a
   simulated agent with bad habits in [`kit/tasks/improve_sim.py`](kit/tasks/improve_sim.py); Part C runs the
   Harness² demo from `lab3_google_harness2/`, which keeps its name.
10. **Capstone** handout and the held-out suite (kept private).
11. **README and syllabus update:** renumber labs, update the course map slide in the master deck, and pacing.

Before the course runs, a pilot pass through all labs should set the real token budgets, check every upstream
API again (the README's "fast-moving APIs" note applies), and time each lab.

---

## 6. Open questions

- **Lab numbering.** This plan numbers labs by lecture (0–14). The master deck and README call today's labs
  Lab 1, 2 and 3. Renumbering touches the master deck's course map; keeping both names for a term is the
  alternative.
- **Hosted model and credits.** Which provider and model to standardize on, and whether students get credits.
- **GPU access.** Labs 2 and 5 need a GPU; a shared course server running vLLM would let every student hit the
  same endpoint and still read its metrics.
- **How much TypeScript.** OpenClaw labs need it; they could become optional for Python-only students.

---

## Sources

University courses

- Harvard CS2680, Modern AI Systems: Agents and System Optimizations: <https://cs2680.seas.harvard.edu/>
- CMU 11-768, AI Agents: <https://www.cmu-agents.com/> (assignment details from a third-party syllabus summary:
  <https://www.heyuan110.com/posts/ai/2026-09-08-cmu-11-768-ai-agents-course/>)
- Stanford CS329Z, Engineering AI Agents: <https://cs329z.stanford.edu/> (assignment details from
  <https://www.heyuan110.com/posts/ai/2026-09-04-stanford-cs329z-engineering-ai-agents/>)
- Stanford CS329A, Self-Improving AI Agents: <https://cs329a.stanford.edu/>
- Michigan EECS 498-016, Applied Agentic Software Engineering: <https://eecs498-aase.github.io/syllabus.html>
- MIT 6.1810, Operating System Engineering: <https://pdos.csail.mit.edu/6.1810/2025/general.html>
- UC Berkeley Agentic AI MOOC, Fall 2025: <https://rdi.berkeley.edu/agentic-ai/f25>

Self-paced courses and tutorials

- Hugging Face Agents Course: <https://huggingface.co/learn/agents-course/unit0/introduction>
- Hugging Face MCP Course: <https://huggingface.co/learn/mcp-course/unit0/introduction>
- Kaggle × Google 5-Day AI Agents Intensive (summary): <https://www.kdnuggets.com/kaggle-googles-free-5-day-agentic-ai-course>
- DeepLearning.AI, LLMs as Operating Systems: Agent Memory: <https://www.deeplearning.ai/short-courses/llms-as-operating-systems-agent-memory>
- DeepLearning.AI, Evaluating AI Agents: <https://www.deeplearning.ai/short-courses/evaluating-ai-agents>
- DeepLearning.AI, Agent Skills with Anthropic: <https://www.deeplearning.ai/short-courses/agent-skills-with-anthropic/>
- DeepLearning.AI, MCP: Build Rich-Context AI Apps with Anthropic: <https://www.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic>
- Microsoft, AI Agents for Beginners: <https://devblogs.microsoft.com/agent-framework/ai-agents-for-beginners-course-10-lessons-teaching-you-how-to-start-building-ai-agents/>
- learn-claude-code: <https://github.com/shareAI-lab/learn-claude-code>
- mini-swe-agent: <https://github.com/SWE-agent/mini-swe-agent>
- nano-vllm: <https://github.com/GeeeekExplorer/nano-vllm> (summary: <https://www.morphllm.com/nano-vllm>)

Benchmarks and tools

- AgentDojo (NeurIPS 2024): <https://arxiv.org/abs/2406.13352>
- τ²-bench: <https://github.com/sierra-research/tau2-bench>
- Terminal-Bench 2.0 and Harbor: <https://www.tbench.ai/docs/run-terminal-bench-2-0>
- Inspect: <https://inspect.aisi.org.uk>
- LMCache benchmark CLI: <https://docs.lmcache.ai/cli/bench.html>
