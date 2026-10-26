# LLM AgentOS course: TODO and roadmap

Future deep-dives for the course. Each item becomes its own PR (one task, one PR, see the repo's
[AGENTS.md](../AGENTS.md)). Tick an item when its PR is merged. Sources listed under "Start from" are
already in [`references/bibliography_and_links.md`](references/bibliography_and_links.md); anything new
must be checked against its primary source before it goes on a slide.

## Status of the nine components

The master deck (slide 5) defines a nine-component reference architecture. Current coverage:

| # | Component | Covered today | Own deep-dive? |
|---|-----------|--------------|----------------|
| 1 | Agent loop | Master deck 19 (hook timeline), harness slides 22–25, **`01_agent_loop_deck.html`** | Yes |
| 2 | Scheduler | Master deck 6–7 (AIOS, Autellix, Continuum, KVFlow, ThunderAgent), **`01_scheduler_deck.html`** | Yes |
| 3 | Context manager (select / compress / isolate) | Master deck 8, Lab 1 context engines, **`01_context_manager_deck.html`** | Yes |
| 4 | Memory manager | Master deck 9–14, **`02_long_term_memory_deck.html`** | Yes (cognitive plane); KV plane in Lab 2 |
| 5 | Tools and execution / sandbox | Master deck 17–18 (component map), **`05_tools_sandbox_deck.html`** | Yes |
| 6 | Skills | Master deck 17–18, Hermes deep-dive 22, **`05_skills_deck.html`** | Yes |
| 7 | Multi-agent | Master deck 17–18, Lab 1C (Letta shared blocks), **`05_multi_agent_deck.html`** | Yes |
| 8 | Access control | Master deck 15–16, **`04_access_control_deck.html`** | Yes |
| 9 | Extension layer (plugins) | Companion deck 28 (plugin APIs side by side), **`05_extension_layer_deck.html`** | Yes |

## A. Component deep-dives

Each deck should follow the same shape as the memory deck: the concept, the design space, how Hermes Agent,
OpenClaw, Letta (and a Rust runtime where relevant) implement it, failure modes, and an exercise.

- [x] **Agent loop.** Done as deck **`slides/01_agent_loop_deck.html`**. Turn structure (think, act, observe), stop conditions, retries and error recovery,
  streaming, interrupts, where each hook fires. Compare Hermes hooks, OpenClaw's gateway with per-session
  run serialization, Letta steps, and Codex CLI / Goose loops.
- [x] **Scheduler.** Done as deck **`slides/01_scheduler_deck.html`**. Request-level vs. program-level scheduling, fairness across agents, priority and
  preemption, rate limits and API quotas, how the scheduler interacts with KV-cache reuse.
  Start from: AIOS, Autellix, Continuum, KVFlow, ThunderAgent.
- [x] **Context manager.** Done as deck **`slides/01_context_manager_deck.html`**. Compression strategies (summaries, pruning, structured notes), context isolation
  through sub-agents, prompt-prefix stability and caching, measuring context quality.
  Start from: Hermes `ContextEngine`, OpenClaw `registerContextEngine`, Manus and Anthropic context-engineering posts.
- [x] **Tools and sandbox.** Done as deck **`slides/05_tools_sandbox_deck.html`**. Tool schemas and selection, MCP, tool-result handling, execution backends
  (local, Docker, remote), filesystem and network isolation, timeouts and resource limits.
- [x] **Skills.** Done as deck **`slides/05_skills_deck.html`**. Progressive disclosure, skill discovery and loading, agents that write their own skills,
  skills vs. tools vs. memory. Start from: Hermes `skills_list` / `skill_view` / `skill_manage`, MemSkill.
- [x] **Multi-agent.** Done as deck **`slides/05_multi_agent_deck.html`**. Delegation (Hermes `delegate_task`, OpenClaw sub-agents with isolated or forked context),
  shared memory (Letta blocks), message passing vs. shared state, cost and failure amplification.
- [x] **Access control and safety.** Done as deck **`slides/04_access_control_deck.html`**. Permission models, user confirmation, credential handling, prompt-injection
  defenses for tools and memory, audit logs. Start from: AIOS access manager, memory security survey, A-MemGuard.
- [x] **Extension layer.** Done as deck **`slides/05_extension_layer_deck.html`**. Plugin manifests, slots and lifecycles, versioning and compatibility, plugin
  security. Start from: companion deck 28 and Lab 1.
- [x] **KV-cache plane, second pass.** Done as deck **`slides/03_kv_cache_deck.html`**. A standalone deck for Module 3 beyond the master-deck slides and Lab 2.
- [x] **MCP, plugins and connectors** (added on request, 2026-10-25). Done as deck **`slides/05_mcp_connectors_deck.html`**: MCP 2026-07-28, OAuth for remote servers, registries and connector directories, tool poisoning.

## B. Observability and evaluation deep-dive (logging, telemetry, analytics, dashboards)

Goal: how to tell whether an AgentOS / harness is healthy and whether its quality is going up or down.
Done as deck **`slides/06_observability_evaluation_deck.html`** (the lab is outlined on its slide 19; lab files still to do).

- [x] **Logging and tracing.**
  - What to record per run: turns, model calls, tool calls, memory reads and writes, compaction, sub-agents,
    and the hook events that wrap them.
  - Trace structure: a run as a tree of spans (agent → turn → LLM call / tool call / retrieval).
  - Standards to check: OpenTelemetry semantic conventions for generative AI.
  - Where each harness exposes events today (Hermes hooks, OpenClaw gateway and plugin hooks, Letta run and step records).
  - Privacy: redacting prompts, user data and secrets in logs.
- [x] **Telemetry and metrics.**
  - System: throughput, latency (TTFT, P50/P99 per turn and per task), token usage and cost, prefix-cache hit
    rate, queueing time, fairness across agents (Jain's index). vLLM serves Prometheus metrics, which
    can link this to Lab 2.
  - Agent behavior: steps per task, tool-call error and retry rates, loop and stall detection, context size
    over time, compaction frequency, memory recall hit rate and injection precision (see memory deck slide 17).
- [x] **Evaluating quality.**
  - Offline: task success on benchmark suites, trajectory evaluation (expected tool sequence), response
    evaluation, LLM-as-judge and its biases, regression sets built from production failures.
  - Online: user feedback signals, sampled human review, A/B tests of harness changes.
  - Link to Module 6 (ADK evaluation) and Module 7 (Harness² probes and judge).
- [x] **Analytics and dashboards.**
  - What a harness dashboard should show: success rate, cost per task, latency, top failing tools,
    regressions after a harness or model change.
  - Tooling to survey and verify before teaching: open-source LLM tracing and evaluation platforms (for
    example Langfuse, Arize Phoenix, OpenLLMetry, MLflow tracing), hosted options (for example LangSmith),
    and a plain Prometheus + Grafana setup for system metrics.
- [x] **Lab idea.** Instrument one harness (for example a Hermes plugin that turns hook events into
  OpenTelemetry spans). Send the spans to a local tracing backend, run a small task suite, and build a
  dashboard that compares two harness configurations.

## C. Course-wide housekeeping

- [ ] Re-verify fast-moving APIs (Hermes, OpenClaw, Letta, vLLM / LMCache, Harness²) before each run of the course.
  The weekly Freshness workflow reports upstream changes and stale checks for the sources in
  [`references/tracked_sources.json`](references/tracked_sources.json); see the repo's `docs/updating-courses.md`.
- [ ] Add a lab for the memory deck exercise (same store, automatic injection vs. search tool).
- [ ] Build the lab track in [`labs/LAB_PLAN.md`](labs/LAB_PLAN.md), one PR per step in its section 5 (lab kit first).
  Done: the lab kit and Labs 0–2 ([`labs/kit/`](labs/kit/)). Next: Lab 3 (context engines).
- [x] Add a link checker as the repo's first CI job (`checks.yml`; external links are checked weekly by `freshness.yml`).
