# LLM AgentOS: Agent Operating Systems, Dual-Plane Memory & Self-Improving Harnesses

An introductory course on treating LLM agents as an **operating-system problem**: who schedules the model, how
memory is paged in and out of the context window, how the serving engine manages KV-cache as a system
resource, and how a harness can rewrite itself between tasks.

The course pairs slide decks with three hands-on labs built on real open-source projects:
Hermes Agent, OpenClaw, Letta, vLLM + LMCache + NIXL, and Google Research's Harness².

> **Audience.** Engineers and graduate students who have built at least one tool-using agent and are comfortable
> with Python. Lab 1B uses TypeScript; Lab 2 needs a CUDA GPU (one GPU is enough for the LMCache part).

---

## Slide decks

Open a deck in the browser; use **← / →** to move between slides and **F** for full screen.

| Deck | What it covers |
|------|----------------|
| [Master deck](slides/00_master_agentos_deck.html) (slides 1–21) | The AgentOS architecture, the nine components, and the map of all seven modules |
| [Harness deep-dives](slides/05_opensource_plugin_apis_slides.html) (slides 22–31) | Hermes Agent, OpenClaw, Letta, AIOS, Rust runtimes, plugin APIs, vLLM KV connectors, Harness² |
| [The agent loop](slides/01_agent_loop_deck.html) | Module 1 lecture: turns, budgets, retries, interrupts, compaction, hooks |
| [The scheduler](slides/01_scheduler_deck.html) | Module 1 lecture: lanes and queues, delegation limits, engine and program-aware scheduling |
| [The context manager](slides/01_context_manager_deck.html) | Module 1 lecture: context rot, prompt assembly, cache-stable prefixes, compaction, isolation, Hermes and OpenClaw context engines |
| [Long-term memory](slides/02_long_term_memory_deck.html) | Module 2 lecture: write path, indexes, retrieval, how harnesses build memory, RAG tooling |
| [The KV-cache plane](slides/03_kv_cache_deck.html) | Module 3 lecture: KV-cache sizing, prefix caching in vLLM, provider prompt caching for agents, disaggregation and tiered offload, MLA and FP8 KV, a cache hygiene checklist |
| [Access control and safety](slides/04_access_control_deck.html) | Module 4 lecture: the lethal trifecta and Rule of Two, OWASP agentic top 10, permission models in OpenClaw / Hermes / Codex, prompt-injection design patterns (CaMeL), audit |
| [Tools and sandbox](slides/05_tools_sandbox_deck.html) | Module 5 lecture: writing tools for models, MCP, code execution, Hermes / OpenClaw / Codex / Claude Code sandboxes, approvals |
| [Skills](slides/05_skills_deck.html) | Module 5 lecture: the Agent Skills format, progressive disclosure, how Hermes / OpenClaw / Codex load skills, self-written skills, skill supply-chain risks |
| [Multi-agent systems](slides/05_multi_agent_deck.html) | Module 5 lecture: when multi-agent helps, orchestrator-worker delegation, Hermes / OpenClaw / Letta mechanisms, A2A, failure modes (MAST) |
| [The extension layer](slides/05_extension_layer_deck.html) | Module 5 lecture: plugin manifests, discovery, slots, hooks, plugins vs. MCP servers, versioning and plugin security in Hermes / OpenClaw / Claude Code |
| [MCP, plugins and connectors](slides/05_mcp_connectors_deck.html) | Module 5 lecture: MCP 2026-07-28 (stateless core, transports, multi round-trip requests), OAuth for remote servers, registries and connector directories, how harnesses mount MCP, tool poisoning and defenses |
| [Observability and evaluation](slides/06_observability_evaluation_deck.html) | Module 6 lecture: traces, OpenTelemetry, metrics, evals, dashboards |

## Syllabus (7 modules)

| # | Module | Core ideas | Key sources |
|---|--------|-----------|-------------|
| 1 | **AgentOS architecture, AIOS kernel & program-aware scheduling** | A nine-component reference architecture (loop, scheduler, context manager, memory, tools/sandbox, skills, multi-agent, access control, extensions); agent "system calls", context snapshot/restore; scheduling whole agent programs; context engineering (select, compress, isolate). Separate lecture deck on the agent loop: ReAct and workflows vs. agents, Hermes / OpenClaw / Codex / Letta turn structure, budgets and stop conditions, retries and fallback, interrupts and steering queues, compaction, hooks. Separate lecture deck on the scheduler: three scheduling levels, OpenClaw lanes and queue modes, Hermes delegation limits, AIOS and vLLM schedulers, program-aware serving (Autellix, Continuum, KVFlow, ThunderAgent), fairness. Separate lecture deck on the context manager: context rot and lost-in-the-middle, write / select / compress / isolate, cache-aware assembly, compaction strategies, Hermes `ContextEngine` and OpenClaw context-engine slots | AIOS (arXiv:2403.16971), ThunderAgent (2602.13692), Continuum (2511.02230), Autellix (2502.13965), KVFlow (2507.07400) |
| 2 | **Plane 1: user / cognitive memory** | MemGPT-style virtual context, core vs. archival memory, bi-temporal knowledge graphs, Zettelkasten-style note evolution, single-pass extraction, memory as skills. Separate lecture deck on how long-term memory is built: write path, indexes (keyword / vector / graph), retrieval scoring, automatic injection vs. agent-invoked search tools, how Hermes, OpenClaw and Letta do it, RAG and memory tooling, evaluation | MemGPT (2310.08560), Generative Agents (2304.03442), Zep/Graphiti (2501.13956), A-MEM (2502.12110), Mem0 (2504.19413), MemSkill (2602.02474), GraphRAG (2404.16130), LongMemEval (2410.10813) |
| 3 | **Plane 2: system / KV-cache memory** | PagedAttention, RadixAttention prefix sharing, chunked (non-prefix) KV reuse, multi-tier offload (HBM → DRAM → NVMe → RDMA), cache engineering for agents. Separate lecture deck on the KV-cache plane: sizing, vLLM hash-chained prefix caching and `cache_salt`, provider prompt caching economics, DistServe / Mooncake disaggregation, MLA and FP8 KV, cache hygiene for agent harnesses | vLLM, SGLang, AsymCache / Multi-Segment Attention (2606.02964), Predictive Multi-Tier KV (2604.26968), AKV / Cache Engineering for Agents (2610.28577) |
| 4 | **Tool arbitration & memory security** | Memory lifecycle threat model (write → store → retrieve → execute → share → forget), memory poisoning, consensus-based defenses. Separate lecture deck on access control and safety: the lethal trifecta and Meta's Rule of Two, OWASP Top 10 for agentic applications, adaptive attacks, OpenClaw DM policies and security audit, Hermes approvals, design patterns and CaMeL for prompt-injection resistance | Long-term memory security survey (2604.16548), A-MemGuard (2510.02373, ICML 2026) |
| 5 | **How open harnesses implement AgentOS** | Component-by-component map of Hermes Agent, OpenClaw, Letta and AIOS, one deep-dive per harness, Rust runtimes (Goose, OpenFang, ZeroClaw) and the Python / TypeScript / Rust trade-off, and their plugin APIs: Hermes `MemoryProvider` / `ContextEngine`, OpenClaw `registerContextEngine` + `plugins.slots`, Letta shared blocks and sleep-time agents. Component lecture decks: tools and sandbox (tool design, MCP, code execution, isolation backends, approvals); skills (Agent Skills format, progressive disclosure, Hermes / OpenClaw / Codex skill loading, agents that write skills, SkillsBench, skill supply-chain attacks); multi-agent systems (orchestrator-worker cost and benefit, Hermes `delegate_task`, OpenClaw sub-agents and bindings, Letta shared memory, A2A, MAST failure taxonomy); the extension layer (manifests, discovery, slots, hook points, plugins vs. MCP servers, versioning, plugin security); MCP, plugins and connectors (MCP 2026-07-28 stateless core, Streamable HTTP, OAuth 2.1 authorization, MCP Registry and connector directories, MCP Apps, tool poisoning and MCPTox) | Hermes Agent docs, OpenClaw docs, Letta docs, AIOS / Cerebrum |
| 6 | **OS-level & trajectory evaluation** | Throughput, P99 latency, Jain's fairness index across agents, prefix-cache hit rate; trajectory and response evaluation for agents. Separate lecture deck on observability: agent traces and OpenTelemetry GenAI conventions, OpenClaw / Hermes / vLLM exporters, a harness metric catalog, graders, pass@k vs. pass^k, LLM-as-judge, dashboards, telemetry privacy | Google ADK evaluation (adk.dev/evaluate), OpenTelemetry GenAI semconv, τ-bench (2406.12045), Anthropic agent-evals guide |
| 7 | **Self-improving harnesses (Harness²)** | Task-level recursion (propose harness edits → probe them on the same task → compose a refined harness) and harnessing-level recursion (keep adopted harnesses + evidence-backed lessons) over 8 editable components | Harness² (arXiv:2610.14224, github.com/google-research/harness2) |

Suggested pacing: one module per week, with Lab 1 after Module 5, Lab 2 after Module 3, and Lab 3 after Module 7.

---

## Repository layout

```
llm-agentos-course/
├── README.md                                  # This file: overview, syllabus, reading list
├── LICENSE                                    # Apache-2.0
├── .gitignore                                 # Python, Node, vLLM/LMCache and Harness² run artifacts
├── TODO.md                                    # Roadmap: component deep-dives, observability & evaluation
├── slides/
│   ├── 00_master_agentos_deck.html            # Master deck, slides 1–21: AgentOS architecture & component map
│   ├── 01_agent_loop_deck.html                # Module 1 lecture, 17 slides: the agent loop (turns, budgets, retries, interrupts, hooks)
│   ├── 01_scheduler_deck.html                 # Module 1 lecture, 16 slides: scheduling agents (lanes, delegation limits, vLLM, program-aware)
│   ├── 01_context_manager_deck.html           # Module 1 lecture, 14 slides: context engineering (assembly, caching, compaction, isolation)
│   ├── 02_long_term_memory_deck.html          # Module 2 lecture, 20 slides: how long-term memory is built
│   ├── 03_kv_cache_deck.html                  # Module 3 lecture, 14 slides: the KV-cache plane (prefix caching, prompt caching, offload)
│   ├── 04_access_control_deck.html            # Module 4 lecture, 15 slides: access control and safety (trifecta, permissions, injection defenses)
│   ├── 05_tools_sandbox_deck.html             # Module 5 lecture, 14 slides: tool design, MCP, code execution, sandboxes, approvals
│   ├── 05_skills_deck.html                    # Module 5 lecture, 14 slides: skills (format, loading, self-improvement, security)
│   ├── 05_multi_agent_deck.html               # Module 5 lecture, 15 slides: multi-agent (delegation, routing, shared memory, A2A, failures)
│   ├── 05_extension_layer_deck.html           # Module 5 lecture, 14 slides: the extension layer (manifests, slots, hooks, plugin security)
│   ├── 05_mcp_connectors_deck.html            # Module 5 lecture, 16 slides: MCP, plugins and connectors (protocol, auth, distribution, attacks)
│   ├── 05_opensource_plugin_apis_slides.html  # Slides 22–31: harness deep-dives (incl. Rust), plugin APIs, vLLM KV, Harness²
│   └── 06_observability_evaluation_deck.html  # Module 6 lecture, 20 slides: tracing, telemetry, evals, dashboards
├── labs/
│   ├── LAB_PLAN.md                            # Plan: one practice lab per lecture deck, survey of similar courses
│   ├── lab1_memory_plugins/
│   │   ├── hermes_custom_memory.py            # Hermes Agent MemoryProvider + ContextEngine plugin template
│   │   ├── openclaw_context_engine.ts         # OpenClaw registerContextEngine + plugins.slots config
│   │   └── letta_shared_blocks.py             # Letta multi-agent shared core-memory blocks
│   ├── lab2_aios_kv_cache/
│   │   ├── vllm_lmcache_nixl_config.py        # vLLM MultiConnector (LMCache 256-token chunks + NIXL)
│   │   └── run_vllm_lmcache.sh                # `vllm serve` launcher with 64 GiB CPU DRAM KV offload
│   └── lab3_google_harness2/
│       ├── run_harness2_demo.sh               # Quickstart for google-research/harness2
│       └── sample_lesson_entry.jsonl          # Course schema for an evidence-backed lesson record
└── references/
    ├── bibliography_and_links.md              # Papers (2023–2026), repos, docs and video lectures
    └── tracked_sources.json                   # Fast-moving sources (MCP spec, vLLM, LMCache, ...) and the version last checked
```

### Viewing the slides

Open any HTML file directly in a browser; no build step or network is needed. Use **← / →** (or click) to move
between slides, **F** for full screen. Slides are laid out at 16:9 (1280×720) and scale to the window.

---

## Labs

The plan for a practice lab per lecture deck (a shared lab kit, fifteen labs and a capstone, informed by a survey
of similar university courses and tutorials) is in [`labs/LAB_PLAN.md`](labs/LAB_PLAN.md). The three labs below
exist today; the plan slots them in as Labs 12, 5 and 14.

### Lab 1: memory and context plugins (Module 5)

Three small plugins that hook the same idea, "an OS-managed memory plane", into three different runtimes.

- **1A Hermes Agent** (`hermes_custom_memory.py`): a `MemoryProvider` that recalls notes before each turn and
  extracts facts in `on_pre_compress` before the context engine compresses history, plus a `ContextEngine` that
  only compresses when the token threshold is crossed. Drop the file into `~/.hermes/plugins/course-graph-mem/`
  as `__init__.py` with a `plugin.yaml`, then set `memory.provider: course-graph-mem`.
- **1B OpenClaw** (`openclaw_context_engine.ts`): a context engine registered with `api.registerContextEngine`
  and selected through `plugins.slots.contextEngine`.
- **1C Letta** (`letta_shared_blocks.py`): a supervisor and a worker agent that share one memory block through
  `block_ids`, so a write by one is visible in the other's context on the next turn.

### Lab 2: the system memory plane in vLLM (Module 3)

- `run_vllm_lmcache.sh` starts `vllm serve` with LMCache as the KV offloading backend and a 64 GiB CPU DRAM tier.
- `vllm_lmcache_nixl_config.py` builds a `KVTransferConfig` using `MultiConnector`, which saves KV to every child
  connector and loads from the first one that has it: LMCache (256-token chunks, local CPU) first, then NIXL for
  KV transfer between a prefill and a decode instance.
- Exercise: send the same long system prompt with different user questions and compare time-to-first-token with
  and without offloading.

### Lab 3: a self-improving harness (Module 7)

- `run_harness2_demo.sh` clones `google-research/harness2`, installs it with `uv`, and runs the bundled
  `harness2-demo --output runs/demo --k 2`. Pass `--lab` to also set up and run one LAB task end to end
  (needs Vertex AI credentials, see the upstream README).
- `sample_lesson_entry.jsonl` is a **course-defined** schema for recording an evidence-backed lesson
  (target component, failure signature, proposed edit, probe evidence, decision). Harness² stores lessons and
  adopted harnesses under `runs/`, but its README does not publish a file schema, so treat this file as a
  worksheet for reading the run outputs, not as the upstream format.

---

## Reading list (start here)

1. Packer et al., *MemGPT: Towards LLMs as Operating Systems* (arXiv:2310.08560)
2. Mei et al., *AIOS: LLM Agent Operating System* (arXiv:2403.16971)
3. Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (arXiv:2309.06180)
4. Rasmussen et al., *Zep: A Temporal Knowledge Graph Architecture for Agent Memory* (arXiv:2501.13956)
5. Lin et al., *A Survey on Long-Term Memory Security in LLM Agents* (arXiv:2604.16548)
6. Xu et al., *Harness²: Recursive Agent Harnessing for an Open World* (arXiv:2610.14224)

The full list, with repos, docs and lectures, is in [`references/bibliography_and_links.md`](references/bibliography_and_links.md).

---

## A note on fast-moving APIs

Every plugin API used here was checked against the project's own documentation in October 2026, and the source
page is linked at the top of each lab file. These projects ship weekly; if a lab breaks, check the linked page
first. Known caveats:

- Letta's Python `letta-client` (V1 SDK) is being deprecated in favor of the TypeScript-only Letta Agent SDK.
  The shared-block calls used in Lab 1C are still the documented V1 calls.
- LMCache's in-process mode (environment variables / `LMCACHE_CONFIG_FILE`) is deprecated in favor of LMCache
  MP mode; `--kv-offloading-backend lmcache` is the current `vllm serve` path.
- `kv_role="kv_both"` is deprecated for `NixlConnector`; Lab 2 uses explicit producer/consumer roles for NIXL.

## Roadmap

Planned deep-dives (the remaining AgentOS components, and logging, telemetry, dashboards and quality evaluation
for harnesses) are tracked in [`TODO.md`](TODO.md).

## License

Apache License 2.0, see [LICENSE](LICENSE). Third-party projects referenced here keep their own licenses.
