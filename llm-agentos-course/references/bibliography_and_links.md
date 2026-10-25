# Bibliography, Repositories & Video Lectures

Every arXiv ID, repository, doc page and video below was opened and checked in October 2026. Titles are given as they
appear on arXiv. Venues are listed only where a primary source states them.

## 1. AgentOS foundations & program-aware scheduling (Module 1)

- **AIOS: LLM Agent Operating System**, Mei et al. (Rutgers), arXiv:2403.16971.
  https://arxiv.org/abs/2403.16971 · Kernel: https://github.com/agiresearch/AIOS · SDK (Cerebrum): https://github.com/agiresearch/Cerebrum
- **MemGPT: Towards LLMs as Operating Systems**, Packer et al. (UC Berkeley), arXiv:2310.08560.
  https://arxiv.org/abs/2310.08560 · Now Letta: https://github.com/letta-ai/letta
- **LLM as OS, Agents as Apps: Envisioning AIOS, Agents and the AIOS-Agent Ecosystem**, Ge et al. (Rutgers), Dec 2023, arXiv:2312.03815.
  https://arxiv.org/abs/2312.03815
- **ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System**, Kang et al., Feb 2026, arXiv:2602.13692.
  https://arxiv.org/abs/2602.13692
- **Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live**, Li et al., Nov 2025, arXiv:2511.02230.
  https://arxiv.org/abs/2511.02230
- **Autellix: An Efficient Serving Engine for LLM Agents as General Programs**, Feb 2025, arXiv:2502.13965.
  https://arxiv.org/abs/2502.13965
- **KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows**, NeurIPS 2025, arXiv:2507.07400.
  https://arxiv.org/abs/2507.07400

## 2. Plane 1: user / cognitive memory (Module 2)

- **Zep: A Temporal Knowledge Graph Architecture for Agent Memory**, Rasmussen et al., Jan 2025, arXiv:2501.13956.
  https://arxiv.org/abs/2501.13956 · Graphiti: https://github.com/getzep/graphiti
- **A-MEM: Agentic Memory for LLM Agents**, Xu et al., NeurIPS 2025, arXiv:2502.12110.
  https://arxiv.org/abs/2502.12110 · https://github.com/WujiangXu/A-mem
- **Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory**, arXiv:2504.19413.
  https://arxiv.org/abs/2504.19413 · https://github.com/mem0ai/mem0
- **MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents**, arXiv:2602.02474.
  https://arxiv.org/abs/2602.02474
- **Sleep-time Compute: Beyond Inference Scaling at Test-time**, Lin, Snell, Wang, Packer, Wooders, Stoica, Gonzalez, Apr 2025, arXiv:2504.13171
  (basis of Letta's sleep-time agents). https://arxiv.org/abs/2504.13171
- **Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers** (survey), Du, Mar 2026, arXiv:2603.07670.
  https://arxiv.org/abs/2603.07670

## 2b. Building long-term memory: papers, docs and tools (Module 2, deck `02_long_term_memory_deck.html`)

### Papers

- **MemGPT: Towards LLMs as Operating Systems**, Packer et al., arXiv:2310.08560 (see §1). https://arxiv.org/abs/2310.08560
- **Generative Agents: Interactive Simulacra of Human Behavior**, Park, O'Brien, Cai, Morris, Liang, Bernstein, UIST 2023, arXiv:2304.03442
  (memory stream, retrieval by recency + importance + relevance, reflection). https://arxiv.org/abs/2304.03442
- **From Local to Global: A Graph RAG Approach to Query-Focused Summarization** (GraphRAG), Edge et al., arXiv:2404.16130.
  https://arxiv.org/abs/2404.16130 · https://github.com/microsoft/graphrag
- **LightRAG: Simple and Fast Retrieval-Augmented Generation**, Guo, Xia, Yu, Ao, Huang, arXiv:2410.05779.
  https://arxiv.org/abs/2410.05779 · https://github.com/HKUDS/LightRAG
- **HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models**, Gutiérrez et al., arXiv:2405.14831.
  https://arxiv.org/abs/2405.14831 · https://github.com/OSU-NLP-Group/HippoRAG
- **RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval**, Sarthi et al., arXiv:2401.18059.
  https://arxiv.org/abs/2401.18059
- **Evaluating Very Long-Term Conversational Memory of LLM Agents** (LoCoMo), Maharana, Lee, Tulyakov, Bansal, Barbieri, Fang, arXiv:2402.17753.
  https://arxiv.org/abs/2402.17753
- **LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory**, Wu, Wang, Yu, Zhang, Chang, Yu, arXiv:2410.10813
  (five abilities: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention). https://arxiv.org/abs/2410.10813

### How the harnesses do it (docs)

- Hermes Agent, memory providers (built-in MEMORY.md / USER.md, one external provider, per-provider recall modes and tools):
  https://hermes-agent.nousresearch.com/docs/user-guide/features/memory-providers
- OpenClaw, memory architecture (MEMORY.md, USER.md, daily notes, dreaming, trigger injection, standing intents):
  https://docs2.openclaw.ai/concepts/memory-architecture
- OpenClaw, memory search (hybrid vector + BM25, temporal decay, embedding providers):
  https://docs.openclaw.ai/concepts/memory-search
- OpenClaw, Active Memory (escalation recall sub-agent, default mode `escalate`): https://docs.openclaw.ai/concepts/active-memory
- OpenClaw, `@openclaw/memory-lancedb` plugin (auto-recall, auto-capture, `memory_store` / `memory_recall` / `memory_forget`):
  https://hub.openclaw.ai/openclaw/plugins/memory-lancedb
- Letta, archival memory (`archival_memory_insert` / `archival_memory_search`, passages, tags):
  https://docs.letta.com/guides/agents/archival-memory
- Mem0, how it works (extraction, storage, search signals, scoping): https://docs.mem0.ai/core-concepts/how-it-works
  · add: https://docs.mem0.ai/core-concepts/memory-operations/add · search: https://docs.mem0.ai/core-concepts/memory-operations/search
- LangGraph, memory concepts (semantic / episodic / procedural; hot path vs. background writes) and stores:
  https://docs.langchain.com/oss/python/langgraph/memory
- Anthropic, Introducing Contextual Retrieval (contextual embeddings + contextual BM25 + reranking):
  https://www.anthropic.com/news/contextual-retrieval

### Common tools for building memory and RAG

| Layer | Tool | Link |
|---|---|---|
| Vector store | pgvector (Postgres) | https://github.com/pgvector/pgvector |
| Vector store | Qdrant | https://github.com/qdrant/qdrant |
| Vector store | Chroma | https://github.com/chroma-core/chroma |
| Vector store | LanceDB (embedded) | https://github.com/lancedb/lancedb |
| Vector store | Milvus | https://github.com/milvus-io/milvus |
| Vector index | FAISS | https://github.com/facebookresearch/faiss |
| Vector index | sqlite-vec | https://github.com/asg017/sqlite-vec |
| Keyword index | SQLite FTS5 | https://www.sqlite.org/fts5.html |
| Keyword index | Tantivy | https://github.com/quickwit-oss/tantivy |
| Embeddings / reranker | BGE (FlagEmbedding) | https://github.com/FlagOpen/FlagEmbedding |
| Graph memory | Graphiti (Zep) | https://github.com/getzep/graphiti |
| Graph RAG | Microsoft GraphRAG, LightRAG, HippoRAG | see papers above |
| RAG framework | LlamaIndex | https://github.com/run-llama/llama_index |
| RAG framework | Haystack | https://github.com/deepset-ai/haystack |
| Memory layer | Mem0 | https://github.com/mem0ai/mem0 |
| Memory layer | Letta | https://github.com/letta-ai/letta |
| Memory layer | LangMem (LangChain) | https://github.com/langchain-ai/langmem |
| Memory layer | Honcho | https://github.com/plastic-labs/honcho |
| Memory layer | Supermemory | https://github.com/supermemoryai/supermemory |
| Memory layer | Hindsight (Vectorize) | https://vectorize.io/blog/introducing-hindsight-agent-memory-that-works-like-human-memory |
| Memory layer | Cognee | https://github.com/topoteretes/cognee |

## 3. Plane 2: system / KV-cache memory (Module 3)

- **Efficient Memory Management for Large Language Model Serving with PagedAttention** (vLLM), Kwon et al., SOSP 2023, arXiv:2309.06180.
  https://arxiv.org/abs/2309.06180 · https://github.com/vllm-project/vllm
- **SGLang: Efficient Execution of Structured Language Model Programs** (RadixAttention), Zheng et al., arXiv:2312.07104.
  https://arxiv.org/abs/2312.07104 · https://github.com/sgl-project/sglang
- **Multi-Segment Attention: Enabling Efficient KV-Cache Management for Faster Large Language Model Serving**
  (the AsymCache system), Shi et al., Jun 2026, arXiv:2606.02964.
  https://arxiv.org/abs/2606.02964
- **Predictive Multi-Tier Memory Management for KV Cache in Large-Scale GPU Inference**, Ganjihal, Apr 2026 (v2 Aug 2026),
  arXiv:2604.26968. Six tiers: GPU HBM, CPU DRAM, CXL memory, NVMe via GPUDirect Storage, RDMA fabric, parallel filesystem.
  https://arxiv.org/abs/2604.26968
- **Cache Engineering for Agents** (the AKV serving engine), Fu et al., 20 Oct 2026, arXiv:2610.28577.
  https://arxiv.org/abs/2610.28577
- **LMCache** https://github.com/LMCache/LMCache · Docs: https://docs.lmcache.ai/
- **CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion**, Yao et al., arXiv:2405.16444.
  Non-prefix KV reuse by recomputing a small subset of tokens; available in LMCache as blending (https://docs.lmcache.ai/kv_cache_optimizations/blending.html).
  https://arxiv.org/abs/2405.16444
- **NVIDIA NIXL** (Inference Xfer Library) https://github.com/ai-dynamo/nixl · vLLM guide: https://docs.vllm.ai/en/latest/features/nixl_connector_usage/
- **vLLM MultiConnector** API: https://docs.vllm.ai/api/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.html

## 4. Memory security (Module 4)

- **A Survey on Long-Term Memory Security in LLM Agents: Attacks, Defenses, and Governance Across the Memory Lifecycle**,
  Lin et al., Apr 2026, arXiv:2604.16548. Six lifecycle phases: Write, Store, Retrieve, Execute, Share & Propagate, Forget & Rollback.
  https://arxiv.org/abs/2604.16548
- **A-MemGuard: A Proactive Defense Framework for LLM-Based Agent Memory**, Wei et al., arXiv:2510.02373
  (ICML 2026 per the authors' repository). https://arxiv.org/abs/2510.02373 · https://github.com/TangciuYueng/AMemGuard
- **Agent libOS: A Runtime Substrate for Capability-Controlled Self-Evolving LLM Agents**, Zhang, Jun 2026, arXiv:2606.03895.
  https://arxiv.org/abs/2606.03895

## 5. Production harnesses & plugin APIs (Module 5, Lab 1)

- **Hermes Agent** (Nous Research, MIT) https://github.com/NousResearch/hermes-agent
  - Architecture: https://hermes-agent.nousresearch.com/docs/developer-guide/architecture
  - Built-in memory: https://hermes-agent.nousresearch.com/docs/user-guide/features/memory
  - Skills: https://hermes-agent.nousresearch.com/docs/user-guide/features/skills
  - Delegation (subagents): https://hermes-agent.nousresearch.com/docs/guides/delegation-patterns
  - Plugins and hooks: https://hermes-agent.nousresearch.com/docs/user-guide/features/plugins
  - Memory provider plugins: https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin
  - Context engine plugins: https://hermes-agent.nousresearch.com/docs/developer-guide/context-engine-plugin
- **OpenClaw** (MIT; "OpenClaw 2.0" is the name of the Aug 30 2026 release) https://github.com/openclaw/openclaw
  - Context engine: https://docs.openclaw.ai/context-engine
  - Plugin internals: https://docs.openclaw.ai/plugins/architecture-internals/context-engines
  - Plugin manifest: https://docs.openclaw.ai/plugins/manifest
  - Gateway architecture: https://docs.openclaw.ai/concepts/architecture
  - Agent loop: https://docs.openclaw.ai/concepts/agent-loop
  - Memory: https://docs.openclaw.ai/concepts/memory
  - Skills: https://docs.openclaw.ai/tools/skills · Sub-agents: https://docs.openclaw.ai/tools/subagents
  - Sandboxing: https://docs.openclaw.ai/gateway/sandboxing
- **Rust agent runtimes** (slides 26–27)
  - Goose (Block, Apache-2.0): https://github.com/block/goose · Docs: https://goose-docs.ai/ (smart context management, subagents, permission modes)
  - OpenFang, "Open-source Agent Operating System": https://github.com/RightNow-AI/openfang
  - ZeroClaw: https://github.com/zeroclaw-labs/zeroclaw · IronClaw (Rust, OpenClaw-inspired): https://github.com/nearai/ironclaw
  - OpenAI Codex CLI (`codex-rs`): https://github.com/openai/codex · Why it moved to Rust: https://www.devclass.com/ai-ml/2025/06/02/nodejs-frustrating-and-inefficient-openai-rewrites-ai-coding-tool-in-rust/1619589
  - Rivet agentOS v0.2 (Rust rewrite): https://rivet.dev/changelog/2026-06-25-introducing-agentos-v0-2/
  - OSS Insight, *Rust AI agent infrastructure* (Apr 2026): https://ossinsight.io/blog/rust-ai-agent-infrastructure-2026
- **Letta** shared memory blocks: https://docs.letta.com/guides/core-concepts/memory/shared-memory
  - Archival memory: https://docs.letta.com/guides/core-concepts/memory/archival-memory
  - Sleep-time agents: https://docs.letta.com/guides/agents/architectures/sleeptime
- **Context engineering & harness engineering (practitioner guides)**
  - Yichao "Peak" Ji, *Context Engineering for AI Agents: Lessons from Building Manus*, Jul 2025:
    https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
  - Anthropic, *Effective context engineering for AI agents*, Sep 2025:
    https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
  - Addy Osmani, *Agent Harness Engineering*, Apr 2026: https://addyosmani.com/blog/agent-harness-engineering/

## 5b. Component lecture decks (Modules 1 and 5)

One subsection per component deck. Every link was checked against its source when the deck was written.

### Agent loop (deck `01_agent_loop_deck.html`)

- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR 2023, arXiv:2210.03629. https://arxiv.org/abs/2210.03629
- Anthropic, *Building effective agents* (workflows vs. agents), Dec 19 2024: https://www.anthropic.com/research/building-effective-agents
- Michael Bolin (OpenAI), *Unrolling the Codex agent loop*, Jan 23 2026: https://openai.com/index/unrolling-the-codex-agent-loop/
- Letta, *Rearchitecting Letta's agent loop* (the v1 agent), Oct 14 2025: https://www.letta.com/blog/letta-v1-agent
- Hermes Agent, agent loop (`run_conversation`, `IterationBudget`, fallback, compression): https://hermes-agent.nousresearch.com/docs/developer-guide/agent-loop
- Hermes Agent, delegation guide (`delegation.max_iterations`): https://hermes-agent.nousresearch.com/docs/user-guide/features/delegation
- OpenClaw, agent loop (per-session lanes, timeouts, plugin hooks): https://docs.openclaw.ai/concepts/agent-loop
- OpenClaw, queue steering (`steer`, `followup`, `collect`, interrupts): https://docs.openclaw.ai/concepts/queue-steering
### Scheduler (deck `01_scheduler_deck.html`)

Papers (AIOS, Autellix, Continuum, KVFlow, ThunderAgent) are in §1. Harness and engine docs:

- OpenClaw, command queue (lanes `main` / `subagent`, `maxConcurrent`, queue `mode`, `debounceMs`, `cap`, `drop`): https://docs.openclaw.ai/concepts/queue
- Hermes Agent, delegation limits (`max_concurrent_children`, `max_spawn_depth`, `max_iterations`, stall monitor): https://hermes-agent.nousresearch.com/docs/user-guide/features/delegation
- vLLM, `SchedulerConfig` (`policy` fcfs / priority, `max_num_seqs`, `max_num_batched_tokens`, chunked prefill): https://docs.vllm.ai/en/stable/api/vllm/config/scheduler/
- AIOS source (FIFO and round-robin schedulers): https://github.com/agiresearch/AIOS
### Context manager (deck `01_context_manager_deck.html`)

- Anthropic, *Effective context engineering for AI agents* (attention budget, compaction, just-in-time retrieval, sub-agents), Sep 29 2025: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Yichao "Peak" Ji, *Context Engineering for AI Agents: Lessons from Building Manus* (KV-cache hit rate, mask don't remove, file system as context), Jul 18 2025: https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
- LangChain, *Context Engineering* (write / select / compress / isolate; poisoning, distraction, confusion, clash), Jul 2 2025: https://www.langchain.com/blog/context-engineering-for-agents
- Chroma, *Context Rot: How Increasing Input Tokens Impacts LLM Performance* (18 models), Jul 14 2025: https://research.trychroma.com/context-rot
- Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, TACL 2024, arXiv:2307.03172. https://arxiv.org/abs/2307.03172
- Hermes Agent, context engine plugins: https://hermes-agent.nousresearch.com/docs/developer-guide/context-engine-plugin
- OpenClaw, context engine (`ingest`, `assemble`, `compact`, `afterTurn`, `maintain`; `legacy` default): https://docs.openclaw.ai/concepts/context-engine
- lossless-claw (SQLite + DAG summaries, `lcm_grep` / `lcm_describe` / `lcm_expand`): https://clawhub.ai/martian-engineering/plugins/lossless-claw
### Tools and sandbox (deck `05_tools_sandbox_deck.html`)

- Anthropic, *Writing effective tools for agents* (consolidation, namespacing, concise vs. detailed responses, 25,000-token default cap in Claude Code), Sep 11 2025: https://www.anthropic.com/engineering/writing-tools-for-agents
- Anthropic, *Code execution with MCP* (150k → 2k tokens), Nov 4 2025: https://www.anthropic.com/engineering/code-execution-with-mcp
- Anthropic, *Claude Code sandboxing* (bubblewrap / Seatbelt, network proxy, 84% fewer prompts), Oct 20 2025: https://www.anthropic.com/engineering/claude-code-sandboxing
- Model Context Protocol specification 2026-07-28, Tools (`tools/list` with `ttlMs`, `tools/call`, `isError`, `list_changed` over `subscriptions/listen`, security requirements): https://modelcontextprotocol.io/specification/2026-07-28/server/tools
- Hermes Agent, tools and toolsets (`terminal.backend`: local, docker, ssh, modal, daytona, …): https://hermes-agent.nousresearch.com/docs/user-guide/features/tools
- OpenClaw, sandboxing (`mode`, `scope`, `workspaceAccess`, backends, `tools.elevated`): https://docs.openclaw.ai/gateway/sandboxing
- OpenAI Codex, sandboxing and approvals (`read-only`, `workspace-write`, `danger-full-access`; `on-request`, `never`): https://developers.openai.com/codex/concepts/sandboxing

### Skills (deck `05_skills_deck.html`)

- Anthropic, *Equipping agents for the real world with Agent Skills* (progressive disclosure), Oct 2025: https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- Agent Skills specification (`name` ≤ 64 characters, `description` ≤ 1024): https://agentskills.io/specification
- Hermes Agent, skills (`skills_list`, `skill_view`, `skill_manage`): https://hermes-agent.nousresearch.com/docs/user-guide/features/skills
- OpenClaw, skills (precedence, `metadata.openclaw` gating, snapshots): https://docs.openclaw.ai/tools/skills
- OpenAI Codex, skills (REPO / USER / ADMIN / SYSTEM scopes): https://developers.openai.com/codex/skills
- SkillsBench (curated vs. self-generated skills): https://arxiv.org/abs/2602.12670
- Voyager (skill library for lifelong learning): https://arxiv.org/abs/2305.16291
- MemSkill: https://arxiv.org/abs/2602.02474
- Agent Skills in the Wild (vulnerabilities in public skills): https://arxiv.org/abs/2601.10338
- ClawHavoc: 341 malicious skills on ClawHub, Feb 2026: https://thehackernews.com/2026/02/researchers-find-341-malicious-clawhub.html

### Multi-agent systems (deck `05_multi_agent_deck.html`)

- Anthropic, *How we built our multi-agent research system* (90.2% better than single agent on the internal eval, about 15× tokens), Jun 2025: https://www.anthropic.com/engineering/multi-agent-research-system
- Cognition, *Don't build multi-agents* (share full context, actions carry implicit decisions), Jun 2025: https://cognition.com/blog/dont-build-multi-agents
- Cemri et al., *Why Do Multi-Agent LLM Systems Fail?* (MAST, 14 failure modes): https://arxiv.org/abs/2503.13657
- A2A protocol specification (Agent Card at `/.well-known/agent-card.json`): https://a2a-protocol.org/latest/specification/
- KVFlow (workflow-aware KV-cache management): https://arxiv.org/abs/2507.07400
- Hermes Agent, delegation (`delegate_task`): https://hermes-agent.nousresearch.com/docs/user-guide/features/delegation
- OpenClaw, sub-agents (`sessions_spawn`): https://docs.openclaw.ai/tools/subagents
- OpenClaw, multi-agent routing (bindings, `tools.agentToAgent`): https://docs.openclaw.ai/concepts/multi-agent
- Letta, shared memory: https://docs.letta.com/concepts/shared-memory

## 6. Evaluation (Module 6)

- **Google Agent Development Kit (ADK) evaluation**: https://adk.dev/evaluate/ · https://github.com/google/adk-python
- **Survey on Evaluation of LLM-based Agents**, Yehudai et al., Mar 2025, arXiv:2503.16416. https://arxiv.org/abs/2503.16416
- Jain, Chiu & Hawe, *A Quantitative Measure of Fairness and Discrimination for Resource Allocation in Shared Computer Systems*,
  DEC-TR-301, 1984 (Jain's fairness index).

### Observability and evaluation deep-dive (deck `06_observability_evaluation_deck.html`)

- **OpenTelemetry semantic conventions for generative AI** (operations `chat`, `invoke_agent`, `execute_tool`, …; `gen_ai.*` attributes;
  status: Development). Moved to https://github.com/open-telemetry/semantic-conventions-genai
  (old page with the move notice: https://opentelemetry.io/docs/specs/semconv/gen-ai/). Explainer: https://greptime.com/blogs/2026-05-09-opentelemetry-genai-semantic-conventions
- **OpenInference** semantic conventions (Arize, Apache-2.0): https://github.com/Arize-ai/openinference
- **OpenClaw OpenTelemetry export** (`diagnostics-otel` plugin, `diagnostics.otel` config, exported spans and metrics, content-capture privacy):
  https://docs.openclaw.ai/gateway/opentelemetry
- **Hermes Agent `hermes-otel` plugin** (community; hook-to-span tree, OpenInference + `gen_ai.*`, `content_capture`):
  https://hermes-agent.nousresearch.com/docs/plugins/hermes_otel
- **Langfuse integration with Hermes Agent** (bundled `observability/langfuse` plugin): https://langfuse.com/integrations/other/hermes
- **vLLM metrics** (`/metrics`, `vllm:time_to_first_token_seconds`, `vllm:kv_cache_usage_perc`, `vllm:prefix_cache_hits`, …):
  https://docs.vllm.ai/en/stable/design/metrics/
- **ADK evaluation criteria** (`tool_trajectory_avg_score`, `response_match_score`, rubric-based and multi-turn criteria): https://adk.dev/evaluate/criteria/
- **Demystifying evals for AI agents**, Anthropic Engineering, Jan 2026 (tasks, trials, graders, transcript vs. outcome, pass@k vs. pass^k):
  https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- **τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains**, Yao, Shinn, Razavi, Narasimhan, arXiv:2406.12045.
  https://arxiv.org/abs/2406.12045
- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**, Zheng et al., NeurIPS 2023 Datasets & Benchmarks, arXiv:2306.05685
  (position, verbosity and self-enhancement biases). https://arxiv.org/abs/2306.05685
- Tracing and evaluation platforms: Langfuse https://langfuse.com (SDK MIT) · Arize Phoenix https://arize.com/phoenix/ (Elastic License 2.0)
  · OpenLLMetry https://github.com/traceloop/openllmetry (Apache-2.0) · MLflow Tracing https://mlflow.org/docs/latest/genai/tracing/ (Apache-2.0)
  · LangSmith https://www.langchain.com/langsmith (hosted).
- Metrics and dashboards: Prometheus https://prometheus.io · Grafana https://grafana.com/oss/grafana/

## 7. Self-improving harnesses (Module 7, Lab 3)

- **Harness²: Recursive Agent Harnessing for an Open World**, Xu, Chen, CuiZhu, Dalvi Mishra, Ming, Yu, Han, Lee, Pfister
  (Google Research), 9 Oct 2026, arXiv:2610.14224.
  https://arxiv.org/abs/2610.14224 · https://github.com/google-research/harness2
  Eight editable harness components: system prompt, guardrails, memory, tool guidance, subagents, skills, scripts, plugins.

## 8. Video lectures & MOOCs

- **DeepLearning.AI × Letta: LLMs as Operating Systems: Agent Memory** (course trailer): https://youtu.be/RQ5HzSAx27I
- **UC Berkeley RDI, Advanced LLM Agents MOOC (Spring 2025)**: https://llmagents-learning.org/sp25 · Channel: https://www.youtube.com/@BerkeleyRDI
  - Yu Su, *Reasoning, Memory & Planning of Language Agents*: https://youtu.be/zvI4UN2_i-w
- **Hung-yi Lee (NTU), Introduction to Generative AI & Machine Learning, Fall 2025** (Mandarin): https://speech.ee.ntu.edu.tw/~hylee/GenAI-ML/2025-fall.php
  · Playlist: https://www.youtube.com/playlist?list=PLJV_el3uVTsMMGi5kbnKP5DrDHZpTX0jT
  - Lecture 2, *Context Engineering: the key technology behind AI agents*: https://youtu.be/lVdajtNpaGI
    · Slides: https://speech.ee.ntu.edu.tw/~hylee/GenAI-ML/2025-fall-course-data/Agent.pdf (.pptx also available).
    Its select / compress / multi-agent framing is used on master-deck slide 8.
  - *How AI agents work in one lecture* (ML 2025): https://youtu.be/M2Yg1kwPpts
- **Yun-Nung (Vivian) Chen (NTU), Applied Deep Learning, ADL 8.1: LLM Adaptation** (2023, slides credited to Hung-yi Lee):
  https://www.youtube.com/watch?v=DLHCC8Biiqo. Background on adapting / fine-tuning LLMs; not agent-specific.
