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

## 6. Evaluation (Module 6)

- **Google Agent Development Kit (ADK) evaluation**: https://adk.dev/evaluate/ · https://github.com/google/adk-python
- **Survey on Evaluation of LLM-based Agents**, Yehudai et al., Mar 2025, arXiv:2503.16416. https://arxiv.org/abs/2503.16416
- Jain, Chiu & Hawe, *A Quantitative Measure of Fairness and Discrimination for Resource Allocation in Shared Computer Systems*,
  DEC-TR-301, 1984 (Jain's fairness index).

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
