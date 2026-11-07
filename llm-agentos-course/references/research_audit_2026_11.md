# Research audit — AgentOS, 2026-11-07

Reviewed all 15 deck topics, tracked-source records and representative memory/plugin/kit boundaries against current primary literature and upstream material. Distinguish memory correctness, useful retrieval, safe writes and cache performance. [Repository-wide audit and registry snapshot](../../docs/research-audit-2026-11-07.md).

## Primary sources and observations

| Source / inspected scope | Finding and course decision |
|---|---|
| [MCP specification, dated 2026-07-28 revision](https://modelcontextprotocol.io/specification/2026-07-28), overview; schema-directory metadata | The checked revision still matches the tracker and the course's stateless/per-request framing. Retain that dated contract; a schema-directory check alone does not validate every transport/auth lab. |
| [OpenClaw 2026.11.1 release](https://github.com/openclaw/openclaw/releases/tag/v2026.11.1), conversation recovery/compaction changes; [Hermes v0.23.0 release](https://github.com/NousResearch/hermes-agent/releases/tag/v0.23.0), compaction and session changes | New release metadata warrants a compatibility pass. Recovery across compaction is a useful invariant to test. Do not infer that a particular provider method changed just because a release number changed. The attempted OpenClaw context-engine documentation URLs could not be read in this pass. |
| [Anthropic's long-running harness study](https://www.anthropic.com/research/effective-harnesses-for-long-running-agents), environment setup and session handoff | Durable progress artifacts and explicit startup checks bridge finite contexts. This implementation example is not proof of unlimited autonomous reliability. Add a restart/handoff experiment rather than more harness branding. |
| [MemDelta: Controlled Baselines and Hidden Confounds in Agent Memory Evaluation, 2606.29914v1](https://arxiv.org/html/2606.29914v1), controlled comparisons, cost and limitations | Match the model, embedding/retrieval setup and evaluated subset before attributing gains to memory design. Its partial-coverage and synthetic-dialogue limits preclude a blanket claim that extraction or persistent memory is useless. |
| [MemEvoBench: Benchmarking Memory MisEvolution in LLM Agents, 2604.15774v1](https://arxiv.org/html/2604.15774v1), memory injection/noisy-tool protocol | Persistent state can turn repeated misleading inputs into later behavioral failures. Extend the course's existing access-control/red-team material with multi-round state provenance and recovery, not just a one-turn refusal score. Its scenarios do not establish universal deployment risk rates. |

## Compatibility findings

The registry scan found vLLM `0.32.0 → 0.33.0`, OpenClaw `2026.10.7 → 2026.11.1`, and Hermes `v0.22.5 → v0.23.0`. LMCache `0.5.6 → 0.5.7` and NIXL `1.5.0 → 1.5.2` are patch movements below the configured comparison threshold. Letta's tracked client version was unchanged. These observations are metadata, not successful lab runs.

For vLLM, PyPI reported 0.33.0, but the corresponding GitHub tagged release body could not be verified: the web fetch exposed a shell and the release API returned 404. Leave feature-level migration unresolved until a matching artifact/source revision is identified. Do not overwrite a tested configuration merely to make the freshness report quiet.

## Proposed edits

| ID / priority | Target | Execution / acceptance |
|---|---|---|
| AGT-1 / P1 compatibility | `labs/lab1_memory_plugins/`, `labs/kit/nanoos/ports.py`; extension/harness/skills decks | In isolated environments, pin new OpenClaw/Hermes artifacts, inspect the actually imported interfaces, run provider/context clone and compaction fixtures, then test a failed/retried session. Update EN/ZH claims and tracker only after recording the supported contract. Preserve old tested environment as a comparison. |
| AGT-2 / P1 evaluation | Long-term-memory and observability decks; memory kit | Add matched no-memory/full-context/verbatim-retrieval/extracted-memory conditions. Keep questions, generator and token budgets fixed; report write/read costs, latency and per-type quality. Different evaluated subsets need separate rows rather than one leaderboard. |
| AGT-3 / P1 safety evaluation | Access-control/context decks; existing red-team tasks | Design a multi-round contaminated-memory fixture with source labels, write decisions, retrieval and later action checks. Include benign corrections and a negative control. Measure recovery after deleting or superseding bad state; do not equate refusal with preserved task utility. |
| AGT-4 / P2 reliability | Agent-loop/context decks; kit | Interrupt a task between sessions, replay durable progress and verify current goal, pending tool state and completed work. Compare explicit handoff with naive transcript truncation at equal context budget. No live service needs to be contacted for the deterministic fixture. |

## Potential projects

**Memory evaluation under matched budgets (replication).** Week 1 freeze task/model/retrieval settings; week 2 implement matched baselines; week 3 record write/read costs and question-type results; week 4 rerun adverse histories and report uncertainty. Ask whether gains survive both cost matching and a new task distribution; novelty is unestablished.

**Memory provenance and repair (engineering/research extension).** Week 1 define trust/write rules; week 2 create benign and misleading multi-round fixtures; week 3 implement correction/deletion with an auditable trail; week 4 assess both harmful-action avoidance and retained utility. All observations are proposed; no model-agent benchmark or new plugin installation ran in this audit.
