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

## AGT-2 teaching update prepared for review — 2026-11-07

Long-term-memory slides 17/19 and observability slide 19, with their Chinese twins, now introduce four controlled memory conditions, write/read accounting and a constructed matched-subset checkpoint. Lab 4's optional protocol and the Lab 13 cross-link define controls, eligibility, missing outcomes and deliverables. The existing kit's four read modes remain a different experiment; no full-history/verbatim adapters or model benchmark were implemented. This addresses teaching/protocol material only; AGT-2's executable comparison and measured evaluation remain pending. Before/after screenshots accompany review. Per the user's new instruction, merging requires explicit approval.

## AGT-3 bounded state experiment — 2026-11-09

The [memory provenance/repair fixture](memory_repair_controls_2026_11.md) executes six fixed-rule arms of one constructed history, with two initial bad claims, a repair, a benign owner correction and another bad tool result. It records host source labels, write decisions, retrieval/proposals and supersession lineage; independent tests include newest-only deletion failure, deny-all utility loss and recontamination after one-time repair. Access-control deck slides 9/14 are synchronized in EN/ZH. The provided fixture preserves Lab 4 TODOs and uses its own episodic semantics. This completes the bounded AGT-3 state/teaching experiment, not a model safety study, authentication/semantic retrieval test or persistence/index recovery. AGT-2's measured benchmark remains pending beyond PR #120.

## AGT-4 bounded session-boundary fixture — 2026-11-09

The [handoff/receipt experiment](session_handoff_controls_2026_11.md) executes six main cases plus three short-tail controls, each in a fresh Python worker. At three fixed boundaries it checks goal, completed work, pending operations and simulated effects. Both inputs receive an exact 512-ASCII-byte cap, explicitly not a real tokenizer budget. Negative controls cover short-tail success, changed-key duplication, payload conflicts and missing/incomplete state. Context slides 12/13 are synchronized in EN/ZH; Lab 3's student code and real-model compaction exercise remain intact. This completes the bounded deterministic fixture, not LLM/production-harness recovery, concurrency, write atomicity or power-loss qualification. AGT-2's measured benchmark remains pending beyond PR #120.

## vLLM 0.33.0 source check — 2026-11-11

The open question above (no matching source revision for 0.33.0) is now settled: the compute course's [source-identity record](../../llm-compute-systems-course/references/vllm_source_identity_2026_11.md) ties the PyPI 0.33.0 sdist to tag `v0.33.0`, commit `4ba787ec9`. The 0.32.0 and 0.33.0 sdists from PyPI were then compared for every vLLM interface this course teaches (KV-cache deck, plugin-APIs deck slide 9, Lab 5):

| Interface | 0.32.0 | 0.33.0 |
|---|---|---|
| `--enable-prefix-caching`, `--prefix-caching-hash-algo` (default `sha256`; `sha256_cbor`, `xxhash`, `xxhash_cbor`) | present | unchanged |
| `--kv-cache-dtype fp8` | present | unchanged (new non-FP8 options added) |
| `--kv-offloading-backend` (`native`, `lmcache`), `--kv-offloading-size`, `--disable-hybrid-kv-cache-manager` | present | unchanged |
| `LMCacheConnectorV1`, `NixlConnector`, `MultiConnector`, `OffloadingConnector` in the connector factory | registered | unchanged |
| `NixlConnector` warning that `kv_role="kv_both"` is deprecated | present | unchanged |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` / `_PORT` defaults `localhost` / `5600` | present | unchanged |
| `MultiConnector` save-to-all / load-from-first-hit | present | unchanged; adds sleep-mode support and tolerates children without Prometheus metrics |
| Metrics `vllm:prefix_cache_queries`, `vllm:kv_cache_usage_perc`, `vllm:time_to_first_token_seconds` | present | unchanged |

No slide or lab text changes. The tracker now records 0.33.0. This is a source-level check: Lab 5 was not re-run on a GPU, and the compute course keeps its own 0.32.0 cohort until its runtime gates pass.

## OpenClaw 2026.11.3 contract check — 2026-11-11

The npm packages for 2026.10.7 and 2026.11.3 ship their docs, so the context-engine and manifest pages were compared directly (`docs/concepts/context-engine.md`, `docs/plugins/architecture-internals/context-engines.md`, `docs/plugins/manifest.md`).

- Unchanged: `api.registerContextEngine`, `plugins.slots.contextEngine` with `"legacy"` as the default, manifest `kind: "context-engine"` and `configSchema`, `info.acceptedHostParams`, `assemble()` receiving `availableTools` / `citationsMode`, `contextProjection`, `openclaw plugins install -l`, `delegateCompactionToRuntime` and `buildMemorySystemPromptAddition` in `openclaw/plugin-sdk/core`, and the quarantine rule (an engine that throws is quarantined for the Gateway process and turns fall back to `legacy`).
- Changed: `ownsCompaction: true` still turns off the built-in auto-compaction, but no longer turns off the host's pre-prompt check. The runtime now checks every outgoing request, including the first, against its own estimate; a request over budget enters recovery before it reaches the provider, and the engine's `estimatedTokens` no longer decides this. `promptAuthority` now only affects diagnostics.

Course change: the kit's `OpenClawHost` (`labs/kit/nanoos/hosts.py`) now counts the assembled request itself instead of trusting `estimatedTokens` before calling `compact()`. Slide, handout and Lab 12B text stay as they are (they only say that owning compaction turns the built-in compactor off). With the reference solutions, the Lab 12 and kit tests pass as before; Lab 12B was not run against a live Gateway.

## Hermes Agent v0.23.0 contract check — 2026-11-11

Compared `agent/memory_provider.py`, `agent/context_engine.py` and the docs pages `developer-guide/memory-provider-plugin.md`, `developer-guide/context-engine-plugin.md` and `user-guide/features/skills.md` at tags `v0.22.5` and `v0.23.0` (raw files from GitHub).

- `ContextEngine` is unchanged: `should_compress`, `compress`, `select_context`, `clone_for_agent`, `threshold_percent` default 0.75.
- `MemoryProvider` only gained optional hooks: `on_profile_created`, `on_profile_renamed`, and `read_config_fields` / `write_config_fields` for providers that own their config storage. Every method Lab 12 implements keeps its signature. `spawn_context_thread` (already in v0.22.5) is the documented way to start background work; the Lab 12 handout already asks students to use it.
- Manual `/compress` on every surface now calls the engine's `compress(force=True)` and then `on_session_start(..., boundary_reason="compression")`.
- Skills: `skills_list` / `skill_view` / `skill_manage`, the skill roots and the gating frontmatter on the skills deck are unchanged; v0.23.0 adds `hermes curator lock <skill>` and more trusted vendor skill repos.

No slide or lab change is needed; the tracker moves to v0.23.0.

## AGT-1 bounded import / clone check — 2026-11-14 UTC

The [executed import note](hermes_import_contracts_2026_11.md) checks the actual Hermes v0.22.5/v0.23.0 base classes in isolated source cohorts, provided adapter construction/clone isolation, and five loading controls. A narrow fix stops nested missing dependencies or removed symbols from silently selecting offline stand-ins; the old loader fails three controls. The existing kit passes 36/36 with loopback access. EN/ZH plugin-API slide 28 visualizes the boundary. Full installation, compaction, failed/retried sessions, profile/persistence behavior and OpenClaw remain unqualified, so AGT-1 remains open. PR #120 still supplies only AGT-2 teaching/protocol, not its measured benchmark.
