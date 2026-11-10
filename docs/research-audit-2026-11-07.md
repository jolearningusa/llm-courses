# All-course research audit — 2026-11-07

This round covers **all six courses** at baseline `aeadc78`: ML foundations, Transformer architecture, training/post-training, efficient computing systems, AgentOS and accelerator software. It is a literature/tutorial/repository and compatibility-triage pass, not a benchmark run or a declaration that every slide has been fact-checked. The result is six course-specific reading reviews, 26 retain/update recommendations and ten potential projects with execution and acceptance gates.

## Course reports

| Course | Current inventory | Main result | Detailed evidence and execution plans |
|---|---|---|---|
| ML foundations | 21 English decks | Retain the canonical sequence; strengthen gradient-check diagnosis and fair optimizer comparisons. Existing labs already contain centered differences and learning-rate searches. | [Foundations report](../ml101-neural-networks-course/references/research_audit_2026_11.md) |
| Transformer architecture | 44 English + 44 Chinese decks | Add optional Mamba-3 and dynamic byte-chunking mechanisms; evaluate context quality separately from cache/mask correctness. | [Architecture report](../llm-transformer-architecture-course/references/research_audit_2026_11.md) |
| Training and post-training | 34 English decks | Add controlled coverage/reward/exploration ablations and a rollout/learner consistency walkthrough; improve environment reproducibility. | [Training report](../llm-training-course/references/research_audit_2026_11.md) |
| Efficient computing systems | 41 English + 41 Chinese decks | Prioritize export/runtime compatibility, target-specific kernel contracts, dated hardware claims and recovery evidence. Preserve the published figures/worked examples. | [Compute report](../llm-compute-systems-course/references/research_audit_2026_11.md) |
| AgentOS | 15 English + 15 Chinese decks | Recheck plugin compatibility; add matched memory-cost/quality and multi-round provenance/recovery experiments. | [AgentOS report](../llm-agentos-course/references/research_audit_2026_11.md) |
| Accelerator software | 8 English + 8 Chinese decks | Improve proof that intended work executed, invalid-run classification and one bounded target-adapter path. Keep CPU and physical evidence distinct. | [Accelerator report](../ai-accelerator-software-course/references/research_audit_2026_11.md) |

There are 163 English originals and 108 translated twins, **271 deck files**. Inventory/syllabus/bibliography coverage spans all six courses; detailed code inspection was selective. English-only courses are not claimed to have Chinese twins. The scope is primary technical sources plus existing course material, not social-media consensus or popularity rankings.

## What was searched and read

Search families: introductory gradient/optimization teaching; language-model construction assignments; recurrent and byte-level architectures; long-context evaluation; RLVR coverage/exploration; production training/rollout stacks; agent context/memory evaluation; CUDA kernel evaluation; DMA/driver tests; target qualification; accelerator/runtime documentation. Course reports cite exact papers, sections or repository reading targets and distinguish their conclusions from our proposed experiments.

The reading set includes CS231n, CS336 and its assignment repository; Google Tuning Playbook; Mamba-3, BLT, H-Net and RULER; RL coverage/exploration papers and verl/OLMo-core; MemDelta, MemEvoBench and the long-running harness study; FA4, TileLang, FSDP2 and TorchTitan; Linux DMA/KUnit, Twister, ComputeEval and KernelBench; vendor Rubin/ROCm/Ironwood material. New papers are candidate readings, not automatically established best practices. Canonical sources remain useful even when they are not newly published.

We read primary full-text mechanisms, methods, limitations or implementation descriptions for the selected papers, rather than promoting a search snippet to a verified finding. We did not re-derive every proof, audit entire upstream repositories or reproduce paper tables. GitHub main branches and `latest`/`stable` documentation can move; any later implementation must pin a revision/environment.

## Registry results and a monitoring blind spot

[Machine-readable upstream snapshot](research-audit-2026-11-07-versions.json) records all 15 existing tracked entries across AgentOS and compute, the observed upstream metadata, comparison result and affected paths. Four other courses have no existing tracker; absence of one is not evidence of freshness. The `check_freshness.py --no-links` run reported five changed rows, representing four distinct upstream packages:

| Source | Existing course baseline | Observed upstream | Decision |
|---|---|---|---|
| vLLM (two course entries) | 0.32.0 | 0.33.0 | Identify a matching artifact/source and inspect the relevant API before compatibility testing. Tagged release API returned 404; registry existence is not release-note verification. |
| OpenClaw | 2026.10.7 | 2026.11.1 | Read recovery changes and test actual provider/context interfaces. Failed documentation fetches remain unresolved. |
| Hermes Agent | v0.22.5 | v0.23.0 | Read compaction/session changes; test the actual imported interfaces and restart behavior. |
| PyTorch CPU lab pin | 2.9.1 | 2.15.0 | Intentional older tested baseline. Run a separate isolated compatibility pilot before changing it. |
| ExecuTorch — **not flagged by current rule** | 1.0.0 | 1.6.0 | `compare: 1` suppresses this minor-version movement. Review monitoring granularity and qualify export/lower/load/execute before changing the pin. The inspected moving docs rendered 1.5, so metadata and documentation are not treated as one qualified bundle. |

LMCache, NIXL and Ollama also moved at patch level below configured comparison thresholds; the snapshot preserves those observations. MCP's dated schema revision still matches its tracker. No tracker `verified` date or tested dependency was advanced: metadata retrieval is not runtime qualification. Do not close [freshness issue #40](https://github.com/jolearningusa/llm-courses/issues/40) on the strength of this audit.

## Implementation order after this audit

Course reports contain 36 distinct cited primary-source URLs. P1 means a recommended next bounded compatibility/teaching pass, not a confirmed critical bug; P2 means an optional later extension. “Retain” is a deliberate no-change decision.

These are bounded follow-up proposals, not background jobs or claims of implementation. Use one latest-main branch and one PR at a time; merge before proceeding.

1. **Compatibility and evidence hygiene.** Establish the vLLM artifact/source pair; recheck AgentOS plugin interfaces; fix ExecuTorch monitoring granularity while preserving the old qualified pin. Record real supported environments for the new training course. Do not mix unrelated package migrations into one PR.
2. **Highest-value teaching changes.** Add matched memory evaluation (AGT-2), numerical-gradient diagnostics (FND-1), long-context task/position controls (ARC-3), and RL coverage/reward ablations (TRN-1), each as its own bounded change. Reuse existing CPU kits before writing new harnesses.
3. **Modern mechanism supplements.** Mamba-3, byte/dynamic chunking, and concrete FA4/TileLang source walkthroughs. Change EN/ZH twins together where translations exist; preserve paper cohorts and avoid unbounded model catalogues.
4. **Target-backed projects.** Deployment compatibility, rollout/learner consistency, evaluator coverage and a recoverable target-adapter pilot. Start with independent acceptance/negative controls; record unsupported and not-run outcomes explicitly.

## Research questions versus established gaps

The reports propose ten projects, including a gradient-check atlas, recurrent-state tradeoff study, dynamic-chunk workbench, coverage/exploration ablation, rollout consistency ledger, matched memory evaluation, memory repair, deployment matrix, cross-backend kernel study and evaluator-coverage project. An engineering prototype is valuable without a novelty claim. Questions such as transfer across task distributions, total-compute fairness and portability under fixed contracts are **our candidate research questions**. This audit does not prove they are unanswered by all existing literature. A focused prior-art sweep is a prerequisite to a research proposal.

## Limitations and verification

The checked-in changes are reading reports, source links and a registry snapshot. No slides, model weights, lab implementations, dependency pins, firmware or hardware settings were changed. No GPU, distributed-training, agent-model, driver or thermal experiment ran. Existing CPU test results remain historical validation records rather than being re-certified here.

The stale freshness issue contains synthetic `.example` attack fixtures and access-refused pages as well as genuine link leads. Do not delete deliberate red-team fixtures to reduce a link count, and do not treat a 403 as proof a paper is obsolete. This pass checked selected primary sources, not every external URL in the repository. Inaccessible URLs and the release API discrepancy are disclosed in the relevant report.

## Follow-up: CMP-2 source identity — 2026-11-08

The [compute source-identity supplement](../llm-compute-systems-course/references/vllm_source_identity_2026_11.md) resolves the v0.33.0 annotated tag, verifies the PyPI sdist digest and embedded commit, and matches two pinned NIXL files. Deck 35's EN/ZH evidence diagram distinguishes these checks from runtime qualification. The release REST 404 remains a separate observation. No wheel was inspected/installed, no target was qualified and no connector/benchmark ran; the rest of CMP-2 remains open and all historical trackers/pins stay intact. PR #120 still completes only AGT-2's teaching/protocol portion, not its measured memory benchmark.

## Follow-up: CMP-3 TileLang source reading — 2026-11-09

The [compute GEMM supplement](../llm-compute-systems-course/references/tilelang_gemm_walkthrough_2026_11.md) follows a fixed TileLang author revision and adds EN/ZH memory-flow and edge-validation diagrams to deck 40, with an L4 handoff. Only source inspection and integer coverage were executed. This is the TileLang source portion, not full CMP-3 completion: FA4/CuTeDSL, generated device code, numerical/tail validation and matched target benchmarks remain pending.
