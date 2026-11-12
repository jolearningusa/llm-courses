# Research audit — efficient LLM computing systems, 2026-11-07

Reviewed all 41 deck topics, the syllabus/tracker, prior research notes and representative deployment/verification lab boundaries. Preserve the visual and worked-example changes from PRs #103/#104. This pass proposes updates; it does not silently migrate lab environments. [Repository-wide audit and registry snapshot](../../docs/research-audit-2026-11-07.md).

## Primary sources checked

| Source / inspected scope | Finding and decision |
|---|---|
| [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) and [TorchTitan](https://github.com/pytorch/torchtitan), overview and training responsibilities | Extend source walkthroughs with an actual pinned configuration and a resume/equivalence gate. Reading current examples does not validate the older pinned CPU lab or a distributed run. |
| [ExecuTorch getting started](https://docs.pytorch.org/executorch/stable/getting-started.html), export/delegate workflow | The moving `stable` page rendered 1.5 while PyPI metadata reported 1.6.0. Keep documentation version and install artifact separate. The tested 1.0 CPU pin remains the baseline until a new export/lower/load/execute pilot succeeds. |
| [FlashAttention-4 author README](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/README.md) and [TileLang author repository](https://github.com/tile-ai/tilelang), implementation/target descriptions | Target-specific realization matters more than DSL name alone. Decks 12/33/40 should link a pinned kernel with its shape/dtype/mask constraints. Source availability is not tested support for every backend. |
| [vLLM disaggregated-prefill documentation](https://docs.vllm.ai/en/latest/features/disagg_prefill/), experimental status | Keep placement, transfer cost and failure ownership in the comparison. A connector/configuration example is not end-to-end throughput evidence. PyPI 0.33.0 was observed, but a matching tagged GitHub release body was unavailable in this pass. |
| [NVIDIA Rubin platform](https://www.nvidia.com/en-us/data-center/technologies/rubin/), platform scope/status; [Google TPU7x/Ironwood](https://docs.cloud.google.com/tpu/docs/tpu7x), system documentation; [AMD ROCm](https://rocm.docs.amd.com/en/latest/), components and compatibility links | Re-read dated device/library claims against vendor pages and per-target matrices. Rubin's current page describes production and agentic/reasoning infrastructure; do not turn platform marketing into a measured model-specific speedup. ROCm's moving landing page is not a universal supported combination. |

## Freshness findings beyond the automatic headline

The 15-entry repository scan reports five rows above configured comparison thresholds, including compute's PyTorch `2.9.1 → 2.15.0` and vLLM `0.32.0 → 0.33.0`. The L3T Triton pin still matches observed 3.9.0 metadata. The shared snapshot records patch movements too.

**ExecuTorch is a monitoring gap:** observed `1.0.0 → 1.6.0` is suppressed by `compare: 1`. Minor-version movement can still matter to export/runtime APIs. Recommend comparing minor versions in a separate monitoring change while retaining the tested pin until qualification; advancing `verified` would wrongly suggest a successful compatibility check. The rendered 1.5 docs / 1.6 registry difference must also be resolved against versioned documentation.

## Proposed changes

| ID / priority | Exact target | Execution and acceptance |
|---|---|---|
| CMP-1 / P1 compatibility | Decks 05/06/22/23; L2/L11 requirements and deployment reports | Create isolated old/new environment lanes, freeze model/input/export constraints and compare eager/export/lowered output. Test one documented shape rejection and cold versus warm timing. Save actual delegate partition, versions and artifact identity. Update pins only after the reproducible pilot. |
| CMP-2 / P1 compatibility | Decks 16/18/35; AgentOS cache lab | Identify an installable vLLM 0.33 artifact/source pair first. Recheck connector roles, failure/cancel ownership and cold/shared-prefix controls on a qualified target. No version-only feature claim or simulator timing substituted for engine evidence. |
| CMP-3 / P1 mechanism | Decks 12/33/40; L4/L5 | Select one FA4/CuTeDSL and one TileLang implementation at revisions. Map tile layout, staging, synchronization, masks and supported shapes. Begin with numerical/tail cases; benchmark matched contracts on supported hardware only. A source-only report lists missing generated-code/device evidence. |
| CMP-4 / P1 source accuracy | Decks 24/40; vendor notes | Recheck each hardware table row against a dated vendor source, separating chip, rack and cluster units; peak precision/sparsity from delivered performance; current status from roadmap. Preserve historical statements as explicitly dated history. Any unverifiable product specification stays unresolved rather than being repeated as fact. |
| CMP-5 / P2 training reliability | Decks 25/28/29; distributed/recovery labs | Pin a TorchTitan/FSDP2 path and require uninterrupted-versus-resumed progress equivalence, including data cursor/RNG and optimizer state. Then measure checkpoint stalls and useful completed work. CPU sharding arithmetic alone cannot qualify distributed recovery. |

## Potential projects

**Deployment compatibility matrix (engineering).** Week 1 freeze old/new environments and constraints; week 2 export/lower/load and test allowed/rejected inputs; week 3 collect delegate/fallback/memory evidence; week 4 publish reproducible manifests and unsupported cases. Do not confuse new documentation with successfully installed runtimes.

**Kernel portability under a fixed contract (replication/research extension).** Week 1 choose operation and quality tolerance; week 2 inspect pinned implementations and validate irregular inputs; week 3 measure one supported target; week 4 add a second eligible backend and explain regressions using generated code and counters. Novelty requires a broader prior-art check, not simply writing the same GEMM in two DSLs.

No GPU or distributed experiment, new model run, hardware purchase recommendation or dependency upgrade was performed. This is a scoped source/registry audit; it is not an exhaustive verification of every external link or every existing device claim.


## Follow-up progress — 2026-11-07

**ExecuTorch monitoring gap, prepared for review:** changed only that tracker's comparison granularity from one to two components; the 1.0.0 / 2026-10-27 tested baseline and all dependency pins stay intact. Deck 23's final slide now visualizes registry/docs/qualified-environment evidence and export/lower/reload/execute gates in English and Chinese. The dated monitoring note and L11 handout define the later candidate pilot. Registry 1.6.0 was re-observed; stable docs rendered 1.5; matching versioned 1.6 docs and new runtime compatibility remain unqualified. This completes the bounded alert-rule/teaching correction, not CMP-1's runtime matrix or a package migration.

## CMP-2 progress — 2026-11-08

**Source-identity gate completed; runtime qualification still open.** The [dated supplement and manifest](vllm_source_identity_2026_11.md) resolve the annotated v0.33.0 tag to `4ba787ec9959e566d3b587a796dff07673324da5`, verify the downloaded sdist digest and matching embedded commit, and compare two NIXL files with the pinned source. Release REST still returns 404; this does not mean a missing Git tag. Deck 35 slide 20 now visualizes this boundary in EN/ZH. Published Linux wheels were listed, not downloaded or installed. No full-tree/build-provenance proof, target dependency resolution, imported connector, cancellation/failure test or cold/shared-prefix benchmark ran. Historical tracker/pins remain unchanged. This is only CMP-2's source/artifact identification prerequisite; an **installable** pair still needs the qualified environment gate.

## CMP-3 progress — TileLang source portion, 2026-11-09

The [pinned GEMM walkthrough](tilelang_gemm_walkthrough_2026_11.md) maps launch, shared/fragment layout responsibilities, K staging, synchronization ownership and edge checks to one exact author revision. Deck 40 slides 19/21 and both reference handouts now provide diagrams and a concrete full/tail/small-M worksheet; L4 links the same contract. Integer coverage arithmetic ran, with no tensor/compiler/device execution. This completes only the bounded TileLang source-reading supplement. Generated code, irregular-shape/numerical checks on target, performance, FA4/CuTeDSL and cross-backend evidence remain open. No package pin or tracker date changed.

## CMP-3 progress — FA4 source portion, 2026-11-09

The [FA4 SM100 walkthrough](fa4_sm100_walkthrough_2026_11.md) pins one ordinary forward path, interface guards and softmax/correction sources. Deck 12 slides 15/17 now visualize two-way readiness and candidate full/tail/small-N validation in EN/ZH; L5 links a scalar checkpoint with two failing omissions. Only text inspection and standard-library scalar arithmetic ran. Together with the TileLang supplement this completes the bounded source-reading pair, not CMP-3's generated-code/device safety/numerical/performance or cross-backend work. No tracker or dependency changes.

## CMP-4 selected Rubin rows — 2026-11-10

The [dated evidence ledger](rubin_spec_evidence_2026_11.md) rechecks product/PDF/architecture units, footnotes, workload comparisons and vendor-reported status. EN/ZH deck 24 slides 6/7 now separate GPU/Superchip/NVL72, the PDF's NVL4 column, sparse/dense peaks and conditioned versus locally measured performance. The 19.2/3 versus 22/3.6 TB/s source discrepancy remains unresolved; no SKU/clock explanation is invented. This completes only the bounded Rubin claim review. Other vendors, actual library support and hardware/workload validation remain open.

## CMP-4 selected Ironwood rows — 2026-11-11

The [Google evidence ledger](ironwood_spec_evidence_2026_11.md) rechecks chip/chiplet/JAX-device counts, nominal versus usable HBM, bandwidth units and service preview/GA milestones. Deck 40 slide 13 has synchronized original EN/ZH diagrams. The source table/prose rate difference is retained without an invented explanation. This completes the bounded Google claim review, not target/software qualification or all CMP-4 vendors. No runtime or benchmark ran. PR #120 still leaves AGT-2's measured benchmark pending.

## CMP-4 selected Zhenwu claims — 2026-11-12 UTC

The [Alibaba evidence ledger](zhenwu_claim_evidence_2026_11.md) rechecks the M890/V900 announcements, 810E deployment template, capacity/link scope and two vendor 3× claims. EN/ZH deck 40 slides 15/16 distinguish documentation, announcements and forecast. The old keynote timed out twice; no fresh product-family classification is inferred from it. This completes the bounded selected Zhenwu review, not SAIL/toolchain qualification, matched performance or all CMP-4 vendor rows. No runtime or benchmark ran. PR #120 still leaves AGT-2's measured benchmark pending.
