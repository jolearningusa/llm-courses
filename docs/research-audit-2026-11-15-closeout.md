# Research-audit disposition and explicit deferrals — 2026-11-15

**16 bounded implementations · 4 retain decisions · 6 explicit deferrals.**

This record closes the **bounded 2026-11-07 audit pass** when this PR is approved and merged. On the unmerged branch it is a proposal: the monitor must wait, not report completion. Merging the record adopts the explicit deferrals below; it does not convert partial evidence into a pass. No target, package migration, model benchmark or physical qualification is newly executed by this documentation update.

The counts refer to the original 26 recommendation IDs in the [six course reports](research-audit-2026-11-07.md#course-reports), including four retain recommendations. “Implemented” means the stated bounded teaching/control/pilot scope, with its limits preserved; it does not mean every downstream research extension or every course claim has been validated. ARC-4 includes its completed cross-course handoff while retaining historical paper/model cohorts.

## Every original recommendation, once

| ID | Disposition for this audit pass | Completed evidence and boundary |
| --- | --- | --- |
| FND-1 | implemented | [Smooth-step/kink diagnostics, explicit scales and rejected wrong gradients.](../ml101-neural-networks-course/labs/L1_autograd_mlp/README.md) |
| FND-2 | implemented | [Equal tuning budgets, validation selection frozen before held-out learning scores; all trials retained.](../ml101-neural-networks-course/references/heldout_optimizer_study_2026_11.md) |
| FND-3 | implemented | [Shared shape/gradient/causality worksheet and reproduced future-leakage failure with restoration.](../ml101-neural-networks-course/references/decoder_bridge_worksheet_2026_11.md) |
| FND-4 | retained | [Canonical sequence and executable examples retained; no unsolicited model catalogue.](../ml101-neural-networks-course/references/research_audit_2026_11.md) |
| ARC-1 | implemented | [Mamba-3 mechanism, streamed/unrolled recurrence, carry/complex/MIMO fixtures and EN/ZH diagrams.](../llm-transformer-architecture-course/labs/L6_block_variants/README.md) |
| ARC-2 | implemented | [Fixed-byte reading/accounting fixture: local/global position ledger, causal boundaries and round-trip controls; neural work is not measured.](../llm-transformer-architecture-course/references/byte_chunk_accounting_2026_11.md) |
| ARC-3 | implemented | [Length/placement/task plan plus a frozen real-checkpoint pilot, actual token spans and retained failed short controls.](../llm-transformer-architecture-course/references/context_model_pilot_2026_11.md) |
| ARC-4 | retained | [Historical paper/model cohorts retained and missing compute 12/35 EN/ZH handoff implemented in #162.](../llm-transformer-architecture-course/references/attention_execution_links_2026_11.md) |
| TRN-1 | implemented | [Coverage/reward ablation, gaming controls, fixed holdout, seed-level cost/quality ledger and negative outcomes.](../llm-training-course/labs/L7_grpo/README.md) |
| TRN-2 | deferred | [Pinned verl path and executed CPU behavior/anchor/learner probability controls.](../llm-training-course/references/rollout_consistency_2026_11.md) **Deferred remainder:** Actual selected rollout/learner backend, worker/queue staleness, distributed timing and end-to-end version/weight publication qualification. |
| TRN-3 | implemented | [No/random/learned steering with frozen checkpoints, held-out scores and replay/time/memory accounting; poor toy generalization retained.](../llm-training-course/references/steering_ablation_2026_11.md) |
| TRN-4 | implemented | [Fresh joint hashed CPU lock plus all-course CPU checks; all L1 reference cells and original L7 demo have separately dated two-cohort replay records.](../llm-training-course/references/l1_remaining_tables_2026_11.md) |
| TRN-5 | retained | [Canonical objectives/derivations retained; small demo differences are not a universal ranking.](../llm-training-course/references/research_audit_2026_11.md) |
| CMP-1 | implemented | [Two isolated export/lower/load/execute cohorts, four cross-version CPU cells, actual partitions, input rejection and cold/warm samples.](../llm-compute-systems-course/references/executorch_cpu_compatibility_2026_11.md) |
| CMP-2 | deferred | [vLLM 0.33 sdist/tag/source identity and NIXL interface reading.](../llm-compute-systems-course/references/vllm_source_identity_2026_11.md) **Deferred remainder:** Install/qualify the selected vLLM/NIXL target cohort; execute connector failure/cancellation/ownership and cold/shared-prefix controls. |
| CMP-3 | deferred | [Pinned TileLang/FA4 walkthrough pair, integer coverage and scalar rescale negative controls.](../llm-compute-systems-course/references/fa4_sm100_walkthrough_2026_11.md) **Deferred remainder:** Author-kernel compilation/dispatch, generated code, numerical/tail/safety checks and matched supported-device/cross-backend measurements. |
| CMP-4 | implemented | [Dated Rubin, Ironwood, Zhenwu, AMD and Ascend evidence ledgers distinguish boundaries/units/support/status; unresolved claims stay flagged.](../llm-compute-systems-course/references/research_audit_2026_11.md) |
| CMP-5 | deferred | [Pinned TorchTitan path and real two-rank CPU DDP/Gloo/DCP restart, cursor/RNG/optimizer comparisons, omission controls and stall/work records.](../llm-compute-systems-course/references/distributed_resume_2026_11.md) **Deferred remainder:** The selected actual TorchTitan/FSDP2 target profile and its worker-loss/topology/async or crash-durability qualification; CPU results do not establish these. |
| AGT-1 | deferred | [Imported old/new Hermes bases/clone controls, actual old/new OpenClaw registry error/abort probes and teaching-host first-compaction fix.](../llm-agentos-course/references/openclaw_context_runtime_2026_11.md) **Deferred remainder:** Full integrated provider/SDK/Gateway/Legacy compaction, clone/profile/persistence and failed/retried-session qualification in a frozen old/new deployment cohort. |
| AGT-2 | deferred | [Teaching/protocol plus later actual four-arm fixed-checkpoint pilot with write/read costs and strict per-type zeros; #120 itself is teaching only.](../llm-agentos-course/references/measured_memory_pilot_2026_11.md) **Deferred remainder:** The broader matched memory benchmark: qualified larger/adverse paired cohorts, utility/scoring adjudication, uncertainty and generalization beyond the tiny strict pilot. |
| AGT-3 | implemented | [Multi-round source/write/retrieval/action-proposal fixture, benign correction, deletion/supersession, negative controls and recontamination.](../llm-agentos-course/references/memory_repair_controls_2026_11.md) |
| AGT-4 | implemented | [Fresh-process equal-cap handoff/truncation comparisons with goals, pending tool state, fake receipts and acknowledgement-loss controls.](../llm-agentos-course/references/session_handoff_controls_2026_11.md) |
| ACC-1 | implemented | [Harness-owned CPU execution-coverage protocol, no-launch/malformed/replay/error/legal controls; no sanitizer pass inferred.](../ai-accelerator-software-course/labs/examples/execution_coverage_report.json) |
| ACC-2 | implemented | [Stage/outcome/scope policy with actual bytecode build/model run and unavailable-board rejection.](../ai-accelerator-software-course/references/qualification_matrix_2026_11.md) |
| ACC-3 | implemented | [Actual Metal shared-buffer ownership/cancel/drain and known-good recovery pilot; host pointers/runtime boundary documented.](../ai-accelerator-software-course/references/metal_lifetime_2026_11.md) |
| ACC-4 | retained | [Physical/PVT/power/thermal/interrupt evidence stays distinct from emulation/model observations.](../ai-accelerator-software-course/references/research_audit_2026_11.md) |

## Six explicit deferrals and concrete resumption gates

These are scope decisions for this pass. In particular, AGT-1/2 are not declared impossible on this computer; they need a new integrated deployment/evaluation protocol. Deferral is proposed to keep those larger studies separate from the completed bounded fixtures and avoid an unending expansion of this monitor. If continued execution is preferred, do not adopt this closeout: select the gate and supply/choose its target/cohort below.

### AGT-1

**Deferred:** Full integrated provider/SDK/Gateway/Legacy compaction, clone/profile/persistence and failed/retried-session qualification in a frozen old/new deployment cohort.

**Reason / current evidence limit:** Scope choice, not a demonstrated hardware impossibility. Imported base/registry fixtures are narrower than full host/provider execution.

**Resume with:** Select an integrated old/new host installation, provider (local or service), session/clone/persistence protocol and supported model/configuration; record full installation/import/retry results.

### AGT-2

**Deferred:** The broader matched memory benchmark: qualified larger/adverse paired cohorts, utility/scoring adjudication, uncertainty and generalization beyond the tiny strict pilot.

**Reason / current evidence limit:** Scope choice, not a GPU prerequisite. The tiny pilot completed but strict scoring/formatting failures and two profiles do not establish broad useful memory behavior.

**Resume with:** Freeze a fresh larger/adverse matched cohort, model/retriever/budgets and appropriate predeclared quality adjudication; retain all writes/reads, failures and uncertainty. Do not overwrite old zeros.

### TRN-2

**Deferred:** Actual selected rollout/learner backend, worker/queue staleness, distributed timing and end-to-end version/weight publication qualification.

**Reason / current evidence limit:** The executed evidence is a local CPU probability contract. It does not run the pinned upstream multi-worker backend/queue path.

**Resume with:** Choose and qualify the actual rollout/learner/backend deployment and worker resources; log behavior/anchor/learner identities, queue/staleness, masks and stage timing under fixed workload.

### CMP-2

**Deferred:** Install/qualify the selected vLLM/NIXL target cohort; execute connector failure/cancellation/ownership and cold/shared-prefix controls.

**Reason / current evidence limit:** The current host/process is Darwin arm64 with no CUDA/HIP build/device; the observed vLLM wheels are Linux candidates. No separate qualified deployment is recorded.

**Resume with:** Provide or choose an eligible Linux/target cohort and supported NIXL/backend/layout profile; install, import and run failure/cancel/ownership and cold/shared-prefix comparisons.

### CMP-3

**Deferred:** Author-kernel compilation/dispatch, generated code, numerical/tail/safety checks and matched supported-device/cross-backend measurements.

**Reason / current evidence limit:** The selected FA4 SM100/CUDA kernel and TileLang target contracts have no eligible local CUDA/ROCm execution evidence. Source/scalar checks cannot supply it.

**Resume with:** Use supported owned target(s), pinned compiler/runtime/builds and actual dispatch; predeclare independent numerical tolerances/tails/safety and equal-work timing before cross-backend comparison.

### CMP-5

**Deferred:** The selected actual TorchTitan/FSDP2 target profile and its worker-loss/topology/async or crash-durability qualification; CPU results do not establish these.

**Reason / current evidence limit:** The executed restart is real CPU DDP/DCP, not the pinned TorchTitan/FSDP2 target recipe. Local eligibility observations do not qualify that GPU deployment.

**Resume with:** Choose the target/process topology and checkpoint/storage/failure contract; execute actual recipe uninterrupted/resumed state/data/RNG/work/stall comparisons and declared worker-loss/durability controls.

## Eligibility/source review, not new qualification

The [read-only evidence record](research-audit-2026-11-15-closeout-evidence.json) records this local Darwin/arm64 Python process, Torch 2.9.1 with no CUDA/HIP build, zero reported CUDA devices and no `nvcc`, `nvidia-smi`, `hipcc` or `rocminfo` in its command path. This is not a remote-machine inventory or proof that no eligible resource exists elsewhere. It makes no inference about Apple Metal availability; the previous real Metal pilot remains distinct, valid recorded evidence within its scope.

The source anchors for the selected [verl rollout-correction document](https://github.com/verl-project/verl/blob/2b4ed6826a39205838d0ca004bdd942334dbe4b6/docs/algo/rollout_corr.md), [TorchTitan checkpointer guide](https://github.com/pytorch/torchtitan/blob/631d544b1f6794ecfeca734a7bba861e03ddd5e3/torchtitan/components/checkpointer/README.md) and [FA4 README](https://github.com/Dao-AILab/flash-attention/blob/c055bdc6444a1a1984082a64b7fd62d6cb42a93d/flash_attn/cute/README.md) were re-fetched from official pinned raw URLs and match the prior SHA-256 values. Their source/configuration guidance is not proof the upstream deployment was executed. The raw web reader failed; direct official-source fetch succeeded. The existing [vLLM artifact/source ledger](../llm-compute-systems-course/references/vllm_source_identity_2026_11.md) preserves the Linux wheel candidates and earlier sdist check; no wheel is installed or qualified here.

AGT-1 retains actual imported Hermes-base/clone controls and actual OpenClaw registry error/abort controls, **not** full integrated-host equivalence. AGT-2 retains real 32-answer cost/strict-score evidence, **not** just the teaching in #120; broad utility/generalization remains deferred. CMP-5 retains real two-rank CPU DDP/Gloo/DCP restart evidence, **not** merely sharding arithmetic; the selected TorchTitan/FSDP2 target/failure profile remains deferred. No prior observed result or failed control is overwritten.

## Optional research projects and other extensions

The ten potential projects in the reports are explicitly deferred from this audit pass, including larger gradient atlases, learned Mamba/dynamic-chunk quality and complete device work, informative long-context quality curves, cross-task RL/memory studies, kernel portability studies and physical evaluator extensions. This is a schedule/scope disposition, not a novelty finding or a claim these projects ran.

Likewise, deployed semantic/authenticated memory repair, production exactly-once/durable handoff, generalized ABI/quantized/accelerator compatibility, all vendor SDK qualification, physical DMA/IOMMU/driver/reset/fault/HIL/PVT/power/thermal/interrupt evidence and cross-platform/GPU/distributed course-wide certification remain unqualified. The bounded ACC-1/2 policy controls and ACC-3 Metal runtime pilot remain implemented; these larger qualifications are deferred extensions, not hidden passes. Unresolved vendor-source discrepancies stay flagged in their dated ledgers.

## Completion/monitor rule

1. While the closeout PR is open, wait without new audit work or unchanged-status notifications.
2. If it closes without merging, stop dependent work and request the unresolved scope decision.
3. Only after actual merge confirms adoption of this record: all 26 original recommendation rows are implemented, retained or explicitly deferred for this pass. Report scoped completion and delete the `continue-course-audit-after-pr-merge` monitor. Do not merge or enable auto-merge to reach that condition.

This does not close [freshness issue #40](https://github.com/jolearningusa/llm-courses/issues/40), advance tracker `verified` dates, change tested pins, or claim all future course maintenance is finished. A later new target/cohort/project starts a separately approved task and protocol.

[Before/after document review captures](reviews/research-audit-closeout/README.md) accompany this proposal. Original seven audit documents, dated progress, raw results and acceptance limitations remain available.
