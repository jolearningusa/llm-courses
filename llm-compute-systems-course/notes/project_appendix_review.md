# Deck review and project appendix integration

Review date: 2026-10-29. The current checkout contains **36 English decks and 36 Simplified Chinese twins**. The review inspected their titles/teaching progression and focused on the project-relevant content in measurement, compiler/kernel, cache, deployment, whole-system and supplement decks. It is a curriculum/content review, not a fresh hardware validation of their examples. Browser checks cover numbering, runtime errors and the edited navigation slides.

## What the decks already teach

| Teaching material | Project implication | Appendix coverage |
|---|---|---|
| 01–04: workload budgets, timing boundaries, host data paths | A project must diagnose a measurable bottleneck and distinguish queue, device and input time. | Common experiment contract; P01, P16, P19 |
| 05–12 and 33: capture/guards, libraries, kernel contracts, tiles and pipelines | Select among existing optimized implementations before generating kernels; programming model comparisons need actual target support. | P06–P08, P22 |
| 13–14: precision, metadata, packing, dispatch and quality | Reduced bytes and a valid low-precision artifact do not establish full-request acceleration. | P10; precision gates across projects |
| 15–18 and 35: engine behavior, cache identity, scheduling and transfer | Engine names/API similarity do not make benchmarks comparable. Preserve exact workload/format/resource differences. | P01–P04 |
| 19–21: MoE, speculation and multi-call agent state | Measure dispatch/skew, correct accept/reject and complete task outcomes. | P03, P11, P17, P23 |
| 22–24 and 34: export, target lowering and vendor stacks | A functioning driver is insufficient; qualify operators, compiler/runtime, partitions and fallback. | P05, P14–P15, P25 |
| 25–30: training state, collectives, layouts, restart and useful work | Preserve update/data progression and maximum-rank completion; include lost work and search cost. | P12–P13, P20, P24 |
| 31–32: capstone and optimization decisions | Baselines, bounded actions and independent evaluation already exist in the curriculum. | P09, P19, P21 and project selection |

## Changes prompted by the supplied strategy and papers

The previous catalogue gives twenty independent scopes but only a common three-milestone execution pattern. [Appendix A](../projects/COURSE_APPENDIX.md) adds per-project weekly tasks, target/access gates, workload plans, baseline comparisons, adverse tests, concrete deliverables and suggested resource caps. P01–P20 IDs remain stable.

Five extensions have narrower research objectives: P21 studies transfer/stopping rather than merely storing history; P22 tests validator detection/cost; P23 addresses operation-dependent cache identity; P24 evaluates distributed candidates with straggler/failure accounting; P25 tests two-model edge contention over sustained operation. Their open-question status is a hypothesis qualified by the [research review](../references/project_research_review.md), not a claim that nobody has studied the topic.

The supplied automated-optimization architecture is useful as a roadmap, but substantial pieces already appear in Ansor, Mirage, FlashInfer-Bench and KernelEvolve. A controller written in Rust, an LLM wrapper or a new benchmark UI does not establish differentiation. The larger P09 experiment therefore compares layer selection and total search cost with existing tuners and simple heuristics.

AKV distinguishes a new semantic boundary: after cache drop/reposition operations, token identity alone does not characterize reusable computation. This does not invalidate deck 16's ordinary exact-prefix contract. P23 explicitly adds operation history and compares against the same declared operation semantics; task quality versus full attention is a separate test. P03 continues to study exact-state retain/offload/recompute policies.

## Navigation and teaching use

The README, syllabus, release guide and capstone contract link the appendix. The existing master, capstone and optimization supplement link it from their final reading/assignment slides in both languages; Chinese links identify the appendix as English. There is no new lecture deck or renumbering: full execution plans belong in the readable appendix, while deck 31 remains the assessment workshop.

Assign P01/P06/P13 as approachable engineering entry points; P03 for AgentOS interests; P14/P25 for edge experience; P21/P22 for bounded research studies. P09 needs a working evaluator before a planner. P04/P11/P12/P24 need real target allocations before claims about accelerated distributed behavior.

## Scope and dates

No appendix project was executed as part of authoring this review. Existing CPU lab validation stays in [VALIDATION](../VALIDATION.md); GPU/edge/student pilots remain separate. New resource estimates and weekly schedules are proposed planning assumptions.

Some existing supplement source-review labels say 2026-10-30, later than this review's 2026-10-29 date. Those historical labels and release tables are not used as proof of freshness here. Project W1 gates require independent current-source and installed-release checks; this change does not silently rewrite unrelated release claims.
