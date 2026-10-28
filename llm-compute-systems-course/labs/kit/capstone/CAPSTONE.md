# Capstone — explain and validate one systems improvement

Choose **one** training, serving, deployment or bounded optimization-controller question. Reuse the course lab evidence path; the [potential projects](../../../projects/POTENTIAL_PROJECTS.md) are optional proposal sources, not a requirement to implement an entire platform.

## Proposal contract

Name a concrete workload and bottleneck hypothesis, the current implementation, one intervention and one adverse condition. Declare numerical/task-quality constraints, hardware/software/input identity, observation window, resources and search budget. Identify the source/trace signature that could falsify the mechanism. Submit a one-page proposal before collecting the final held-out results.

Three paths:

- **Training:** fixed effective batch/data progression, model/optimizer/RNG/checkpoint state, transient/peak memory and maximum-rank step/useful throughput. Compare a supported sharding, overlap, precision or recovery intervention.
- **Serving/agentic:** fixed offered arrivals and complete task/request accounting, compatible prefix identity/ownership, latency/quality SLOs, cache/transfer/recompute and tool-wait effects. Compare one supported scheduler/cache/speculation policy.
- **Deployment/kernel:** fixed computation contract, current library/compiler baseline, export/shape/stride/target support, cold/build/load/warm costs and numerical validation. Include required gradients when claiming a training operator.

Without target hardware, select a **source/analytical capstone** and label every result accordingly. It can earn credit for a correct mechanism, sensitivity analysis and falsifiable experiment design, but cannot make a measured acceleration claim.

## Required artifact package

1. A report with problem, mechanism, baseline, intervention, correctness/quality, evidence, uncertainty, adverse case and limitations.
2. Source/configuration/environment manifests and exact rerun commands. Keep incompatible SDKs in separate environments.
3. Raw samples/events with provenance and accounting for failed/rejected/timed-out/unfinished work. Do not publish only a best-run mean.
4. Correctness tests and a deliberate failing control. A speedup cannot override a failed contract.
5. One ablation and one held-out workload. If composing changes, evaluate the combination as a new candidate.
6. A brief presentation with an execution/dataflow diagram, one worked budget and the practical decision.

Use `capstone_manifest.example.json` as a **proposal example**, not a results file or validator. Replace all example identities with actual selected ones before execution. `critique_example.md` shows how to challenge an overstated analytical result. The [course report template](../../report_template.md) supplies the measurement structure.

## Rubric — 100 points

| Criterion | Points | Evidence |
|---|---:|---|
| Problem/workload and falsifiable hypothesis | 10 | Specific boundary, input and expected signature |
| Credible baseline and environment | 15 | Current eligible implementation; reproducible target |
| Correctness / numerical or task quality | 25 | Declared thresholds, edge cases, negative control |
| Measurement / analytical validity | 20 | Raw evidence, correct timing/units/resources and uncertainty |
| Mechanism explanation | 15 | Dataflow/state/critical-path account consistent with results |
| Ablation, adverse case and held-out test | 10 | Conditions that limit or reverse the benefit |
| Reproducibility and honest limits | 5 | Rerun package, provenance and explicit unsupported paths |

A defensible null result can score well. Unsupported measured claims, omitted correctness failures or fabricated traces cannot. The grader reviews evidence; the manifest alone does not certify correctness, authenticity or performance.

## Workshop and presentation

In week 16, use the workshop to critique a proposed baseline, check the budget and predeclare acceptance. Then present the final result and decision: retain, revise or reject the intervention. Duration estimates remain provisional until a student pilot is recorded; scale model/workload sizes to the actual device. Instructor checks should trace each headline result back to a raw record and a test of the claimed contract.
