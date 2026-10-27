# L8 — Parallel layouts, recomputation and source tracing

**Released:** analytical layout calculator, CPU activation-checkpoint correctness experiment and pinned source-reading assignment. **Unverified:** measured multi-accelerator throughput/memory. Estimated 5–8 hours, student pilot pending.

From `labs/kit/`:

```sh
python3 parallelism_lab.py > layouts.json
python3 parallelism_lab.py --checkpoint > layouts-and-checkpoint.json
python3 -m unittest discover -s tests -v
```

The analytical path uses the standard library; `--checkpoint` needs the pinned PyTorch environment. Inspect the complete references, then independently derive the formulas before modifying them.

## A. Compare two feasible plans

The supplied eight-rank plans use DP/TP/PP/CP = 2/2/2/1 and 2/1/4/1. Both assume full persistent-state sharding across data ranks, even layer/tensor partitioning, one billion parameters at 16 state bytes/parameter, 2 GB activation and 1 GB workspace **per rank**, eight microbatches and local batch two. Both give 5 GB estimated peak and global batch 32; ideal fill/drain utilization differs (8/9 versus 8/11). These are classroom assumptions, not measured allocations or schedule implementations.

Use a 6 GB/rank ceiling. Identify at least three omitted effects that could invalidate feasibility: gathered parameters, prefetch, padding/head divisibility, imbalance or real allocator overhead. Context parallelism replicates parameters in this calculator; it does not magically reduce all state. Change one assumption to make a plan infeasible. Add a communication ledger for each layout and explain placement; no automated network timing is inferred.

## B. Recomputation

Run `checkpoint_check` with non-reentrant activation checkpointing. Require outputs and input/parameter gradients to pass torch tolerances. This CPU fixture has no stochastic operations. Add dropout as an exercise and analyze RNG preservation and repeated side effects using the pinned API documentation. On a GPU, measure synchronized time and peak allocated/reserved memory; without one, derive a saved-activation estimate and label it analytical.

## C. Source walkthrough

Complete the [pinned TorchTitan/Megatron walkthrough](../training_source_walkthrough.md). Provide file/symbol/commit, input/output layouts, constraints and evidence limits for each step. Neither stack was executed on the authoring host. Submit raw JSON and the [report](../../report_template.md), separating analytical, source-inspected and measured evidence.
