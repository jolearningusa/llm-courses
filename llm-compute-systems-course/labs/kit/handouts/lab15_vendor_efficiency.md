# L15 — Rubin/AMD dossier and whole-system efficiency

**Released:** source-grounded architecture/library worksheet and standard-library analytical phase accounting. **Not validated:** Rubin/AMD hardware, vendor library dispatch or power/energy measurements. Estimated 5–8 hours, student pilot pending.

From `labs/kit/` run `python3 system_efficiency.py > efficiency-model.json` and the unit suite. Inspect the complete reference and reproduce its result independently: 12,000 useful tokens, 100 s disjoint phase ledger, 60 s compute sped up 2× → 70 s total and 1.4286× useful throughput. The costs are constructed and vendor-independent. A trace with overlapping stages needs a critical-path/interval model rather than this sum.

Complete the [dossier template](../vendor_dossier_template.md) using [Rubin](../../../notes/vera_rubin.md), [NVIDIA](../../../notes/nvidia_stack.md) and [AMD](../../../notes/amd_stack.md) primary references. Draw both training and agentic serving dataflows; choose three compute/state/communication optimizations and name the layer and evidence that could validate each. Compare responsibilities, not presumed drop-in equivalence.

The 2026-10-27 source review found different published rate scopes: the current NVL72 product table lists 19.2 TB/s HBM and 3 TB/s NVLink per GPU, whereas the July architecture article gives up-to peak values 22 and 3.6 TB/s. Record this discrepancy; do not mix values into an unlabeled synthetic SKU. Likewise keep sparse NVFP4 inference specifications separate from dense BF16/training figures.

Critique one NVIDIA vendor comparison by recording model, input/output lengths, baseline, precision/sparsity, power boundary and whether performance is projected/subject to change. It is neither a local result nor a general guarantee. For AMD, identify HIP, hipBLASLt/rocBLAS, CK/AITER, RCCL and profiler responsibilities and check the selected architecture/operator support.

Revisit L10: include lost/repeated work and recovery in useful throughput. For serving use completed tasks/accepted tokens under declared quality and latency; for post-training explain rollout/trainer model-version identity and cache invalidation. If hardware exists, validate one supported intervention and compare unprofiled end-to-end results. Otherwise submit an analytical/source dossier with explicit unmeasured cells. Submit JSON and [report](../../report_template.md).
