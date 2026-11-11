# Ironwood / TPU7x selected specification review — 2026-11-11

Companion to [deck 40, slide 13](../slides/40_inference_accelerator_architectures.html#slide-13) and its [Chinese twin](../slides/zh/40_inference_accelerator_architectures.html#slide-13). This is the Google portion of CMP-4, checked against vendor documentation, not a runtime or hardware benchmark. The original 2026-11-01 review remains dated history; this note supersedes its interpretation of the selected Google row only.

## Primary source ledger

- [TPU7x system documentation](https://docs.cloud.google.com/tpu/docs/tpu7x), observed 2026-11-11: system table, memory hierarchy, dual-chiplet/programming model and supported configurations. Moving documentation, not a pinned installed stack.
- [Cloud TPU release notes](https://docs.cloud.google.com/tpu/docs/release-notes), entries dated 2025-11-24 and 2026-03-31: service preview and GA respectively.
- [Google Cloud announcement](https://cloud.google.com/blog/products/compute/ironwood-tpus-and-new-axion-based-vms-for-your-ai-workloads), published 2025-11-06: forward-looking availability and vendor performance positioning. Its title announces general availability while the body says availability in coming weeks. Use the service release log for the dated service milestones; do not invent a reconciliation of rollout scopes.

## Units and exposure

| Selected claim | Source scope | Interpretation / unresolved boundary |
|---|---|---|
| 2,307 BF16 and 4,614 FP8 TFLOP/s | System table, per chip peak | Retained, with precision visible. The selected row does not label a sparsity mode; do not invent one or interpret the 2× ratio as a measured application speedup. |
| 192 GiB HBM per chip | Table and memory hierarchy | Nominal physical capacity; runtime/reliability reservations lower framework-visible capacity. No usable-memory measurement was made. |
| Two chiplets, each 96 GiB | Dual-chiplet architecture | Separate memory spaces, with intra-chip D2D communication managed through collectives. Their sum is a capacity count, not one contiguous allocation. |
| Four chips per VM; eight devices exposed by JAX | Programming-model example | Device count differs from chip count. This is Google's documented example, not output captured from a course VM. |
| 7,380 GB/s HBM per chip in table; approximately 7.37 TB/s in prose | Two sections of the same page | Decimal conversion of the table is 7.38 TB/s. Both round to 7.4 at one decimal, but the source does not establish the reason for the difference. Earlier “rounding” language must not imply a verified cause. |
| 1,200 GB/s bidirectional ICI per chip | Table and configurations | Six links × 200 GB/s bidirectional; distinct from D2D, HBM and whole-Pod bisection bandwidth. No collective rate measured. |
| Up to 9,216 chips per Pod | Configurations | Maximum scope, not evidence of a particular allocation. Large slice shapes above 2,048 chips require account-team contact in this observed documentation. |

No sum of nominal capacities is advertised as application-allocatable unified memory. No TFLOP/s or link sum is converted into tokens/s. The diagram is original and conceptual, not a vendor floorplan.

## Status and software boundary

The service log records preview on **2025-11-24**, then GA on **2026-03-31**. These dated vendor milestones do not establish regional capacity, account quota, reservation success or a tested local runtime. The current TPU7x page lists JAX and PyTorch support and excludes TensorFlow; generic older TPU/XLA framework guidance is not enough to qualify TPU7x. No framework package, image, model recipe or compiler was installed or imported for this review. Software compatibility and delivered performance remain unqualified.

## Scope and validation

Only deck 40 slide 13 and the selected Google reference/handout statements change, synchronized in English/Chinese. Other hardware rows and prior lab result tables are untouched. Review captures show before/after at native resolution; browser numbering/text bounds, course consistency, site links and all-deck checks cover publication. No new unit test mirrors documentation facts or arithmetic. No hardware run, library dispatch, latency/power measurement or dependency/tracker upgrade occurred. Other CMP-4 vendor rows remain open; PR #120 still leaves AGT-2's measured benchmark pending.
