# L15 vendor and whole-system dossier

## Boundary and source ledger

| Fact / hypothesis | Value and units | Primary source + date | Scope / conditions | Evidence type |
|---|---|---|---|---|
| Target configuration | Fill | Fill | SKU, topology, memory, firmware | Vendor-documented / measured |
| Kernel / dtype support | Fill | Exact release/support source | Shape, precision, layout | Documented / executed / unknown |
| Rate used in an estimate | Fill | Fill | Peak versus achieved, directionality | Analytical input / measured |

Resolve conflicting source values explicitly; do not silently choose a larger rate. Preserve dense/sparse and datatype footnotes. A marketing speedup needs baseline/model/sequence/power/quality conditions, not just a multiple.

For each rate, also name the exact table column and resource boundary (GPU, Superchip, NVL4/NVL72 or fleet), direction convention and any rounding. Record availability separately with source/publication/access dates and evidence class; vendor production status is not local qualification. The [selected Rubin example](../../references/rubin_spec_evidence_2026_11.md) leaves unresolved source differences explicit. A PDF metadata timestamp is not automatically its publication date.

## Two dataflows

Draw a training step (input → forward/backward → collective → optimizer → checkpoint) and an agentic serving task (queue → prefill/decode → tool wait → resume). Label CPU, GPU/HBM, scale-up, scale-out and storage boundaries. Map each operation to an implementation/library and a validation source. Identify state identity/lifetime and the critical path.

## NVIDIA / AMD responsibility comparison

| Responsibility | NVIDIA candidate + release | AMD candidate + release | Reused / retuned / replaced / unsupported | Local evidence |
|---|---|---|---|---|
| Runtime / launch | Fill | Fill | Fill | Fill |
| GEMM / attention / precision | Fill | Fill | Fill | Fill |
| Collective / one-sided transfer | Fill | Fill | Fill | Fill |
| Framework / serving / profiler | Fill | Fill | Fill | Fill |

A mapping is not feature parity. Record current upstream locations, eligible architecture and actual selected backend. Without both targets, leave performance comparison cells “not measured.”

## Useful-work accounting

Record useful tokens/tasks, complete observation time, failures/retries/repeated work, startup/build, queueing, save/recovery and power boundary where measured. Use disjoint elapsed intervals or a proper dependency/union model; never sum overlapping stage times as wall time. Reconcile a component gain with the whole-job result and include an adverse case.

## Decision and evidence gaps

State one proposed optimization, the expected trace signature, numerical/quality gate, resource/search budget and rejection rule. List unavailable hardware/software evidence and what experiment would resolve it. Do not label an analytical Rubin projection a local benchmark.

### Worked AMD support counterexample — 2026-11-12 UTC

Use the [selected matrix/library ledger](../../references/amd_support_evidence_2026_11.md) to distinguish documentation from execution. Record the selector conditions and exact ROCm/library/target/OS/kernel cohort. Explain why a FNUZ-on-gfx950 support claim contradicts the selected hipBLASLt note, then leave runtime rejection, firmware/driver, operation contract, dispatch and numerical evidence unmeasured until actually tested. Do not fill local qualification from a general platform listing.
