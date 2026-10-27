# L9 — Route, group and combine expert work

**Released:** CPU local float32 reference, torch 2.9.1. No fused GPU GEMM, network dispatcher or router-training validation. Estimated 4–6 hours, student pilot pending.

From `labs/kit/` run `python3 moe_lab.py > moe-cpu.json` and the unit suite. Inspect `routed` and derive the token/slot mapping independently before comparing with `token_reference`. Student extension: build an explicit permutation and inverse-index array, then show equivalence to the index-add reference.

The fixture has 17 tokens, four experts, top-2 distinct expert assignments and normalized gates. Check uniform/skewed routes, empty experts, all 34 contributions and weighted combine (rtol/atol 1e-5). Count assignments, not unique tokens, in expert GEMM shapes. The Python expert loop is not an accelerated grouped kernel. Its single wall-time call includes setup/dispatch and is diagnostic only; do not present it as a stable benchmark.

Draw the distributed dispatch/compute/combine path and derive activation payload using a declared dtype and placement. Include backward and metadata separately. On hardware, compare a supported grouped GEMM/dispatcher with matched routing and numerical contract; time phases with appropriate synchronization, record rank skew and fabric/versions. Without hardware, submit the payload ledger and primary-source mechanism critique as analytical work.

Submit JSON, correctness and load tables plus the [report](../../report_template.md). Instructor diagnostic: reducing active expert FLOPs does not guarantee resident weights fit; changed capacity/drop policy is a semantic change requiring quality evaluation.
