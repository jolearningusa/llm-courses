# AI-assisted development and verification lab

Companion to decks 37–39. CPU baseline: Python 3.9+, standard library only. Optional CUDA extension needs a qualified compiler/device and Compute Sanitizer; it was not executed on the authoring host. Estimated 3–5 hours for the CPU path, pending student pilot.

## Learning contract

An AI may propose a candidate or a test, but neither its explanation nor its score grants correctness. Keep the implementation under test separate from the acceptance contract. Test the evaluator with both an incorrect candidate it must reject and a legal transformation it must accept.

Two cases run locally: an RMSNorm-style numerical operation and an explicit queue-event trace. The trace is a list of authored events, not a concurrent device simulator. The follow-on accelerator course develops that model further.

## Run the baseline

From `llm-compute-systems-course/labs/kit/`:

```sh
python3 verification_lab.py
python3 -m unittest discover -s tests -p test_verification.py -v
python3 verification_lab.py --output verification-report.json
```

The output path must not already exist. The JSON records candidate source hashes and per-case decisions; it is a teaching record, not a signed attestation. Expect reference/reordered numerical candidates to pass, three numerical mutants to fail, early buffer release and stale completion to fail. No performance data is produced.

## A. AI-proposed numerical change

1. Read `reference`, its finite-input scope and the fixed tolerance. Predict which fixture rejects each mutant. The teaching inputs avoid overflow in squared values; this is not a production-general stable norm.
2. Ask an assistant to propose one legal optimization and one adversarial input family. Review the patch before adding it as a named candidate; this lab never loads or executes arbitrary generated files.
3. Freeze held-out inputs before evaluating the candidate. Check shape, complete writes, finite values and the combined absolute/relative error contract. Add a valid reduction-order variant so the evaluator cannot “win” merely by rejecting everything.
4. Compare cases near zero and with nonzero output. Explain why an all-zero test alone cannot reject a constant-zero implementation.
5. For a real low-precision/GPU path, select dtype-specific error bounds and model-quality checks. CPU higher precision is a numerical reference, not a universal bit-exact oracle.

## B. AI-proposed queue change

1. Follow submit → start → complete → release. State who owns the buffer at each boundary.
2. Review the “release immediately after submission” proposal. Inject the early-release trace and explain why outstanding work makes it invalid.
3. Test queued cancellation separately from in-flight cancellation. In this contract in-flight work needs completion or reset; a client disconnect alone is insufficient.
4. Reset increments generation and drains ownership in the teaching contract. An old completion must be rejected. Actual hardware needs a proven quiesce/reset boundary before buffers can be reclaimed.
5. Add duplicate completion, cancellation/release and missing-release cases. A passing event list does not explore all possible interleavings or establish memory-ordering correctness.

## C. Optional target extension

Choose one reviewed CUDA reduction or runtime API path, lock the environment and run numerical checks before timing. Use memcheck for memory accesses, racecheck for shared-memory hazards, synccheck for synchronization usage and initcheck for supported uninitialized reads. These have bounded scope and do not prove an entire runtime race-free. Run one deliberate failing control in an isolated test process. Preserve logs and an explicit missing-tool/not-run status.

Do not flash firmware or inject faults into production hosts. A hardware extension uses a designated test device, supported recovery procedure and an approved operating envelope. Functional simulation cannot establish silicon power/thermal behavior.

## Deliver

Submit the candidate diff, frozen contract, generated versus human-authored provenance, complete JSON outcomes, held-out/negative-control fixtures, one minimized failure and a retain/revise/reject decision. If timing a target extension, report cold/build/search cost and unprofiled end-to-end results separately. The released CPU exercise demonstrates evaluator/state-contract behavior only.

## D. Design a firmware/runtime qualification campaign

Deck 38 adds three design exercises; deck 39 turns them into a release plan. Budget 60–90 minutes for the worksheet, separately from the executed CPU baseline. This is a proposed extension: the released lab does not implement graph fuzzing, real interrupts, silicon telemetry or firmware deployment.

1. **Freeze the claim and oracle.** Pick one change: fused reduction, graph pass, queue cancellation or layout optimization. Record supported shapes, layouts, input/accumulator/output dtypes, exceptional-value behavior and exact versus tolerance-based fields. For finite floating-point comparisons, a possible rule is `abs(candidate-reference) <= atol + rtol * abs(reference)`; justify operation-specific bounds before evaluating candidates. Include near-zero, cancellation, tail and dynamic-range cases. For quantization, add held-out model-quality acceptance and record scaling/calibration. A CPU reference is not automatically bit-exact.
2. **Separate graph campaigns.** For valid graphs, vary documented shapes/operators/pass combinations and compare independently evaluated results. For malformed metadata, choose the exact parser/API boundary and expected error; verify no unintended submission or leaked state. Give valid but expensive cases explicit time/memory budgets. Keep these outcomes separate from numerical mismatches. Save seeds, input tensors, graph, pass settings, versions, stderr and a minimized reproducer. Use process/resource isolation for experimental compiler tests.
3. **Write a fault schedule.** Start with `submit → DMA starts → lost IRQ → timeout`. State who owns the memory and what establishes quiescence. Add delayed completion, reset and stale-generation notifications. Name both a safety invariant and a progress/recovery bound, plus assumptions about completion, interrupts and reset. Replay seeds and minimize the failing trace. A bounded state exploration reports depth and abstractions; it does not prove all concurrent hardware executions.
4. **Design observability.** Assign request IDs, generations, queue-depth/progress counters, error/reset causes and bounded event buffers. Define timestamp domains and correlation, log retention across reset and instrumentation overhead. Specify which raw events reproduce the ownership failure. Software hooks complement hardware counters; they do not replace silicon DFT.
5. **Plan the physical comparison.** Use a designated supported device and identical workload/quality gates. Record baseline/candidate identities, warm-up, cooling, power limits, duration, repetitions and run order. Name each sensor/counter, boundary, units and sampling cadence. Mark unavailable current/voltage/bandwidth fields unsupported. Integrate measured power over a matched completion window to estimate energy; distinguish device/board/system energy and gross/idle-subtracted values. Board temperature cannot diagnose a local SRAM hotspot; a functional emulator or FPGA cannot certify final-silicon thermal behavior.
6. **Define HIL promotion.** Reserve one recoverable test board; review the artifact and supported deployment method; run functional/numerical/fault/soak checks; collect raw logs; restore a known-good artifact and verify a fresh command. State stop conditions within the supported operating envelope. Runtime/kernel changes need not entail firmware flashing. Expand the device pool only after single-board recovery and evidence collection work.

Submit this matrix with a reason for every unsupported or not-run entry:

| Claim | Target / input family | Oracle and stop condition | Artifact / raw evidence | Status |
|---|---|---|---|---|
| Numerical equivalence | Declared dtype, shapes and exceptional cases | Frozen bounds plus rejected mutant and accepted legal transform | Inputs, outputs, reference and candidate identity | Not-run until measured |
| Ownership and progress | Lost IRQ, delayed completion, reset | No reclaim before quiescence; bounded recovery under stated assumptions | Seed, event trace, minimized witness | Not-run until exercised |
| Graph robustness | Valid graphs and invalid metadata, separately | Numerical agreement or specified rejection / resource outcome | Graph, pass config, logs, minimized case | Not-run until exercised |
| Sustained target behavior | Named silicon, fixed workload and operating conditions | Declared quality, latency, resource and recovery gates | Time-aligned telemetry, counter definitions, completed work | Not-run without device |
| Release recovery | One designated recoverable target | Known-good restore and fresh-command success | Deployment identity, device logs, recovery record | Not-run without device |

Use the [companion verification lecture](../../../../ai-accelerator-software-course/slides/05_verification.html), [qualification lecture](../../../../ai-accelerator-software-course/slides/06_qualification.html) and [target evidence checklist](../../../../ai-accelerator-software-course/labs/TARGET_EVIDENCE.md) to distinguish the executable virtual-device exercises from physical extensions.
