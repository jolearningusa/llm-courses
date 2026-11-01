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
