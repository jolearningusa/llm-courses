# L5 — Bounded state exploration and AI changes

Companion to [the topic deck](../../slides/05_verification.html). Run from `ai-accelerator-software-course/labs/` using Python 3.9+ with assertions enabled. CPU functional path; no root, GPU or flashing. Suggested 2–3 hours is provisional pending a student pilot.

## Baseline command

```sh
python3 experiments.py explore
```

## Tasks

1. Read the abstract state: phase and buffer-live flag. Enumerate enabled submit/start/cancel/complete/release transitions.

2. Run safe and deliberately unsafe policies. Expect no counterexample within the safe model scope and a submit → release witness for the unsafe policy.

3. Reduce the depth bound to one in a Python call to explore. Explain why failure to find the two-action bug is inconclusive.

4. Use an assistant to suggest a changed state rule and an independent test. Freeze the ownership property; compare both policies without editing the acceptance criterion.

## Expected evidence and deliverable

Deliver states/depth, the counterexample, a legal passing control and an explanation of missing weak-memory/concurrency behavior. This is not an external formal-tool proof.

Use [ABI](../ABI.md), [AI prompt boundaries](../AI_PROMPTS.md) and the [target evidence template](../TARGET_EVIDENCE.md). Keep simulated time, source findings and real measurements separate.


## Execution coverage controls

Run the ACC-1 extension from `labs/`, still using only Python 3.9+ and the standard library:

```sh
python3 -m unittest discover -s tests -v
python3 execution_coverage.py --output /tmp/coverage-new.json
```

The output path must be new. CLI completion means the report was produced, not that every control passed: the tests require the declared failed/invalid outcomes. [The executed reference report](../examples/execution_coverage_report.json) records all nine named controls, raw modeled traces, separate coverage/diagnostic channels, numerical outcome, Python/platform and source hashes. It reports **1 passed, 2 failed, 6 invalid** under this declared CPU contract; these are constructed control outcomes, not statistical error rates. The original 24 tests plus eight new evaluator tests all ran successfully on Python 3.9.6 on 2026-11-08.

### Who owns the evidence?

`execution_coverage.py` creates the `Device`, nonce, literal expected output and channel records. Start/completion counts come from that owned model's trace for the designated four-byte copy, not a candidate's launch claim. The exact workload requires one start and one completion, the expected target ID and matching per-run nonce; extra/missing launches, wrong targets, boolean/float counts, duplicate JSON fields and replayed records are rejected. The parser checks schema as well as presence.

A no-launch control returns the right literal output and a fabricated stdout claim of 1,000 launches. Its numerical dimension passes but its coverage is invalid. `classify()` has no candidate-stdout argument. Other controls delete/corrupt the harness channel **after** running the virtual copy, exercising a transport failure separately from no execution.

This is an in-process protocol demonstration using only course-owned controls. It does not load arbitrary candidate modules, establish a process/container boundary, sign reports or authenticate an imported JSON document. The nonce binds records within this run; publishing it in a teaching report is not a hardware attestation. Keep the trusted observer independent of candidate code in a later real runner.

### Keep failures and invalid evidence separate

| Control | Numerical dimension | Coverage / diagnostics | Overall CPU gate |
|---|---|---|---|
| Legal copy | Right output | One observed start/completion; valid empty findings | Passed |
| No launch + fake stdout | Right hardcoded output | Zero starts/completions | Invalid |
| Missing / malformed / replayed coverage | Right output | Coverage transport invalid | Invalid (three controls) |
| Missing diagnostics | Right output | Coverage valid, diagnostic record absent | Invalid |
| Wrong returned output | Wrong output | Coverage valid, no model finding | Failed |
| Modeled DMA error | Wrong output | Coverage valid; explicit error completion | Failed |
| Bounds rejected before submit | No output | Zero starts; bounds finding preserved | Invalid, with failed diagnostic dimension |

Overall `invalid` takes precedence when required evidence is unavailable; diagnostic/correctness failures remain visible in their own fields. A known error must never become an accepted result. The bounds control attempts a five-byte request against four-byte buffers, which the model rejects before launch. It does **not** execute an out-of-bounds memory access or validate real memcheck detection. An error completion can count as execution coverage while still failing correctness/diagnostics; `completed` does not mean successful.

### Exercises and later target acceptance

1. Predict each dimension before reading the reference report. Explain why an empty finding list alone cannot establish execution.
2. Add a target-identity mismatch and replayed diagnostic record to a local test without relaxing the acceptance contract. Keep no-launch and legal controls in the test population.
3. Keep the original safe/unsafe state exploration above: execution evidence cannot replace the ownership invariant or a bounded counterexample.
4. For a later GPU adapter, pin target/tool/harness versions, independently observe the intended launched workload, capture required tool summaries/logs and retain failed/invalid/time-out runs. Include a supported known-bad access control and a known-good kernel. Neither was run on a GPU here. A clean memory check still does not establish race/synchronization/physical correctness.

The reading is [ComputeEval at revision 00fbad9](https://github.com/NVIDIA/compute-eval/blob/00fbad91b63957a002b501f33a06510cd25beaae/README.md#compute-sanitizer), whose opt-in sanitizer path distinguishes execution evidence from candidate stdout and preserves invalid runs. This CPU protocol is an adaptation of that idea, not a ComputeEval installation or reproduction. NNSmith's valid-graph/differential testing remains separate prior art; a new wrapper is not by itself a verification novelty.
