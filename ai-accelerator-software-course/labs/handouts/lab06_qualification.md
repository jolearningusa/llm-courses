# L6 — Qualification evidence and HIL plan

Companion to [the topic deck](../../slides/06_qualification.html). Run from `ai-accelerator-software-course/labs/` using Python 3.9+ with assertions enabled. CPU functional path; no root, GPU or flashing. Suggested 2–3 hours is provisional pending a student pilot.

## Baseline command

```sh
python3 experiments.py report --output result.json
```

## Tasks

1. Run in a fresh output path. Inspect execution, model_tick, source hashes and explicit hardware/power-thermal not_run fields.

2. Compare the functional gate with the hardware gate. A CPU model pass must not promote a real target.

3. Fill TARGET_EVIDENCE.md with known facts; leave all unavailable board/SDK/physical execution rows not-run. Design one-board reservation, logging, recovery and rollback.

4. If a designated supported target is actually available, define a separate reviewed adapter experiment before any deployment. Distinguish real measurements from the model report.

## Expected evidence and deliverable

Deliver the complete status matrix, logs and a reproducible dry-run plan. Filling a manifest or quoting a vendor peak does not validate a device.

## Executed matrix and policy controls

```sh
PYTHONHASHSEED=0 python3 qualification_matrix.py --output qualification-new.json
python3 -m unittest discover -s tests -v
```

Use a new output filename. The runner compiles three course Python files to temporary CPython bytecode, hashes the artifacts, then executes the existing functional model. This is neither a firmware build nor an ISA emulator. The [reference report](../examples/qualification_report.json) retains the full model report and a separate decision for each stage. Model ticks remain simulated time.

| Actual evidence | Execution stage | Decision |
|---|---|---|
| Three bytecode artifacts and compilation log | `built_only` | Incomplete for model execution |
| Scenarios, invariant checks and bounded exploration | `simulator_run` | Passed for the declared CPU model checks |
| Device functionality, recovery and power/thermal | `not_run` | Physical decision incomplete; no board, adapter or target image |

Record **stage** separately from each required check's **outcome**: `passed`, `failed`, `unsupported` or `not_run`. A missing board is `not_run`; `unsupported` requires an explicit capability limitation, not a guessed hardware property. These are course terms, not a lossless Twister export schema. A real adapter must preserve its native statuses and raw logs too, including build errors, timeouts and skips.

The harness declares expected source/build/target/run identity and required checks before evaluating results. Missing checks, mismatched identity or absent evidence references invalidate a record. Failed checks remain visible even when evidence is also invalid. Unsupported or unexecuted requirements block acceptance; they are not silently removed from the denominator.

The nine new tests include **fabricated policy records** for a complete positive control, absent board despite claimed device execution, build/simulator promotion attempts, failed checks with missing logs, unsupported/not-run reasons, identity mismatch and malformed/missing checks. Their passing result validates these policy cases, not any device. The actual report has no device-run row.

Exercise: remove a required check, substitute another build/run identity, then claim a device run with `available=False`. Explain each refusal. For a future target, freeze the required properties and instruments, obtain the supported build/board identity, execute bounded known-good work and recovery, and archive observed logs before asking for physical acceptance. Physical criteria are target-specific; the three worksheet properties are not a universal silicon qualification suite.

The JSON, hashes and evidence paths are reproducibility aids. The policy does not read or authenticate those referenced logs, isolate malicious code, reserve a board, run Twister, flash firmware, measure telemetry or authorize deployment. See the [source and execution note](../../references/qualification_matrix_2026_11.md).

Use [ABI](../ABI.md), [AI prompt boundaries](../AI_PROMPTS.md) and the [target evidence template](../TARGET_EVIDENCE.md). Keep simulated time, source findings and real measurements separate.
