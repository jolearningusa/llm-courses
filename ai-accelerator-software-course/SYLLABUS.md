# Syllabus — AI accelerator software systems

Eight weeks, one topic seminar and one lab session per week. Suggested 4–6 hours of practice weekly is a planning estimate pending a student pilot. The model/target split is explicit: a functional CPU path supports protocol experiments; target paths require device documentation, tools and a recoverable test setup.

| Week | Topic | Lab evidence |
|---|---|---|
| 1 | Software stack, ABI and trust/ownership boundaries | Reject unsupported ABI/opcode/shape before consuming queue/buffer resources |
| 2 | Queues, DMA, memory visibility and completion | Trace immutable descriptors, bounded queues and complete output visibility |
| 3 | Firmware boot, compatibility and state transitions | RESET/READY/FAULT lifecycle, compatibility rejection and generation ledger |
| 4 | Driver/runtime lifecycle | Distinguish queued cancellation, in-flight work, timeout and safe reclaim |
| 5 | Verification, fuzzing and fault injection | Replay fixed seeds, preserve failing inputs, test DMA fault/lost interrupt |
| 6 | State models and AI-assisted changes | Explore bounded event orderings; compare a proposed lifetime change with invariants |
| 7 | HIL, telemetry and release evidence | Produce a dry-run target matrix and rollback record; optional separately qualified board run |
| 8 | Capstone integration and critique | Reproduce one failure/fix and defend an evidence-scoped release decision |

## Six topic units

1. Interface contracts: framework/compiler/HAL/runtime/driver/firmware boundaries; ABI, addresses and command schema.
2. Execution and movement: descriptor ownership, queues, DMA lifetimes, publication/completion, interrupts and polling.
3. Firmware lifecycle: boot, capabilities, image compatibility, watchdog, quiesce/reset and recovery generations.
4. Host runtime/driver: admission, mapping/pinning concepts, cancellation, timeout, cleanup and concurrency assumptions.
5. Verification engineering: numerical/protocol oracles, valid/invalid inputs, fault injection, differential/metamorphic tests and bounded state exploration.
6. Target qualification: build-only versus simulation versus hardware execution, resource/thermal observations, artifact identity and staged rollback.

## Capstone contract

Choose one bounded change to the functional reference or one documented target adapter. Submit a frozen contract, baseline, AI-assistance provenance, independent tests, deliberately failing control, minimized failure, rerun commands and raw outcomes. State which properties were tested, modeled, source-reviewed or not run.

A simulation path must not claim actual DMA coherency, interrupt timing, device performance, secure boot or physical safety. A board path identifies exact SDK/firmware/hardware revisions, a recovery procedure and supported instrumentation. Firmware flashing is outside the default lab and needs a designated recoverable test setup.

Use a 100-point rubric: contract/baseline 20; correctness and failure detection 30; mechanism/state explanation 20; reproducibility and negative controls 20; release/limitation decision 10. A broken invariant blocks any performance claim. Do not grade students by a fixed speedup on different devices.
