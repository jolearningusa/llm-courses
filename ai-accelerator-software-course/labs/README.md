# Functional lab kit

Python 3.9+ standard library only. Run from this directory; use the normal interpreter (assertions enabled, no `-O`). No package install, GPU, root access, kernel modules or firmware flashing is needed.

```sh
python3 -m unittest discover -s tests -v
python3 experiments.py scenarios
python3 experiments.py explore
python3 experiments.py fuzz --seed 17 --steps 300
python3 experiments.py report --output result.json
python3 execution_coverage.py --output coverage-new.json
PYTHONHASHSEED=0 python3 qualification_matrix.py --output qualification-new.json
```

`result.json` must not already exist; reports are created exclusively to avoid overwriting a previous run. Save raw JSON plus exact source/environment identity. The report contains model ticks, CPU execution provenance and explicit hardware/power-thermal `not_run` fields. A scope gate accepts only the supported functional evidence; this is a teaching decision, not a signed attestation or hardware qualification.

## Path through the labs

| Lab | Handout | Main evidence |
|---|---|---|
| L0 | [ABI and rejection](handouts/lab00_abi.md) | Binary descriptor and no-side-effect rejection |
| L1 | [Queues, DMA and completion](handouts/lab01_queue_dma.md) | FIFO order, visible output, lost IRQ and polling |
| L2 | [Firmware lifecycle](handouts/lab02_firmware.md) | Boot, fault, watchdog, reset and generations |
| L3 | [Runtime ownership](handouts/lab03_runtime.md) | Timeout/cancel/reap and buffer lifetime |
| L4 | [Fault injection and seeded testing](handouts/lab04_faults.md) | Reproducible adverse paths and invariant checks |
| L5 | [State exploration and AI patch](handouts/lab05_state_exploration.md) | Bounded counterexample and independent evaluator |
| L6 | [Target evidence and HIL plan](handouts/lab06_qualification.md) | Functional-only gate and honest target matrix |
| L7 | [Capstone release decision](handouts/lab07_capstone.md) | Baseline/change/negative control/reproduction package |

Read [ABI](ABI.md) before editing. `virtual_accelerator.py` is the deterministic reference. `experiments.py` supplies five scenarios, a small bounded ownership model, seeded state exercising and a report. Tests include deliberate rejection controls and correctness expectations independent of the report formatter. The model retains command history for instruction, so it is not a bounded-memory production service.

Seeded state exercising is not coverage-guided fuzzing. The abstract explorer is not an external formal tool or ISA memory model. Real target work starts with the [target evidence template](TARGET_EVIDENCE.md), a supported adapter and a recoverable designated test device. No physical telemetry or hardware timing is synthesized by the model.


The separate [L5 execution-evidence controls](handouts/lab05_state_exploration.md#execution-coverage-controls) use the same virtual worker to distinguish a legal run, no launch, invalid channels and observed failures. They add eight tests (32 total) and an executed reference report. They do not change `experiments.py`'s older functional gate or authenticate that report. GPU and Compute Sanitizer remain not-run.

[L6's qualification matrix](handouts/lab06_qualification.md#executed-matrix-and-policy-controls) adds nine policy tests (41 total) and an actual bytecode-build/CPU-model report. It retains unavailable device rows as not-run. Fabricated device records are policy controls only; no target adapter or physical experiment is implemented.

## Optional real Metal runtime boundary — 2026-11-14 UTC

The [executed Apple-Silicon pilot](../references/metal_lifetime_2026_11.md) is separate from the default Python model: three fresh-process runs,24 actual GPU completions, independent output/tail checks, host lifetime guards and logical cancellation followed by drain and known-good reuse. Shared buffer pointers are not physical DMA addresses. This is a bounded runtime functional result; physical qualification, driver/firmware/reset, sanitizers and fault injection remain not-run. See the note for tool/target identity, raw evidence and commands; do not promote it to the default physical acceptance gate.
