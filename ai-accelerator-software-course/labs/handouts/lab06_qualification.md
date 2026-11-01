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

Use [ABI](../ABI.md), [AI prompt boundaries](../AI_PROMPTS.md) and the [target evidence template](../TARGET_EVIDENCE.md). Keep simulated time, source findings and real measurements separate.
