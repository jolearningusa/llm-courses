# L2 — Firmware lifecycle and generations

Companion to [the topic deck](../../slides/03_firmware_lifecycle.html). Run from `ai-accelerator-software-course/labs/` using Python 3.9+ with assertions enabled. CPU functional path; no root, GPU or flashing. Suggested 2–3 hours is provisional pending a student pilot.

## Baseline command

```sh
python3 experiments.py scenarios
```

## Tasks

1. Boot the default supported image/ABI. Reject an unsupported image or ABI; record FAULT and the required reset before retry.

2. Inject hang and follow the modeled watchdog to FAULT. Confirm buffers remain pinned while the worker may still own them.

3. Reset, boot again and deliver an old completion ticket. Confirm it is rejected and increments the stale-completion counter.

4. Reuse a cookie in the new generation, preserving separate ticket identity. Explain why a generation tag cannot itself stop real old DMA writes.

## Expected evidence and deliverable

Deliver the state/generation timeline, compatibility failure and old-event rejection. The firmware identifier is not a compiled image or secure-boot implementation.

Use [ABI](../ABI.md), [AI prompt boundaries](../AI_PROMPTS.md) and the [target evidence template](../TARGET_EVIDENCE.md). Keep simulated time, source findings and real measurements separate.
