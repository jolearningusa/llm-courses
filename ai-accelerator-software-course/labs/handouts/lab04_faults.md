# L4 — Fault injection and seeded state testing

Companion to [the topic deck](../../slides/05_verification.html). Run from `ai-accelerator-software-course/labs/` using Python 3.9+ with assertions enabled. CPU functional path; no root, GPU or flashing. Suggested 2–3 hours is provisional pending a student pilot.

## Baseline command

```sh
python3 experiments.py fuzz --seed 17 --steps 300
```

## Tasks

1. Reproduce the seed twice and compare accepted/rejected counts and action populations. This is a state exerciser, not coverage-guided fuzzing.

2. Test dma_error, drop_irq and hang separately. Check destination contents, terminal state, pins and reset/recovery requirements.

3. Run allocation and outstanding-slot limits. Keep ordinary backpressure distinct from a watchdog fault or numerical failure.

4. Add one invalid descriptor family and one valid control. Preserve a minimal reproducer and explain which invariant, not merely which exception, detects the issue.

## Expected evidence and deliverable

Deliver seed/configuration, raw results, failure taxonomy and minimized test. Rejection counts alone do not measure useful coverage.

Use [ABI](../ABI.md), [AI prompt boundaries](../AI_PROMPTS.md) and the [target evidence template](../TARGET_EVIDENCE.md). Keep simulated time, source findings and real measurements separate.
