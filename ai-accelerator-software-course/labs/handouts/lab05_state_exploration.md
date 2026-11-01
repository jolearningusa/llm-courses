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
