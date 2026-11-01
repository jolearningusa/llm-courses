# L7 — Capstone release decision

Companion to [the topic deck](../../slides/07_capstone.html). Run from `ai-accelerator-software-course/labs/` using Python 3.9+ with assertions enabled. CPU functional path; no root, GPU or flashing. Suggested 2–3 hours is provisional pending a student pilot.

## Baseline command

```sh
python3 -m unittest discover -s tests -v
```

## Tasks

1. Choose one change: descriptor rejection, queue ownership, lost-interrupt recovery, watchdog/reset or bounded verification. Freeze the claim and baseline first.

2. Preserve an AI prompt/diff provenance record if used, plus independent correctness and deliberately failing controls. Keep unsupported behavior explicit.

3. Run held-out seeds/event cases and minimize a failure. Reproduce the final result from a clean source checkout with exact commands.

4. Present the contract, state/dataflow diagram, evidence matrix and retain/revise/reject decision. A null result with a clear explanation is valid.

## Expected evidence and deliverable

Deliver code/configuration, raw records, independent tests, one adverse case, rerun instructions and a short presentation. Physical/driver-performance claims need a separately qualified target.

Use [ABI](../ABI.md), [AI prompt boundaries](../AI_PROMPTS.md) and the [target evidence template](../TARGET_EVIDENCE.md). Keep simulated time, source findings and real measurements separate.
