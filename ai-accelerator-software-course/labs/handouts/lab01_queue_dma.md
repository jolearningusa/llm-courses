# L1 — Queues, DMA and completion

Companion to [the topic deck](../../slides/02_queues_dma.html). Run from `ai-accelerator-software-course/labs/` using Python 3.9+ with assertions enabled. CPU functional path; no root, GPU or flashing. Suggested 2–3 hours is provisional pending a student pilot.

## Baseline command

```sh
python3 experiments.py scenarios
```

## Tasks

1. Run copy and add-one scenarios. Check exact bytes, including modulo-256 wraparound and a partial-copy tail preserved by the unit test.

2. Locate submit, start, output_visible, completion and reap in the trace. Prove output publication precedes completion in this model.

3. Compare interrupt delivery with drop_irq. Without polling the wait times out even when device work is done. Poll, reap and verify output.

4. Queue two independent commands. Check FIFO order and the timeout needed for both service intervals; do not assume the first command and whole queue have the same deadline.

## Expected evidence and deliverable

Deliver annotated traces, expected output bytes, one lost-interrupt recovery and one capacity-rejection explanation. Model ticks are event ordering, not hardware latency.

Use [ABI](../ABI.md), [AI prompt boundaries](../AI_PROMPTS.md) and the [target evidence template](../TARGET_EVIDENCE.md). Keep simulated time, source findings and real measurements separate.
