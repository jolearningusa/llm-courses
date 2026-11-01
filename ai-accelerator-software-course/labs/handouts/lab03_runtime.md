# L3 — Host runtime ownership

Companion to [the topic deck](../../slides/04_runtime_lifetimes.html). Run from `ai-accelerator-software-course/labs/` using Python 3.9+ with assertions enabled. CPU functional path; no root, GPU or flashing. Suggested 2–3 hours is provisional pending a student pilot.

## Baseline command

```sh
python3 -m unittest discover -s tests -p test_virtual_accelerator.py -v
```

## Tasks

1. Trace host write → submit/pin → complete/observe → reap → read/free. Check pins after every transition.

2. Attempt write, read and free before safe reap; confirm rejection. Even a completed command remains pinned until the modeled host observes/reaps it.

3. Compare queued cancellation with in-flight cancellation. Only the queued case is supported. A client timeout is not device cancellation.

4. Ask an assistant for a bounded cleanup patch using AI_PROMPTS.md. Keep tests fixed and show that early-release proposals are rejected.

## Expected evidence and deliverable

Deliver an ownership table, a failing early-release patch or trace, one accepted legal edit and a cancellation/timeout decision. No driver/kernel memory ordering is proven.

Use [ABI](../ABI.md), [AI prompt boundaries](../AI_PROMPTS.md) and the [target evidence template](../TARGET_EVIDENCE.md). Keep simulated time, source findings and real measurements separate.
