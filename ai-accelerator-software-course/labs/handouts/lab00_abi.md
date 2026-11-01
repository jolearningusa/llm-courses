# L0 — ABI and rejection

Companion to [the topic deck](../../slides/01_interfaces.html). Run from `ai-accelerator-software-course/labs/` using Python 3.9+ with assertions enabled. CPU functional path; no root, GPU or flashing. Suggested 2–3 hours is provisional pending a student pilot.

## Baseline command

```sh
python3 -m unittest discover -s tests -p test_virtual_accelerator.py -v
```

## Tasks

1. Read the 24-byte descriptor in ABI.md. Encode/decode one valid packet and annotate every field, width and endianness. Explain why a buffer handle is not a DMA bus address.

2. Mutate ABI, opcode, reserved bits, generation, length and cookie independently. Predict rejection before running the corresponding tests.

3. Compare queue, command records and pin counts before/after rejection. A rejected command must consume none of those resources.

4. Read the Linux DMA guide address section and list the APIs/assumptions the functional model omits.

## Expected evidence and deliverable

Valid packet roundtrip passes; malformed/bounds/unsupported packets fail without resource changes. Deliver a field table, two minimal rejected packets and the resource ledger.

Use [ABI](../ABI.md), [AI prompt boundaries](../AI_PROMPTS.md) and the [target evidence template](../TARGET_EVIDENCE.md). Keep simulated time, source findings and real measurements separate.
