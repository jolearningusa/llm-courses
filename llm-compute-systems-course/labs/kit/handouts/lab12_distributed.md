# L12 — Distributed updates and exposed communication

**Released:** actual two-process CPU Gloo DDP correctness experiment and analytical ring model. **Unverified extension:** accelerator FSDP2/NCCL/RCCL memory and overlap. Python 3.12 / torch 2.9.1; local process communication is required. Estimated 5–8 hours, student pilot pending.

From `labs/kit/` with the pinned PyTorch environment:

```sh
python3 distributed_lab.py > ddp-cpu.json
python3 distributed_lab.py --rank-delay 0.02 > ddp-skew.json
python3 -m unittest discover -s tests -v
```

## A. Preserve training semantics

Inspect the complete reference before changing it. Each rank receives the same initialization and global data fixture, takes a disjoint 16-example shard and uses mean loss. Over three SGD steps, DDP averaged gradients must match a full 32-example reference (rtol 1e-5, atol 1e-6). The program asserts each parameter comparison. Explain why equal shard sizes make this correct. Student task: design weighted loss normalization for unequal token counts; do not apply an unweighted mean of means.

Record backend, process count, PyTorch version, data coverage and parameter error. The local CPU run validates DDP semantics, not FSDP or GPU performance. The program requires exactly two ranks and contains no dropout/BatchNorm; do not extend its equivalence claim to rank-dependent stochastic layers without controlling their behavior.

## B. Latency and skew

Compare raw maximum-rank step wall times with and without an injected 20 ms sleep on rank 1. These include CPU dispatch, autograd and update; startup and the alignment barrier are excluded. They cannot isolate collective duration. Reproduce the ring estimate for eight ranks and 256 MB/rank: 9.03 ms under assumed 5 µs/step and 50 GB/s. Compare small/large messages. Keep this estimate separate from measured Gloo timings; model parameters are classroom assumptions.

## C. GPU extension or analytical alternative

At fixed effective batch, compare DDP with FSDP2 on 2–8 supported accelerators. Follow the selected-release FSDP2 tutorial: shard blocks then root, record mixed-precision and reshard policy, initialize compatible optimizer state, and verify updates. Record persistent and peak allocated/reserved memory, group size, all-gather/reduce-scatter bytes, rank placement and maximum-rank step time. Vary group/bucket size once and inspect a trace for exposed waits and rank skew. Run NCCL on supported NVIDIA hardware or RCCL on supported AMD hardware; record software/driver/fabric versions.

Without accelerators, use L0 state accounting plus the lecture timeline: show ideal per-rank persistent storage, add gathered layers/activations/prefetch buffers, identify a possible peak, and derive an exposed-communication example. Explicitly label the result analytical. No accelerator launcher or benchmark is advertised as validated here.

Submit both CPU JSON files, correctness explanation, accounting table and [report](../../report_template.md). Instructor diagnostic: a faster step after reducing effective batch is not a matched training improvement; a Gloo pass is not an NCCL/RCCL pass.
