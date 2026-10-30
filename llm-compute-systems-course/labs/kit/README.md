# Computing systems lab kit

Released paths are indexed below, with validation limits in each handout. Standard-library exercises use Python 3.9+; PyTorch paths use the separate pinned Python 3.12 environment. Analytical fixtures and measured CPU work are labeled separately. GPU build/run claims require their own recorded validation.

From this directory:

```sh
python3 budgets.py --demo
python3 measurement.py --repeats 15 --warmup 3
python3 budgets.py --m 1 --n 4096 --k 4096 --compute-tflops 100 --bandwidth-tbps 1
python3 -m unittest discover -s tests -v
```

Read the [L0 handout](handouts/lab00_budgets.md), the [workload primer](../../notes/workload_primer.md) and the [report template](../report_template.md). The two rates in the example command are explicit classroom assumptions. Do not label the output as performance measured on Rubin or any other device.

## Functions and scope

`gemm` counts `2MNK` operations and one ideal read of each input plus one output write. It computes an overlapping math/memory lower bound using decimal TFLOP/s and TB/s. It excludes launch latency, repeated transfers, collectives and dependencies between phases.

`kv_bytes` counts logical MHA/GQA K/V storage for variable retained sequence lengths. It excludes metadata, padding, quantization scales, replication and workspace, and is not an MLA formula.

`training_state` separates parameter, gradient and optimizer/master-copy storage under simplified ZeRO-style stage accounting. It estimates persistent per-rank state; transient gathers, activations and padding remain additional. A floating result represents ideal even partitioning, not a real allocator's granularity.

## Validation and release record

The analytical unit checks and CLI demo ran on the course-authoring CPU host with Python 3.9.6. No GPU is required or validated in this release. Tests cover the worked examples, unit conversion, shape-dependent bottleneck transition, variable context lengths, sharding and invalid inputs. Lab duration remains a planning estimate until a student pilot is recorded.

## L1 measurement path

[Measurement handout](handouts/lab01_measurement.md) pairs actual CPU wall timing with analytical interval/pipeline exercises. `measurement.py` records raw samples and environment, uses explicit warmup, and reports nearest-rank p95. Its interval-union helper does not infer causal bottlenecks. The CLI ran on Python 3.9.6 and its accounting/warmup checks passed; no CUDA timeline was collected.

## L2 pinned PyTorch environment

Create a separate Python 3.12 environment and install `requirements-pytorch.txt` (direct dependency) or `requirements-pytorch-lock.txt` (the resolved CPU authoring environment). Read [L2](handouts/lab02_pytorch.md). The standard-library labs do not require PyTorch. Without it, the L2 tests explicitly skip.

Validated on macOS arm64, Python 3.12.14 and PyTorch 2.9.1: eager forward/backward, Inductor compiled forward/backward, two batch sizes and dynamic-shape execution, with output/input/parameter-gradient checks. Cold first use and warm samples are recorded separately; no CUDA/ROCm performance claim or student-duration pilot is made. PyTorch's optional NumPy bridge is not used by this kit.

## L3 reductions and CUDA example

Read [L3](handouts/lab03_reductions.md). `reductions.py` and its numerical/invalid-input checks ran on the CPU host. `cuda/rmsnorm.cu` is an educational float32 standalone source example: no CUDA compiler, GPU execution or sanitizer run was available here. The integration/backward task is explicitly a student extension, not a supplied training operator.

## L4 GEMM and pipeline study

[L4](handouts/lab04_gemm.md) uses `gemm_lab.py` in the pinned PyTorch environment for actual CPU matrix/shape measurements and K-tiling correctness. It includes ragged expert groups with an empty expert. Python tiling/group dispatch is explicitly pedagogical, not an optimized CUDA/ROCm implementation. CUDA/library pipeline benchmarking remains a hardware extension not validated on this host.

## Lab index

- [L0 — Predict resource use before profiling](handouts/lab00_budgets.md)
- [L1 — Measurement and host data paths](handouts/lab01_measurement.md)
- [L2 — Follow eager execution into compilation](handouts/lab02_pytorch.md)
- [L3 — Reductions, CUDA execution and operator contracts](handouts/lab03_reductions.md)
- [L4 — GEMM shapes, tiling and pipelines](handouts/lab04_gemm.md)
- [L5 — Online attention and matched backend comparison](handouts/lab05_attention.md)
- [L6 — Precision contracts and measured numerical behavior](handouts/lab06_precision.md)
- [L7 — State ownership and inference-cache mechanisms](handouts/lab07_cache.md)
- [L8 — Replay scheduling and budget disaggregation](handouts/lab08_scheduling.md)
- [L9 — Route, group and combine expert work](handouts/lab09_moe.md)
- [L10 — Speculative correction and agent-wait policies](handouts/lab10_agent_policy.md)
- [L11 — Export, compile, lower and reload real artifacts](handouts/lab11_deployment.md)
- [L12 — Distributed updates and exposed communication](handouts/lab12_distributed.md)
- [L13 — Parallel layouts, recomputation and source tracing](handouts/lab13_parallelism.md)
- [L14 — Recover the computation, not only the weights](handouts/lab14_recovery.md)
- [L15 — Rubin/AMD dossier and whole-system efficiency](handouts/lab15_vendor_efficiency.md)

## L5 attention execution

[L5](handouts/lab05_attention.md) provides CPU full/tiled single-head self-attention, causal/noncausal checks, tail handling and gradient comparisons. `attention_lab.py` ran in the pinned PyTorch environment. Its Python autograd graph is not a fused training-memory implementation, and its CPU wall timings do not establish GPU backend performance.

## L6 precision

[L6](handouts/lab06_precision.md) includes actual packed int4/int8 array storage and a CPU BF16 autocast loss/gradient comparison. These numerical experiments do not establish TorchAO/TE GPU acceleration or LLM task quality.

## L7 serving and caches

[L7](handouts/lab07_cache.md) tests a logical full-block prefix model with identity, shared ownership, pressure and cancellation. It stores no KV tensors. A vLLM/llama.cpp comparison worksheet is provided; no engine run is claimed.

## L8 scheduling and disaggregation

[L8](handouts/lab08_scheduling.md) replays a finite workload under explicit virtual-time costs and compares chunked/unbounded prefill. The transfer calculator is analytical; it is not a full split-cluster simulator or measured disaggregated deployment.

## L9 MoE

[L9](handouts/lab09_moe.md) validates local routed expert combine. It does not substitute for a distributed dispatcher.

## L10 speculation and agent policy

[L10](handouts/lab10_agent_policy.md) validates exact one-step categorical correction and samples a synthetic distribution. Its wait-policy accounting is analytical, with explicit identity and byte-second boundaries; no engine/model throughput is claimed.

## L11 deployment

[L11](handouts/lab11_deployment.md) builds and loads actual CPU AOTInductor .pt2 and ExecuTorch .pte artifacts, including fresh-process reload, output checks and shape/stride/dtype guard rejection. ExecuTorch uses a separate pinned environment; CUDA, edge devices and quantized native extensions are not validated.

## L12 distributed training

[L12](handouts/lab12_distributed.md) launches two actual CPU Gloo DDP processes and compares three updates against a global-batch reference. A separate ring model is analytical. FSDP2 and accelerator collective/overlap measurements require additional hardware validation.

## L13 parallelism

[L13](handouts/lab13_parallelism.md) separates an analytical eight-rank layout model, actual CPU checkpoint output/gradient checks and pinned TorchTitan/Megatron source inspection. Multi-GPU feasibility and timing remain unverified.

## L14 recovery

[L14](handouts/lab14_recovery.md) validates actual CPU checkpoint/resume with a missing-RNG negative control. It does not substitute for DCP failure testing.

## L15 vendor and whole-system study

[L15](handouts/lab15_vendor_efficiency.md) uses a source-grounded dossier and analytical disjoint phase ledger. It preserves conflicting published rate scopes and separates dense/sparse precision. No Rubin/AMD or power benchmark is claimed.

## Capstone and release

The [capstone package](capstone/CAPSTONE.md) supplies three engineering paths plus an analytical option, proposal manifest, critique exercise and rubric. [Release validation](../../VALIDATION.md) records author checks separately from student/hardware pilots.

## Optional inference bottleneck diagnosis

The [diagnosis worksheet](inference_bottleneck_worksheet.md) connects L1/L8/L10 with deck 36: identify weight/KV traffic, compute, host gaps, sampling/beam-state work and queues. It is a profiling protocol with a source/analytical path, not a new validated GPU harness.
