# Computing systems lab kit

Released labs are **L0: resource budgets**, **L1: measurement and host data paths**, **L2: PyTorch execution/compiler**, and **L3: reduction references/CUDA exercise**. L0/L1 use Python 3.9+ and the standard library; L2 has a separate pinned Python 3.12/PyTorch environment. Analytical examples are labeled separately from measured CPU work. Later kits land through serial release units.

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
