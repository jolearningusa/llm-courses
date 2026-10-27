# Computing systems lab kit

Released labs are **L0: resource budgets** and **L1: measurement and host data paths**. It uses Python 3.9 or later and only the standard library. Its numbers are analytical estimates, not GPU measurements. Later lab implementations will be added as their serial release units land.

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
