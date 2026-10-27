# L3 — Reductions, CUDA execution and operator contracts

**Released/validated:** Python standard-library numerical references and tests. **Provided but unverified:** standalone CUDA float32 RMSNorm example. The authoring host has no CUDA compiler/device; no CUDA build, execution, sanitizer result or bandwidth measurement is claimed. Expected student time: 5–8 hours on a supported CUDA host, pending pilot.

## A. Read and test the numerical contract

From the kit directory:

```sh
python3 reductions.py --cols 257
python3 -m unittest discover -s tests -v
```

Explain stable softmax for logits [1000,1001]. Verify RMSNorm for zero input and lengths 1,33,257 with nonuniform weights. State finite-input, positive-epsilon and shape requirements. The serial reference is a correctness oracle, not an optimized CPU baseline or a guarantee that extreme magnitudes cannot overflow.

## B. CUDA correctness exercise

On a supported NVIDIA CUDA host:

```sh
nvcc -O2 -std=c++17 cuda/rmsnorm.cu -o /tmp/course-rmsnorm
/tmp/course-rmsnorm
compute-sanitizer --tool memcheck /tmp/course-rmsnorm
compute-sanitizer --tool racecheck /tmp/course-rmsnorm
```

Exit 77 means unavailable hardware, not a passed check. The provided source uses exactly 256 threads and fixed example dimensions. Do not change the launch size without changing its shared-memory/tree assumptions. Inspect error checks, tail bounds and barrier participation. Extend the harness to several row counts and dimensions, reject unsupported layouts, and preserve the declared numerical tolerance. Record compiler/device/runtime and all real validation results.

## C. Improve the baseline

Measure a correctly synchronized warm kernel scope separately from allocation/copies. Compare a legal alternative reduction/layout, then compare with a credible framework/library path at matching semantics. Explain source-level rereads versus HBM traffic and the row-count limit on parallelism. Provide one shape where the change loses. Do not extrapolate a bandwidth result from the Python reference.

## D. Framework/training extension

Using the current official custom-operator tutorial, define schema, shape/dtype/device checks and metadata behavior. Register and test backward against an eager reference, including input and weight gradients, before calling the operator training-ready. Then check a required compile/export path independently. This extension is student work; the standalone CUDA example is not advertised as an integrated PyTorch operator.

## Submit

Include the [report](../../report_template.md), numerical checks, launch/input contract, source change, timings and profiler/sanitizer evidence where available. Without CUDA hardware, complete A and a line-by-line B analysis and label C/D unverified. A numerical or CPU pass never stands in for GPU memory/race validation.
