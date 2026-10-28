# L14 — Export, compile, lower and reload real artifacts

**Validated:** macOS arm64, Python 3.12.14, torch 2.9.1 AOTInductor CPU and a separate ExecuTorch 1.0.0 + torch 2.9.1 XNNPACK CPU environment. **Not validated:** CUDA, TensorRT, native C++-only application, edge device or quantized backend. Estimated 5–8 hours, student pilot pending.

## A. Separate the environments

Use the pinned PyTorch environment for `aoti`. For ExecuTorch create a **separate Python 3.12 environment**, activate it and install `requirements-executorch.txt` or the full `requirements-executorch-lock.txt` authoring lock. The latter includes platform-specific/transitive exporter dependencies; it is not a universal cross-platform lock. Do not combine it with every other lab/serving SDK.

The tested ExecuTorch wheel supplied `flatc` in its environment's bin directory. The script checks that bundled location if `flatc` is not on PATH. Its optional TorchAO C++ extensions were disabled by the torch-version check; **no quantized TorchAO path is claimed**. The runtime also reported that its InternalConsistency verifier was unavailable in that build. Numerical and explicit input-contract checks below were performed; they do not replace that missing internal verifier.

## B. Produce and measure

From `labs/kit/`, choose result directories outside the course source, for example on macOS/Linux:

```sh
# Pinned PyTorch environment
python3 deployment_lab.py --backend aoti --output /tmp/course-l14-aoti > aoti.json
python3 deployment_lab.py --backend aoti --output /tmp/course-l14-aoti --load-existing > aoti-load.json

# Separate ExecuTorch environment
python3 deployment_lab.py --backend executorch --output /tmp/course-l14-et > executorch.json
python3 deployment_lab.py --backend executorch --output /tmp/course-l14-et --load-existing > executorch-load.json
```

The complete reference uses an unquantized synthetic MLP, contiguous float32 CPU input shape (3,8). It records export/build/load/first-execution and ten warm wall samples, plus eager/exported baselines. These tiny host-call measurements are a pipeline demonstration, not a stable LLM speed benchmark. Include environment, compiler cache state and timing scope; do not compare archive sizes as a universal runtime footprint ranking.

Inspect the native AOTInductor package entries, and the ExecuTorch delegate count (one XNNPACK segment in the authoring run). `fixture.pt` contains only the produced input/expected tensors for a trusted local numerical replay. AOT compiled `.pt2` and `torch.export.save` graph `.pt2` have different producers/loaders despite the shared extension. The example `.pte` embeds its data; it produces no separate `.ptd`.

## C. Reload and enforce the contract

Run each `--load-existing` command in a fresh process. This path loads the artifact/fixture without constructing the model. Require output rtol 1e-5/atol 1e-6 and rejection of the wrong shape. The explicit checked loader rejects non-CPU, non-float32, non-contiguous or non-(3,8) input **before** native execution; do not bypass this guard to probe an arbitrary compiled artifact. Export also rejects a changed static batch, with exception type varying by release.

Run the unit suite in the PyTorch environment. It includes exported-graph save/load equivalence and a guard negative control that proves an invalid input never calls the execution function. Actual compile/runtime validation is the separate CLI experiment, not hidden in every unit-test run.

## D. Student extension and report

Declare bounded dynamic shapes and re-export, or select one supported alternate backend. Inspect partition/fallback and buffer/workspace lifetimes, validate held-out eligible inputs and an unsupported case, and measure cold/warm costs separately. Use target-specific precision/quality rules if quantizing. Consult current ExecuTorch CUDA-backend docs for the AOTInductor/Triton relationship without assuming that the older pinned CPU environment validates it.

Submit artifacts/producer-loader manifests, raw JSON, delegate/partition evidence and [report](../../report_template.md). Instructor diagnostic: an export succeeding does not prove target execution, a file suffix does not prove format compatibility, and two CPU packages do not establish GPU or edge performance.

Recorded authoring runs: [AOTInductor CPU JSON](../examples/l14_aoti_cpu.json) and [ExecuTorch CPU JSON](../examples/l14_executorch_cpu.json). They are actual small-fixture host measurements with stated scope, not performance targets or portable benchmark scores.
