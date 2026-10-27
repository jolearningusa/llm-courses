# L2 — Follow eager execution into compilation

**Environment:** a separate Python 3.12 environment with `pip install -r requirements-pytorch.txt`. This lab pins PyTorch 2.9.1 for a reproducible CPU exercise, rather than claiming the newest release. Compiler execution also needs a working platform C++ toolchain. See the kit's release record for actually tested paths. GPU variants need a separate validated environment. Estimated student time: 4–6 hours, pending pilot.

## A. Eager execution and saved state

From the kit directory run:

```sh
python3 pytorch_lab.py --training > eager-training.json
python3 -m unittest discover -s tests -v
```

The control compares two copies of the same small Linear/LayerNorm/GELU block at two batch sizes. Inspect the FX graph and saved-tensor records. Trace the linear forward/backward shapes. Explain why summing saved logical bytes does not measure unique live storage or allocator peak. The script clears gradients each repetition, validates parameter/input gradients and excludes optimizer updates.

## B. Compiled execution

```sh
TORCH_LOGS=graph_breaks,recompiles python3 pytorch_lab.py --compile --training > compiled-training.json 2> compiler.log
python3 pytorch_lab.py --compile > compiled-forward.json
python3 pytorch_lab.py --compile --training --dynamic > dynamic-training.json
```

Inspect logs rather than inferring compile counts from time alone. Compare first use for each batch with warm raw samples. A later process may reuse compiler artifacts; use a fresh documented cache location when measuring cold compilation. Record the cache policy and toolchain. Do not interpret the script's first-use timer as a guarantee of an empty on-disk compile cache.

## C. Diagnose one change

Introduce a layout change, supported shape change, or a Python branch that capture cannot handle. Keep a correct eager reference and distinguish graph breaks, guard-triggered recompilation and ordinary slow execution. Full-graph mode may reject unsupported capture; report that outcome rather than suppressing the error and claiming the compiled variant passed.

## D. Explain the result

What changed in the graph and actual implementation? Is the tiny workload large enough to amortize compilation? Does forward+backward differ from forward-only? Which costs are included in both timing paths? Compare equivalent initial weights and numerical tolerances before reporting speedup. The CPU scalar checksum is included in both warm paths and may matter for such a small workload.

Use the [report template](../../report_template.md), include JSON/logs, actual versions, raw samples and an adverse case. A validated CPU compiler path is not a measured CUDA result or a portable deployment package. AOTInductor and ExecuTorch are separate later labs.
