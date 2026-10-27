# L4 — GEMM shapes, tiling and pipelines

**Released/validated:** CPU PyTorch shape study and numerical K-tiled reference in the L2 pinned environment. **Hardware extension:** CUDA/CUTLASS/cuBLASLt or AMD hipBLASLt/CK implementation study and benchmarking; not executed on the authoring host. Expected time: 5–8 hours plus environment setup, pending pilot.

## A. Reuse and arithmetic

Derive the input-only arithmetic intensity of a square T×T output tile across a K-step of T. Include the final output write and discuss partial accumulations. Explain how increasing T changes register/shared-memory needs and the number of independent tiles. Use the L0 calculator for whole-matrix work/traffic, not as a substitute for the tile-level model.

## B. Run the shape study

```sh
python3 gemm_lab.py --repeats 15 > gemm-cpu.json
python3 -m unittest discover -s tests -v
```

The code compares CPU `torch.mm`-equivalent execution with an explicit Python K-tile loop. Both validate against a higher-precision reference and include Python dispatch/checksum in wall timing. These small shapes illustrate prefill-like, decode-like and small-expert regimes; they do not reproduce a frontier model or use GPU Tensor Cores. Vary tile K and shape, inspect tolerance and identify overhead from extra dispatch and accumulation.

## C. Write or inspect a real kernel

On compatible hardware, implement a guarded shared-memory tiled GEMM or inspect a recorded CUTLASS/CK example. Identify how threads load tiles, synchronize, accumulate and store edge tiles. State dtype/accumulation/layout support. Validate irregular dimensions and memory/race behavior before performance work. The CPU K-loop is not the CUDA implementation: mapping each Python tile to a separate GPU matmul does not create the intended on-chip pipeline.

## D. Compare libraries and pipelines

Choose a supported cuBLASLt/CUTLASS or hipBLASLt/CK path and record version, algorithm/layout, workspace and actual backend. Match the mathematical contract and numerical threshold. Compare one tile/pipeline/epilogue choice at a time across all three regimes. Include kernel-only and allocation/copy-inclusive scope where relevant. Report first use, warm repetitions and an adverse case. Higher-level source syntax alone is not evidence of generated instructions or overlap.

Without device access, complete A/B and a source walkthrough for C/D, label hardware results unverified, and state the measurements needed to test the hypothesis. Use the [report template](../../report_template.md).
