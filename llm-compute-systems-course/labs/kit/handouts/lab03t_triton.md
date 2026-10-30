# L3T — Fused softmax and RMSNorm in Triton

**Released/validated:** `triton_lab.py` softmax and RMSNorm kernels run in Triton's CPU interpreter (`TRITON_INTERPRET=1`) against PyTorch references, with tails, strided rows, large logits, low-precision inputs and contract rejection tested. **Provided but unverified:** the GPU timing path. The authoring host had no GPU, so no compiled-kernel run, timing or bandwidth is claimed. Expected student time: 4–6 hours on a supported GPU, pending pilot.

L3T is a companion to [L3](lab03_reductions.md), for week 4 or 5. L3 writes the reduction in CUDA C++ and handles threads, shared memory and barriers yourself. Here the same row-wise reductions are written in Triton, where you describe one block of a row per program and the compiler chooses the thread mapping, vectorization and shared-memory use. Compare what you had to write in each.

## Setup

Use a separate Python 3.12 environment on Linux x86_64 with [`requirements-triton.txt`](../requirements-triton.txt) (torch 2.15.0, triton 3.9.0). The CPU interpreter path needs no GPU. For the GPU path, use a CUDA GPU supported by that Triton release, or a ROCm build of PyTorch with its matching Triton; record which one you used.

From the kit directory:

```sh
python triton_lab.py --interpret > triton-interpret.json      # CPU correctness only
python -m unittest tests.test_triton_lab -v                    # interpreter on CPU, compiled on GPU
python triton_lab.py --dtype float16 > triton-gpu.json         # GPU: checks, then do_bench timings
```

Without a GPU and without `--interpret`, the CLI exits with code 77 (unavailable hardware, not a pass). The interpreter executes each program with NumPy on the CPU. It catches indexing, masking and numerical errors, but it ignores `num_warps` and says nothing about speed, so interpreter timings are not reported.

## A. Read the kernels

Read `softmax_kernel` and `rmsnorm_kernel`. Each program handles one row: `tl.arange(0, BLOCK)` gives the column offsets, `BLOCK` is the next power of two of the row length, and the mask covers the tail.

1. Why does the softmax load use `other=-inf`, while RMSNorm uses `other=0`? Predict what goes wrong if you swap them, then try it in the interpreter.
2. Where are the values converted to float32, and why is the output converted back to the input dtype only at the store? (This early conversion is also why the bfloat16 cases run in the interpreter, which [does not support bfloat16 arithmetic](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html).)
3. Why do the kernels take a row stride but require the last dimension to be contiguous? The run uses a sliced view whose row stride differs from its length; show that this checks the stride argument.
4. Why is the row index cast to int64 before multiplying by the stride? Give a shape where int32 offsets would overflow.

## B. Correctness before speed

Run the tests in the interpreter. Then add your own cases: a row length of 1, a length just above a power of two (such as 1025), rows with all-equal values and rows mixing very large positive and negative logits. Explain the tolerances in `tolerance()` for float32, float16 and bfloat16, and why the reference computes in float32 before converting.

Break a kernel on purpose (remove the mask, or drop the max subtraction) and confirm that a test fails. A test suite that cannot fail tells you nothing.

## C. Measure on a GPU

On a supported GPU, `run()` checks each shape, then times five paths with `triton.testing.do_bench`, which times with device events and clears the cache before each repeat:

| Path | What it is |
|---|---|
| `triton_softmax`, `triton_rmsnorm` | The fused kernels: one read and one write per element |
| `torch_softmax`, `torch_rms_norm` | The library operators |
| `eager_unfused_softmax` | Softmax written as five eager ops (max, subtract, exp, sum, divide), each with its own memory round trip |

`effective_GBps_at_ideal_bytes` divides one read plus one write of the matrix by the time. It is a lower bound on the traffic, not a profiler measurement.

1. Before running, predict the ratio between the unfused and fused softmax from the number of memory passes. Then compare with the measurement and explain the gap.
2. Compare the effective bandwidth with your GPU's published memory bandwidth. Which shapes come close, and which do not?
3. Sweep `--num-warps` (2, 4, 8, 16) at 1,000 and 16,384 columns. Explain the result in terms of registers per thread and the number of rows (programs) available to fill the GPU.
4. Run 65,536 columns. One program now holds the whole row; use the profiler (Nsight Compute or `rocprof`) to check for register spills, and explain why long rows need a multi-pass or split-row kernel.
5. The [official fused-softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) launches a fixed number of persistent programs that each loop over many rows, instead of one program per row. Port that change and explain when it helps (many short rows) and when it does not.
6. Find a shape where the library operator beats your kernel, and one where yours wins. Do not claim a fixed speedup.

Record the GPU, driver/runtime, torch and Triton versions from the JSON, and keep the raw output.

## D. Look at what the compiler produced (extension)

Set `TRITON_CACHE_DIR` to an empty directory, run once and inspect the cached intermediate files (`.ttir`, `.ttgir`, `.llir`, and `.ptx` or the AMD equivalent). Find where the row reduction becomes warp-level shuffles and where vectorized loads appear. Compare the generated reduction with the shared-memory tree in L3's `cuda/rmsnorm.cu`.

## E. Training extension

The kernels are forward only. Write the RMSNorm backward (input and weight gradients) as a second Triton kernel, wrap both in a `torch.autograd.Function`, and check gradients against `reference_rmsnorm` with `torch.autograd.gradcheck` in float64 on the interpreter before timing anything on the GPU. Explain which forward values you saved and why.

## Submit

Include the [report](../../report_template.md), both JSON outputs, the tests you added and the failing-test evidence from B. Without a GPU, complete A and B on the interpreter, give written predictions for C, and label C–E unverified. A pass in the interpreter never stands in for a compiled GPU run.
