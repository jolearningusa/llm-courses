# PyTorch: from tensor program to deployed execution

The engineering question is where overhead, data movement, allocation and specialization appear in the execution path. Compilation, quantization and deployment packaging address different parts of that path.

## Stack map

```text
Python nn.Module + tensor operations
  ├─ eager → dispatcher / ATen → libraries or native kernels → device runtime
  ├─ autograd → backward operations + saved tensors → gradients → optimizer
  ├─ torch.compile → Dynamo / FX → Inductor → generated kernels + library calls
  │                   └─ training paths may use AOTAutograd for graph partitioning
  └─ torch.export → ExportedProgram + shape constraints
        ├─ AOTInductor → compiled .pt2 package → Python / C++ deployment
        └─ ExecuTorch edge lowering + target delegation → .pte → edge runtime

TorchAO quantization / sparsity recipes intersect these paths where supported.
CUDA driver/runtime, allocator, streams, hardware libraries and OS underpin execution.
```

This is a conceptual map, not a promise that every operation follows every intermediate stage. See the [PyTorch 2 paper](https://pytorch.org/assets/pytorch2-2.pdf) and [compiler documentation](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html).

## First trace: eager and training

Start with a linear layer, normalization and activation. Inspect shape, stride, dtype, device and aliasing. Map a Python operation to its dispatcher registration and actual backend call; do not assume every matmul invokes the same library algorithm. Measure Python dispatch, launch count and device time separately.

For training, inspect saved tensors and the backward graph. Explain which intermediate must survive, which could be recomputed and when it becomes free. A forward-only fast kernel is not a complete training operator: backward behavior, gradients, mutation/aliasing declarations and compiler integration matter. Use a supported [PyTorch custom-operator workflow](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html) rather than hiding unsupported side effects from the compiler.

## Compilation experiment

Compare eager with `torch.compile` on a fixed shape, then vary batch and sequence length. Separate first invocation, warmup, stable execution and a workload that repeatedly changes shape. Read graph-break and recompilation logs; identify guards and layout changes. Compare full-graph capture with partial capture, and inspect generated kernels or library calls.

Ask whether the gain comes from fusion, fewer launches, better layout or a different kernel. If a graph break splits the critical region, remove it only while preserving semantics. If specialization inflates compile time, compare shape bucketing and supported dynamic shapes. AOTAutograd is a training compiler component; it is distinct from AOTInductor deployment. Required references: [compile API](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) and [dynamic-shape guide](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html).

## Export, artifacts and the likely “AOI” terminology

“AOI native format” is not used as a defined format in this course. The likely intended term is **AOTInductor / AOTI**. If a different artifact is intended, add its exact producer/runtime rather than treating all native formats as interchangeable.

| Object | Meaning | Does serialization alone optimize execution? |
|---|---|---|
| State dictionary / checkpoint | Weights, and possibly separately saved training state | No; a runtime still chooses execution |
| ExportedProgram | Captured graph, parameters and input constraints from `torch.export` | No; capture is a stage for later transformation |
| AOTInductor package | Compiled deployment artifacts packaged in a PT2 archive, commonly `.pt2` | Compilation can help; gain depends on generated execution and workload |
| ExecuTorch `.pte` | Edge program after export/lowering; can include delegated target-specific work | Backend lowering/runtime can help; validate delegation and memory planning |
| ExecuTorch `.ptd` | Optional separately stored tensor data | Data separation itself is not a compute optimization |

A `.pt2` extension can refer to an exported-program archive or compiled package: record the producer and loader, not just the suffix. An AOTInductor package is not universally portable across GPU architectures, ABIs, operating systems or library versions. Check the target and dependencies. Sources: [AOTInductor](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_aot_inductor.html), [ExecuTorch architecture](https://docs.pytorch.org/executorch/stable/intro-how-it-works.html), [file formats](https://docs.pytorch.org/executorch/stable/file-formats-advanced.html).

## TorchAO and numerical execution

[TorchAO](https://arxiv.org/abs/2507.16099) supplies quantization and sparsity workflows, not a single deployment format. Compare weight-only and weight-plus-activation recipes, scale granularity, packed storage and the kernel that consumes that storage. Add calibration/quality evaluation when the recipe requires it.

The [official inference guide](https://docs.pytorch.org/ao/stable/workflows/inference.html) includes hardware requirements and examples where quantization does not help. Use it to select candidate recipes; do not assume its benchmark environment matches a current install. Keep training low precision distinct from inference PTQ. Compare TorchAO with NVIDIA Transformer Engine and Model Optimizer by the actual transformation and target runtime, not the brand name.

## ExecuTorch as a contrasting systems design

Study target partitioning, delegate boundaries, memory planning, operator coverage and runtime footprint. Start with a small computation on an available CPU backend; extend to an LLM after the export/runtime path is validated. Quantized operator coverage and supported dynamic shapes must be checked for the chosen backend and version.

ExecuTorch is central to understanding constrained deployment, while the main distributed training/serving exercises use PyTorch-native and data-center stacks. Its documentation also lists a CUDA backend; this does not establish that it is the preferred Rubin rack-scale runtime. Inspect actual target support and workload fit. See [ExecuTorch upstream](https://github.com/pytorch/executorch).

## Required comparison report

Use identical weights/inputs for eager, compiled, exported-and-reloaded, AOTInductor and one ExecuTorch target where supported. Report unsupported paths explicitly. Measure export/build time, fresh-process load and first execution, steady-state latency, memory footprint and artifact size separately. A small artifact is not evidence of low execution latency. A fast repeated forward pass is not evidence of improved online serving under load.
