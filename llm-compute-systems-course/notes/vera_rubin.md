# Vera Rubin: hardware/software co-design case study

**Source-check date: 2026-10-26.** Use NVIDIA's technical descriptions for the architecture and library support matrices for deployability. No Rubin hardware was benchmarked for this course blueprint. The classroom questions and projections below are proposed experiments.

Vera Rubin supports training and inference, with agentic AI and sustained reasoning/inference a major design emphasis. Its relevance to AgentOS comes from the workload, rather than a requirement to run a particular agent operating system. See the [agentic cache topic](agentic_caching.md).

## Establish the system boundary

The [platform article](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/) describes Vera Rubin NVL72 as a scale-up system with 72 Rubin GPUs and 36 Vera CPUs. Vera has 88 custom Olympus Arm-compatible cores. NVLink-C2C joins CPU and GPU, NVLink 6 joins GPUs through switches, and ConnectX-9/Spectrum-X provide scale-out networking. BlueField-4 supports infrastructure offload. These links serve different traffic paths; CPU–GPU coherence does not make all rack memory equally local.

The January article carries an update: broader platform coverage later includes a seventh chip and additional rack types. Teach the GPU/CPU NVL72 configuration separately from the wider [March POD announcement](https://developer.nvidia.com/blog/nvidia-vera-rubin-pod-seven-chips-five-rack-scale-systems-one-ai-supercomputer). LPX and CPX are extensions to examine when relevant, not interchangeable names for a Rubin GPU or a prerequisite for the core labs.

## GPU mechanisms to investigate

NVIDIA's [July GPU architecture article](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/) describes two compute dies in one GPU, 224 SMs, 896 Tensor Cores, up to 288 GB HBM4 and 22 TB/s peak HBM bandwidth. It gives 3.6 TB/s NVLink and 1.8 TB/s NVLink-C2C bandwidth figures. Its execution changes include inline TMA descriptor updates, larger Tensor Core K operations, faster exponentials, finer-grained producer/consumer coordination and counted writes for device-initiated communication. These mechanisms motivate experiments on MoE movement, GEMM, softmax and synchronization. Peak NVFP4 figures are not BF16 training throughput.

Do not add link bandwidths into an application throughput prediction. Directionality, protocol overhead, contention, topology and achievable rates need separate treatment. Do not infer exact instruction syntax or a compiler target flag from a marketing diagram; consult the [CUDA/PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/) and the installed toolchain.

## Hardware-to-software experiment map

The mapping is a course design inference: it identifies which software layer to inspect for a hardware-related bottleneck. It does not assert that every listed release already uses every Rubin feature.

| Resource or mechanism | Software to inspect | Course experiment |
|---|---|---|
| Tensor Core throughput and layouts | cuBLASLt, CUTLASS/CuTe DSL, Transformer Engine | Large training GEMM versus small decode/expert shapes; isolate layout, scales and epilogue costs |
| HBM capacity/bandwidth | CUDA allocator, attention backends, KV manager | Build a live-byte budget; compare it to allocation peaks and memory traffic |
| TMA and on-chip staging | CUDA/PTX, CUTLASS, compiler output | Compare descriptor/setup work and overlapped data movement for grouped workloads |
| Attention softmax and matmul balance | cuDNN, FlashAttention, FlashInfer | Profile the same mask/dtype/head shape; identify non-matmul bottlenecks |
| CPU–GPU path | PyTorch input/runtime, CUDA transfers and graph replay | Separate host starvation, transfers, page faults and launch gaps |
| NVLink scale-up | NCCL, NVSHMEM, expert/tensor parallel runtimes | Sweep payload size and rank placement; inspect the critical synchronization path |
| Scale-out fabric | NCCL transports, GPUDirect RDMA, NIXL | Compare actual transfer cost with KV recomputation or communication overlap |
| Storage/infrastructure | cuFile/GDS, distributed checkpointing, DOCA where applicable | Separate storage throughput from checkpoint pause and recovery duration |

See [NVIDIA acceleration software](nvidia_stack.md) for primary documentation and training/inference roles.

## Assumptions ledger

| Claim category | What can be stated | What remains to validate |
|---|---|---|
| Hardware description | Vendor-documented configuration and mechanisms, with date/source | Actual SKU, firmware, topology and available capacity on the lab machine |
| Software support | A named release/recipe explicitly states a supported configuration | Driver/toolkit, dtype, model, shape and backend combination |
| Vendor speedup | A result under the vendor's stated comparison | Workload, accuracy, latency, power boundary and reproduction on the course workload |
| Classroom projection | An analytical estimate with assumptions | Achieved bandwidth/compute, overlap, queueing and end-to-end measurement |

NVIDIA's [platform announcement](https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer) reports up-to generational improvements. The course does not turn those claims into expected student speedups. Have students locate the baseline, precision, workload and measurement boundary before attempting a comparison.

## A reusable case-study worksheet

1. Draw two GPUs, their CPU connection, a switch, NIC and remote node. Label weight, gradient, activation, KV and checkpoint traffic.
2. Compute a training-state budget with explicit dtypes and a serving budget with the [workload primer](workload_primer.md). Keep logical, allocated and reserved memory distinct.
3. Predict the limiting phase for a large GEMM, a small decode GEMM and skewed MoE routing. State the assumptions that could reverse each prediction.
4. Select a software change from the library map and name the exact code path that should change. Identify an observable trace signature.
5. Measure on accessible hardware. Make a Rubin projection separately and list the missing evidence. If Rubin access exists, replace assumptions with measured rates; record software support and numerical checks.
6. Include a case where faster compute exposes a new softmax, memory, CPU or communication bottleneck.

The aim is architectural reasoning that transfers across generations, while keeping architecture-specific support claims auditable.
