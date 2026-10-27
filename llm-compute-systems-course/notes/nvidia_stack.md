# NVIDIA acceleration stack for training and inference

This is a map of software responsibilities and teaching targets, checked against official documentation/upstream repositories on 2026-10-26. **“Relevant to Rubin” is not the same as “this release, model and dtype are validated on Rubin.”** Confirm a runnable combination before each lab release. Libraries, compilers, frameworks and operational products occupy different layers.

## Kernels and numerical execution

| Component | Training / inference | Mechanism and course use | Primary source |
|---|---|---|---|
| CUDA runtime/driver and CUDA Graphs | Both | Allocation, streams, launches, synchronization and replay; establish overhead and buffer-lifetime contracts | [Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) |
| cuBLAS / cuBLASLt | Both | Optimized matrix multiplication; layout, dtype, workspace, algorithm and epilogue selection; baseline for student GEMMs | [cuBLAS docs](https://docs.nvidia.com/cuda/cublas/contents.html) |
| CUTLASS / CuTe DSL | Both | Build and inspect specialized tiled kernels; compare layouts, pipelining, grouped GEMM and epilogues | [CUTLASS](https://github.com/NVIDIA/cutlass) |
| CUDA Tile / cuTile / Tile IR | Both, subject to operator/toolchain support | Tile-oriented programming and compiler interface; compare abstraction level and generated execution with SIMT CUDA and Triton | [CUDA Tile](https://developer.nvidia.com/cuda/tile), [TileGym](https://github.com/NVIDIA/TileGym) |
| cuDNN | Both | Graph-based primitives/fusions including SDPA; compare backend selection and supported training/inference configurations | [cuDNN overview](https://docs.nvidia.com/cudnn/overview) |
| Transformer Engine | Primarily a training integration target; relevant kernels/precision concepts also apply to inference | Low-precision recipes, scales/amax metadata and fused transformer operations; verify loss and gradients | [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) |
| Model Optimizer / ModelOpt | Optimization for downstream inference, with training-assisted methods where required | Quantization, sparsity and speculative workflows; inspect export and runtime support rather than assume conversion guarantees speed | [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) |

NVIDIA's Transformer Engine README explicitly lists Rubin FP8 training support and a September 2026 release adding Rubin support. This is stronger evidence than generic CUDA compatibility; it still does not validate every precision/model recipe. Keep NVFP4 inference, FP8 training and a specific FP4 training recipe distinct. The [Rubin platform article](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/) provides the broad software context.

## Communication, transfers and data

| Component | Main role | Course use and boundary | Primary source |
|---|---|---|---|
| NCCL | Multi-GPU collective and point-to-point communication | Training gradient/state exchange, tensor/expert parallel traffic and inference collectives; measure latency as well as bandwidth | [NCCL guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) |
| NCCL device API | Communication invoked within CUDA kernels | Study computation/communication fusion, symmetric-memory requirements and supported transports; avoid teaching NCCL as exclusively CPU-initiated | [Device API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html) |
| NVSHMEM | GPU-accessible symmetric memory and one-sided communication | Fine-grained communication/compute overlap and synchronization reasoning; contrast programming model with collectives | [NVSHMEM](https://developer.nvidia.com/nvshmem) |
| GPUDirect RDMA | NIC/GPU direct data path under supported topology | Inspect the transport, registration and topology requirements rather than assuming RDMA is available | [NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html) |
| NIXL | Transfer abstraction across memory/storage backends | KV handoff/offload; separate transfer time from routing, metadata and ownership management | [NIXL](https://github.com/ai-dynamo/nixl) |
| GPUDirect Storage / cuFile | Direct storage-to-GPU path where supported | Dataset, checkpoint or offload transfer experiments; record filesystem and actual direct/fallback path | [GDS overview](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html) |
| DALI | Data loading and preprocessing, particularly image/video/audio | Optional multimodal input-pipeline comparison; do not add it to a token-only workload without a measured input bottleneck | [DALI](https://github.com/NVIDIA/DALI) |

BlueField/DOCA belongs to infrastructure offload and programmable networking, not tensor-kernel compilation. Include it in the topology discussion; building DPU applications is optional scope. Source: [DOCA](https://developer.nvidia.com/doca).

## Training and serving frameworks

| Component | Position in stack | How students use it | Primary source |
|---|---|---|---|
| Megatron Core / Megatron-LM | Distributed training library / reference training application | Trace TP/PP/DP/CP/EP, precision, overlap and checkpoint behavior through one recorded configuration | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) |
| NeMo Framework | Training/post-training framework and integrations | Optional production configuration comparison after core parallelism is understood | [NeMo docs](https://docs.nvidia.com/nemo/), [Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/index.html) |
| TensorRT-LLM | Optimized LLM inference execution/runtime | Compare optimized kernels, KV handling, batching and supported parallelism with vLLM/SGLang under a matched workload | [Repository](https://github.com/NVIDIA/TensorRT-LLM), [support matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html) |
| Dynamo | Distributed inference serving framework | Routing, cache locality, disaggregation and resource planning around execution engines | [Dynamo](https://github.com/ai-dynamo/dynamo) |
| TensorRT | General neural-network inference optimization/runtime | Optional contrast with LLM-specific execution; do not equate its engine format with a PyTorch export archive | [TensorRT](https://github.com/NVIDIA/TensorRT) |
| Triton Inference Server | Serving/orchestration product | Optional endpoint/deployment integration after execution is understood | [Triton server](https://github.com/triton-inference-server/server) |
| Nsight Systems / Nsight Compute / Compute Sanitizer | Measurement and correctness tools | System timeline, per-kernel bottlenecks, race/memory diagnostics | [Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html), [Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) |

**Name collision:** the Triton kernel language/compiler and NVIDIA Triton Inference Server are distinct projects. Likewise NVIDIA Dynamo is distinct from PyTorch TorchDynamo, the `torch.compile` capture frontend. The course must spell out which one is meant.

## What is not an NVIDIA library

PyTorch/TorchTitan/TorchAO/ExecuTorch, Triton language, vLLM/SGLang, FlashAttention/FlashInfer and DeepGEMM/DeepEP are independent upstream projects or broader ecosystem projects, even when NVIDIA contributes or their code targets NVIDIA GPUs. Discuss them alongside NVIDIA software while attributing ownership correctly. No serving engine is assumed to call every library in this table.

The legacy `NVIDIA/NeMo` repository now redirects to Speech; follow current NeMo/Megatron Bridge documentation for LLM training rather than an old repository URL. The reviewed TensorRT-LLM support matrix did not explicitly name Rubin, so this blueprint leaves release-specific Rubin support unconfirmed for that engine.

## Trace assignments

For training: choose a TorchTitan or Megatron job, identify the actual GEMM/attention implementation, low-precision recipe, collective sequence, input pipeline and checkpoint path. For serving: choose vLLM/SGLang or TensorRT-LLM, identify the attention backend, KV allocator, kernel/library calls and parallelism; then add Dynamo/NIXL only when the experiment needs distributed routing or transfer.

For both, make a table of **operation → implementation → hardware resource → evidence in the trace → relevant version/support source**. This prevents a software catalog from substituting for an explanation of performance.
