# Annotated bibliography, tutorials and implementation references

Checked against primary web sources on **2026-10-26**. **Core** means assigned foundation; **advanced** means selected sections/project reading; **reference** means implementation or documentation to consult. Paper abstracts establish scope; deeper design/evaluation review is identified in [research_review.md](research_review.md). Repository review here covers documentation/interfaces, not installed or benchmarked code.

## Workloads and performance models

- **Core — How To Scale Your Model: All About Rooflines**, Austin et al., 2025. [Book chapter](https://jax-ml.github.io/scaling-book/roofline/). Compute, memory and interconnect bounds; adapt rates to the actual device.
- **Reference — A Survey of LLM Inference Systems**, arXiv:2506.21901. [Paper](https://arxiv.org/abs/2506.21901). Taxonomy and vocabulary; use primary system papers for mechanism claims.
- **Reference — A Survey on Efficient Inference for Large Language Models**, arXiv:2404.14294. [Paper](https://arxiv.org/abs/2404.14294). Broad data/model/system optimization landscape; context beyond the core systems syllabus.
- **Core primer — GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**, Ainslie et al., arXiv:2305.13245. [Paper](https://arxiv.org/abs/2305.13245). KV-head reduction as an architecture/quality choice.
- **Core primer — DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**, DeepSeek-AI et al., arXiv:2405.04434. [Paper](https://arxiv.org/abs/2405.04434). MLA and expert workloads; inspect actual cached-state representation.
- **Core selected sections — DeepSeek-V3 Technical Report**, DeepSeek-AI et al., arXiv:2412.19437. [Paper](https://arxiv.org/abs/2412.19437). Low-precision and communication-aware systems example; model quality results are background.
- **Advanced primer — Mistral 7B**, Jiang et al., arXiv:2310.06825. [Paper](https://arxiv.org/abs/2310.06825). GQA/windowed attention example; architecture support matters.
- **Advanced primer — Mamba: Linear-Time Sequence Modeling with Selective State Spaces**, Gu and Dao, arXiv:2312.00752. [Paper](https://arxiv.org/abs/2312.00752). Different state/computation contract; do not label it exact Transformer attention.

## Kernels, attention and precision

- **Core — FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**, Dao et al., arXiv:2205.14135. [Paper](https://arxiv.org/abs/2205.14135). Tiling and online softmax reduce HBM traffic.
- **Core — FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**, Dao, arXiv:2307.08691. [Paper](https://arxiv.org/abs/2307.08691). Parallelism/work partitioning beyond the IO argument.
- **Advanced — FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision**, Shah et al., arXiv:2407.08608. [Paper](https://arxiv.org/abs/2407.08608). Hopper-specific pipeline/precision example.
- **Advanced — FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling**, Zadouri et al., arXiv:2603.05451. [Paper](https://arxiv.org/abs/2603.05451), [design/evaluation](https://arxiv.org/html/2603.05451v1). Blackwell-era pipelines, load balance and CuTe DSL; baseline versions affect comparisons.
- **Core selected sections — FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving**, Ye et al., arXiv:2501.01005. [Paper](https://arxiv.org/abs/2501.01005), [runtime design](https://arxiv.org/html/2501.01005v2). Storage formats, planning and dynamic scheduling alongside kernels.
- **Advanced — SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**, Xiao et al., arXiv:2211.10438. [Paper](https://arxiv.org/abs/2211.10438). Activation/weight transformations for W8A8.
- **Advanced — AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**, Lin et al., arXiv:2306.00978. [Paper](https://arxiv.org/abs/2306.00978). Weight-only quantization and activation-aware scaling.
- **Core — TorchAO: PyTorch-Native Training-to-Serving Model Optimization**, Or et al., arXiv:2507.16099. [Paper](https://arxiv.org/abs/2507.16099). Training/serving optimization across PyTorch integrations.

## Distributed training

- **Core — ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**, Rajbhandari et al., arXiv:1910.02054. [Paper](https://arxiv.org/abs/1910.02054). Shard training state; account for each dtype and transient buffer.
- **Core — PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel**, Zhao et al., arXiv:2304.11277. [Paper](https://arxiv.org/abs/2304.11277). State lifetime/collective scheduling; pair historical design with current FSDP2 APIs.
- **Advanced foundation — Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**, Shoeybi et al., arXiv:1909.08053. [Paper](https://arxiv.org/abs/1909.08053). Intra-layer model parallelism.
- **Core — Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM**, Narayanan et al., arXiv:2104.04473. [Paper](https://arxiv.org/abs/2104.04473). Composing parallelism and pipeline schedules.
- **Core — Reducing Activation Recomputation in Large Transformer Models**, Korthikanti et al., arXiv:2205.05198. [Paper](https://arxiv.org/abs/2205.05198). Sequence parallelism and selective recomputation; connect memory savings to step time.
- **Reference — FSDP2 tutorial**. [Official tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html). Implementation path to check when releasing L7.
- **Reference — Distributed checkpointing**. [Official API](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html). Save/load and distributed-state handling; recovery experiment must also restore RNG and data position.
- **Reference — TorchTitan**. [Repository](https://github.com/pytorch/torchtitan). Primary PyTorch-native training integration target; record a compatible environment/commit.
- **Reference — Megatron-LM/Core**. [Repository](https://github.com/NVIDIA/Megatron-LM). Training components and reference applications; complementary implementation study.
- **Reference — NeMo / Megatron Bridge**. [Current NeMo docs](https://docs.nvidia.com/nemo/), [Bridge docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/index.html). Current NVIDIA training entry points; legacy NeMo links may lead to Speech.
- **Reference — llm.c**. [Repository](https://github.com/karpathy/llm.c). Inspectable small training path; not a substitute for distributed systems validation.

## Inference systems

- **Core — Orca: A Distributed Serving System for Transformer-Based Generative Models**, Yu et al., OSDI 2022. [USENIX paper/talk](https://www.usenix.org/conference/osdi22/presentation/yu). Iteration scheduling and selective batching.
- **Core — Efficient Memory Management for Large Language Model Serving with PagedAttention**, Kwon et al., arXiv:2309.06180. [Paper](https://arxiv.org/abs/2309.06180). KV allocation/sharing, distinct from attention arithmetic reduction.
- **Core — Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve**, Agrawal et al., arXiv:2403.02310. [Paper](https://arxiv.org/abs/2403.02310). Chunked prefill and scheduling tradeoffs.
- **Core — DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**, Zhong et al., arXiv:2401.09670. [Paper](https://arxiv.org/abs/2401.09670), [evaluation](https://arxiv.org/html/2401.09670v3). Placement and SLO goodput; check transfer costs.
- **Advanced — SGLang: Efficient Execution of Structured Language Model Programs**, Zheng et al., arXiv:2312.07104. [Paper](https://arxiv.org/abs/2312.07104). Structured execution and prefix reuse; separate paper architecture from today's implementation.
- **Advanced — Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving**, Qin et al., arXiv:2407.00079. [Paper](https://arxiv.org/abs/2407.00079). KV-centric serving with heterogeneous storage tiers.
- **Core — Fast Inference from Transformers via Speculative Decoding**, Leviathan, Kalman and Matias, arXiv:2211.17192. [Paper](https://arxiv.org/abs/2211.17192). Exact-distribution verification/correction with a draft model.
- **Advanced — Accelerating Large Language Model Decoding with Speculative Sampling**, Chen et al., arXiv:2302.01318. [Paper](https://arxiv.org/abs/2302.01318). Complementary speculative sampling formulation.
- **Reference — nano-vLLM**. [Repository](https://github.com/GeeeekExplorer/nano-vllm). Small code-reading target before production runtimes.
- **Reference — vLLM**. [Repository](https://github.com/vllm-project/vllm). Serving integration target; select a backend and record its dispatch.
- **Reference — SGLang**. [Repository](https://github.com/sgl-project/sglang). Alternative serving integration target and prefix-cache study.
- **Reference — TensorRT-LLM**. [Repository](https://github.com/NVIDIA/TensorRT-LLM), [support matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html). Optimized NVIDIA inference comparison; model/dtype/GPU support is configuration-specific.
- **Reference — Dynamo / NIXL**. [Dynamo](https://github.com/ai-dynamo/dynamo), [NIXL](https://github.com/ai-dynamo/nixl). Distributed serving versus transfer abstraction; different roles.
- **Reference — Mooncake / LMCache**. [Mooncake code](https://github.com/kvcache-ai/Mooncake), [LMCache](https://github.com/LMCache/LMCache). KV transfer/storage implementation studies.
- **Reference — llama2.c**. [Repository](https://github.com/karpathy/llama2.c). Small C inference path for tracing operator/state lifetimes.

## PyTorch compilers and deployment

- **Core — PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation**, Ansel et al., ASPLOS 2024. [Paper](https://pytorch.org/assets/pytorch2-2.pdf), [official paper/tutorial announcement](https://pytorch.org/blog/pytorch-pytorch-2-paper-tutorial/). Framework-to-compiler path.
- **Reference — torch.compiler and torch.compile**. [Compiler](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html), [API](https://docs.pytorch.org/docs/stable/generated/torch.compile.html), [dynamic shapes](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html). Capture, guards and specialization; pin docs with the lab version.
- **Reference — Custom operators**. [Official tutorials](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html). Registration, correctness and compiler integration.
- **Core — AOTInductor deployment**. [Official guide](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_aot_inductor.html). Exported graph → compiled PT2 archive; C++ deployment and input constraints.
- **Core — ExecuTorch**. [Architecture](https://docs.pytorch.org/executorch/stable/intro-how-it-works.html), [file formats](https://docs.pytorch.org/executorch/stable/file-formats-advanced.html), [upstream](https://github.com/pytorch/executorch). Target lowering, delegates, program/data artifacts and runtime.
- **Reference — TorchAO inference workflows**. [Official guide](https://docs.pytorch.org/ao/stable/workflows/inference.html), [upstream](https://github.com/pytorch/ao). Recipe/hardware support and quality/performance tradeoffs; benchmarks use their own recorded versions.

## NVIDIA hardware and software references

- **Core — Inside the NVIDIA Vera Rubin Platform: Six New Chips, One AI Supercomputer**, NVIDIA, Jan 2026 with update. [Technical article](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/). Platform topology and software context; track changes to platform scope.
- **Core — Inside NVIDIA Rubin GPU Architecture: Powering the Era of Agentic AI**, Alvarez, Mehta and Ghodsian, July 2026. [Technical article](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/). Memory, execution and synchronization mechanisms; vendor source, not independent benchmark.
- **Advanced — NVIDIA Vera Rubin POD: Seven Chips, Five Rack-Scale Systems, One AI Supercomputer**, NVIDIA, March 2026. [Article](https://developer.nvidia.com/blog/nvidia-vera-rubin-pod-seven-chips-five-rack-scale-systems-one-ai-supercomputer). Broader platform boundary; distinguish from NVL72 alone.
- **Reference — Rubin launch claims**. [NVIDIA announcement](https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer). Attribution exercise for up-to generational claims, not a lab performance target.
- **Core/reference — CUDA and PTX**. [Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/), [Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/), [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/), [samples](https://github.com/NVIDIA/cuda-samples). Execution/memory/synchronization contracts and architecture support.
- **Reference — Kernel construction and attention**. [CUTLASS](https://github.com/NVIDIA/cutlass), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [CUDA Tile](https://developer.nvidia.com/cuda/tile), [TileGym](https://github.com/NVIDIA/TileGym). Ownership and support differ; see the library map.
- **Reference — cuBLAS and cuDNN**. [cuBLAS](https://docs.nvidia.com/cuda/cublas/contents.html), [cuDNN](https://docs.nvidia.com/cudnn/overview). Library baselines and graph/epilogue/backend choices.
- **Reference — Transformer Engine and Model Optimizer**. [TE](https://github.com/NVIDIA/TransformerEngine), [ModelOpt](https://github.com/NVIDIA/Model-Optimizer). Training precision management versus model optimization/deployment workflows.
- **Reference — Communication**. [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html), [device API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html), [NVSHMEM](https://developer.nvidia.com/nvshmem), [NCCL topology/RDMA troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html). Collective and one-sided/GPU-initiated operations.
- **Reference — MoE kernels and communication**. [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [DeepEP](https://github.com/deepseek-ai/DeepEP). Independent projects for investigating grouped compute and expert dispatch.
- **Reference — I/O and infrastructure**. [GDS/cuFile](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html), [DALI](https://github.com/NVIDIA/DALI), [DOCA](https://developer.nvidia.com/networking/doca). Different data paths; optional multimodal/DPU scope.
- **Reference — General inference deployment**. [TensorRT](https://github.com/NVIDIA/TensorRT), [Triton Inference Server](https://github.com/triton-inference-server/server). Distinct from LLM execution engines and the Triton kernel compiler.
- **Core/reference — Measurement tools**. [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html), [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html). Timeline, per-kernel analysis and memory/race correctness.
- **Reference — Linux heterogeneous memory**. [HMM](https://docs.kernel.org/mm/hmm.html). Host/device virtual-memory coordination; not a guarantee that managed memory is always faster.

## AMD and portable implementation references

- **Core comparison — HIP**. [Official documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/). Programming/runtime model and porting reference.
- **Core comparison — rocBLAS / hipBLASLt**. [rocBLAS](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/), [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/). BLAS and flexible GEMM/tuning roles.
- **Core comparison — Composable Kernel**. [Current monorepository](https://github.com/ROCm/rocm-libraries), [legacy documentation/mirror](https://github.com/ROCm/composable_kernel). Kernel construction; source migration is explicitly documented.
- **Core comparison — AITER**. [Upstream](https://github.com/ROCm/aiter). AI operator library for inference and training, with multiple implementation backends.
- **Core comparison — RCCL / rocSHMEM**. [RCCL](https://rocm.docs.amd.com/projects/rccl/en/latest/), [rocSHMEM](https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/). Multi-GPU collectives and GPU-centric communication.
- **Reference — ROCm profilers**. [Official overview](https://rocm.docs.amd.com/en/latest/components/profiling-tools-overview.html). System/per-kernel measurement and profiling SDK.
- **Reference — PyTorch/ROCm deployment compatibility**. [Installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html), [compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html). Select validated combinations instead of assuming the latest packages interoperate.

## Courses and tutorials

- **Reference — Stanford CS349E, Efficient ML Inference at Scale**. [Course](https://cs349e.github.io/). Hardware, compilers, kernels and disaggregated serving course structure.
- **Reference — Stanford CS336, Language Modeling from Scratch, Spring 2025**. [Course](https://cs336.stanford.edu/spring2025/). Systems assignment and build/measure teaching style.
- **Reference — Triton tutorials**. [Official gallery](https://triton-lang.org/main/getting-started/tutorials/index.html). Progressive kernel examples; language/compiler, not Triton Inference Server.
- **Reference — GPU MODE**. [Lecture materials](https://github.com/gpu-mode/lectures). Selected talks and code; choose a bounded subset for each lab.
- **Reference — NVIDIA TileGym tutorials**. [Tutorial repository](https://github.com/NVIDIA/TileGym). Tile kernel tutorials and examples.

## AgentOS inference and caching

- **Core — Autellix: An Efficient Serving Engine for LLM Agents as General Programs**, arXiv:2502.13965. [Paper](https://arxiv.org/abs/2502.13965). Program-level serving objectives and scheduling; abstract reviewed.
- **Core — Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live**, arXiv:2511.02230. [Paper](https://arxiv.org/abs/2511.02230). Multi-turn KV retention/TTL; abstract reviewed.
- **Core reference — vLLM Automatic Prefix Caching**. [Design](https://docs.vllm.ai/en/latest/design/prefix_caching/). Hash/block identity, references, eviction and isolation; design sections checked.
- **Core reference — llama.cpp server**. [Upstream documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server). Prompt reuse, slots, RAM caching and save/restore; documented behavior checked, not benchmarked.
- **Reference — GPTCache**. [Repository](https://github.com/zilliztech/GPTCache). Semantic response caching, distinct from KV/prefix caching.
- **Advanced — NVIDIA Vera Rubin Maximizes Intelligence per Dollar for Post-Training**. [NVIDIA article](https://blogs.nvidia.com/blog/nvidia-vera-rubin-post-training-intelligence-per-dollar/). Rollout/training economics; illustrative assumptions and vendor comparisons require attribution.
- **Reference — Vera Rubin platform overview**. [NVIDIA product page](https://www.nvidia.com/en-us/data-center/technologies/rubin/). Current platform positioning alongside detailed architecture articles.

## Pinned L2 implementation references

- **Reference — PyTorch 2.9 autograd**. [Versioned API](https://docs.pytorch.org/docs/2.9/autograd.html). Saved-tensor inspection and backward interface used by the CPU kit.
- **Reference — PyTorch 2.9 compile**. [Versioned API](https://docs.pytorch.org/docs/2.9/generated/torch.compile.html). Full-graph capture and dynamic setting used in the tested L2 environment.

- **Reference — CUTLASS efficient GEMM**. [Current guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html). Hierarchical tiling, reuse and pipeline implementation study for L4.

## Revalidation policy

Before teaching a module, read its assigned design/evaluation sections and upstream code at a recorded commit. Before shipping executable labs, record driver/runtime/compiler/framework/library versions, target GPU, supported dtype/model/backend and test results. `latest`, `stable` and repository `main` are moving references. Track pinned API dependencies with the repository's freshness workflow after executable kits exist.

- **L6 numerical contracts**. [PyTorch 2.9 AMP](https://docs.pytorch.org/docs/2.9/amp.html), [Transformer Engine FP8/FP4 primer](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html). Reviewed 2026-10-26 for operation-specific autocast, scales and target-specific recipes. TorchAO stable docs displayed 0.17; the lab does not claim that library was installed.

- **L7 distributed mechanisms**. [DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html), [FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html), [NCCL collective semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html), [RCCL](https://rocm.docs.amd.com/projects/rccl/en/latest/). Reviewed 2026-10-26; tutorial web versions move independently of the pinned torch 2.9.1 CPU implementation.

- **L8 parallelism**. [Training layouts tutorial](https://jax-ml.github.io/scaling-book/training/), [PyTorch 2.9 activation checkpoint API](https://docs.pytorch.org/docs/2.9/checkpoint.html). Pinned inspected TorchTitan/Megatron snapshots and symbols are documented in [the source walkthrough](../labs/kit/training_source_walkthrough.md); no production training-stack execution claimed.

- **L9–L10 mechanisms**. [Megatron Core MoE](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html), [DeepEP](https://github.com/deepseek-ai/DeepEP), [async DCP recipe](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html). Reviewed 2026-10-26 for routing phases, dispatch/combine and checkpoint staging/completion. Production paths not executed.
