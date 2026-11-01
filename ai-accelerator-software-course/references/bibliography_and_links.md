# Primary sources and reading scope

Reviewed 2026-10-31. These sources define responsibilities and test methodology; they do not certify the course's virtual model or establish a universal accelerator ABI. Kernel/docs main branches move: record a release/revision for any target adapter.

| Source | Reading goal and boundary |
|---|---|
| [Linux DMA mapping guide](https://docs.kernel.org/core-api/dma-api-howto.html) | CPU virtual, physical and DMA/bus addresses differ; mapping/lifetime/synchronization responsibilities. The virtual lab does not implement these APIs. |
| [Linux memory barriers](https://docs.kernel.org/core-api/wrappers/memory-barriers.html) | Ordering and visibility need architecture/API-specific reasoning. Event order in Python is not an ISA memory model. |
| [IREE HAL](https://iree.dev/reference/mlir-dialects/HAL/) | Command buffers, queue execution, buffers and synchronization as runtime responsibilities; distinguish framework-level lowering from driver ownership. |
| [Linux KUnit](https://docs.kernel.org/dev-tools/kunit/index.html) | Kernel unit-test structure and isolation. Not executed in the CPU-only reference. |
| [Linux fault injection](https://docs.kernel.org/fault-injection/fault-injection.html) | Allocation and other failure mechanisms; a model injector is not evidence of kernel behavior. |
| [Zephyr Twister](https://docs.zephyrproject.org/latest/develop/twister/index.html) | Build-only/simulation/device execution, board fixtures and results reporting. An embedded test methodology example, not an AI-accelerator SDK. |
| [OpenTitan DV methodology](https://opentitan.org/book/doc/contributing/dv/methodology/index.html) | Testplans, simulation, properties and coverage. OpenTitan is a hardware-verification example, not the target accelerator of this course. |
| [NNSmith, arXiv:2207.13066](https://arxiv.org/html/2207.13066v2) | Valid graph generation and differential compiler testing (§§3–5). Malformed-input rejection is separate. |
| [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | Separate supported memory, shared-memory hazard, initialization and synchronization checks. Passing does not prove the whole stack race-free. |
| [NVIDIA floating-point guide](https://docs.nvidia.com/cuda/floating-point/index.html) | Error/tolerance versus exact protocol matching; CPU higher precision is not a universal bit-exact oracle. |
| [ComputeEval](https://github.com/NVIDIA/compute-eval) | Existing generated-code evaluation and sanitizer integration; compare prior art before claiming a novel AI validation loop. |
| [NVIDIA telemetry](https://docs.nvidia.com/deploy/nvidia-smi/) | Available power, clocks, temperature and errors depend on the target. No physical values are synthesized in the functional lab. |

## AI-assisted development and claim discipline

An assistant may propose a bounded patch, input generator or test. Independent contracts and held-out/negative controls decide acceptance. Keep failed proposals, source/environment identity and the cost of finding a candidate. Do not let the proposer silently relax tests to manufacture a pass.

Use exact checks for protocol/ABI/integer behavior when specified; numerical tolerances and task-quality gates for floating-point or approximate paths. HIL automation need not begin with a large farm. Software observability helps testability but does not replace silicon DFT. Functional simulation, FPGA prototypes and final-silicon power/PVT characterization are distinct evidence paths.
