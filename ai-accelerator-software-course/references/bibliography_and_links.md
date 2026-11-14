# Primary sources and reading scope

Reviewed 2026-10-31; diagram sources rechecked 2026-11-01. These sources define responsibilities and test methodology; they do not certify the course's virtual model or establish a universal accelerator ABI. Kernel/docs main branches move: record a release/revision for any target adapter.

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

## Mechanism diagrams

See the [figure-source inventory](figure_sources.md) for direct links to the inspected Linux, IREE, OpenTitan and Zephyr illustrations and all 32 bilingual diagram locations. The [IREE C API runtime component diagrams](https://iree.dev/reference/bindings/c-api/#runtime-api) and [OpenTitan OTBN DV testbench](https://opentitan.org/book/hw/ip/otbn/dv/index.html#block-diagram) supplement the existing HAL/methodology readings. All distributed illustrations are original teaching schematics with explicit scope, not copied vendor figures.


## Execution coverage supplement — verified 2026-11-08

[ComputeEval README at 00fbad91b63957a002b501f33a06510cd25beaae](https://github.com/NVIDIA/compute-eval/blob/00fbad91b63957a002b501f33a06510cd25beaae/README.md#compute-sanitizer): sanitizer checking is opt-in; its hidden harness supplies run-associated launch evidence separately from candidate stdout. Missing required evidence yields an invalid sanitizer result, with ordinary correctness reported separately. The course's L5 CPU controls adapt the trust/coverage distinction; they do not execute that framework or its CUDA path.

[Compute Sanitizer manual](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html), tool descriptions and command-line options: launch filters/summaries, execution scope and finding policy matter in a real run. Memory, shared-memory hazard, initialization and synchronization checks remain separate. No new claim of a clean GPU-tool run is made.

## Qualification reporting supplement — verified 2026-11-10

[Twister introduction](https://docs.zephyrproject.org/latest/develop/twister/index.html) and [native status reference](https://docs.zephyrproject.org/latest/develop/twister/twister_statuses.html): build-only and execution coverage are distinct; retain native outcomes in any future adapter. The [course execution note](qualification_matrix_2026_11.md) explains why its broader not-run vocabulary is not a Twister status mapping, and documents the actual CPU report and unavailable-board controls. No Twister or device path was run.

## Metal runtime pilot — verified 2026-11-14 UTC

Apple Metal: [shared storage](https://developer.apple.com/documentation/metal/mtlstoragemode/shared.md), [buffer contents](https://developer.apple.com/documentation/metal/mtlbuffer/contents().md), [resource retention](https://developer.apple.com/documentation/metal/mtlcommandbuffer/retainedreferences.md), [completion wait](https://developer.apple.com/documentation/metal/mtlcommandbuffer/waituntilcompleted().md), [event gate](https://developer.apple.com/documentation/metal/mtlcommandbuffer/encodewaitforevent(_:value:).md), [host event signal](https://developer.apple.com/documentation/metal/mtlsharedevent/signaledvalue.md), [command-buffer interface](https://developer.apple.com/documentation/metal/mtlcommandbuffer.md), [CPU/GPU synchronization sample](https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work.md). [Dated source/execution note](metal_lifetime_2026_11.md) distinguishes the actual local device fixture from DMA/driver/firmware/physical qualification. Deck04 slide10 uses original inline bilingual diagrams; existing managed illustrations are unchanged.
