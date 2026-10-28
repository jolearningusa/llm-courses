# Author validation record and remaining pilots

Recorded 2026-10-26–27. This is an **author walkthrough and evidence ledger**, not a student pilot or a GPU qualification report.

## Package checks

The complete release has 34 decks, 34 teaching SVGs and 16 lab handouts plus capstone. Repository checks cover source/bibliography consistency, exact README file tree and slide counts, site-relative links, listed decks, browser console/navigation and continuous numbering. Additional author checks cover content bounds, SVG label clipping/overlap, accessible SVG titles/descriptions and idempotent figure regeneration. All illustrated slides were rendered/reviewed; dense/new slides and corrected diagram boundaries were inspected separately.

A 2026-10-27 availability audit of 160 unique external URLs in course Markdown returned HTTP 200 for every HEAD request. This verifies reachability, not the correctness of every page; research depth is recorded separately in the reference review.

The pinned PyTorch CPU unit suite passes **48 tests**. GPU tests were not simulated or counted as GPU passes. Executable smoke records below identify the paths actually exercised; measured values are not portable benchmark scores.

## Lab evidence ledger

| Lab | Author-executed / inspected evidence | Limit |
|---|---|---|
| L0 | CPU budget CLI and arithmetic/GQA/state checks | Analytical rates/state model |
| L1 | CPU wall-time CLI, raw samples and interval/pipeline checks | No CUDA profiler timeline |
| L2 | CPU eager/Inductor forward/backward, dynamic shapes and gradient checks | No CUDA/ROCm compiler performance |
| L3 | CPU reduction references/tails/stability checks; CUDA source supplied | CUDA build/run/sanitizer/custom-op integration unverified |
| L4 | Actual CPU GEMM shape/tiling/group reference and checks | Python grouping is not fused GPU GEMM |
| L5 | CPU full/online attention outputs, masks, tails and gradients | Python autograd is not fused GPU training-memory validation |
| L6 | Packed CPU int4/int8 bytes/error checks and CPU BF16 loss/gradient comparison | No LLM quality or accelerated TorchAO/TE recipe |
| L7 | Actual two-process CPU Gloo DDP, three matched-batch updates and delayed-rank run | No FSDP2 GPU peak/overlap or NCCL/RCCL/fabric measurement |
| L8 | Analytical layouts, actual CPU checkpoint output/gradient check, pinned training-stack source reading | Production stacks/multi-GPU layouts not executed |
| L9 | CPU local routing/group/combine reference, skew and empty experts | No distributed dispatcher or fused GPU kernel |
| L10 | Actual CPU checkpoint file save/load; identical loss/parameter/optimizer replay; missing-RNG negative control | No process-kill/distributed durable save qualification |
| L11 | Logical prefix/identity/refcount/eviction/cancel simulation and checks; engine-doc review | Stores no KV tensors; two-engine benchmark not run |
| L12 | Deterministic arrival replay, output conservation and analytical transfer budget | Constructed costs; no full split-cluster simulation or serving benchmark |
| L13 | Exact one-step categorical marginal/sample check; analytical tool-wait accounting | No autoregressive draft/target model or engine policy execution |
| L14 | Actual CPU native AOT .pt2 and XNNPACK .pte build/execute, fresh-process reload and guarded-input rejection | Unquantized host only; other devices/runtimes unverified |
| L15 | Analytical phase-ledger CLI/checks; primary vendor/source dossier | No Rubin/AMD/power benchmark |

L7 parameter difference from the full-batch reference was at most 1.49e-8 in the author run. L14 fresh reload errors were 0 for AOTInductor and 5.96e-8 for ExecuTorch, with one XNNPACK delegated segment. See the [recorded deployment JSON](labs/kit/examples/l14_aoti_cpu.json) and [ExecuTorch JSON](labs/kit/examples/l14_executorch_cpu.json) for the actual small-fixture timing scope.

## Environment qualifications

- Standard-library checks/CLIs: author host Python 3.9.6; no target acceleration inferred.
- PyTorch numerical/framework/AOT checks: macOS arm64, Python 3.12.14, torch 2.9.1; pinned manifests in the kit. Optional NumPy bridge is unused in the baseline environment.
- ExecuTorch CPU: separate environment, 1.0.0 + torch 2.9.1. Exporter uses bundled `flatc`. Optional TorchAO C++ extensions were disabled and the wheel reported its internal consistency verifier unavailable; numerical and explicit loader-contract checks passed. No native quantized or unavailable-verifier claim is made.
- No CUDA GPU/compiler, multi-GPU accelerator cluster, AMD target, Rubin system or edge device was available for this release's accelerated extensions.

## Remaining pilots before full course delivery

1. Run L3–L6 CUDA/library/sanitizer performance paths on a declared supported GPU; inspect numerical and timing/traffic evidence.
2. Run multi-accelerator FSDP/parallelism/MoE/collective/recovery extensions with fixed training progression and real topology.
3. Run the two-engine cache matrix, measured disaggregated/speculative/agent-policy extensions and failure accounting.
4. Validate selected AMD/Rubin/edge paths only when target support and access are established.
5. Conduct a student setup/pacing pilot, record actual task duration and revise estimates. No student-duration record is fabricated here.

The capstone/source/analytical paths are complete as assessment materials. A student analytical submission remains analytical even if every structural check passes.
