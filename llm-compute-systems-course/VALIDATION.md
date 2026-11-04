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
| L3T | Triton softmax/RMSNorm kernels in the CPU interpreter: tails, strided rows, large logits, float16/bfloat16, contract rejection (7 tests) | Compiled GPU kernels, timings and bandwidth unverified |
| L4 | Actual CPU GEMM shape/tiling/group reference and checks | Python grouping is not fused GPU GEMM |
| L5 | CPU full/online attention outputs, masks, tails and gradients | Python autograd is not fused GPU training-memory validation |
| L6 | Packed CPU int4/int8 bytes/error checks and CPU BF16 loss/gradient comparison | No LLM quality or accelerated TorchAO/TE recipe |
| L7 | Logical prefix/identity/refcount/eviction/cancel simulation and checks; engine-doc review | Stores no KV tensors; two-engine benchmark not run |
| L8 | Deterministic arrival replay, output conservation and analytical transfer budget | Constructed costs; no full split-cluster simulation or serving benchmark |
| L9 | CPU local routing/group/combine reference, skew and empty experts | No distributed dispatcher or fused GPU kernel |
| L10 | Exact one-step categorical marginal/sample check; analytical tool-wait accounting | No autoregressive draft/target model or engine policy execution |
| L11 | Actual CPU native AOT .pt2 and XNNPACK .pte build/execute, fresh-process reload and guarded-input rejection | Unquantized host only; other devices/runtimes unverified |
| L12 | Actual two-process CPU Gloo DDP, three matched-batch updates and delayed-rank run | No FSDP2 GPU peak/overlap or NCCL/RCCL/fabric measurement |
| L13 | Analytical layouts, actual CPU checkpoint output/gradient check, pinned training-stack source reading | Production stacks/multi-GPU layouts not executed |
| L14 | Actual CPU checkpoint file save/load; identical loss/parameter/optimizer replay; missing-RNG negative control | No process-kill/distributed durable save qualification |
| L15 | Analytical phase-ledger CLI/checks; primary vendor/source dossier | No Rubin/AMD/power benchmark |

L12 parameter difference from the full-batch reference was at most 1.49e-8 in the author run. L11 fresh reload errors were 0 for AOTInductor and 5.96e-8 for ExecuTorch, with one XNNPACK delegated segment. See the [recorded deployment JSON](labs/kit/examples/l11_aoti_cpu.json) and [ExecuTorch JSON](labs/kit/examples/l11_executorch_cpu.json) for the actual small-fixture timing scope.

## Environment qualifications

- Standard-library checks/CLIs: author host Python 3.9.6; no target acceleration inferred.
- PyTorch numerical/framework/AOT checks: macOS arm64, Python 3.12.14, torch 2.9.1; pinned manifests in the kit. Optional NumPy bridge is unused in the baseline environment.
- ExecuTorch CPU: separate environment, 1.0.0 + torch 2.9.1. Exporter uses bundled `flatc`. Optional TorchAO C++ extensions were disabled and the wheel reported its internal consistency verifier unavailable; numerical and explicit loader-contract checks passed. No native quantized or unavailable-verifier claim is made.
- Triton (L3T): separate environment, Linux x86_64, Python 3.12.3, torch 2.15.0 and triton 3.9.0 with `TRITON_INTERPRET=1`; the 7 L3T tests and the CLI ran in the interpreter only.
- No CUDA GPU/compiler, multi-GPU accelerator cluster, AMD target, Rubin system or edge device was available for this release's accelerated extensions.

## Remaining pilots before full course delivery

1. Run L3–L6 CUDA/library/sanitizer performance paths on a declared supported GPU; inspect numerical and timing/traffic evidence.
2. Run the two-engine cache matrix, measured disaggregated/speculative/agent-policy and distributed MoE extensions and failure accounting.
3. Run multi-accelerator FSDP/parallelism/collective/recovery extensions with fixed training progression and real topology.
4. Validate selected AMD/Rubin/edge paths only when target support and access are established.
5. Conduct a student setup/pacing pilot, record actual task duration and revise estimates. No student-duration record is fabricated here.

The capstone/source/analytical paths are complete as assessment materials. A student analytical submission remains analytical even if every structural check passes.

## Inference bottleneck supplement — 2026-10-30

Deck 36 adds 18 slides in English and 18 in Simplified Chinese, with an original dependency/batching SVG. The new slides were browser-rendered and inspected; body and SVG label bounds were checked. Course consistency, site links and all 104 repository decks pass their checks. The 11 external URLs in the bottleneck review responded successfully to availability checks. Worked weight/KV arithmetic was independently checked. The optional worksheet and P26 are proposed experiment protocols: no GPU sampler, beam-search, communication or engine performance experiment was executed for this addition. Existing CPU validation remains unchanged.

## AI-assisted verification unit — 2026-10-31

Executed the standard-library evaluator and eight tests: correct/reordered numerical implementations accepted, three numerical mutants rejected, ownership/generation controls checked. Authored event traces are not concurrent device execution. GPU sanitizers, formal tools, HIL, power/thermal and firmware flashing were not run. The new bilingual decks and diagrams undergo browser/layout/site checks; student pacing remains unpiloted.

## Course-wide visual revision — 2026-11-01

All 41 English/Chinese deck pairs retain their page counts (713 slides per language). Each now contains four SVG mechanism pages: 41 retained companion illustrations and 123 text pages rewritten in place. There are 164 concepts and 328 localized standalone SVGs, embedded in HTML for offline use. Original explanations and authored notes remain accessible through the existing presenter controls.

Primary diagram/documentation review covers CUDA pipelines, PyTorch compiler/autograd/FSDP/DCP, Triton, FlashAttention/FlashAttention-2, GQA, PagedAttention, Sarathi-Serve, DistServe, speculative decoding, ZeRO, NCCL, ExecuTorch, IREE, AMD HIP, TVM, PJRT, GGUF, SGLang and FlashInfer. See [the source guide](references/figure_sources.md) for precise scope and destination pages. Online NCCL, PagedAttention and ExecuTorch illustrations were also inspected visually. Distributed artwork is original geometry; no upstream image assets or benchmark values are reproduced.

Rendered the initial full 1,426-slide course and all 328 final diagram pages at 1280×720; checked text/caption/SVG bounds. Reviewed bilingual contact sheets and representative full-size diagrams. Corrected edge labels, connector/text intersections, tensor-grid dimension consistency, terminal request state, accumulation dependencies and explicit ZeRO-stage representation. Regeneration hash check covers 411 outputs: 82 HTML decks, 328 SVGs and the manifest. Repeated unified generation is idempotent.

Course consistency: three courses, zero problems. Site audit: 818 pages, zero broken relative links, unlisted decks or missing home links. Full browser audit: 128 decks, zero console/numbering problems. Presenter audit: 98 embedded copies, bilingual navigation, notes, timer, resize and popup fallback passed. Lab implementation and recorded reports were unchanged. No new GPU, hardware, HIL, physical measurement or student-pacing result is claimed.

## Worked-example / lab handoff pilot — 2026-11-03

- Revised two existing pages per lecture in decks 10/16/36/38, EN and ZH; counts and all mechanism illustrations unchanged. Added bilingual prediction/reveal notes and linked four executable checkpoints in existing handouts/worksheet.
- Executed all four published Python snippets from `labs/kit/`: pipeline 32/23 μs and 1.391 ratio; cache B 8 matched/1 recomputed with ownership counts; selection 9.2/6.0 ms and 1.087/1.667 ratios; numerical/trace decisions exactly as documented. These are constructed arithmetic or existing CPU fixture outcomes, not GPU measurements.
- Existing cache (3) and verification (8) unit tests passed from the documented kit working directory. An initial test invocation from the repository root failed module imports; rerunning from the kit directory passed without code changes. No lab implementation changed.
- Course check: 4 courses, zero problems. Site build/link check: 956 pages, zero broken relative links/unlisted decks/missing home links. Slide check: 216 decks, zero problems. Presenter check: 98 canonical embedded copies and all behavior checks passed.
- Rendered all 138 pages across the eight revised deck files at 1280×720; bounds check found no issues. Visually inspected EN/ZH contact sheets for the eight revised page positions. Regenerating the illustration library left all 82 compute HTML decks byte-identical.
- Rechecked primary inference analysis, CUDA pipeline, vLLM prefix-cache and Compute Sanitizer documentation. No device benchmark, firmware execution, classroom timing or student comprehension claim added; checkpoint durations remain estimates.
