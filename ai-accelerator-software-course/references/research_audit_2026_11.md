# Research audit — AI accelerator software, 2026-11-07

Reviewed the eight-deck syllabus, bibliography and `labs/TARGET_EVIDENCE.md`. Existing material already separates Python event traces, numerical tests and physical qualification. The next improvement is stronger evidence that the intended code actually executed, followed by one bounded target adapter. [Repository-wide audit](../../docs/research-audit-2026-11-07.md).

## Primary sources checked

| Source / inspected scope | Finding and decision |
|---|---|
| [Linux DMA mapping guide](https://docs.kernel.org/core-api/dma-api-howto.html), addressing and mapping responsibilities | CPU virtual/physical and device DMA addresses are different concepts. Keep the course's explicit ownership abstraction, and map it to actual device/driver rules before a target extension. Event-list ordering cannot establish hardware memory ordering. |
| [Linux KUnit documentation](https://docs.kernel.org/dev-tools/kunit/index.html), isolation, fake-device and execution entries | Useful for a driver-facing unit-test adapter. KUnit is not automatic physical qualification; a model test and in-kernel execution need different evidence labels. |
| [Zephyr Twister](https://docs.zephyrproject.org/latest/develop/twister/index.html), built-only, emulator and hardware execution outcomes | “Built” and “executed” are separate statuses, and absent hardware/harnesses affect coverage. Preserve them in a future HIL report rather than collapsing every successful build to passed. Twister is an orchestration example, not an accelerator SDK. |
| [Compute Sanitizer manual](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html), individual tool scopes | Memory, shared-memory hazards, initialization and synchronization checks are distinct. Keep the bounded-coverage claim in deck 05. No clean result follows from not running the intended kernel. |
| [NVIDIA ComputeEval](https://github.com/NVIDIA/compute-eval), sanitizer mode and hidden-test coverage handshake | Sanitizer use is opt-in. Current documentation separates ordinary correctness from invalid sanitizer runs and requires hidden-test evidence of a target launch. This motivates testing the evaluator itself; it does not authenticate arbitrary candidate-generated logs. |
| Ouyang et al., [KernelBench: Can LLMs Write Efficient GPU Kernels?, 2502.10517v1](https://arxiv.org/html/2502.10517v1), metric and hardware variation; [author repository](https://github.com/ScalingIntelligence/KernelBench) | Correctness and a speed threshold jointly define an optimization outcome. Compare search budget and target, not just the fastest surviving kernel. Existing benchmarks are prior art, not proof a new course evaluator is novel. |

## Proposed next changes

| ID / priority | Target | Execution / acceptance |
|---|---|---|
| ACC-1 / P1 verification | Deck 05; target-evidence template | Add an execution-coverage field owned by the trusted test harness. Exercise no-launch, missing/malformed coverage, deliberate failing access and legal candidate controls. Invalid evidence must not become a clean sanitizer pass. First reproduce the protocol on a toy CPU adapter; GPU/tool conclusions require actual tool logs. |
| ACC-2 / P1 qualification | Deck 06; target report / runner | Record built-only, simulator-run, device-run, failed, unsupported and not-run distinctly. Associate every result with source/build/target identity and required checks. Acceptance: an unavailable test board cannot yield a physical pass. |
| ACC-3 / P2 target pilot | Decks 01/02/04; L2/L4 conceptual material | Pick one supported runtime/driver boundary. Map host pointers, DMA mappings, ownership and quiesce conditions to its documentation; add lifetime/cancellation fixtures. Require a known-good recovery run before extending to fault injection. Do not infer portability to all GPU/TPU/NPU firmware. |
| ACC-4 / retain | Decks 03/06; physical-evidence worksheet | Keep physical measurements distinct from emulator results. No new source inspected establishes that the CPU reference can qualify PVT, power, thermal corners or real interrupt behavior. |

## Potential project: evaluator coverage and failure classification

Engineering first; research question is whether an independent coverage gate catches false-clean outcomes without rejecting legal work. Week 1 freeze trust boundaries and accepted/rejected controls. Week 2 implement hidden coverage records and missing-evidence cases. Week 3 add a single supported tool/target adapter and archive raw outputs. Week 4 report false acceptance/rejection, timeout/invalid populations and overhead. Compare ComputeEval/KernelBench facilities before claiming novelty. Without a GPU, deliver only the CPU protocol and a target plan marked not-run.

No driver, firmware image, sanitizer, device adapter or physical experiment was executed in this research pass. The existing 24-test release is not re-certified by reading upstream documentation.


## Follow-up progress — 2026-11-08

**ACC-1 CPU protocol, prepared for review:** deck 05 slides 9/10 (EN/ZH), L5 and target template now separate harness-owned execution coverage, numerical correctness and diagnostics. The existing virtual worker supplies observed start/completion traces; nine constructed controls include no launch with numerically correct output, absent/corrupt/replayed evidence, a rejected bounds request, an error completion, wrong output and a legal copy. Eight new tests plus the original 24 pass; the report retains all outcomes (1 passed / 2 failed / 6 invalid) and source identity. The original virtual model and older functional gate are unchanged. This completes the bounded CPU protocol/control exercise, not a hardened isolation boundary, ComputeEval reproduction, GPU sanitizer result or ACC-2/3 target qualification.
