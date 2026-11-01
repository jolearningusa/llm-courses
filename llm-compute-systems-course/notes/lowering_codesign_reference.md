# Model lowering and co-design: reference and worksheet

Companion to [deck 34](../slides/34_model_lowering_codesign.html) · [简体中文](zh/lowering_codesign_reference.md). Reviewed 2026-11-01. The lecture keeps the mechanism path; this handout supplies optional integration detail. It is not an executed compiler lab or a universal API recipe.

## Who chooses what?

| Parameter | Model/quality decision | Hardware and implementation constraint |
|---|---|---|
| Layer dimensions | Hidden/FFN sizes, attention heads, vocabulary | Supported instruction shapes; tile, split-K and dispatch selection |
| Weight representation | Quality tolerance and quantization recipe | Native formats, packed layouts, scale application and conversion |
| Group/block size | Recipe-specific quantization groups | Native block-scaled instructions may impose another granularity; do not identify the two automatically |
| KV state | Actual MHA/GQA/MLA representation | Capacity, cache layout, scale granularity and supported attention kernel |
| Convolution | Kernel, stride, channels and padding | Algorithm choice, workspace, arithmetic/transform error and target performance |

For fixed silicon, map the workload to available resources. For an ASIC/FPGA design, array dimensions and local buffers can also be design variables. Neither case guarantees that a preferred tile or lower bit width improves the job.

## Compiler and runtime are different responsibilities

| Stack | Graph/compiler route | Device/runtime integration | Evidence to demand |
|---|---|---|---|
| IREE | Imported MLIR lowered through target-specific compilation | HAL device/driver and compatible artifact | Compile, load, buffer transfer, execution, output checks |
| Apache TVM | Relax graph and TensorIR; external dispatch can partition subgraphs | Runtime module, target codegen and external libraries | Every operator has a legal target or explicit fallback; boundary cost is measured |
| OpenXLA | Incoming IR through a compatible target compiler | PJRT plugin for buffers, compile and execution interfaces | API conformance plus compiler coverage and numerical behavior |

Use [IREE deployment configurations](https://iree.dev/guides/deployment-configurations/) and its target-specific examples rather than copying a generic command across machines. Compiler backend, target triple, runtime device and ABI must match. A `.vmfb` deployment path and a bare-metal static artifact are different configurations; see the [bare-metal guide](https://iree.dev/guides/deployment-configurations/bare-metal/).

For TVM, read [architecture](https://tvm.apache.org/docs/arch/index.html) and [external library dispatch/BYOC](https://tvm.apache.org/docs/arch/external_library_dispatch.html). Pattern matching, grouping and external code generation can coexist with ordinary lowering. **Partitioning alone does not supply missing operator implementations.** An unsupported remainder must have an actual legal path; CPU fallback, if used, carries placement, transfer and synchronization costs. Do not silently count a host fallback as full accelerator coverage.

For OpenXLA, read [PJRT integration](https://openxla.org/xla/pjrt/pjrt_integration). A runtime plugin is not an automatic code generator for a new ISA. Test the matching compiler, driver and plugin together. Existing deployment routes in decks 22/23 are useful contrasts, not interchangeable artifacts.

## Bring-up gates

1. Buffer round-trip: allocation, transfer, trivial operation, completion and cleanup are correct.
2. Anchor operation: validate shapes/tails/precision before comparing kernel time with an applicable bound.
3. Whole model: validate outputs and actual placement, including remaining operators and temporary memory.
4. Integration: preserve ordering, lifetime, error propagation and the declared framework contract.
5. Tuning: identify exposed time fractions, then compare both component and end-to-end outcomes.

This sequence is a planning framework. It does not assert that every model is GEMM-dominated or that one anchor kernel completes a port.

## Co-design audit worksheet

Choose one decoder and two target environments. Record M/N/K for both prefill and decode, model/quality constraints, a weight representation, actual KV tensors and expected memory traffic. Then name the compiler/runtime boundary for each target and every fallback. Require three independent checks: numerical/quality agreement, execution coverage/placement, and scoped timing with transfer/startup costs identified. An unexecuted target dossier must remain labeled analytical.

The lecture prediction uses a GEMM-plus-normalization graph on a GEMM-only target. A plugin cannot create the missing normalization implementation; the plan must supply lowering or an explicit fallback and account for the boundary. Sources for formats, convolution algorithms and historical workload time fractions remain in the [annotated bibliography](../references/bibliography_and_links.md#model-lowering-and-hardware-co-design-reviewed-2026-10-30); historical percentages are not universal assumptions for this exercise.
