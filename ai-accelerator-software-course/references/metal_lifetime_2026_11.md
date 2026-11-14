# Metal shared-buffer ownership pilot — 2026-11-14 UTC

ACC-3 now has one bounded runtime/device adapter, executed on the local Apple M4 Max through Metal. Across three fresh processes, 24 command buffers completed, 2,511 integer outputs matched an independent CPU calculation, and all 24 host guard attempts were rejected. Known-good work passes before the cancellation exercise and after draining/recycling the same buffers. This qualifies only the declared runtime-level functional fixture; physical qualification, DMA/driver implementation, firmware/reset, sanitizers and fault injection remain not-run.

## Contract and actual target

The [frozen protocol](../labs/metal_lifetime_protocol.json) and original [MSL kernel](../labs/metal_lifetime.metal) were committed at `5732ad1`; the [Swift host adapter](../labs/metal_lifetime.swift) at `b63ca6e` before execution. The target reports Apple M4 Max, registry ID4294968539, unified memory, macOS27.0.1 build26A434. The installed compiler is Apple Swift6.4.2, language mode6, optimization `-O`, selected Xcode/SDK identity retained in the [build/environment ledger](../labs/examples/metal_lifetime/build_environment.json). Selected macOS SDK is27.2; the OS is27.0.1. This observed combination is not a general SDK/OS compatibility guarantee. Driver/firmware build identity is not exposed by the selected API and is explicitly unqualified.

One reusable slot owns input/output/counter MTLBuffers of capacity257 uint32 elements (counter one element), allocated in shared storage. The kernel computes `(input[i] XOR 0xA5A5A5A5) + i modulo2^32`, counts each valid index with an atomic increment and guards padded lanes with `i<n`. CPU input is17*i+19*run. Fixed baseline lengths1/63/64/65/257 cover exact and partial64-lane groups; reuse65, cancellation65 and recovery257 add three commands per process. All cases and all runs are retained; no search or result selection occurs.

## Primary-source mapping to the runtime boundary

Read 2026-11-14 UTC, using Apple's linked Markdown pages because the HTML shells require JavaScript:

| Concept | Documented contract and selected implementation |
|---|---|
| Host pointer vs device address | [`MTLBuffer.contents()`](https://developer.apple.com/documentation/metal/mtlbuffer/contents().md) supplies CPU access to shared allocation. The adapter binds MTLBuffer objects plus offsets for the shader; it does not pass a CPU pointer as a physical/DMA address or expose a Linux `dma_addr_t`. |
| Shared storage | [`MTLStorageMode.shared`](https://developer.apple.com/documentation/metal/mtlstoragemode/shared.md) provides CPU/GPU system-memory access with synchronization responsibility. The app finishes CPU writes before commit and waits for completion before reading/reusing. Shared memory alone does not imply concurrent access is safe. |
| Lifetime | [`retainedReferences`](https://developer.apple.com/documentation/metal/mtlcommandbuffer/retainedreferences.md) controls command-buffer resource retention. The adapter explicitly sets it true, also holds strong slot references and uses tracked default resource hazards. It never intentionally frees pending resources. |
| Quiesce | [`waitUntilCompleted()`](https://developer.apple.com/documentation/metal/mtlcommandbuffer/waituntilcompleted().md) waits for GPU execution and completion handlers. The adapter also requires successful completed status, no command error and exactly one callback before host access. |
| Deterministic pending case | [`encodeWaitForEvent`](https://developer.apple.com/documentation/metal/mtlcommandbuffer/encodewaitforevent(_:value:).md) blocks subsequent passes; [`MTLSharedEvent.signaledValue`](https://developer.apple.com/documentation/metal/mtlsharedevent/signaledvalue.md) can release the gate from the host. The event starts0, command waits for1; raw pending snapshots show no callback and pending ownership before four host guards. The host then signals1 and drains. |
| Cancellation | The selected public [`MTLCommandBuffer`](https://developer.apple.com/documentation/metal/mtlcommandbuffer.md) interface/header has no cancellation method used here. The adapter implements logical publication suppression, not GPU abort: pre-submit cancellation prevents commit; post-submit cancellation retains resources until drain. A defer path releases the held event and waits even if a guard assertion fails. |

Apple's [CPU/GPU synchronization sample](https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work.md) motivates disciplined writes/commit/completion/reuse. The [Linux DMA guide](https://docs.kernel.org/core-api/dma-api-howto.html) distinguishes CPU virtual, physical and bus/DMA addresses. These are separate interface layers: this Metal application never calls a kernel DMA mapping API, observes IOMMU mappings or qualifies arbitrary firmware portability. The compiled SDK's selected public header digests are in the build ledger; no Apple source/artwork is copied into the lesson.

## Results and controls

| Fresh process | Actual completions | Correct uint32 outputs | Host guard rejections | Logical-cancel GPU status / publication | Known-good after drain | Runtime pipeline build seconds |
|---|---|---|---|---|---|---|
| 1 | 8/8 | 837/837 | 8/8 | completed / suppressed | passed | 0.076123 |
| 2 | 8/8 | 837/837 | 8/8 | completed / suppressed | passed | 0.000746 |
| 3 | 8/8 | 837/837 | 8/8 | completed / suppressed | passed | 0.000650 |

[Independent verification](../labs/examples/metal_lifetime/verification.json), [run1](../labs/examples/metal_lifetime/run1.json), [run2](../labs/examples/metal_lifetime/run2.json), [run3](../labs/examples/metal_lifetime/run3.json) retain actual device metadata, raw outputs, tail checks, per-command status/callbacks, shader-counter observations, source/protocol/binary hashes and intervals. Python independently recomputes every returned value; kernel-side counters equal valid lengths and all unused output tails retain sentinel0xDEADBEEF. This is a trusted closed teaching workload: callback/status plus outputs/counter are useful execution evidence, not an authenticated isolation channel for hostile candidate code or a sanitizer result.

Eight guard families per run reject: zero count, count beyond capacity, submission after pre-submit cancel, read/write/recycle/resubmit while pending, and publication after drained logical cancellation. They are **adapter host-policy rejections before the forbidden operation**, not demonstrated GPU illegal-access errors. Legal baseline/reuse/recovery cases are accepted, so the guard is not an always-reject gate. The cancelled kernel still computes its65 elements successfully, but the public result remains suppressed; raw harness verification after quiesce is retained as diagnostic evidence.

All41 existing Python model/policy tests pass; [raw log](../labs/examples/metal_lifetime/python-tests.log) is retained. The original virtual accelerator and its older physical acceptance gate remain unchanged. This target report's `device_run` stage is paired with a bounded runtime-fixture outcome, while physical/sanitizer/fault fields remain not-run. It is not automatically promoted through the existing CPU qualification policy.

Timing is descriptive only. Runtime pipeline compilation includes source-to-library/pipeline work and varies with persistent compiler caches; fresh processes do not imply a cold OS/device/compiler cache. Per-command commit-to-wait intervals include CPU guard/cancel/event handling in the held case. They are not GPU kernel-only latency, end-to-end application throughput or a speed ranking. No sustained load, memory-growth series, error-rate population, power/temperature/clock telemetry or physical corners were measured.

## Reproduce and remaining gates

From the repository root, on an eligible Apple-Silicon Mac with Metal and installed Swift/Xcode tools:

```bash
xcrun swiftc -swift-version 6 -O ai-accelerator-software-course/labs/metal_lifetime.swift -o /tmp/metal-lifetime
/tmp/metal-lifetime 1 ai-accelerator-software-course/labs/metal_lifetime.metal ai-accelerator-software-course/labs/metal_lifetime_protocol.json ai-accelerator-software-course/labs/metal_lifetime.swift /tmp/new-metal-run1.json
```

Choose a new output filename; repeat run IDs2/3 in fresh processes. The adapter refuses to overwrite results. A missing device is not-run; lack of unified memory/shared-event support is outside this fixture. Binary hashes identify the executed local build; the binary is not distributed and exact rebuild byte identity is not promised.

This completes the bounded ACC-3 supported-runtime ownership/cancellation/known-good-after-drain pilot. It establishes no device reset or physical DMA mapping result. Before fault injection, an owned designated test board needs its actual supported driver/firmware identity, quiesce/reset contract and independent known-good recovery procedure. Illegal access, hung kernels, driver faults, hardware resets, firmware flashing and electrical/thermal experiments were not attempted. Broader ACC-2 physical qualification and other audit gates remain open; PR #120 still supplies AGT-2 teaching/protocol only, with later measured pilots separately scoped.
