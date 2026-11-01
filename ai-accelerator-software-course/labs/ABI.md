# Functional accelerator ABI v1

This is a course-owned teaching ABI, not NVIDIA, AMD, TPU, NPU or IREE compatibility. Buffers are modeled byte arrays; integer handles are not CPU pointers, physical addresses, DMA bus addresses or IOMMU mappings.

## Descriptor

Little-endian 24-byte packet, Python struct `<BBHIIIII`. `Descriptor.encode()` checks scalar widths; `Device.submit()` checks semantic eligibility before any pin/queue mutation.

| Field | Width | Contract |
|---|---:|---|
| abi | 8 bits | 1 |
| opcode | 8 bits | 1 = byte copy; 2 = add one modulo 256 |
| reserved | 16 bits | 0 |
| generation | 32 bits | Current device generation |
| src, dst | 32 bits each | Existing distinct buffer handles; exclusive ownership |
| length | 32 bits | Positive and within both buffers |
| cookie | 32 bits | Nonzero, unique within generation even after reap |

No arbitrary graph, code binary, pointer or firmware image is executed. Copy/add-one writes exactly `length` bytes; destination tail is unchanged. Neither operation models a floating-point inference kernel. This fixed integer contract allows exact output checks while the parent course supplies numerical/quality experiments.

## State and ownership

Firmware states: RESET → READY after supported image/ABI boot; FAULT after rejected boot or watchdog expiry; reset returns to RESET and advances generation. `tutorial-fw-v1` is an identifier, not a compiled image, signed firmware or secure boot implementation.

Command states: queued → inflight → done/error → reaped. Queued work may cancel → cancelled → reaped. In-flight cancellation is deliberately unsupported. Submission pins both buffers; host read/write/free is blocked until safe reap. Completed-but-unreaped work still consumes an outstanding slot. The FIFO worker runs one command at a time; no concurrent execution or shared-read aliasing is modeled.

At completion the modeled output write precedes the completion record. A delivered interrupt marks it observed. With `drop_irq`, work can be done but unobserved; polling discovers it. `wait` timeout advances only the selected number of model ticks and never unpins buffers. Normal operation latency and watchdog deadlines use model ticks, not measured milliseconds.

Reset atomically quiesces the **modeled** worker, clears pending work, releases pins and increments generation. Old completion tickets are rejected. A real driver must establish actual DMA cessation before reclaim; generation tagging alone does not stop old physical writes.

## Registers and injected behavior

`registers()` reports status, generation, model tick, queued count, running ticket, allocated bytes and stale completions. These are Python observations, not real MMIO addresses or performance counters.

Fault choices are `none`, `drop_irq`, `dma_error`, `hang`. DMA error completes without modifying destination; hang eventually faults the watchdog and remains pinned until reset. Allocation and queue limits produce explicit rejection. No retries or hidden fallback occur.

The model assumes coherent instantaneous output visibility at its completion action. It has no weak memory, real cache maintenance, DMA mapping, preemption, interrupt controller, ISA timing, DVFS, voltage or temperature. Event-order tests cannot prove those properties.
