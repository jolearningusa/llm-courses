# Figure sources and reading guide

Reviewed 2026-11-01. The eight decks retain ten slides each. Each deck now has four mechanism diagrams: 32 original concepts and 64 localized, editable SVG files. All diagrams are embedded in the HTML, so slides work offline. Press **S** for retained explanations and **P** for the presenter window.

## Attribution and scope

These are original course drawings informed by the primary-source figures and documentation below. They do not trace or redistribute upstream artwork. Labels, connections and layouts simplify the teaching question. Original SVGs follow the course license; linked upstream illustrations retain their own authorship and licensing. An attribution is a reading source, not vendor endorsement.

| Reference | Online figure or section inspected | What the course changes |
|---|---|---|
| [IREE C API runtime diagrams / HAL](https://iree.dev/reference/bindings/c-api/#runtime-api) | [Original figure / section](https://iree.dev/reference/bindings/c-api/#low-level-concepts) | Original conceptual diagram informed by IREE component diagrams; generic firmware and course roles are additions. |
| [Linux DMA mapping guide](https://docs.kernel.org/core-api/dma-api-howto.html#cpu-and-dma-addresses) | [Original figure / section](https://docs.kernel.org/core-api/dma-api-howto.html#cpu-and-dma-addresses) | Original simplified diagram informed by the Linux address-space illustration; no traced artwork. |
| [OpenTitan OTBN testbench architecture](https://opentitan.org/book/hw/ip/otbn/dv/index.html#block-diagram) | [Original figure / section](https://opentitan.org/book/hw/ip/otbn/dv/doc/tb.svg) | Original simplified testbench informed by OTBN DV; not a reproduction of its architecture or signoff flow. |
| [Zephyr Twister device fixtures](https://docs.zephyrproject.org/latest/develop/twister/index.html#fixtures) | [Original figure / section](https://docs.zephyrproject.org/latest/_images/fixtures.svg) | Original teaching schematic informed by Twister fixtures and evidence separation; not an accelerator SDK. |
| [Linux memory-barrier guide](https://docs.kernel.org/core-api/wrappers/memory-barriers.html) | Documentation and definitions | Original pedagogical ordering diagram informed by the documented memory model; no universal barrier recipe. |
| [NVIDIA SMI telemetry definitions](https://docs.nvidia.com/deploy/nvidia-smi/) | Documentation and definitions | Original measurement schematic; no samples or performance values reproduced. |

The Linux DMA guide supplies an ASCII address-space illustration. IREE publishes runtime component diagrams as Mermaid. OpenTitan OTBN publishes an SVG testbench diagram, and Twister publishes an SVG showing board/fixture/test matching. The linked figures were inspected alongside their surrounding explanations. The remaining drawings visualize the course-owned ABI, executable state model, fault controls, evidence contract and capstone. They are not diagrams of NVIDIA/AMD hardware.

## Diagram inventory

| Deck / slide | Mechanism | Reading source |
|---|---|---|
| [00 / 3](../slides/00_accelerator_software.html#slide-3) · [简体中文](../slides/zh/00_accelerator_software.html#slide-3) | Host and device execution boundaries | [IREE C API runtime diagrams / HAL](https://iree.dev/reference/bindings/c-api/#runtime-api) |
| [00 / 5](../slides/00_accelerator_software.html#slide-5) · [简体中文](../slides/zh/00_accelerator_software.html#slide-5) | One command through the virtual device | [Course-owned ABI v1](../labs/ABI.md) |
| [00 / 7](../slides/00_accelerator_software.html#slide-7) · [简体中文](../slides/zh/00_accelerator_software.html#slide-7) | Four evidence scopes | [Zephyr Twister device fixtures](https://docs.zephyrproject.org/latest/develop/twister/index.html#fixtures) |
| [00 / 8](../slides/00_accelerator_software.html#slide-8) · [简体中文](../slides/zh/00_accelerator_software.html#slide-8) | Independent proposal and acceptance | [Course AI task boundaries](../labs/AI_PROMPTS.md) |
| [01 / 3](../slides/01_interfaces.html#slide-3) · [简体中文](../slides/zh/01_interfaces.html#slide-3) | Tensor semantics and command contracts | [IREE C API runtime diagrams / HAL](https://iree.dev/reference/bindings/c-api/#runtime-api) |
| [01 / 4](../slides/01_interfaces.html#slide-4) · [简体中文](../slides/zh/01_interfaces.html#slide-4) | Two address paths to one RAM buffer | [Linux DMA mapping guide](https://docs.kernel.org/core-api/dma-api-howto.html#cpu-and-dma-addresses) |
| [01 / 5](../slides/01_interfaces.html#slide-5) · [简体中文](../slides/zh/01_interfaces.html#slide-5) | The 24-byte teaching descriptor | [Course-owned ABI v1](../labs/ABI.md) |
| [01 / 6](../slides/01_interfaces.html#slide-6) · [简体中文](../slides/zh/01_interfaces.html#slide-6) | Validation before resource commitment | [Course-owned ABI v1](../labs/ABI.md) |
| [02 / 3](../slides/02_queues_dma.html#slide-3) · [简体中文](../slides/zh/02_queues_dma.html#slide-3) | Ownership across submission and reap | [Course-owned ABI v1](../labs/ABI.md) |
| [02 / 4](../slides/02_queues_dma.html#slide-4) · [简体中文](../slides/zh/02_queues_dma.html#slide-4) | FIFO service and outstanding slots | [Course-owned ABI v1](../labs/ABI.md) |
| [02 / 6](../slides/02_queues_dma.html#slide-6) · [简体中文](../slides/zh/02_queues_dma.html#slide-6) | Output visibility before completion | [Linux memory-barrier guide](https://docs.kernel.org/core-api/wrappers/memory-barriers.html) |
| [02 / 7](../slides/02_queues_dma.html#slide-7) · [简体中文](../slides/zh/02_queues_dma.html#slide-7) | Completed work with a lost interrupt | [Course-owned ABI v1](../labs/ABI.md) |
| [03 / 3](../slides/03_firmware_lifecycle.html#slide-3) · [简体中文](../slides/zh/03_firmware_lifecycle.html#slide-3) | Firmware states and guarded transitions | [Course-owned ABI v1](../labs/ABI.md) |
| [03 / 5](../slides/03_firmware_lifecycle.html#slide-5) · [简体中文](../slides/zh/03_firmware_lifecycle.html#slide-5) | A hung command keeps its buffers | [Course-owned ABI v1](../labs/ABI.md) |
| [03 / 6](../slides/03_firmware_lifecycle.html#slide-6) · [简体中文](../slides/zh/03_firmware_lifecycle.html#slide-6) | Quiescence before memory reclamation | [Linux DMA mapping guide](https://docs.kernel.org/core-api/dma-api-howto.html#cpu-and-dma-addresses) |
| [03 / 7](../slides/03_firmware_lifecycle.html#slide-7) · [简体中文](../slides/zh/03_firmware_lifecycle.html#slide-7) | Same cookie, different generation | [Course-owned ABI v1](../labs/ABI.md) |
| [04 / 3](../slides/04_runtime_lifetimes.html#slide-3) · [简体中文](../slides/zh/04_runtime_lifetimes.html#slide-3) | Runtime, driver interface and device | [IREE C API runtime diagrams / HAL](https://iree.dev/reference/bindings/c-api/#runtime-api) |
| [04 / 4](../slides/04_runtime_lifetimes.html#slide-4) · [简体中文](../slides/zh/04_runtime_lifetimes.html#slide-4) | Exclusive buffer ownership | [Course-owned ABI v1](../labs/ABI.md) |
| [04 / 5](../slides/04_runtime_lifetimes.html#slide-5) · [简体中文](../slides/zh/04_runtime_lifetimes.html#slide-5) | Cancellation depends on command state | [Course-owned ABI v1](../labs/ABI.md) |
| [04 / 6](../slides/04_runtime_lifetimes.html#slide-6) · [简体中文](../slides/zh/04_runtime_lifetimes.html#slide-6) | Completion, observation and reclamation | [Course-owned ABI v1](../labs/ABI.md) |
| [05 / 3](../slides/05_verification.html#slide-3) · [简体中文](../slides/zh/05_verification.html#slide-3) | Stimulus, reference and independent checks | [OpenTitan OTBN testbench architecture](https://opentitan.org/book/hw/ip/otbn/dv/index.html#block-diagram) |
| [05 / 4](../slides/05_verification.html#slide-4) · [简体中文](../slides/zh/05_verification.html#slide-4) | Controls that test the evaluator | [Course AI task boundaries](../labs/AI_PROMPTS.md) |
| [05 / 5](../slides/05_verification.html#slide-5) · [简体中文](../slides/zh/05_verification.html#slide-5) | Faults at different execution boundaries | [Linux fault-injection methodology + course ABI](https://docs.kernel.org/fault-injection/fault-injection.html) |
| [05 / 7](../slides/05_verification.html#slide-7) · [简体中文](../slides/zh/05_verification.html#slide-7) | The two-action ownership counterexample | [Course bounded-state explorer](../labs/handouts/lab05_state_exploration.md) |
| [06 / 3](../slides/06_qualification.html#slide-3) · [简体中文](../slides/zh/06_qualification.html#slide-3) | Functional pass and missing target evidence | [Course target-evidence contract](../labs/TARGET_EVIDENCE.md) |
| [06 / 5](../slides/06_qualification.html#slide-5) · [简体中文](../slides/zh/06_qualification.html#slide-5) | One board, a runner and a recovery path | [Zephyr Twister device fixtures](https://docs.zephyrproject.org/latest/develop/twister/index.html#fixtures) |
| [06 / 6](../slides/06_qualification.html#slide-6) · [简体中文](../slides/zh/06_qualification.html#slide-6) | Power boundary and measurement window | [NVIDIA SMI telemetry definitions](https://docs.nvidia.com/deploy/nvidia-smi/) |
| [06 / 9](../slides/06_qualification.html#slide-9) · [简体中文](../slides/zh/06_qualification.html#slide-9) | A safe rollback boundary | [Course target-evidence contract](../labs/TARGET_EVIDENCE.md) |
| [07 / 3](../slides/07_capstone.html#slide-3) · [简体中文](../slides/zh/07_capstone.html#slide-3) | The capstone decision and its evidence | [Course capstone contract](../labs/handouts/lab07_capstone.md) |
| [07 / 4](../slides/07_capstone.html#slide-4) · [简体中文](../slides/zh/07_capstone.html#slide-4) | A bounded patch inside the stack | [Course AI task boundaries](../labs/AI_PROMPTS.md) |
| [07 / 6](../slides/07_capstone.html#slide-6) · [简体中文](../slides/zh/07_capstone.html#slide-6) | A shared evaluator for baseline and candidate | [Course capstone contract](../labs/handouts/lab07_capstone.md) |
| [07 / 7](../slides/07_capstone.html#slide-7) · [简体中文](../slides/zh/07_capstone.html#slide-7) | The reproducible artifact package | [Course capstone contract](../labs/handouts/lab07_capstone.md) |

## Reading conventions

- Arrows denote commands, data dependencies, notifications or state transitions as labeled. Only the descriptor fields are drawn to byte-width scale. Timing diagrams show logical ordering, not measured duration.
- The functional worker is single-FIFO. Pins are exclusive for both source and destination, including done-but-unreaped work. Host access restrictions are teaching-model rules, not a claim that every DMA API enforces this policy.
- Reset atomically quiesces the modeled worker. A target adapter must establish real quiescence before reclaim; generations alone cannot stop old writes.
- Testbench and evidence diagrams describe responsibilities. The CPU model is not OpenTitan RTL, IREE firmware, a Linux driver or a Twister board adapter.
- The power curve is schematic with no numeric samples. No hardware, energy, performance or thermal result is synthesized.

## Editing and regeneration

Edit `slides/illustrations/figure_specs.json` for bilingual titles, captions, anchors and attribution. Edit `build_illustrations.py` for geometry. Run:

```sh
python3 ai-accelerator-software-course/slides/illustrations/build_illustrations.py
```

The generator updates four matching slides in each deck in place and preserves explanatory notes. It embeds the repository's existing offline presenter controls from `llm-compute-systems-course/slides/presenter_mode.js`. There is no runtime network dependency. Re-running the generator should leave all outputs unchanged.


## Additional authored diagrams — 2026-11-08

Deck 05 slides 9/10 (EN/ZH) add original inline SVGs: three independent acceptance dimensions and the executed CPU control matrix. The first is a schematic gate; the second summarizes the nine named cases in `labs/examples/execution_coverage_report.json`. No box width represents timing, risk or statistical coverage. They sit outside the 32 managed bilingual diagram concepts above and must survive regeneration. The [pinned ComputeEval reading](bibliography_and_links.md#execution-coverage-supplement-verified-2026-11-08) motivates the protocol; no upstream artwork was copied and no GPU run is depicted.
