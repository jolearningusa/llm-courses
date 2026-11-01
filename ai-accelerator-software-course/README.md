# AI Accelerator Software Systems: Runtime, Firmware and Verification

Build and verify the software boundary between an AI framework and an accelerator: command ABI, queues, DMA, interrupts, device lifecycle, host ownership, failure recovery and release evidence. AI assistance is used for bounded implementation and testing tasks; an independent contract and evaluator decide acceptance.

**Blueprint release, 2026-10-31.** The syllabus, lab contracts and build sequence are defined. Teaching decks and executable lab implementation follow in the next serial PR. No device, firmware binary, board support or performance result is implied by this blueprint.

## Start here

- [Syllabus](SYLLABUS.md): eight-week path, prerequisites, assessed outcomes and capstone.
- [Build plan](BUILD_PLAN.md): release gates and implementation order.
- [Primary sources](references/bibliography_and_links.md): Linux DMA/memory ordering, IREE HAL, verification tooling and hardware-test references.
- [Validation boundaries](VALIDATION.md): functional model, real target and physical evidence remain separate.

This follows the [efficient LLM computing systems course](../llm-compute-systems-course/), especially its AI-assisted development/verification unit. That course optimizes computation and serving. This one studies the software that safely submits, executes and recovers accelerator work. It is not a GPU ISA, RTL-design or semiconductor signoff course.

## Audience and learning outcomes

Prerequisites: C/C++ reading, Python, basic operating systems, concurrent state machines and tensor inference. Linux kernel/embedded experience helps. CPU-laptop students can execute the planned functional path; real DMA, interrupts, memory ordering and board qualification require a supported test target.

Students should be able to specify a bounded command ABI; trace host/device buffer ownership; distinguish ordering, visibility and completion; validate timeout/cancellation/reset behavior; reproduce and minimize injected failures; compare exact protocol and numerical contracts; and assemble a target-specific release/rollback dossier.

## Slide decks

The next release supplies a master plus six topic decks and a capstone deck, with English/Chinese twins and editable conceptual diagrams. No empty decks or runnable-lab placeholders are published here.

## Experiment spine

A deterministic, CPU-only virtual accelerator will model a versioned copy/add-one command, explicit buffers, bounded outstanding work, firmware states, completion/interrupt delivery and generation-changing reset. Every modeled assumption is specified; it will not emulate GPU ISA timing, Linux DMA coherency or real silicon behavior.

The exercises will extend one model rather than unrelated toy programs: ABI rejection, queued/in-flight ownership, DMA boundaries, firmware compatibility, failure injection, bounded ordering exploration, evidence manifests and release rollback. A later board adapter must be separately qualified before any hardware claim.

## Assessment

Protocol/correctness labs 45%, source/trace critiques 15%, capstone 40%. A minimized negative result can earn full credit. Speed without the stated contract, failed work omitted from reports or simulated ticks presented as milliseconds cannot.

## Repository layout

```text
ai-accelerator-software-course/
├── references/
│   └── bibliography_and_links.md
├── BUILD_PLAN.md
├── LICENSE
├── README.md
├── SYLLABUS.md
└── VALIDATION.md
```
