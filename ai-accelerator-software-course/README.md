# AI Accelerator Software Systems: Runtime, Firmware and Verification

Build and verify the software boundary between an AI framework and an accelerator: command ABI, queues, DMA, interrupts, device lifecycle, host ownership, failure recovery and release evidence. AI assistance is used for bounded implementation and testing tasks; an independent contract and evaluator decide acceptance.

**Functional-reference release, 2026-10-31; visual revision, 2026-11-01.** Eight decks in English and Simplified Chinese (80 slides per language), 32 original mechanism diagrams (64 localized editable SVGs), L0–L7 handouts and an executable standard-library virtual accelerator are supplied. Twenty-four regression tests, five scenarios, seeded state exercising and a bounded ownership explorer validate their stated functional scopes. Real-device, firmware-binary, driver, HIL and physical qualification remain target-dependent.

[Research audit, 2026-11-07](references/research_audit_2026_11.md): checked readings, compatibility findings, prioritized updates and potential projects. Proposed experiments are distinguished from executed results.

## Start here

- [Syllabus](SYLLABUS.md): eight-week path, prerequisites, assessed outcomes and capstone.
- [Build plan](BUILD_PLAN.md): release gates and implementation order.
- [Primary sources](references/bibliography_and_links.md): Linux DMA/memory ordering, IREE HAL, verification tooling and hardware-test references.
- [Lab kit and commands](labs/), [teaching ABI](labs/ABI.md), [AI task boundaries](labs/AI_PROMPTS.md) and [target evidence template](labs/TARGET_EVIDENCE.md).
- [Visual reading guide and figure sources](references/figure_sources.md): attributed mechanisms, diagram scope and regeneration. Press **S** for detailed notes or **P** for a presenter window.
- [Validation boundaries](VALIDATION.md): functional model, real target and physical evidence remain separate.

This follows the [efficient LLM computing systems course](../llm-compute-systems-course/), especially its AI-assisted development/verification unit. That course optimizes computation and serving. This one studies the software that safely submits, executes and recovers accelerator work. It is not a GPU ISA, RTL-design or semiconductor signoff course.

## Audience and learning outcomes

Prerequisites: C/C++ reading, Python, basic operating systems, concurrent state machines and tensor inference. Linux kernel/embedded experience helps. CPU-laptop students can execute the planned functional path; real DMA, interrupts, memory ordering and board qualification require a supported test target.

Students should be able to specify a bounded command ABI; trace host/device buffer ownership; distinguish ordering, visibility and completion; validate timeout/cancellation/reset behavior; reproduce and minimize injected failures; compare exact protocol and numerical contracts; and assemble a target-specific release/rollback dossier.

## Slide decks

| Deck | Teaching use |
|---|---|
| [Course overview](slides/00_accelerator_software.html) (10 slides) · [简体中文](slides/zh/00_accelerator_software.html) | Course map |
| [Interface contracts and command ABI](slides/01_interfaces.html) (10 slides) · [简体中文](slides/zh/01_interfaces.html) | Topic 1 |
| [Queues, DMA and completion](slides/02_queues_dma.html) (10 slides) · [简体中文](slides/zh/02_queues_dma.html) | Topic 2 |
| [Firmware lifecycle and recovery](slides/03_firmware_lifecycle.html) (10 slides) · [简体中文](slides/zh/03_firmware_lifecycle.html) | Topic 3 |
| [Runtime and driver lifetimes](slides/04_runtime_lifetimes.html) (10 slides) · [简体中文](slides/zh/04_runtime_lifetimes.html) | Topic 4 |
| [Verification and AI-assisted changes](slides/05_verification.html) (10 slides) · [简体中文](slides/zh/05_verification.html) | Topic 5 |
| [Target qualification, HIL and release](slides/06_qualification.html) (10 slides) · [简体中文](slides/zh/06_qualification.html) | Topic 6 |
| [Capstone: one invariant, one decision](slides/07_capstone.html) (10 slides) · [简体中文](slides/zh/07_capstone.html) | Week 8 |

## Experiment spine

A deterministic, CPU-only virtual accelerator models a versioned copy/add-one command, explicit buffers, bounded outstanding work, firmware states, completion/interrupt delivery and generation-changing reset. Every modeled assumption is specified; it does not emulate GPU ISA timing, Linux DMA coherency or real silicon behavior.

The exercises extend one model rather than unrelated toy programs: ABI rejection, queued/in-flight ownership, DMA boundaries, firmware compatibility, failure injection, bounded ordering exploration, evidence manifests and release rollback. A later board adapter must be separately qualified before any hardware claim.

## Assessment

Protocol/correctness labs 45%, source/trace critiques 15%, capstone 40%. A minimized negative result can earn full credit. Speed without the stated contract, failed work omitted from reports or simulated ticks presented as milliseconds cannot.

## Repository layout

```text
ai-accelerator-software-course/
├── labs/
│   ├── examples/
│   │   ├── functional_report.json
│   │   ├── execution_coverage_report.json
│   │   └── qualification_report.json
│   ├── handouts/
│   │   ├── lab00_abi.md
│   │   ├── lab01_queue_dma.md
│   │   ├── lab02_firmware.md
│   │   ├── lab03_runtime.md
│   │   ├── lab04_faults.md
│   │   ├── lab05_state_exploration.md
│   │   ├── lab06_qualification.md
│   │   └── lab07_capstone.md
│   ├── tests/
│   │   ├── test_virtual_accelerator.py
│   │   ├── test_execution_coverage.py
│   │   └── test_qualification_matrix.py
│   ├── ABI.md
│   ├── AI_PROMPTS.md
│   ├── README.md
│   ├── TARGET_EVIDENCE.md
│   ├── experiments.py
│   ├── execution_coverage.py
│   ├── qualification_matrix.py
│   └── virtual_accelerator.py
├── references/
│   ├── bibliography_and_links.md
│   ├── figure_sources.md
│   ├── qualification_matrix_2026_11.md
│   └── research_audit_2026_11.md
├── slides/
│   ├── illustrations/
│   │   ├── zh/
│   │   │   ├── accelerator-00-independent.svg
│   │   │   ├── accelerator-00-scope.svg
│   │   │   ├── accelerator-00-spine.svg
│   │   │   ├── accelerator-00.svg
│   │   │   ├── accelerator-01-addresses.svg
│   │   │   ├── accelerator-01-admission.svg
│   │   │   ├── accelerator-01-packet.svg
│   │   │   ├── accelerator-01.svg
│   │   │   ├── accelerator-02-fifo.svg
│   │   │   ├── accelerator-02-lostirq.svg
│   │   │   ├── accelerator-02-visibility.svg
│   │   │   ├── accelerator-02.svg
│   │   │   ├── accelerator-03-generation.svg
│   │   │   ├── accelerator-03-quiesce.svg
│   │   │   ├── accelerator-03-watchdog.svg
│   │   │   ├── accelerator-03.svg
│   │   │   ├── accelerator-04-cancel.svg
│   │   │   ├── accelerator-04-pins.svg
│   │   │   ├── accelerator-04-reap.svg
│   │   │   ├── accelerator-04.svg
│   │   │   ├── accelerator-05-bounded.svg
│   │   │   ├── accelerator-05-controls.svg
│   │   │   ├── accelerator-05-faults.svg
│   │   │   ├── accelerator-05.svg
│   │   │   ├── accelerator-06-energy.svg
│   │   │   ├── accelerator-06-hil.svg
│   │   │   ├── accelerator-06-rollback.svg
│   │   │   ├── accelerator-06.svg
│   │   │   ├── accelerator-07-baseline.svg
│   │   │   ├── accelerator-07-boundedpatch.svg
│   │   │   ├── accelerator-07-dossier.svg
│   │   │   └── accelerator-07.svg
│   │   ├── README.md
│   │   ├── accelerator-00-independent.svg
│   │   ├── accelerator-00-scope.svg
│   │   ├── accelerator-00-spine.svg
│   │   ├── accelerator-00.svg
│   │   ├── accelerator-01-addresses.svg
│   │   ├── accelerator-01-admission.svg
│   │   ├── accelerator-01-packet.svg
│   │   ├── accelerator-01.svg
│   │   ├── accelerator-02-fifo.svg
│   │   ├── accelerator-02-lostirq.svg
│   │   ├── accelerator-02-visibility.svg
│   │   ├── accelerator-02.svg
│   │   ├── accelerator-03-generation.svg
│   │   ├── accelerator-03-quiesce.svg
│   │   ├── accelerator-03-watchdog.svg
│   │   ├── accelerator-03.svg
│   │   ├── accelerator-04-cancel.svg
│   │   ├── accelerator-04-pins.svg
│   │   ├── accelerator-04-reap.svg
│   │   ├── accelerator-04.svg
│   │   ├── accelerator-05-bounded.svg
│   │   ├── accelerator-05-controls.svg
│   │   ├── accelerator-05-faults.svg
│   │   ├── accelerator-05.svg
│   │   ├── accelerator-06-energy.svg
│   │   ├── accelerator-06-hil.svg
│   │   ├── accelerator-06-rollback.svg
│   │   ├── accelerator-06.svg
│   │   ├── accelerator-07-baseline.svg
│   │   ├── accelerator-07-boundedpatch.svg
│   │   ├── accelerator-07-dossier.svg
│   │   ├── accelerator-07.svg
│   │   ├── build_illustrations.py
│   │   └── figure_specs.json
│   ├── zh/
│   │   ├── 00_accelerator_software.html
│   │   ├── 01_interfaces.html
│   │   ├── 02_queues_dma.html
│   │   ├── 03_firmware_lifecycle.html
│   │   ├── 04_runtime_lifetimes.html
│   │   ├── 05_verification.html
│   │   ├── 06_qualification.html
│   │   └── 07_capstone.html
│   ├── 00_accelerator_software.html
│   ├── 01_interfaces.html
│   ├── 02_queues_dma.html
│   ├── 03_firmware_lifecycle.html
│   ├── 04_runtime_lifetimes.html
│   ├── 05_verification.html
│   ├── 06_qualification.html
│   └── 07_capstone.html
├── BUILD_PLAN.md
├── LICENSE
├── README.md
├── SYLLABUS.md
└── VALIDATION.md
```

[Execution-coverage extension](labs/handouts/lab05_state_exploration.md#execution-coverage-controls), 2026-11-08: 32 CPU tests and nine explicit pass/fail/invalid controls; real sanitizer/device validation remains separate.
