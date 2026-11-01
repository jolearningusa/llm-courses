# Course build plan

## Serial release sequence

1. **Blueprint:** syllabus, source map, validation boundaries and concrete virtual-device lab contract. No runnable code is claimed.
2. **Teaching and functional reference:** eight bilingual decks (master, six topic units, capstone), one editable conceptual SVG per deck; deterministic virtual accelerator, executable scenarios, bounded ordering explorer, reproducibility tests and eight student handouts.
3. **Target qualification:** a separately scoped future project after actual hardware/SDK access. This is not required to use the functional release and is not scheduled by publishing this plan.

Follow the repository workflow: fresh main → one branch/PR → required checks → authorized squash merge → delete branch → refresh main before the next unit. User approval on 2026-10-31 authorizes this course and the existing-course supplement.

## Functional-reference exit gates

- Versioned descriptors validate exact fields, types, opcode and length before changing ownership.
- Host writes/free are rejected while outstanding work owns a buffer; queued cancellation releases via a defined reap path.
- Completion occurs after modeled output writes; lost interrupts can be recovered through polling.
- Timeout does not reclaim live state. Reset quiesces modeled work before a new generation; stale completions are rejected.
- Allocation/queue/DMA failures, double completion and invalid input have explicit outcomes and deterministic regression tests.
- Bounded event exploration reports its depth/state scope and a counterexample for an intentionally incorrect lifetime rule.
- Result manifests label CPU execution, simulated time, source review and real-device evidence distinctly. Target “not run” cannot be promoted as passed.
- README tree/counts, master links, translated twins, site and browser checks pass. All student commands execute or name their missing target prerequisite.

## Physical-target scope

The functional model has no real registers, IOMMU, cache coherency, interrupt controller, ISA or thermal behavior. Real target work requires a documented adapter and instrumentation. Clock/power/temperature values must come from actual telemetry or instruments; emulator ticks and TDP are not measured energy. Use a designated test device and recovery plan, never silently replace those prerequisites with fabricated data.
