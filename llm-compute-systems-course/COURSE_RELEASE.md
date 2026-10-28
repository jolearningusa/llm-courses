# Course materials release — 2026-10-27

The authored package contains **33 HTML decks** (master, 30 core lectures, capstone and optimization supplement), **33 editable SVG teaching figures**, **L0–L15 handouts/reference kits**, a capstone assessment package, source notes/bibliography and **20 separate potential-project specifications**. Every deck contains a companion vector illustration; standalone figures and their generator/placement manifest are supplied.

This is a materials and CPU-reference release. It is **not** a claim that all accelerated labs, vendor targets or student pacing have been piloted. See the [validation record](VALIDATION.md) for tested paths and pending work.

## Student entry path

1. Read [the syllabus](SYLLABUS.md), [master overview](slides/00_master_compute_systems.html) and [lab setup](labs/kit/).
2. Start with standard-library budgets and measurement. Use the separate pinned PyTorch environment for numerical/framework/deployment exercises, and the separate ExecuTorch environment for its producer/runtime.
3. Follow each handout's **released**, **analytical**, **source-only** and **unverified extension** labels. Missing GPUs do not turn a skipped experiment into a passed one.
4. Use [deck 32](slides/32_optimization_decision_systems.html) and the [technique map](notes/optimization_techniques.md) to connect observed bottlenecks with libraries, compilers, kernels and serving policies.
5. Finish with the [capstone contract](labs/kit/capstone/CAPSTONE.md), preserving the baseline, correctness gate, raw evidence, adverse case and practical decision.

## Instructor entry path

Read the [component audit](notes/course_component_audit.md), [build plan](BUILD_PLAN.md) and validation record. Choose a hardware path before assigning target-specific extensions. Pilot workload sizes, setup/build duration and student pacing; current handout times are estimates. A CUDA compiler/device and small multi-GPU allocation are needed to validate the accelerated training/kernel track. AMD, Rubin and edge targets each need their own supported environment. No Rubin allocation is required for the source-based case study.

## Reproducibility boundaries

The Python 3.9+ standard-library path is independent of vendor SDKs. The tested PyTorch CPU environment is Python 3.12.14 / torch 2.9.1. ExecuTorch uses its own Python 3.12 environment with ExecuTorch 1.0.0 / torch 2.9.1 and an authoring lock. These are intentional tested pins, not claims to be latest or universal cross-platform locks.

Actual AOTInductor and ExecuTorch artifacts were built/loaded on macOS arm64 for a small unquantized CPU MLP. Other target binaries are not shipped or claimed validated. Raw CPU deployment examples include environment and timing scope. Analytical scheduling/cache/efficiency outputs are explicitly simulated/constructed, never substituted for engine/GPU traces.

Rubin hardware descriptions, software capabilities and vendor results carry source dates and conditions. Conflicting product-versus-architecture rate scopes remain explicit. AMD/NVIDIA library maps compare responsibilities, not blanket feature parity or interchangeable binary formats.

## Maintaining the package

Follow the repository's serial PR workflow: fresh `main` → one bounded change → local/CI checks → authorized squash merge → delete branch → refresh `main`. Regenerate SVG placements after rebuilding decks, synchronize actual README counts/tree and master links, then run course/site/slide checks. Dependency or source drift triggers a new compatibility pilot; update [tracked sources](references/tracked_sources.json) with evidence rather than bumping versions blindly.

The [project catalogue](projects/POTENTIAL_PROJECTS.md) is a separate proposal collection. Publishing those specs does not implement, schedule or launch all projects.
