# Course release and follow-up roadmap

The core authoring sequence has produced the master, 30 lectures, capstone workshop and L0–L15 kits, plus the optimization-decision supplement, editable SVG illustrations and separate potential-project specifications. The [build plan](BUILD_PLAN.md) records the serial PR milestones; each previous PR merged before the next branch began.

## Available materials

- [Syllabus](SYLLABUS.md), [deck directory](./#slide-decks), [kit setup](labs/kit/) and [capstone](labs/kit/capstone/CAPSTONE.md).
- [Optimization-technique map](notes/optimization_techniques.md), [coverage audit](notes/course_component_audit.md) and [20 potential projects](projects/POTENTIAL_PROJECTS.md).
- [Release guide](COURSE_RELEASE.md), [author validation/evidence ledger](VALIDATION.md), primary references and versioned environment manifests.

## Follow-up validation work

Accelerated kernel/library, multi-GPU training, real serving-engine, Rubin/AMD and edge paths need the target-specific pilots listed in the validation record. Student setup/pacing also needs a real pilot. These are explicit validation gaps, not empty promised implementation files or claims of measured performance.

For each follow-up use one bounded PR from fresh `origin/main`: declare the target, reproduce the reference, validate correctness, collect unprofiled timings plus a focused trace, update the handout/source ledger and only then mark that path tested. Preserve prior evidence and adverse cases. Optional new projects remain separately scoped proposals until selected for implementation.
