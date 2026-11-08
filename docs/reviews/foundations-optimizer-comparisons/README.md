# Foundations: fair optimizer comparisons — review

Prepared 2026-11-07, audit FND-2. **Explicit user approval is required before merge.** Before screenshots come from `a4b5b32daba1c5064efb913112f16dd3f32d7f1a` (merged PR #121); after screenshots are from this independent change.

[Before/after comparison](changes.png): old page on the left, proposed page on the right. Rows: deck 08 slide 10, then deck 09 slide 7. Native captures were 1280×720 and active page numbers verified. The HTML contains original editable SVG diagrams. This course is English-only; both decks retain ten pages. Full-size captures and a local gallery are supplied in the chat outputs.

The first page separates per-optimizer learning-rate studies, validation selection and locked held-out evaluation. The second accounts for a constructed tuning allocation (3,600 update slots per optimizer), separate final refits (600) and evaluation cost. These are caps, not measured runtime; failed/early-terminated runs may execute fewer updates.

The L2 handout supplies a study contract, all-trial ledger, selection/reporting rules and a six-trial executable snippet using the unchanged optimizer code. On Python 3.9.6 / NumPy 2.0.2 the snippet produced all six records, including the nonfinite SGD learning-rate-0.1 trial. At learning rate 0.01 the toy favored SGD; the declared grid also contained a better Adam setting at 0.1. This is a deterministic optimization example, not held-out model-quality or general optimizer superiority evidence. The snippet does not instrument actual steps/time, and the worksheet explicitly asks for them in a complete experiment.

All twelve existing L2 optimizer tests passed. Existing noisy-regression tests emitted NumPy matmul runtime warnings on this host but completed successfully; optimizer code was not modified. No optimizer implementation, reference-run data, requirements or trained-model benchmark was changed. The learned-data validation/test experiment remains a student protocol. Original optimizer hand calculations remain in the handout and other learning-rate questions remain in the schedule deck.

Primary source rechecked: [Google Tuning Playbook — experimental design](https://github.com/google-research/tuning_playbook). Numerical allocation and study diagrams are original teaching material.
