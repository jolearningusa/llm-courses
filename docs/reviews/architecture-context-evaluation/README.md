# Architecture long-context evaluation — visual review

Prepared 2026-11-07 for audit ARC-3. **Explicit user approval is required before merge.** Before images come from `ec9f9fefb6d87a23b2e220338e5369927d3fc12e` (merged PR #122); after images are from this independent update.

[English comparison](en.png) · [Chinese comparison](zh.png). Each row is before on the left and after on the right: PI/YaRN slide 9; sliding-window/sinks slide 10. Native captures are 1280×720, with active page numbers and text/SVG bounds verified. Full-size images and a local gallery are supplied in chat outputs. Deck counts remain 9/10 per language. The diagrams are original editable SVGs, not copied vendor/paper figures.

The matrix varies synthetic record length, evidence placement and task family. Bars are normalized schematics; each marker denotes a placement condition, expanded by task and fixture variant. They are not model-token counts. The second diagram distinguishes position identities, execution equivalence and learned task quality.

L3 adds an original standard-library fixture builder and seven tests. The default produces 81 fixtures across 27 cells, preserves question/answer/filler identities under movement, checks a literal chain and rejects a changed-evidence answer. Debug-oracle agreement is 81; model runs are zero. This is not RULER, a model adapter or measured long-context quality. Tokenization/template calibration and trained-checkpoint evaluation remain a separate protocol.

Validation: all 16 L3 tests (nine existing plus seven new) and all ten existing L5 tests passed on Python 3.9.6 / NumPy 2.0.2. Existing L5 tests emitted NumPy matmul runtime warnings on this host but completed successfully; original position and attention implementations were unchanged. An initial debug-oracle failure on the two-field distractor format was fixed before the passing fixture run. No model/API or GPU experiment ran.

Source checked: Hsieh et al., [RULER v1](https://arxiv.org/html/2404.06654v1), task families and error analysis, plus the [author repository](https://github.com/NVIDIA/RULER). Historical model results were not imported as present-day rankings. The original math/cache closing questions are retained in the linked L3 handout.
