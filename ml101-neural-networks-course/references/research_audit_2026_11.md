# Research audit — ML foundations, 2026-11-07

This is a research and teaching-design review, not a new validation of all numerical results. Reviewed the syllabus, deck inventory, bibliography and representative L1/L2 lab code against the primary resources below. The historical progression remains appropriate; a new model release is not a reason to remove backpropagation, CNNs or recurrent networks. [Repository-wide audit](../../docs/research-audit-2026-11-07.md).

## Readings checked and decisions

| Primary source / inspected material | Finding and course decision |
|---|---|
| [Stanford CS231n learning notes](https://cs231n.github.io/neural-networks-3/), gradient checks and sanity checks | Centered finite differences, scale-sensitive error and nondifferentiable points matter. L1 already implements a centered check and relative error; extend its diagnostic explanation rather than claiming gradient checks are absent. |
| [Google Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook), experimental design and scientific/nuisance/fixed hyperparameters | Comparing optimizers at one shared learning rate can confound the result. L2 already searches learning rates; make the search budget and validation-selection boundary visible in the exercise/report. |
| [Stanford CS336, Spring 2026](https://cs336.stanford.edu/), prerequisites and assignments | The modeling course assumes stronger implementation and systems preparation than introductory ML. Use it as a downstream bridge, not the expected workload of a first numerical-gradient exercise. |
| [CS336 assignment 1 repository](https://github.com/stanford-cs336/assignment1-basics), student scaffolding/tests | A useful example of student-facing adapters and implementation tasks. Compare assessment design without replacing the NumPy labs with a much larger language-model assignment. |

## Proposed changes, in execution order

| ID / priority | Exact target | Bounded execution and acceptance |
|---|---|---|
| FND-1 / P1 teaching | Decks 06/19; L1 `test_mlp.py` | Add a prediction exercise sweeping finite-difference step size on a smooth function, then show a ReLU kink. Plot absolute and relative errors separately near zero. Submit an intentionally wrong gradient that the check rejects and a correct smooth case it accepts. No universal error threshold across all dtypes/functions. |
| FND-2 / P1 teaching | Decks 08/09; L2 `best_lr` / `best_final` | Have students declare optimizer as the scientific variable, learning rate as a nuisance variable, and equal search budgets. Select on validation data and evaluate the selected setting once on a held-out set. Report all searched settings and seeds; do not present the smallest loss from one seed as a general optimizer ranking. |
| FND-3 / P2 integration | Deck 20; Transformer L1 and training L1 | Link a shared shape/gradient worksheet: MLP block → residual block → causal decoder. Require students to identify which tests check numerical derivatives versus causality. Acceptance: one reproducible failure when a future token affects an earlier output. Reuse existing downstream tests where possible. |
| FND-4 / retain | Core decks 01–18 and L0–L6 | Keep canonical papers and the small executable examples. No source inspected in this pass establishes that a historical mechanism has become incorrect. Modern architecture catalogues belong in the architecture course. |

## Potential project: a gradient-check failure atlas

Engineering/teaching project, not claimed new research. Week 1: freeze smooth, saturated, broadcast and reused-variable fixtures plus a finite-difference reference. Week 2: introduce scale changes, kinks and deliberate backward mutants; minimize one failure in each family. Week 3: make an annotated plot and diagnostic guide with an independent accepted control. Deliver code, inputs, raw errors and explanations. A CPU/NumPy path is sufficient; compare to an external autograd system only in a separately recorded environment. Success is correctly diagnosing the failure class, not merely lowering one tolerance.

## Scope and remaining evidence

These are proposed additions, not executed experiments. Existing tests were inspected, not rerun as a claim of full course qualification. Suggested classroom durations require a student pilot. Four sources were read for mechanisms/assignment design; this is not a census of all introductory ML tutorials.

## FND-1 implementation prepared for review — 2026-11-07

Deck 06 slide 10 and deck 19 slides 4/10 now show executed float64 step/error plots and a ReLU branch-crossing schematic. L1 adds a reproducible diagnostic runner, six control tests and a report with standalone SVGs. The runner exercises the existing cubic autograd path, rejects two mutants, and separates absolute/relative scale from nonsmooth checks. Classroom comprehension remains untested; the wider gradient-failure-atlas project is not complete. Before/after screenshots are part of review, and explicit user approval is required before merge.

## FND-2 teaching/protocol update prepared for review — 2026-11-07

Deck 08 slide 10 and deck 09 slide 7 now show independent per-optimizer tuning and a search/refit/evaluation budget. The L2 worksheet declares controlled splits/seeds, validation selection, held-out boundaries, failed trials and actual cost. A six-trial command reuses existing optimizers on a deterministic quadratic; it is optimization-only and does not establish held-out model quality. Original optimizer implementations and reference results are unchanged. This completes the teaching/protocol addition; a student validation-selected learning experiment remains to be run. Screenshots accompany review and explicit user approval is required before merge.

## FND-3 shared validation bridge — 2026-11-10

Deck 20 slides 9/10 and the [shared worksheet](decoder_bridge_worksheet_2026_11.md) connect MLP/residual shape and gradient contracts to architecture/training L1. Ten reused NumPy baseline cases pass, with two smooth residual numerical checks and an identity-path control. A temporary all-allowed decoder mask makes the exact existing future-invariance test fail while attention-definition equivalence still passes; restoring it passes again. Two selected existing PyTorch model tests pass in a separate lane. The bounded integration/expected-failure acceptance is implemented, not full Transformer gradient coverage, target-shift/packing verification, new training or learned-quality validation.

## FND-2 validation-selected learning study — 2026-11-12 UTC

The [executed NumPy study](heldout_optimizer_study_2026_11.md) reuses L0 regression loss/gradient and L2 optimizers on disjoint fixed train/validation/test IDs. Eighteen complete fits use 2,700 updates per optimizer; committed validation selection precedes six held-out scores with no refit. All 16 L2 tests pass, including leakage/selection/overflow/checkpoint controls. Deck 08 slide 10 visualizes actual outcomes. This completes the bounded FND-2 reference learning experiment, not a universal ranking or classroom pilot. Other audit items and AGT-2's measured benchmark beyond PR #120 remain open.
