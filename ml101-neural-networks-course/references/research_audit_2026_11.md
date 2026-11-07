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
