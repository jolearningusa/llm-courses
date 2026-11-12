# Held-out optimizer learning study — 2026-11-12 UTC

Base `adc454f` after merged PR #147. No open PR overlaps at selection. One FND-2 experiment; foundations is English-only, deck 08 remains 10 slides and only slide 10 changes.

[Before left / after right, slide 10](changes.png): native 1280×720 frames, labeled 2560×760 sheet. Active numbering and slide/SVG bounds passed, sheet visually inspected. Original protocol SVG remains in notes; the new inline SVG reports executed validation/test results, not constructed values or a general ranking.

[Protocol, raw artifacts and limits](../../../ml101-neural-networks-course/references/heldout_optimizer_study_2026_11.md): protocol commit precedes implementation/search; committed validation selection precedes the sole six-checkpoint held-out invocation. Existing L0 loss/gradient and L2 optimizers are unchanged. Eighteen finite fits, 5,400 updates, zero refit; equal updates do not equal runtime. Paired direction changes by seed. No student pilot, neural-network quality result, package migration or old result-table rewrite.

Validation: 16 passing L2 tests (12 existing + four new methods with independent oracles/leakage/selection/failure/checkpoint guards); training/validation-only replay matches except timing fields; course consistency, site build/links, all-deck browser check, exact section/count comparison, screenshot bounds/visual review and `git diff --check`. Held-out evaluation is not repeated for replay. User approval is required before merge; auto-merge remains disabled. Other audit items, including AGT-2's measured benchmark beyond PR #120, remain open.
