# Execution coverage controls — visual review, 2026-11-08

Prepared from latest main `5018b3b`, after confirmed merge of PR #125. One bounded independent ACC-1 CPU-protocol update, awaiting user approval; no auto-merge.

[English before/after](en.png) and [Simplified Chinese before/after](zh.png): before left, after right. Rows are deck 05 slide 9 and slide 10; each deck still has ten slides. Eight native frames were captured at 1280×720, active slide numbers/bounds checked and final layouts inspected.

The first original SVG separates execution coverage, diagnostics and numerical correctness; the second summarizes all nine constructed CPU controls. Raw report and source hashes live in `labs/examples/execution_coverage_report.json`. The observed control population is 1 passed / 2 failed / 6 invalid; it is not a statistical coverage estimate. Original NNSmith/prior-art caution and state-exploration assignment remain in the linked L5 handout.

All 32 CPU tests (24 original + eight new) passed on Python 3.9.6. Original model and older functional gate were checked unchanged. The protocol uses only course-owned in-process controls; it neither runs arbitrary candidates nor authenticates imported reports. GPU / Compute Sanitizer remain not-run, and the rejected-bounds case does not execute a real out-of-bounds access.

Primary source: ComputeEval README at `00fbad91b63957a002b501f33a06510cd25beaae`, sanitizer section, and NVIDIA's tool-scope manual. Illustration regeneration preserves all course files. Course check: six courses, zero problems. Site check: 1,117 pages, zero broken relative links/unlisted decks/missing home links. Slide check: 271 decks, zero problems.
