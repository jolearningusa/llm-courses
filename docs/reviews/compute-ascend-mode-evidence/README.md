# Ascend working-mode source review — 2026-11-12 UTC

Base `c2dba91` after merged PR #146. No open PR overlaps at selection. Selected Ascend portion of CMP-4; deck 40 remains 23 slides in each language, only slide 14 changes.

| Screenshot | Mapping |
|---|---|
| [English](en.png) | Before left / after right; slide 14 |
| [简体中文](zh.png) | Before left / after right; slide 14 |

Native frames 1280×720, labeled sheets 2560×760. All four pass active-number and slide/SVG text-bounds checks. Both sheets and full-size Chinese after frame visually inspected. Original inline SVG, no vendor artwork copied, outside the 164 managed concepts. Earlier TBE explanation remains in notes with scope identified.

[Source identities and selected claims](../../../llm-compute-systems-course/references/ascend_mode_evidence_2026_11.md): existing commercial/800 TBE overview and CE/850 Ascend C architecture. The newer 910 English lead timed out; no newest-SDK claim is made. Diagram distinguishes shared/independent Scalar scheduling, symbolic 1:N ratio and Atlas 200I/500 A2 hardware/programming-mode caveat. No physical layout, fixed N, core totals, buffers, peak throughput or local execution is inferred. No SDK install, platform query, compile, NPU or performance run. Other vendor/library gates and AGT-2's measured benchmark remain pending.

Validation: course consistency, site build/links, all-deck browser check, native frame bounds/visual review, exact section/count comparison, private unified illustration regeneration preserving all 417 existing files, source response digests and `git diff --check`. No lab code, old numerical results or tracker/pin changes; no tests mirror prose/arithmetic. Explicit approval is required before merge; auto-merge remains disabled.
