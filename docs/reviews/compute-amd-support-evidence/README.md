# AMD selected target/library support review — 2026-11-12 UTC

Base `d6bac58` after merged PR #145. No open PR overlaps at selection. Bounded AMD portion of CMP-4; deck 24 remains 17 slides per language and only slide 14 changes.

| Screenshot | Mapping |
|---|---|
| [English](en.png) | Before left / after right; slide 14 |
| [简体中文](zh.png) | Before left / after right; slide 14 |

Native frames 1280×720; labeled sheets 2560×760. All four pass active-number and slide/SVG bounds checks; both sheets and the full-size Chinese after frame were visually inspected. Original inline SVG outside the 164 managed concepts; no copied vendor artwork. Earlier general rule remains in notes.

[Source and selected-row identities](../../../llm-compute-systems-course/references/amd_support_evidence_2026_11.md): ROCm 10.1.0 conditional compatibility matrix, selected hardware rows and hipBLASLt 1.4.1 precision/install documentation. The versioned matrix could not be read by the web reader but was fetched HTTP 200 with Python; response digests are recorded. FNUZ-on-gfx950 is a documentation counterexample, not an executed rejection. The partial target/OS/library tuple does not qualify missing driver/firmware, complete operation contracts or local dispatch/numerics. No AMD GPU call, build/install, model or performance run; other vendor/library rows and AGT-2's measured benchmark remain pending.

Validation: course consistency, site build/links, all-deck browser check, selected-frame bounds/visual review, exact changed-section/count comparison, private illustration regeneration preserving all 417 existing files and `git diff --check`. No lab code, old result table or tracker/pin changes. No tests mirror prose/arithmetic. User approval is required before merge; auto-merge remains disabled.
