# Ironwood chip/device specification review — 2026-11-11

Base `766b7fe` after merged PR #143; no open PR overlaps. One selected Google CMP-4 claim review. Deck 40 remains 23 slides per language; only slide 13 changes.

| Screenshot | Mapping |
|---|---|
| [English](en.png) | Before left / after right, slide 13 |
| [简体中文](zh.png) | Before left / after right, slide 13 |

Native frames 1280×720, labeled contact sheets 2560×760. Active slide numbering and slide/SVG text bounds passed in all four frames; both contact sheets were visually inspected. Original slide explanation remains in hidden notes. The new original inline SVG is outside the 164 managed concepts and copies no vendor artwork.

[Source ledger](../../../llm-compute-systems-course/references/ironwood_spec_evidence_2026_11.md): system specifications, dual-chiplet exposure, framework-visible memory reservation and dated service preview/GA entries. Preserve the same-page HBM rate difference without assuming a cause. No TPU, runtime, model, throughput or power test ran; no old numerical table, dependency pin or tracker date changed. Other CMP-4 vendors and AGT-2's measured benchmark remain open.

Validation: course consistency, site build/links, all-deck browser check, selected frame bounds/visual review, private regeneration preserving all 417 existing slide/asset files and `git diff --check`. User approval is required before merge; auto-merge remains disabled.
