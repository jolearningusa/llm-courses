# Zhenwu product-claim review — 2026-11-12 UTC

Base `62de1e0` after merged PR #144. No open PR overlaps at selection. Bounded Alibaba portion of CMP-4; deck 40 remains 23 slides per language and only slides 15/16 change.

| Screenshot | Mapping |
|---|---|
| [English](en.png) | Before left / after right; slide 15 top, 16 bottom |
| [简体中文](zh.png) | Before left / after right; slide 15 top, 16 bottom |

Native frames 1280×720; labeled contact sheets 2560×1520. All eight frames passed active-number and slide/SVG text-bounds checks. Both sheets and full-size Chinese after frames were visually inspected. Original manual inline SVGs sit outside the 164 managed concepts; no vendor artwork copied. Earlier explanations remain in notes with their historical/unverified status identified.

[Source ledger](../../../llm-compute-systems-course/references/zhenwu_claim_evidence_2026_11.md) records the May/September Alibaba announcements, October PAI template and two timed-out fetches of the old keynote link. Diagrams distinguish announced GB/inter-chip GB/s, the V900 forecast and the missing matched benchmark contract. No chip packaging, usable HBM, link direction, PE layout or verified 9× result is inferred. No TPU/GPU/runtime/compiler/model run or new performance result; SAIL/toolchain qualification and other vendor/library work remain open. PR #120 still leaves AGT-2's measured benchmark pending.

Validation: course consistency, site build/links, all-deck browser check, native-frame bounds/visual review, exact changed-section/count comparison, private illustration regeneration preserving all 417 existing files and `git diff --check`. No lab code or tracker/pin changes. No unit tests mirror prose/arithmetic. User approval is required before merge; auto-merge remains disabled.
