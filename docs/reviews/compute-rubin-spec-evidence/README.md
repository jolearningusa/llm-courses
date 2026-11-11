# Compute Rubin specification evidence — 2026-11-10

Base `8089d06` (latest main after this chat's merged #139 and separate AgentOS updates). One bounded CMP-4 review of selected Rubin rows. Deck 24 remains 17 slides in EN/ZH; only slides 6/7 change. Other vendor rows and software qualification remain open.

| Screenshot | Mapping |
|---|---|
| [English](en.png) | Before left / after right; slide 6 top, slide 7 bottom |
| [简体中文](zh.png) | Before left / after right; slide 6 top, slide 7 bottom |

Native frames are 1280×720; contact sheets 2560×1520 with labels. All eight frames passed active-number and slide/SVG text bounds checks; both sheets and the full-size Chinese slide 7 were visually inspected. The original SVGs are embedded offline, outside the 164 managed concepts. Private illustration regeneration preserves the deck/assets; the inventory README now identifies that managed scope. Original slide explanations remain in notes.

[Source note and PDF identity](../../../llm-compute-systems-course/references/rubin_spec_evidence_2026_11.md): vendor product page, July architecture article, linked 11-page datasheet and dated production release. PDF printed pages 3/8/9 were rendered with Poppler and inspected in full, including column headings and footnotes. Its SHA-256 is recorded. No vendor image/PDF is redistributed. The web table's third configuration column is GPU; the PDF's is NVL4. Published HBM/NVLink pairs remain unresolved instead of being combined into an invented SKU or upgrade. Decimal arithmetic is a comparison, not an explanation of operating points or achieved rates.

Diagrams distinguish GPU/Superchip/NVL72, sparse/dense peaks, conditioned vendor workload comparisons and vendor availability from unexecuted local validation. The 2026-10-27 case-study review remains dated history, with a separate selected recheck. No hardware, driver/library dispatch, model-quality, latency or power benchmark ran; no lab Python, prior results or tracker pins changed.

Validation: course consistency (6 courses), site build/link check (1,152 pages), all-deck browser check (271 decks), exact slide/count comparison in both languages, before/after frame bounds and visual review, PDF digest/decimal arithmetic, private regeneration preservation and `git diff --check`. No new tests mirror the arithmetic or prose. User approval is required before merge; auto-merge must remain disabled.
