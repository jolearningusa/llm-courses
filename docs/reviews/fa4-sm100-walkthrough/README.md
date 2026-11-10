# FA4 SM100 source walkthrough — visual review, 2026-11-09

Prepared from latest main `f3018f6` after confirmed merge of PR #130. No overlapping open PR was present. One bounded CMP-3 source-reading update, awaiting user approval; no auto-merge.

[English before/after](en.png) and [Simplified Chinese before/after](zh.png): before left, after right; rows are deck 12 slides 15 and 17. Eight native 1280×720 frames were captured with fresh page loads; active slide numbers, slide element bounds and SVG text bounds passed. All four final frames were visually inspected. Each twin retains 17 slides and only pages 15/17 changed. A private illustration-generator rerun preserves these pages byte-for-byte in both languages.

Original SVGs trace S readiness, P/old-O readiness and the candidate qualification sequence. QK and PV share the MMA role; the diagram is a logical dependency view, not a measured timeline. The source note fixes author commit `c055bdc6444a1a1984082a64b7fd62d6cb42a93d`, records interface/forward/softmax hashes and separates ordinary SM100 from other dispatch paths. Existing historical paper material remains intact.

Executed: text-only source inspection and Python 3.9.6 scalar rescale arithmetic. The correct value is 4; intentionally omitting old-output rescaling gives 4.666666666666667, and omitting old-denominator rescaling gives 3. No FA4 installation/import/compile, upstream tests, tensor numerical comparison, generated device code, sanitizer, device trace or benchmark ran. Candidate shapes are proposed tests, not qualified support. CMP-3 target qualification remains open; PR #120 still does not include the measured AGT-2 memory benchmark.

Validation: course consistency — six courses, zero problems; site build/check — 1,123 pages, zero broken relative links/unlisted decks/missing home links. Browser slide check — 271 decks, zero problems. All tracker and requirement files are byte-identical to main; no lab implementation changed. L5 links the independent scalar checkpoint and the not-run target worksheet.
