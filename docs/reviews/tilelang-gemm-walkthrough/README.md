# TileLang GEMM source walkthrough — visual review, 2026-11-09

Prepared from latest main `a5543cc` after confirmed merge of PR #128. No overlapping open PR was present. One bounded CMP-3 TileLang source-reading update, awaiting user approval; no auto-merge.

[English before/after](en.png) and [Simplified Chinese before/after](zh.png): before left, after right; rows are deck 40 slides 19 and 21. Eight native 1280×720 frames were captured with a fresh page load for each frame. Active slide number, element bounds and SVG text bounds passed; all four final frames were visually inspected. Each twin retains 23 slides, with only pages 19/21 changed. The managed illustration generator was run in a private copy and preserved both new pages byte-for-byte in both languages.

Original SVGs map a pinned author's GEMM to global/shared/fragment storage, then give full/tail/small-M validation controls. Logical byte counts do not claim compiled resource allocation or per-thread register use; stage count does not establish overlap. The source note retains exact author links, two file hashes and the compiler/device evidence still required. The L4 helper's input-dtype accumulation is explicitly distinguished from this source kernel's FP32 fragment accumulator.

Executed: text-only source inspection at TileLang commit `72d986e1c45ba8e81f7611eb6835c9bece097d60`; Python 3.9.6 standard-library coverage arithmetic gives output-grid/K-step results (8×8,32), (2×2,2), (1×1,1). No TileLang install/import/compile, upstream tests, tensor numerical comparisons, generated device code, sanitizer or benchmark ran. This is the TileLang source portion only; FA4/CuTeDSL and device qualification remain pending.

Validation: course consistency — six courses, zero problems; site build/check — 1,122 pages, zero broken relative links/unlisted decks/missing home links; browser slide check — 271 decks, zero problems. All trackers and requirement files are byte-identical to main. No lab implementation changed; the L4 handout links the source-only extension.
