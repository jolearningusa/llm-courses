# Architecture byte/chunk supplement — 2026-11-10

Base `189c8a5` (main after #135). One bounded ARC-2 reading/CPU-fixture update. Deck 07 remains nine slides in EN and ZH; only slides 8/9 changed. Earlier historical mechanisms/results are unchanged. Original closing text remains in speaker notes; exercise questions also remain in the L2 handout.

| Screenshot | Before / after mapping |
|---|---|
| [English](en.png) | Before left / after right; slide 8 top, slide 9 bottom |
| [简体中文](zh.png) | Before left / after right; slide 8 top, slide 9 bottom |

Native frames are 1280×720; contact sheets are 2560×1520 with labels. All eight frames passed active-page and slide/SVG-text bounds checks; both final contact sheets were visually inspected. Original SVGs are embedded in the single-file HTML decks. No upstream artwork was copied. Paper figure reading and pinned author source paths/hashes are in the [dated note](../../../llm-transformer-architecture-course/references/byte_chunk_accounting_2026_11.md).

Actual CPU work: deterministic byte/BPE/fixed/toy-prefix-entropy segmentation on three common inputs, raw-byte reconstruction and causal scalar smoothing. Twelve rows retain exact input/piece bytes, configuration and source/corpus digests. Fifteen L2 tests passed on Python 3.12.14 (six original, nine new), including deliberately future-reading negative controls. Two fresh report invocations were identical; all digests were verified. The original tokenizer and corpus files are unchanged.

Neural local/global/router/head counts are **declared architecture positions**, not executed FLOPs or measured runtime. No author package, trained BLT/H-Net model, learned routing, quality/BPB, memory or GPU timing was run. This completes the bounded reading/fixture portion; those target-backed ARC-2 extensions remain open.

Validation: course consistency (6 courses); site build/link check (1,141 pages); all-deck browser checks (271 decks); exact changed-slide/count comparison; source/corpus/report repeat checks; frame bounds; `git diff --check`. Existing dependency pins/trackers remain untouched. Approval is required before merge; auto-merge must remain disabled.
