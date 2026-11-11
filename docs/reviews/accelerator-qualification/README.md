# Accelerator qualification reporting — 2026-11-10

Base: `9c8de10` (main after #134). One bounded ACC-2 update. Deck 06 remains 10 slides in both languages; only slides 2 and 4 changed. All other slide sections match the base. EN/ZH sources and original SVGs are embedded in the single-file decks.

| Screenshot | Before / after mapping |
|---|---|
| [English](en.png) | Left before / right after; slide 2 top, slide 4 bottom |
| [简体中文](zh.png) | Left before / right after; slide 2 top, slide 4 bottom |

Each source frame is 1280×720; contact sheets are 2560×1520 with labels. Checked active slide numbers, slide and SVG text bounds, then visually inspected both contact sheets. The original explanations remain in speaker notes. Illustration regeneration preserves both edited decks byte-for-byte.

The new L6 runner actually compiles three Python files to temporary bytecode and executes the existing CPU model. The raw JSON retains separate build/model/physical decisions. Nine new policy tests bring the suite to 41 passing tests; fabricated device records are policy controls only. No target firmware, GPU, Twister, KUnit, HIL, telemetry or physical qualification was executed. The unavailable-board control cannot earn a physical pass even with a claimed device-run stage. Schema consistency and hash/log references do not authenticate logs.

Primary sources: [Twister runner](https://docs.zephyrproject.org/latest/develop/twister/index.html), [Twister native statuses](https://docs.zephyrproject.org/latest/develop/twister/twister_statuses.html), [KUnit introduction](https://docs.kernel.org/dev-tools/kunit/index.html). The [execution note](../../../ai-accelerator-software-course/references/qualification_matrix_2026_11.md) explains scope and the broader course not-run vocabulary.

Validation: 41 lab tests; two fresh fixed-hash-seed reports identical after excluding run IDs; source hashes checked; course consistency (6 courses), site build/link check (1,137 pages), all-deck browser check (271 decks), bilingual frame/bounds checks, illustration regeneration and `git diff --check`. Existing tracker versions and older reports remain unchanged. Real target adapter and physical qualification remain pending. User approval is required before merge; auto-merge must remain disabled.
