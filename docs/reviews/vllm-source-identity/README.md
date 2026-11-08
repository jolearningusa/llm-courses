# vLLM source-identity gate — visual review, 2026-11-08

Prepared from latest main `6b41378` after confirmed merge of PR #127. No overlapping open PRs were present. One independent CMP-2 prerequisite update, awaiting user approval; no auto-merge.

[Before/after contact sheet](changes.png): English first row, Simplified Chinese second row; before left, after right. Four native 1280×720 frames were captured. Active page number, slide bounds and SVG text bounds were checked; both final images were visually inspected. Only deck 35 slide 20 changed in each twin; both retain 21 slides and the dated version table on slide 4 is intact.

Original SVGs show PyPI archive/tag/declared commit and two-file equality evidence separately from not-run runtime gates. The source note and JSON manifest preserve full identities, digests, exact URLs and read-only reproduction steps. The source archive was hashed and read without extraction, installation, import or build; the two sampled source members were byte-equal with the full Git commit. Published wheel metadata was recorded, but wheels were not downloaded. No complete-source/build-provenance proof or installation/runtime/connector/performance result is claimed. CMP-2 remains open beyond this first source gate.

Validation: course consistency — six courses, zero problems; site build/check — 1,121 pages, zero broken relative links/unlisted decks/missing home links; browser slide check — 271 decks, zero problems. All trackers and requirement files are byte-identical to `origin/main`. No lab implementation changed. The source checks used Python 3.9.6 standard-library archive/hash/metadata readers plus the GitHub API. No GPU, Linux deployment or upstream connector test was run.
