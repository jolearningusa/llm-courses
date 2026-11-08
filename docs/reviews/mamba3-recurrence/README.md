# Optional Mamba-3 recurrence — visual review, 2026-11-08

Prepared from latest main `903940f` after confirmed merge of PR #126. One bounded independent ARC-1 mechanism/algebra update, awaiting user approval; no auto-merge.

[English before/after](en.png) and [Simplified Chinese before/after](zh.png): before left, after right. Rows are deck 23 slides 9 and 10. Eight native 1280×720 frames were captured, active slide numbers/bounds checked and final layouts inspected. Both twins remain ten slides; original Mamba/Mamba-2 evidence on slides 1–8 is intact.

Original inline SVGs show the two-endpoint/complex recurrence and per-token MIMO rank around one core state. The matrix fragment explicitly omits full-block mechanisms. The complex caption states its actual coefficient formula. Original closing critique questions are preserved in L6. The pinned author cache reading distinguishes core SSM storage from rank-dependent auxiliary state.

Validation: eight new standard-library tests on Python 3.9.6; fourteen total L6 tests on Python 3.12.14 / torch 2.9.1 CPU, one thread. PyTorch warned NumPy was absent; no NumPy conversion was used. Literal states [1,4], independent unrolled comparison, causal prefix, chunk carry negative control, rotation equivalence and literal MIMO shapes were checked. No historical multi-seed model table, trained Mamba-3 quality, optimized SSD kernel or GPU result was reproduced.

Primary reading: Lahoti et al. 2603.15569v1 §§3.1–3.3 / Appendix C; author module at `e9594ce1c732d97440f0332fdc43170a2294dbfa`, `step`/cache allocation. Course check: six courses, zero problems. Site check: 1,119 pages, zero broken relative links/unlisted decks/missing home links. Slide check: 271 decks, zero problems.
