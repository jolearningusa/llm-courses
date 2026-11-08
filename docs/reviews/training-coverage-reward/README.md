# Coverage and reward controls — visual review

Prepared 2026-11-07 from clean main `66062b6`, after confirmed merge of PR #123. This independent update awaits user approval; no auto-merge.

[Before/after comparison](changes.png): before left, after right; rows are deck 26 slide 9 and deck 28 slide 8. Native frames are 1280×720, with unchanged 9-slide counts. The training course currently has English decks only.

The first original SVG is a schematic control/relative-reward diagram. The second is a measured table computed from all twelve CPU runs in L7's `coverage_reference.json`, with no selected checkpoint. It separates strict held-out pass@k from the deliberately weak width reward. The complete seed-level outcomes, costs and negative result remain in the handout and ledger. Old closing derivations are retained in L7 rather than deleted.

Primary reading: Clay et al. 2608.24949v1 §§4–5; DAPO §3.2; Chen et al. 2107.03374v2 §2.1 / Appendix A. This is a local arithmetic experiment, not a reproduction of those LLM results.

Validation: twelve L7 tests (six existing, six new); all twelve experiment arms executed on Python 3.12.14 / torch 2.9.1, CPU, one thread, float32. NumPy was absent and PyTorch warned at import; no NumPy operation was used. Course check: 6 courses, 0 problems. Site check: 1,113 pages, 0 broken relative links/unlisted decks/missing home links. Slide check: 271 decks, 0 problems. Four native frames were captured, visually inspected and checked for bounds/active slide numbers. Ledger hashes, paired base states, totals and exact two-slide scope were also checked.
