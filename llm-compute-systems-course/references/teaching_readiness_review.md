# Teaching readiness review of the slide decks

Prepared 2026-10-31 from a page-level audit of every deck in `slides/` and `slides/zh/`, re-verified against `main` at #87 (2026-11-01, decks 37–40 included). Scope: this course only. Method: scripted extraction of per-slide word counts, table/code/figure/citation density and speaker-note state across all 41 decks (690 slides), followed by full reads of representative decks (01, 10, 35, 40). Numbers below are measurements, not judgments; findings cite them.

The repository's own checks pass (`check_course.py`: 0 problems; bilingual twins, README rows and the layout tree all agree). The gaps below are therefore about lecture delivery, not mechanical consistency: the decks are an outline of what to cover, not yet the material to teach from.

## Headline metrics

| Metric | Value |
|---|---|
| Decks (English) / slides | 41 / 690 |
| Total words on slides | 25,393 |
| Median words per slide | 27 (mean 37) |
| Slides under 25 words | 242 (35%) |
| Slides above 130 words | 11 |
| Slides without a `.cite` line | 196 |
| Speaker-note asides with content | 0 of 690 (English and Chinese) |
| Decks with a closing recap/summary/next-lecture page | 0 of 41 |
| Decks mentioning another lecture or Part | 15 of 41 |

## Findings

### F1 · There is no lecture script anywhere (blocking)

All 690 `<aside class="notes">` elements are empty, in both languages. The template also hides them (`.notes{display:none}`) and the keyboard offers only ←/→, Home/End and F — there is no notes toggle and no presenter view, so even filled notes would be invisible at the lectern. With a median of 27 words per slide and ~17 slides per 75–90 minute lecture, the spoken layer — derivations, common misconceptions, live examples — exists nowhere. Two instructors would teach two different courses from these files.

Fix: write full notes for Part I (decks 01–04) as a template first, and add a presenter mode (`S` toggles notes; presenter view with a timer) in the same effort. Then decide the depth for the remaining decks based on who teaches them.

### F2 · No deck closes the loop (blocking)

Every deck runs: p1 outcomes → content → last page jumps to the lab handoff. There is no recap, no unresolved-question page, no pointer to the next lecture — in any of the 41 decks. Two related inconsistencies: p1 carries an `Outcomes` line in 35 of 41 decks (missing: 00, 36, and all four newest decks 37–40), and the final page hands off to a lab in 36 of 41 (not: 00, 31, 32, 35, 36).

Fix: one closing slide per deck — three takeaways, one question the lecture did not answer, what the next lecture builds on. Roughly three minutes of speaking; the highest ratio of value to effort in this list.

### F3 · One figure per seventeen slides

Each deck embeds exactly one companion SVG. Tiling, double buffering and TMA, collectives, KV paging and chunked prefill are spatial or temporal structures that prose describes poorly — for example, the 137.44 GFLOP → 1.374 ms versus 33.55 MFLOP → 33.6 µs contrast in deck 01 (slides 10–11) is a picture, not two paragraphs. The `slides/illustrations/build_illustrations.py` generator already exists, so extending coverage is cheap relative to other work.

Fix: add 2–3 mechanism figures (timing/state diagrams first) to each of Part II (05–12) and Part IV (15–21).

### F4 · Four supplements have no lecture slot

`SYLLABUS.md` provides 32 lecture slots (decks 01–31 plus the capstone); decks 34, 35, 36 and the new deck 40 appear nowhere in it, and `BUILD_PLAN.md` does not place them either — even though deck 35 self-describes as "Supplement to Part IV, after deck 16". Anyone scheduling from the syllabus cannot place them; teaching them costs four additional sessions.

Fix: add a "Supplements" section to the syllabus listing each supplement's possible insertion points and the trade-off of teaching or skipping it.

### F5 · Citations are strong late, weak early

196 slides carry no `.cite`. Deck 00 is at 20% and deck 13 at 17%, while Part IV onward is near 100%. Deck 01 slide 9 labels its 137.44 GFLOP as a constructed example, but slide 10's 1.374 ms carries neither label nor source — and L0 asks students to reproduce exactly these numbers.

Fix: do not chase per-slide citations; add one "Sources for this lecture" page per deck for 00 and 01–14.

### F6 · Density polarization: two courses in one

Core decks run 400–800 words over 15–22 pages; the newer supplements run far denser: deck 35 is 2,183 words over 19 pages (115 words/slide, 16 uses of `.dense`/`.small`), deck 33 is 1,850 over 19 (12 tables), deck 34 is 1,273 over 13, and the new deck 40 is 1,275 over 13 — at ~98 words per slide on the standard 23px type, with no compressed classes to absorb it. Six decks (09, 10, 14, 33, 34, 35) define compressed type — tables down to 18px, body text to 20px — breaking the projector floor exactly on the kernel, quantization and engine-comparison material. The worst single slide is 159 words (deck 10, "Parallel without locks").

Fix: either split 34/35 into two lectures each or cut to core-deck density; keep `.dense` tables to three readable rows and move the rest to a handout.

### F7 · No in-class checkpoints, thin cross-lecture navigation

No slide in 690 asks students to predict an outcome before it is revealed or to discuss in pairs; the course's `predict → implement → validate → profile → explain → compare` philosophy lives entirely in the labs, while the 90-minute classroom has no checkpoint. Only 15 of 41 decks reference another lecture or Part, although the dependency chain (15 → 16 → 17) is strong. Deck 35 is the counterexample: its comparison tables cite "deck 01" and "decks 15, 17" cell by cell, which is exactly the pattern to generalize.

Fix: one "Predict" slide per deck (a number students guess before it is shown — the spirit already exists in deck 01's "An experiment that can disprove the prediction"), and a prerequisites/supports line on each deck's first page.

## What already works

- The p1 outcomes-plus-motivation line is the most consistent structure in the course, and half of the fix for F2.
- The `.cite` habit and "constructed example, not measured performance" self-labeling (Parts IV onward).
- Strict bilingual twins enforced by `check_course.py` — every change here has a known, bounded cost.
- Lab linkage: most decks end on a specific numbered exercise (L0–L15).

## Suggested order of work

| Priority | Item | Size |
|---|---|---|
| P0 | Speaker notes + recap pages for decks 01–04 as a template | 4 decks |
| P0 | Presenter mode (notes toggle, timer) | one JS/CSS change, all 82 deck files |
| P1 | Split or slim decks 34/35/40; place supplements in `SYLLABUS.md` | 2–3 decks + 2 docs |
| P1 | 2–3 mechanism figures per deck for Parts II and IV | ~12 SVGs |
| P2 | "Sources for this lecture" page for decks 00, 01–14 | 15 decks |
| P2 | One "Predict" slide per deck; prerequisites line on p1 | 41 decks |

Every item must update the `slides/zh/` twins in the same PR (enforced by `check_course.py`).

## Open decision before P0

Who delivers the lectures? Self-delivery needs waymarker notes (2–3 sentences per page: what to say first, what to draw on the board, which number students compute themselves); delegated delivery or recording needs self-contained full scripts — an order of magnitude more work, best batched part by part. This review was produced with a 16-week graduate/engineering-course delivery in mind.

## Implementation response — 2026-11-01

See the [response and remaining batches](teaching_readiness_response.md) for the 01–04 teaching pilot, presenter controls, supplemental schedule and remaining work. It also corrects the deck-40 page-count interpretation and distinguishes pre-existing placement/questions from the absence of a consistent teaching pattern. The metrics above are the historical pre-pilot audit, not measurements of the revised decks.
