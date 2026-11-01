# Teaching-readiness response and implementation order

Prepared 2026-11-01 in response to the [review](teaching_readiness_review.md) merged in PRs #88/#89. This response starts with the review's foundation pilot before extending spoken material across the full course. It does not claim that four revised lectures make all 41 lectures classroom-ready.

## What the pilot changes

| Finding | Implemented response | Limit / remaining work |
|---|---|---|
| F1: no spoken layer | Every slide in decks 01–04 has English/Chinese instructor waymarkers: explanations, board calculations, misconceptions and questions. | 76 populated notes per language after adding the teaching pages. These are preparation notes, not full verbatim recording scripts; the other 37 decks still need topic-specific notes. |
| F1: notes inaccessible | All 82 compute-course deck files embed offline S/P controls, a separate presenter window, synchronized current/next titles, and start/pause/reset timer. | Popup fallback and inline mode are supplied. No claim of physical dual-display or classroom testing. Timer resets with page reload. |
| F2: no closure | Each foundation lecture ends with three takeaways, an open question and a linked next lecture. Existing lab handoffs remain. | Extend after an instructor pilots the pattern. |
| F4: supplemental scheduling | The syllabus now maps all supplements 32–40 to insertion points and explicitly describes excerpt/replacement/extra-session choices. | Durations are estimates. Full supplement coverage requires additional time or declared lost core coverage. |
| F5: weak source framing | Each foundation lecture has a source page; deck 01's assumed-rate examples explicitly label analytical scope. | Add source pages to 00 and 05–14 next, using existing primary references where sufficient. |
| F7: passive lecture flow | One prediction checkpoint precedes the numerical answer in each foundation lecture; opening slides show prerequisite/support relationships. | Validate discussion timing and misconceptions with students before scaling the pattern. |

The pilot uses existing cited course mechanisms. Numerical checks retain decimal FLOP/rate units and binary memory units where stated. No new device benchmark, timing estimate disguised as a measurement, or hardware qualification is introduced.

## Review claims that need a narrower reading

The historical headline baseline of 41 English decks / 690 slides agrees with the checkout before this response. However, deck 40 has **22**, not 13, slides; the F6 per-slide density claim based on 13 pages should not drive a blind split. Its code/table pages still deserve projector and pacing review. The syllabus already contained a separate placement paragraph for deck 36, and build-plan supplements contained several placement notes; the actual gap was a consolidated, time-budgeted schedule, especially for 34/35/40.

“Zero recap” and “zero checkpoint” describe a missing consistent teaching pattern, not proof that no existing page asks a question or points onward. For example, deck 01 already asked for a disconfirming experiment and deck 40 linked related lectures. The pilot adds explicit prediction-before-reveal and final recap patterns rather than deleting useful existing handoffs. The original review is preserved as an audit record.

## Next bounded batches

1. **Pilot acceptance.** An instructor rehearses 01–04 with the presenter window and records actual duration, unclear notes and whether students can answer the closing questions. Compare early predictions with final explanations. Fix the pilot before expanding scripts.
2. **Dense supplement edits (F6).** Inspect 34/35/40 page by page at projector size. Keep a teachable mechanism path, move catalogue/reference rows into a linked handout and record what is optional. Split only if rehearsal supports a second session; maintain bilingual twins and navigation/counts together.
3. **Mechanism figures (F3).** Start with deck 10 double-buffer ownership and deck 17 prefill/decode timing, then attention, KV paging and collectives. Each diagram needs explicit dependencies/units, an accompanying question, a primary reference and accessible labels. Expand the generator to multiple figure entries per deck before promising 2–3 figures in every affected lecture.
4. **Sources and notes by part (F1/F5).** Add lecture-specific source pages to 00 and 05–14, then write Part II notes against the illustrated mechanisms. Repeat per part after rehearsal. Choose full scripts only when recording/delegated delivery actually requires them.
5. **Course-wide closure/checkpoints (F2/F7).** Extend the validated opening/prerequisite, prediction, recap and next-lecture pattern. Do not add generic filler questions solely to satisfy a count.

Each batch should be one short-lived branch and one PR from refreshed main, with English/Chinese changes, checks and visual review. These are implementation recommendations, not scheduled background work or a claim that later batches are complete.
