## What changed


## Why


## Verification

- [ ] Branch is rebased on the latest `origin/main`
- [ ] One task in this PR; no stacking on another unmerged branch
- [ ] Branch's PR state checked before the last push (not already merged)
- [ ] `node scripts/check_slides.mjs`: no console errors, slide numbering continuous
- [ ] `python scripts/check_site.py`: relative links resolve; external references checked
- [ ] Changed lab code runs (or the missing prerequisite is stated)
- [ ] Course README syllabus and layout match the files

## Merge

Squash merge into `main`, then delete this branch.
