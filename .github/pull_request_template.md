## What changed


## Why


## Prompt history

<!-- The user's requests that led to this PR, quoted with their dates, and the instructions the agent
     followed (AGENTS.md, coordinator notes). For a PR written by hand, say so. -->


## Verification

- [ ] Branch is rebased on the latest `origin/main`
- [ ] One task in this PR; no stacking on another unmerged branch
- [ ] Branch's PR state checked before the last push (not already merged)
- [ ] `python scripts/check_course.py`: sources in the bibliography, tracked versions current, README layout and slide counts match
- [ ] `node scripts/check_slides.mjs`: no console errors, slide numbering continuous
- [ ] `python scripts/check_site.py`: relative links resolve; external references checked
- [ ] Changed lab code runs (or the missing prerequisite is stated)
- [ ] Course README syllabus and layout match the files
- [ ] If this PR adds or edits an English deck in a course with translations (`slides/zh/` or another `slides/<lang>/`), the twin in every language folder is updated in this PR and each translated link stays on the same README row

## Merge

Squash merge into `main`, then delete this branch.
