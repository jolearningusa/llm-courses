# Agent instructions for llm-courses

This repository holds LLM course material. Each course lives in its own top-level directory
(for example `llm-agentos-course/`) with its own README, slides, labs and references. Never run
`git init` inside a course directory, and keep shared, repo-wide files (this file, `docs/`,
`.github/`) at the root.

## Website (GitHub Pages)

`.github/workflows/pages.yml` runs `scripts/build_site.py` on every push to `main` and deploys the
result to GitHub Pages; pull requests only build it. Every top-level directory with a `README.md`
is published as a course, with no list to maintain: READMEs and other Markdown are rendered,
`slides/*.html` decks are served unchanged, and lab files get a highlighted view plus the raw file.
Preview locally with `pip install markdown pygments && python scripts/build_site.py`, then open
`_site/index.html`. Add new repo-wide directories that are not courses to `SKIP_DIRS` in the script.
The root `README.md` describes the repository and is not published as a course (only directories are);
when a course is added, also add a row to its course table. Links from a course to repo-wide files
(`AGENTS.md`, `docs/`) are rewritten to point at GitHub, since those files are not on the site.
Each course README has a "Slide decks" table that links every deck in its `slides/` folder;
`scripts/check_site.py` fails if a deck is missing from it, so add a row with every new deck.
The site is only deployed by this workflow when **Settings → Pages → Source** is "GitHub Actions". If it
is "Deploy from a branch", GitHub's own Jekyll build ("pages build and deployment") also runs on each
push, finishes later and overwrites the site with a plain rendering of the READMEs.

## Git workflow: trunk-based, one task one PR

`main` is the only long-lived branch and must always be in a publishable state. There is no
`develop`, `staging` or long-lived release branch. Full rules: [docs/git-workflow.md](docs/git-workflow.md).

**Required sequence for each task:**

1. Fetch `origin/main` and fast-forward a clean local `main`.
2. Create one short-lived branch for the current task from the verified latest `origin/main`:
   `course/*` (new course or large course change), `content/*` (slides, labs, references),
   `fix/*`, `docs/*` or `chore/*`. Agent sessions that are assigned a branch name (such as
   `claude/*`) use that name.
3. Make the change, keeping it to one intent. Run the checks in "Before pushing" below.
4. **Before pushing**, check the state of the branch's PR (the user often merges within minutes):
   if it is already merged, restart the branch from the latest `origin/main` instead of pushing
   onto merged history. Then fetch `origin/main` and rebase the branch onto it.
5. Push the branch and open a PR against `main`, filled in from `.github/pull_request_template.md`.
   Its **Prompt history** section quotes the user's requests that led to the PR (with dates) and lists
   the instructions the agent followed, so readers can learn how the PR was produced.
6. When the user has authorized the merge, **squash-merge** the PR into `main`.
7. **Delete the branch after the merge**: GitHub deletes the remote head branch automatically
   (see repository settings below); delete it manually only after confirming the merge. Then fetch
   `origin/main`, fast-forward local `main` and delete the local topic branch.
8. **Only then** start the next task, from the updated `main`.

**Critical rules:**

- Never push directly to `main`, never force-push `main`, never fix things on `main`.
- Do not create a branch or PR for the next task until the current PR is merged. Do not stack
  PRs on top of each other's unmerged branches.
- Never reuse a merged branch for new work; restart from the latest `main`.
- A branch should live one or two days at most. Split a large course into several PRs
  (outline, then modules, then labs) rather than one long-running branch.
- When several sessions work on the repo at once, each owns its own files (for example one session
  the slide decks, another the root README and tooling). Do not edit files another open PR changes;
  if that cannot be avoided, wait for that PR to merge and rebase.

## Before pushing

`.github/workflows/checks.yml` runs two checks on every PR, and `pages.yml` builds the site. Run the
same checks locally before pushing:

```bash
python scripts/check_course.py                                 # course agrees with itself (see below)
python scripts/build_site.py && python scripts/check_site.py   # relative links resolve
node scripts/check_slides.mjs                                  # decks: console errors, numbering
python scripts/check_freshness.py                              # optional, needs network: external links, upstream versions
```

- Slides (`*.html`) open in a browser without console errors and slide numbering is continuous
  (`check_slides.mjs`; a deck that continues another, such as a split master deck, must start on the
  slide after the one it continues).
- Relative links in Markdown and HTML resolve (`check_site.py`); external references have been checked
  (`check_freshness.py`, or by hand when a site blocks scripted requests).
- Every arXiv ID cited in a deck, README or lab is in the course's `references/`, tracked facts (such as the
  MCP spec version in `references/tracked_sources.json`) are cited with their current value, and README
  slide counts and the "Repository layout" tree match the files (`check_course.py`).
- Lab code that was changed still runs (or the PR says which prerequisite, such as a GPU, was
  missing so it could not run).
- The course README (syllabus, repository layout) matches the files actually in the course.

## Keeping courses current

The weekly **Freshness** workflow (`.github/workflows/freshness.yml`, `scripts/check_freshness.py`) compares
each course's `references/tracked_sources.json` with upstream (PyPI, npm, GitHub) and checks every external
link, and keeps one open issue labeled `freshness` with the report. To update a course with new findings
(from that issue, a new paper or a release), follow [docs/updating-courses.md](docs/updating-courses.md):
check the primary source, update the bibliography, update `version` and `verified` in the tracked entry, then
the decks and labs, one PR per change.
