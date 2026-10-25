# Agent instructions for llm-courses

This repository holds LLM course material. Each course lives in its own top-level directory
(for example `llm-agentos-course/`) with its own README, slides, labs and references. Never run
`git init` inside a course directory, and keep shared, repo-wide files (this file, `docs/`,
`.github/`) at the root.

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
4. **Before pushing**, fetch `origin/main` and rebase the branch onto it.
5. Push the branch and open a PR against `main`, filled in from `.github/pull_request_template.md`.
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

## Before pushing

There is no CI yet, so check locally:

- Slides (`*.html`) open in a browser without console errors and slide numbering is continuous.
- Relative links in Markdown and HTML resolve; external references have been checked.
- Lab code that was changed still runs (or the PR says which prerequisite, such as a GPU, was
  missing so it could not run).
- The course README (syllabus, repository layout) matches the files actually in the course.
