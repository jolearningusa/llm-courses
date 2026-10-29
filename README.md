# LLM Courses

Course material on large language model systems: slide decks, hands-on labs and annotated reading lists,
one self-contained course per directory.

**Website:** <https://jolearningusa.github.io/llm-courses/>. Every course is published there automatically
when a change lands on `main`. Slide decks open full-screen in the browser; use ← / → to move between
slides and F for full screen.

## Courses

| Course | What it covers | Materials |
|--------|----------------|-----------|
| [LLM AgentOS](llm-agentos-course/) ([website](https://jolearningusa.github.io/llm-courses/llm-agentos-course/)) | Treating LLM agents as an operating-system problem: scheduling, dual-plane memory (user memory and KV-cache), tool arbitration and memory security, how open harnesses (Hermes Agent, OpenClaw, Letta, AIOS) implement it, evaluation, and self-improving harnesses (Harness²) | Slide decks, a lab kit with 15 labs and a capstone, bibliography |
| [Efficient LLM Computing Systems](llm-compute-systems-course/) ([website](https://jolearningusa.github.io/llm-courses/llm-compute-systems-course/)) | Inference-first execution stack: PyTorch, compilers, CUDA kernels and the other vendors' programming models, quantization, serving, agentic caching and deployment, with distributed training as the final part; Vera Rubin/NVIDIA software and AMD ROCm comparison | 34 illustrated decks, L0–L15 reference kits, capstone, research notes and 20 potential-project specs; CPU validation and GPU pilot limits documented |

## Repository layout

```
llm-courses/
├── README.md                    # This file
├── AGENTS.md                    # Rules for contributors and coding agents (CLAUDE.md points here)
├── docs/
│   ├── git-workflow.md          # Trunk-based git workflow and GitHub settings
│   └── updating-courses.md      # Consistency and freshness checks; how to update a course with new findings
├── scripts/
│   ├── build_site.py            # Builds the website from every course into _site/
│   ├── check_course.py          # Fails when a course disagrees with itself (bibliography, versions, layout)
│   ├── check_freshness.py       # Reports upstream changes to tracked sources and broken external links
│   ├── check_site.py            # Fails on broken relative links in the built site
│   ├── check_slide_review.py    # Every other Monday: quote slide claims, check them on the web, ask first
│   └── check_slides.mjs         # Opens every deck in Chromium: console errors, slide numbering
├── .github/
│   ├── workflows/pages.yml      # Builds and deploys the website on every push to main
│   ├── workflows/checks.yml     # Consistency, link and slide checks on every pull request
│   ├── workflows/freshness.yml  # Weekly freshness report, kept in one "freshness" issue
│   ├── workflows/slide_review.yml # Every other Monday: ask before changing up to four slides
│   └── pull_request_template.md
└── <course>/                    # One directory per course, for example llm-agentos-course/
    ├── README.md                # Overview, syllabus, repository layout (required)
    ├── slides/*.html            # Self-contained HTML slide decks
    ├── labs/<lab>/              # Lab code and scripts
    └── references/              # Reading lists (*.md) and tracked_sources.json (optional)
```

## Adding a course

1. Create a top-level directory named after the course, for example `llm-rag-course/`. Do not run
   `git init` inside it.
2. Add a `README.md` whose first `#` heading is the course title and whose first paragraph is a one-line
   summary; the website uses both for the course card. Any top-level directory with a `README.md` is
   published, so there is no list to update for the site.
3. Put decks in `slides/` (each deck a single self-contained `.html` file with `.slide` sections and a
   `N / TOTAL` number on each slide), labs in `labs/` and reading lists in `references/`.
4. Add a row to the [Courses](#courses) table above.
5. Preview and check locally, then open a pull request as described in [AGENTS.md](AGENTS.md):

   ```bash
   pip install markdown pygments
   python scripts/check_course.py                                 # bibliography, versions, layout
   python scripts/build_site.py && python scripts/check_site.py   # then open _site/index.html
   node scripts/check_slides.mjs                                  # needs: npm install playwright
   ```

To keep a course current after it lands, list the sources it depends on in
`references/tracked_sources.json`; see [docs/updating-courses.md](docs/updating-courses.md).

Large courses go in several pull requests (outline, then modules, then labs), one at a time.

## Contributing

`main` is the only long-lived branch. Each change is one short-lived branch and one pull request,
squash-merged into `main`. The full rules are in [AGENTS.md](AGENTS.md) and
[docs/git-workflow.md](docs/git-workflow.md).

Each course carries its own license file (the AgentOS course is Apache-2.0).
