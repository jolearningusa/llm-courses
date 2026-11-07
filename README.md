# LLM Courses

Course material on large language model systems: slide decks, hands-on labs and annotated reading lists,
one self-contained course per directory.

**Website:** <https://jolearningusa.github.io/llm-courses/>. Every course is published there automatically
when a change lands on `main`. Slide decks open full-screen in the browser; use ← / → to move between
slides and F for full screen.

## Courses

| Course | What it covers | Materials |
|--------|----------------|-----------|
| [ML 101: Neural Network Foundations](ml101-neural-networks-course/) ([website](https://jolearningusa.github.io/llm-courses/ml101-neural-networks-course/)) | The foundations before LLMs, in the spirit of a first university course on machine learning and neural networks: linear and logistic regression, gradient descent and backpropagation, MLPs and initialization, then optimizers, regularization, CNNs, RNNs, LSTMs and seq2seq attention, ending with a bridge to the Transformer course | Complete: course map and 20 decks in six parts, CPU labs L0–L6 in NumPy, syllabus, bibliography |
| [LLM AgentOS](llm-agentos-course/) ([website](https://jolearningusa.github.io/llm-courses/llm-agentos-course/)) | Treating LLM agents as an operating-system problem: scheduling, dual-plane memory (user memory and KV-cache), tool arbitration and memory security, how open harnesses (Hermes Agent, OpenClaw, Letta, AIOS) implement it, evaluation, and self-improving harnesses (Harness²) | Slide decks, a lab kit with 15 labs and a capstone, bibliography |
| [Efficient LLM Computing Systems](llm-compute-systems-course/) ([website](https://jolearningusa.github.io/llm-courses/llm-compute-systems-course/)) | Inference-first execution stack: PyTorch, compilers, CUDA kernels and the other vendors' programming models, quantization, serving, agentic caching and deployment, with distributed training as the final part; Vera Rubin/NVIDIA software and AMD ROCm comparison | 41 illustrated decks in English and Chinese, L0–L15 reference kits, capstone, research notes and 26 detailed project plans; CPU validation and GPU pilot limits documented |
| [AI Accelerator Software Systems](ai-accelerator-software-course/) | Runtime, firmware, driver contracts, queues/DMA, lifecycle, verification and AI-assisted development | 8 illustrated bilingual decks, L0–L7 handouts, executable CPU virtual accelerator, 24 tests and failure/state experiments; real-target qualification separate |
| [LLM Transformer Architecture](llm-transformer-architecture-course/) ([website](https://jolearningusa.github.io/llm-courses/llm-transformer-architecture-course/)) | The architecture papers behind language, vision, diffusion and speech models, one deck per paper: the 2017 Transformer and its three families, tokenization and positions (RoPE, ALiBi, YaRN), chat templates, attention variants (MQA, GQA, MLA) and the KV cache, MoE, newer efficiency designs (SSMs, hybrids, sparse attention), and the Llama, Qwen, DeepSeek and Gemma reports | Course map, all 43 paper and capstone decks, labs L0–L9, syllabus, bibliography of 70 checked sources; every deck also in Simplified Chinese |
| [LLM Training and Post-Training](llm-training-course/) ([website](https://jolearningusa.github.io/llm-courses/llm-training-course/)) | How language models are trained, one paper per method: pre-training recipes and data, training at scale, long context, SFT and LoRA, RLHF and DPO, RL for reasoning (GRPO, RLVR), distillation, safety tuning and evaluation | Course map, Part I decks 01–04 (AdamW and schedules, batch size and μP, stability, Muon), Part II decks 05–08 (web corpora, dedup and filtering, mixtures, code, math and synthetic data), Part III decks 09–10 (parallelism, low precision), Part IV decks 11–12 (long context, continual pre-training), Part V decks 13–15 (instruction tuning, instruction data, Tülu 3), Part VI decks 16–17 (adapters and LoRA, QLoRA and DoRA), labs L0–L4, syllabus (34 decks, 10 CPU labs), bibliography of 97 checked sources; later parts to follow |

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
