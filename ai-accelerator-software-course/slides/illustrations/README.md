# Mechanism diagrams

32 original concepts, each localized in English and Simplified Chinese, replace text-heavy content in the eight decks. Source drawings and scope are documented in the [figure reading guide](../../references/figure_sources.md). The course owns these simplified schematics; upstream figures are linked as references, not redistributed.

- `figure_specs.json`: slide anchors, bilingual titles/captions and source/figure attribution.
- `build_illustrations.py`: editable SVG geometry, in-place slide replacement, notes preservation and offline presenter embedding.
- `accelerator-*.svg`: 32 English figures.
- `zh/accelerator-*.svg`: 32 Chinese figures.

From the repository root:

```sh
python3 ai-accelerator-software-course/slides/illustrations/build_illustrations.py
```

Re-running must be idempotent. HTML slides embed SVG and presenter code directly and work offline. The generator reuses the existing canonical presenter source in `llm-compute-systems-course/slides/presenter_mode.js`; the full repository is needed to rebuild, but each generated HTML file runs independently. Press **S** for notes and **P** for the presenter window. Original slide explanations remain in notes.

Only descriptor field widths are proportional. Timelines use logical event order. Power curves are schematic, not measurements. The CPU model cannot establish real DMA, firmware, weak-memory, HIL or silicon behavior.
