# Original mechanism illustrations

The managed set for the 41 decks contains four diagrams per deck: 41 existing companion figures plus 123 text pages rewritten in place. All 164 concepts have English and Chinese editable SVG assets (328 files). HTML embeds SVG directly and remains self-contained offline. Detailed original explanations and instructor notes remain accessible with **S** or **P**.

The [source and reading guide](../../references/figure_sources.md) identifies the primary-source illustrations/documentation, destination slides and interpretation limits. New artwork is original simplified geometry, not traced or redistributed upstream figures. The inventory distinguishes external mechanism readings from course-owned experiment/verification designs. Existing hardware specification tables and pinned APIs are preserved.

- `build_illustrations.py`: the 41 established companion drawings and unified regeneration entry point.
- `visual_specs.json`: bilingual labels/captions, in-place page anchors, source assignments and drawing parameters for the 123 replacements.
- `visual_explanations.py`: editable mechanism layouts and notes-preserving page replacement.
- `manifest.json`: all 164 concepts, deck names and asset paths.
- `*.svg` / `zh/*.svg`: localized standalone assets for reuse.

From the repository root:

```sh
python3 llm-compute-systems-course/slides/illustrations/build_illustrations.py
```

Regeneration preserves other figures in a deck, all retained explanations and instructor notes, and continuous numbering. Running it twice must leave all 82 decks, 328 SVGs and the manifest unchanged. Use this unified entry point rather than running the in-place helper alone.

Color supports the labels: blue/teal identify work/data, amber retained state/constraints and red rejected or unsafe paths. Equal ledger boxes do not encode sizes. Schedules show logical dependencies, not milliseconds. Small head/rank/lane/block examples are teaching subsets. Power plots and occupancy grids are explicitly schematic, not measurements or chip floorplans.

Dated manual supplements, including deck 24 slides 6/7, sit outside the 164 managed concepts. Their attribution and scope are listed in the source guide; regeneration must preserve them.
