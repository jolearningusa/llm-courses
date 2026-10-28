# Original SVG teaching illustrations

These figures are editable vectors with native text, paths and shapes. They are conceptual diagrams and constructed accounting examples, not photographs, hardware die layouts or measured performance charts. Colors distinguish computation (blue/teal), retained state or synchronization (amber), and inactive/context elements (gray); labels carry the meaning independently of color.

Every released deck 00–24 has one companion visual slide next to the relevant mechanism. The original slide remains intact. Each figure has an accessible SVG title/description, a concise teaching caption and the neighboring slide's primary-source citation. Standalone `.svg` files are supplied for reuse; the HTML also embeds their contents so decks remain self-contained offline.

`manifest.json` maps each figure to its deck and teaching anchor. `build_illustrations.py` contains the editable drawing source. To regenerate all current figures and their companion slides:

```sh
python3 llm-compute-systems-course/slides/illustrations/build_illustrations.py
```

The operation is idempotent and renumbers slides. When adding/rebuilding lectures, regenerate diagrams, synchronize the README slide counts/tree and master links, then run the repository course/site/slide checks. New topics require their own explicitly reviewed figure entries; the generator does not invent diagrams for unknown decks.

## Review questions

Can a student follow the data or state ownership without reading a paragraph? Are time bars labeled schematic when they are not measurements? Do shard/shape/precision labels match the lecture contract? Are transfer and completion boundaries unambiguous? Do the caption and source separate the mechanism from evidence of performance?
