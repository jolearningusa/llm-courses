# Validation record

## Blueprint — 2026-10-31

This stage defines contracts and teaching scope only. No executable lab, board firmware, runtime adapter, GPU/FPGA run, formal-tool proof or power/thermal result is claimed. Primary sources support mechanisms rather than validation of the future implementation.

The functional-reference release will record exact commands, executed regression/scenario results, bounded explorer scope and browser/site checks. Hardware extensions remain separately marked unsupported/not-run until a real supported target is qualified. Student pacing estimates require a later pilot.

## Functional reference — 2026-10-31

Executed with Python 3.9.6 on the macOS authoring host, standard library only:

```sh
cd ai-accelerator-software-course/labs
python3 -m unittest discover -s tests -v
python3 experiments.py scenarios
python3 experiments.py explore
python3 experiments.py fuzz --seed 17 --steps 300
python3 experiments.py report --output result.json
```

All 24 tests pass. The test suite exercises 20 seeds × 300 state actions (6,000 invariant checks), plus reproducibility and negative controls. Five deterministic scenarios validate copy, modulo-256 add-one, lost-interrupt polling recovery, DMA error without output write, and watchdog/reset/stale-completion handling. The generated report exercises an additional ten seeds × 200 actions and carries source hashes, Python version and model-tick scope; [executed example](labs/examples/functional_report.json).

The bounded abstract explorer has five reachable states under each searched safe/unsafe policy before termination/counterexample; safe depth-six search reports no counterexample in that small model, while the deliberately unsafe rule produces `submit → release`. Depth one cannot reveal that two-action bug. This is neither an external formal-tool proof nor a weak-memory/concurrent-hardware model. Assertions must remain enabled.

The functional gate passes its required model checks and the hardware gate rejects promotion. Real hardware and power/thermal are explicitly not-run. Descriptor/API rejection, bounds, partial-copy tails, exclusive ownership, timeout pinning, queued cancellation, double completion/reap, FIFO and reset-generation reuse have dedicated tests.

No real DMA, MMIO, IRQ controller, kernel driver, firmware binary, FPGA/RTL, sanitizer, formal tool, HIL, silicon performance or physical safety qualification was executed. The model assumes atomic quiesce/reset and coherent writes; target adapters must establish real equivalents. Student pacing remains an estimate.

## Publication checks — 2026-10-31

- Course consistency: three courses, zero problems.
- Site build/link audit: 454 pages, zero broken relative links, unlisted decks or missing deck navigation.
- Full browser audit: 126 decks, zero console/numbering problems.
- New-course layout audit: all 160 English/Chinese slides rendered at 1280×720, with no detected text/diagram bounds violations; contact sheets and representative full-size slides visually inspected.
- SVG regeneration is idempotent: sixteen decks and eight editable SVGs unchanged on a second run.
- All twelve primary-source reference URLs returned HTTP 200 on the publication check.
- Example report source hashes match the shipped implementation. Functional reports are teaching evidence, not authenticated release attestations.

The documented lab commands run from the course's `labs/` directory. The 24-test suite passed from that directory; a repository-root discovery attempt without the lab import path was not a supported invocation.

## Mechanism-figure revision — 2026-11-01

All eight decks retain ten slides in each language. Four pages per deck now use original mechanism schematics: 32 concepts, 64 localized SVG assets, embedded in 16 standalone HTML decks. Detailed original explanations are retained in offline speaker notes. The repository's existing presenter controls now work in this course too; the browser presenter check verifies all 98 embedded copies across both systems courses and exercises the companion's bilingual notes, navigation and presenter popup.

The [source inventory](references/figure_sources.md) identifies each upstream diagram/section and each destination slide. Inspected the official IREE runtime component diagrams, Linux DMA address-space illustration, OpenTitan OTBN testbench SVG and Zephyr fixture SVG with their documentation. Artwork is original and simplified; no upstream image assets are redistributed. Model state/ABI figures refer to the course-owned lab, not vendor hardware.

Rendered all 160 slides at 1280×720 and inspected all 64 bilingual mechanism pages in contact sheets, with full-size checks for dense labels and timing diagrams. Corrected connector/text crossings and dim labels. Layout bounds passed. Regeneration was checked by hashing all 80 HTML/SVG outputs before and after a second run; no output changed. Course consistency, full 128-deck browser checks and site links passed. No lab implementation or recorded functional report changed. No target/hardware measurements were performed; the power plot is schematic.
