# Editable teaching diagrams

Eight original vector diagrams identify the responsibility/state sequence of the master, six topic units and capstone. Labels, paths and shapes remain editable. These are conceptual teaching diagrams, not hardware schematics, measured timing or physical instrumentation.

Run `python3 ai-accelerator-software-course/slides/illustrations/build_illustrations.py` from the repository root. `figure_specs.json` carries English/Chinese labels and teaching anchors. The standard-library generator updates both language twins, writes standalone English SVG assets and renumbers slides idempotently. Review labels, source context and layout after any change.
