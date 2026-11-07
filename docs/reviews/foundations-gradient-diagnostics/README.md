# Foundations gradient diagnostics — review

Prepared 2026-11-07, audit FND-1. **User approval is required before merge.** Before screenshots come from main commit `752494c56d0d02952b57c3258979fe0277205d10`; after screenshots are from this change. Each row shows before on the left and after on the right:

1. Backpropagation, slide 10: finite-difference step/error plot.
2. Debugging recipe, slide 4: ReLU kink schematic and executed numeric examples.
3. Debugging recipe, slide 10: separate absolute/relative near-zero error plots.

[Visual comparison](changes.png). Native screenshots were captured at 1280×720 and active slide numbers checked. A local full-size gallery is also supplied in the chat outputs. This course currently has English originals only; both decks remain at ten pages. Plots are derived from the included JSON report using optional Matplotlib; HTML/SVG remains the editable slide source.

Validation: all 17 L1 tests passed (six new diagnostics plus eleven existing tests). Existing MLP tests emitted NumPy matmul runtime warnings on this host but completed successfully; no autograd/MLP implementation was changed. The diagnostic cases passed. Numerical report: Python 3.9.6 / NumPy 2.0.2, optional plot export Matplotlib 3.9.4. No trained-model performance or GPU result is claimed. Course/site checks passed; layout bounds were checked and all six before/after pages inspected.

Retained hand calculations and general debugging prompts are linked from the extended L1 handout. This is a focused diagnostic exercise, not completion of the broader research atlas project.
