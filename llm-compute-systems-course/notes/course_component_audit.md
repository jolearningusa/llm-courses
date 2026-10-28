# Slide and component audit — 2026-10-27

The audit compares the computing-systems course with the relevant optimization discussion supplied by the user. It evaluates coverage and evidence, not a claim that every referenced implementation has been run.

## Findings and changes

| Area | Existing coverage | Gap / action | Current evidence |
|---|---|---|---|
| Teaching illustrations | Mostly prose, formulas and text arrows | Add mechanism-specific SVG companion slides while retaining originals | PR #62: 25 figures across decks 00–24; browser/label/layout review passed |
| Tool taxonomy | CUDA, libraries and compiler topics existed separately | Explicitly distinguish platform, operator library, graph engine and kernel DSL | Supplementary deck 32 + technique map |
| TileLang | CUDA/Triton/CuTe/CUDA Tile in kernel study | Add TileLang tooling, TileKernels distinction and target-support boundary | README/tools review; no installed TileLang/GPU run |
| Automatic search | Predict/profile/measure loop | Teach existing vLLM auto-tune, TileLang tuning and FlashInfer-Bench as baselines | Primary README/docs/paper scope reviewed; no reproduced gains |
| Choosing the optimization layer | Topic-specific mechanisms | Map observed bottleneck to candidate, component, adverse case and acceptance gate | Technique map and controller/evaluator exercise |
| Deployment naming | AOTInductor/ExecuTorch planned and notes available | Disambiguate “AOI,” add Torch-TensorRT and CUDA delegate composition | Primary deployment documentation; L14 CPU artifacts now validated; other targets unverified |
| Agent lifecycle hints | Cache validity/retain/offload/recompute background | Add current RFC/implementation distinction and existing agentic-api context | vLLM #52113 open RFC at review date; not execution evidence |
| DeepInfer | Not a central course dependency | Source-grounded optional specialized-engine/service case | Pinned README/support review; no model/target run |
| Cross-device optimization | NVIDIA/AMD core coverage | Add bounded edge/backend comparison as an optional extension | Primary backend entry points; hardware-specific execution pending |
| New projects | Capstone scope existed | Separate potential-project catalogue/specifications from released lab code | [Twenty independent specifications](../projects/POTENTIAL_PROJECTS.md); no project implementation implied |

## Integration guide

Use [optimization techniques](optimization_techniques.md) alongside decks 03–14 for diagnosis and implementation choices, decks 15–20 for training/state, decks 21–24 for serving/placement, and decks 25–30 for agent policy/deployment/vendor integration. Deck 32 provides the cross-topic optimization section. Keep modeling background concise and preserve the training emphasis as well as inference.

The practical addition to L4 is a source-comparison worksheet: identify a library baseline and two eligible kernel/compiler choices, explain their layout/synchronization responsibilities, then decide whether a hardware experiment is warranted. Unsupported or unavailable targets remain source-reading exercises. A project proposal must supply a new testable objective beyond wrapping an existing tuner or benchmark UI.

## Outstanding validation

CPU reference/numerical tests and analytical simulations have bounded, recorded scopes in their handouts. CUDA reductions, accelerated kernel/library paths, FSDP/cluster experiments, real engine comparisons and edge targets remain unverified on this authoring host. Core P13–P16 materials are authored; student/hardware pilots remain on the build plan. This audit does not turn those into completed experiments.
