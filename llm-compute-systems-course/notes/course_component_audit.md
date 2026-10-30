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
| Deployment naming | AOTInductor/ExecuTorch planned and notes available | Disambiguate “AOI,” add Torch-TensorRT and CUDA delegate composition | Primary deployment documentation; L11 CPU artifacts now validated; other targets unverified |
| Agent lifecycle hints | Cache validity/retain/offload/recompute background | Add current RFC/implementation distinction and existing agentic-api context | vLLM #52113 open RFC at review date; not execution evidence |
| DeepInfer | Not a central course dependency | Source-grounded optional specialized-engine/service case | Pinned README/support review; no model/target run |
| Cross-device optimization | NVIDIA/AMD core coverage | Add bounded edge/backend comparison as an optional extension | Primary backend entry points; hardware-specific execution pending |
| New projects | Capstone scope existed | Separate potential-project catalogue/specifications from released lab code | [Twenty independent specifications](../projects/POTENTIAL_PROJECTS.md); no project implementation implied |

## Integration guide

Use [optimization techniques](optimization_techniques.md) alongside decks 03–14 for diagnosis and implementation choices, decks 15–21 for serving, placement and agent policy, decks 22–24 for deployment and vendor integration, and decks 25–30 for training state and whole-job efficiency. Deck 32 provides the cross-topic optimization section, and deck 33 compares programming models across vendors and edge targets. Keep modeling background concise; inference is the core and training at scale is the final part (renumbered 2026-10-28).

The practical addition to L4 is a source-comparison worksheet: identify a library baseline and two eligible kernel/compiler choices, explain their layout/synchronization responsibilities, then decide whether a hardware experiment is warranted. Unsupported or unavailable targets remain source-reading exercises. A project proposal must supply a new testable objective beyond wrapping an existing tuner or benchmark UI.

## Course-wide review — 2026-10-28

After deck 33 (PR #71) and the inference-first renumbering (PR #72), every deck, note and handout was searched for version, date and support-status claims, and each claim was checked against its current primary source. Most content teaches mechanisms and was still accurate; the changes below are the stale or newly relevant facts.

| Claim checked | Finding on 2026-10-28 | Action |
|---|---|---|
| Operator/kernel, fusion versus graph replay, edge compilers | Not covered before the user's questions | Slides added to decks 05, 06 and 23 (PR #71); deck 33 links from 07, 24 and 32 |
| PyTorch lab pin 2.9.1 | PyPI is at 2.14.1; the weekly freshness report flags it | Pin kept: a compatibility pilot needs the tested macOS environment, and the PyTorch wheel index is blocked from the authoring sandbox |
| AOTInductor API (deck 22) | 2.14 docs still document `torch._inductor.aoti_compile_and_package` | Slide notes the 2.14 status |
| ExecuTorch (deck 23, L11) | Stable docs are 1.5; PyPI 1.5.1; CUDA backend still described as AOTInductor plus Triton | Slide notes the current release; lab pin 1.0.0 kept |
| TorchAO version (bibliography) | PyPI 0.18.0 | Bibliography updated |
| vLLM session-aware hint RFC #52113 (deck 21) | Still open, no linked PR | No change |
| TensorRT-LLM Rubin support (NVIDIA note) | Support matrix still does not name Rubin | No change |
| PyTorch/XLA, Pallas-to-Triton | PyTorch/XLA deprecated for TorchTPU; Pallas-to-Triton deprecated | Recorded in deck 33; no other course file relied on either |

## Outstanding validation

CPU reference/numerical tests and analytical simulations have bounded, recorded scopes in their handouts. CUDA reductions, accelerated kernel/library paths, FSDP/cluster experiments, real engine comparisons and edge targets remain unverified on this authoring host. Core P13–P16 materials are authored; student/hardware pilots remain on the build plan. This audit does not turn those into completed experiments.

## Autoregressive bottleneck review — 2026-10-30

Decks 01, 04, 15–21 and 30 already explain individual performance mechanisms. A connected diagnosis was missing: serial output dependence does not establish GPU idle time, low FLOP utilization does not establish host starvation, and vocabulary selection differs from projection, sparse-attention top-k and MoE routing. [Deck 36](../slides/36_inference_bottlenecks.html), [literature review](../references/inference_bottlenecks_review.md) and [worksheet](../labs/kit/inference_bottleneck_worksheet.md) fill that gap. The English/Chinese supplement includes a dependency/batching vector diagram and constructed budgets. P26 extends the appendix to 26 detailed plans; no accelerated experiment was run for this addition.
