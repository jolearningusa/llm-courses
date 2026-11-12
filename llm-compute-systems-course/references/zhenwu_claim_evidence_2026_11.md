# Zhenwu selected product-claim review — 2026-11-12 UTC

Companion to deck 40 [slide 15](../slides/40_inference_accelerator_architectures.html#slide-15) / [slide 16](../slides/40_inference_accelerator_architectures.html#slide-16) and the [Chinese twin](../slides/zh/40_inference_accelerator_architectures.html#slide-15). Bounded Alibaba portion of CMP-4; no hardware, compiler, runtime or model benchmark ran. The 2026-11-01 review remains historical; this recheck applies only to the selected Zhenwu claims.

## Sources and selected observations

| Primary source | Publication / displayed update | Inspected scope and evidence |
|---|---|---|
| [Alibaba May announcement](https://www.alibabacloud.com/en/press-room/alibaba-announces-comprehensive-full-stack-ai?_p_lc=1) | 2026-05-20 | M890: 144 GB GPU memory; 800 GB/s inter-chip link; FP32 through FP4 support claimed; vendor 3× versus 810E. These are announcement claims, not a qualified configuration. |
| [Alibaba September roadmap](https://www.alibabacloud.com/en/press-room/alibaba-unveils-roadmap-on-full-stack-ai-strategy) | 2026-09-22 | V900: 216 GB GPU memory; 1,200 GB/s inter-chip; FP8/FP4 named; vendor 3× versus M890. Mass production/commercial release is forecast for Q1 2027. |
| [PAI model deployment guide](https://www.alibabacloud.com/help/en/pai/use-cases/one-click-deploy-deepseek) | Last updated 2026-10-13 | Resource Quota instructions pair a Zhenwu PPU with SGLang and PPU-810E card type. This documents a deployment choice, not an executed course deployment or numerical 810E specification. |
| [Older full-stack keynote link](https://www.alibabacloud.com/blog/603595) | Unresolved on this recheck | Two browser fetches timed out. Do not interpret failure as proof of deletion, or treat its earlier GPU-family classification as newly verified. The visible revised slides rely on the accessible announcement/deployment evidence instead. |

## Claim boundaries

**Capacity and bandwidth.** Keep the vendor's GB label. The selected announcements do not establish memory technology, binary GiB capacity, allocatable capacity after runtime reservations, packaging granularity or a tested board/driver configuration. Inter-chip GB/s is distinct from HBM bandwidth, switch aggregate bandwidth and full-rack bisection. Link direction and measurement conditions are not supplied by the selected specification sentences; no bidirectional interpretation is inserted. Do not scale capacity or bandwidth by an assumed accelerator count into a usable unified-memory or delivered-network result.

**Precision and ratios.** Named FP formats do not supply TFLOP/s, accumulation precision, sparsity, accuracy, model quality or which path realizes the 3× claim. The two announcements do not disclose a common model/shape/precision/quality/latency/resource/power contract sufficient for a matched cross-generation benchmark. Multiplying them into a verified 9× application result is unsupported. No new PE-grid, systolic or SIMT classification is established by the selected public announcements.

**Status.** M890 is introduced in May and described as released in May by the September source; that is dated vendor evidence, not proof of local inventory, region availability or installed software. V900 remains a September announcement with a Q1 2027 forecast in the inspected source, not an observed commercial deployment. The PAI 810E template is a separate software/service fact; its general engine list does not establish every engine on every card. No cloud resources were provisioned or charged.

## Teaching and remaining work

The original bilingual diagrams show evidence classes and the missing common benchmark contract. Only deck 40 slides 15/16 change; the handout twins receive dated selected-row annotations. Other vendors, earlier arithmetic examples, lab code, measured tables and tracker pins are unchanged. No SAIL repository, SDK/driver support matrix, installable toolchain or Triton/TileLang integration is qualified here; those gates remain open. Other CMP-4 vendor/library rows remain open, and PR #120 still does not complete AGT-2's measured benchmark.

Publication checks cover course consistency, site links, browser numbering, native before/after screenshots and SVG/text bounds. No unit tests mirror prose or arithmetic, and source review is not promoted to execution evidence.
