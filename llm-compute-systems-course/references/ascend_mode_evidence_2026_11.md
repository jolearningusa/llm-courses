# Ascend AI Core working-mode evidence — 2026-11-12 UTC

Companion to [deck 40 slide 14](../slides/40_inference_accelerator_architectures.html#slide-14) and its [Chinese twin](../slides/zh/40_inference_accelerator_architectures.html#slide-14). Selected Ascend source portion of CMP-4, not an SDK migration or hardware qualification. [Response identities and claim scope](ascend_mode_evidence_2026_11.json) record fetched bytes/digests; these identify observed responses, not immutable publications or verified installations.

## Primary reading cohort

- [Earlier TBE AI Core overview, commercial/800 URL](https://www.hiascend.com/doc_center/source/en/canncommercial/800/opdevg/tbeaicpudevg/atlasopdev_10_0008.html): three compute-unit responsibilities, local storage, movement units and queue/event synchronization. This remains useful historical/background reading; its TBE development constraints and scheduling discussion are not universal Ascend C support claims. No published date was established from the fetched page.
- [Ascend C Basic Architecture, CANNCommunityEdition/850 URL](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html): definitions, working modes, product/programming-mode list, and the Atlas A2 example. This selected version path is the cohort used for the revised slide; it is not claimed to be the newest CANN release. No published date was established; observation is dated 2026-11-12 UTC.
- [Newer 910 English page lead](https://www.hiascend.com/en-sg/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/programming_guide/advanced_programming/hardware_implementation/basic_architecture.md) timed out in the web reader. Search excerpts did not become newly qualified current SDK support. No PDF was used as the implementation source and no vendor figure was copied.

## Selected architecture and product boundaries

In coupled mode, Cube and Vector share one Scalar scheduler. In separated mode, the Cube Core and each Vector Core have independent Scalar schedulers and DMA resources. A group is described as one Cube Core plus N Vector Cores; the guide says to obtain N through hardware platform information, and counts grouped AI Cores by Cube Core. No platform API was called here. One drawn Vector box represents a symbolic group, not a measured 1:1 ratio, a universal 1:2 ratio or a chip floorplan.

The selected guide lists Atlas A2/A3 training and inference families in separated mode. It also distinguishes hardware capability from the programming mode for Atlas 200I/500 A2: hardware can support both modes, but this guide's Ascend C programming scope supports coupled mode. The general occurrence of “A2” therefore cannot select a working mode by itself. These are the selected guide's family labels, not exhaustive SKU/firmware/SDK compatibility certification.

The diagram concerns scheduling resources only. It does not supply device core totals, buffer sizes, instruction throughput, delivered model performance or product availability. Do not multiply a symbolic N into unqueried hardware totals. The original Cube/Vector/Scalar responsibilities remain valid as this guide's abstraction, without implying that every resource is colocated behind one Scalar scheduler.

## Student dossier check and remaining gates

Record exact product, CANN/Ascend C version, mode, platform-reported core counts/ratio and memory sizes before choosing launch dimensions or staging. A dossier claiming “every A2 is separated in Ascend C” conflicts with the selected product caveat; this is a documentation counterexample, not an executed compiler/runtime rejection. Freeze API/dtype/layout/memory/synchronization constraints and compile/run on an eligible target before claiming operator support or portability. Queue independence does not remove data dependencies, and earlier TBE auto-scheduling text cannot establish every manual Ascend C synchronization contract.

Only deck 40 slide 14 changes in both languages. Handout twins and the earlier evidence review receive dated annotations preserving 2026-11-01 history. No lab code, old numerical table, dependency pin or tracker date changes. No CANN installation/import, platform query, generated code, target run or numerical/performance validation occurred. This completes the bounded working-mode source clarification, not all vendor/library claims or target qualification. PR #120 still leaves AGT-2's measured benchmark pending.
