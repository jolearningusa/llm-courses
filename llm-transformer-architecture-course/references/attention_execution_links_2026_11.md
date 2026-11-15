# Attention mathematics, device kernels and serving — ARC-4, 2026-11-15 UTC

This completes ARC-4's bounded cross-course link in architecture deck14, slide9 (English and Chinese), while keeping its historical paper results and original closing questions. Slides1–8 and model-family decks28–33 are unchanged. This is a reading/evidence map, not a package migration, hardware benchmark or new model catalogue.

## Follow three different contracts

| Level | Question / evidence | Course route | Current boundary |
| --- | --- | --- | --- |
| Formula, scale and mask | Does this blocked computation match the declared direct attention on the same inputs? | Architecture [L5](../labs/L5_attention_variants/) and FlashAttention lecture | Existing NumPy fixture, causal/noncausal small arrays; random weights do not establish learned quality |
| Device implementation | Which target, kernel class, dtype/layout, tile pipeline, mask and synchronization actually execute? | Compute [12: attention/kernel co-design](../../llm-compute-systems-course/slides/12_attention_kernel_codesign.html#slide-15), [pinned SM100 reading](../../llm-compute-systems-course/references/fa4_sm100_walkthrough_2026_11.md) | Source-reading/scalar controls already recorded; FA4 device compilation/numerics/safety/performance still unqualified |
| Serving and KV movement | Which engine/request owns the KV blocks, and which completion permits use or reuse? | Compute [35: inference engines](../../llm-compute-systems-course/slides/35_inference_engines_compared.html#slide-20), [vLLM source identity](../../llm-compute-systems-course/references/vllm_source_identity_2026_11.md) | Artifact/source identity and feature-specific source checks, not a qualified vLLM/NIXL target run |

The Chinese slide routes to the Chinese compute twins with the same anchors. These are related views of a workload, **not a claim of one mandatory serial pipeline or that every engine uses FA4/NIXL**. A correctly implemented transfer preserves the intended KV contents, but the transfer path is not the softmax algorithm. Incorrect transport, stale ownership, a changed mask or a different representation can still change outputs; each needs its own oracle.

## Primary-source recheck

- [FlashAttention v1](https://arxiv.org/html/2205.14135v1), §3.1 and Appendix B: exact dense attention is reorganized with tiling/recomputation and an online normalization state. This motivates L5's mathematical comparison. Original paper speed/quality results stay in their2022 evaluated cohorts; no new speed figure is transferred to modern kernels or serving.
- [FA4 author README](https://github.com/Dao-AILab/flash-attention/blob/c055bdc6444a1a1984082a64b7fd62d6cb42a93d/flash_attn/cute/README.md) and [interface](https://github.com/Dao-AILab/flash-attention/blob/c055bdc6444a1a1984082a64b7fd62d6cb42a93d/flash_attn/cute/interface.py), revision `c055bdc6444a1a1984082a64b7fd62d6cb42a93d`: the CuTeDSL package and dispatch/device checks are implementation contracts, not an architecture-independent qualification. The selected SM100 dependency reading remains separately scoped in compute12.
- [vLLM NIXL facade](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py) and [compatibility document](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/docs/features/nixl_connector_compatibility.md), revision `4ba787ec9959e566d3b587a796dff07673324da5`: the facade forwards work to scheduler/worker classes, distinguishes prefill/decode roles and exposes loading/completion-related hooks. Feature support is a separate matrix; no general “all KV layouts supported” claim follows. Facade existence alone does not prove correct completion/recovery or inspect every underlying worker path.

The web reader could not retrieve the two pinned raw pages; a read-only fetch from the official raw URLs succeeded. [Source record](attention_execution_sources_2026_11.json) saves URLs, sizes and SHA-256 values. FA4 interface and both vLLM files match the prior three digest anchors; README identity is newly recorded. No upstream code was imported, package installed, CUDA built, engine launched or target result inferred. No tracker/version/verified date advances.

## Existing CPU check and its limit

Reran exactly the existing `test_attention.TestMasksAndBlocks.test_blockwise_equals_direct` on CPython3.12.14 /NumPy2.3.5, macOS arm64: **one unittest passes**, covering causal blocks1/3/5/N and a noncausal block4 comparison for its fixed seeded input. The assertions use `atol=1e-12` and NumPy's default relative tolerance; this is not a newly specified absolute-only1e-12 guarantee. The [raw log](attention_execution_blockwise_2026_11.txt) is retained. Neither lab implementation nor tests changed, and no new tests were invented for these reversible links.

```sh
# From labs/L5_attention_variants/, in the matching recorded NumPy environment:
python -m unittest test_attention.TestMasksAndBlocks.test_blockwise_equals_direct -v
```

The pass does not execute FA4, a GPU synchronization path, remote KV movement or a trained model. For an actual kernel, freeze shape/dtype/scale/mask and use independent numerics, generated code/resources and relevant safety checks. For transport, add actual engine/request identities, completion/ownership/retry controls and supported layout/backend evidence. Both remain separate acceptance gates; source links do not close CMP-2/3 or broad AGT-2 evaluation.

## Teaching handoff

For each linked implementation, fill three rows: mathematical invariant and reference; selected kernel/configuration with eligible target evidence; engine/KV lifetime with completion evidence. Identify one counterexample to promoting a CPU equivalence pass into an end-to-end deployment claim. Keep original lecture questions on extra backward FLOPs versus IO, online max/sum/output updates, training versus decode and modeling versus speed results; they remain in slide notes and L5's existing tasks.

[Review captures](../../docs/reviews/architecture-attention-execution-links/README.md) show both translated before/after slides. This closes only the bounded ARC-4 cross-link/retention task. Author-kernel execution, vLLM target qualification, portability, model quality and hardware speedups remain outside the update. PR#120 remains AGT-2 teaching/protocol only; later measured pilots do not complete its broader benchmark.
