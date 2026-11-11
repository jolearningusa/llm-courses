# Research audit — Transformer architecture, 2026-11-07

Reviewed the 44-deck inventory, syllabus/bibliography and L5/L6 lab descriptions. Current historical/model-family decks form a coherent paper cohort, not a claim to list the newest model in every family. Add mechanisms and controlled evaluation before expanding model catalogues. [Repository-wide audit](../../docs/research-audit-2026-11-07.md).

## Primary literature and code checked

| Source / reading scope | Finding, limitation and decision |
|---|---|
| Lahoti et al., [Mamba-3: Improved Sequence Modeling using State Space Principles, 2603.15569v1](https://arxiv.org/html/2603.15569v1), §§3–4, §6 and kernel appendix | Exponential-trapezoidal recurrence, complex state and MIMO connect expressivity to decode arithmetic intensity. Extend deck 23's Mamba/Mamba-2 story. The paper's experiments do not prove every constant-state model is faster or better than every Transformer. |
| Pagnoni et al., [Byte Latent Transformer: Patches Scale Better Than Tokens, 2412.09871v1](https://arxiv.org/html/2412.09871v1), patching/architecture and limitations; [author repository](https://github.com/facebookresearch/blt) | Entropy-based byte patches allocate global-model work dynamically. This is a missing comparison for deck 07's fixed-token pipeline, not a drop-in tokenizer replacement for existing checkpoints. Include local encoder/decoder and entropy-model costs. |
| Hwang, Wang and Gu, [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, 2507.07955v1](https://arxiv.org/html/2507.07955v1), §2 and autoregressive inference; [H-Net code](https://github.com/goombalab/hnet) | Learned chunking differs from BLT's entropy patching. Preserve causal routing and dechunking in any toy example; do not infer production latency from fewer global positions. The guessed `state-spaces/hnet` URL failed; use the author-linked goombalab repository. |
| Hsieh et al., [RULER: What's the Real Context Size of Your Long-Context Language Models?, 2404.06654v1](https://arxiv.org/html/2404.06654v1), task families and error analysis | Retrieval, multi-hop tracking and aggregation probe different context behaviors. A supported window or one needle test is not effective context quality. Historical reported model scores are not current rankings. |
| [FlashAttention-4 author implementation README](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/README.md) | The CuTeDSL implementation gives a modern backend reading. Deck 14 should distinguish exact attention mathematics from which target-specific kernel realizes it; the compute course owns tuning/benchmarking. No install or device test performed. |

## Proposed changes

| ID / priority | Target | Execution and acceptance |
|---|---|---|
| ARC-1 / P1 mechanism | Deck 23; L6 block variants | Add an optional Mamba-3 comparison after Mamba-2: state update, inputs/outputs and MIMO reuse diagram. Derive a tiny recurrence and compare step versus reference evaluation with explicit tolerances. Kernel results require a separate target run; cite the actual scale/setting before discussing paper gains. |
| ARC-2 / P2 mechanism | Deck 07; L2 BPE; capstone 43 | Prepare one byte/BPE/entropy-patch/learned-chunk reading supplement. Keep raw byte sequence and data budget fixed; count local work and global positions separately. Acceptance: causal boundaries, round-trip byte reconstruction and complete workload accounting. No bits-per-token comparison across different vocabularies; use an appropriate common byte-normalized metric. |
| ARC-3 / P1 evaluation | Decks 08–10/15; L3 positions / L5 attention | Add a length × target-position × task-type evaluation plan based on RULER. Separate position encoding correctness, valid cache/mask behavior and learned task quality. Existing random-weight equivalence tests prove implementation relationships, not long-context comprehension. |
| ARC-4 / retain and cross-link | Deck 14 and model-family decks 28–33 | Keep the original paper results dated to their source. Link modern execution details to compute 12/35; add a new model only when it changes an identified mechanism or enables a meaningful comparison. |

## Potential projects

**Recurrent-state tradeoff study (research replication, novelty unestablished).** Week 1 freeze recurrence fixtures and matched baselines; week 2 validate streamed versus reference states, including long streams; week 3 measure state size and task retention; week 4, if hardware is available, compare prefill/decode latency at matched batch and quality. Ask whether kernel efficiency persists under changing batch, precision and state size. Report any unsupported target explicitly.

**Dynamic-chunk accounting workbench (engineering first).** Week 1 implement byte/BPE count baselines; week 2 add deterministic patch boundaries and causal tests; week 3 inspect author H-Net routing/dechunking; week 4 report byte-normalized quality and total work. A CPU boundary simulator is not a trained BLT/H-Net reproduction. Learned-quality claims need trained checkpoints and a budgeted evaluation.

All projects/edits above are proposed. A prior-art review and a fixed experimental protocol must precede any novelty claim; no new architecture or hardware benchmark was executed in this audit.

## ARC-3 teaching/fixture update prepared for review — 2026-11-07

Deck 10 slide 9 and deck 15 slide 10, with Chinese twins, now separate a length/placement/task matrix from the distinct position-math, cache and learned-quality oracles. L3 adds 81 small record-based fixtures and seven construction/scoring tests; L5 links to the tokenized model-evaluation protocol. Records are not model tokens, the debug solver is not a model, and no trained-checkpoint quality or target performance is measured. The original math/attention implementations are unchanged. Screenshots accompany review; explicit approval is required before merge.


## ARC-1 mechanism/algebra update prepared for review — 2026-11-08

Deck 23 slides 9/10 and Chinese twins add optional two-endpoint/complex recurrence and shared-state MIMO diagrams after the existing Mamba-2 evidence. L6 adds an original standard-library streamed/unrolled fixture, chunk carry with a previous-input negative control, complex/real rotation and literal MIMO shape checks. Eight new tests pass on Python 3.9.6; all fourteen L6 tests also pass in the existing torch 2.9.1 CPU environment. The author implementation reading is pinned to e9594ce, separating rank-independent core SSM state from rank-dependent auxiliary cache. Original model/training code and paper result pages are unchanged. This completes a bounded mechanism/algebra supplement, not learned Mamba-3 quality, optimized SSD equivalence or target-kernel validation.

## ARC-2 reading/accounting fixture — 2026-11-10

The [supplement](byte_chunk_accounting_2026_11.md) and EN/ZH deck 07 slides 8/9 distinguish byte/BPE, BLT entropy patches and H-Net learned chunks. L2 executes twelve fixed segmentation rows, raw-byte reconstruction, prefix-invariance controls and isolated scalar dechunking algebra; 15 tests pass (nine new). The model-position ledger is declared, not neural execution or complete FLOP/timing measurement. Author code is inspected at pinned revisions but not imported. This completes the bounded reading/CPU accounting fixture; learned router training, author-kernel equivalence, trained BPB/quality and total device-work measurement remain open.
