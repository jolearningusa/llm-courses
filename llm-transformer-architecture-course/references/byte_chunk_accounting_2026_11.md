# Byte/chunk mechanisms and accounting — 2026-11-10

ARC-2's bounded reading/CPU-fixture portion, based on main `189c8a5` after #135. [Deck 07](../slides/07_subword_tokenization.html) slides 8/9 and their [Chinese twin](../slides/zh/07_subword_tokenization.html) introduce the comparison. [L2](../labs/L2_bpe/#optional-byte-and-dynamic-chunk-accounting) provides execution instructions and the [raw report](../labs/L2_bpe/chunk_accounting_report.json). These are original teaching illustrations, not copied paper artwork.

## Primary reading: separate mechanisms

- **BLT**, [arXiv:2412.09871 (v1), §§2–4](https://arxiv.org/html/2412.09871v1): entropy patching uses next-byte distributions conditioned on the observed prefix. Its local byte encoder/decoder surround a global patch model. BPE segmentation need not be stable under suffix extension, so lossless BPE encoding alone does not establish incremental patching. Read Fig. 3/5 and §4.5–4.6 for accounting and normalization, not as current speed guarantees.
- **H-Net**, [arXiv:2507.07955 (v1), §§2.2/2.4](https://arxiv.org/html/2507.07955v1): learned routing selects encoder representations; dechunking smooths and expands main-network outputs while retaining a fine-grained residual path. Current/past hidden states determine routing for next-step prediction. Equations 5/8 motivate the isolated scalar fixture; it omits learned projections, gradients and full model execution.

Neither mechanism is a tokenizer swap that preserves an existing token model's behavior. A common raw-byte budget and complete system boundary are needed for comparisons. No paper score, speed ratio or trained-checkpoint result is reproduced here.

## Pinned author-code reading; not installed or executed

| File / revision | Inspected path and limits | SHA-256 |
|---|---|---|
| [BLT patcher](https://github.com/facebookresearch/blt/blob/9774ed4fcc78313f9f218295f3d7e4decdadf2ae/bytelatent/data/patcher.py), `9774ed4fcc78313f9f218295f3d7e4decdadf2ae` | `entropy()` uses natural logarithms. `find_entropy_patch_start_ids()` distinguishes fixed thresholds from a whole-sequence top-k branch when threshold is absent; only the threshold path is described there as incremental. Special start IDs, shifts and `include_next_token` also matter. Our bit threshold is not a copied author hyperparameter. | `8fa2eab345f981250364580e3a2a01e8c4809e9e66755294367b9c50e7b75ba1` |
| [H-Net dynamic chunking](https://github.com/goombalab/hnet/blob/3673fe1217ebeb0d1438c7c71d58d32bdd190ec2/hnet/modules/dc.py), `3673fe1217ebeb0d1438c7c71d58d32bdd190ec2` | `RoutingModule.forward/step` compares projected previous/current states and retains routing state; the code uses a strict `>0.5` step decision. `ChunkLayer` selects marked vectors. `DeChunkLayer` clamps probabilities and uses a scan/gather in forward, recursive state in step. The scalar fixture is not numerical equivalence to these kernels. | `b0a30a75245a0f6911a92a56e6be80a3725fc27aa859bd0e9905c239fe87bb38` |

Sources were downloaded as text and inspected. No author dependencies, trained weights or kernels were installed/imported, and the complete repositories were not audited. Paper formulas and implementation edge conventions must not be silently interchanged.

## Actual CPU exercise and observed counts

Python 3.12.14, standard library only, assertions enabled. Existing `bpe.py` is unchanged. Train its deterministic 64-merge tokenizer on the four existing corpus files in filename order; evaluate the three fixed strings in `chunk_accounting.py`, without normalization. Corpus/source digests, merges, exact input bytes, piece bytes, flags and entropy values are retained. This is segmentation accounting, not a language-quality evaluation or a tokenizer contamination study.

| Input fixture | Raw UTF-8 bytes | Byte pieces | BPE64 pieces | Fixed-8 patches | Prefix-count toy patches |
|---|---:|---:|---:|---:|---:|
| repeat | 23 | 23 | 17 | 3 | 6 |
| multilingual | 18 | 18 | 16 | 3 | 6 |
| code | 23 | 23 | 21 | 3 | 7 |

The toy distribution is document-local byte counts with Laplace-one smoothing over 256 values. Before byte `i`, compute entropy in bits from bytes strictly before `i`; start at index zero, at entropy above 7.99, or when the previous patch has eight bytes. Reset per document. This is not BLT's neural entropy predictor or H-Net's learned router. All methods receive identical bytes; concatenating stored pieces reconstructs those bytes exactly. This does **not** show that compressed neural representations are lossless.

15 tests passed: six existing BPE tests and nine new fixture tests. Every prefix is checked against the full boundary sequence and an altered suffix. Deliberately future-sensitive routing and whole-patch mean broadcasting fail the corresponding invariance controls. A separate literal smoothing example gives `[10,10,10,20,20]` for boundary positions `[0,3]`, values `[10,30]` and probabilities `[1,0.5]`; changing the later value leaves the earlier outputs unchanged. It is not a trained dechunking model.

## Count positions without claiming total runtime

The JSON **declares** an architecture-position ledger; it does not execute neural stages. For hierarchical byte paths, retain `n` local encoder, `n` local decoder, `n` byte-output and, for the toy entropy rule, `n` router positions, alongside `m` global positions. The actual estimator performs `n` 256-bin entropy evaluations. Byte/BPE baselines declare a global/head position per byte/token, respectively. Widths, layer counts and operator types differ; these columns cannot be added as equal-cost operations.

Tokenization/training, packing and padding, transfers, cross-attention/dechunking, output heads, model training and entropy-model training/inference all belong in a real comparison. FLOPs, memory, latency and quality are explicitly unmeasured, not zero. Fewer positions do not establish a speedup or a fair model comparison.

For capstone 43, propose a fixed checkpoint/data protocol before measuring quality: identical held-out raw bytes/documents, declared context in bytes, allowed normalization, output likelihood conventions and selection budget. Use total negative log-likelihood in bits divided by scored bytes where a valid likelihood contract exists; do not compare bits/token across vocabularies or invent BPB for this counting fixture. Then obtain actual complete-work timing/memory and retain failures. Learned routing, author forward/step equivalence, trained quality and device performance remain open ARC-2 extensions.
