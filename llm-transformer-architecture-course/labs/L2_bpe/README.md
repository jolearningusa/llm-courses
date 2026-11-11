# Lab L2: byte-level BPE

Used with deck 07. CPU only, standard library only.

`bpe.py` trains a byte-level BPE tokenizer: Sennrich et al.'s merge loop (deck 07) run on UTF-8 bytes, with
GPT-2's rule that merges never cross a letter/digit/punctuation boundary. `corpus/` holds the same short
passage about tokenizers written in English, German, Spanish and Chinese.

## Run

```bash
python bpe.py --merges 300          # trains on corpus/, prints characters per token for each language
python -m unittest test_bpe.py
```

## Tasks

1. Read `train()` and match each line to Algorithm 1 of Sennrich et al. What replaces the end-of-word symbol?
2. Print the first 30 merges. Which language's pieces appear first, and why?
3. The four files say the same thing. Count tokens for each at 100, 300 and 1,000 merges. Which language
   gains least from more merges, and what does that mean for its share of a 128K-token context window?
4. Remove `pre_split` (encode the whole text as one chunk). What kinds of tokens appear, and why does GPT-2
   forbid them?
5. Add a fifth language file of your own and retrain. How does adding it change the English token count?

## Checks

Encoding and decoding are lossless for every corpus file and for unseen text (emoji, rare scripts, tabs);
every ID is a byte or a learned merge, so there is no unknown token; merges at least halve the English byte
count; a frequent word becomes one token; and the same content costs more tokens in Chinese than in English.

## Optional byte and dynamic chunk accounting

Companion to deck 07 slides 8/9 and [the dated reading/execution note](../../references/byte_chunk_accounting_2026_11.md). Standard library only; tested with Python 3.12.14. This supplement leaves the original tokenizer and its six tests unchanged.

```sh
python -m unittest test_bpe.py test_chunk_accounting.py -v
python chunk_accounting.py --output chunk-new.json
```

Use a fresh output path; existing files are not overwritten. [Reference JSON](chunk_accounting_report.json) contains all twelve sample/method rows, source/corpus hashes, merge rules, raw input/piece bytes, boundary flags and the scalar smoothing fixture. Fifteen tests pass, including nine new tests.

1. Compare byte, BPE64, fixed-eight-byte and prefix-count entropy partitions on exactly the same raw bytes. Reassemble byte pieces **before** decoding UTF-8; one piece can split a character. This tests byte storage/segmentation, not losslessness of neural latents.
2. Change a suffix and verify all earlier toy entropy boundaries are unchanged. Predict how the deliberately future-sensitive router fails. Explain why `encode('a')=[97]` and `encode('ab')=[256]` under the one-merge test tokenizer is lossless but not incremental byte patching.
3. Inspect all position-ledger columns. For the 23-byte repeat fixture, six toy patches still have 23 declared local encoder, decoder and router positions. These neural stages are **not executed**. The actual entropy routine evaluates 256 probabilities at each byte. Tokenizer training/preprocessing and later model/packing/transfer/head costs must not disappear from a system budget.
4. Trace the scalar EMA/forward-fill example, then change its later selected value. Contrast with broadcasting a whole-patch mean to earlier positions, which leaks future inputs. This fixture omits learned routing, residuals, gradient estimators and author kernels.
5. Read the pinned author paths in the note. Write the missing experiment required before claiming trained quality, BPB or a GPU speedup. For capstone [43](../../slides/43_capstone.html), first freeze raw bytes, checkpoint, context, scoring and compute-budget conventions; the count table supplies no model likelihoods.

The entropy toy uses Laplace-one **prefix counts**, threshold 7.99 **bits** and a maximum of eight bytes per patch; it is not a neural BLT/H-Net reproduction. Declared architecture positions are distinct from executed CPU bookkeeping, FLOPs and latency. No trained model, quality, memory or device timing was measured.

Retained deck questions: run three merges by hand on `{low×5, lowest×2, newer×6, wider×3}`; explain byte coverage versus unknown characters; calculate embedding growth from 32K to 128K at width 4096 for tied/untied heads; explain checkpoint implications of changing token IDs.
