# Lab L2: a pre-training data pipeline

Used with decks 05 to 08. CPU only, standard library only.

Web-scale pipelines (CCNet, RefinedWeb, Dolma, FineWeb, DCLM) run the same stages: drop pages that fail
simple rules, remove exact and near duplicates, then keep what a quality classifier ranks highest. This lab
runs those stages on a small synthetic crawl in which every document is labeled with what it is, so you can
see what each stage removes and what it lets through.

| File | What it holds | Deck |
|---|---|---|
| `corpus.py` | A synthetic crawl: prose, boilerplate menus, repeated lines, near and exact duplicates, keyword spam | 05 |
| `quality.py` | Three of FineWeb's line rules, and a naive Bayes classifier over words and word bigrams | 05, 06 |
| `dedup.py` | CCNet-style normalized exact hashing; MinHash over word 5-grams; LSH bands | 06 |
| `pipeline.py` | Runs the stages and prints the funnel by document type | 05, 06 |
| `mixture.py` | Sampling a mixture, epochs per domain for a token budget, the value of repeated tokens | 07 |

## Run

```bash
python -m unittest test_data_pipeline.py       # about 2 seconds
python pipeline.py
python pipeline.py --bands 20 --rows 5 --threshold 0.6
python pipeline.py --top 0.85
```

## Reference output

```text
stage                   docs       prose    near_dup   exact_dup boilerplate  repetitive        spam
crawl                    600         337          56          30          90          47          40
after rules              463         337          56          30           0           0          40
after exact dedup        433         337          56           0           0           0          40
after MinHash dedup      386         337           9           0           0           0          40
classifier, top 90%      347         337           9           0           0           0           1
```

- The rules remove all menus and repeated-line pages but none of the spam, whose lines are full sentences.
- Exact dedup removes byte-for-byte copies only. MinHash removes most near-duplicates; the 9 left are short
  documents where two edited words change enough 5-grams to fall below 75% similarity.
- The classifier, trained on a separate crawl, removes the spam. Keeping the top 85% instead removes 16 good
  documents as well: a classifier threshold always trades recall of good text against precision.

## Checks

- Normalized exact keys ignore case, digits and punctuation, as CCNet's do.
- The MinHash estimate is within 0.07 of the true Jaccard similarity with 512 hashes.
- FineWeb's 14 bands of 8 rows: a pair at similarity (1/14)^(1/8) ≈ 0.72 becomes a candidate with probability
  1 − (1 − 1/14)^14 ≈ 0.65; at 0.5 under 6%; at 0.9 over 99%.
- The pipeline removes every menu, repeated-line page and exact copy, at least 95% of the spam and 75% of the
  near-duplicates, and keeps at least 85% of the prose.
- Sampled mixture proportions match their weights within 1%; epochs per domain follow from the budget.
- Repeated tokens (Muennighoff et al.): 4 epochs are worth over 90% of 4 epochs of new data; the total never
  exceeds U(1 + R*), with R* = 15.39; at R = R* repeats keep 1 − 1/e of their value.

## Tasks

1. Plot `candidate_probability` for 9,000 hashes in 20 bands of 450 (Lee et al., RefinedWeb) and for 112
   hashes in 14 bands of 8 (FineWeb). Which similarity does each target, and what does each cost per document?
2. Change `near_duplicate` to make 1, 2, 4 and 8 edits and record how many near-duplicates survive. Why do
   short documents survive first?
3. FineWeb found that deduplicating each crawl snapshot on its own beat deduplicating across all of them.
   Split the corpus into "snapshots" and compare the documents each approach keeps. Which kind of document does
   global deduplication keep more of?
4. Replace the classifier's positives with only the first 20 prose documents of the reference crawl. How does
   the funnel change? What does that say about choosing a classifier's positive examples (DCLM used instruction
   data and ELI5 posts)?
5. You have 50B unique tokens of math and need 400B math tokens. Use `effective_data` to compare repeating it
   with adding 50B tokens of code, and relate the answer to Muennighoff et al.'s code-augmentation finding.
