# Lab L9: pass@k, ratings from pairwise votes, contamination

Used with deck 32 and the capstone (deck 33). CPU only, Python 3.10+ and NumPy. Every run takes a few
seconds.

An evaluation number is only as good as its estimator. This lab writes three estimators that decide
headline numbers: pass@k for code and reasoning, Bradley-Terry ratings for an arena of pairwise votes, and
two checks for a benchmark leaking into training data.

| File | What it holds | Deck |
|---|---|---|
| `passk.py` | The unbiased pass@k estimator in its stable product form, the binomial-coefficient form, brute-force enumeration, the biased plug-in estimate, and a simulation of both errors | 32 |
| `ratings.py` | Simulated votes, Bradley-Terry by Newton's method, online Elo on the same votes, bootstrap intervals | 32 |
| `contamination.py` | A toy corpus and benchmark, a bigram language model, n-gram overlap, and Oren et al.'s permutation test on example order | 32 |

## Run

```bash
python -m unittest test_eval.py   # about a second
python passk.py
python ratings.py
python contamination.py
```

## Checks

- **pass@k** equals brute-force enumeration over every size-k subset of n results, for n = 5 and 8, every
  c and every k. Its expectation over Binomial(n, p) outcomes equals 1 − (1 − p)<sup>k</sup> exactly, while
  the plug-in 1 − (1 − c/n)<sup>k</sup> is lower for every k > 1 (Chen et al.,
  [arXiv:2107.03374](https://arxiv.org/abs/2107.03374), App. A).
- **Bradley-Terry** recovers five generating strengths to within 0.06 from 20,000 votes; the gradient is
  zero at the fit; reordering the votes leaves the fit unchanged, while online Elo changes (Chiang et al.,
  [arXiv:2403.04132](https://arxiv.org/abs/2403.04132)).
- **Contamination:** a clean model shares no 8-gram with the benchmark and passes the permutation test; one
  copy in canonical order is detected by both; copies in a shuffled order pass the permutation test though
  every 8-gram matches (Oren et al., [arXiv:2310.17623](https://arxiv.org/abs/2310.17623)).

## Reference results

On our machine (NumPy 2.5). `python passk.py`:

```text
n = 20 samples per problem, 20,000 simulated problems per row
    p    k    true  unbiased err  naive err
 0.05    1   0.050        0.0005     0.0005
 0.05    5   0.226        0.0009    -0.0187
 0.05   10   0.401       -0.0006    -0.0652
 0.20    1   0.200       -0.0009    -0.0009
 0.20    5   0.672       -0.0017    -0.0418
 0.20   10   0.893       -0.0015    -0.0654
 0.50    1   0.500       -0.0004    -0.0004
 0.50    5   0.969        0.0003    -0.0164
 0.50   10   0.999       -0.0001    -0.0042
```

`python ratings.py`:

```text
7 models, 5,000 votes; ratings on the Elo scale (1000 = average)
model    true      BT          95% CI     Elo  Elo, shuffled  BT, shuffled
    0    1208    1211    1191-1236       1191           1208          1211
    1    1139    1143    1125-1161       1175           1139          1143
    2    1087    1099    1083-1119       1106           1099          1099
    3    1000     981     962-998         963            987           981
    4     931     924     904-941         926            944           924
    5     844     852     834-869         848            836           852
    6     792     791     772-807         791            787           791
```

`python contamination.py`:

```text
benchmark: 200 examples of 12 tokens; corpus: 200,000 random tokens plus copies of the benchmark
     copies in the corpus   8-gram  permutation p
                        0     0.00          0.810
                        1     1.00          0.005
                        2     1.00          0.005
                        4     1.00          0.005
   4, in a shuffled order     1.00          0.875
4, 30% of tokens reworded     0.06          0.005
```

The smallest p-value 199 shuffles can give is 0.005. Each check misses something the other catches: the
permutation test needs the canonical order, and exact n-gram matching misses reworded copies. The
reworded copies are still caught here only because most example boundaries survived the rewording.
A toy bigram model memorizes one copy perfectly; Oren et al.'s 1.4B model could not be reliably caught at
one copy, and their threshold was about four.

## Tasks

1. Plot the naive estimator's bias against k for n = 10, 20 and 200. Why did Chen et al. sample n = 200
   to report k up to 100?
2. Make the votes intransitive (A beats B, B beats C, C beats A, each 70% of the time). What does
   Bradley-Terry report, and what does that hide?
3. Add a position bias: whoever is shown first wins an extra 10% of close votes. Fit Bradley-Terry with
   and without randomizing the order shown, and with a first-position term in the model.
4. Implement Oren et al.'s sharded test: split the benchmark into shards, compare canonical and shuffled
   log-likelihood in each, and combine with a t-test. How many shards give the most power at one copy?
5. Replace exact 8-gram matching with matching after lowercasing-style normalization (map each token to
   token // 10). How much of the reworded contamination does it recover, and at what false-positive rate?
