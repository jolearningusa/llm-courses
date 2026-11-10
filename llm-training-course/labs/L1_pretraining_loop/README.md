# Lab L1: a tiny pre-training loop

Used with decks 01 to 04. CPU only. PyTorch 2 (`pip install torch`) is a broad prerequisite, not an exact reproduction recipe; see the [observed environment and its limits](../../references/cpu_environment_evidence_2026_11.md).

The parts of a pre-training recipe that a model card summarizes in one line (the optimizer, its
settings, the learning-rate schedule, the stability tricks) are each a few lines of code. This lab writes
them out and trains a small decoder-only model with each.

| File | What it holds | Deck |
|---|---|---|
| `optim.py` | AdamW step by step (bias correction, decoupled weight decay); Muon with five Newton-Schulz steps and Moonlight's update scaling | 01, 04 |
| `schedules.py` | Constant, warm-up and cosine, and warm-up-stable-decay (WSD) with a linear or 1-sqrt cool-down | 01 |
| `model.py` | A 2-layer character model (RMSNorm, causal attention, GELU MLP) with optional QK-norm and z-loss | 03 |
| `train.py` | Trains runs on the same data, budget and seeds and reports final validation loss as mean ± stdev | 01–04 |

The data is generated lines of additions such as `417+86=503;`. The operands are random, so the loss has
a floor well above zero, and a model that learns to add reaches a lower loss than one that only learns the
format. Hidden 2-D weights get the run's optimizer with weight decay 0.1; embeddings, the output head and
norm gains get AdamW without weight decay, as OLMo 2 does for embeddings.

## Run

```bash
python -m unittest test_pretraining.py        # about 5 seconds
python train.py                               # 4 runs x 3 seeds x 600 steps, about a minute on 4 cores
python train.py --steps 1500                  # about 3 minutes
python train.py --lr 3e-2 --runs adamw-constant adamw-constant-qk-z
```

## Reference results

Our run on a 4-core CPU with PyTorch 2.15, peak learning rate 3e-3, 5% warm-up, 20% cool-down for WSD.
Your numbers will differ with the PyTorch version and the hardware.

| Run | 600 steps | 1,500 steps |
|---|---|---|
| adamw-constant | 1.734 ± 0.018 | 1.587 ± 0.114 |
| adamw-cosine (to 10% of peak) | 1.748 ± 0.010 | 1.655 ± 0.013 |
| adamw-wsd (1-sqrt cool-down) | 1.726 ± 0.015 | 1.575 ± 0.102 |
| muon-wsd | 1.716 ± 0.006 | 1.538 ± 0.139 |

| 600 steps, peak learning rate 3e-2 | Final validation loss |
|---|---|
| adamw-constant | 1.758 ± 0.009 |
| adamw-constant with QK-norm and z-loss 1e-4 | 1.717 ± 0.005 |

Read them like a paper's table, three seeds each:

- At 600 steps the four runs lie within 0.032 of each other, two to three standard deviations. Muon and WSD come out
  ahead, in the direction decks 01 and 04 report, but three seeds at this size do not establish it.
- At 1,500 steps the spread across seeds jumps to about 0.1 for every run except cosine. Some seeds start to
  learn the carries of the addition and others do not yet; plot `curves.csv` to see the step where each seed's
  loss drops. A single seed would have given any ranking you wanted.
- At ten times the learning rate, QK-norm and z-loss together lower the loss and its spread. At this size nothing
  diverges, which is why Wortsman et al. (deck 03) measure sensitivity across a learning-rate sweep, not one run.

## Checks

- The hand-written AdamW matches `torch.optim.AdamW` to float precision over 25 steps, and its first step
  moves every weight by exactly the learning rate (bias correction).
- Five Newton-Schulz steps bring every singular value of a matrix with condition number 100 into [0.5, 1.3]
  without changing its shape, and Muon's scaled update has an RMS of about 0.2, matching AdamW's.
- Schedules: warm-up reaches the peak; cosine ends at its floor; WSD is flat until the cool-down and ends at 0.
- The model is causal; z-loss leaves cross-entropy unchanged when all logits shift and penalizes the shift.
- AdamW and Muon both lower the loss below that of a uniform guess within 120 steps.

## Tasks

1. Plot validation loss against step for `adamw-cosine` and `adamw-wsd`. Where does WSD's loss fall
   fastest, and how does that match MiniCPM's description of the decay phase?
2. Change the WSD cool-down to 5% and 40% of the run and to `shape="linear"`. Which matters more here, the
   length or the shape? Compare with Hägele et al.'s Figure 5.
3. Save a checkpoint at step 1,200 of a constant-rate run and start two cool-downs from it, one 300 and one
   600 steps long. Why does this make scaling-law sweeps cheaper than one cosine run per budget?
4. Sweep the peak learning rate over 1e-3, 3e-3, 1e-2, 3e-2 and 1e-1 with and without `qk_norm` and z-loss,
   and compute Wortsman et al.'s learning-rate sensitivity for each.
5. Remove Moonlight's 0.2·sqrt(max(A, B)) factor from `Muon.step` and find Muon's best learning rate again.
   How far is it from AdamW's? Moonlight adds the factor so that AdamW's learning rate and weight decay
   can be reused unchanged.

## Environment and split evidence — 2026-11-09

The [TRN-4 manifest and reporting worksheet](../../references/cpu_environment_evidence_2026_11.md) records 10 existing L1 tests passing on CPython 3.12.14 / PyTorch 2.9.1, CPU, one thread, with NumPy absent (import warning retained). This is an observed existing environment, not a fresh-install lock or reproduction of the historical PyTorch 2.15 tables above. The training assertions use two fixed 120-step configurations and select no winner.

In this Python 3.12.14 run, the generated train/validation cohorts (seeds 1/2) share **43 unique exact examples**; different seeds do not make them strictly disjoint. Their content hashes and overlap definition are in the manifest. Treat this as a toy mechanism/validation check, not a clean held-out generalization study. Freeze a disjoint split and a selection rule before conducting such a study; do not silently alter the historical experiment.
