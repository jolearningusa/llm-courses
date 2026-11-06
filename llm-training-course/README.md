# LLM Training and Post-Training

Read the papers behind how today's language models are trained, one method at a time: pre-training data and recipes, training at scale, long context, supervised fine-tuning, parameter-efficient tuning, RLHF, direct preference methods, RL for reasoning, distillation, safety tuning and evaluation.

**Status, 2026-11-06: Part I published.** The course map deck, the four Part I decks on the pre-training recipe (AdamW and learning-rate schedules, batch size and μP, training stability, Muon), labs L0 and L1, the syllabus (34 decks in eleven parts, ten CPU labs, weekly plan, assessment) and the bibliography (96 sources, each checked on its arXiv or blog page) are in place. The other parts follow one at a time.

## Start here

- [Course map deck](slides/00_course_map.html): the whole course on 11 slides.
- [Syllabus](SYLLABUS.md): parts, decks, weekly plan, labs and assessment.
- [Bibliography](references/bibliography.md): every paper, grouped by the deck that reads it.
- [Labs](labs/): CPU-only exercises with their own checks (`python -m unittest` in each lab folder).

## What the course covers

The course follows one model from raw web text to a chat and reasoning model.

- **The pre-training recipe.** The next-token loss, AdamW, warm-up and WSD schedules, batch size and μP, loss spikes and how reports keep training stable, Muon.
- **Pre-training data.** Web corpora, deduplication and quality filtering, data mixtures and repetition, code, math and synthetic data, and mid-training.
- **Training at scale.** Data, tensor and pipeline parallelism as a model builder sees them, and mixed-precision and FP8 training.
- **Long context and continued pre-training.** Extending the context window with continued training, and adapting a model to a new domain without forgetting.
- **Supervised and parameter-efficient fine-tuning.** Instruction tuning, instruction data, an open post-training recipe end to end, LoRA and QLoRA.
- **Preferences.** Reward models, PPO and InstructGPT, RLHF at scale, AI feedback, DPO and its variants, online preference learning.
- **RL for reasoning.** Verifiers and process rewards, GRPO, RL with verifiable rewards (DeepSeek-R1), and the fixes that followed.
- **Distillation, safety and evaluation.** Off- and on-policy distillation, safety tuning and its attacks, benchmarks, judges and contamination.

Every paper deck follows the same five steps: the problem, the method, the evidence in the paper, the training impact, and what later reports kept or dropped.

## Audience and prerequisites

Engineers and students who can read a model card and want to know how each training stage behind it works, what it costs and when it fails. You need Python, NumPy, probability and the basics of neural-network training (a loss, backpropagation, SGD, softmax and cross-entropy); labs L1, L3, L5 and L7 also use PyTorch. All labs run on a laptop CPU with tiny models or toy distributions.

Classic architectures such as MLPs, CNNs, RNNs and LSTMs, and general machine learning, are not taught here; Stanford's CS229 and CS231n cover that ground. Model architecture is taught in [LLM Transformer Architecture](../llm-transformer-architecture-course/) and the systems side of training (kernels, collectives, sharding, checkpointing) in [Efficient LLM Computing Systems](../llm-compute-systems-course/). This course links to their decks where topics overlap instead of repeating them; the [syllabus](SYLLABUS.md#decks-this-course-links-to) lists each link.

## Slide decks

| Deck | Teaching use |
|---|---|
| [Course map](slides/00_course_map.html) (11 slides) | Week 1: the stages from web text to a chat model, the loss each stage optimizes, how each paper deck is built |
| **Part I. The pre-training recipe** | |
| [01 The training loop: AdamW and learning-rate schedules](slides/01_training_loop.html) (10 slides) | Week 1: Adam, decoupled weight decay, warm-up and cosine, WSD and cool-downs; labs L0, L1 |
| [02 Batch size and hyperparameter transfer](slides/02_batch_size_mup.html) (11 slides) | Week 1: the gradient noise scale and critical batch, batch ramps, μP and μTransfer; lab L0 |
| [03 Training stability](slides/03_training_stability.html) (10 slides) | Week 2: PaLM's loss spikes and z-loss, small-scale proxies and QK-norm, the OLMo 2 recipe; lab L1 |
| [04 Muon](slides/04_muon.html) (10 slides) | Week 2: orthogonalized updates by Newton-Schulz, Moonlight's scaling, QK-Clip in Kimi K2; labs L0, L1 |

## Repository layout

```text
llm-training-course/
├── labs/
│   ├── L0_training_budget/
│   │   ├── README.md
│   │   ├── budget.py                # FLOPs, Chinchilla tokens, days at an MFU, ZeRO memory, batch trade-off
│   │   └── test_budget.py
│   ├── L1_pretraining_loop/
│   │   ├── README.md
│   │   ├── model.py                 # Tiny decoder with optional QK-norm and z-loss
│   │   ├── optim.py                 # AdamW and Muon written out step by step
│   │   ├── schedules.py             # Constant, warm-up + cosine, WSD
│   │   ├── test_pretraining.py
│   │   └── train.py                 # Same data, budget and seeds across optimizers and schedules
│   └── README.md                    # Lab index
├── references/
│   └── bibliography.md              # Every paper, grouped by deck; sources checked 2026-11-06
├── slides/
│   ├── 00_course_map.html           # Course map deck
│   ├── 01_training_loop.html
│   ├── 02_batch_size_mup.html
│   ├── 03_training_stability.html
│   └── 04_muon.html
├── LICENSE                          # Apache-2.0
├── README.md                        # This file
└── SYLLABUS.md                      # Parts, decks, weekly plan, labs and assessment
```

## License

Apache-2.0, see [LICENSE](LICENSE).
