# LLM Training and Post-Training

Read the papers behind how today's language models are trained, one method at a time: pre-training data and recipes, training at scale, long context, supervised fine-tuning, parameter-efficient tuning, RLHF, direct preference methods, RL for reasoning, distillation, safety tuning and evaluation.

**Status, 2026-11-06: outline published.** The course map deck, the syllabus (34 decks in eleven parts, ten CPU labs, weekly plan, assessment) and the bibliography (96 sources, each checked on its arXiv or blog page) are in place. Paper decks and labs follow part by part.

## Start here

- [Course map deck](slides/00_course_map.html): the whole course on 11 slides.
- [Syllabus](SYLLABUS.md): parts, decks, weekly plan, labs and assessment.
- [Bibliography](references/bibliography.md): every paper, grouped by the deck that reads it.

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

Engineers and students who can read a model card and want to know how each training stage behind it works, what it costs and when it fails. You need Python, NumPy, probability and the basics of neural-network training (a loss, backpropagation, SGD, softmax and cross-entropy); some labs also use PyTorch. All labs run on a laptop CPU with tiny models or toy distributions.

Classic architectures such as MLPs, CNNs, RNNs and LSTMs, and general machine learning, are not taught here; Stanford's CS229 and CS231n cover that ground. Model architecture is taught in [LLM Transformer Architecture](../llm-transformer-architecture-course/) and the systems side of training (kernels, collectives, sharding, checkpointing) in [Efficient LLM Computing Systems](../llm-compute-systems-course/). This course links to their decks where topics overlap instead of repeating them; the [syllabus](SYLLABUS.md#decks-this-course-links-to) lists each link.

## Slide decks

| Deck | Teaching use |
|---|---|
| [Course map](slides/00_course_map.html) (11 slides) | Week 1: the stages from web text to a chat model, the loss each stage optimizes, how each paper deck is built |

## Repository layout

```text
llm-training-course/
├── references/
│   └── bibliography.md              # Every paper, grouped by deck; sources checked 2026-11-06
├── slides/
│   └── 00_course_map.html           # Course map deck
├── LICENSE                          # Apache-2.0
├── README.md                        # This file
└── SYLLABUS.md                      # Parts, decks, weekly plan, labs and assessment
```

## License

Apache-2.0, see [LICENSE](LICENSE).
