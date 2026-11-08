# Bibliography

Every source the course cites, grouped by the part that reads it. On 2026-11-06 each entry's arXiv ID, title
and first author was checked on its arXiv page, and each other entry on its publisher or proceedings page.
The year is that of first publication (arXiv v1 where there is one). Decks cite sections, figures and tables
of these sources; numbers that come from the course's own labs say so on the slide.

## Textbooks used throughout

| Decks | Book | Authors, year | Source |
|---|---|---|---|
| all | Deep Learning | Goodfellow, Bengio and Courville, 2016 | [deeplearningbook.org](https://www.deeplearningbook.org/) (free online) |
| 01–04 | The Elements of Statistical Learning, 2nd edition | Hastie, Tibshirani and Friedman, 2009 | [hastie.su.domains/ElemStatLearn](https://hastie.su.domains/ElemStatLearn/) (free PDF) |

## Part I. Learning from data

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 03 | A Stochastic Approximation Method | Robbins, 1951 | [doi:10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586) |
| 03 | Optimization Methods for Large-Scale Machine Learning | Bottou, 2016 | [arXiv:1606.04838](https://arxiv.org/abs/1606.04838) |
| 04 | Understanding deep learning requires rethinking generalization | Zhang, 2016 | [arXiv:1611.03530](https://arxiv.org/abs/1611.03530) |
| 04 | Reconciling modern machine learning practice and the bias-variance trade-off | Belkin, 2018 | [arXiv:1812.11118](https://arxiv.org/abs/1812.11118) |
| 04 | Deep Double Descent: Where Bigger Models and More Data Hurt | Nakkiran, 2019 | [arXiv:1912.02292](https://arxiv.org/abs/1912.02292) |

## Part II. Neural networks

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 05 | The perceptron: a probabilistic model for information storage and organization in the brain | Rosenblatt, 1958 | [doi:10.1037/h0042519](https://doi.org/10.1037/h0042519) |
| 05 | Approximation by superpositions of a sigmoidal function | Cybenko, 1989 | [doi:10.1007/BF02551274](https://doi.org/10.1007/BF02551274) |
| 05 | Deep Sparse Rectifier Neural Networks | Glorot, 2011 | [AISTATS 2011, PMLR 15](https://proceedings.mlr.press/v15/glorot11a.html) |
| 05 | Attention Is All You Need (the feed-forward sub-layer; see also Part III) | Vaswani, 2017 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| 06 | Learning representations by back-propagating errors | Rumelhart, 1986 | [doi:10.1038/323533a0](https://doi.org/10.1038/323533a0) |
| 06 | Automatic differentiation in machine learning: a survey | Baydin, 2015 | [arXiv:1502.05767](https://arxiv.org/abs/1502.05767) |
| 07 | Understanding the difficulty of training deep feedforward neural networks | Glorot, 2010 | [AISTATS 2010, PMLR 9](https://proceedings.mlr.press/v9/glorot10a.html) |
| 07 | Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification | He, 2015 | [arXiv:1502.01852](https://arxiv.org/abs/1502.01852) |
| 07 | Learning long-term dependencies with gradient descent is difficult | Bengio, 1994 | [doi:10.1109/72.279181](https://doi.org/10.1109/72.279181) |
| 07 | Improving Language Understanding by Generative Pre-Training (GPT-1) | Radford, 2018 | [OpenAI PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) |
| 07 | Language Models are Unsupervised Multitask Learners (GPT-2) | Radford, 2019 | [OpenAI PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |

## Part III. Optimization and regularization

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 08 | On the importance of initialization and momentum in deep learning | Sutskever, 2013 | [ICML 2013, PMLR 28](https://proceedings.mlr.press/v28/sutskever13.html) |
| 08 | Adaptive Subgradient Methods for Online Learning and Stochastic Optimization (AdaGrad) | Duchi, 2011 | [JMLR 12](https://jmlr.org/papers/v12/duchi11a.html) |
| 08 | Neural Networks for Machine Learning, lecture 6e (RMSProp) | Hinton, 2012 | [lecture slides](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) |
| 08 | Adam: A Method for Stochastic Optimization | Kingma, 2014 | [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) |
| 08, 10 | Decoupled Weight Decay Regularization (AdamW) | Loshchilov, 2017 | [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) |
| 08–10 | Llama 2: Open Foundation and Fine-Tuned Chat Models (optimizer and schedule settings) | Touvron, 2023 | [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) |
| 09 | SGDR: Stochastic Gradient Descent with Warm Restarts (cosine annealing) | Loshchilov, 2016 | [arXiv:1608.03983](https://arxiv.org/abs/1608.03983) |
| 09 | Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour | Goyal, 2017 | [arXiv:1706.02677](https://arxiv.org/abs/1706.02677) |
| 09 | On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima | Keskar, 2016 | [arXiv:1609.04836](https://arxiv.org/abs/1609.04836) |
| 09 | An Empirical Model of Large-Batch Training | McCandlish, 2018 | [arXiv:1812.06162](https://arxiv.org/abs/1812.06162) |
| 09 | Language Models are Few-Shot Learners (GPT-3; batch-size ramp) | Brown, 2020 | [arXiv:2005.14165](https://arxiv.org/abs/2005.14165) |
| 09 | MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies (warm-up, stable, decay) | Hu, 2024 | [arXiv:2404.06395](https://arxiv.org/abs/2404.06395) |
| 09, 10, 14 | Attention Is All You Need (schedule, dropout, label smoothing, residual connections) | Vaswani, 2017 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| 10 | Improving neural networks by preventing co-adaptation of feature detectors | Hinton, 2012 | [arXiv:1207.0580](https://arxiv.org/abs/1207.0580) |
| 10 | Dropout: A Simple Way to Prevent Neural Networks from Overfitting | Srivastava, 2014 | [JMLR 15](https://jmlr.org/papers/v15/srivastava14a.html) |
| 11 | Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift | Ioffe, 2015 | [arXiv:1502.03167](https://arxiv.org/abs/1502.03167) |
| 11 | How Does Batch Normalization Help Optimization? | Santurkar, 2018 | [arXiv:1805.11604](https://arxiv.org/abs/1805.11604) |
| 11 | Layer Normalization | Ba, 2016 | [arXiv:1607.06450](https://arxiv.org/abs/1607.06450) |

## Part IV. Convolutional networks

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 12 | Backpropagation Applied to Handwritten Zip Code Recognition | LeCun, 1989 | [doi:10.1162/neco.1989.1.4.541](https://doi.org/10.1162/neco.1989.1.4.541) |
| 12 | Gradient-based learning applied to document recognition (LeNet-5, MNIST) | LeCun, 1998 | [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791) |
| 10, 13 | ImageNet Classification with Deep Convolutional Neural Networks (AlexNet) | Krizhevsky, 2012 | [NeurIPS 2012](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) |
| 13 | ImageNet Large Scale Visual Recognition Challenge | Russakovsky, 2014 | [arXiv:1409.0575](https://arxiv.org/abs/1409.0575) |
| 13 | Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG) | Simonyan, 2014 | [arXiv:1409.1556](https://arxiv.org/abs/1409.1556) |
| 09, 13, 14 | Deep Residual Learning for Image Recognition (ResNet) | He, 2015 | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| 14 | Identity Mappings in Deep Residual Networks | He, 2016 | [arXiv:1603.05027](https://arxiv.org/abs/1603.05027) |
| 14 | Highway Networks | Srivastava, 2015 | [arXiv:1505.00387](https://arxiv.org/abs/1505.00387) |
| 14 | Language Models are Unsupervised Multitask Learners (GPT-2; residual-branch initialization) | Radford, 2019 | [OpenAI PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |

## Part V. Sequence models

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 15 | Finding Structure in Time | Elman, 1990 | [doi:10.1207/s15516709cog1402_1](https://doi.org/10.1207/s15516709cog1402_1) |
| 15 | Backpropagation through time: what it does and how to do it | Werbos, 1990 | [doi:10.1109/5.58337](https://doi.org/10.1109/5.58337) |
| 15, 16 | Learning long-term dependencies with gradient descent is difficult | Bengio, 1994 | [doi:10.1109/72.279181](https://doi.org/10.1109/72.279181) |
| 15 | On the difficulty of training recurrent neural networks (gradient clipping) | Pascanu, 2012 | [arXiv:1211.5063](https://arxiv.org/abs/1211.5063) |
| 15 | Generating Sequences With Recurrent Neural Networks | Graves, 2013 | [arXiv:1308.0850](https://arxiv.org/abs/1308.0850) |
| 16 | Long Short-Term Memory | Hochreiter, 1997 | [doi:10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735) |
| 16 | Learning to Forget: Continual Prediction with LSTM | Gers, 2000 | [doi:10.1162/089976600300015015](https://doi.org/10.1162/089976600300015015) |
| 16, 17 | Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (GRU) | Cho, 2014 | [arXiv:1406.1078](https://arxiv.org/abs/1406.1078) |
| 16 | Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling | Chung, 2014 | [arXiv:1412.3555](https://arxiv.org/abs/1412.3555) |
| 16, 19 | LSTM: A Search Space Odyssey | Greff, 2015 | [arXiv:1503.04069](https://arxiv.org/abs/1503.04069) |
| 16 | An Empirical Exploration of Recurrent Network Architectures | Jozefowicz, 2015 | [ICML 2015](https://proceedings.mlr.press/v37/jozefowicz15.html) |
| 17 | Sequence to Sequence Learning with Neural Networks | Sutskever, 2014 | [arXiv:1409.3215](https://arxiv.org/abs/1409.3215) |
| 17 | On the Properties of Neural Machine Translation: Encoder-Decoder Approaches | Cho, 2014 | [arXiv:1409.1259](https://arxiv.org/abs/1409.1259) |
| 17 | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5) | Raffel, 2019 | [arXiv:1910.10683](https://arxiv.org/abs/1910.10683) |
| 18 | Neural Machine Translation by Jointly Learning to Align and Translate | Bahdanau, 2014 | [arXiv:1409.0473](https://arxiv.org/abs/1409.0473) |
| 18 | Effective Approaches to Attention-based Neural Machine Translation | Luong, 2015 | [arXiv:1508.04025](https://arxiv.org/abs/1508.04025) |
| 18 | Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation | Wu, 2016 | [arXiv:1609.08144](https://arxiv.org/abs/1609.08144) |
| 18 | In-context Learning and Induction Heads | Olsson, 2022 | [arXiv:2209.11895](https://arxiv.org/abs/2209.11895) |

## Part VI. Practice and the bridge

| Deck | Source | Author, year | Link |
|---|---|---|---|
| 19 | A Recipe for Training Neural Networks (blog post) | Karpathy, 2019 | [karpathy.github.io](https://karpathy.github.io/2019/04/25/recipe/) |
| 08, 09, 19 | Deep Learning Tuning Playbook | Godbole, 2023 | [github.com/google-research/tuning_playbook](https://github.com/google-research/tuning_playbook) |
| 19 | Practical Recommendations for Gradient-Based Training of Deep Architectures | Bengio, 2012 | [arXiv:1206.5533](https://arxiv.org/abs/1206.5533) |
| 19 | Random Search for Hyper-Parameter Optimization | Bergstra, 2012 | [JMLR 13](https://jmlr.org/papers/v13/bergstra12a.html) |
| 18, 20 | Attention Is All You Need (self-attention, Table 1 on per-layer cost) | Vaswani, 2017 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| 20 | Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention | Katharopoulos, 2020 | [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) |

## Gradient-check diagnostic supplement — checked 2026-11-07

- Stanford CS231n, [Neural Networks Part 3: Learning](https://cs231n.github.io/neural-networks-3/), gradient checks: centered differences, step-size tradeoffs, kinks and scale-aware error. Used in decks 06/19 and L1. Course curves are executed toy float64 calculations, not copied upstream figures; the mixed tolerance is fixture-specific.

Fair-comparison supplement checked 2026-11-07: Google Tuning Playbook, experimental design / scientific, nuisance and fixed hyperparameters. Decks 08/09 and the L2 worksheet adapt these principles into an original study diagram and a constructed update-budget example; neither is a measured optimizer ranking.
