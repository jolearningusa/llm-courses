# Lab L8: ViT patches and a CLIP-style contrastive loss

Used with decks 34 to 36. CPU only, NumPy only.

`vit_clip.py` builds the two pieces that most vision-language models share: the ViT input layer, which
turns an image into tokens, and CLIP's contrastive loss, which aligns image and text embeddings.

| Code | Idea | Deck |
|---|---|---|
| `patchify()`, `n_tokens()` | Image → P × P patches, flattened row by row | 34 |
| `PatchEmbed` | Linear projection E, a [class] token and learned 1D position embeddings | 34 |
| `conv2d_strided()` | The same projection written as a convolution with kernel and stride P | 34 |
| `interpolate_pos()` | 2D interpolation of position embeddings for fine-tuning at a higher resolution | 34 |
| `clip_loss()`, `clip_loss_grad()` | Symmetric cross-entropy over the N × N cosine-similarity matrix, and its gradient | 35 |
| `train_toy_clip()` | Two linear encoders learn to pair toy "images" and "captions" that share a hidden concept | 35 |

## Run

```bash
python -m unittest test_vit_clip.py
python vit_clip.py
```

`python vit_clip.py` prints token counts for common image and patch sizes (197 for 224 px with 16 px
patches, 577 for 336 px with 14 px patches) and trains the toy model with batch 64 and scale 1 / 0.07.
Our run:

| | Loss |
|---|---|
| Chance (all embeddings identical): ln 64 | 4.159 |
| Step 1 (random encoders) | 8.254 |
| Step 50 | 0.110 |
| Step 400 | 0.076 |
| Held-out batch | 0.067, with 100% image → caption accuracy |

The first step is above ln 64 because random encoders are confidently wrong: with a scale of about 14,
a wrong pairing with higher cosine similarity is punished hard.

The tests check the deck numbers (197 tokens for ViT at 224 px, LLaVA's 256 patches); that patchify keeps
every pixel once; that the patch embedding equals a strided convolution; that interpolated position
embeddings keep the [class] row and the grid corners; that the loss is ln N when every embedding is the same
and near 0 for perfect pairing at CLIP's maximum scale of 100; that it is symmetric; that the gradient matches
finite differences; and that toy training learns the pairing.

## Tasks

1. Change the patch size in `PatchEmbed` from 4 to 2 on a 16 × 16 image. How do the token count and the
   attention cost change? Which part of the parameter count changes, and which does not (compare DiT, deck 39)?
2. Replace the learned 1D position embeddings with none at all. Without positions, which images can a ViT
   not tell apart? Build two such images.
3. Train the toy model with batch 8, 64 and 512 at the same number of steps. How do the final loss and the
   held-out accuracy (on a batch of the same size) compare? Why does contrastive learning like large batches
   (deck 35)?
4. Train with scale 1, 1 / 0.07 and 100. Which trains fastest, and which ends with the best retrieval?
5. Drop the column term from `clip_loss` (image → text only). Is retrieval in the other direction (caption →
   image) still learned on this toy data? Explain.
6. LLaVA (deck 36) maps CLIP patch features into an LLM with one matrix W. Using `PatchEmbed` as a stand-in
   encoder, count how many LLM input tokens a 336 × 336 image costs at patch size 14, and compare with
   Gemma 3's 256 (deck 32).
