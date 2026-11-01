"""Lab L8: ViT patch embedding and a CLIP-style contrastive loss. NumPy only.

- patchify() and PatchEmbed: the ViT input (deck 34): image -> P x P patches -> linear projection,
  a [class] token and learned 1D position embeddings.
- conv2d_strided(): the same projection written as a convolution with stride P, for the check.
- interpolate_pos(): 2D interpolation of position embeddings for a larger grid (ViT fine-tuning).
- clip_loss() and its gradient: the symmetric cross-entropy over an N x N similarity matrix (deck 35).
- train_toy_clip(): two linear encoders learn to pair "images" and "captions" that share a hidden concept.

    python vit_clip.py   # prints token counts and a toy CLIP training run
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- ViT patches (deck 34)

def patchify(img: np.ndarray, p: int) -> np.ndarray:
    """(H, W, C) image -> (N, p*p*C) flattened patches, row by row. H and W must be multiples of p."""
    H, W, C = img.shape
    assert H % p == 0 and W % p == 0, "image size must be a multiple of the patch size"
    x = img.reshape(H // p, p, W // p, p, C).transpose(0, 2, 1, 3, 4)
    return x.reshape((H // p) * (W // p), p * p * C)


def n_tokens(h: int, w: int, p: int, cls: bool = True) -> int:
    return (h // p) * (w // p) + (1 if cls else 0)


class PatchEmbed:
    """Linear patch projection E, a learnable [class] token and learned 1D position embeddings."""

    def __init__(self, img_size: int, p: int, c: int, d: int, rng: np.random.Generator):
        self.p, self.c, self.d = p, c, d
        self.grid = img_size // p
        self.E = rng.normal(0, (p * p * c) ** -0.5, (p * p * c, d))
        self.b = np.zeros(d)
        self.cls = rng.normal(0, 0.02, d)
        self.pos = rng.normal(0, 0.02, (self.grid * self.grid + 1, d))

    def __call__(self, img: np.ndarray) -> np.ndarray:
        tokens = patchify(img, self.p) @ self.E + self.b
        return np.vstack([self.cls, tokens]) + self.pos


def conv2d_strided(img: np.ndarray, kernel: np.ndarray, bias: np.ndarray, stride: int) -> np.ndarray:
    """Plain loops: kernel (kh, kw, C, D). Returns (H_out * W_out, D) in row-major order."""
    kh, kw, C, D = kernel.shape
    H, W, _ = img.shape
    out = []
    for i in range(0, H - kh + 1, stride):
        for j in range(0, W - kw + 1, stride):
            window = img[i:i + kh, j:j + kw, :]
            out.append(np.tensordot(window, kernel, axes=([0, 1, 2], [0, 1, 2])) + bias)
    return np.array(out)


def interpolate_pos(pos: np.ndarray, old_grid: int, new_grid: int) -> np.ndarray:
    """Bilinear 2D interpolation of patch position embeddings; the [class] row (row 0) is kept."""
    cls, grid = pos[:1], pos[1:].reshape(old_grid, old_grid, -1)
    coords = np.linspace(0, old_grid - 1, new_grid)
    i0 = np.floor(coords).astype(int)
    i1 = np.minimum(i0 + 1, old_grid - 1)
    t = coords - i0
    rows = grid[i0] * (1 - t)[:, None, None] + grid[i1] * t[:, None, None]
    cols = rows[:, i0] * (1 - t)[None, :, None] + rows[:, i1] * t[None, :, None]
    return np.vstack([cls, cols.reshape(new_grid * new_grid, -1)])


# ---------------------------------------------------------------- CLIP loss (deck 35)

def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def log_softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=axis, keepdims=True))


def clip_loss(img_emb: np.ndarray, txt_emb: np.ndarray, scale: float = 1 / 0.07):
    """Symmetric contrastive loss. Row i of each input is pair i. Returns (loss, logits).

    scale = 1 / tau; CLIP learns it, starting at 1 / 0.07 and clipping it at 100.
    """
    I, T = l2_normalize(img_emb), l2_normalize(txt_emb)
    logits = scale * I @ T.T
    n = len(logits)
    loss_i = -np.trace(log_softmax(logits, axis=1)) / n   # image -> its caption, over each row
    loss_t = -np.trace(log_softmax(logits, axis=0)) / n   # caption -> its image, over each column
    return (loss_i + loss_t) / 2, logits


def clip_loss_grad(img_emb: np.ndarray, txt_emb: np.ndarray, scale: float = 1 / 0.07):
    """Gradients of clip_loss with respect to the unnormalized embeddings."""
    ni = np.linalg.norm(img_emb, axis=1, keepdims=True)
    nt = np.linalg.norm(txt_emb, axis=1, keepdims=True)
    I, T = img_emb / ni, txt_emb / nt
    logits = scale * I @ T.T
    n = len(logits)
    eye = np.eye(n)
    p_rows = np.exp(log_softmax(logits, axis=1))
    p_cols = np.exp(log_softmax(logits, axis=0))
    dlogits = ((p_rows - eye) + (p_cols - eye)) / (2 * n)
    dI = scale * dlogits @ T
    dT = scale * dlogits.T @ I
    # back through x / |x|: dx = (dy - y (y . dy)) / |x|
    d_img = (dI - I * (I * dI).sum(1, keepdims=True)) / ni
    d_txt = (dT - T * (T * dT).sum(1, keepdims=True)) / nt
    return d_img, d_txt


# ---------------------------------------------------------------- toy CLIP training

def toy_pairs(n: int, rng: np.random.Generator, d_concept=8, d_img=48, d_txt=32, noise=0.3, mix=None):
    """Pairs that share a hidden concept z: image = z A + noise, caption = z B + noise."""
    if mix is None:
        mix = (rng.normal(size=(d_concept, d_img)), rng.normal(size=(d_concept, d_txt)))
    z = rng.normal(size=(n, d_concept))
    A, B = mix
    img = z @ A + noise * rng.normal(size=(n, d_img))
    txt = z @ B + noise * rng.normal(size=(n, d_txt))
    return img, txt, mix


def retrieval_accuracy(img_emb, txt_emb) -> float:
    """Fraction of images whose most similar caption is their own."""
    sims = l2_normalize(img_emb) @ l2_normalize(txt_emb).T
    return float((sims.argmax(axis=1) == np.arange(len(sims))).mean())


def train_toy_clip(steps=400, batch=64, d_embed=16, lr=0.5, scale=1 / 0.07, seed=0):
    """Train two linear encoders with plain SGD. Returns a dict with the loss curve and test accuracy."""
    rng = np.random.default_rng(seed)
    _, _, mix = toy_pairs(1, rng)
    d_img, d_txt = mix[0].shape[1], mix[1].shape[1]
    Wi = rng.normal(0, d_img ** -0.5, (d_img, d_embed))
    Wt = rng.normal(0, d_txt ** -0.5, (d_txt, d_embed))
    losses = []
    for _ in range(steps):
        img, txt, _ = toy_pairs(batch, rng, mix=mix)
        ei, et = img @ Wi, txt @ Wt
        loss, _ = clip_loss(ei, et, scale)
        gi, gt = clip_loss_grad(ei, et, scale)
        Wi -= lr * img.T @ gi
        Wt -= lr * txt.T @ gt
        losses.append(loss)
    img, txt, _ = toy_pairs(batch, rng, mix=mix)
    return {"losses": losses, "test_loss": clip_loss(img @ Wi, txt @ Wt, scale)[0],
            "test_accuracy": retrieval_accuracy(img @ Wi, txt @ Wt), "chance_loss": float(np.log(batch))}


def main():
    print("Tokens per image (with [class]):")
    for h, p in [(224, 16), (224, 14), (336, 14), (384, 16)]:
        print(f"  {h} x {h}, patch {p}: {n_tokens(h, h, p)}")
    r = train_toy_clip()
    l = r["losses"]
    print(f"\nToy CLIP, batch 64: chance loss ln 64 = {r['chance_loss']:.3f}")
    print(f"  training loss: step 1 {l[0]:.3f}, step 50 {l[49]:.3f}, step 400 {l[-1]:.3f}")
    print(f"  held-out batch: loss {r['test_loss']:.3f}, image->caption accuracy {r['test_accuracy']:.2%}")


if __name__ == "__main__":
    main()
