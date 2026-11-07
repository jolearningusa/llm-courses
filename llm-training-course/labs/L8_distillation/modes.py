"""Forward against reverse KL on a two-mode teacher, and on-policy distillation (decks 29 and 30).

The teacher p is a mixture of two Gaussians on a grid of points. The student q is a single Gaussian on the
same grid (normalized over the grid), so it cannot represent both modes and must choose:

    forward KL(p || q) = E_p[log p - log q]   what supervised KD minimizes: mode covering
    reverse KL(q || p) = E_q[log q - log p]   what MiniLLM and on-policy GKD minimize: mode seeking
    JSD(beta)                                  GKD's interpolation between the two

On-policy distillation estimates the reverse-KL gradient from the student's own samples:
grad = E_q[(log q(x) - log p(x)) grad log q(x)], the policy-gradient form MiniLLM uses.
"""
import numpy as np

GRID = np.linspace(-8, 8, 801)


def normalize(logits):
    l = logits - logits.max()
    p = np.exp(l)
    return p / p.sum()


def teacher(m1=-3.0, m2=3.0, s=0.8, w=0.5):
    dens = w * np.exp(-0.5 * ((GRID - m1) / s) ** 2) + (1 - w) * np.exp(-0.5 * ((GRID - m2) / s) ** 2)
    return dens / dens.sum()


def student(params):
    mu, log_s = params
    return normalize(-0.5 * ((GRID - mu) / np.exp(log_s)) ** 2)


def forward_kl(p, q):
    m = p > 0
    return float((p[m] * (np.log(p[m]) - np.log(q[m]))).sum())


def reverse_kl(p, q):
    return forward_kl(q, p)


def jsd(p, q, beta):
    """Generalized JSD of GKD: beta KL(p || m) + (1 - beta) KL(q || m), m = beta p + (1 - beta) q.
    Divided by beta(1 - beta), small beta behaves like forward KL and beta near 1 like reverse KL."""
    m = beta * p + (1 - beta) * q
    return beta * forward_kl(p, m) + (1 - beta) * forward_kl(q, m)


def score(params, x):
    """d log q(x) / d (mu, log_s) for the grid-normalized Gaussian student, at grid indices x."""
    mu, log_s = params
    s = np.exp(log_s)
    q = student(params)
    g_mu = (GRID - mu) / s ** 2
    g_ls = ((GRID - mu) / s) ** 2
    return np.stack([g_mu[x] - (q * g_mu).sum(), g_ls[x] - (q * g_ls).sum()], -1)


def reverse_kl_grad(params, p):
    """Exact gradient of KL(q || p): sum over the grid of q (log q - log p) d log q."""
    q = student(params)
    idx = np.arange(len(GRID))
    w = q * (np.log(q + 1e-300) - np.log(p + 1e-300))
    return (w[:, None] * score(params, idx)).sum(0)


def forward_kl_grad(params, p):
    """Exact gradient of KL(p || q) = -sum_x p(x) d log q(x)."""
    idx = np.arange(len(GRID))
    return -(p[:, None] * score(params, idx)).sum(0)


def on_policy_grad(params, p, n, rng):
    """Monte Carlo estimate of the reverse-KL gradient from n student samples, with a mean baseline."""
    q = student(params)
    x = rng.choice(len(GRID), size=n, p=q)
    r = np.log(q[x]) - np.log(p[x] + 1e-300)
    return ((r - r.mean())[:, None] * score(params, x)).sum(0) / (n - 1)


def fit(grad_fn, params=(1.5, -0.5), steps=3000, lr=0.05):
    """Adam on (mu, log sigma). The default start is a narrow student slightly right of centre."""
    th = np.array(params, dtype=float)
    m = np.zeros(2)
    v = np.zeros(2)
    for t in range(1, steps + 1):
        g = grad_fn(th)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        th -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return th


if __name__ == "__main__":
    p = teacher(w=0.6)
    rng = np.random.default_rng(0)
    rows = [
        ("forward KL (supervised KD)", fit(lambda th: forward_kl_grad(th, p))),
        ("reverse KL, exact", fit(lambda th: reverse_kl_grad(th, p))),
        ("reverse KL, on-policy samples", fit(lambda th: on_policy_grad(th, p, 64, rng), steps=1500)),
        ("reverse KL, start at mu=-1.5", fit(lambda th: reverse_kl_grad(th, p), params=(-1.5, -0.5))),
        ("reverse KL, start wide", fit(lambda th: reverse_kl_grad(th, p), params=(0.5, 0.5), lr=0.02)),
    ]
    print("teacher: modes at -3 and +3 (weights 0.6 and 0.4), std 0.8 each; mixture mean -0.6, std 3.05")
    print("student: one Gaussian, starting at mu=1.5, sigma=0.61 unless stated")
    print(f"{'objective':>30} {'mu':>7} {'sigma':>7} {'KL(p||q)':>9} {'KL(q||p)':>9}")
    for name, th in rows:
        q = student(th)
        print(f"{name:>30} {th[0]:7.3f} {np.exp(th[1]):7.3f} {forward_kl(p, q):9.3f} {reverse_kl(p, q):9.3f}")
    print("\nReverse KL picks the mode nearest its start, even the lighter one; from a wide start it can stall")
    print("in a broad local optimum that straddles both modes. Forward KL reaches the same answer from any start.")
