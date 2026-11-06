"""Two optimizers written out step by step (decks 01 and 04).

AdamW:  Kingma and Ba (arXiv:1412.6980) Algorithm 1 with the weight decay of Loshchilov and Hutter
        (arXiv:1711.05101) Algorithm 2 applied outside the adaptive division. Like torch.optim.AdamW,
        the decay is multiplied by the current learning rate.
Muon:   Jordan (2024): momentum, then an approximate orthogonalization of the update by five
        Newton-Schulz steps, for 2-D hidden weights only. Scaled by 0.2 * sqrt(max(rows, cols)) so its
        update RMS matches AdamW's, with decoupled weight decay (Moonlight, arXiv:2502.16982 section 2.2).
"""
from __future__ import annotations

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1
                t, m, v, g = state["t"], state["m"], state["v"], p.grad
                p.mul_(1 - group["lr"] * group["weight_decay"])       # decoupled weight decay
                m.mul_(b1).add_(g, alpha=1 - b1)                       # first moment
                v.mul_(b2).addcmul_(g, g, value=1 - b2)                # second moment
                m_hat = m / (1 - b1 ** t)                              # bias correction
                v_hat = v / (1 - b2 ** t)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(group["eps"]), value=-group["lr"])


def newton_schulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Push every singular value of g toward 1 while keeping its singular vectors."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g / (g.norm() + eps)
    tall = x.size(0) > x.size(1)
    if tall:
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        x = a * x + (b * aa + c * aa @ aa) @ x
    return x.T if tall else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.01, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.ndim == 2, "Muon is for 2-D hidden weights; use AdamW for the rest"
                buf = self.state[p].setdefault("m", torch.zeros_like(p))
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                update = newton_schulz5(g) * 0.2 * max(p.shape) ** 0.5
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
