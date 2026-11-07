"""A toy world of responses, a hidden gold reward, and a Bradley-Terry reward model (deck 18).

Each of N candidate responses is a point phi(a) in two dimensions. The gold reward, which stands for
what people actually want, likes one direction but punishes extremes:

    r*(a) = w* . phi(a) - c * |phi(a)|^2

The reference policy (the SFT model) mostly produces moderate responses near the origin, so the
preference data comes from there. The reward model is linear in phi: it fits the data well where the
data is, and keeps rising in the liked direction where there is no data. Deck 20's overoptimization
comes from exactly that gap.
"""
import torch

W_GOLD = torch.tensor([1.0, 0.5])
CURVATURE = 0.5


def make_world(n_actions=400, spread=2.5, ref_width=0.7, seed=0):
    """Return phi (n, 2) and the reference policy's log-probabilities (n,)."""
    g = torch.Generator().manual_seed(seed)
    phi = (torch.rand(n_actions, 2, generator=g) * 2 - 1) * spread
    ref_logits = -(phi ** 2).sum(1) / (2 * ref_width ** 2)
    return phi, torch.log_softmax(ref_logits, 0)


def gold_reward(phi):
    return phi @ W_GOLD - CURVATURE * (phi ** 2).sum(-1)


def sample_preferences(phi, ref_logp, n_pairs, seed=0, label_noise=True):
    """Draw pairs of responses from the reference policy and label them by Bradley-Terry on the gold
    reward: P(a wins over b) = sigmoid(r*(a) - r*(b)). Returns (winner, loser) index tensors."""
    g = torch.Generator().manual_seed(seed)
    a = torch.multinomial(ref_logp.exp(), n_pairs, replacement=True, generator=g)
    b = torch.multinomial(ref_logp.exp(), n_pairs, replacement=True, generator=g)
    r = gold_reward(phi)
    p_a = torch.sigmoid(r[a] - r[b])
    a_wins = torch.rand(n_pairs, generator=g) < p_a if label_noise else p_a > 0.5
    return torch.where(a_wins, a, b), torch.where(a_wins, b, a)


def bt_loss(r_winner, r_loser):
    """Bradley-Terry negative log-likelihood: mean of -log sigmoid(r_w - r_l)."""
    return -torch.nn.functional.logsigmoid(r_winner - r_loser).mean()


class LinearRM(torch.nn.Module):
    """Reward model r(a) = w . phi(a). Bradley-Terry only fixes differences, so no bias is needed."""

    def __init__(self, dim=2):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, phi):
        return phi @ self.w


def train_rm(phi, winners, losers, steps=300, lr=0.1):
    rm = LinearRM(phi.shape[1])
    opt = torch.optim.Adam(rm.parameters(), lr=lr)
    for _ in range(steps):
        r = rm(phi)
        loss = bt_loss(r[winners], r[losers])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return rm


@torch.no_grad()
def pair_accuracy(rm, phi, a, b):
    """Share of pairs that the reward model orders the same way as the gold reward (ties excluded)."""
    gold = gold_reward(phi)
    keep = gold[a] != gold[b]
    r = rm(phi)
    return ((r[a] > r[b]) == (gold[a] > gold[b]))[keep].float().mean().item()
