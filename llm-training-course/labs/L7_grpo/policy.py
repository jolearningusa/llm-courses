"""A tiny autoregressive policy over the 12-token vocabulary (PyTorch)."""
import torch
import torch.nn as nn

from task import VOCAB, MAX_LEN, EOS


class Policy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        V = len(VOCAB)
        self.a = nn.Embedding(10, hidden)
        self.b = nn.Embedding(10, hidden)
        self.pos = nn.Embedding(MAX_LEN, hidden)
        self.prev = nn.Embedding(V + 1, hidden)  # index V = start of response
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, V))

    def logits(self, a, b, prev, pos):
        return self.mlp(self.a(a) + self.b(b) + self.prev(prev) + self.pos(pos))

    def logprobs(self, a, b, tokens):
        """Log-probabilities of given responses. a, b: (N,); tokens: (N, T) padded with EOS.
        Returns per-token log-probs (N, T) and a mask (N, T) that is 1 up to and including the first EOS."""
        N, T = tokens.shape
        start = torch.full((N, 1), len(VOCAB), dtype=torch.long)
        prev = torch.cat([start, tokens[:, :-1]], 1)
        pos = torch.arange(T).expand(N, T)
        lg = self.logits(a[:, None].expand(N, T), b[:, None].expand(N, T), prev, pos)
        lp = torch.log_softmax(lg, -1).gather(-1, tokens[..., None]).squeeze(-1)
        is_eos = (tokens == EOS).long()
        before = torch.cumsum(is_eos, 1) - is_eos       # number of EOS strictly before t
        mask = (before == 0).float()
        return lp, mask

    @torch.no_grad()
    def sample(self, a, b, greedy=False, generator=None):
        N = a.shape[0]
        tokens = torch.full((N, MAX_LEN), EOS, dtype=torch.long)
        prev = torch.full((N,), len(VOCAB), dtype=torch.long)
        done = torch.zeros(N, dtype=torch.bool)
        for t in range(MAX_LEN):
            lg = self.logits(a, b, prev, torch.full((N,), t, dtype=torch.long))
            if greedy:
                nxt = lg.argmax(-1)
            else:
                nxt = torch.multinomial(torch.softmax(lg, -1), 1, generator=generator).squeeze(-1)
            nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
            tokens[:, t] = nxt
            done |= nxt == EOS
            prev = nxt
        return tokens
