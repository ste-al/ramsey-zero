"""Learning components.  Both start from nothing and use search outcomes only.

UCB          The bandit that chooses the next local-search arm (weak symmetry, descend,
             ascend, full) from the rewards of earlier pulls.  Always on.

OrbitPolicy  The "apprentice" of an AlphaZero-style expert-iteration loop, optional
             (learn=True / --learn; off by default because it measured neutral):
               AlphaZero:  policy net proposes -> MCTS improves -> net imitates MCTS.
               Here:       policy proposes an orbit colouring -> exact tabu search
                           improves it -> policy imitates the improved colourings of
                           the best episodes (ranked reward, top quantile).
             Targets are first mapped into the policy's gauge using the space's
             symmetries (multipliers x->a*x, swaps of equal colours) so that isomorphic
             solutions reinforce each other instead of cancelling.  The policy starts
             uniform.

The third learned element lives in the C core: the complete search orders its variables by
conflict activity carried across restarts (see ox_run in _core.c).
"""
from __future__ import annotations

import itertools
import math

import numpy as np


class OrbitPolicy:
    def __init__(self, V, r, eps=0.02, lr=0.25, rng=None):
        self.V, self.r, self.eps, self.lr = V, r, eps, lr
        self.p = np.full((V, r), 1.0 / r)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.updates = 0

    def sample(self, B):
        u = self.rng.random((B, self.V, 1))
        cdf = np.cumsum(self.p, axis=1)[None, :, :]
        return np.minimum((u > cdf).sum(axis=2), self.r - 1).astype(np.int8)

    def loglik(self, x):
        return float(np.log(self.p[np.arange(self.V), x]).sum())

    def align(self, x, orbit_perms, color_perms):
        """Return the symmetric image of x that is most likely under the policy."""
        best, best_ll = x, -math.inf
        idx = np.arange(self.V)
        logp = np.log(self.p)
        for g in orbit_perms:
            y = np.empty_like(x)
            y[g] = x
            for cp in color_perms:
                z = cp[y]
                ll = logp[idx, z].sum()
                if ll > best_ll:
                    best, best_ll = z, ll
        return best

    def update(self, X):
        X = np.asarray(X)
        freq = np.stack([(X == c).mean(axis=0) for c in range(self.r)], axis=1)
        self.p = (1 - self.lr) * self.p + self.lr * freq
        self.p = np.maximum(self.p, self.eps)
        self.p /= self.p.sum(axis=1, keepdims=True)
        self.updates += 1

    def entropy(self):
        return float(-(self.p * np.log(self.p)).sum() / (self.V * math.log(self.r)))

    def reset(self):
        self.p[:] = 1.0 / self.r
        self.updates = 0


def color_permutations(ks):
    """Permutations of colours that preserve the problem (only among equal k)."""
    r = len(ks)
    groups = {}
    for c, k in enumerate(ks):
        groups.setdefault(k, []).append(c)
    perms = []
    for choice in itertools.product(*[list(itertools.permutations(g)) for g in groups.values()]):
        p = np.arange(r)
        for g, img in zip(groups.values(), choice):
            for a, b in zip(g, img):
                p[a] = b
        perms.append(p.astype(np.int8))
    return perms[:24]


class UCB:
    def __init__(self, n_arms, c=0.35, rng=None):
        self.n = np.zeros(n_arms)
        self.s = np.zeros(n_arms)
        self.c = c
        self.rng = rng if rng is not None else np.random.default_rng()

    def select(self, allowed=None):
        idx = np.arange(len(self.n)) if allowed is None else np.asarray(allowed)
        untried = [i for i in idx if self.n[i] == 0]
        if untried:
            return int(untried[0])
        total = self.n[idx].sum()
        ucb = self.s[idx] / self.n[idx] + self.c * np.sqrt(np.log(total + 1) / self.n[idx])
        ucb = ucb + 1e-9 * self.rng.random(len(idx))
        return int(idx[int(np.argmax(ucb))])

    def update(self, i, reward):
        self.n[i] += 1
        self.s[i] += reward

    def skip(self, i):
        """Count a turn the arm sat out: its mean reward is unchanged, its bonus shrinks."""
        if self.n[i] > 0:
            self.s[i] += self.s[i] / self.n[i]
        self.n[i] += 1

    def means(self):
        return np.where(self.n > 0, self.s / np.maximum(self.n, 1), np.nan)
