"""OptLLM (Liu et al., 2024) - optimal *batch* assignment of queries to LLMs.

1. Predictor: multi-label classifier (one output per LLM = will it succeed?) with uncertainty
   estimated by a bootstrap ensemble; the robust estimate is mean - kappa * std.
2. Optimiser: produces a whole set of non-dominated (cost, accuracy) assignments for the batch.
   Start from the cheapest solution and the max-accuracy solution, then iteratively *destruct*
   (unassign a random subset) and *reconstruct* (greedily reassign by marginal accuracy per cost
   within the budget), keeping the Pareto set. The user picks a point matching their budget.
Here the knob is a budget fraction (of the all-strongest cost); each knob returns the best
assignment found under that budget.
"""
from __future__ import annotations

import numpy as np

from ..nn import MLP
from .base import Decision, Router


class OptLLMRouter(Router):
    name = "optllm"
    paper = "OptLLM (2405.15130)"

    def __init__(self, n_boot=5, kappa=0.5, iters=60, destroy_frac=0.15, seed=0):
        self.n_boot, self.kappa, self.iters, self.destroy, self.seed = n_boot, kappa, iters, destroy_frac, seed

    def fit(self, train, val=None):
        rng = np.random.default_rng(self.seed)
        self.models = []
        for b in range(self.n_boot):
            idx = rng.integers(0, train.N, train.N)
            self.models.append(MLP(train.X.shape[1], train.M, hidden=96, loss="bce", epochs=40, seed=b)
                               .fit(train.X[idx], train.correct[idx]))
        self.cost = train.mean_cost()
        return self

    def predict(self, X):
        P = np.stack([m.predict(X) for m in self.models])
        return np.clip(P.mean(0) - self.kappa * P.std(0), 0, 1)

    def knobs(self):
        return list(np.linspace(0.0, 1.0, 21))

    def _assign(self, P, budget, rng):
        N, M = P.shape
        c = self.cost
        a = np.full(N, int(np.argmin(c)))      # start: everything on the cheapest model
        spent = c[a].sum()

        def reconstruct(a, spent, pool):
            # greedy upgrades by (delta acc / delta cost) for queries in `pool`
            gains = []
            for i in pool:
                for m in range(M):
                    dc, da = c[m] - c[a[i]], P[i, m] - P[i, a[i]]
                    if da > 0:
                        gains.append((da / max(dc, 1e-12) if dc > 0 else np.inf, i, m, dc, da))
            gains.sort(key=lambda t: -t[0])
            for _, i, m, dc, da in gains:
                cur_da = P[i, m] - P[i, a[i]]
                cur_dc = c[m] - c[a[i]]
                if cur_da > 0 and spent + cur_dc <= budget:
                    spent += cur_dc
                    a[i] = m
            return a, spent

        a, spent = reconstruct(a, spent, range(N))
        best, best_val = a.copy(), P[np.arange(N), a].sum()
        for _ in range(self.iters):  # destruct-and-reconstruct local search
            cand = best.copy()
            drop = rng.choice(N, max(1, int(self.destroy * N)), replace=False)
            cand[drop] = int(np.argmin(c))
            sp = c[cand].sum()
            cand, sp = reconstruct(cand, sp, rng.permutation(drop))
            v = P[np.arange(N), cand].sum()
            if v > best_val and sp <= budget:
                best, best_val = cand, v
        return best

    def decide(self, data, knob):
        rng = np.random.default_rng(self.seed)
        P = self.predict(data.X)
        cmin, cmax = self.cost.min() * data.N, self.cost.max() * data.N
        budget = cmin + knob * (cmax - cmin)
        return Decision.single(self._assign(P, budget, rng), data.M)
