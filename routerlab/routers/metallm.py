"""MetaLLM (Nguyen et al., 2024) - routing as a contextual multi-armed bandit.

Arms = LLMs. Context = query embedding. Reward = 1[correct] - p * cost_m  (p trades
accuracy for money). A linear reward model per arm is learned; at inference the arm with
the highest predicted reward is chosen. Paper targets classification / multiple-choice,
where correctness is cheap to observe - so this router is evaluated on cls+mcq only.

mode="offline": full-information ridge regression per arm (what the paper trains).
mode="online" : LinUCB over the train stream with *bandit feedback only* (you only learn the
                reward of the arm you pulled) - the realistic production setting.
"""
from __future__ import annotations

import numpy as np

from .base import Decision, Router


class MetaLLMRouter(Router):
    paper = "MetaLLM (2407.10834)"
    kinds = {"cls", "mcq"}
    supports_live = True

    def __init__(self, mode="offline", ridge=1.0, ucb_alpha=0.5, n_knobs=14, seed=0):
        self.mode, self.ridge, self.alpha, self.seed = mode, ridge, ucb_alpha, seed
        self.ps = [0.0] + list(np.geomspace(0.01, 5, n_knobs - 1))
        self.name = f"metallm-{mode}"

    def _feat(self, X):
        return np.hstack([X, np.ones((len(X), 1))])

    def fit(self, train, val=None):
        self.c = train.mean_cost() / train.mean_cost().max()
        F, M = self._feat(train.X), train.M
        d = F.shape[1]
        self.theta = {}
        for p in self.ps:
            R = train.correct - p * self.c[None, :]
            if self.mode == "offline":
                A = F.T @ F + self.ridge * np.eye(d)
                self.theta[p] = np.linalg.solve(A, F.T @ R)          # [d, M]
            else:
                rng = np.random.default_rng(self.seed)
                Ainv = np.stack([np.eye(d) / self.ridge] * M)
                bvec = np.zeros((M, d))
                for i in rng.permutation(len(F)):
                    x = F[i]
                    th = np.einsum("mij,mj->mi", Ainv, bvec)
                    ucb = th @ x + self.alpha * np.sqrt(np.einsum("i,mij,j->m", x, Ainv, x))
                    a = int(np.argmax(ucb))
                    Ax = Ainv[a] @ x                                   # Sherman-Morrison
                    Ainv[a] -= np.outer(Ax, Ax) / (1 + x @ Ax)
                    bvec[a] += R[i, a] * x
                self.theta[p] = np.einsum("mij,mj->mi", Ainv, bvec).T
        return self

    def knobs(self):
        return self.ps

    def decide(self, data, knob):
        return Decision.single(np.argmax(self._feat(data.X) @ self.theta[knob], 1), data.M)

    def route_one(self, emb, knob):
        p = min(self.ps, key=lambda x: abs(x - knob))
        return int(np.argmax(self._feat(emb[None]) @ self.theta[p]))
