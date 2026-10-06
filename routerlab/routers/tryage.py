"""Tryage (Hari & Thomson, 2023) - perceptive router over a library of expert models.

The router (a small LM in the paper) predicts, for each expert model, the *downstream loss*
it would incur on the prompt (they used masked-LM loss; we use the causal NLL of the gold
answer, collected with extras.answer_nll). Selection then minimises
      predicted_loss_m  +  lambda * constraint_m
where constraints are user 'flags' such as model size, recency, security or verbosity. Sweeping
lambda traces the Pareto front between accuracy and the secondary objective (here: cost/size).
Tryage shines with *specialised* experts (code, math, bio, legal ...): add them to the pool
(aux.experts in models.yaml) and re-collect to reproduce the paper's library-of-experts setup.
Also reports the router's 'optimal model identification' accuracy (the paper's 50.9% metric).
"""
from __future__ import annotations

import numpy as np

from ..nn import MLP
from .base import Decision, Router, lambda_grid


class TryageRouter(Router):
    name = "tryage"
    paper = "Tryage (2308.11601)"
    needs = {"nll"}
    supports_live = True

    def __init__(self, constraint: dict | None = None, seed=0):
        self.constraint, self.seed = constraint, seed

    def fit(self, train, val=None):
        assert train.nll is not None, "run collection with extras.answer_nll=true"
        Y = np.nan_to_num(train.nll, nan=np.nanmax(train.nll))
        self.mu, self.sd = Y.mean(0), Y.std(0) + 1e-6
        self.net = MLP(train.X.shape[1], train.M, hidden=128, loss="mse", epochs=60, seed=self.seed).fit(
            train.X, (Y - self.mu) / self.sd)
        if self.constraint:     # e.g. {"qwen2.5-32b-awq": 1.0, "qwen2.5-1.5b": 0.05} size/recency flags
            self.flag = np.array([self.constraint.get(m, 0.0) for m in train.models])
        else:
            c = train.mean_cost()
            self.flag = c / c.max()
        return self

    def predict_loss(self, X):
        return self.net.predict(X) * self.sd + self.mu

    def knobs(self):
        return lambda_grid()

    def _pick(self, X, lam):
        L = self.predict_loss(X)
        L = (L - L.mean(1, keepdims=True)) / (L.std(1, keepdims=True) + 1e-6)
        return np.argmin(L + lam * self.flag[None], 1)

    def decide(self, data, knob):
        return Decision.single(self._pick(data.X, knob), data.M)

    def route_one(self, emb, knob):
        return int(self._pick(emb[None], knob)[0])

    def extra_report(self, train, test):
        return {"optimal_model_identification_acc": self.identification_accuracy(test),
                "chance": 1 / test.M}

    def identification_accuracy(self, data):
        true = np.argmin(np.nan_to_num(data.nll, nan=np.inf), 1)
        return float((self._pick(data.X, 0.0) == true).mean())
