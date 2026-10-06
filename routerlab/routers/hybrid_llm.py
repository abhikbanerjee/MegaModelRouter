"""Hybrid LLM (Ding et al., ICLR 2024) - quality-aware small/large routing.

Router predicts whether the SMALL model's response will be (almost) as good as the LARGE
model's, from the query alone, so the large model is called only when needed. Quality q(.)
was BARTScore in the paper; here it is the reward-model score (or correctness if no RM).
Three label constructions:
  det   : y = 1[q(S) >= q(L)]
  prob  : y = P[q(S) >= q(L)] - soft label; the paper averages over 10 samples per model,
          we approximate it with sigmoid((q(S)-q(L))/sigma)
  trans : y = 1[q(S) >= q(L) - t], with t chosen so labels are balanced on the train set.
          This relaxation is what lets the router learn when queries are 'easy' even if the
          small model rarely wins outright (the paper's best variant at high cost savings).
Route to the small model if predicted y >= threshold (knob).
"""
from __future__ import annotations

import numpy as np

from ..nn import MLP, sigmoid
from .base import Decision, Router
from .utils import quantile_knobs, strong_weak


class HybridLLMRouter(Router):
    paper = "Hybrid LLM (2404.14618)"
    needs = {"reward"}
    supports_live = True

    def __init__(self, variant="trans", small=None, large=None, seed=0):
        self.variant, self.small, self.large, self.seed = variant, small, large, seed
        self.name = f"hybrid-{variant}"

    def _quality(self, d):
        return d.reward if d.reward is not None else d.correct

    def fit(self, train, val=None):
        self.L, self.S = strong_weak(train, self.large, self.small)
        q = self._quality(train)
        gap = q[:, self.S] - q[:, self.L]
        if self.variant == "det":
            y = (gap >= 0).astype(float)
        elif self.variant == "prob":
            y = sigmoid(gap / (gap.std() + 1e-6) * 2)
        else:
            self.t = float(np.quantile(-gap, 0.5))   # balance labels
            y = (gap >= -self.t).astype(float)
        self.clf = MLP(train.X.shape[1], 1, hidden=128, loss="bce", epochs=50, seed=self.seed).fit(train.X, y)
        self.knob_list = quantile_knobs(self.clf.predict(train.X)[:, 0])
        return self

    def knobs(self):
        return self.knob_list

    def decide(self, data, knob):
        p = self.clf.predict(data.X)[:, 0]
        return Decision.single(np.where(p >= knob, self.S, self.L), data.M)

    def route_one(self, emb, knob):
        return self.S if self.clf.predict(emb[None])[0, 0] >= knob else self.L
