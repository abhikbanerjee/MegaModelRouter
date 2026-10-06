"""Zooter - Routing to the Expert (Lu et al., 2023): reward-guided routing by distillation.

1. Run every LLM on the training queries; score every response with a reward model (QwenRM in
   the paper; OpenAssistant DeBERTa RM by default here).
2. Turn per-query rewards into a target distribution over models: softmax(r / tau).
3. Tag-based label enhancement: queries are noisy, so mix each target with the average
   distribution of queries sharing a tag (paper: InsTag tags; here: task/subject, or k-means
   cluster ids if you pass tags=None)  y = (1-beta) y_q + beta y_tag.
4. Distil into a small router with KL / soft cross-entropy. At inference: one forward pass of a
   small encoder, then ONE LLM call - unlike reward-model ranking which needs all LLMs.
The original only maximises quality; knob lambda adds a cost penalty on the log-probabilities.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from ..nn import MLP, softmax
from .base import Decision, Router, lambda_grid


class ZooterRouter(Router):
    name = "zooter"
    paper = "Zooter (2311.08692)"
    needs = {"reward"}
    supports_live = True

    def __init__(self, tau=1.0, beta=0.3, tag_col="subject", seed=0):
        self.tau, self.beta, self.tag_col, self.seed = tau, beta, tag_col, seed

    def fit(self, train, val=None):
        r = train.reward if train.reward is not None else train.correct
        r = (r - r.mean()) / (r.std() + 1e-6)
        Y = softmax(r / self.tau)
        tags = (train.q[self.tag_col].values if self.tag_col else
                KMeans(32, n_init=3, random_state=self.seed).fit_predict(train.X))
        Yt = np.zeros_like(Y)
        for t in np.unique(tags):
            Yt[tags == t] = Y[tags == t].mean(0)
        Y = (1 - self.beta) * Y + self.beta * Yt
        self.net = MLP(train.X.shape[1], train.M, hidden=128, loss="softce", epochs=60, seed=self.seed).fit(train.X, Y)
        self.c = train.mean_cost() / train.mean_cost().max()
        return self

    def knobs(self):
        return lambda_grid()

    def _pick(self, X, lam):
        return np.argmax(np.log(self.net.predict(X) + 1e-9) - lam * self.c[None], 1)

    def decide(self, data, knob):
        return Decision.single(self._pick(data.X, knob), data.M)

    def route_one(self, emb, knob):
        return int(self._pick(emb[None], knob)[0])
