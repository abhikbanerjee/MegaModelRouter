"""Harnessing the Power of Multiple Minds (Srivatsa et al., 2024) - is per-query routing
feasible on reasoning tasks?

The paper trains routers that predict which LLM will solve a query, using
  * classification  - multi-label 'will model m be correct' heads on a query encoder
  * clustering      - k-means over query embeddings, route each cluster to its best model
and compares them with the best single model and the oracle. Their finding: the oracle gap is
huge, but learned routers only sometimes beat the best single model -> routing is promising,
not automatic. `fit` stores those diagnostics (`self.report`) so you see the same analysis.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from ..nn import MLP
from .base import Decision, Router, lambda_grid, utility_select


class MultipleMindsRouter(Router):
    paper = "Multiple Minds (2405.00467)"
    supports_live = True

    def __init__(self, variant="classifier", n_clusters=20, seed=0):
        self.variant, self.k, self.seed = variant, n_clusters, seed
        self.name = f"minds-{variant}"

    def fit(self, train, val=None):
        self.cost = train.mean_cost()
        if self.variant == "classifier":
            self.clf = MLP(train.X.shape[1], train.M, hidden=128, loss="bce", epochs=50, seed=self.seed).fit(
                train.X, train.correct)
        else:
            self.km = KMeans(self.k, n_init=5, random_state=self.seed).fit(train.X)
            lab = self.km.labels_
            prior = train.correct.mean(0)
            self.cacc = np.stack([(train.correct[lab == j].sum(0) + 3 * prior) / ((lab == j).sum() + 3)
                                  for j in range(self.k)])
        best_single = train.correct.mean(0).max()
        oracle = train.correct.max(1).mean()
        self.report = {"best_single_acc": float(best_single), "oracle_acc": float(oracle),
                       "oracle_gap_pts": float(100 * (oracle - best_single)),
                       "share_where_best_single_is_wrong_but_some_model_right":
                           float(((train.correct[:, train.correct.mean(0).argmax()] == 0) &
                                  (train.correct.max(1) == 1)).mean())}
        return self

    def extra_report(self, train, test):
        return self.report

    def predict(self, X):
        if self.variant == "classifier":
            return self.clf.predict(X)
        return self.cacc[self.km.predict(X)]

    def knobs(self):
        return lambda_grid()

    def decide(self, data, knob):
        return Decision.single(utility_select(self.predict(data.X), self.cost, knob), data.M)

    def route_one(self, emb, knob):
        return int(utility_select(self.predict(emb[None]), self.cost, knob)[0])
