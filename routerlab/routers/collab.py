"""Merge, Ensemble, and Cooperate (Lu et al., 2024 survey) - collaborative baselines.

The survey has no single algorithm; it organises the space into
  * Merge     - fuse weights of same-architecture models  -> scripts/merge_models.py builds a
                merged checkpoint (linear / task-arithmetic); add it to models.yaml and re-collect.
  * Ensemble  - run several models and combine outputs (before/during/after inference).
  * Cooperate - models with different roles (cascades, verification, speculative decoding).
The routing papers here are 'cooperate' or 'ensemble-before-inference' methods. This module
gives the after-inference ensemble reference line so you can see what routing buys you:

  knob k: weighted majority vote over the k models with the best validation accuracy-per-cost.
  Weights = log-odds of each model's validation accuracy.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .base import Decision, Router
from .utils import correct_of


class EnsembleVoteRouter(Router):
    name = "collab-vote"
    paper = "Merge/Ensemble/Cooperate survey (2407.06089)"

    def fit(self, train, val=None):
        v = val if val is not None and val.N else train
        acc = np.clip(v.correct.mean(0), 0.02, 0.98)
        self.w = np.log(acc / (1 - acc)) + 1e-3
        cost = train.mean_cost()
        # add models greedily: start with the most accurate, then best accuracy-per-cost
        order = [int(np.argmax(acc))]
        rest = [m for m in range(train.M) if m not in order]
        rest.sort(key=lambda m: -acc[m] / cost[m])
        self.order = order + rest
        return self

    def knobs(self):
        return list(range(1, len(self.order) + 1))

    def decide(self, data, knob):
        S = self.order[: int(knob)]
        preds = []
        for i in range(data.N):
            score = defaultdict(float)
            for m in S:
                score[data.pred[i, m]] += self.w[m]
            preds.append(max(score, key=score.get))
        calls = np.zeros((data.N, data.M), bool)
        calls[:, S] = True
        return Decision(final=np.full(data.N, -1), calls=calls, correct_override=correct_of(data, preds))
