"""Fly-Swat or Cannon? (Sakota, Peyrard, West - WSDM 2024) - meta-model + cost-aware assignment.

1. Meta-model: given only the input, predict the probability each LM answers correctly
   (paper: fine-tuned DistilBERT with one sigmoid head per LM; here an MLP head on frozen
   embeddings, or finetune.py for the full DistilBERT recipe).
2. Assignment: choose one LM per input to maximise expected accuracy subject to a total
   budget B - an Integer Linear Program:
        max  sum_{q,m} p_qm x_qm   s.t.  sum_m x_qm = 1,   sum_{q,m} c_m x_qm <= B,  x binary
   solved exactly with SciPy's HiGHS MILP solver. Also the paper's simpler 'cost-sensitive' rule is available
   (variant="greedy": argmax p - lambda c).
Paper: matches the largest LM at 63% lower cost.
"""
from __future__ import annotations

import numpy as np

from ..nn import MLP
from .base import Decision, Router, lambda_grid, utility_select


class FlySwatRouter(Router):
    paper = "Fly-Swat or Cannon (2308.06077)"
    supports_live = True

    def __init__(self, variant="ilp", seed=0):
        self.variant, self.seed = variant, seed
        self.name = f"flyswat-{variant}"

    def fit(self, train, val=None):
        self.meta = MLP(train.X.shape[1], train.M, hidden=128, loss="bce", epochs=50, seed=self.seed).fit(
            train.X, train.correct)
        self.cost = train.mean_cost()
        return self

    def knobs(self):
        return list(np.linspace(0, 1, 15)) if self.variant == "ilp" else lambda_grid()

    def _ilp(self, P, budget):
        """Exact ILP with SciPy's HiGHS MILP solver (no extra dependency)."""
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix

        N, M = P.shape
        rows = np.repeat(np.arange(N), M)
        one_per_query = csr_matrix((np.ones(N * M), (rows, np.arange(N * M))), shape=(N, N * M))
        budget_row = csr_matrix(np.tile(self.cost, N)[None, :])
        cons = [LinearConstraint(one_per_query, 1, 1), LinearConstraint(budget_row, -np.inf, budget)]
        res = milp(c=-P.ravel(), constraints=cons, integrality=np.ones(N * M), bounds=Bounds(0, 1),
                   options={"time_limit": 30, "mip_rel_gap": 1e-3})
        if res.x is None:      # infeasible budget -> cheapest model everywhere
            return np.full(N, int(np.argmin(self.cost)))
        return res.x.reshape(N, M).argmax(1)

    def decide(self, data, knob):
        P = self.meta.predict(data.X)
        if self.variant == "greedy":
            return Decision.single(utility_select(P, self.cost, knob), data.M)
        budget = data.N * (self.cost.min() + knob * (self.cost.max() - self.cost.min())) * (1 + 1e-9)
        return Decision.single(self._ilp(P, budget), data.M)

    def route_one(self, emb, knob):
        return int(utility_select(self.meta.predict(emb[None]), self.cost, knob)[0])
