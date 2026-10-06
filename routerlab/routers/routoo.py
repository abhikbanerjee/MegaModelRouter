"""Routoo (Mohammadshahi et al., 2024) - performance predictor + cost-aware selector.

* Performance predictor: a lightweight model that scores every candidate LLM for a prompt
  WITHOUT running them. The paper uses a small decoder LM with learned per-model embeddings;
  here: shared MLP over [query embedding ; model embedding ; their elementwise product].
* Cost-aware selector: given a budget (avg $/query) and optional latency cap, pick per query
  argmax_m  p(m|q) - lambda * cost_m, with lambda found by bisection so the batch meets the
  budget exactly. That's what gives Routoo its controllable quality/cost dial.
Paper result on MMLU with open models: Mixtral-level accuracy at ~1/3 lower cost.
"""
from __future__ import annotations

import numpy as np

from ..nn import Adam, sigmoid
from .base import Decision, Router


class RoutooRouter(Router):
    name = "routoo"
    paper = "Routoo (2401.13979)"
    supports_live = True

    def __init__(self, k=32, hidden=128, epochs=40, max_latency_s=None, seed=0):
        self.k, self.h, self.epochs, self.max_lat, self.seed = k, hidden, epochs, max_latency_s, seed

    def fit(self, train, val=None):
        rng = np.random.default_rng(self.seed)
        d, M = train.X.shape[1], train.M
        f = d + 2 * self.k
        self.P = {"Wq": rng.normal(0, 1 / np.sqrt(d), (d, self.k)), "E": rng.normal(0, 0.3, (M, self.k)),
                  "W1": rng.normal(0, np.sqrt(2 / f), (f, self.h)), "b1": np.zeros(self.h),
                  "w2": rng.normal(0, 0.1, self.h), "b2": np.zeros(1)}
        opt = Adam(self.P, 2e-3, 1e-4)
        X, Y = train.X, train.correct
        for _ in range(self.epochs):
            for b in np.array_split(rng.permutation(len(X)), max(1, len(X) // 128)):
                out, cache = self._fwd(X[b])
                g = (sigmoid(out) - Y[b]) / Y[b].size           # [B,M]
                F, H, A, Zq = cache
                gA = g[..., None] * self.P["w2"] * (H > 0)        # [B,M,h]
                gF = gA @ self.P["W1"].T                          # [B,M,f]
                _, ge, gp = gF[..., :d], gF[..., d:d + self.k], gF[..., d + self.k:]
                gZ = (gp * self.P["E"][None]).sum(1)
                opt.step({"w2": np.einsum("bm,bmh->h", g, A), "b2": np.array([g.sum()]),
                          "W1": np.einsum("bmf,bmh->fh", F, gA), "b1": gA.sum((0, 1)),
                          "E": ge.sum(0) + (gp * Zq[:, None]).sum(0), "Wq": X[b].T @ gZ})
        self.cost = train.mean_cost()
        self.lat = train.latency.mean(0)
        return self

    def _fwd(self, X):
        B, M = len(X), len(self.P["E"])
        Zq = X @ self.P["Wq"]
        F = np.concatenate([np.repeat(X[:, None], M, 1), np.repeat(self.P["E"][None], B, 0),
                            Zq[:, None] * self.P["E"][None]], -1)
        H = F @ self.P["W1"] + self.P["b1"]
        A = np.maximum(H, 0)
        return A @ self.P["w2"] + self.P["b2"], (F, H, A, Zq)

    def predict(self, X):
        return sigmoid(self._fwd(X)[0])

    def select(self, P, budget):
        allowed = np.ones(len(self.cost), bool)
        if self.max_lat:
            allowed &= self.lat <= self.max_lat
        Pm = np.where(allowed[None], P, -np.inf)
        lo, hi = 0.0, 1e4 / self.cost.max()
        for _ in range(50):                      # bisection on the Lagrange multiplier
            lam = (lo + hi) / 2
            a = np.argmax(Pm - lam * self.cost[None], 1)
            if self.cost[a].mean() > budget:
                lo = lam
            else:
                hi = lam
        return np.argmax(Pm - hi * self.cost[None], 1)

    def knobs(self):
        return list(np.linspace(0, 1, 21))   # budget as fraction between cheapest and priciest

    def _budget(self, knob):
        return self.cost.min() + knob * (self.cost.max() - self.cost.min())

    def decide(self, data, knob):
        return Decision.single(self.select(self.predict(data.X), self._budget(knob)), data.M)

    def route_one(self, emb, knob):
        # for a single live query use the multiplier calibrated offline at this budget
        p = self.predict(emb[None])
        return int(self.select(p, self._budget(knob))[0])
