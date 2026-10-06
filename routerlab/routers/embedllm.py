"""EmbedLLM (Zhuang et al., 2024) - learn compact *model embeddings*.

Correctness of model m on query q is modelled as
    P(correct | q, m) = sigmoid( v . (E_m * (W x_q)) + b_m )
i.e. a matrix factorisation between a learned embedding E_m per LLM and a linear
projection of the query embedding. Trained with BCE on the (query x model) correctness
matrix. The model embeddings are reusable: they cluster similar models, predict
benchmark accuracy, and a NEW model can be added by fitting only its E_m on a few
hundred labelled queries (`add_model`) - the main practical benefit claimed.
"""
from __future__ import annotations

import numpy as np

from ..nn import Adam, sigmoid
from .base import Decision, Router, lambda_grid, utility_select


class EmbedLLMRouter(Router):
    name = "embedllm"
    paper = "EmbedLLM (2410.02223)"
    supports_live = True

    def __init__(self, k=32, epochs=80, lr=3e-3, wd=1e-4, bs=512, seed=0):
        self.k, self.epochs, self.lr, self.wd, self.bs, self.seed = k, epochs, lr, wd, bs, seed

    def fit(self, train, val=None):
        rng = np.random.default_rng(self.seed)
        X, Y = train.X, train.correct
        d, M = X.shape[1], train.M
        self.P = {"W": rng.normal(0, 1 / np.sqrt(d), (d, self.k)), "E": rng.normal(0, 0.3, (M, self.k)),
                  "v": rng.normal(0, 0.3, self.k), "b": np.zeros(M)}
        opt = Adam(self.P, self.lr, self.wd)
        n = len(X)
        for _ in range(self.epochs):
            for s in range(0, n, self.bs):
                b = rng.permutation(n)[: self.bs]
                Z = X[b] @ self.P["W"]                   # [B,k]
                H = Z[:, None, :] * self.P["E"][None]   # [B,M,k]
                logit = H @ self.P["v"] + self.P["b"]   # [B,M]
                g = (sigmoid(logit) - Y[b]) / Y[b].size  # [B,M]
                gH = g[..., None] * self.P["v"]          # [B,M,k]
                grads = {"v": np.einsum("bm,bmk->k", g, H), "b": g.sum(0),
                         "E": np.einsum("bmk,bk->mk", gH, Z),
                         "W": X[b].T @ np.einsum("bmk,mk->bk", gH, self.P["E"])}
                opt.step(grads)
        self.mean_cost = train.mean_cost()
        return self

    def predict(self, X):
        Z = X @ self.P["W"]
        return sigmoid((Z[:, None, :] * self.P["E"][None]) @ self.P["v"] + self.P["b"])

    def add_model(self, X_probe, y_probe, steps=300, lr=1e-2):
        """Fit an embedding for a brand-new LLM from a small probe set, everything else frozen."""
        e, b = np.zeros(self.k), np.zeros(1)
        P = {"e": e, "b": b}
        opt = Adam(P, lr)
        Z = X_probe @ self.P["W"]
        for _ in range(steps):
            logit = (Z * P["e"]) @ self.P["v"] + P["b"]
            g = (sigmoid(logit) - y_probe) / len(y_probe)
            opt.step({"e": (g[:, None] * Z * self.P["v"]).sum(0), "b": np.array([g.sum()])})
        self.P["E"] = np.vstack([self.P["E"], P["e"]])
        self.P["b"] = np.append(self.P["b"], P["b"])
        return len(self.P["E"]) - 1

    def model_embeddings(self):
        return self.P["E"]

    def knobs(self):
        return lambda_grid()

    def decide(self, data, knob):
        return Decision.single(utility_select(self.predict(data.X), self.mean_cost, knob), data.M)

    def route_one(self, emb, knob):
        return int(utility_select(self.predict(emb[None]), self.mean_cost, knob)[0])
