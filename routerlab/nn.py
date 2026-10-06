"""Tiny numpy neural nets so every router trains on CPU in seconds (no torch needed).

Router heads in these papers are small (a linear layer or 1-hidden-layer MLP on top of a
frozen encoder), so this is faithful. For the papers that fine-tune a full encoder
(RouteLLM-BERT, Hybrid-LLM DeBERTa, FrugalGPT DistilBERT, Zooter mDeBERTa, Fly-Swat
DistilBERT) see routerlab/finetune.py which does that on GPU with transformers.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class Adam:
    def __init__(self, params: dict, lr=1e-3, wd=0.0, b1=0.9, b2=0.999):
        self.p, self.lr, self.wd, self.b1, self.b2, self.t = params, lr, wd, b1, b2, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: dict):
        self.t += 1
        for k, g in grads.items():
            g = g + self.wd * self.p[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


class MLP:
    """x -> [hidden ReLU] -> out. loss in {'bce','softce','mse'}.
    'bce'   : multi-label sigmoid (Y in [0,1], shape [N,K]) with optional mask (missing labels)
    'softce': softmax cross-entropy with soft targets (Zooter's KL distillation)
    'mse'   : regression (Tryage loss prediction, Hybrid-LLM quality gap)"""

    def __init__(self, d_in, d_out, hidden=128, loss="bce", lr=2e-3, wd=1e-4, epochs=60, bs=128,
                 dropout=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.loss, self.lr, self.wd, self.epochs, self.bs, self.dropout = loss, lr, wd, epochs, bs, dropout
        self.hidden = hidden
        if hidden:
            self.P = {"W1": rng.normal(0, np.sqrt(2 / d_in), (d_in, hidden)), "b1": np.zeros(hidden),
                      "W2": rng.normal(0, np.sqrt(1 / hidden), (hidden, d_out)), "b2": np.zeros(d_out)}
        else:
            self.P = {"W2": rng.normal(0, 0.01, (d_in, d_out)), "b2": np.zeros(d_out)}
        self.rng = rng

    def _fwd(self, X, train=False):
        if self.hidden:
            h = X @ self.P["W1"] + self.P["b1"]
            a = np.maximum(h, 0)
            if train and self.dropout:
                mask = (self.rng.random(a.shape) > self.dropout) / (1 - self.dropout)
                a = a * mask
            return a @ self.P["W2"] + self.P["b2"], (h, a)
        return X @ self.P["W2"] + self.P["b2"], (None, X)

    def fit(self, X, Y, mask=None, sample_weight=None):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        if Y.ndim == 1:
            Y = Y[:, None]
        mask = np.ones_like(Y) if mask is None else np.asarray(mask, float)
        sw = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, float)
        opt = Adam(self.P, self.lr, self.wd)
        n = len(X)
        for _ in range(self.epochs):
            perm = self.rng.permutation(n)
            for s in range(0, n, self.bs):
                b = perm[s:s + self.bs]
                out, (h, a) = self._fwd(X[b], train=True)
                if self.loss == "bce":
                    d = (sigmoid(out) - Y[b]) * mask[b]
                elif self.loss == "softce":
                    d = softmax(out) - Y[b]
                else:
                    d = (out - Y[b]) * mask[b]
                d = d * sw[b, None] / len(b)
                g = {"W2": a.T @ d, "b2": d.sum(0)}
                if self.hidden:
                    dh = (d @ self.P["W2"].T) * (h > 0)
                    g |= {"W1": X[b].T @ dh, "b1": dh.sum(0)}
                opt.step(g)
        return self

    def raw(self, X):
        return self._fwd(np.asarray(X, float))[0]

    def predict(self, X):
        z = self.raw(X)
        if self.loss == "bce":
            return sigmoid(z)
        if self.loss == "softce":
            return softmax(z)
        return z
