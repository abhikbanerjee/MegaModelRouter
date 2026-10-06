"""Predictive Human Preference (Chip Huyen, 2024) - from model ranking to model routing.

Instead of one global leaderboard (Elo over all prompts), learn a *prompt-conditioned*
Bradley-Terry model:   P(A beats B | x) = sigmoid(s_A(x) - s_B(x)),  s_m(x) = w_m . x + b_m
trained on pairwise preferences. Two uses from the post:
  1. per-prompt ranking of models;
  2. routing: if a cheap model is predicted to be as good as (win-or-tie vs.) the strong one
     for this prompt, send it to the cheap model.
Preferences come from human votes (Chatbot Arena) or, here by default, from your own
outcomes: correctness first, reward score to break ties - an automated preference oracle.
Routing rule: the cheapest model m with P(m >= strongest | x) >= threshold (knob).
"""
from __future__ import annotations

import numpy as np

from ..nn import Adam, sigmoid
from .base import Decision, Router, cheapest_first


class PredictivePreferenceRouter(Router):
    name = "predictive-pref"
    paper = "Predictive Human Preference (Huyen 2024)"
    supports_live = True

    def __init__(self, epochs=40, lr=3e-3, wd=1e-3, extra_pairs=None, seed=0):
        self.epochs, self.lr, self.wd, self.extra, self.seed = epochs, lr, wd, extra_pairs, seed

    def _pairs(self, d):
        util = d.correct + (0.1 * np.tanh(d.reward) if d.reward is not None else 0)
        I, A, B, Y = [], [], [], []
        for a in range(d.M):
            for b in range(a + 1, d.M):
                diff = util[:, a] - util[:, b]
                keep = np.abs(diff) > 1e-9
                I += list(np.where(keep)[0]); A += [a] * keep.sum(); B += [b] * keep.sum()
                Y += list((diff[keep] > 0).astype(float))
        return np.array(I), np.array(A), np.array(B), np.array(Y)

    def fit(self, train, val=None):
        rng = np.random.default_rng(self.seed)
        I, A, B, Y = self._pairs(train)
        X = train.X
        if self.extra is not None:     # (X_extra, A, B, Y) e.g. Chatbot Arena votes mapped to your pool
            Xe, Ae, Be, Ye = self.extra
            I = np.concatenate([I, np.arange(len(Xe)) + len(X)]); X = np.vstack([X, Xe])
            A, B, Y = np.concatenate([A, Ae]), np.concatenate([B, Be]), np.concatenate([Y, Ye])
        d = X.shape[1]
        self.P = {"W": np.zeros((d, train.M)), "b": np.zeros(train.M)}
        opt = Adam(self.P, self.lr, self.wd)
        for _ in range(self.epochs):
            for bt in np.array_split(rng.permutation(len(I)), max(1, len(I) // 512)):
                x = X[I[bt]]
                s = x @ self.P["W"] + self.P["b"]
                z = s[np.arange(len(bt)), A[bt]] - s[np.arange(len(bt)), B[bt]]
                g = (sigmoid(z) - Y[bt]) / len(bt)
                G = np.zeros((len(bt), train.M))
                np.add.at(G, (np.arange(len(bt)), A[bt]), g)
                np.add.at(G, (np.arange(len(bt)), B[bt]), -g)
                opt.step({"W": x.T @ G, "b": G.sum(0)})
        self.order = list(cheapest_first(train))
        self.strong = int(np.argmax(train.correct.mean(0)))
        return self

    def scores(self, X):
        return X @ self.P["W"] + self.P["b"]

    def rank(self, X):
        """Per-prompt model ranking (best first)."""
        return np.argsort(-self.scores(X), 1)

    def knobs(self):
        return list(np.linspace(0.05, 0.95, 19)) + [1.01]

    def _choose(self, S, knob):
        out = np.full(len(S), self.strong)
        done = np.zeros(len(S), bool)
        for m in self.order:
            if m == self.strong:
                continue
            ok = ~done & (sigmoid(S[:, m] - S[:, self.strong]) >= knob)
            out[ok] = m
            done |= ok
        return out

    def decide(self, data, knob):
        return Decision.single(self._choose(self.scores(data.X), knob), data.M)

    def route_one(self, emb, knob):
        return int(self._choose(self.scores(emb[None]), knob)[0])
