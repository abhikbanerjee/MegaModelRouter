"""RouteLLM (Ong et al., 2024) - binary strong/weak routing from preference data.

The router estimates P(strong model wins | q). Route to the strong model iff that
probability >= alpha. Four router families in the paper; three are implemented here
(the causal-LLM classifier is a fine-tuned Llama-3-8B; see finetune.py for the
encoder-classifier recipe you can extend):

  mf          matrix factorisation  s(m,q) = w2 . (v_m * (W1 x_q)),  P = sigmoid(s_strong - s_weak)
  sw_ranking  similarity-weighted Bradley-Terry: train pairs weighted by cosine similarity to q
  clf         classifier on the query embedding (BERT router stand-in; finetune.py = real BERT)

Training data (the paper's key idea):
  * human preferences (Chatbot Arena) mapped onto your strong/weak tiers   -> `load_arena_pairs`
  * + data augmentation with golden labels from benchmarks: strong wins if strong is correct
    and weak is wrong; weak 'wins' (is good enough) otherwise. That is what `fit` builds from
    the train split, and it is what made the paper's routers generalise.
Headline metrics: APGR and CPT(50%/80%) (fraction of strong calls needed).
"""
from __future__ import annotations

import numpy as np

from ..nn import MLP, Adam, sigmoid
from .base import Decision, Router
from .utils import quantile_knobs, strong_weak


def golden_label_pairs(train, s, w):
    """y=1: strong needed. Ties where both are right count as weak-is-enough (y=0); both wrong are dropped
    with weight 0.5 (neither is better) - mirrors RouteLLM's tie handling."""
    cs, cw = train.correct[:, s], train.correct[:, w]
    y = ((cs == 1) & (cw == 0)).astype(float)
    wt = np.where((cs == 0) & (cw == 0), 0.5, 1.0)
    return train.X, y, wt


def load_arena_pairs(strong_models: set[str], weak_models: set[str], embedder, limit=None):
    """Chatbot Arena 55k -> (X, y) for strong-vs-weak. Map arena model names into your tiers
    (the paper clustered arena models into 10 Elo tiers; pass tier sets here)."""
    import json

    from datasets import load_dataset

    ds = load_dataset("lmsys/lmsys-arena-human-preference-55k", split="train")
    xs, ys = [], []
    for r in ds:
        a, b = r["model_a"], r["model_b"]
        if a in strong_models and b in weak_models:
            sw = {"a": 1.0, "b": 0.0}
        elif b in strong_models and a in weak_models:
            sw = {"a": 0.0, "b": 1.0}
        else:
            continue
        if r["winner_tie"]:
            continue
        y = sw["a"] if r["winner_model_a"] else sw["b"]
        xs.append(json.loads(r["prompt"])[0])
        ys.append(y)
        if limit and len(xs) >= limit:
            break
    return embedder.encode(xs), np.array(ys)


class RouteLLMRouter(Router):
    paper = "RouteLLM (2406.18665)"
    supports_live = True

    def __init__(self, variant="mf", strong=None, weak=None, extra_pairs=None, k=64, epochs=60, seed=0):
        self.variant, self.strong, self.weak, self.extra, self.k, self.epochs, self.seed = \
            variant, strong, weak, extra_pairs, k, epochs, seed
        self.name = f"routellm-{variant}"

    def fit(self, train, val=None):
        self.s, self.w = strong_weak(train, self.strong, self.weak)
        X, y, wt = golden_label_pairs(train, self.s, self.w)
        if self.extra is not None:  # arena preference pairs (X_a, y_a)
            X = np.vstack([X, self.extra[0]])
            y = np.concatenate([y, self.extra[1]])
            wt = np.concatenate([wt, np.ones(len(self.extra[1]))])
        self.Xtr, self.ytr, self.wtr = X, y, wt
        if self.variant == "mf":
            self._fit_mf(X, y, wt)
        elif self.variant == "clf":
            self.clf = MLP(X.shape[1], 1, hidden=128, loss="bce", epochs=self.epochs, seed=self.seed).fit(
                X, y, sample_weight=wt)
        elif self.variant == "sw_ranking":
            self.Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        self.knob_list = quantile_knobs(self.score(val.X if val is not None and val.N else X))
        return self

    def _fit_mf(self, X, y, wt):
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        P = {"W1": rng.normal(0, 1 / np.sqrt(d), (d, self.k)), "V": rng.normal(0, 0.3, (2, self.k)),
             "w2": rng.normal(0, 0.3, self.k), "b": np.zeros(1)}
        opt = Adam(P, 3e-3, 1e-4)
        for _ in range(self.epochs):
            for b in np.array_split(rng.permutation(len(X)), max(1, len(X) // 256)):
                Z = X[b] @ P["W1"]
                dv = P["V"][0] - P["V"][1]              # strong - weak
                logit = (Z * dv) @ P["w2"] + P["b"]
                g = (sigmoid(logit) - y[b]) * wt[b] / len(b)
                gz = g[:, None] * dv * P["w2"]
                opt.step({"w2": (g[:, None] * Z * dv).sum(0), "b": np.array([g.sum()]),
                          "V": np.stack([(g[:, None] * Z * P["w2"]).sum(0), -(g[:, None] * Z * P["w2"]).sum(0)]),
                          "W1": X[b].T @ gz})
        self.P = P

    def score(self, X):
        """P(strong model needed)."""
        if self.variant == "mf":
            Z = X @ self.P["W1"]
            return sigmoid((Z * (self.P["V"][0] - self.P["V"][1])) @ self.P["w2"] + self.P["b"])
        if self.variant == "clf":
            return self.clf.predict(X)[:, 0]
        # similarity-weighted ranking: weights = exp(gamma * cos), BT reduces to weighted win-rate here
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        sim = Xn @ self.Xn.T
        W = np.exp(10 * (sim - sim.max(1, keepdims=True))) * self.wtr
        return (W @ self.ytr + 0.5) / (W.sum(1) + 1.0)

    def knobs(self):
        return self.knob_list

    def decide(self, data, knob):
        p = self.score(data.X)
        return Decision.single(np.where(p >= knob, self.s, self.w), data.M)

    def route_one(self, emb, knob):
        return self.s if self.score(emb[None])[0] >= knob else self.w
