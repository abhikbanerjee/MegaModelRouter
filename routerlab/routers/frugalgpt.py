"""FrugalGPT (Chen, Zaharia, Zou - 2023) - learned LLM cascade.

Three cost levers in the paper: prompt adaptation, LLM approximation (completion cache /
model fine-tuning), and the LLM cascade - the one that delivers most of the savings:
  * query models in a chosen order; after each answer a *scoring function* g(q, a) in [0,1]
    (paper: DistilBERT regressor on query+answer) estimates reliability;
  * stop if g >= tau_j for that stage, else go to the next model;
  * the model list AND thresholds are optimised jointly to maximise accuracy under a budget.
Implementation: per-model scorer (MLP on [query emb ; answer features]); exhaustive search
over ordered chains of length <=3 and quantile threshold grids on the train split; for each
budget knob return the best chain/thresholds found. variant="cache" adds LLM approximation:
a semantic completion cache that reuses an earlier answer when a new query is near-identical.
Paper: matched GPT-4 with up to 98% cost reduction (on API price ratios far larger than
between open models - expect smaller but real savings on a GPU pool).
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from ..nn import MLP
from .base import Decision, Router, cheapest_first

_HV = HashingVectorizer(n_features=64, alternate_sign=False, norm="l2")


def live_features(emb, pred, margin_min, mean_logprob, n_out):
    """Same features as answer_features, for ONE live answer (used by the gateway)."""
    A = _HV.transform([str(pred)]).toarray()[0]
    stats = np.array([0.5 if margin_min is None else margin_min, mean_logprob or 0.0, np.log1p(n_out) / 7])
    return np.concatenate([emb, A, stats])[None]


def answer_features(data, m):
    """Features of model m's answer: hashed tokens of the extracted answer + confidence stats."""
    A = _HV.transform([str(p) for p in data.pred[:, m]]).toarray()
    stats = np.stack([np.nan_to_num(data.margin_min[:, m], nan=0.5), np.nan_to_num(data.mean_logprob[:, m]),
                      np.log1p(data.n_out[:, m]) / 7], 1)
    # agreement with other models is NOT used: it would require calling them (cost leak)
    return np.hstack([data.X, A, stats])


class FrugalGPTRouter(Router):
    paper = "FrugalGPT (2305.05176)"

    def __init__(self, max_len=3, n_thr=6, variant="cascade", cache_sim=0.97, seed=0):
        self.max_len, self.n_thr, self.variant, self.cache_sim, self.seed = max_len, n_thr, variant, cache_sim, seed
        self.name = "frugalgpt" + ("-cache" if variant == "cache" else "")

    def fit(self, train, val=None):
        self.scorers = [MLP(answer_features(train, 0).shape[1], 1, hidden=64, loss="bce", epochs=40, seed=m)
                        .fit(answer_features(train, m), train.correct[:, m]) for m in range(train.M)]
        opt = val if val is not None and val.N >= 50 else train
        G = self._scores(opt)
        order = list(cheapest_first(train))
        chains = [list(c) for L in range(1, self.max_len + 1) for c in itertools.combinations(order, L)]
        cands = []
        for ch in chains:
            grids = [np.quantile(G[:, m], np.linspace(0, 0.95, self.n_thr)) for m in ch[:-1]]
            for thr in itertools.product(*grids):
                acc, cost = self._run(opt, G, ch, thr, score_only=True)
                cands.append((cost, acc, ch, thr))
        cands.sort(key=lambda t: t[0])
        self.frontier, best = [], -1
        for c in cands:              # Pareto frontier of (cost, acc) on the optimisation split
            if c[1] > best:
                self.frontier.append(c)
                best = c[1]
        self.cmin, self.cmax = self.frontier[0][0], self.frontier[-1][0]
        return self

    def _scores(self, data):
        return np.stack([s.predict(answer_features(data, m))[:, 0] for m, s in enumerate(self.scorers)], 1)

    def _run(self, data, G, chain, thr, score_only=False):
        N = data.N
        final = np.full(N, chain[-1])
        calls = np.zeros((N, data.M), bool)
        active = np.ones(N, bool)
        for j, m in enumerate(chain):
            calls[active, m] = True
            if j < len(chain) - 1:
                stop = active & (G[:, m] >= thr[j])
                final[stop] = m
                active &= ~stop
        if score_only:
            return data.correct[np.arange(N), final].mean(), (calls * data.cost).sum(1).mean()
        return final, calls

    def knobs(self):
        return list(np.linspace(0, 1, 15))

    # ---- live serving: the gateway walks the chain itself and asks us for the plan and the scores
    def live_plan(self, knob):
        budget = self.cmin + knob * (self.cmax - self.cmin)
        _, _, chain, thr = max((c for c in self.frontier if c[0] <= budget + 1e-15), key=lambda c: c[1])
        return list(chain), list(thr)

    def live_score(self, m, emb, pred, margin_min, mean_logprob, n_out) -> float:
        return float(self.scorers[m].predict(live_features(emb, pred, margin_min, mean_logprob, n_out))[0, 0])

    def decide(self, data, knob):
        budget = self.cmin + knob * (self.cmax - self.cmin)
        _, _, chain, thr = max((c for c in self.frontier if c[0] <= budget + 1e-15), key=lambda c: c[1])
        G = self._scores(data)
        final, calls = self._run(data, G, chain, thr)
        if self.variant == "cache":     # LLM approximation: reuse answers of near-duplicate earlier queries
            Xn = data.X / (np.linalg.norm(data.X, axis=1, keepdims=True) + 1e-9)
            override = data.correct[np.arange(data.N), final].astype(float)
            for i in range(1, data.N):
                sims = Xn[:i] @ Xn[i]
                j = int(np.argmax(sims))
                if sims[j] >= self.cache_sim:
                    calls[i] = False
                    same = data.q.answer.values[i] == data.q.answer.values[j]
                    override[i] = override[j] * same
            return Decision(final=np.full(data.N, -1), calls=calls, sequential=True, correct_override=override)
        return Decision(final=final, calls=calls, sequential=True)
