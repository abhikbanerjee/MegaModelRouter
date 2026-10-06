"""AutoMix (Madaan et al., 2023/2024) - few-shot self-verification + meta-verifier cascade.

1. The small model answers.
2. The SAME small model is prompted k times (T=0.7) to verify its own answer against the
   context; the fraction of 'correct' verdicts is a noisy confidence (collect.py, VERIFY_TMPL).
3. A meta-verifier decides whether to trust that noisy signal or escalate:
     variant="threshold": escalate if confidence < theta
     variant="pomdp"    : the paper's POMDP treats the true answer quality as a hidden state and
                          the verifier output as an observation. With a one-step horizon this
                          reduces to: estimate the belief P(correct | verifier bin) from training
                          data (Beta-smoothed), escalate if that belief < theta. Handles verifiers
                          that are systematically over/under-confident.
Works for multi-step chains (small -> medium -> large). Metric of choice: dIBC.
"""
from __future__ import annotations

import numpy as np

from .base import Decision, Router, cheapest_first


class AutoMixRouter(Router):
    paper = "AutoMix (2310.12963)"
    needs = {"verify"}

    def __init__(self, variant="pomdp", chain: list[str] | None = None, bins=9):
        # chain=None -> [cheapest, strongest]; pass e.g. ["qwen2.5-1.5b","qwen2.5-7b","qwen2.5-32b-awq"] for 3 levels
        self.variant, self.chain_names, self.bins = variant, chain, bins
        self.name = f"automix-{variant}"

    def fit(self, train, val=None):
        assert train.verify is not None, "run collection with extras.self_verify=true"
        if self.chain_names:
            self.chain = [train.models.index(m) for m in self.chain_names]
        else:  # paper default: cheapest verifiable model -> strongest model
            order = list(cheapest_first(train))
            ver = [m for m in order if not np.all(np.isnan(train.verify[:, m]))]
            strongest = int(np.argmax(train.correct.mean(0)))
            self.chain = [m for m in ver if m != strongest][:1] + [strongest]
        self.edges = np.linspace(0, 1, self.bins + 1)
        self.belief = {}
        for m in self.chain[:-1]:
            v, c = train.verify[:, m], train.correct[:, m]
            b = np.clip(np.digitize(v, self.edges) - 1, 0, self.bins - 1)
            prior = c.mean()
            self.belief[m] = np.array([(c[b == j].sum() + 2 * prior) / ((b == j).sum() + 2) for j in range(self.bins)])
        return self

    def _conf(self, data, m):
        v = np.nan_to_num(data.verify[:, m], nan=0.0)
        if self.variant == "threshold":
            return v
        b = np.clip(np.digitize(v, self.edges) - 1, 0, self.bins - 1)
        return self.belief[m][b]

    def knobs(self):
        return list(np.linspace(0, 1.0001, 21))

    def decide(self, data, knob):
        N = data.N
        final = np.full(N, self.chain[-1])
        calls = np.zeros((N, data.M), bool)
        extra = np.zeros(N)
        extra_lat = np.zeros(N)
        active = np.ones(N, bool)
        for m in self.chain[:-1]:
            calls[active, m] = True
            extra[active] += np.nan_to_num(data.verify_cost[active, m])
            extra_lat[active] += 0.5 * data.latency[active, m]   # verification pass ~ half a generation
            accept = active & (self._conf(data, m) >= knob)
            final[accept] = m
            active &= ~accept
        calls[active, self.chain[-1]] = True
        return Decision(final=final, calls=calls, extra_cost=extra, sequential=True, extra_latency=extra_lat)
