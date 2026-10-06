"""Uncertainty-based two-tier selection (Ramirez, Birch, Titov - COLM 2024).

No router model at all: run the small LLM, read the token probabilities it already produced,
and escalate to the large LLM only when it is uncertain. Uncertainty = Margin Sampling,
p(top-1) - p(top-2) at the decision token (smallest margin over the answer tokens). The paper
found this beats trained routers and cascades in 25/27 setups. Zero training, zero extra calls.

signal: "margin_min" (default), "margin_first", or "mean_logprob".
"""
from __future__ import annotations

import numpy as np

from .base import Decision, Router
from .utils import quantile_knobs, strong_weak


class TwoTierRouter(Router):
    paper = "Uncertainty two-tier (2405.02134)"
    needs = {"logprobs"}

    def __init__(self, signal="margin_min", small=None, large=None):
        self.signal, self.small, self.large = signal, small, large
        self.name = f"two-tier-{signal.replace('_', '')}"

    def fit(self, train, val=None):
        self.L, self.S = strong_weak(train, self.large, self.small)
        self.knob_list = quantile_knobs(getattr(train, self.signal)[:, self.S])
        return self

    def knobs(self):
        return self.knob_list

    def decide(self, data, knob):
        conf = np.nan_to_num(getattr(data, self.signal)[:, self.S], nan=-np.inf)
        esc = conf < knob
        calls = np.zeros((data.N, data.M), bool)
        calls[:, self.S] = True
        calls[esc, self.L] = True
        return Decision(final=np.where(esc, self.L, self.S), calls=calls, sequential=True)
