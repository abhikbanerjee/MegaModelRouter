"""LLM-Blender (Jiang, Ren, Lin - ACL 2023) - ensemble by pairwise ranking + generative fusion.

All N models answer. PairRanker (a cross-encoder that compares two candidates jointly with the
input, much better than scoring each alone) ranks them; GenFuser (a seq2seq model) fuses the
top-K into one answer. This trades MORE cost (every model runs) for quality above the best
single model - the upper-right reference point that routers try to approach cheaply.

Operating points:
  knob k = 1..M : run the k best models (by validation accuracy), return PairRanker's top-1
                  (ranking signal = PairRM if collected, else the reward model)
  knob 0        : full LLM-Blender: all models + GenFuser over top-3 (fusion.parquet)
"""
from __future__ import annotations

import numpy as np

from .base import Decision, Router


class LLMBlenderRouter(Router):
    name = "llm-blender"
    paper = "LLM-Blender (2306.02561)"
    needs = {"reward"}

    def fit(self, train, val=None):
        v = val if val is not None and val.N else train
        self.order = list(np.argsort(-v.correct.mean(0)))
        return self

    def knobs(self):
        return list(range(1, len(self.order) + 1)) + [0]

    def decide(self, data, knob):
        N = data.N
        calls = np.zeros((N, data.M), bool)
        if knob == 0:
            if data.fusion_correct is None:
                knob = len(self.order)
            else:
                calls[:] = True
                return Decision(final=np.full(N, -1), calls=calls, correct_override=data.fusion_correct,
                                extra_cost=data.fusion_cost - data.cost.sum(1))
        S = self.order[: int(knob)]
        calls[:, S] = True
        score = data.reward if data.reward is not None else data.mean_logprob
        sub = np.where(calls, np.nan_to_num(score, nan=-1e9), -np.inf)
        return Decision(final=np.argmax(sub, 1), calls=calls)
