"""Cache & Distil / neural caching (Ramirez et al., ACL Findings 2024).

An ONLINE setting: requests arrive one by one. A small *student* (here: a logistic-regression
head on query embeddings, retrained every `retrain_every` requests on the teacher's labels so
far) answers when it can; a *policy* decides which requests go to the expensive teacher LLM,
whose answers both serve the user and become training data for the student.
Policies from the paper (classic active-learning criteria):
  front       - front-loading: call the teacher for the first B requests, then never
  random      - call with probability = budget rate
  margin      - Margin Sampling: call when student's top1-top2 prob margin is small  (best)
  qbc         - Query-by-Committee: disagreement among bootstrap students             (best)
Budget knob = fraction of requests that may hit the teacher. Classification tasks only
(student predicts a label), as in the paper. Teacher = the strongest model in the pool.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import Decision, Router
from .utils import is_correct


class CacheDistilRouter(Router):
    paper = "Cache & Distil (2310.13561)"
    kinds = {"cls"}

    def __init__(self, policy="margin", retrain_every=50, committee=4, window=200, seed=0):
        self.policy, self.every, self.K, self.window, self.seed = policy, retrain_every, committee, window, seed
        self.name = f"cache-distil-{policy}"

    def fit(self, train, val=None):
        self.T = int(np.argmax(train.correct.mean(0)))   # teacher
        return self   # the student is learned online during decide()

    def knobs(self):
        return [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

    def _fit_students(self, X, y, rng):
        if len(set(y)) < 2:
            return None
        ms = []
        for k in range(self.K if self.policy == "qbc" else 1):
            idx = rng.integers(0, len(X), len(X)) if self.policy == "qbc" else np.arange(len(X))
            if len(set(y[idx])) < 2:
                idx = np.arange(len(X))
            ms.append(LogisticRegression(max_iter=300, C=2.0).fit(X[idx], y[idx]))
        return ms

    def decide(self, data, knob):
        rng = np.random.default_rng(self.seed)
        N = data.N
        budget = int(round(knob * N))
        order = np.arange(N)                       # arrival order = dataset order (already shuffled)
        tasks = data.q.task.values
        teacher_pred = data.pred[:, self.T]
        labels_X, labels_y, labels_t = [], [], []
        students: dict = {}
        used, recent = 0, []
        calls = np.zeros((N, data.M), bool)
        final = np.full(N, -1)
        override = np.zeros(N)
        since = 0
        for step, i in enumerate(order):
            t = tasks[i]
            st = students.get(t)
            remaining = N - step
            # uncertainty of the student for this request (higher = more uncertain)
            if st is None:
                u, spred = np.inf, None
            else:
                classes = sorted(set().union(*[set(m.classes_) for m in st]))
                probs = []
                for m in st:   # align bootstrap students that saw different label subsets
                    pp = np.zeros(len(classes))
                    pp[[classes.index(c) for c in m.classes_]] = m.predict_proba(data.X[i:i + 1])[0]
                    probs.append(pp)
                p = np.mean(probs, 0)
                spred = classes[int(np.argmax(p))]
                if self.policy == "qbc":
                    votes = [int(np.argmax(pp)) for pp in probs]
                    u = 1 - max(votes.count(v) for v in set(votes)) / len(votes)
                else:
                    s = np.sort(p)[::-1]
                    u = 1 - (s[0] - (s[1] if len(s) > 1 else 0))
            if self.policy == "front":
                ask = used < budget
            elif self.policy == "random":
                ask = used < budget and rng.random() < budget / N
            else:  # adaptive threshold: spend the remaining budget on the most uncertain share
                rate = (budget - used) / max(remaining, 1)
                window = recent[-self.window:]
                thr = np.quantile(window, 1 - rate) if len(window) >= 20 and 0 < rate < 1 else (-np.inf if rate >= 1 else np.inf)
                ask = used < budget and (u >= thr or spred is None)
            recent.append(u if np.isfinite(u) else 1.0)
            if ask:
                used += 1
                calls[i, self.T] = True
                final[i] = self.T
                labels_X.append(data.X[i]); labels_y.append(teacher_pred[i]); labels_t.append(t)
                since += 1
                if since >= self.every or st is None:
                    since = 0
                    lt = np.array(labels_t)
                    for task in set(labels_t):
                        msk = lt == task
                        students[task] = self._fit_students(np.array(labels_X)[msk], np.array(labels_y, dtype=object)[msk].astype(str), rng) or students.get(task)
            else:
                pred = spred if spred is not None else ""
                override[i] = is_correct(pred, data.q.answer.values[i], "cls")
        return Decision(final=final, calls=calls, correct_override=override)
