"""LLM Routing with Benchmark Datasets (Shnitzer et al., 2023).

Reuse existing benchmark results (which model got which question right) to pick the best LLM
for a NEW task with no labels. Reduces to one binary 'correctness predictor' g_m(x) per model
(paper: kNN on sentence embeddings). For a new task with inputs x_1..x_n:
   S1(m) = mean_i g_m(x_i)                                    - naive estimated accuracy
   S2(m) = mean_i [ g_m(x_i) p_m + (1-g_m(x_i))(1-p_m) ]      - accounts for how accurate the
           correctness predictor itself is on this (possibly out-of-distribution) task, p_m,
           which is estimated from embedding distance to the training tasks
   S3    - use S2, but fall back to the best-on-average model when the predicted gain over it
           is smaller than the uncertainty of the estimate.
The paper's caveat: correctness predictors are weak out-of-distribution, so S2/S3 matter.

This router does per-query routing (argmax g_m(x) - lambda cost) for the shared cost curve,
and `task_level_report` reproduces the paper's leave-one-task-out model selection.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from .base import Decision, Router, lambda_grid, utility_select


class BenchmarkRoutingRouter(Router):
    name = "benchmark-knn"
    paper = "Routing with benchmark datasets (2309.15789)"
    supports_live = True

    def __init__(self, k=10, task_col="subject"):
        self.k, self.task_col = k, task_col

    def _fit_g(self, X, Y):
        gs = []
        for m in range(Y.shape[1]):
            y = Y[:, m]
            gs.append(KNeighborsClassifier(self.k, metric="cosine").fit(X, y) if len(set(y)) > 1 else float(y[0]))
        return gs

    @staticmethod
    def _pred(gs, X):
        return np.stack([g.predict_proba(X)[:, -1] if not isinstance(g, float) else np.full(len(X), g) for g in gs], 1)

    def fit(self, train, val=None):
        self.gs = self._fit_g(train.X, train.correct)
        self.cost = train.mean_cost()
        self.train = train
        return self

    def knobs(self):
        return lambda_grid()

    def decide(self, data, knob):
        return Decision.single(utility_select(self._pred(self.gs, data.X), self.cost, knob), data.M)

    def route_one(self, emb, knob):
        return int(utility_select(self._pred(self.gs, emb[None]), self.cost, knob)[0])

    def extra_report(self, train, test):
        rows = self.task_level_report(train)
        if not rows:
            return {}
        keys = ["acc_best_true", "acc_best_on_avg", "acc_S1", "acc_S2", "acc_S3"]
        return {"n_tasks": len(rows), "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys}, "tasks": rows}

    def task_level_report(self, data, min_items=20):
        """Leave-one-task-out: for each task, learn from all OTHER tasks, pick a model with S1/S2/S3."""
        tasks = data.q[self.task_col].values
        uniq = [t for t in np.unique(tasks) if (tasks == t).sum() >= min_items]
        # 1) per-task OOD-ness -> predictor accuracy regression (fit on held-out tasks)
        feats, accs, rows = [], [], []
        cache = {}
        for t in uniq:
            tr, te = tasks != t, tasks == t
            gs = self._fit_g(data.X[tr], data.correct[tr])
            G = self._pred(gs, data.X[te])
            nn = NearestNeighbors(n_neighbors=self.k, metric="cosine").fit(data.X[tr])
            dist = nn.kneighbors(data.X[te])[0].mean()
            p_true = ((G > 0.5) == (data.correct[te] == 1)).mean(0)      # predictor accuracy per model
            cache[t] = (G, dist, tr, te)
            for m in range(data.M):
                feats.append([dist]); accs.append(p_true[m])
        reg = LinearRegression().fit(np.array(feats), np.array(accs))
        best_avg = int(np.argmax(data.correct.mean(0)))
        for t in uniq:
            G, dist, tr, te = cache[t]
            p = float(np.clip(reg.predict([[dist]])[0], 0.5, 1.0))
            s1 = G.mean(0)
            s2 = (G * p + (1 - G) * (1 - p)).mean(0)
            unc = np.sqrt(s2 * (1 - s2) / te.sum()) + (1 - p)
            s3 = int(np.argmax(s2)) if s2.max() - s2[best_avg] > unc[best_avg] else best_avg
            true = data.correct[te].mean(0)
            rows.append({"task": t, "n": int(te.sum()), "best_true": data.models[int(np.argmax(true))],
                         "acc_best_true": float(true.max()), "acc_best_on_avg": float(true[best_avg]),
                         "acc_S1": float(true[int(np.argmax(s1))]), "acc_S2": float(true[int(np.argmax(s2))]),
                         "acc_S3": float(true[s3]), "est_predictor_acc": p})
        return rows
