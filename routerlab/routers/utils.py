from __future__ import annotations

import numpy as np

from ..table import RoutingData


def is_correct(pred: str, gold: str, kind: str) -> float:
    if pred is None:
        return 0.0
    if kind == "math":
        try:
            return float(abs(float(pred) - float(gold)) < 1e-6)
        except (TypeError, ValueError):
            return 0.0
    return float(str(pred).strip().lower() == str(gold).strip().lower())


def correct_of(data: RoutingData, preds) -> np.ndarray:
    return np.array([is_correct(p, g, k) for p, g, k in zip(preds, data.q.answer, data.q.kind)])


def quantile_knobs(scores: np.ndarray, n: int = 21) -> list[float]:
    """Thresholds at evenly spaced quantiles of a score, so a sweep covers 0..100% escalation."""
    qs = np.quantile(scores[~np.isnan(scores)], np.linspace(0, 1, n))
    return sorted(set([-np.inf] + list(qs) + [np.inf]))


def strong_weak(data: RoutingData, strong: str | None = None, weak: str | None = None) -> tuple[int, int]:
    acc, cost = data.correct.mean(0), data.mean_cost()
    s = data.models.index(strong) if strong else int(np.argmax(acc))
    w = data.models.index(weak) if weak else int(np.argmin(cost))
    return s, w


def standardize(train_X, *others):
    mu, sd = train_X.mean(0), train_X.std(0) + 1e-6
    return [(train_X - mu) / sd] + [(o - mu) / sd for o in others]
