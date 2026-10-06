from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..table import RoutingData


@dataclass
class Decision:
    """What a router did for each query in a batch.

    final     [N] int   index of the model whose answer is returned (-1 => use `correct_override`)
    calls     [N, M]    which models were actually invoked (cascades call several)
    extra_cost [N]      verifier / scorer / fuser cost not covered by `calls`
    sequential bool     True for cascades (latency adds up); False for one-shot routing / parallel ensembles
    correct_override [N] correctness for answers not produced by a single model (fusion, voting)
    """
    final: np.ndarray
    calls: np.ndarray
    extra_cost: np.ndarray | None = None
    sequential: bool = False
    correct_override: np.ndarray | None = None
    extra_latency: np.ndarray | None = None

    @staticmethod
    def single(final: np.ndarray, M: int) -> "Decision":
        calls = np.zeros((len(final), M), dtype=bool)
        calls[np.arange(len(final)), final] = True
        return Decision(final=final, calls=calls)


class Router:
    name: str = "base"
    paper: str = ""
    needs: set[str] = set()      # extra signals: verify, reward, nll, fusion, logprobs

    def fit(self, train: RoutingData, val: RoutingData | None = None):
        return self

    def knobs(self) -> list[float]:
        """Values of the cost/quality trade-off knob swept to trace a curve."""
        return [0.0]

    def decide(self, data: RoutingData, knob: float) -> Decision:
        raise NotImplementedError

    # live gateway support: choose a model from the prompt alone (pre-generation routers)
    supports_live: bool = False

    def route_one(self, emb: np.ndarray, knob: float) -> int:
        raise NotImplementedError


def cheapest_first(data: RoutingData) -> np.ndarray:
    """Model order by mean cost (used by cascades)."""
    return np.argsort(data.mean_cost())


def lambda_grid(n=30, lo=1e-2, hi=20.0):
    return [0.0] + list(np.geomspace(lo, hi, n))


def utility_select(pred_acc: np.ndarray, mean_cost: np.ndarray, lam: float) -> np.ndarray:
    """argmax_m  P(correct_m) - lam * cost_m / max(cost). The shared selector used by most
    'predict-then-select' papers (Routoo, Fly-Swat, EmbedLLM, MetaLLM ...)."""
    c = mean_cost / mean_cost.max()
    return np.argmax(pred_acc - lam * c[None, :], axis=1)
