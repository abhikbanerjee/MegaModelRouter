"""Loads a collected run into dense (queries x models) matrices for fast offline replay."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RoutingData:
    models: list[str]
    q: pd.DataFrame                  # one row per query
    X: np.ndarray                    # [N, d] query embeddings
    correct: np.ndarray              # [N, M] 0/1
    cost: np.ndarray                 # [N, M] USD
    latency: np.ndarray              # [N, M] seconds
    n_out: np.ndarray                # [N, M]
    pred: np.ndarray                 # [N, M] extracted answers (object)
    text: np.ndarray                 # [N, M] raw outputs (object)
    margin_min: np.ndarray           # [N, M]
    margin_first: np.ndarray
    mean_logprob: np.ndarray
    verify: np.ndarray | None = None       # [N, M] P(self-verifier says correct); nan if not collected
    verify_cost: np.ndarray | None = None
    reward: np.ndarray | None = None       # [N, M]
    nll: np.ndarray | None = None          # [N, M]
    fusion_correct: np.ndarray | None = None  # [N]
    fusion_cost: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def N(self):
        return len(self.q)

    @property
    def M(self):
        return len(self.models)

    def subset(self, mask) -> "RoutingData":
        idx = np.where(mask)[0] if np.asarray(mask).dtype == bool else np.asarray(mask)
        kw = {}
        for f in ("X", "correct", "cost", "latency", "n_out", "pred", "text", "margin_min", "margin_first",
                  "mean_logprob", "verify", "verify_cost", "reward", "nll", "fusion_correct", "fusion_cost"):
            v = getattr(self, f)
            kw[f] = None if v is None else v[idx]
        return replace(self, q=self.q.iloc[idx].reset_index(drop=True), **kw)

    def split(self, name: str) -> "RoutingData":
        return self.subset(self.q["split"].values == name)

    def tasks(self) -> np.ndarray:
        return self.q["task"].values

    def mean_cost(self) -> np.ndarray:
        return self.cost.mean(0)


def _mat(df, qids, models, col, fill=np.nan, dtype=float):
    p = df.pivot(index="qid", columns="model", values=col).reindex(index=qids, columns=models)
    return p.to_numpy(dtype=dtype, na_value=fill) if dtype is not object else p.to_numpy(dtype=object)


def load_run(run_dir: str | Path, models: list[str] | None = None) -> RoutingData:
    d = Path(run_dir)
    q = pd.read_parquet(d / "queries.parquet")
    q["choices"] = q["choices"].apply(json.loads)
    X = np.load(d / "embeddings.npy")
    o = pd.read_parquet(d / "outcomes.parquet")
    models = models or list(dict.fromkeys(o.model))
    qids = q.qid.tolist()

    def opt(name, col):
        p = d / f"{name}.parquet"
        return _mat(pd.read_parquet(p), qids, models, col) if p.exists() else None

    fus = d / "fusion.parquet"
    fc = fcost = None
    if fus.exists():
        f = pd.read_parquet(fus).set_index("qid").reindex(qids)
        fc, fcost = f.correct.values.astype(float), f.cost_usd.values.astype(float)
    meta = json.loads((d / "cost_meta.json").read_text()) if (d / "cost_meta.json").exists() else {}
    return RoutingData(
        models=models, q=q, X=X,
        correct=_mat(o, qids, models, "correct", 0.0), cost=_mat(o, qids, models, "cost_usd", 0.0),
        latency=_mat(o, qids, models, "latency_s", 0.0), n_out=_mat(o, qids, models, "n_out", 0.0),
        pred=_mat(o, qids, models, "pred", dtype=object), text=_mat(o, qids, models, "text", dtype=object),
        margin_min=_mat(o, qids, models, "margin_min"), margin_first=_mat(o, qids, models, "margin_first"),
        mean_logprob=_mat(o, qids, models, "mean_logprob"),
        verify=opt("verify", "p_verify"), verify_cost=opt("verify", "cost_usd"),
        reward=opt("reward", "reward"), nll=opt("nll", "nll"),
        fusion_correct=fc, fusion_cost=fcost, meta=meta,
    )
