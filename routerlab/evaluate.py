"""Cost-quality evaluation shared by all routers.

Metrics (all on the held-out test split):
  * curve           (cost/query, accuracy) at every knob value, plus its Pareto frontier
  * AUC             area under the Pareto frontier for cost in [cheapest, strongest] (normalised)
  * APGR            RouteLLM's average Performance-Gap-Recovered across the cost range
  * CPT(50/80)      RouteLLM's cost fraction (of the strongest model) needed to recover 50/80% of the gap
  * savings@best    % cost saved while matching the best single model's accuracy (-0.5pt tolerance)
  * max_acc         best accuracy reached (can exceed the best single model: complementary strengths)
  * dIBC            AutoMix's incremental-benefit-per-cost vs. the naive weak->strong interpolation
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .routers.base import Decision, Router
from .table import RoutingData


@dataclass
class Point:
    knob: float
    acc: float
    cost: float          # mean USD per query
    latency: float       # mean seconds per query
    share: list          # fraction of queries answered by each model
    calls_per_query: float


def score_decision(data: RoutingData, d: Decision) -> tuple[float, float, float, np.ndarray]:
    N = data.N
    if d.correct_override is not None:
        corr = np.where(d.final >= 0, data.correct[np.arange(N), np.clip(d.final, 0, None)], d.correct_override)
    else:
        corr = data.correct[np.arange(N), d.final]
    cost = (d.calls * data.cost).sum(1)
    if d.extra_cost is not None:
        cost = cost + d.extra_cost
    lat_mat = d.calls * data.latency
    lat = lat_mat.sum(1) if d.sequential else lat_mat.max(1)
    if d.extra_latency is not None:
        lat = lat + d.extra_latency
    return float(corr.mean()), float(cost.mean()), float(lat.mean()), corr


def sweep(router: Router, data: RoutingData) -> list[Point]:
    pts = []
    for k in router.knobs():
        d = router.decide(data, k)
        acc, cost, lat, _ = score_decision(data, d)
        share = [float((d.final == m).mean()) for m in range(data.M)]
        pts.append(Point(float(k), acc, cost, lat, share, float(d.calls.sum(1).mean())))
    return pts


def baselines(data: RoutingData) -> dict:
    acc = data.correct.mean(0)
    cost = data.cost.mean(0)
    lat = data.latency.mean(0)
    oracle_idx = []
    order = np.argsort(cost)
    for i in range(data.N):  # cheapest model that gets it right, else cheapest overall
        ok = [m for m in order if data.correct[i, m] > 0]
        oracle_idx.append(ok[0] if ok else order[0])
    oracle_idx = np.array(oracle_idx)
    o_acc = float(data.correct[np.arange(data.N), oracle_idx].mean())
    o_cost = float(data.cost[np.arange(data.N), oracle_idx].mean())
    return {"single": {m: {"acc": float(a), "cost": float(c), "latency": float(l)}
                       for m, a, c, l in zip(data.models, acc, cost, lat)},
            "oracle": {"acc": o_acc, "cost": o_cost}}


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(points)
    front, best = [], -1.0
    for c, a in pts:
        if a > best:
            front.append((c, a))
            best = a
    return front


def _acc_at(front, c):
    """Best accuracy achievable at cost <= c (step function)."""
    vals = [a for cc, a in front if cc <= c + 1e-15]
    return max(vals) if vals else np.nan


def metrics(points: list[Point], base: dict) -> dict:
    single = base["single"]
    names = list(single)
    costs = np.array([single[m]["cost"] for m in names])
    accs = np.array([single[m]["acc"] for m in names])
    weak, strong = int(np.argmin(costs)), int(np.argmax(accs))
    a_w, a_s, c_w, c_s = accs[weak], accs[strong], costs[weak], costs[strong]
    front = pareto([(p.cost, p.acc) for p in points] + [(c_w, a_w)])
    grid = np.linspace(c_w, max(c_s, c_w * 1.0001), 200)
    acc_grid = np.array([_acc_at(front, c) for c in grid])
    gap = max(a_s - a_w, 1e-9)
    pgr = np.clip((acc_grid - a_w) / gap, -1, 2)

    def cpt(x):
        ok = [p.cost for p in points if (p.acc - a_w) / gap >= x]
        return float(min(ok) / c_s) if ok else None

    best_single_acc = float(accs.max())
    match = [p.cost for p in points if p.acc >= best_single_acc - 0.005]
    savings = float(1 - min(match) / c_s) if match else None
    # AutoMix dIBC: slope of each operating point vs. the weak->strong straight line
    ibc_base = (a_s - a_w) / max(c_s - c_w, 1e-12)
    ibcs = [(p.acc - a_w) / (p.cost - c_w) for p in points if p.cost > c_w * 1.001]
    dibc = float((np.median(ibcs) - ibc_base) / abs(ibc_base) * 100) if ibcs and ibc_base else None
    return {
        "auc": float(np.nanmean(acc_grid)),
        "apgr": float(np.nanmean(pgr)),
        "cpt50": cpt(0.5), "cpt80": cpt(0.8),
        "savings_at_best_single_pct": None if savings is None else 100 * savings,
        "max_acc": float(max(p.acc for p in points)),
        "gain_over_best_single_pts": 100 * float(max(p.acc for p in points) - best_single_acc),
        "dIBC_pct": dibc,
        "weak": names[weak], "strong": names[strong],
    }


def evaluate_router(router: Router, train: RoutingData, val: RoutingData, test: RoutingData) -> dict:
    kinds = getattr(router, "kinds", None)
    if kinds:  # some papers only apply to classification / MCQ
        train, val, test = (d.subset(d.q.kind.isin(kinds).values) for d in (train, val, test))
    t0 = time.perf_counter()
    router.fit(train, val)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pts = sweep(router, test)
    route_ms = 1000 * (time.perf_counter() - t0) / max(1, len(pts) * test.N)
    base = baselines(test)
    out = {"router": router.name, "paper": router.paper, "fit_s": fit_s, "route_ms_per_query": route_ms,
           "points": [asdict(p) for p in pts], "metrics": metrics(pts, base), "baselines": base,
           "models": test.models, "n_test": test.N, "kinds": sorted(kinds) if kinds else "all"}
    if hasattr(router, "extra_report"):
        out["report"] = router.extra_report(train, test)
    return out


def plot_results(results: list[dict], out_png: str | Path, title="Cost vs accuracy (test split)"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 7))
    base = results[0]["baselines"]
    for m, v in base["single"].items():
        ax.scatter(v["cost"], v["acc"], marker="s", s=70, color="black", zorder=5)
        ax.annotate(m, (v["cost"], v["acc"]), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.scatter(base["oracle"]["cost"], base["oracle"]["acc"], marker="*", s=180, color="gold",
               edgecolor="black", zorder=6, label="oracle")
    cmap = plt.get_cmap("tab20")
    for i, r in enumerate(results):
        fr = pareto([(p["cost"], p["acc"]) for p in r["points"]])
        if not fr:
            continue
        c, a = zip(*fr)
        ax.plot(c, a, "-o", ms=3, lw=1.4, color=cmap(i % 20), label=f"{r['router']} (APGR {r['metrics']['apgr']:.2f})")
    ax.set_xscale("log")
    ax.set_xlabel("mean cost per query (USD, measured on your GPUs)")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def save(results, path):
    Path(path).write_text(json.dumps(results, indent=1, default=float))
