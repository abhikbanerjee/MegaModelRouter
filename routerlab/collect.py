"""Phase 1: run EVERY model on EVERY query once and store everything routers need.

This is how RouteLLM / RouterBench / EmbedLLM evaluate: once you have the full
(query x model) outcome table, any router - including cascades - can be replayed
offline exactly, for free, at every threshold. The GPU is only used here (and in the
live gateway).

Output (run dir):
  queries.parquet      qid, task, kind, prompt, answer, subject, choices, split
  embeddings.npy       query embeddings (row order == queries.parquet)
  outcomes.parquet     qid, model, text, pred, correct, n_in, n_out, latency_s, cost_usd,
                       margin_min, margin_first, mean_logprob
  verify.parquet       qid, model, p_verify, cost_usd          (AutoMix)
  reward.parquet       qid, model, reward                      (Zooter, Hybrid LLM)
  nll.parquet          qid, model, nll                         (Tryage)
  fusion.parquet       qid, text, correct, cost_usd            (LLM-Blender GenFuser)
  cost_meta.json       measured $/token per model, throughput
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .aux_models import Embedder
from .backends import Backend, MockBackend
from .config import PoolConfig
from .tasks import Query, score

PREFILL_WEIGHT = 0.15  # a prompt token costs ~15% of a generated token on GPU


def assign_splits(qs: list[Query], fr: dict, seed=0) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    out = {}
    for q in qs:
        u = rng.random()
        out[q.qid] = "train" if u < fr["train"] else ("val" if u < fr["train"] + fr["val"] else "test")
    return out


def _margins(g) -> tuple[float, float, float]:
    if not g.top_logprobs:
        return float("nan"), float("nan"), float("nan")
    ms = []
    for tops in g.top_logprobs:
        p = [math.exp(x) for x in tops[:2]] + [0.0, 0.0]
        ms.append(p[0] - p[1])
    mean_lp = float(np.mean(g.token_logprobs)) if g.token_logprobs else float("nan")
    return float(min(ms)), float(ms[0]), mean_lp


VERIFY_TMPL = ("[VERIFY] Context and question:\n{prompt}\n\nProposed answer: {answer}\n\n"
               "Is the proposed answer correct? Think briefly, then reply 'Verification: The answer is correct' "
               "or 'Verification: The answer is incorrect'.")


def collect(queries: list[Query], pool: PoolConfig, backend: Backend, exp: dict, out_dir: str | Path,
            embedder: Embedder | None = None, reward_model=None, pair_ranker=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gcfg, extras = exp["generation"], exp.get("extras", {})
    splits = assign_splits(queries, exp["split"], exp.get("seed", 0))
    qdf = pd.DataFrame([q.to_dict() | {"split": splits[q.qid]} for q in queries])
    qdf["choices"] = qdf["choices"].apply(json.dumps)
    qdf.to_parquet(out / "queries.parquet")
    embedder = embedder or Embedder(None)
    np.save(out / "embeddings.npy", embedder.encode([q.prompt for q in queries]))

    prompts, qids = [q.prompt for q in queries], [q.qid for q in queries]
    rows, meta = [], {}
    for spec in pool.models:
        m = spec.name
        print(f"[collect] {m} on {len(queries)} queries")
        backend.load(m)
        gens = []
        t0 = time.perf_counter()
        for kind in ("mcq", "cls", "math", "code"):  # group by output length for efficient batching
            idx = [i for i, q in enumerate(queries) if q.kind == kind]
            if not idx:
                continue
            mt = gcfg["max_tokens_mcq"] if kind in ("mcq", "cls") else gcfg["max_tokens_gen"]
            res = backend.generate(m, [prompts[i] for i in idx], mt, gcfg["temperature"], gcfg["logprobs"],
                                   qids=[qids[i] for i in idx])
            gens += list(zip(idx, res))
        wall = time.perf_counter() - t0
        if isinstance(backend, MockBackend):
            wall = sum(r[0].latency_s for _, r in gens) / 8   # pretend 8-way batching
        w_tokens = sum(r[0].n_out + PREFILL_WEIGHT * r[0].n_in for _, r in gens)
        usd_per_tok = (pool.gpu_hourly_usd * spec.num_gpus * wall / 3600) / max(w_tokens, 1)
        meta[m] = {"wall_s": wall, "usd_per_weighted_token": usd_per_tok,
                   "tokens_per_s": sum(r[0].n_out for _, r in gens) / max(wall, 1e-9),
                   "num_gpus": spec.num_gpus}
        for i, res in gens:
            g, q = res[0], queries[i]
            pred, ok = score(g.text, q)
            if spec.price_out_per_mtok is not None:
                cost = (g.n_in * (spec.price_in_per_mtok or 0) + g.n_out * spec.price_out_per_mtok) / 1e6
            else:
                cost = usd_per_tok * (g.n_out + PREFILL_WEIGHT * g.n_in)
            mm, mf, ml = _margins(g)
            rows.append(dict(qid=q.qid, model=m, text=g.text, pred=pred, correct=ok, n_in=g.n_in, n_out=g.n_out,
                             latency_s=g.latency_s, cost_usd=cost, margin_min=mm, margin_first=mf, mean_logprob=ml))
        # --- AutoMix self-verification (k samples at T=0.7) by this same model on its own answer
        if extras.get("self_verify") and m in extras.get("self_verify_models", []):
            mine = {r["qid"]: r for r in rows if r["model"] == m}
            vp = [VERIFY_TMPL.format(prompt=q.prompt, answer=mine[q.qid]["text"][-300:]) for q in queries]
            vres = backend.generate(m, vp, 64, 0.7, 0, n=extras.get("self_verify_k", 8), qids=qids)
            vrows = []
            for q, gs in zip(queries, vres):
                yes = np.mean([("is correct" in g.text.lower()) for g in gs])
                # k samples share one prefill (n>1 + prefix caching); each sample pays its own decode
                c = usd_per_tok * (PREFILL_WEIGHT * gs[0].n_in + sum(g.n_out for g in gs))
                vrows.append(dict(qid=q.qid, model=m, p_verify=float(yes), cost_usd=c))
            _append(out / "verify.parquet", pd.DataFrame(vrows))
        # --- Tryage: NLL of the gold answer under each model
        if extras.get("answer_nll"):
            gold = [f"Answer: {q.answer}" if q.kind != "code" else "" for q in queries]
            nll = backend.answer_nll(m, prompts, gold, qids=qids)
            _append(out / "nll.parquet", pd.DataFrame(dict(qid=qids, model=m, nll=nll)))
    backend.unload()
    odf = pd.DataFrame(rows)
    odf.to_parquet(out / "outcomes.parquet")
    (out / "cost_meta.json").write_text(json.dumps(meta, indent=2))

    # --- reward model scores over all outputs (Zooter, Hybrid LLM)
    if extras.get("reward_scores"):
        if reward_model is not None:
            pm = dict(zip(qids, prompts))
            rw = reward_model.score([pm[q] for q in odf.qid], odf.text.tolist())
        else:  # mock: noisy correctness signal
            rng = np.random.default_rng(1)
            rw = odf.correct.values * 2.0 + rng.normal(0, 0.9, len(odf))
        pd.DataFrame(dict(qid=odf.qid, model=odf.model, reward=rw)).to_parquet(out / "reward.parquet")

    # --- LLM-Blender GenFuser over top-3 PairRM-ranked candidates
    if extras.get("ensemble_fusion"):
        _fusion(queries, odf, pool, out, pair_ranker)
    print(f"[collect] done -> {out}")
    return out


def _append(path: Path, df: pd.DataFrame):
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path)


def _fusion(queries, odf, pool, out, pair_ranker, k=3):
    names = pool.names
    piv = odf.pivot(index="qid", columns="model", values="text")
    qmap = {q.qid: q for q in queries}
    cands = [[piv.loc[q.qid, m] for m in names] for q in queries]
    if pair_ranker is not None:
        ranks = pair_ranker.rank([q.prompt for q in queries], cands)
        topk = [[c[j] for j in np.argsort(r)[:k]] for c, r in zip(cands, ranks)]
        fused = pair_ranker.fuse([q.prompt for q in queries], topk)
        fuse_cost = 0.0   # 3B fuser; small. Add its measured cost if you care.
    else:  # mock: rank by reward, fuse = majority over top-k predictions (ties -> best ranked)
        rw = pd.read_parquet(out / "reward.parquet").pivot(index="qid", columns="model", values="reward")
        pr = odf.pivot(index="qid", columns="model", values="pred")
        fused = []
        for q in queries:
            order = rw.loc[q.qid, names].sort_values(ascending=False).index[:k]
            preds = [pr.loc[q.qid, m] for m in order]
            best = max(set(preds), key=lambda p: (preds.count(p), -preds.index(p)))
            fused.append(f"Answer: {best}")
        fuse_cost = 0.0
    rows = []
    allc = odf.groupby("qid").cost_usd.sum()
    for q, t in zip(queries, fused):
        _, ok = score(t, qmap[q.qid])
        rows.append(dict(qid=q.qid, text=t, correct=ok, cost_usd=float(allc[q.qid]) + fuse_cost))
    pd.DataFrame(rows).to_parquet(out / "fusion.parquet")
