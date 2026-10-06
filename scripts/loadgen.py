#!/usr/bin/env python
"""Replay held-out benchmark queries through the gateway for several strategies at once, score
every answer, send the score back as feedback (-> Grafana accuracy panels + Phoenix annotations),
and print a side-by-side comparison.

  python scripts/loadgen.py --run runs/mock --strategies single/qwen2.5-32b-awq,two-tier,automix,predict/routellm-mf --n 200 --qps 8
  python scripts/loadgen.py --run runs/mock --strategies all --n 100
Each query is sent once per strategy (interleaved), so all strategies see the same traffic.
"""
import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
import pandas as pd  # noqa: E402

from routerlab.tasks import Query, score  # noqa: E402


async def main(a):
    q = pd.read_parquet(Path(a.run) / "queries.parquet")
    q = q[q.split == "test"]
    q = q.sample(min(a.n, len(q)), random_state=a.seed) if a.n else q
    async with httpx.AsyncClient(timeout=600) as c:
        avail = (await c.get(f"{a.gateway}/strategies")).json()
        names = [s["strategy"] for s in avail["strategies"]]
        strategies = names if a.strategies == "all" else a.strategies.split(",")
        bad = [s for s in strategies if s not in names and not (s.startswith("cache+") and s[6:] in names)]
        if bad:
            sys.exit(f"unknown strategies {bad}; available: {names}")
        jobs = [(r, s) for r in q.itertuples() for s in strategies]
        if not a.ordered:
            random.Random(a.seed).shuffle(jobs)
        rows, sem = [], asyncio.Semaphore(a.concurrency)

        async def one(r, s):
            qq = Query(r.qid, r.task, r.kind, r.prompt, r.answer, r.subject, json.loads(r.choices))
            mt = 16 if r.kind in ("mcq", "cls") else 512
            hdr = {"x-session-id": f"loadgen-{a.seed}"}
            if a.knob is not None:
                hdr["x-router-knob"] = str(a.knob)
            async with sem:
                t0 = time.perf_counter()
                resp = await c.post(f"{a.gateway}/v1/chat/completions", headers=hdr,
                                    json={"model": f"router:{s}", "max_tokens": mt,
                                          "messages": [{"role": "user", "content": r.prompt}]})
                lat = time.perf_counter() - t0
            if resp.status_code != 200:
                rows.append({"strategy": s, "qid": r.qid, "error": resp.text[:200]})
                return
            js = resp.json()
            rl = js["routerlab"]
            _, ok = score(js["choices"][0]["message"]["content"], qq)
            await c.post(f"{a.gateway}/feedback", json={"request_id": rl["request_id"], "correct": bool(ok)})
            rows.append({"strategy": s, "qid": r.qid, "task": r.task, "answered_by": rl["answered_by"],
                         "n_calls": len(rl["calls"]), "correct": ok, "cost_usd": rl["cost_usd"],
                         "baseline_cost_usd": rl["baseline_cost_usd"], "latency_s": lat,
                         "escalated": rl["escalated"], "cache_hit": rl["cache_hit"], "trace_id": rl["trace_id"]})

        tasks = []
        for r, s in jobs:
            tasks.append(asyncio.create_task(one(r, s)))
            if a.qps:
                await asyncio.sleep(1 / a.qps)
        await asyncio.gather(*tasks)
    df = pd.DataFrame(rows)
    out = Path(a.run) / f"live_compare_{int(time.time())}.csv"
    df.to_csv(out, index=False)
    if "error" in df:
        errs = df[df.error.notna()]
        if len(errs):
            print(f"{len(errs)} errors, e.g. {errs.iloc[0].to_dict()}")
        df = df[df.error.isna()]
    summ = df.groupby("strategy").agg(n=("qid", "count"), accuracy=("correct", "mean"),
                                      cost_per_1k_usd=("cost_usd", lambda x: 1000 * x.mean()),
                                      baseline_per_1k_usd=("baseline_cost_usd", lambda x: 1000 * x.mean()),
                                      p50_s=("latency_s", "median"), p95_s=("latency_s", lambda x: x.quantile(.95)),
                                      calls_per_q=("n_calls", "mean"), escalated=("escalated", "mean"),
                                      cache_hits=("cache_hit", "mean"))
    summ["saving_vs_strongest_%"] = 100 * (1 - summ.cost_per_1k_usd / summ.baseline_per_1k_usd)
    print(summ.sort_values("accuracy", ascending=False).to_string(float_format=lambda x: f"{x:.4g}"))
    print(f"\nper-query rows -> {out}\ntraces -> Phoenix (project 'llm-routing'), metrics -> Grafana 'LLM Routing (OTel)'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gateway", default="http://localhost:9000")
    ap.add_argument("--strategies", default="single/qwen2.5-32b-awq,two-tier,automix,frugalgpt,predict/routellm-mf")
    ap.add_argument("--knob", type=float, default=None, help="override every strategy's knob (default: knobs.json)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--qps", type=float, default=10, help="request launch rate (0 = as fast as possible)")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--ordered", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    asyncio.run(main(ap.parse_args()))
