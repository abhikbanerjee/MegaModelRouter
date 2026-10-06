#!/usr/bin/env python
"""EcoAssistant, full method, live against the vLLM pool (scripts/serve_pool.sh first).

Compares, on the same GSM8K stream:
  A) strongest model alone               B) hierarchy only
  C) demonstrations only (strongest)     D) hierarchy + demonstrations (EcoAssistant)
and prints success rate and $ cost (from the run's measured cost_meta.json).

  python scripts/run_ecoassistant.py --run runs/x --n 300
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from routerlab.aux_models import Embedder  # noqa: E402
from routerlab.backends import OpenAIBackend  # noqa: E402
from routerlab.collect import PREFILL_WEIGHT  # noqa: E402
from routerlab.config import load_pool  # noqa: E402
from routerlab.routers.ecoassistant import EcoAssistantLive  # noqa: E402
from routerlab.tasks import Query  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--models", default="configs/models.yaml")
ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()
pool = load_pool(a.models)
meta = json.loads((Path(a.run) / "cost_meta.json").read_text())
q = pd.read_parquet(Path(a.run) / "queries.parquet")
q = q[q.task == "gsm8k"].head(a.n)
qs = [Query(r.qid, r.task, r.kind, r.prompt, r.answer, r.subject) for r in q.itertuples()]
order = sorted(pool.names, key=lambda m: meta[m]["usd_per_weighted_token"])
strong = order[-1]


def checker(qq, out):  # stands in for user feedback (paper: GPT-4 as the user); swap in an LLM judge if you like
    try:
        return abs(float(out.replace(",", "")) - float(qq.answer)) < 1e-6
    except ValueError:
        return False


emb = Embedder(pool.aux.get("embedder"))
configs = {"A_strong_only": ([strong], 0), "B_hierarchy": (order, 0),
           "C_demos_strong": ([strong], 2), "D_ecoassistant": (order, 2)}
summary = []
for name, (chain, k) in configs.items():
    eco = EcoAssistantLive(OpenAIBackend(pool), pool, emb, chain, checker, k_demos=k or 1, sim_threshold=0.5 if k else 2.0)
    logs = [eco.solve(x) for x in qs]
    cost = sum(meta[t["model"]]["usd_per_weighted_token"] * (t["n_out"] + PREFILL_WEIGHT * t["n_in"])
               for lg in logs for t in lg["attempts"])
    sr = sum(lg["solved_by"] is not None for lg in logs) / len(logs)
    by = pd.Series([lg["solved_by"] for lg in logs]).value_counts().to_dict()
    summary.append({"config": name, "success": sr, "cost_usd": cost, "solved_by": by})
    print(summary[-1])
pd.DataFrame(summary).to_csv(Path(a.run) / "ecoassistant_live.csv", index=False)
