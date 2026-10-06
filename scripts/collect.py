#!/usr/bin/env python
"""Phase 1 - run every model on every benchmark query (GPU), save the outcome table.

  # real run on GPUs (loads one model at a time with vLLM offline batching):
  python scripts/collect.py --backend vllm --out runs/qwen_llama

  # against already-running `vllm serve` endpoints (scripts/serve_pool.sh):
  python scripts/collect.py --backend openai --out runs/served

  # CPU smoke test with the simulated pool:
  python scripts/collect.py --mock --out runs/mock
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routerlab.aux_models import Embedder, PairRanker, RewardModel  # noqa: E402
from routerlab.backends import MockBackend, OpenAIBackend, VLLMOfflineBackend  # noqa: E402
from routerlab.collect import collect  # noqa: E402
from routerlab.config import load_experiment, load_pool  # noqa: E402
from routerlab.mock import synthetic_queries  # noqa: E402
from routerlab.tasks import load_tasks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--exp", default="configs/experiment.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", choices=["vllm", "openai"], default="vllm")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-n", type=int, default=3000)
    a = ap.parse_args()
    pool, exp = load_pool(a.models), load_experiment(a.exp)
    out = a.out or exp["out_dir"]
    if a.mock:
        qs = synthetic_queries(a.mock_n, exp.get("seed", 0))
        collect(qs, pool, MockBackend(pool, qs), exp, out)
        return
    qs = load_tasks(exp["tasks"], exp.get("seed", 0))
    backend = VLLMOfflineBackend(pool) if a.backend == "vllm" else OpenAIBackend(pool)
    ex = exp.get("extras", {})
    rm = RewardModel(pool.aux.get("reward_model")) if ex.get("reward_scores") else None
    pr = None
    if ex.get("ensemble_fusion"):
        try:
            pr = PairRanker(pool.aux.get("pairranker", "llm-blender/PairRM"))
        except ImportError:
            print("[collect] llm-blender not installed -> fusion uses reward-ranked voting")
    collect(qs, pool, backend, exp, out, embedder=Embedder(pool.aux.get("embedder")), reward_model=rm, pair_ranker=pr)


if __name__ == "__main__":
    main()
