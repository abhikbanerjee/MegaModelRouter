# MegaModelRouter

[![ci](https://github.com/abhikbanerjee/MegaModelRouter/actions/workflows/ci.yml/badge.svg)](https://github.com/abhikbanerjee/MegaModelRouter/actions/workflows/ci.yml)

Reproduce, compare and **serve** 19 LLM-routing papers on open-weight models running on your own GPUs (vLLM). It uses one shared benchmark and a cost model measured on your hardware. A live OpenAI-compatible gateway serves 30 routing strategies side by side, traced into **Arize Phoenix** (per-request $ cost) and metered with **OpenTelemetry** into Grafana.

- **[docs/PAPERS.md](docs/PAPERS.md):** the 19 papers, with the gist, the benefits and how each is tested here.
- **[docs/GUIDE.md](docs/GUIDE.md):** a step-by-step setup guide, from a no-GPU demo in 10 minutes to real GPUs.

```bash
git clone https://github.com/abhikbanerjee/MegaModelRouter && cd MegaModelRouter
pip install -e ".[live,dev]"
make demo          # no GPU: Phoenix + OTel + Grafana + simulated pool + gateway + control room
```

## How it works

```
 ┌──────────────── Phase 1: collect (GPU) ─────────────────┐      ┌──── Phase 2: route (CPU, seconds) ────┐
 │ benchmarks (MMLU, ARC, HellaSwag, GSM8K, SST-2, AG News, │      │ 29 routers / variants from 19 papers   │
 │ MBPP)  ×  every model in configs/models.yaml (vLLM)     │ ───► │ fit on train, sweep knob on test       │
 │ + logprobs, self-verification, reward scores, gold NLL, │      │ → cost-accuracy curves, APGR, CPT,     │
 │   PairRM/GenFuser fusion, measured $/token per model    │      │   savings, dIBC, per-paper diagnostics │
 └─────────────────────────────────────────────────────────┘      └────────────────────────────────────────┘
                                         │
                ┌────────────── Phase 3: live (GPU) ─────────────────────────────────────────┐
                │ vLLM servers ◄── routing gateway (OpenAI API, /metrics) ◄── loadgen replay │
                │ Prometheus + Grafana + DCGM: $ saved vs always-strongest, traffic share,   │
                │ live accuracy, p50/p95 latency, escalations, GPU util/memory/power, vLLM KV │
                └────────────────────────────────────────────────────────────────────────────┘
```

Phase 1 runs every model on every query once. Because the full (query × model) outcome table is stored, every router — including cascades — is replayed offline exactly, at every threshold, for free. Cost is **measured**: during collection the bench times each model's batch and converts GPU-hours (`gpu_hourly_usd × num_gpus`) into $ per token for that model on your hardware.

## Quick start

```bash
pip install -e ".[live,gpu,dev]"     # or: pip install -r requirements.txt
huggingface-cli login                      # Llama weights are gated

# 0) CPU smoke test with the simulated pool (no GPU, no downloads, ~1 min)
python scripts/collect.py --mock --out runs/mock
python scripts/run_routers.py --run runs/mock
pytest -q tests/

# 1) real collection (edit configs/models.yaml + configs/experiment.yaml first)
python scripts/collect.py --backend vllm --out runs/v1          # one model at a time, offline batching

# 2) all routers, leaderboard and plots
python scripts/run_routers.py --run runs/v1
streamlit run scripts/dashboard.py -- --run runs/v1

# 3) live routing: every strategy behind one OpenAI-compatible gateway, traced + metered
make routers RUN=runs/v1                                         # train/pickle live routers, pick knobs
bash scripts/serve_pool.sh                                       # vllm serve per model (ports 8001-8004)
docker compose -f monitoring/docker-compose.yml --profile gpu up -d   # Phoenix, OTel collector, Prometheus, Grafana, DCGM
make gateway RUN=runs/v1                                         # :9000
make load RUN=runs/v1 STRATEGIES=single/qwen2.5-32b-awq,two-tier,automix,frugalgpt,predict/routellm-mf
make control-room RUN=runs/v1                                    # :8501
```

**No GPU?** `make demo` (Docker) or `make demo-local` (pip only) runs the entire live stack against a simulated four-model pool: Phoenix, OTel, Grafana, gateway, control room, and a first comparison run.

## Live routing, traces and metrics

```
client ──OpenAI API──► gateway :9000 ──► vLLM pool (or mock pool :8000)
                         │  model="router:<strategy>", header x-router-knob
                         └─OTLP──► OTel collector :4318 ──traces──► Arize Phoenix :6006  (LLM spans, $ per trace, correctness)
                                                        └─metrics─► Prometheus :9090 ──► Grafana :3000 + control room :8501
```

- **30 strategies** (`GET /strategies`): `single/<model>`, `random`, `task-static`, `predict/<any of 15 trained routers>`, `two-tier`, `automix`, `frugalgpt`, `consistency`, `llm-classifier`, `vote`, `blender`, `cache-distil`, `metallm-online`, and `cache+<any>` (semantic cache in front).
- **Phoenix.** Every request is one trace. A `route` span carries the strategy, knob, answering model, $ cost, $ baseline (always-strongest) and savings. It has a `router.decide` child with the decision and scores, and one OpenInference `LLM` span per model call (model, tokens, messages). At startup the gateway writes the $/token measured on your GPUs into Phoenix's model-price table, so Phoenix's cost column is your real cost. Correctness feedback from `/feedback` becomes a `correctness` span annotation.
- **OTel metrics** (Prometheus names): `routerlab_requests_total`, `routerlab_cost_usd_total`, `routerlab_baseline_cost_usd_total`, `routerlab_feedback_total{correct}`, `routerlab_escalations_total`, `routerlab_cache_hits_total`, `routerlab_llm_calls_total`, `routerlab_llm_tokens_total`, `routerlab_request_duration_seconds`, `routerlab_llm_duration_seconds`, `routerlab_router_decide_duration_seconds` and `routerlab_gpu_*`. All are labelled by `strategy` and/or `model`.
- **Grafana "LLM Routing (OTel)"** (`monitoring/grafana/build_dashboard.py`) shows cost saved, live accuracy, a per-strategy scorecard, accuracy-vs-$ scatter, latency, traffic share, escalations, cache hits, router overhead, per-model $/latency/tokens, GPU (DCGM/NVML) and vLLM queue/KV-cache/TTFT.

**GPU sizing.** The default pool (Qwen2.5-1.5B, Llama-3.1-8B, Qwen2.5-7B, Qwen2.5-32B-AWQ) fits one 80 GB GPU for offline collection (one model loaded at a time). For live serving, put the 32B model on its own GPU, or replace it. With a 2–4 GPU node, add Llama-3.3-70B as the strong model (commented out in `models.yaml`). On a 24 GB card, drop the 32B model and use Qwen2.5-14B-AWQ as the strong model.

## Papers → code

| # | Paper | Module | Registry names | Needs |
|---|---|---|---|---|
| 1 | EmbedLLM | `routers/embedllm.py` | `embedllm` | – |
| 2 | RouteLLM | `routers/routellm.py` | `routellm-mf`, `-sw_ranking`, `-clf` | (optional arena data) |
| 3 | MetaLLM | `routers/metallm.py` | `metallm-offline`, `metallm-online` | cls+mcq only |
| 4 | Merge/Ensemble/Cooperate survey | `routers/collab.py`, `scripts/merge_models.py` | `collab-vote` | – |
| 5 | AutoMix | `routers/automix.py` | `automix-pomdp`, `automix-threshold` | `self_verify` |
| 6 | OptLLM | `routers/optllm.py` | `optllm` | – |
| 7 | Uncertainty two-tier | `routers/two_tier.py` | `two-tier-marginmin` | logprobs |
| 8 | Multiple Minds | `routers/multiple_minds.py` | `minds-classifier`, `minds-cluster` | – |
| 9 | Hybrid LLM | `routers/hybrid_llm.py` | `hybrid-det`, `-prob`, `-trans` | `reward_scores` |
| 10 | Predictive Human Preference | `routers/predictive_pref.py` | `predictive-pref` | (optional arena data) |
| 11 | Routoo | `routers/routoo.py` | `routoo` | – |
| 12 | Fly-Swat or Cannon | `routers/flyswat.py` | `flyswat-ilp`, `flyswat-greedy` | – |
| 13 | Zooter (Routing to the Expert) | `routers/zooter.py` | `zooter` | `reward_scores` |
| 14 | Cache & Distil | `routers/cache_distil.py` | `cache-distil-margin`, `-qbc`, `-front` | cls only (online) |
| 15 | EcoAssistant | `routers/ecoassistant.py`, `scripts/run_ecoassistant.py` | `ecoassistant-hierarchy-oraclefb` | live run for demos |
| 16 | Routing with Benchmark Datasets | `routers/benchmark_routing.py` | `benchmark-knn` | – |
| 17 | Tryage | `routers/tryage.py` | `tryage` | `answer_nll` |
| 18 | LLM-Blender | `routers/llm_blender.py` | `llm-blender` | `reward_scores`, `ensemble_fusion` |
| 19 | FrugalGPT | `routers/frugalgpt.py` | `frugalgpt`, `frugalgpt-cache` | – |

Every module's docstring states the paper's method, what is faithful, and what is approximated.

## Metrics (all on the held-out test split)

- **APGR** (RouteLLM): the average share of the weak→strong accuracy gap recovered across the cost range. 1.0 means strong-model quality at every budget.
- **CPT(50/80)**: the fraction of the strong model's cost needed to recover 50% or 80% of the gap.
- **cost_saving_at_best_single_%**: the cost saved while matching the best single model's accuracy (with a 0.5-point tolerance).
- **gain_vs_best_single_pts**: accuracy above the best single model, which is possible when models are complementary.
- **dIBC_%** (AutoMix): the incremental benefit per cost, compared with linear weak/strong mixing.
- **Oracle**: the cheapest model that answers each query correctly. This is the ceiling for any router.

## Faithfulness notes

- **Router heads.** Router heads run on frozen sentence embeddings (`BAAI/bge-small-en-v1.5`) so that all 19 papers train in seconds. Where a paper fine-tunes a full encoder (RouteLLM-BERT, Hybrid-LLM DeBERTa, FrugalGPT DistilBERT, Fly-Swat DistilBERT, Zooter mDeBERTa), use `routerlab/finetune.py` (GPU) and plug the result in with `FinetunedHead`.
- **Hybrid LLM.** The quality score is the reward model's score rather than BARTScore. The `prob` labels use one sample per model instead of ten.
- **AutoMix POMDP.** The meta-verifier is the one-step belief version of the POMDP.
- **EcoAssistant.** The offline replay uses oracle user feedback and no demonstrations, so it is an upper bound for the hierarchy alone. `scripts/run_ecoassistant.py` runs the full method (code execution, retrieved demonstrations) live.
- **Tryage.** The router predicts the causal-LM NLL of the gold answer rather than masked-LM loss. Add domain experts (`aux.experts`) to reproduce its library-of-experts setting.
- **RouteLLM and Predictive Preference.** Both accept Chatbot Arena pairs (`load_arena_pairs`) in addition to the golden-label augmentation built from your own outcomes.
- **Mock pool.** The simulated pool only checks the plumbing. Its numbers say nothing about real models.

## Repository layout

```
routerlab/            library
  routers/            the 19 papers as offline routers (one file per paper, 29 variants)
  live/               live strategies (30) + async model-pool client
  gateway.py          OpenAI-compatible routing gateway
  telemetry.py        OpenTelemetry traces/metrics, Phoenix price sync + annotations
  collect.py          GPU outcome collection (vLLM) with measured $/token
  evaluate.py         cost-accuracy metrics (APGR, CPT, AUC, savings, dIBC)
  mockserver.py       simulated pool for no-GPU runs
scripts/              collect, run_routers, train_router, serve_pool.sh, loadgen, control_room, demo.sh, ...
monitoring/           docker compose (Phoenix+Postgres, OTel collector, Prometheus, Grafana, DCGM, vLLM)
configs/              models.yaml (pool + $/GPU-hour), models.mock.yaml, experiment.yaml
docs/                 PAPERS.md, GUIDE.md
tests/                offline routers + every live strategy against the mock pool (CPU)
```
