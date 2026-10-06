# Step-by-step guide: live routing with Arize Phoenix + OpenTelemetry

The gateway puts 30 routing strategies behind one OpenAI-compatible API. Every request is traced into Arize Phoenix with its real GPU cost, and OpenTelemetry metrics feed a Grafana dashboard and a Streamlit control room, so you can compare cost, quality and latency per strategy side by side.

## How it fits together

```
 clients (your app / loadgen / control room)
        │  OpenAI API, model="router:<strategy>", header x-router-knob
        ▼
 routing gateway :9000 ──chat completions──► model pool: vLLM :8001-8004  (or mock pool :8000, no GPU)
        │ OTLP :4318                                        │ /metrics (scraped)
        ▼                                                   ▼
 OTel collector ──traces──► Arize Phoenix :6006      Prometheus :9090 ──► Grafana :3000 + control room :8501
                └─metrics──────────────────────────────────►┘
```

The gateway also writes your GPU-measured $/token into Phoenix's model price table and posts correctness feedback as Phoenix span annotations.

## Part A — try it in 10 minutes, no GPU

The demo runs the whole live stack against a simulated four-model pool with different skills and per-topic strengths.

1. Install Python 3.10+ and Docker (Compose v2), then:
   ```bash
   git clone https://github.com/abhikbanerjee/MegaModelRouter && cd MegaModelRouter
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[live,dev]"
   ```
2. `make demo` — simulates collection, trains every live router, starts Phoenix (+Postgres), the OTel collector, Prometheus and Grafana in Docker, then the mock pool (:8000), gateway (:9000) and control room (:8501), and sends 100 questions through 8 strategies.
3. Open Phoenix http://localhost:6006 (project **llm-routing**), Grafana http://localhost:3000 (dashboard **LLM Routing (OTel)**), control room http://localhost:8501.
4. More traffic: `STRATEGIES=all N=200 bash scripts/demo.sh load`.
5. Stop: `make demo-down`.

No Docker? `pip install -e ".[phoenix]"` then `make demo-local` (traces to Phoenix only; Grafana metrics need the collector).

## Part B — real GPUs

1. `pip install -e ".[live,gpu,dev]"` and `huggingface-cli login` (Llama weights are gated).
2. Edit `configs/models.yaml` (pool, ports, GPU memory shares, `gpu_hourly_usd`) and `configs/experiment.yaml` (benchmarks, sample sizes).
3. `make collect RUN=runs/v1` — every model answers every question (vLLM offline batching) plus logprobs, self-verification, reward scores, gold NLL; $/token measured from wall-clock time.
4. `make bench RUN=runs/v1` — leaderboard + cost-accuracy plots for the 29 offline router variants.
5. `make routers RUN=runs/v1` — pickles every live router; default knob = ~50% of the strongest model's cost (`--target-cost-frac`).
6. `make serve` — one `vllm serve` per model on 8001–8004 (or `docker compose -f monitoring/docker-compose.yml --profile vllm up -d`).
7. `make stack` — Phoenix, OTel collector, Prometheus, Grafana, DCGM GPU exporter.
8. `make gateway RUN=runs/v1` — syncs prices into Phoenix, serves :9000.
9. `make load RUN=runs/v1 STRATEGIES=single/qwen2.5-32b-awq,two-tier,automix,frugalgpt,predict/routellm-mf N=300`.
10. `make control-room RUN=runs/v1`, plus Grafana and Phoenix.

## Strategy catalogue

| Strategy | Pattern | From | Knob | LLM calls / request |
| --- | --- | --- | --- | --- |
| `single/<model>` | Always one model | baseline | — | 1 |
| `random` | Random weak/strong mix | baseline | P(strongest) | 1 |
| `task-static` | Detect task, use that task's best model | common practice | cost weight λ | 1 |
| `predict/<router>` | Predict-then-select: `embedllm`, `routellm-mf`, `routellm-sw_ranking`, `routellm-clf`, `metallm-offline`, `minds-classifier`, `minds-cluster`, `hybrid-trans`, `hybrid-prob`, `predictive-pref`, `routoo`, `flyswat-greedy`, `zooter`, `benchmark-knn`, `tryage` | 11 papers | router-specific | 1 |
| `two-tier` | Cascade on logprob margin | Two-tier uncertainty | margin threshold | 1–2 |
| `automix` | Cascade on self-verification + meta-verifier | AutoMix | belief threshold | 2–3 |
| `frugalgpt` | Learned cascade with answer scorer | FrugalGPT | budget 0–1 | 1–3 |
| `consistency` | Cascade on agreement of k samples | self-consistency cascades | agreement share | 1–2 |
| `llm-classifier` | Small LLM labels EASY/HARD, then routes | LLM-as-router | — | 2 |
| `vote` | Parallel ensemble, weighted majority | survey | k models | k |
| `blender` | Parallel ensemble, PairRM/confidence pick | LLM-Blender | — | all |
| `cache-distil` | Online student answers when confident | Cache & Distil | student confidence | 0–1 |
| `metallm-online` | LinUCB bandit learning from `/feedback` | MetaLLM | cost weight | 1 |
| `cache+<any>` | Semantic completion cache in front | FrugalGPT | inner knob | 0 on hit |

EcoAssistant needs code execution with a success check, so it runs as `scripts/run_ecoassistant.py`.

## Reading the results

- **Phoenix:** one trace per request. The root `route` span holds strategy, knob, answering model, `routerlab.cost_usd`, `routerlab.baseline_cost_usd`, `routerlab.savings_usd` and the decision in `metadata`; child LLM spans hold messages, tokens and latency. Cost comes from the synced prices (*Settings → Models*, `routerlab/<model>`). Filter `attributes['routerlab.strategy'] == 'two-tier'`; the `correctness` annotation finds cheap-and-wrong answers.
- **Grafana "LLM Routing (OTel)":** cost saved, live accuracy, spend; per-strategy scorecard and accuracy-vs-$ scatter; latency; traffic share, escalations, cache hits, router overhead; per-model $/latency/tokens; GPU (DCGM/NVML) and vLLM queue/KV cache/TTFT. Regenerate with `make grafana-json`.
- **Control room:** try a request (decision, cost, Phoenix trace link, mark right/wrong), compare strategies live, OTel scorecard from Prometheus, recent Phoenix traces.

## Calling the gateway from your app

```python
from openai import OpenAI
import httpx

client = OpenAI(base_url="http://localhost:9000/v1", api_key="unused")
r = client.chat.completions.create(
    model="router:predict/routellm-mf",          # or router:two-tier, router:cache+frugalgpt, ...
    messages=[{"role": "user", "content": "Explain KV caching in two sentences."}],
    extra_headers={"x-router-knob": "0.4", "x-session-id": "demo-user-1"},
)
info = r.model_extra["routerlab"]   # answered_by, cost_usd, baseline_cost_usd, trace_id, decision
httpx.post("http://localhost:9000/feedback", json={"request_id": info["request_id"], "correct": True})
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Grafana panels empty | Send traffic (`scripts/loadgen.py`); check the collector logs and that `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318` |
| Phoenix fills slowly under load | Phoenix returns 503 when busy; the collector buffers and retries. Lower `--qps`; the compose stack uses Postgres |
| Phoenix cost column empty | Restart the gateway with `PHOENIX_URL` set; check *Settings → Models* for `routerlab/<model>` |
| Collector: "address already in use :4317" | A pip-installed Phoenix holds 4317; use the Docker stack or stop it |
| `unknown strategy` | `make routers`, then `curl :9000/strategies`; AutoMix needs `extras.self_verify`, Tryage `extras.answer_nll` |
| vLLM / GPU panels empty | Prometheus reaches host vLLM via `host.docker.internal`; start the stack with `--profile gpu` or install `nvidia-ml-py` |
| Live accuracy flat | Send feedback (`loadgen.py` does) |

Tested end to end on CPU with the mock pool, Phoenix 20.19, OTel collector 0.115 and Prometheus 2.54; every dashboard query was checked against live Prometheus data. Grafana rendering, DCGM and real vLLM servers were not available in that environment.
