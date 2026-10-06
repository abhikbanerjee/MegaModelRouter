#!/usr/bin/env bash
# One-command demo of the live routing stack WITHOUT a GPU (simulated model pool).
#
#   bash scripts/demo.sh up        # docker: Phoenix + OTel collector + Prometheus + Grafana, then mock pool,
#                                  #         gateway, control room, and a first comparison run
#   bash scripts/demo.sh up-local  # no docker: Phoenix via pip; traces only (no Grafana metrics)
#   bash scripts/demo.sh load      # send another comparison batch
#   bash scripts/demo.sh down      # stop everything this script started
#
# For real GPUs, follow the guide: collect.py --backend vllm, train_router.py --all, serve_pool.sh, gateway.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-runs/mock}
PIDS=.demo_pids
mkdir -p logs

start() {  # name, command...
  local name=$1; shift
  nohup "$@" > "logs/$name.log" 2>&1 &
  echo "$! $name" >> "$PIDS"
  echo "  started $name (pid $!, log logs/$name.log)"
}

wait_http() {  # url, name
  for _ in $(seq 1 90); do curl -fsS "$1" >/dev/null 2>&1 && { echo "  $2 is up"; return 0; }; sleep 2; done
  echo "  $2 did not come up - check logs/"; return 1
}

prepare_data() {
  if [ ! -f "$RUN/outcomes.parquet" ]; then
    echo "[1/5] simulated collection -> $RUN"
    python scripts/collect.py --mock --out "$RUN" > logs/collect.log
  fi
  if [ ! -f "$RUN/routers/knobs.json" ]; then
    echo "[2/5] training every live router"
    python scripts/train_router.py --run "$RUN" --all | tail -22
  fi
}

serve_app() {  # otel endpoint, metrics on/off
  echo "[4/5] mock model pool :8000, gateway :9000, control room :8501"
  start mockpool env RUN_DIR="$RUN" MODELS_YAML=configs/models.mock.yaml MOCK_LATENCY_SCALE=0.3 \
    python -m uvicorn routerlab.mockserver:app --port 8000
  wait_http http://localhost:8000/health "mock pool"
  start gateway env RUN_DIR="$RUN" MODELS_YAML=configs/models.mock.yaml EMBEDDER="" \
    OTEL_EXPORTER_OTLP_ENDPOINT="$1" ROUTERLAB_METRICS="$2" PHOENIX_URL=http://localhost:6006 \
    python -m uvicorn routerlab.gateway:app --host 0.0.0.0 --port 9000
  wait_http http://localhost:9000/health gateway
  start control-room python -m streamlit run scripts/control_room.py --server.headless true --server.port 8501 -- --run "$RUN"
}

first_load() {
  echo "[5/5] first comparison: 100 questions x 8 strategies"
  python scripts/loadgen.py --run "$RUN" --n 100 --qps 20 \
    --strategies single/qwen2.5-32b-awq,single/qwen2.5-1.5b,random,two-tier,automix,frugalgpt,predict/routellm-mf,predict/embedllm
}

case "${1:-up}" in
  up)
    prepare_data
    echo "[3/5] observability stack (docker compose)"
    docker compose -f monitoring/docker-compose.yml up -d phoenix-db phoenix otel-collector prometheus grafana
    wait_http http://localhost:6006/healthz Phoenix
    serve_app http://localhost:4318 1
    first_load
    cat <<EOF

  Phoenix (traces + cost)      http://localhost:6006   project "llm-routing"
  Grafana (OTel metrics)       http://localhost:3000   dashboard "LLM Routing (OTel)"
  Control room                 http://localhost:8501
  Prometheus                   http://localhost:9090
  Gateway (OpenAI API)         http://localhost:9000/v1   model="router:<strategy>"  (list: /strategies)
EOF
    ;;
  up-local)
    prepare_data
    echo "[3/5] Phoenix (pip, no docker)"
    command -v phoenix >/dev/null || pip install -q arize-phoenix
    start phoenix phoenix serve
    wait_http http://localhost:6006/healthz Phoenix
    serve_app http://localhost:6006 0      # traces straight to Phoenix; metrics off (no collector)
    first_load
    echo; echo "  Phoenix http://localhost:6006 · Control room http://localhost:8501 (OTel-metrics tab needs the docker stack)"
    ;;
  load)
    python scripts/loadgen.py --run "$RUN" --n "${N:-200}" --qps "${QPS:-20}" --strategies "${STRATEGIES:-all}"
    ;;
  down)
    if [ -f "$PIDS" ]; then
      while read -r pid name; do kill "$pid" 2>/dev/null && echo "  stopped $name"; done < "$PIDS"
      rm -f "$PIDS"
    fi
    docker compose -f monitoring/docker-compose.yml stop 2>/dev/null || true
    ;;
  *) echo "usage: $0 up|up-local|load|down"; exit 1 ;;
esac
