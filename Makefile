# Shortcuts. `make help` lists them.  RUN=runs/v1 make routers   overrides the run directory.
RUN ?= runs/v1
PY  ?= python

.PHONY: help install test demo demo-local demo-down collect bench routers serve stack gateway load control-room grafana-json

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n",$$1,$$2}'

install:        ## pip install everything (GPU box: vllm included)
	pip install -r requirements.txt

test:           ## CPU smoke test of all routers on the simulated pool
	$(PY) -m pytest -q tests/

demo:           ## no-GPU demo with docker: Phoenix, OTel, Prometheus, Grafana, mock pool, gateway, control room
	bash scripts/demo.sh up

demo-local:     ## no-GPU, no-docker demo (Phoenix via pip, traces only)
	bash scripts/demo.sh up-local

demo-down:      ## stop the demo
	bash scripts/demo.sh down

collect:        ## GPU: run every model on every benchmark question (vLLM offline batching)
	$(PY) scripts/collect.py --backend vllm --out $(RUN)

bench:          ## offline: all 29 router variants -> leaderboard + cost/accuracy plots
	$(PY) scripts/run_routers.py --run $(RUN)

routers:        ## train + pickle every live router, pick default knobs (50% of strongest cost)
	$(PY) scripts/train_router.py --run $(RUN) --all

serve:          ## GPU: start one vLLM server per model in configs/models.yaml
	bash scripts/serve_pool.sh configs/models.yaml

stack:          ## observability stack (+DCGM on a GPU host)
	docker compose -f monitoring/docker-compose.yml --profile gpu up -d

gateway:        ## routing gateway on :9000 (real pool)
	RUN_DIR=$(RUN) MODELS_YAML=configs/models.yaml $(PY) -m uvicorn routerlab.gateway:app --host 0.0.0.0 --port 9000

load:           ## compare strategies live: STRATEGIES=a,b,c N=200
	$(PY) scripts/loadgen.py --run $(RUN) --n $${N:-200} --strategies $${STRATEGIES:-all}

control-room:   ## Streamlit control room on :8501
	$(PY) -m streamlit run scripts/control_room.py -- --run $(RUN)

grafana-json:   ## regenerate the Grafana dashboard JSON
	$(PY) monitoring/grafana/build_dashboard.py
