#!/usr/bin/env bash
# Start one `vllm serve` per model in configs/models.yaml (OpenAI-compatible, with /metrics).
# Models share GPUs via gpu_memory_utilization fractions; put big ones on their own GPU with
# CUDA_VISIBLE_DEVICES mapping below.  Usage:  bash scripts/serve_pool.sh [configs/models.yaml]
set -euo pipefail
CFG=${1:-configs/models.yaml}
mkdir -p logs
python - "$CFG" <<'EOF' > /tmp/routerlab_serve_cmds.sh
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
gpu = 0
for m in cfg["models"]:
    tp = m.get("num_gpus", 1)
    # simple placement: small models share GPU 0; anything using >=0.85 of a GPU gets the next free one(s)
    if m.get("gpu_memory_utilization", 0.9) >= 0.85 or tp > 1:
        gpu += 1
        devs = ",".join(str(gpu + i) for i in range(tp)); gpu += tp - 1
    else:
        devs = "0"
    args = [f"--port {m['port']}", f"--tensor-parallel-size {tp}",
            f"--gpu-memory-utilization {m.get('gpu_memory_utilization', 0.9)}",
            f"--max-model-len {m.get('max_model_len', 4096)}", "--enable-prefix-caching", "--max-logprobs 20"]
    if m.get("quantization"):
        args.append(f"--quantization {m['quantization']}")
    print(f"CUDA_VISIBLE_DEVICES={devs} nohup vllm serve {m['hf_id']} {' '.join(args)} > logs/{m['name']}.log 2>&1 &")
    print("sleep 20")   # stagger startups so memory profiling doesn't collide on a shared GPU
EOF
cat /tmp/routerlab_serve_cmds.sh
echo "Single-GPU box? Edit placement above or set every model's gpu_memory_utilization so they sum to <0.95."
bash /tmp/routerlab_serve_cmds.sh
echo "Waiting for health checks..."
python - "$CFG" <<'EOF'
import sys, time, yaml, urllib.request
for m in yaml.safe_load(open(sys.argv[1]))["models"]:
    url = f"http://localhost:{m['port']}/health"
    for _ in range(180):
        try:
            urllib.request.urlopen(url, timeout=2); print("up:", m["name"]); break
        except Exception:
            time.sleep(5)
    else:
        print("NOT UP:", m["name"], "- see logs/")
EOF
