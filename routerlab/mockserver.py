"""A stand-in for the whole vLLM pool on ONE port, so the full live stack (gateway, Phoenix,
OTel collector, Prometheus, Grafana, control room) runs on a laptop without a GPU.

    RUN_DIR=runs/mock MODELS_YAML=configs/models.mock.yaml uvicorn routerlab.mockserver:app --port 8000

Answers come from the simulated pool used by `collect.py --mock`: each model has skills and
per-topic strengths, so correctness, logprob margins, self-verification and latency all behave
like a real small/large pool. Prompts from runs/mock/queries.parquet are recognised (so loadgen
replays get scored); unknown prompts get a synthetic question.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re

import pandas as pd
from fastapi import FastAPI, Request

from .backends import MockBackend
from .config import load_pool
from .tasks import Query

POOL = load_pool(os.environ.get("MODELS_YAML", "configs/models.mock.yaml"))
RUN = os.environ.get("RUN_DIR", "runs/mock")
SPEED = float(os.environ.get("MOCK_LATENCY_SCALE", "1.0"))   # 0 = instant, 1 = simulated latency
BY_HF = {m.hf_id: m.name for m in POOL.models} | {m.name: m.name for m in POOL.models}

qs = []
if os.path.exists(f"{RUN}/queries.parquet"):
    df = pd.read_parquet(f"{RUN}/queries.parquet")
    qs = [Query(r.qid, r.task, r.kind, r.prompt, r.answer, r.subject, json.loads(r.choices)) for r in df.itertuples()]
BY_PROMPT = {q.prompt: q.qid for q in qs}
BACK = MockBackend(POOL, qs)
app = FastAPI(title="mock vLLM pool")


def _qid_for(prompt: str) -> str:
    if prompt in BY_PROMPT:
        return BY_PROMPT[prompt]
    qid = "live-" + hashlib.sha1(prompt.encode()).hexdigest()[:12]
    if qid not in BACK.q:
        m = re.match(r"\[(\w+)\]", prompt)
        topic = m.group(1) if m and m.group(1) in BACK.TOPICS else "history"
        BACK.register([Query(qid, "live", "mcq", prompt, "ABCD"[int(qid[-1], 16) % 4], topic, ["a", "b", "c", "d"])])
        BY_PROMPT[prompt] = qid
    return qid


@app.post("/v1/chat/completions")
async def chat(req: Request):
    b = await req.json()
    model = BY_HF.get(b["model"], POOL.models[0].name)
    prompt = b["messages"][-1]["content"]
    n = int(b.get("n", 1))
    if "Reply with exactly one word: EASY or HARD" in prompt:   # LLM-as-router request
        inner = prompt.split("Question:\n", 1)[-1].split("\n\nReply with exactly")[0]
        p_small = BACK.p_correct(POOL.models[0].name, _qid_for(inner))
        noisy = p_small + (int(hashlib.sha1(inner.encode()).hexdigest()[:4], 16) / 65535 - 0.5) * 0.5
        return {"id": "mock-cls", "object": "chat.completion", "model": b["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "EASY" if noisy > 0.5 else "HARD"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 2, "total_tokens": len(prompt) // 4 + 2}}
    if "Proposed answer:" in prompt:                         # AutoMix self-verification request
        inner = prompt.split("Context and question:\n", 1)[-1].split("\n\nProposed answer:")[0]
        qid = _qid_for(inner)
        gens = BACK.generate(model, ["[VERIFY] " + prompt], 64, b.get("temperature", 0), n=n, qids=[qid])[0]
    else:
        qid = _qid_for(prompt)
        gens = BACK.generate(model, [prompt], b.get("max_tokens", 512), b.get("temperature", 0), n=n, qids=[qid])[0]
    if SPEED:
        await asyncio.sleep(min(gens[0].latency_s * SPEED, 5))
    choices = []
    for i, g in enumerate(gens):
        lp = None
        if b.get("logprobs") and g.top_logprobs:
            k = int(b.get("top_logprobs", 1))
            lp = {"content": [{"token": "X", "logprob": tops[0],
                               "top_logprobs": [{"token": f"t{j}", "logprob": v} for j, v in enumerate(tops[:k])]}
                              for tops in g.top_logprobs]}
        choices.append({"index": i, "message": {"role": "assistant", "content": g.text}, "logprobs": lp,
                        "finish_reason": "stop"})
    n_in = gens[0].n_in
    return {"id": f"mock-{qid}", "object": "chat.completion", "model": b["model"], "choices": choices,
            "usage": {"prompt_tokens": n_in, "completion_tokens": sum(g.n_out for g in gens),
                      "total_tokens": n_in + sum(g.n_out for g in gens)}}


@app.get("/health")
def health():
    return {"ok": True, "models": list(BY_HF.values())}


@app.get("/v1/models")
def models():
    return {"data": [{"id": m.hf_id} for m in POOL.models]}


