"""Routing gateway: OpenAI-compatible endpoint that serves every routing strategy side by side,
traced into Arize Phoenix and metered with OpenTelemetry.

    uvicorn routerlab.gateway:app --host 0.0.0.0 --port 9000

Environment
    RUN_DIR                      collected run with trained routers (runs/<name>, routers/*.pkl)
    MODELS_YAML                  pool config (configs/models.yaml, or configs/models.mock.yaml for the demo)
    EMBEDDER                     sentence-transformer id; "" = hashing features (mock runs)
    DEFAULT_STRATEGY             used for model="router" (default two-tier)
    OTEL_EXPORTER_OTLP_ENDPOINT  OTel collector (default http://localhost:4318)
    PHOENIX_URL                  Phoenix base URL for price sync + feedback annotations (http://localhost:6006)

Calls
    POST /v1/chat/completions  model="router:<strategy>" | "router" | "<pool model>"; header x-router-knob
    POST /feedback             {"request_id": ..., "correct": true}  -> metric + Phoenix annotation + online learners
    GET  /strategies           every strategy, its knob meaning and default
"""
from __future__ import annotations

import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import telemetry as tel
from .aux_models import Embedder
from .config import load_pool
from .live.pool import Pool
from .live.strategies import Ctx, Req, build, describe, resolve

RUN = os.environ.get("RUN_DIR", "runs/default")
POOLCFG = load_pool(os.environ.get("MODELS_YAML", "configs/models.yaml"))
DEFAULT = os.environ.get("DEFAULT_STRATEGY", "two-tier")
PHOENIX = os.environ.get("PHOENIX_URL", "http://localhost:6006")

tel.init("routerlab-gateway")
POOL = Pool(POOLCFG, RUN)
emb_id = os.environ.get("EMBEDDER", POOLCFG.aux.get("embedder"))
CTX = Ctx(POOL, RUN, Embedder(emb_id or None))
STRATS = build(CTX)
ANNOT = tel.PhoenixAnnotator(PHOENIX)
RECORDS: dict[str, dict] = {}
app = FastAPI(title="routerlab gateway")
print(f"[gateway] strategies: {', '.join(STRATS)}")


@app.on_event("startup")
async def _sync_prices():
    if PHOENIX and POOL.meta:
        tel.sync_phoenix_prices(PHOENIX, POOL.prices_per_mtok())


def _strategy_name(model: str | None) -> str:
    if model in POOL.names:
        return f"single/{model}"
    if model and model.startswith("router:"):
        return model.split(":", 1)[1]
    return DEFAULT


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    messages = body["messages"]
    name = _strategy_name(body.get("model"))
    try:
        strat = resolve(STRATS, CTX, name)
    except KeyError:
        return JSONResponse({"error": f"unknown strategy '{name}'", "available": list(STRATS)}, 404)
    knob = req.headers.get("x-router-knob")
    r = Req(messages, messages[-1]["content"], float(knob) if knob not in (None, "") else None,
            int(body.get("max_tokens", 512)), uuid.uuid4().hex)
    t0 = time.perf_counter()
    attrs = {"input.value": r.prompt, "routerlab.strategy": strat.name, "routerlab.request_id": r.request_id,
             "routerlab.knob": strat.knob(r)}
    if req.headers.get("x-session-id"):
        attrs["session.id"] = req.headers["x-session-id"]
    with tel.chain_span("route", **attrs) as root:
        try:
            out = await strat.run(r)
        except Exception as e:
            root.record_exception(e)
            root.set_attribute("otel.status_code", "ERROR")
            return JSONResponse({"error": str(e), "strategy": strat.name}, 502)
        dt = time.perf_counter() - t0
        cost = sum(c.cost for c in out.calls)
        ans = out.call
        n_in = ans.n_in or max(1, len(r.prompt) // 4)
        n_out = (ans.n_out / max(1, len(ans.texts))) if ans.n_out else 32   # per completion (n>1 samples)
        baseline = POOL.usd(CTX.strong, n_in, n_out)
        text = out.text or ans.text
        meta = {"strategy": strat.name, "answered_by": ans.model, "calls": [c.model for c in out.calls],
                "escalated": out.escalated, "cache_hit": out.cache_hit, "decision": out.decision}
        tel._set(root, {"output.value": text, "routerlab.answered_by": ans.model, "routerlab.cost_usd": cost,
                        "routerlab.baseline_cost_usd": baseline, "routerlab.savings_usd": baseline - cost,
                        "routerlab.escalated": out.escalated, "routerlab.cache_hit": out.cache_hit,
                        "routerlab.n_calls": len(out.calls), "routerlab.decision": out.decision,
                        "metadata": meta, "tag.tags": [strat.name, ans.model]})
        span_id, trace_id = tel.span_id_hex(root), tel.trace_id_hex(root)
    tel.M.requests.add(1, {"strategy": strat.name, "model": ans.model})
    tel.M.request_duration.record(dt, {"strategy": strat.name})
    tel.M.baseline.add(baseline, {"strategy": strat.name})
    RECORDS[r.request_id] = {"strategy": strat.name, "model": ans.model, "span_id": span_id}
    if len(RECORDS) > 200_000:
        RECORDS.pop(next(iter(RECORDS)))
    resp = dict(ans.raw) if ans.raw else {"object": "chat.completion", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
    if out.text:
        resp["choices"] = [{"index": 0, "message": {"role": "assistant", "content": out.text}, "finish_reason": "stop"}]
    resp["model"] = f"router:{strat.name}"
    resp["routerlab"] = {"request_id": r.request_id, "strategy": strat.name, "answered_by": ans.model,
                         "calls": [c.model for c in out.calls], "cost_usd": cost, "baseline_cost_usd": baseline,
                         "escalated": out.escalated, "cache_hit": out.cache_hit, "latency_s": dt,
                         "trace_id": trace_id, "span_id": span_id, "decision": out.decision}
    return resp


@app.post("/feedback")
async def feedback(req: Request):
    b = await req.json()
    rec = RECORDS.get(b["request_id"])
    if rec is None:
        return JSONResponse({"error": "unknown request_id"}, 404)
    ok = bool(b["correct"])
    tel.M.feedback.add(1, {"strategy": rec["strategy"], "model": rec["model"], "correct": str(ok).lower()})
    ANNOT.add(rec["span_id"], ok, rec["strategy"], rec["model"])
    try:
        resolve(STRATS, CTX, rec["strategy"]).feedback(b["request_id"], ok)
    except KeyError:
        pass
    return {"ok": True}


@app.get("/strategies")
def strategies():
    return {"default": DEFAULT, "strongest_model": CTX.strong, "strategies": describe(STRATS)}


@app.get("/health")
def health():
    return {"ok": True, "run": RUN, "models": POOL.names, "strategies": len(STRATS)}


@app.get("/")
def root():
    return {"see": ["/strategies", "/docs"], "phoenix": PHOENIX, "metrics": "via OTLP -> collector"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 9000)))
