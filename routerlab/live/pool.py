"""Async client for the model pool. Every call is one OpenInference LLM span + OTel metrics."""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .. import telemetry as tel
from ..collect import PREFILL_WEIGHT
from ..config import PoolConfig

_ANS = re.compile(r"answer\s*[:：]\s*\(?([^\n\)]*)", re.I)


def extract_answer(text: str) -> str:
    """Task-agnostic answer extraction (the gateway doesn't know the task): the value after the last
    'Answer:' (letter, label or number), else the last line."""
    m = list(_ANS.finditer(text or ""))
    cand = (m[-1].group(1) if m else (text or "").strip().split("\n")[-1]).strip().rstrip(".").lower()
    num = re.fullmatch(r"-?[\d,]*\.?\d+", cand.replace("$", ""))
    return cand.replace(",", "") if num else cand


@dataclass
class Call:
    model: str
    text: str
    texts: list[str]
    n_in: int
    n_out: int
    cost: float
    latency: float
    margin_min: float | None = None
    mean_logprob: float | None = None
    raw: dict = field(default_factory=dict)

    @property
    def answer(self) -> str:
        return extract_answer(self.text)


class Pool:
    def __init__(self, pool: PoolConfig, run_dir: str | Path):
        self.cfg = pool
        self.names = pool.names
        meta_p = Path(run_dir) / "cost_meta.json"
        self.meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        self.client = httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=512))

    # $ per call from the throughput measured during collection (or fixed prices in models.yaml)
    def usd(self, model, n_in, n_out) -> float:
        spec = self.cfg.get(model)
        if spec.price_out_per_mtok is not None:
            return (n_in * (spec.price_in_per_mtok or 0) + n_out * spec.price_out_per_mtok) / 1e6
        r = self.meta.get(model, {}).get("usd_per_weighted_token", 0.0)
        return r * (n_out + PREFILL_WEIGHT * n_in)

    def prices_per_mtok(self) -> dict[str, tuple[float, float]]:
        return {m: (self.usd(m, 1_000_000, 0), self.usd(m, 0, 1_000_000)) for m in self.names}

    def cost_order(self) -> list[str]:
        return sorted(self.names, key=lambda m: self.usd(m, 500, 200))

    async def call(self, model, messages, strategy, max_tokens=512, temperature=0.0, logprobs=0, n=1,
                   span_name_suffix="") -> Call:
        spec = self.cfg.get(model)
        params = {"max_tokens": max_tokens, "temperature": temperature, "n": n, "logprobs": logprobs}
        with tel.llm_span(model, messages, params, **{"routerlab.strategy": strategy,
                                                       "routerlab.step": span_name_suffix or "answer"}) as sp:
            body = {"model": spec.hf_id, "messages": messages, "max_tokens": max_tokens,
                    "temperature": temperature, "n": n}
            if logprobs:
                body |= {"logprobs": True, "top_logprobs": logprobs}
            t0 = time.perf_counter()
            r = await self.client.post(f"{spec.base_url}/chat/completions", json=body)
            r.raise_for_status()
            js = r.json()
            dt = time.perf_counter() - t0
            u = js.get("usage", {})
            n_in, n_out = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            cost = self.usd(model, n_in, n_out)
            texts = [c["message"]["content"] or "" for c in js["choices"]]
            mm, ml = _margin(js)
            tel.record_llm_result(sp, texts[0], n_in, n_out, cost)
            if mm is not None:
                sp.set_attribute("routerlab.margin_min", mm)
            attrs = {"strategy": strategy, "model": model}
            tel.M.llm_calls.add(1, attrs)
            tel.M.llm_duration.record(dt, {"model": model})
            tel.M.tokens.add(n_in, {"model": model, "direction": "input"})
            tel.M.tokens.add(n_out, {"model": model, "direction": "output"})
            tel.M.cost.add(cost, attrs)
        return Call(model, texts[0], texts, n_in, n_out, cost, dt, mm, ml, js)


def _margin(js):
    toks = ((js["choices"][0].get("logprobs") or {}).get("content")) or []
    if not toks:
        return None, None
    ms, lps = [], []
    for t in toks:
        p = sorted((math.exp(x["logprob"]) for x in t.get("top_logprobs", [])), reverse=True) + [0.0, 0.0]
        ms.append(p[0] - p[1])
        lps.append(t["logprob"])
    return min(ms), sum(lps) / len(lps)
