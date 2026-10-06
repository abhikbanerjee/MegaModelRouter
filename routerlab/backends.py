"""Inference backends.

* VLLMOfflineBackend  - loads one model at a time with `vllm.LLM`; fastest way to collect
                        outcomes for thousands of queries (continuous batching, full GPU).
* OpenAIBackend       - talks to already-running vLLM servers (OpenAI-compatible API);
                        used by the live gateway and for online/sequential methods.
* MockBackend         - a simulated pool with known ground truth so the whole pipeline
                        (all 19 routers, metrics, dashboards) runs on a laptop CPU.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
from dataclasses import dataclass, field

import numpy as np

from .config import PoolConfig


@dataclass
class Generation:
    text: str
    n_in: int
    n_out: int
    latency_s: float = 0.0
    token_logprobs: list[float] = field(default_factory=list)   # logprob of each emitted token
    top_logprobs: list[list[float]] = field(default_factory=list)  # top-k logprobs per position (sorted desc)


class Backend:
    pool: PoolConfig

    def load(self, model: str):  # offline backends load lazily
        pass

    def unload(self):
        pass

    def generate(self, model: str, prompts: list[str], max_tokens: int, temperature: float = 0.0,
                 logprobs: int = 0, n: int = 1, qids: list[str] | None = None) -> list[list[Generation]]:
        raise NotImplementedError

    def answer_nll(self, model: str, prompts: list[str], answers: list[str],
                   qids: list[str] | None = None) -> list[float]:
        """Mean negative log-likelihood of `answer` given `prompt` (teacher forcing)."""
        raise NotImplementedError


# ============================================================ vLLM offline
class VLLMOfflineBackend(Backend):
    def __init__(self, pool: PoolConfig):
        self.pool = pool
        self._llm = None
        self._name = None

    def load(self, model: str):
        if self._name == model:
            return
        self.unload()
        from vllm import LLM

        spec = self.pool.get(model)
        kw = dict(model=spec.hf_id, tensor_parallel_size=spec.num_gpus, max_model_len=spec.max_model_len,
                  gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=0)
        if spec.quantization:
            kw["quantization"] = spec.quantization
        self._llm = LLM(**kw)
        self._name = model

    def unload(self):
        if self._llm is None:
            return
        import gc

        import torch

        try:
            from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        del self._llm
        self._llm, self._name = None, None
        gc.collect()
        torch.cuda.empty_cache()

    def generate(self, model, prompts, max_tokens, temperature=0.0, logprobs=0, n=1, qids=None):
        from vllm import SamplingParams

        self.load(model)
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n,
                            logprobs=logprobs or None, seed=0 if temperature > 0 else None)
        convs = [[{"role": "user", "content": p}] for p in prompts]
        t0 = time.perf_counter()
        outs = self._llm.chat(convs, sp, use_tqdm=True)
        per_q = (time.perf_counter() - t0) / max(1, len(prompts))  # amortised batch latency
        res = []
        for o in outs:
            gens = []
            for c in o.outputs:
                tl, tops = [], []
                if c.logprobs:
                    for tid, step in zip(c.token_ids, c.logprobs):
                        tl.append(step[tid].logprob if tid in step else min(v.logprob for v in step.values()))
                        tops.append(sorted((v.logprob for v in step.values()), reverse=True))
                gens.append(Generation(c.text, len(o.prompt_token_ids), len(c.token_ids), per_q, tl, tops))
            res.append(gens)
        return res

    def answer_nll(self, model, prompts, answers, qids=None):
        from vllm import SamplingParams

        self.load(model)
        tok = self._llm.get_tokenizer()
        full, n_ans = [], []
        for p, a in zip(prompts, answers):
            head = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            ids_head = tok(head, add_special_tokens=False).input_ids
            ids_full = tok(head + a, add_special_tokens=False).input_ids
            full.append(ids_full)
            n_ans.append(max(1, len(ids_full) - len(ids_head)))
        sp = SamplingParams(max_tokens=1, prompt_logprobs=1)
        outs = self._llm.generate([{"prompt_token_ids": f} for f in full], sp, use_tqdm=True)
        nll = []
        for o, ids, k in zip(outs, full, n_ans):
            lps = [o.prompt_logprobs[i][ids[i]].logprob for i in range(len(ids) - k, len(ids))]
            nll.append(-float(np.mean(lps)))
        return nll


# ============================================================ OpenAI-compatible servers
class OpenAIBackend(Backend):
    """Calls `vllm serve` endpoints. Start them with scripts/serve_pool.sh."""

    def __init__(self, pool: PoolConfig, concurrency: int = 64, timeout: float = 300):
        self.pool, self.concurrency, self.timeout = pool, concurrency, timeout

    async def achat(self, client, model, prompt, max_tokens, temperature=0.0, logprobs=0, n=1):
        import httpx  # noqa: F401

        spec = self.pool.get(model)
        body = {"model": spec.hf_id, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": temperature, "n": n}
        if logprobs:
            body |= {"logprobs": True, "top_logprobs": logprobs}
        t0 = time.perf_counter()
        r = await client.post(f"{spec.base_url}/chat/completions", json=body, timeout=self.timeout)
        r.raise_for_status()
        js = r.json()
        lat = time.perf_counter() - t0
        u = js.get("usage", {})
        gens = []
        for ch in js["choices"]:
            tl, tops = [], []
            for t in ((ch.get("logprobs") or {}).get("content") or []):
                tl.append(t["logprob"])
                tops.append(sorted((x["logprob"] for x in t.get("top_logprobs", [])), reverse=True))
            gens.append(Generation(ch["message"]["content"] or "", u.get("prompt_tokens", 0),
                                   u.get("completion_tokens", 0) // max(1, n), lat, tl, tops))
        return gens

    async def _agenerate(self, model, prompts, **kw):
        import httpx

        sem = asyncio.Semaphore(self.concurrency)
        async with httpx.AsyncClient() as client:
            async def one(p):
                async with sem:
                    return await self.achat(client, model, p, **kw)
            return await asyncio.gather(*(one(p) for p in prompts))

    def generate(self, model, prompts, max_tokens, temperature=0.0, logprobs=0, n=1, qids=None):
        return asyncio.run(self._agenerate(model, prompts, max_tokens=max_tokens, temperature=temperature,
                                           logprobs=logprobs, n=n))

    def answer_nll(self, model, prompts, answers, qids=None):
        # The chat API cannot score a fixed continuation; use VLLMOfflineBackend for Tryage NLL,
        # or the /v1/completions endpoint with echo=True, logprobs=1 if your server allows it.
        raise NotImplementedError("Use VLLMOfflineBackend for answer_nll")


# ============================================================ Mock pool
def _u(*keys) -> float:
    h = hashlib.sha256("|".join(map(str, keys)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


class MockBackend(Backend):
    """Simulated pool. Each model has a general skill, per-topic strengths and a size.

    A query's correctness for model m ~ Bernoulli(sigmoid(4*(skill_m + affinity_m[topic] - difficulty_q))).
    Logprobs, verification verdicts, reward scores and NLL are all correlated with that
    latent probability, so every router has a real (but imperfect) signal to learn from.
    """

    TOPICS = ["algebra", "biology", "history", "law", "physics", "sentiment", "sports", "code"]

    def __init__(self, pool: PoolConfig, queries=None):
        self.pool = pool
        self.q = {q.qid: q for q in (queries or [])}
        rng = np.random.default_rng(7)
        n = len(pool.models)
        self.skill = {m.name: s for m, s in zip(pool.models, np.linspace(-0.4, 0.7, n))}
        # per-topic strengths -> complementary models, the premise of routing
        self.aff = {m.name: dict(zip(self.TOPICS, rng.normal(0, 0.7, len(self.TOPICS)))) for m in pool.models}
        self.size = {m.name: 2.0 ** i for i, m in enumerate(pool.models)}  # relative compute

    def register(self, queries):
        self.q.update({q.qid: q for q in queries})

    def p_correct(self, model, qid) -> float:
        q = self.q[qid]
        topic = q.subject if q.subject in self.TOPICS else self.TOPICS[int(_u("t", qid) * len(self.TOPICS))]
        diff = (_u("d", qid) - 0.5) * 3.0
        return 1 / (1 + math.exp(-2.5 * (self.skill[model] + self.aff[model][topic] - diff)))

    def _wrong(self, q, seed):
        if q.kind == "mcq":
            opts = [c for c in "ABCD" if c != q.answer]
            return opts[int(_u(seed) * len(opts))]
        if q.kind == "cls":
            opts = [c for c in q.choices if c != q.answer]
            return opts[int(_u(seed) * len(opts))]
        return str(int(float(q.answer or 0)) + 1 + int(_u(seed) * 9))

    def generate(self, model, prompts, max_tokens, temperature=0.0, logprobs=0, n=1, qids=None):
        assert qids is not None, "MockBackend needs qids"
        out = []
        for p, qid in zip(prompts, qids):
            gens = []
            for k in range(n):
                if p.startswith("[VERIFY]"):
                    gens.append(self._verify(model, p, qid, k, temperature))
                    continue
                q = self.q[qid]
                pc = self.p_correct(model, qid)
                draw = _u("g", model, qid, k if temperature > 0 else 0)
                ok = draw < pc
                ans = q.answer if ok else self._wrong(q, ("w", model, qid, k))
                conf = pc if ok else 1 - pc
                # top-1 prob tracks how sure the model is; noisy
                p1 = float(np.clip(0.35 + 0.6 * conf + (_u("n", model, qid) - 0.5) * 0.3, 0.26, 0.999))
                p2 = (1 - p1) * (0.6 + 0.4 * _u("p2", model, qid))
                tops = [[math.log(p1), math.log(max(p2, 1e-6)), math.log(max((1 - p1 - p2) / 2, 1e-6))]]
                n_out = 8 if q.kind in ("mcq", "cls") else int(120 + 200 * _u("len", model, qid))
                text = f"Answer: {ans}" if q.kind != "math" else f"Let me think... Answer: {ans}"
                lat = 0.02 * self.size[model] * (1 + n_out / 50)
                gens.append(Generation(text, len(p) // 4, n_out, lat, [tops[0][0]] * 1, tops))
            out.append(gens)
        return out

    def _verify(self, model, prompt, qid, k, temperature):
        q = self.q[qid]
        m = re.search(r"Proposed answer:\s*(.*)", prompt)
        cand = (m.group(1).strip() if m else "")
        truly_ok = cand.lower() == q.answer.lower() or cand == f"Answer: {q.answer}"
        acc = 0.55 + 0.35 * (self.skill[model] + 0.6) / 1.5     # better models verify better
        says_ok = truly_ok if _u("v", model, qid, k) < acc else not truly_ok
        return Generation("Verification: The answer is " + ("correct" if says_ok else "incorrect"),
                          len(prompt) // 4, 10, 0.01 * self.size[model], [], [])

    def answer_nll(self, model, prompts, answers, qids=None):
        return [float(-math.log(max(self.p_correct(model, qid), 1e-4)) + 0.3 * _u("nll", model, qid))
                for qid in qids]
