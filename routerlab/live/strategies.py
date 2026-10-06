"""Live routing strategies - every routing *pattern* from the 19 papers, servable side by side.

Request a strategy with  model="router:<name>"  (OpenAI API), e.g.
    router:single/qwen2.5-32b-awq      baseline: always one model
    router:random                      random weak/strong mix (the line routers must beat)
    router:task-static                 classify the prompt's task, use that task's best model
    router:predict/<router>            ANY pre-generation router trained offline (embedllm, routellm-mf,
                                       routoo, zooter, hybrid-trans, flyswat-greedy, tryage, metallm-offline, ...)
    router:two-tier                    small model + logprob-margin escalation          (Ramirez et al.)
    router:automix                     small model + self-verification + meta-verifier (AutoMix)
    router:frugalgpt                   learned cascade with answer scorer               (FrugalGPT)
    router:consistency                 small model k samples; escalate on disagreement  (self-consistency cascade)
    router:llm-classifier              small LLM labels EASY/HARD first, then routes    (LLM-as-router)
    router:vote                        parallel ensemble, weighted majority             (survey: ensemble)
    router:blender                     parallel ensemble, pick best by PairRM/confidence (LLM-Blender)
    router:cache-distil                online student learns from the teacher           (Cache & Distil)
    router:metallm-online              LinUCB bandit learning from /feedback            (MetaLLM)
    router:cache+<any of the above>    semantic completion cache in front               (FrugalGPT approximation)
Header `x-router-knob` sets the cost/quality knob per request; defaults come from
runs/<run>/routers/knobs.json (written by scripts/train_router.py --all).
"""
from __future__ import annotations

import asyncio
import json
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import telemetry as tel
from ..collect import VERIFY_TMPL
from .pool import Call, Pool, extract_answer


@dataclass
class Req:
    messages: list
    prompt: str
    knob: float | None
    max_tokens: int
    request_id: str
    emb: np.ndarray | None = None


@dataclass
class Out:
    call: Call                              # the call whose text is returned
    calls: list = field(default_factory=list)
    escalated: bool = False
    cache_hit: bool = False
    text: str | None = None                 # override text (vote/student/cache)
    decision: dict = field(default_factory=dict)


class Ctx:
    """Shared state: pool, embedder, trained routers, offline stats."""

    def __init__(self, pool: Pool, run_dir: str | Path, embedder):
        self.pool, self.run = pool, Path(run_dir)
        self.emb = embedder
        self.order = pool.cost_order()            # cheap -> expensive
        self.routers = {}
        rdir = self.run / "routers"
        for p in sorted(rdir.glob("*.pkl")) if rdir.exists() else []:
            try:
                self.routers[p.stem] = pickle.loads(p.read_bytes())
            except Exception as e:
                print(f"[strategies] could not load {p.name}: {e}")
        self.knobs = json.loads((rdir / "knobs.json").read_text()) if (rdir / "knobs.json").exists() else {}
        self.acc = self._offline_acc()
        self.strong = max(self.acc, key=self.acc.get) if self.acc else self.order[-1]

    def _offline_acc(self):
        p = self.run / "outcomes.parquet"
        if not p.exists():
            return {}
        import pandas as pd

        o = pd.read_parquet(p, columns=["model", "correct"])
        return o.groupby("model").correct.mean().to_dict()

    def embed(self, req: Req):
        if req.emb is None:
            req.emb = self.emb.encode([req.prompt])[0]
        return req.emb

    def idx(self, model):
        return self.pool.names.index(model)


class Strategy:
    name = "base"
    knob_help = ""
    default_knob = 0.5

    def __init__(self, ctx: Ctx):
        self.ctx = ctx

    def knob(self, req):
        return req.knob if req.knob is not None else self.ctx.knobs.get(self.name, self.default_knob)

    async def run(self, req: Req) -> Out:
        raise NotImplementedError

    def feedback(self, request_id: str, correct: bool):
        pass

    async def _call(self, model, req, **kw):
        return await self.ctx.pool.call(model, req.messages, self.name, max_tokens=req.max_tokens, **kw)


# --------------------------------------------------------------------------- baselines
class Single(Strategy):
    knob_help = "unused"

    def __init__(self, ctx, model):
        super().__init__(ctx)
        self.model, self.name = model, f"single/{model}"

    async def run(self, req):
        c = await self._call(self.model, req)
        return Out(c, [c])


class RandomMix(Strategy):
    name = "random"
    knob_help = "probability of sending to the strongest model"

    async def run(self, req):
        m = self.ctx.strong if random.random() < self.knob(req) else self.ctx.order[0]
        c = await self._call(m, req)
        return Out(c, [c], decision={"p_strong": self.knob(req)})


class TaskStatic(Strategy):
    """Rule-based routing most teams start with: detect the task type, use a fixed model per task.
    Task = nearest task centroid in embedding space; per-task model = argmax acc - knob * cost."""
    name = "task-static"
    knob_help = "cost weight lambda (0 = always the most accurate model for the task)"
    default_knob = 0.3

    def __init__(self, ctx):
        super().__init__(ctx)
        import pandas as pd

        q = pd.read_parquet(ctx.run / "queries.parquet")
        X = np.load(ctx.run / "embeddings.npy")
        o = pd.read_parquet(ctx.run / "outcomes.parquet").merge(q[["qid", "task", "split"]], on="qid")
        o = o[o.split == "train"]
        self.tasks = sorted(q.task.unique())
        self.cent = np.stack([X[(q.task == t).values].mean(0) for t in self.tasks])
        self.cent /= np.linalg.norm(self.cent, axis=1, keepdims=True) + 1e-9
        g = o.groupby(["task", "model"]).agg(acc=("correct", "mean"), cost=("cost_usd", "mean")).reset_index()
        self.table = {t: g[g.task == t].set_index("model") for t in self.tasks}

    async def run(self, req):
        e = self.ctx.embed(req)
        t = self.tasks[int(np.argmax(self.cent @ (e / (np.linalg.norm(e) + 1e-9))))]
        tb = self.table[t]
        util = tb.acc - self.knob(req) * tb.cost / tb.cost.max()
        m = util.idxmax()
        c = await self._call(m, req)
        return Out(c, [c], decision={"task": t, "utility": util.round(3).to_dict()})


# --------------------------------------------------------------------------- pre-generation routers
class Predict(Strategy):
    knob_help = "router-specific (see knobs.json); same knob as the offline sweep"

    def __init__(self, ctx, router_name):
        super().__init__(ctx)
        self.r = ctx.routers[router_name]
        if not getattr(self.r, "supports_live", False):
            raise ValueError(f"{router_name} cannot route before generation")
        self.name = f"predict/{router_name}"

    async def run(self, req):
        e = self.ctx.embed(req)
        with tel.chain_span("router.decide", **{"routerlab.router": self.name}) as sp:
            import time

            t0 = time.perf_counter()
            m = int(self.r.route_one(e, self.knob(req)))
            tel.M.decide_duration.record(time.perf_counter() - t0, {"strategy": self.name})
            dec = {"model": self.ctx.pool.names[m]}
            if hasattr(self.r, "predict"):
                try:
                    p = np.asarray(self.r.predict(e[None]))[0]
                    dec["predicted_success"] = {n: round(float(v), 3) for n, v in zip(self.ctx.pool.names, p)}
                except Exception:
                    pass
            sp.set_attribute("output.value", json.dumps(dec))
        c = await self._call(self.ctx.pool.names[m], req)
        return Out(c, [c], decision=dec)


# --------------------------------------------------------------------------- cascades
class TwoTier(Strategy):
    name = "two-tier"
    knob_help = "escalate when the small model's top1-top2 token-probability margin < knob"
    default_knob = 0.5

    def __init__(self, ctx):
        super().__init__(ctx)
        r = ctx.routers.get("two-tier-marginmin")
        self.small = ctx.pool.names[r.S] if r else ctx.order[0]
        self.large = ctx.pool.names[r.L] if r else ctx.strong

    async def run(self, req):
        c1 = await self._call(self.small, req, logprobs=5)
        margin = c1.margin_min if c1.margin_min is not None else 1.0
        dec = {"margin": round(margin, 4), "threshold": self.knob(req)}
        if margin >= self.knob(req):
            return Out(c1, [c1], decision=dec)
        tel.M.escalations.add(1, {"strategy": self.name, "from_model": self.small})
        c2 = await self._call(self.large, req)
        return Out(c2, [c1, c2], escalated=True, decision=dec)


class AutoMix(Strategy):
    name = "automix"
    knob_help = "escalate when (calibrated) self-verification confidence < knob"

    def __init__(self, ctx, k=8):
        super().__init__(ctx)
        self.r = ctx.routers.get("automix-pomdp")
        self.k = k
        if self.r:
            self.chain = [ctx.pool.names[i] for i in self.r.chain]
        else:
            self.chain = [ctx.order[0], ctx.strong]

    def _belief(self, model, v):
        if not self.r or self.ctx.idx(model) not in self.r.belief:
            return v
        b = int(np.clip(np.digitize(v, self.r.edges) - 1, 0, self.r.bins - 1))
        return float(self.r.belief[self.ctx.idx(model)][b])

    async def run(self, req):
        calls, dec = [], {}
        for m in self.chain[:-1]:
            c = await self._call(m, req)
            calls.append(c)
            vmsg = [{"role": "user", "content": VERIFY_TMPL.format(prompt=req.prompt, answer=c.text[-300:])
                     .replace("[VERIFY] ", "")}]
            with tel.chain_span("self-verify", **{"routerlab.model": m}) as sp:
                v = await self.ctx.pool.call(m, vmsg, self.name, max_tokens=64, temperature=0.7, n=self.k,
                                             span_name_suffix="verify")
                conf = float(np.mean(["is correct" in t.lower() for t in v.texts]))
                belief = self._belief(m, conf)
                sp.set_attribute("output.value", json.dumps({"raw_confidence": conf, "belief": belief}))
            calls.append(v)
            dec[m] = {"verifier_confidence": conf, "belief": round(belief, 3)}
            if belief >= self.knob(req):
                return Out(c, calls, decision=dec)
            tel.M.escalations.add(1, {"strategy": self.name, "from_model": m})
        c = await self._call(self.chain[-1], req)
        calls.append(c)
        return Out(c, calls, escalated=True, decision=dec)


class FrugalGPT(Strategy):
    name = "frugalgpt"
    knob_help = "budget position 0..1 between the cheapest and priciest learned cascade"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.r = ctx.routers.get("frugalgpt")
        if self.r is None:
            raise ValueError("train it first: scripts/train_router.py --router frugalgpt")

    async def run(self, req):
        e = self.ctx.embed(req)
        chain, thr = self.r.live_plan(self.knob(req))
        calls, dec = [], {"chain": [self.ctx.pool.names[m] for m in chain]}
        for j, m in enumerate(chain):
            c = await self._call(self.ctx.pool.names[m], req, logprobs=5)
            calls.append(c)
            if j == len(chain) - 1:
                break
            s = self.r.live_score(m, e, c.answer, c.margin_min, c.mean_logprob, c.n_out)
            dec[c.model] = {"score": round(s, 3), "threshold": round(float(thr[j]), 3)}
            if s >= thr[j]:
                return Out(c, calls, decision=dec)
            tel.M.escalations.add(1, {"strategy": self.name, "from_model": c.model})
        return Out(calls[-1], calls, escalated=len(calls) > 1, decision=dec)


class Consistency(Strategy):
    """Self-consistency cascade (answer-agreement as confidence, cf. 'LLM cascades with mixture of
    thought representations'): sample the small model k times; accept if the majority share >= knob."""
    name = "consistency"
    knob_help = "accept the small model when >= knob of its k samples agree"
    default_knob = 0.7

    def __init__(self, ctx, k=5):
        super().__init__(ctx)
        self.k, self.small, self.large = k, ctx.order[0], ctx.strong

    async def run(self, req):
        c = await self._call(self.small, req, temperature=0.8, n=self.k)
        answers = [extract_answer(t) for t in c.texts]
        top, cnt = max(((a, answers.count(a)) for a in set(answers)), key=lambda x: x[1])
        share = cnt / len(answers)
        dec = {"samples": answers, "agreement": round(share, 3)}
        if share >= self.knob(req):
            text = next(t for t in c.texts if extract_answer(t) == top)
            return Out(c, [c], text=text, decision=dec)
        tel.M.escalations.add(1, {"strategy": self.name, "from_model": self.small})
        c2 = await self._call(self.large, req)
        return Out(c2, [c, c2], escalated=True, decision=dec)


class LLMJudgeRouter(Strategy):
    """LLM-as-router: the smallest model rates the question's difficulty first (a few tokens),
    then EASY -> small model, HARD -> strongest model. A common production pattern; no training."""
    name = "llm-classifier"
    knob_help = "unused (EASY/HARD from the classifier prompt)"
    PROMPT = ("You are a router. Decide if a small 1-2B language model can answer the question below correctly.\n"
              "Question:\n{q}\n\nReply with exactly one word: EASY or HARD.")

    async def run(self, req):
        msg = [{"role": "user", "content": self.PROMPT.format(q=req.prompt)}]
        with tel.chain_span("router.decide", **{"routerlab.router": "llm-classifier"}) as sp:
            j = await self.ctx.pool.call(self.ctx.order[0], msg, self.name, max_tokens=4, span_name_suffix="classify")
            hard = "easy" not in j.text.lower()
            sp.set_attribute("output.value", "HARD" if hard else "EASY")
        m = self.ctx.strong if hard else self.ctx.order[0]
        c = await self._call(m, req)
        return Out(c, [j, c], decision={"classifier": j.text.strip()[:20], "routed_to": m})


# --------------------------------------------------------------------------- ensembles
class Vote(Strategy):
    name = "vote"
    knob_help = "k = number of models voting (most accurate first)"
    default_knob = 3

    async def run(self, req):
        k = int(self.knob(req))
        models = sorted(self.ctx.pool.names, key=lambda m: -self.ctx.acc.get(m, 0))[:max(1, k)]
        calls = await asyncio.gather(*(self._call(m, req) for m in models))
        w = {m: np.log(max(self.ctx.acc.get(m, .5), .02) / max(1 - self.ctx.acc.get(m, .5), .02)) for m in models}
        score = defaultdict(float)
        for c in calls:
            score[c.answer] += w[c.model]
        best = max(score, key=score.get)
        winner = next(c for c in sorted(calls, key=lambda c: -w[c.model]) if c.answer == best)
        return Out(winner, list(calls), decision={"votes": {c.model: c.answer for c in calls}})


class Blender(Strategy):
    """LLM-Blender: all models answer in parallel; PairRM (if installed) or token confidence picks."""
    name = "blender"
    knob_help = "unused (all models run)"

    def __init__(self, ctx):
        super().__init__(ctx)
        try:
            import llm_blender

            self.bl = llm_blender.Blender()
            self.bl.loadranker("llm-blender/PairRM")
        except Exception:
            self.bl = None

    async def run(self, req):
        calls = await asyncio.gather(*(self._call(m, req, logprobs=1) for m in self.ctx.pool.names))
        with tel.chain_span("rank", **{"routerlab.ranker": "PairRM" if self.bl else "mean-logprob"}) as sp:
            if self.bl:
                ranks = self.bl.rank([req.prompt], [[c.text for c in calls]], return_scores=False)[0]
                best = calls[int(np.argmin(ranks))]
            else:
                best = max(calls, key=lambda c: c.mean_logprob if c.mean_logprob is not None else -9)
            sp.set_attribute("output.value", best.model)
        return Out(best, list(calls), decision={"picked": best.model})


# --------------------------------------------------------------------------- online learners
class CacheDistil(Strategy):
    """Neural caching: a student (online logistic regression on embeddings) answers when confident;
    otherwise the teacher answers and its answer becomes a training label."""
    name = "cache-distil"
    knob_help = "student answers when its top-1 probability >= knob"
    default_knob = 0.8

    def __init__(self, ctx, min_labels=30):
        super().__init__(ctx)
        from sklearn.linear_model import SGDClassifier

        self.teacher = ctx.strong
        self.clf = SGDClassifier(loss="log_loss", alpha=1e-4)
        self.classes = []
        self.X, self.y = [], []
        self.min_labels = min_labels
        self.lock = asyncio.Lock()

    async def run(self, req):
        e = self.ctx.embed(req)
        if len(self.y) >= self.min_labels and len(set(self.y)) > 1:
            p = self.clf.predict_proba(e[None])[0]
            top = float(p.max())
            if top >= self.knob(req):
                label = self.clf.classes_[int(np.argmax(p))]
                with tel.chain_span("student", **{"output.value": label, "routerlab.student_confidence": top}):
                    pass
                fake = Call("student", f"Answer: {label}", [], 0, 0, 0.0, 0.0)
                return Out(fake, [], text=f"Answer: {label}", decision={"student_confidence": round(top, 3)})
        c = await self._call(self.teacher, req)
        ans = c.answer
        if ans and len(ans) <= 40:          # learn short labels only (classification / MCQ)
            async with self.lock:
                self.X.append(e); self.y.append(ans)
                if len(self.y) >= self.min_labels and len(set(self.y)) > 1 and len(self.y) % 10 == 0:
                    self.clf.fit(np.array(self.X), np.array(self.y))
        return Out(c, [c], decision={"labels_seen": len(self.y)})


class MetaLLMOnline(Strategy):
    """LinUCB contextual bandit; reward = correct - knob * normalised cost, learned from /feedback."""
    name = "metallm-online"
    knob_help = "cost weight p in reward = correct - p * cost"
    default_knob = 0.3

    def __init__(self, ctx, alpha=0.5, ridge=1.0):
        super().__init__(ctx)
        self.alpha, self.ridge = alpha, ridge
        self.A = self.b = None
        self.pending = {}
        c = np.array([ctx.pool.usd(m, 500, 200) for m in ctx.pool.names])
        self.c = c / c.max()

    def _init(self, d):
        M = len(self.ctx.pool.names)
        self.Ainv = np.stack([np.eye(d) / self.ridge] * M)
        self.b = np.zeros((M, d))

    async def run(self, req):
        x = np.append(self.ctx.embed(req), 1.0)
        if self.b is None:
            self._init(len(x))
        th = np.einsum("mij,mj->mi", self.Ainv, self.b)
        ucb = th @ x + self.alpha * np.sqrt(np.einsum("i,mij,j->m", x, self.Ainv, x))
        a = int(np.argmax(ucb))
        self.pending[req.request_id] = (x, a, self.knob(req))
        c = await self._call(self.ctx.pool.names[a], req)
        return Out(c, [c], decision={"ucb": {n: round(float(u), 3) for n, u in zip(self.ctx.pool.names, ucb)}})

    def feedback(self, request_id, correct):
        if request_id not in self.pending:
            return
        x, a, p = self.pending.pop(request_id)
        r = float(correct) - p * self.c[a]
        Ax = self.Ainv[a] @ x
        self.Ainv[a] -= np.outer(Ax, Ax) / (1 + x @ Ax)
        self.b[a] += r * x


# --------------------------------------------------------------------------- semantic cache wrapper
class SemanticCache(Strategy):
    knob_help = "inner strategy's knob"

    def __init__(self, ctx, inner: Strategy, threshold=0.97, max_items=50_000):
        super().__init__(ctx)
        self.inner, self.thr, self.max = inner, threshold, max_items
        self.name = f"cache+{inner.name}"
        self.E, self.items = [], []

    async def run(self, req):
        e = self.ctx.embed(req)
        en = e / (np.linalg.norm(e) + 1e-9)
        if self.E:
            sims = np.asarray(self.E) @ en
            j = int(np.argmax(sims))
            if sims[j] >= self.thr:
                tel.M.cache_hits.add(1, {"strategy": self.name})
                text, model = self.items[j]
                with tel.chain_span("cache.hit", **{"routerlab.similarity": float(sims[j]), "output.value": text}):
                    pass
                return Out(Call(f"cache({model})", text, [], 0, 0, 0.0, 0.0), [], cache_hit=True, text=text,
                           decision={"similarity": round(float(sims[j]), 4)})
        out = await self.inner.run(req)
        if len(self.E) < self.max:
            self.E.append(en)
            self.items.append((out.text or out.call.text, out.call.model))
        return out

    def feedback(self, request_id, correct):
        self.inner.feedback(request_id, correct)


# --------------------------------------------------------------------------- registry
def build(ctx: Ctx) -> dict[str, Strategy]:
    S: dict[str, Strategy] = {}

    def add(factory):
        try:
            s = factory()
            S[s.name] = s
        except Exception as e:
            print(f"[strategies] skipped: {e}")

    for m in ctx.pool.names:
        add(lambda m=m: Single(ctx, m))
    for f in (RandomMix, TaskStatic, TwoTier, AutoMix, FrugalGPT, Consistency, LLMJudgeRouter, Vote, Blender,
              CacheDistil, MetaLLMOnline):
        add(lambda f=f: f(ctx))
    for rn, r in ctx.routers.items():
        if getattr(r, "supports_live", False):
            add(lambda rn=rn: Predict(ctx, rn))
    return S


def resolve(S: dict[str, Strategy], ctx: Ctx, name: str) -> Strategy:
    """'cache+X' wrappers are created on first use and kept (so the cache persists)."""
    if name in S:
        return S[name]
    if name.startswith("cache+") and name[6:] in S:
        S[name] = SemanticCache(ctx, S[name[6:]])
        return S[name]
    raise KeyError(name)


def describe(S) -> list[dict]:
    return [{"strategy": n, "knob": s.knob_help, "default_knob": s.ctx.knobs.get(n, s.default_knob)} for n, s in S.items()]


__all__ = ["Ctx", "Req", "Out", "build", "resolve", "describe", "extract_answer"]
