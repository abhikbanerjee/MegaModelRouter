"""EcoAssistant (Zhang et al., 2023) - assistant hierarchy + solution demonstrations for
code-driven queries.

Two ideas:
  1. Assistant hierarchy: try the cheapest LLM first; escalate only when the user (or an
     automatic checker) says the attempt failed.
  2. Solution demonstration: every query that was solved is stored with its working code; for a
     new query, retrieve the most similar solved ones and put them in the prompt as examples.
     This makes the CHEAP models succeed much more often over time - the paper's main gain.

Because (2) changes model outputs as the stream proceeds, it cannot be replayed offline from
the outcome table. So there are two pieces:
  * EcoAssistantRouter      offline replay of the hierarchy alone (the paper's success signal is
                            user feedback - modelled here as an oracle checker; this is an upper
                            bound for the hierarchy without demonstrations).
  * EcoAssistantLive        the full method against live vLLM servers on GSM8K-style math:
                            models write Python, we execute it, a checker decides success,
                            solved (query, code) pairs feed the retrieval memory.
                            Run with scripts/run_ecoassistant.py.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time

import numpy as np

from .base import Decision, Router, cheapest_first


class EcoAssistantRouter(Router):
    name = "ecoassistant-hierarchy-oraclefb"   # oracle user feedback => upper bound
    paper = "EcoAssistant (2310.03046)"

    def fit(self, train, val=None):
        self.order = list(cheapest_first(train))
        return self

    def knobs(self):
        return list(range(1, 1 + len(self.order)))     # max hierarchy depth

    def decide(self, data, knob):
        chain = self.order[: int(knob)]
        N = data.N
        final = np.full(N, chain[-1])
        calls = np.zeros((N, data.M), bool)
        active = np.ones(N, bool)
        for m in chain:
            calls[active, m] = True
            ok = active & (data.correct[:, m] == 1)     # user feedback: "it worked"
            final[ok] = m
            active &= ~ok
        return Decision(final=final, calls=calls, sequential=True)


CODE_PROMPT = """You are a helpful assistant that solves problems by writing Python.
{demos}Question: {question}
Write a Python program that computes the answer and prints ONLY the final number.
Return the code in a ```python block."""


def _exec(code: str, timeout=10.0) -> str | None:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
    try:
        r = subprocess.run([sys.executable, "-I", f.name], capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip().split("\n")[-1] if r.returncode == 0 and r.stdout.strip() else None
    except subprocess.TimeoutExpired:
        return None


class EcoAssistantLive:
    """Full EcoAssistant loop. `checker(q, output) -> bool` stands in for user feedback:
    pass a gold-answer checker for benchmarking, or an LLM judge (the paper used GPT-4 as the user)."""

    def __init__(self, backend, pool, embedder, models_cheap_to_expensive: list[str], checker,
                 k_demos=2, sim_threshold=0.5, max_turns=2):
        self.b, self.pool, self.emb = backend, pool, embedder
        self.chain, self.checker = models_cheap_to_expensive, checker
        self.k, self.thr, self.turns = k_demos, sim_threshold, max_turns
        self.mem_q, self.mem_code, self.mem_e = [], [], []

    def _demos(self, e):
        if not self.mem_e:
            return ""
        sims = np.array(self.mem_e) @ e
        idx = [i for i in np.argsort(-sims)[: self.k] if sims[i] >= self.thr]
        return "".join(f"Example question: {self.mem_q[i]}\n```python\n{self.mem_code[i]}\n```\n\n" for i in idx)

    def solve(self, q) -> dict:
        e = self.emb.encode([q.prompt])[0]
        demos = self._demos(e)
        log = {"qid": q.qid, "attempts": [], "solved_by": None, "tokens": 0, "latency_s": 0.0}
        for m in self.chain:
            prompt = CODE_PROMPT.format(demos=demos, question=q.prompt.split("\n\nSolve step")[0])
            for turn in range(self.turns):
                t0 = time.perf_counter()
                g = self.b.generate(m, [prompt], 512, 0.0, qids=[q.qid])[0][0]
                log["latency_s"] += time.perf_counter() - t0
                log["tokens"] += g.n_in + g.n_out
                mm = re.search(r"```(?:python)?\n(.*?)```", g.text, re.S)
                code = mm.group(1) if mm else g.text
                out = _exec(code)
                ok = out is not None and self.checker(q, out)
                log["attempts"].append({"model": m, "turn": turn, "ok": ok, "n_in": g.n_in, "n_out": g.n_out})
                if ok:
                    log["solved_by"] = m
                    self.mem_q.append(q.prompt.split("\n\nSolve step")[0]); self.mem_code.append(code); self.mem_e.append(e)
                    return log
                prompt += f"\n\nYour previous code:\n```python\n{code}\n```\nOutput: {out!r}. That is wrong or failed. Fix it."
        return log
