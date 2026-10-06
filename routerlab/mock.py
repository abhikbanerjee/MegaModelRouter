"""Synthetic queries for the mock pool (CPU smoke tests). Text carries topic words so that
embedding-based routers have a real signal, exactly like subject matter does for real data."""
from __future__ import annotations

import random

from .tasks import Query

VOCAB = {
    "algebra": "equation solve variable integer sum product ratio percent fraction apples cost total",
    "biology": "cell protein gene enzyme organism dna species evolution membrane tissue",
    "history": "empire war treaty century dynasty revolution king colonial parliament",
    "law": "court statute contract liability plaintiff tort defendant jurisdiction appeal",
    "physics": "force energy velocity mass charge quantum field momentum wave particle",
    "sentiment": "movie film acting plot boring wonderful terrible great loved hated",
    "sports": "team match goal season player coach league championship score tournament",
}
FILLER = "the of and a to in is that for it with as on was this by which".split()


def synthetic_queries(n=3000, seed=0) -> list[Query]:
    rng = random.Random(seed)
    spec = [("mmlu", "mcq", t) for t in ("biology", "history", "law", "physics")] + \
           [("gsm8k", "math", "algebra"), ("sst2", "cls", "sentiment"), ("ag_news", "cls", "sports")]
    out = []
    for i in range(n):
        task, kind, topic = rng.choice(spec)
        words = VOCAB[topic].split()
        txt = " ".join(rng.choice(words) if rng.random() < 0.45 else rng.choice(FILLER) for _ in range(30))
        if kind == "mcq":
            ans, choices = rng.choice("ABCD"), ["w", "x", "y", "z"]
        elif kind == "math":
            ans, choices = str(rng.randint(1, 500)), []
        elif topic == "sentiment":
            choices = ["negative", "positive"]; ans = rng.choice(choices)
        else:
            choices = ["world", "sports", "business", "technology"]; ans = rng.choice(choices)
        out.append(Query(f"syn-{i}", task, kind, f"[{topic}] {txt}", ans, subject=topic, choices=choices))
    # ~5% repeated questions (real traffic repeats; lets FrugalGPT's completion cache show its effect)
    for j in range(n // 20):
        src = rng.choice(out[: n])
        out.append(Query(f"syn-dup{j}", src.task, src.kind, src.prompt + " ?", src.answer, src.subject, src.choices))
    rng.shuffle(out)
    return out
