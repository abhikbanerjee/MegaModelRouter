"""Benchmark loading, prompt formatting, answer extraction and scoring.

Every benchmark is normalised into a `Query`. Routers never see gold answers at
inference time; they only see `prompt` (and its embedding).
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field

LETTERS = "ABCDEFGHIJ"


@dataclass
class Query:
    qid: str
    task: str
    kind: str                 # "mcq" | "math" | "cls" | "code"
    prompt: str               # full user message sent to every model
    answer: str               # gold (letter / number / label / tests json)
    subject: str = ""         # MMLU subject etc. (used as a "task" for benchmark routing)
    choices: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- prompts
def mcq_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"{question.strip()}\n\n{opts}\n\n"
        "Answer with the letter of the correct option only, in the form 'Answer: X'."
    )


def math_prompt(question: str) -> str:
    return (
        f"{question.strip()}\n\nSolve step by step. "
        "Finish with a final line of the form 'Answer: <number>'."
    )


def cls_prompt(text: str, labels: list[str], instruction: str) -> str:
    opts = ", ".join(labels)
    return f"{instruction}\n\nText: {text.strip()}\n\nOptions: {opts}\nReply in the form 'Answer: <option>'."


# ---------------------------------------------------------------- loaders
def _hf(name, *args, split="test"):
    from datasets import load_dataset  # lazy: not needed for mock runs

    return load_dataset(name, *args, split=split)


def load_mmlu(n, rng):
    ds = _hf("cais/mmlu", "all")
    idx = rng.sample(range(len(ds)), min(n, len(ds))) if n else range(len(ds))
    out = []
    for i in idx:
        r = ds[i]
        out.append(Query(f"mmlu-{i}", "mmlu", "mcq", mcq_prompt(r["question"], r["choices"]),
                         LETTERS[r["answer"]], subject=r["subject"], choices=r["choices"]))
    return out


def load_arc(n, rng):
    ds = _hf("allenai/ai2_arc", "ARC-Challenge")
    idx = rng.sample(range(len(ds)), min(n, len(ds))) if n else range(len(ds))
    out = []
    for i in idx:
        r = ds[i]
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        if r["answerKey"] not in labels:
            continue
        gold = LETTERS[labels.index(r["answerKey"])]
        out.append(Query(f"arc-{i}", "arc_challenge", "mcq", mcq_prompt(r["question"], texts), gold,
                         subject="arc", choices=texts))
    return out


def load_hellaswag(n, rng):
    ds = _hf("Rowan/hellaswag", split="validation")
    idx = rng.sample(range(len(ds)), min(n, len(ds))) if n else range(len(ds))
    out = []
    for i in idx:
        r = ds[i]
        q = f"Choose the most plausible continuation.\n\nContext: {r['ctx']}"
        out.append(Query(f"hs-{i}", "hellaswag", "mcq", mcq_prompt(q, r["endings"]), LETTERS[int(r["label"])],
                         subject="hellaswag", choices=r["endings"]))
    return out


def load_gsm8k(n, rng):
    ds = _hf("openai/gsm8k", "main")
    idx = rng.sample(range(len(ds)), min(n, len(ds))) if n else range(len(ds))
    out = []
    for i in idx:
        r = ds[i]
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append(Query(f"gsm-{i}", "gsm8k", "math", math_prompt(r["question"]), gold, subject="gsm8k"))
    return out


def load_sst2(n, rng):
    ds = _hf("stanfordnlp/sst2", split="validation")
    labels = ["negative", "positive"]
    idx = rng.sample(range(len(ds)), min(n, len(ds))) if n else range(len(ds))
    return [Query(f"sst2-{i}", "sst2", "cls",
                  cls_prompt(ds[i]["sentence"], labels, "Classify the sentiment of the text."),
                  labels[ds[i]["label"]], subject="sst2", choices=labels) for i in idx]


def load_ag_news(n, rng):
    ds = _hf("fancyzhx/ag_news")
    labels = ["world", "sports", "business", "technology"]
    idx = rng.sample(range(len(ds)), min(n, len(ds))) if n else range(len(ds))
    return [Query(f"ag-{i}", "ag_news", "cls",
                  cls_prompt(ds[i]["text"], labels, "Classify the topic of the news article."),
                  labels[ds[i]["label"]], subject="ag_news", choices=labels) for i in idx]


def load_mbpp(n, rng):
    ds = _hf("google-research-datasets/mbpp", "sanitized")
    idx = rng.sample(range(len(ds)), min(n, len(ds))) if n else range(len(ds))
    out = []
    for i in idx:
        r = ds[i]
        p = (f"{r['prompt']}\nYour code must pass this test:\n{r['test_list'][0]}\n"
             "Return only the Python code in a ```python block.")
        out.append(Query(f"mbpp-{i}", "mbpp", "code", p, json.dumps(r["test_list"]), subject="mbpp"))
    return out


LOADERS = {
    "mmlu": load_mmlu, "arc_challenge": load_arc, "hellaswag": load_hellaswag,
    "gsm8k": load_gsm8k, "sst2": load_sst2, "ag_news": load_ag_news, "mbpp": load_mbpp,
}


def load_tasks(task_counts: dict, seed: int = 0) -> list[Query]:
    rng = random.Random(seed)
    qs: list[Query] = []
    for task, n in task_counts.items():
        if n == 0:
            continue
        qs.extend(LOADERS[task](n, rng))
    rng.shuffle(qs)
    return qs


# ---------------------------------------------------------------- scoring
_ANS = re.compile(r"answer\s*[:：]\s*\(?([^\n\)]*)", re.I)


def extract(text: str, q: Query) -> str:
    m = list(_ANS.finditer(text))
    cand = m[-1].group(1).strip() if m else text.strip().split("\n")[-1]
    if q.kind == "mcq":
        lm = re.search(r"\b([A-J])\b", cand) or re.search(r"\b([A-J])\b", text[:20])
        return lm.group(1) if lm else ""
    if q.kind == "math":
        nums = re.findall(r"-?\d[\d,]*\.?\d*", cand) or re.findall(r"-?\d[\d,]*\.?\d*", text)
        return nums[-1].replace(",", "").rstrip(".") if nums else ""
    if q.kind == "cls":
        low = cand.lower()
        for c in q.choices:
            if c.lower() in low:
                return c
        for c in q.choices:  # fall back to anywhere in the text
            if c.lower() in text.lower():
                return c
        return ""
    if q.kind == "code":
        m2 = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
        return m2.group(1) if m2 else text
    return cand


def _num_eq(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def run_code_tests(code: str, tests: list[str], timeout: float = 10.0) -> bool:
    """Executes generated code in a subprocess. Run the bench in a container if you enable MBPP."""
    src = code + "\n\n" + "\n".join(tests) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
    try:
        r = subprocess.run([sys.executable, "-I", f.name], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def score(text: str, q: Query) -> tuple[str, float]:
    pred = extract(text, q)
    if q.kind == "math":
        return pred, float(_num_eq(pred, q.answer))
    if q.kind == "code":
        return pred[:200], float(run_code_tests(pred, json.loads(q.answer)))
    return pred, float(pred.lower() == q.answer.lower())
