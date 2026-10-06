"""GPU fine-tuning of encoder routers - the paper-faithful versions of the heads in routers/*.

Several papers fine-tune a full encoder rather than training a head on frozen embeddings:
  RouteLLM 'BERT classifier'   - BERT-base, P(strong wins)                 target: golden/arena labels
  Hybrid LLM                   - DeBERTa-v3-large, P(small is good enough) target: det/prob/trans labels
  Fly-Swat / Multiple Minds    - DistilBERT/RoBERTa, multi-label per-model correctness
  FrugalGPT scorer             - DistilBERT on (query, answer) -> reliability
  Zooter                       - mDeBERTa-v3-base with soft (KL) targets over models

`train_encoder_router` covers all of them: pass texts (or text pairs) and a target matrix,
choose loss 'bce' (multi-label / binary) or 'kl' (soft distribution). Returns a predictor
whose outputs you can plug into the matching router (they all expose a `predict`-style head).
"""
from __future__ import annotations

import numpy as np


def train_encoder_router(texts, Y, model_id="microsoft/deberta-v3-base", loss="bce", text_pairs=None,
                         epochs=3, lr=2e-5, bs=32, max_len=512, device="cuda", val=None):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    Y = np.asarray(Y, np.float32)
    if Y.ndim == 1:
        Y = Y[:, None]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=Y.shape[1]).to(device)
    enc = tok(list(texts), list(text_pairs) if text_pairs is not None else None, truncation=True,
              max_length=max_len, padding="max_length", return_tensors="pt")
    dl = DataLoader(TensorDataset(enc.input_ids, enc.attention_mask, torch.tensor(Y)), batch_size=bs, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * epochs * len(dl)), epochs * len(dl))
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for ids, am, y in dl:
            ids, am, y = ids.to(device), am.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=am).logits.float()
            if loss == "bce":
                l = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            else:
                l = torch.nn.functional.kl_div(torch.log_softmax(logits, -1), y, reduction="batchmean")
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += l.item()
        print(f"epoch {ep}: loss {tot / len(dl):.4f}")
    model.eval()

    @torch.no_grad()
    def predict(texts, pairs=None):
        out = []
        for i in range(0, len(texts), 64):
            e = tok(list(texts[i:i + 64]), list(pairs[i:i + 64]) if pairs is not None else None, truncation=True,
                    max_length=max_len, padding=True, return_tensors="pt").to(device)
            z = model(**e).logits.float()
            out.append((torch.sigmoid(z) if loss == "bce" else torch.softmax(z, -1)).cpu().numpy())
        return np.concatenate(out)

    return predict


class FinetunedHead:
    """Adapter so a fine-tuned encoder can replace an MLP head inside a router:
        r = FlySwatRouter().fit(train); r.meta = FinetunedHead(pred_fn, texts_by_qid)
    `predict(X)` is called with embeddings in routers; we look the texts up by row order instead."""

    def __init__(self, predict_fn, texts):
        self.f, self.texts = predict_fn, texts

    def predict(self, X):
        assert len(X) == len(self.texts), "set .texts to the prompts of the batch being routed"
        return self.f(self.texts)
