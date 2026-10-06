#!/usr/bin/env python
"""'Merge' from the Merge/Ensemble/Cooperate survey: build a merged checkpoint from two or more
fine-tunes of the SAME base model, then add it to configs/models.yaml and re-collect to see
whether one merged model beats routing between its parents.

  # linear average
  python scripts/merge_models.py --out merged/qwen-mix \
      --models Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-Math-7B-Instruct --weights 0.5 0.5
  # task arithmetic: base + sum_i w_i (ft_i - base)
  python scripts/merge_models.py --out merged/qwen-ta --base Qwen/Qwen2.5-7B \
      --models Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-Coder-7B-Instruct --weights 0.7 0.5

For TIES/DARE/SLERP use mergekit (pip install mergekit) - same idea, better conflict handling.
Merging only works across checkpoints sharing the exact architecture and tokenizer.
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--weights", nargs="+", type=float, required=True)
ap.add_argument("--base", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()
assert len(a.models) == len(a.weights)
load = lambda m: AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16, device_map="cpu")  # noqa: E731
sds = [load(m).state_dict() for m in a.models]
target = load(a.base or a.models[0])
base_sd = {k: v.clone() for k, v in target.state_dict().items()}
new = {}
for k, v in base_sd.items():
    if any(k not in sd or sd[k].shape != v.shape for sd in sds):
        new[k] = v                       # e.g. resized embeddings: keep target's
        continue
    if a.base:
        new[k] = v + sum(w * (sd[k].float() - v.float()) for w, sd in zip(a.weights, sds)).to(v.dtype)
    else:
        new[k] = (sum(w * sd[k].float() for w, sd in zip(a.weights, sds)) / sum(a.weights)).to(v.dtype)
target.load_state_dict(new)
target.save_pretrained(a.out)
AutoTokenizer.from_pretrained(a.models[0]).save_pretrained(a.out)
print("saved", a.out, "- add it to configs/models.yaml (hf_id: <path>) and re-run collect.py")
