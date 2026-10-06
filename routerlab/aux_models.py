"""Auxiliary models: query embedder, reward model, PairRM ranker. Each has a cheap
fallback so mock runs need no GPU and no downloads."""
from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, model_id: str | None = None, device: str | None = None, dim_fallback: int = 256):
        self.model_id, self.device, self.dim_fallback = model_id, device, dim_fallback
        self._st = None
        if model_id:
            try:
                from sentence_transformers import SentenceTransformer

                self._st = SentenceTransformer(model_id, device=device)
            except Exception as e:  # pragma: no cover
                print(f"[embedder] falling back to hashing features ({e})")

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._st is not None:
            return np.asarray(self._st.encode(texts, batch_size=128, normalize_embeddings=True,
                                              show_progress_bar=True), dtype=np.float32)
        from sklearn.feature_extraction.text import HashingVectorizer

        hv = HashingVectorizer(n_features=self.dim_fallback, ngram_range=(1, 2), alternate_sign=True, norm="l2")
        return hv.transform(texts).toarray().astype(np.float32)


class RewardModel:
    """Scores (prompt, response). Default: OpenAssistant DeBERTa-v3 reward model.
    Any HF sequence-classification RM with a single logit works."""

    def __init__(self, model_id: str | None, device: str = "cuda", batch_size: int = 16):
        self.model_id, self.device, self.bs = model_id, device, batch_size
        if model_id:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id, torch_dtype=torch.bfloat16).to(device).eval()

    def score(self, prompts: list[str], responses: list[str]) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(prompts), self.bs):
            enc = self.tok(prompts[i:i + self.bs], responses[i:i + self.bs], truncation=True,
                           max_length=1024, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out.append(self.model(**enc).logits[:, 0].float().cpu().numpy())
        return np.concatenate(out)


class PairRanker:
    """LLM-Blender PairRM (pip install llm-blender). Returns per-candidate rank (1 = best)."""

    def __init__(self, model_id: str = "llm-blender/PairRM"):
        import llm_blender

        self.blender = llm_blender.Blender()
        self.blender.loadranker(model_id)

    def rank(self, prompts: list[str], candidates: list[list[str]]) -> np.ndarray:
        return np.asarray(self.blender.rank(prompts, candidates, return_scores=False, batch_size=8))

    def fuse(self, prompts, topk_candidates, fuser_id="llm-blender/gen_fuser_3b"):
        self.blender.loadfuser(fuser_id)
        return self.blender.fuse(prompts, topk_candidates, batch_size=4)
