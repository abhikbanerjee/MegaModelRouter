from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelSpec:
    name: str
    hf_id: str
    port: int = 8000
    num_gpus: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    quantization: str | None = None
    tier: str = "medium"
    price_in_per_mtok: float | None = None
    price_out_per_mtok: float | None = None
    host: str = "localhost"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


@dataclass
class PoolConfig:
    models: list[ModelSpec]
    gpu_hourly_usd: float = 2.0
    aux: dict[str, Any] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.models]

    def get(self, name: str) -> ModelSpec:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(name)


def load_pool(path: str | Path = "configs/models.yaml") -> PoolConfig:
    raw = yaml.safe_load(Path(path).read_text())
    models = [ModelSpec(**m) for m in raw["models"]]
    return PoolConfig(models=models, gpu_hourly_usd=raw.get("gpu_hourly_usd", 2.0), aux=raw.get("aux", {}))


def load_experiment(path: str | Path = "configs/experiment.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())
