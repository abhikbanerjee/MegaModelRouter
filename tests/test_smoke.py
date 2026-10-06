"""CPU smoke test: simulated pool -> collect -> every router -> metrics.   pytest -q tests/"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routerlab.backends import MockBackend  # noqa: E402
from routerlab.collect import collect  # noqa: E402
from routerlab.config import load_experiment, load_pool  # noqa: E402
from routerlab.evaluate import evaluate_router  # noqa: E402
from routerlab.mock import synthetic_queries  # noqa: E402
from routerlab.routers import REGISTRY, make  # noqa: E402
from routerlab.table import load_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_all_routers(tmp_path):
    pool = load_pool(ROOT / "configs/models.yaml")
    exp = load_experiment(ROOT / "configs/experiment.yaml")
    qs = synthetic_queries(1200)
    collect(qs, pool, MockBackend(pool, qs), exp, tmp_path)
    data = load_run(tmp_path)
    tr, va, te = data.split("train"), data.split("val"), data.split("test")
    for name in REGISTRY:
        res = evaluate_router(make(name), tr, va, te)
        assert res["points"], name
        assert 0 <= res["metrics"]["max_acc"] <= 1, name
