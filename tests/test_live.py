"""Every live strategy end-to-end against the simulated pool, in-process (no servers, no GPU).
Needs a trained mock run: python scripts/collect.py --mock --out runs/mock && python scripts/train_router.py --run runs/mock --all"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / "runs/mock"
pytestmark = pytest.mark.skipif(not (RUN / "routers/knobs.json").exists(), reason="no trained mock run")


def test_every_strategy():
    os.environ.update(RUN_DIR=str(RUN), MODELS_YAML=str(ROOT / "configs/models.mock.yaml"), ROUTERLAB_METRICS="0",
                      MOCK_LATENCY_SCALE="0", OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:9")
    import httpx

    from routerlab import mockserver, telemetry
    from routerlab.aux_models import Embedder
    from routerlab.config import load_pool
    from routerlab.live.pool import Pool
    from routerlab.live.strategies import Ctx, Req, build, resolve

    telemetry.init("test")
    pool = Pool(load_pool(ROOT / "configs/models.mock.yaml"), RUN)
    pool.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=mockserver.app), timeout=60)
    ctx = Ctx(pool, RUN, Embedder(None))
    S = build(ctx)
    assert len(S) >= 25, list(S)
    prompt = mockserver.qs[0].prompt

    async def go():
        for name in list(S) + ["cache+two-tier"]:
            st = resolve(S, ctx, name)
            for i in range(2):   # twice: exercises caches / online learners
                out = await st.run(Req([{"role": "user", "content": prompt}], prompt, None, 64, f"{name}-{i}"))
                assert out.call is not None, name
                st.feedback(f"{name}-{i}", True)
    asyncio.run(go())
