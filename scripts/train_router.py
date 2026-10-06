#!/usr/bin/env python
"""Train routers on a collected run, pickle them for the gateway, and pick each one's default knob.

  python scripts/train_router.py --run runs/x --all                 # every live-capable router
  python scripts/train_router.py --run runs/x --router routellm-mf  # just one
  --target-cost-frac 0.5   choose the knob whose test-split cost is closest to 50% of the strongest model's

Writes runs/x/routers/<router>.pkl and runs/x/routers/knobs.json (keys = live strategy names).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routerlab.evaluate import score_decision  # noqa: E402
from routerlab.routers import make  # noqa: E402
from routerlab.table import load_run  # noqa: E402

# offline routers that have a live strategy, and that strategy's name in the gateway
LIVE = {"embedllm": None, "routellm-mf": None, "routellm-sw_ranking": None, "routellm-clf": None,
        "metallm-offline": None, "minds-classifier": None, "minds-cluster": None, "hybrid-trans": None,
        "hybrid-prob": None, "predictive-pref": None, "routoo": None, "flyswat-greedy": None, "zooter": None,
        "benchmark-knn": None, "tryage": None,
        "two-tier-marginmin": "two-tier", "automix-pomdp": "automix", "frugalgpt": "frugalgpt"}

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--router")
ap.add_argument("--all", action="store_true")
ap.add_argument("--target-cost-frac", type=float, default=0.5)
a = ap.parse_args()
data = load_run(a.run)
train, val, test = data.split("train"), data.split("val"), data.split("test")
strong = int(test.correct.mean(0).argmax())
strong_cost = test.cost.mean(0)[strong]
out = Path(a.run) / "routers"
out.mkdir(exist_ok=True)
kp = out / "knobs.json"
knobs = json.loads(kp.read_text()) if kp.exists() else {}
names = list(LIVE) if a.all else [a.router]
print(f"strongest={data.models[strong]} acc={test.correct.mean(0)[strong]:.3f} cost=${strong_cost:.2e}/query\n")
print(f"{'router':22s} {'knob':>9s} {'acc':>6s} {'cost%':>6s}  live strategy")
for n in names:
    r = make(n)
    miss = [k for k in r.needs if k != "logprobs" and getattr(data, k, None) is None]
    if miss:
        print(f"{n:22s} skipped: needs {miss}")
        continue
    r.fit(train, val)
    best = None
    for k in r.knobs():
        acc, cost, _, _ = score_decision(test, r.decide(test, k))
        gap = abs(cost / strong_cost - a.target_cost_frac)
        if best is None or gap < best[0]:
            best = (gap, float(k), acc, cost)
    (out / f"{n}.pkl").write_bytes(pickle.dumps(r))
    live = LIVE.get(n) or f"predict/{n}"
    knobs[live] = best[1] if abs(best[1]) != float("inf") else (1e9 if best[1] > 0 else -1e9)
    print(f"{n:22s} {best[1]:9.4g} {best[2]:6.3f} {100 * best[3] / strong_cost:5.0f}%  router:{live}")
kp.write_text(json.dumps(knobs, indent=1))
print(f"\nsaved routers + {kp}")
