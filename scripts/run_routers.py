#!/usr/bin/env python
"""Phase 2 - train every router on the train split and sweep its cost/quality knob on test.

  python scripts/run_routers.py --run runs/qwen_llama                 # all routers
  python scripts/run_routers.py --run runs/qwen_llama --only routellm-mf,automix-pomdp

Writes  <run>/results/results.json, leaderboard.md, leaderboard.csv, cost_quality.png,
        and one PNG per router.
"""
import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from routerlab.evaluate import evaluate_router, plot_results, save  # noqa: E402
from routerlab.routers import REGISTRY, make  # noqa: E402
from routerlab.table import load_run  # noqa: E402


def fmt(x, pct=False):
    if x is None:
        return "-"
    return f"{x:.1f}%" if pct else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    data = load_run(a.run)
    train, val, test = data.split("train"), data.split("val"), data.split("test")
    print(f"models={data.models}  train={train.N} val={val.N} test={test.N}")
    names = [n for n in a.only.split(",") if n] or list(REGISTRY)
    out = Path(a.run) / "results"
    out.mkdir(exist_ok=True)
    results = []
    for n in names:
        r = make(n)
        miss = [k for k in r.needs if k != "logprobs" and getattr(data, k, None) is None]
        if miss:
            print(f"[skip] {n}: needs {miss} (enable in experiment.yaml extras)")
            continue
        try:
            res = evaluate_router(r, train, val, test)
        except Exception:
            print(f"[fail] {n}\n{traceback.format_exc()}")
            continue
        m = res["metrics"]
        print(f"{n:24s} APGR={m['apgr']:.3f} AUC={m['auc']:.3f} maxAcc={m['max_acc']:.3f} "
              f"save@best={fmt(m['savings_at_best_single_pct'], True)} fit={res['fit_s']:.1f}s")
        results.append(res)
        plot_results([res], out / f"{n}.png", title=f"{n} - {res['paper']}")
    save(results, out / "results.json")
    full = [r for r in results if r["kinds"] == "all"]
    if full:
        plot_results(full, out / "cost_quality.png")
    rows = [{"router": r["router"], "paper": r["paper"], "subset": ",".join(r["kinds"]) if r["kinds"] != "all" else "all",
             "APGR": r["metrics"]["apgr"], "AUC": r["metrics"]["auc"], "CPT50": r["metrics"]["cpt50"],
             "CPT80": r["metrics"]["cpt80"], "max_acc": r["metrics"]["max_acc"],
             "gain_vs_best_single_pts": r["metrics"]["gain_over_best_single_pts"],
             "cost_saving_at_best_single_%": r["metrics"]["savings_at_best_single_pct"],
             "dIBC_%": r["metrics"]["dIBC_pct"], "route_ms/query": r["route_ms_per_query"]} for r in results]
    df = pd.DataFrame(rows).sort_values(["subset", "APGR"], ascending=[True, False])
    df.to_csv(out / "leaderboard.csv", index=False)
    (out / "leaderboard.md").write_text(df.to_markdown(index=False, floatfmt=".3f"))
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
