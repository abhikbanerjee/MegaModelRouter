"""Offline results explorer.   streamlit run scripts/dashboard.py -- --run runs/x

Pick routers, compare cost-quality curves, inspect traffic share per model at any operating
point, per-task accuracy, and each paper's diagnostic report."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/default")
args = ap.parse_args(sys.argv[1:])
R = json.loads((Path(args.run) / "results" / "results.json").read_text())

st.set_page_config(page_title="LLM routing lab", layout="wide")
st.title("LLM routing lab")
lb = pd.read_csv(Path(args.run) / "results" / "leaderboard.csv")
st.subheader("Leaderboard (test split)")
st.dataframe(lb, width="stretch", hide_index=True)

names = [r["router"] for r in R]
pick = st.multiselect("Routers", names, default=names[:6])
rows = []
for r in R:
    if r["router"] in pick:
        for p in r["points"]:
            rows.append({"router": r["router"], "cost": p["cost"], "acc": p["acc"], "latency": p["latency"], "knob": p["knob"]})
base = R[0]["baselines"]["single"]
single = pd.DataFrame([{"router": f"[{m}]", "cost": v["cost"], "acc": v["acc"], "latency": v["latency"], "knob": None}
                       for m, v in base.items()])
df = pd.concat([pd.DataFrame(rows), single])
c1, c2 = st.columns(2)
c1.plotly_chart(px.line(df.sort_values("cost"), x="cost", y="acc", color="router", markers=True, log_x=True,
                        title="Accuracy vs $/query"), width="stretch")
c2.plotly_chart(px.line(df.sort_values("latency"), x="latency", y="acc", color="router", markers=True,
                        title="Accuracy vs latency/query"), width="stretch")

st.subheader("Where does the traffic go?")
r = next(x for x in R if x["router"] == st.selectbox("Router", names))
i = st.slider("Operating point (knob index)", 0, len(r["points"]) - 1, len(r["points"]) // 2)
p = r["points"][i]
st.write(f"knob={p['knob']:.4g}  acc={p['acc']:.3f}  cost=${p['cost']:.6f}/query  calls/query={p['calls_per_query']:.2f}")
st.plotly_chart(px.bar(x=r["models"], y=p["share"], labels={"x": "model", "y": "share of answers"}), width="stretch")
if "report" in r:
    st.subheader("Paper-specific diagnostics")
    st.json(r["report"])
