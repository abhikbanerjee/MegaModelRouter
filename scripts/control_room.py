"""Routing Control Room - one page to try strategies, run live comparisons, and read the results
from OpenTelemetry metrics (Prometheus) and Phoenix traces.

    streamlit run scripts/control_room.py -- --run runs/mock

Tabs
  Try a request   pick a strategy + knob, send a prompt, see the routing decision, $ and the Phoenix trace
  Compare live    fire the same queries at several strategies (loadgen), scatter accuracy vs $
  OTel metrics    per-strategy scorecard + time series, straight from the OTel metrics in Prometheus
  Phoenix traces  recent routed requests from Phoenix: strategy, model, $, savings, correctness
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/mock")
args, _ = ap.parse_known_args(sys.argv[1:])
ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Routing control room", layout="wide")
with st.sidebar:
    st.header("Endpoints")
    GW = st.text_input("Gateway", "http://localhost:9000")
    PX = st.text_input("Phoenix", "http://localhost:6006")
    PROM = st.text_input("Prometheus (OTel metrics)", "http://localhost:9090")
    RUN = st.text_input("Run dir", args.run)
    PROJECT = st.text_input("Phoenix project", "llm-routing")
    WINDOW = st.selectbox("Metrics window", ["15m", "30m", "1h", "6h", "24h"], index=2)


@st.cache_data(ttl=10)
def strategies(gw):
    try:
        return httpx.get(f"{gw}/strategies", timeout=5).json()
    except Exception as e:
        return {"error": str(e), "strategies": []}


@st.cache_data(ttl=60)
def phoenix_project_id(px_url, name):
    try:
        r = httpx.post(f"{px_url}/graphql", json={"query": "{ projects { edges { node { id name } } } }"}, timeout=5).json()
        return next(e["node"]["id"] for e in r["data"]["projects"]["edges"] if e["node"]["name"] == name)
    except Exception:
        return None


def prom(q):
    try:
        r = httpx.get(f"{PROM}/api/v1/query", params={"query": q}, timeout=10).json()
        return {x["metric"].get("strategy", x["metric"].get("model", "all")): float(x["value"][1])
                for x in r["data"]["result"]}
    except Exception:
        return {}


def trace_link(trace_id):
    pid = phoenix_project_id(PX, PROJECT)
    return f"{PX}/projects/{pid}/traces/{trace_id}" if pid else PX


S = strategies(GW)
names = [s["strategy"] for s in S.get("strategies", [])]
st.title("Routing control room")
if "error" in S:
    st.error(f"Gateway not reachable at {GW}: {S['error']}. Start it (see the guide, step 4).")
tab1, tab2, tab3, tab4 = st.tabs(["Try a request", "Compare live", "OTel metrics", "Phoenix traces"])

# ------------------------------------------------------------------ try
with tab1:
    c1, c2 = st.columns([2, 1])
    qpath = Path(RUN) / "queries.parquet"
    sample = pd.read_parquet(qpath).query("split == 'test'") if qpath.exists() else None
    with c2:
        strat = st.selectbox("Strategy", names, index=names.index(S.get("default")) if S.get("default") in names else 0)
        info = next((s for s in S.get("strategies", []) if s["strategy"] == strat), {})
        st.caption(f"Knob: {info.get('knob', '')}")
        use_knob = st.checkbox("Override knob", value=False)
        knob = st.number_input("Knob", value=float(info.get("default_knob") or 0.5), format="%.4f") if use_knob else None
        cache = st.checkbox("Put a semantic cache in front (cache+)")
    with c1:
        if sample is not None and st.button("Load a random test question"):
            st.session_state["prompt"] = sample.sample(1).iloc[0].prompt
        prompt = st.text_area("Prompt", st.session_state.get("prompt", "What is 17 * 23? Answer: <number>"), height=160)
    if st.button("Send", type="primary"):
        hdr = {"x-router-knob": str(knob)} if knob is not None else {}
        name = ("cache+" if cache else "") + strat
        t0 = time.perf_counter()
        r = httpx.post(f"{GW}/v1/chat/completions", headers=hdr, timeout=600,
                       json={"model": f"router:{name}", "max_tokens": 512, "messages": [{"role": "user", "content": prompt}]})
        if r.status_code != 200:
            st.error(r.text)
        else:
            js = r.json()
            rl = js["routerlab"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Answered by", rl["answered_by"])
            m2.metric("Cost", f"${rl['cost_usd']:.6f}")
            m3.metric("Saved vs strongest", f"{100 * (1 - rl['cost_usd'] / max(rl['baseline_cost_usd'], 1e-12)):.0f}%")
            m4.metric("Latency", f"{time.perf_counter() - t0:.2f}s")
            st.markdown(f"**Answer:** {js['choices'][0]['message']['content']}")
            st.write({"calls": rl["calls"], "escalated": rl["escalated"], "cache_hit": rl["cache_hit"]})
            st.json(rl["decision"])
            st.markdown(f"[Open this trace in Phoenix]({trace_link(rl['trace_id'])})")
            st.session_state["last_req"] = rl["request_id"]
    if st.session_state.get("last_req"):
        f1, f2, _ = st.columns([1, 1, 4])
        if f1.button("Mark correct"):
            httpx.post(f"{GW}/feedback", json={"request_id": st.session_state["last_req"], "correct": True})
        if f2.button("Mark wrong"):
            httpx.post(f"{GW}/feedback", json={"request_id": st.session_state["last_req"], "correct": False})

# ------------------------------------------------------------------ compare
with tab2:
    st.caption("Sends each test question once per strategy, scores the answer against the gold label, "
               "sends the score as feedback, and summarises. Traces and metrics land in Phoenix and Grafana.")
    defaults = [s for s in ["single/qwen2.5-32b-awq", "single/qwen2.5-1.5b", "random", "two-tier", "automix",
                            "frugalgpt", "predict/routellm-mf", "predict/embedllm"] if s in names]
    chosen = st.multiselect("Strategies", names, default=defaults)
    n = st.slider("Questions", 10, 500, 100, 10)
    qps = st.slider("Launch rate (req/s, 0 = max)", 0, 100, 20)
    if st.button("Run comparison", type="primary") and chosen:
        cmd = [sys.executable, str(ROOT / "scripts/loadgen.py"), "--run", RUN, "--gateway", GW,
               "--strategies", ",".join(chosen), "--n", str(n), "--qps", str(qps)]
        with st.spinner(f"{n * len(chosen)} requests..."):
            p = subprocess.run(cmd, capture_output=True, text=True)
        st.code(p.stdout[-4000:] or p.stderr[-4000:])
    files = sorted(Path(RUN).glob("live_compare_*.csv"))
    if files:
        df = pd.read_csv(files[-1])
        df = df[df.get("error").isna()] if "error" in df else df
        g = df.groupby("strategy").agg(accuracy=("correct", "mean"), usd_per_1k=("cost_usd", lambda x: 1000 * x.mean()),
                                       p95_s=("latency_s", lambda x: x.quantile(.95)), n=("qid", "count")).reset_index()
        st.subheader(f"Latest comparison ({files[-1].name})")
        fig = px.scatter(g, x="usd_per_1k", y="accuracy", text="strategy", size="n", log_x=True,
                         labels={"usd_per_1k": "$ per 1k requests (log)", "accuracy": "accuracy"})
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(g.sort_values("accuracy", ascending=False), hide_index=True, width="stretch")

# ------------------------------------------------------------------ OTel metrics
with tab3:
    W = WINDOW
    acc = prom(f'sum by (strategy) (increase(routerlab_feedback_total{{correct="true"}}[{W}])) / sum by (strategy) (increase(routerlab_feedback_total[{W}]))')
    cost = prom(f"sum by (strategy) (increase(routerlab_cost_usd_total[{W}]))")
    base = prom(f"sum by (strategy) (increase(routerlab_baseline_cost_usd_total[{W}]))")
    reqs = prom(f"sum by (strategy) (increase(routerlab_requests_total[{W}]))")
    p95 = prom(f"histogram_quantile(0.95, sum by (le, strategy) (increase(routerlab_request_duration_seconds_bucket[{W}])))")
    calls = prom(f"sum by (strategy) (increase(routerlab_llm_calls_total[{W}]))")
    if not reqs:
        st.info(f"No OTel metrics in Prometheus at {PROM} for the last {W}. Is the collector running and traffic flowing?")
    else:
        tb = pd.DataFrame({"requests": reqs, "accuracy": acc, "spend_usd": cost, "baseline_usd": base,
                           "p95_latency_s": p95, "llm_calls": calls}).fillna(0)
        tb["usd_per_1k"] = 1000 * tb.spend_usd / tb.requests.clip(lower=1)
        tb["saved_vs_strongest_%"] = 100 * (1 - tb.spend_usd / tb.baseline_usd.clip(lower=1e-12))
        tb["calls_per_request"] = tb.llm_calls / tb.requests.clip(lower=1)
        tb = tb.reset_index(names="strategy").sort_values("accuracy", ascending=False)
        k1, k2, k3 = st.columns(3)
        k1.metric("Requests", f"{tb.requests.sum():.0f}")
        k2.metric("Spend", f"${tb.spend_usd.sum():.4f}")
        k3.metric("Saved vs strongest", f"{100 * (1 - tb.spend_usd.sum() / max(tb.baseline_usd.sum(), 1e-12)):.1f}%")
        fig = px.scatter(tb, x="usd_per_1k", y="accuracy", text="strategy", size="requests", log_x=True,
                         labels={"usd_per_1k": "$ per 1k requests (log)"})
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(tb, hide_index=True, width="stretch")
        try:
            end = time.time()
            r = httpx.get(f"{PROM}/api/v1/query_range", params={
                "query": "3600 * sum by (strategy) (rate(routerlab_cost_usd_total[1m]))", "start": end - 3600,
                "end": end, "step": 15}).json()["data"]["result"]
            ts = pd.concat([pd.DataFrame(x["values"], columns=["t", "usd_per_hour"]).assign(strategy=x["metric"]["strategy"])
                            for x in r]) if r else pd.DataFrame()
            if len(ts):
                ts["t"] = pd.to_datetime(ts.t, unit="s")
                ts["usd_per_hour"] = ts.usd_per_hour.astype(float)
                st.plotly_chart(px.line(ts, x="t", y="usd_per_hour", color="strategy", title="Spend rate ($/h)"),
                                width="stretch")
        except Exception as e:
            st.caption(f"time series unavailable: {e}")

# ------------------------------------------------------------------ Phoenix
with tab4:
    try:
        r = httpx.get(f"{PX}/v1/projects/{PROJECT}/spans", params={"limit": 1000}, timeout=20).json()
        rows = [s for s in r.get("data", []) if s["name"] == "route"]
    except Exception as e:
        rows = []
        st.error(f"Phoenix not reachable at {PX}: {e}")
    if rows:
        df = pd.DataFrame([{"time": s["start_time"], "strategy": s["attributes"].get("routerlab.strategy"),
                            "answered_by": s["attributes"].get("routerlab.answered_by"),
                            "cost_usd": s["attributes"].get("routerlab.cost_usd"),
                            "savings_usd": s["attributes"].get("routerlab.savings_usd"),
                            "escalated": s["attributes"].get("routerlab.escalated"),
                            "span_id": s["context"]["span_id"], "trace_id": s["context"]["trace_id"],
                            "prompt": (s["attributes"].get("input.value") or "")[:80]} for s in rows])
        try:
            ann = httpx.get(f"{PX}/v1/projects/{PROJECT}/span_annotations",
                            params=[("span_ids", x) for x in df.span_id[:500]] + [("limit", 1000)], timeout=20).json()
            lab = {a["span_id"]: a["result"]["label"] for a in ann.get("data", []) if a["name"] == "correctness"}
            df["correctness"] = df.span_id.map(lab)
        except Exception:
            pass
        df["trace"] = df.trace_id.map(trace_link)
        st.caption(f"{len(df)} most recent routed requests (Phoenix computes per-trace cost from the GPU-measured prices).")
        by = df.groupby("strategy").agg(n=("span_id", "count"), cost_usd=("cost_usd", "sum"),
                                        savings_usd=("savings_usd", "sum")).reset_index()
        st.dataframe(by, hide_index=True, width="stretch")
        st.dataframe(df.drop(columns=["span_id"]), hide_index=True, width="stretch",
                     column_config={"trace": st.column_config.LinkColumn("Phoenix trace", display_text="open")})
    else:
        st.info("No routed traces yet.")
