"""Generates dashboards/llm_routing_otel.json from the OTel metric names the gateway emits.
Run `python monitoring/grafana/build_dashboard.py` after changing panels; Grafana reloads it."""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prom"}
S = 'strategy=~"$strategy"'
panels, _y = [], [0]


def row(title):
    panels.append({"type": "row", "title": title, "collapsed": False, "gridPos": {"x": 0, "y": _y[0], "w": 24, "h": 1},
                   "id": len(panels) + 1, "panels": []})
    _y[0] += 1


def add(title, targets, kind="timeseries", x=0, w=12, h=8, unit=None, stack=False, desc="", decimals=None,
        extra=None, newline=True):
    fc = {"defaults": {}, "overrides": []}
    if unit:
        fc["defaults"]["unit"] = unit
    if decimals is not None:
        fc["defaults"]["decimals"] = decimals
    if stack:
        fc["defaults"]["custom"] = {"stacking": {"mode": "percent" if stack == "percent" else "normal"}, "fillOpacity": 60,
                                    "lineWidth": 0}
    p = {"type": kind, "title": title, "description": desc, "datasource": DS, "fieldConfig": fc,
         "gridPos": {"x": x, "y": _y[0], "w": w, "h": h}, "id": len(panels) + 1, "options": {},
         "targets": [{"refId": chr(65 + i), "expr": e, "legendFormat": l, **(o or {})}
                     for i, (e, l, *rest) in enumerate(targets) for o in [rest[0] if rest else None]]}
    if extra:
        p.update(extra)
    panels.append(p)
    if newline:
        _y[0] += h


R = "[$__rate_interval]"
row("Is routing paying off?  (vs. sending everything to the strongest model)")
add("Cost saved", [(f"100 * (1 - sum(increase(routerlab_cost_usd_total{{{S}}}[$__range])) / sum(increase(routerlab_baseline_cost_usd_total{{{S}}}[$__range])))", "")],
    "stat", 0, 6, 5, "percent", decimals=1, newline=False, desc="1 - routed $ / $ if every request had gone to the strongest model")
add("Live accuracy", [(f'sum(increase(routerlab_feedback_total{{{S},correct="true"}}[$__range])) / sum(increase(routerlab_feedback_total{{{S}}}[$__range]))', "")],
    "stat", 6, 6, 5, "percentunit", decimals=1, newline=False, desc="From /feedback (loadgen scores every answer)")
add("Spend (range)", [(f"sum(increase(routerlab_cost_usd_total{{{S}}}[$__range]))", "")], "stat", 12, 6, 5, "currencyUSD",
    decimals=4, newline=False)
add("Requests (range)", [(f"sum(increase(routerlab_requests_total{{{S}}}[$__range]))", "")], "stat", 18, 6, 5, "short",
    decimals=0)
add("Spend rate: routed vs always-strongest", [(f"3600 * sum(rate(routerlab_cost_usd_total{{{S}}}{R}))", "routed $/h"),
                                                (f"3600 * sum(rate(routerlab_baseline_cost_usd_total{{{S}}}{R}))", "always-strongest $/h")],
    w=24, h=7, unit="currencyUSD")

row("Strategy comparison")
q_tbl = {"format": "table", "instant": True}
add("Scorecard by strategy (range)", [
    (f'sum by (strategy) (increase(routerlab_feedback_total{{{S},correct="true"}}[$__range])) / sum by (strategy) (increase(routerlab_feedback_total{{{S}}}[$__range]))', "", q_tbl),
    (f"1000 * sum by (strategy) (increase(routerlab_cost_usd_total{{{S}}}[$__range])) / sum by (strategy) (increase(routerlab_requests_total{{{S}}}[$__range]))", "", q_tbl),
    (f"100 * (1 - sum by (strategy) (increase(routerlab_cost_usd_total{{{S}}}[$__range])) / sum by (strategy) (increase(routerlab_baseline_cost_usd_total{{{S}}}[$__range])))", "", q_tbl),
    (f"histogram_quantile(0.95, sum by (le, strategy) (increase(routerlab_request_duration_seconds_bucket{{{S}}}[$__range])))", "", q_tbl),
    (f"sum by (strategy) (increase(routerlab_llm_calls_total{{{S}}}[$__range])) / sum by (strategy) (increase(routerlab_requests_total{{{S}}}[$__range]))", "", q_tbl),
    (f"sum by (strategy) (increase(routerlab_requests_total{{{S}}}[$__range]))", "", q_tbl)],
    "table", 0, 24, 10, extra={
        "transformations": [{"id": "merge"}, {"id": "organize", "options": {"excludeByName": {"Time": True}, "renameByName": {
            "Value #A": "accuracy", "Value #B": "$ per 1k requests", "Value #C": "saved vs strongest %",
            "Value #D": "p95 latency (s)", "Value #E": "LLM calls / request", "Value #F": "requests"}}},
                            {"id": "sortBy", "options": {"sort": [{"field": "accuracy", "desc": True}]}}],
        "fieldConfig": {"defaults": {"decimals": 3, "custom": {"cellOptions": {"type": "color-text"}}}, "overrides": [
            {"matcher": {"id": "byName", "options": "accuracy"}, "properties": [{"id": "unit", "value": "percentunit"},
                {"id": "custom.cellOptions", "value": {"type": "gauge", "mode": "lcd"}}, {"id": "min", "value": 0}, {"id": "max", "value": 1}]},
            {"matcher": {"id": "byName", "options": "$ per 1k requests"}, "properties": [{"id": "unit", "value": "currencyUSD"}, {"id": "decimals", "value": 4}]},
            {"matcher": {"id": "byName", "options": "saved vs strongest %"}, "properties": [{"id": "unit", "value": "percent"}, {"id": "decimals", "value": 1}]},
            {"matcher": {"id": "byName", "options": "requests"}, "properties": [{"id": "decimals", "value": 0}]}]}})
add("Accuracy vs $ per 1k requests (each point = a strategy; up-left is better)", [
    (f"1000 * sum by (strategy) (increase(routerlab_cost_usd_total{{{S}}}[$__range])) / sum by (strategy) (increase(routerlab_requests_total{{{S}}}[$__range]))", "", q_tbl),
    (f'sum by (strategy) (increase(routerlab_feedback_total{{{S},correct="true"}}[$__range])) / sum by (strategy) (increase(routerlab_feedback_total{{{S}}}[$__range]))', "", q_tbl)],
    "xychart", 0, 24, 11, extra={
        "transformations": [{"id": "merge"}, {"id": "organize", "options": {"excludeByName": {"Time": True},
                            "renameByName": {"Value #A": "$ per 1k", "Value #B": "accuracy"}}}],
        "options": {"mapping": "manual", "series": [{"x": "$ per 1k", "y": "accuracy", "name": "strategies",
                                                      "pointSize": {"fixed": 9}}],
                    "tooltip": {"mode": "single"}, "legend": {"showLegend": False}},
        "fieldConfig": {"defaults": {"custom": {"show": "points", "pointSize": {"fixed": 9}}}, "overrides": []}})
add("Accuracy over time", [(f'sum by (strategy) (rate(routerlab_feedback_total{{{S},correct="true"}}[5m])) / sum by (strategy) (rate(routerlab_feedback_total{{{S}}}[5m]))', "{{strategy}}")],
    w=12, unit="percentunit", newline=False)
add("$ per request over time", [(f"sum by (strategy) (rate(routerlab_cost_usd_total{{{S}}}{R})) / sum by (strategy) (rate(routerlab_requests_total{{{S}}}{R}))", "{{strategy}}")],
    x=12, w=12, unit="currencyUSD")
add("Latency p50 / p95 by strategy", [(f"histogram_quantile(0.95, sum by (le, strategy) (rate(routerlab_request_duration_seconds_bucket{{{S}}}{R})))", "p95 {{strategy}}"),
                                      (f"histogram_quantile(0.5, sum by (le, strategy) (rate(routerlab_request_duration_seconds_bucket{{{S}}}{R})))", "p50 {{strategy}}")],
    w=12, unit="s", newline=False)
add("Requests / s by strategy", [(f"sum by (strategy) (rate(routerlab_requests_total{{{S}}}{R}))", "{{strategy}}")], x=12, w=12, unit="reqps")

row("Routing behaviour")
add("Which model answers? (share of answers)", [(f"sum by (model) (rate(routerlab_requests_total{{{S}}}{R}))", "{{model}}")],
    w=12, stack="percent", unit="percentunit", newline=False)
add("Escalation rate (cascades)", [(f"sum by (strategy) (rate(routerlab_escalations_total{{{S}}}{R})) / sum by (strategy) (rate(routerlab_requests_total{{{S}}}{R}))", "{{strategy}}")],
    x=12, w=12, unit="percentunit")
add("LLM calls per request", [(f"sum by (strategy) (rate(routerlab_llm_calls_total{{{S}}}{R})) / sum by (strategy) (rate(routerlab_requests_total{{{S}}}{R}))", "{{strategy}}")],
    w=8, newline=False)
add("Semantic-cache hit rate", [(f"sum by (strategy) (rate(routerlab_cache_hits_total{{{S}}}{R})) / sum by (strategy) (rate(routerlab_requests_total{{{S}}}{R}))", "{{strategy}}")],
    x=8, w=8, unit="percentunit", newline=False)
add("Router decision overhead p95", [(f"histogram_quantile(0.95, sum by (le, strategy) (rate(routerlab_router_decide_duration_seconds_bucket{{{S}}}{R})))", "{{strategy}}")],
    x=16, w=8, unit="s")

row("Models & GPUs")
add("$ by model", [("3600 * sum by (model) (rate(routerlab_cost_usd_total[$__rate_interval]))", "{{model}}")], w=8, unit="currencyUSD",
    stack=True, newline=False)
add("Model call latency p95", [("histogram_quantile(0.95, sum by (le, model) (rate(routerlab_llm_duration_seconds_bucket[$__rate_interval])))", "{{model}}")],
    x=8, w=8, unit="s", newline=False)
add("Tokens / s", [('sum by (model) (rate(routerlab_llm_tokens_total{direction="output"}[$__rate_interval]))', "out {{model}}"),
                   ('sum by (model) (rate(routerlab_llm_tokens_total{direction="input"}[$__rate_interval]))', "in {{model}}")],
    x=16, w=8)
add("GPU utilisation", [("DCGM_FI_DEV_GPU_UTIL", "GPU {{gpu}} (DCGM)"), ("routerlab_gpu_utilization", "GPU {{gpu}} (NVML)")],
    w=8, unit="percent", newline=False)
add("GPU memory used", [("DCGM_FI_DEV_FB_USED * 1048576", "GPU {{gpu}} (DCGM)"), ("routerlab_gpu_memory_used_bytes", "GPU {{gpu}} (NVML)")],
    x=8, w=8, unit="bytes", newline=False)
add("GPU power", [("DCGM_FI_DEV_POWER_USAGE", "GPU {{gpu}} (DCGM)"), ("routerlab_gpu_power", "GPU {{gpu}} (NVML)")], x=16, w=8, unit="watt")
add("vLLM queue: running / waiting", [("sum by (model_name) (vllm:num_requests_running)", "running {{model_name}}"),
                                      ("sum by (model_name) (vllm:num_requests_waiting)", "waiting {{model_name}}")], w=8, newline=False)
add("vLLM KV-cache usage", [("vllm:gpu_cache_usage_perc", "{{model_name}}"), ("vllm:kv_cache_usage_perc", "{{model_name}}")],
    x=8, w=8, unit="percentunit", newline=False)
add("vLLM time to first token p95", [("histogram_quantile(0.95, sum by (le, model_name) (rate(vllm:time_to_first_token_seconds_bucket[$__rate_interval])))", "{{model_name}}")],
    x=16, w=8, unit="s")

dash = {
    "title": "LLM Routing (OTel)", "uid": "routerlab-otel", "schemaVersion": 39, "refresh": "10s",
    "time": {"from": "now-30m", "to": "now"}, "tags": ["routing", "otel"], "panels": panels,
    "links": [{"title": "Traces in Arize Phoenix", "type": "link", "url": "http://localhost:6006", "targetBlank": True,
               "icon": "external link"},
              {"title": "Routing control room", "type": "link", "url": "http://localhost:8501", "targetBlank": True}],
    "templating": {"list": [{"name": "strategy", "label": "Strategy", "type": "query", "datasource": DS,
                             "query": {"query": "label_values(routerlab_requests_total, strategy)", "refId": "S"},
                             "definition": "label_values(routerlab_requests_total, strategy)",
                             "multi": True, "includeAll": True, "allValue": ".*", "current": {"text": "All", "value": "$__all"},
                             "refresh": 2, "sort": 1}]},
}
out = Path(__file__).parent / "dashboards" / "llm_routing_otel.json"
out.write_text(json.dumps(dash, indent=1))
print(f"{len(panels)} panels -> {out}")
