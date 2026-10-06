r"""OpenTelemetry for the routing gateway: traces -> Arize Phoenix, metrics -> Prometheus/Grafana.

Data path (docker compose in monitoring/):
    gateway --OTLP/HTTP:4318--> OpenTelemetry Collector --> Phoenix (traces, :6006)
                                                       \--> Prometheus exporter :8889 --> Prometheus --> Grafana
Set OTEL_EXPORTER_OTLP_ENDPOINT (default http://localhost:4318). Without a collector you can
point traces straight at Phoenix (http://localhost:6006) - metrics are then dropped.

Spans follow OpenInference semantic conventions so Phoenix renders them as LLM traces:
  route (CHAIN)            one per request: strategy, knob, chosen model, $ cost, $ baseline, savings
    router.decide (CHAIN)  the routing decision and its scores
    llm <model> (LLM)      one per model call: llm.model_name, token counts, messages, latency
    verify / score / vote  cascade & ensemble steps
Phoenix computes $ from llm.model_name + token counts using its model price table;
`sync_phoenix_prices` writes the $/token measured on YOUR GPUs into that table, so Phoenix's
cost column matches the gateway's own cost metric (also stored on spans as routerlab.cost_usd).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import contextmanager

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

PROJECT = os.environ.get("PHOENIX_PROJECT_NAME", "llm-routing")
_tracer = None
M = None   # metric instruments (namespace object)

LAT_BUCKETS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
DECIDE_BUCKETS = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25]


class _Instruments:
    def __init__(self, meter):
        c = meter.create_counter
        h = meter.create_histogram
        self.requests = c("routerlab.requests", description="Requests answered, by strategy and answering model")
        self.llm_calls = c("routerlab.llm.calls", description="LLM calls made (cascades/ensembles make several)")
        self.cost = c("routerlab.cost.usd", description="Measured $ of the LLM calls made")
        self.baseline = c("routerlab.baseline.cost.usd", description="$ if the request had gone to the strongest model")
        self.tokens = c("routerlab.llm.tokens", description="Tokens processed")
        self.escalations = c("routerlab.escalations", description="Cascade escalations to a bigger model")
        self.cache_hits = c("routerlab.cache.hits", description="Semantic-cache hits (no LLM call)")
        self.feedback = c("routerlab.feedback", description="Correctness feedback (label correct=true|false)")
        self.request_duration = h("routerlab.request.duration", unit="s", description="End-to-end latency")
        self.llm_duration = h("routerlab.llm.duration", unit="s", description="Per-model call latency")
        self.decide_duration = h("routerlab.router.decide.duration", unit="s", description="Router overhead")


def init(service_name="routerlab-gateway", endpoint: str | None = None, export_interval_ms=5000):
    """Idempotent. Returns (tracer, instruments)."""
    global _tracer, M
    if _tracer is not None:
        return _tracer, M
    endpoint = (endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")).rstrip("/")
    res = Resource.create({"service.name": service_name, "openinference.project.name": PROJECT})
    tp = TracerProvider(resource=res)
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"),
                                             max_queue_size=50_000, max_export_batch_size=1024,
                                             schedule_delay_millis=1000))
    # Deliberately NOT registered as the global provider: frameworks that auto-emit spans
    # (recent FastAPI emits fastapi.* spans for every endpoint) would otherwise flood Phoenix.
    if os.environ.get("ROUTERLAB_METRICS", "1") == "1":
        reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics"),
                                               export_interval_millis=export_interval_ms)
        views = [View(instrument_name="routerlab.router.decide.duration",
                      aggregation=ExplicitBucketHistogramAggregation(DECIDE_BUCKETS)),
                 View(instrument_name="routerlab.request.duration", aggregation=ExplicitBucketHistogramAggregation(LAT_BUCKETS)),
                 View(instrument_name="routerlab.llm.duration", aggregation=ExplicitBucketHistogramAggregation(LAT_BUCKETS))]
        metrics.set_meter_provider(MeterProvider(resource=res, metric_readers=[reader], views=views))
    meter = metrics.get_meter("routerlab")
    M = _Instruments(meter)
    _gpu_gauges(meter)
    _tracer = tp.get_tracer("routerlab")
    return _tracer, M


def _gpu_gauges(meter):
    """GPU util / memory / power via NVML as OTel observable gauges (DCGM exporter also works)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
    except Exception:
        return
    hs = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]

    def util(_):
        return [Observation(pynvml.nvmlDeviceGetUtilizationRates(h).gpu, {"gpu": str(i)}) for i, h in enumerate(hs)]

    def mem(_):
        return [Observation(pynvml.nvmlDeviceGetMemoryInfo(h).used, {"gpu": str(i)}) for i, h in enumerate(hs)]

    def power(_):
        return [Observation(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, {"gpu": str(i)}) for i, h in enumerate(hs)]

    meter.create_observable_gauge("routerlab.gpu.utilization", [util], description="GPU utilisation %")
    meter.create_observable_gauge("routerlab.gpu.memory.used", [mem], unit="By", description="GPU memory used")
    meter.create_observable_gauge("routerlab.gpu.power", [power], description="GPU power draw (W)")


# ------------------------------------------------------------------ span helpers
def _set(span, attrs: dict):
    for k, v in attrs.items():
        if v is None:
            continue
        if isinstance(v, (dict, list)) and not (isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v)):
            v = json.dumps(v, default=str)
        span.set_attribute(k, v)


@contextmanager
def chain_span(name: str, **attrs):
    with _tracer.start_as_current_span(name) as sp:
        _set(sp, {"openinference.span.kind": "CHAIN", **attrs})
        yield sp


@contextmanager
def llm_span(model: str, messages: list[dict], params: dict, **attrs):
    with _tracer.start_as_current_span(f"llm {model}") as sp:
        a = {"openinference.span.kind": "LLM", "llm.model_name": model, "llm.provider": "vllm",
             "llm.system": "vllm", "llm.invocation_parameters": params, "input.value": messages[-1]["content"],
             **attrs}
        for i, m in enumerate(messages):
            a[f"llm.input_messages.{i}.message.role"] = m["role"]
            a[f"llm.input_messages.{i}.message.content"] = m["content"]
        _set(sp, a)
        yield sp


def record_llm_result(sp, text: str, n_in: int, n_out: int, cost: float):
    _set(sp, {"output.value": text, "llm.output_messages.0.message.role": "assistant",
              "llm.output_messages.0.message.content": text, "llm.token_count.prompt": n_in,
              "llm.token_count.completion": n_out, "llm.token_count.total": n_in + n_out,
              "routerlab.cost_usd": cost, "llm.cost.total": cost})


def span_id_hex(sp) -> str:
    return format(sp.get_span_context().span_id, "016x")


def trace_id_hex(sp) -> str:
    return format(sp.get_span_context().trace_id, "032x")


# ------------------------------------------------------------------ Phoenix integration
def sync_phoenix_prices(phoenix_url: str, prices_per_mtok: dict[str, tuple[float, float]]):
    """Create/refresh one Phoenix model-price entry per pool model: name -> (input $/1M, output $/1M)."""
    import httpx

    gql = f"{phoenix_url.rstrip('/')}/graphql"
    try:
        existing = httpx.post(gql, json={"query": "{ generativeModels(first: 1000) { edges { node { id name } } } }"},
                              timeout=10).json()["data"]["generativeModels"]["edges"]
    except Exception as e:
        print(f"[phoenix] price sync skipped ({e})")
        return
    ids = {e["node"]["name"]: e["node"]["id"] for e in existing}
    for model, (pin, pout) in prices_per_mtok.items():
        name = f"routerlab/{model}"
        costs = [{"tokenType": "input", "kind": "PROMPT", "costPerMillionTokens": float(pin)},
                 {"tokenType": "output", "kind": "COMPLETION", "costPerMillionTokens": float(pout)}]
        if name in ids:
            q = ("mutation($i: UpdateModelMutationInput!) { updateModel(input: $i) { __typename } }",
                 {"id": ids[name], "name": name, "namePattern": f"^{re.escape(model)}$", "provider": "vllm", "costs": costs})
        else:
            q = ("mutation($i: CreateModelMutationInput!) { createModel(input: $i) { __typename } }",
                 {"name": name, "namePattern": f"^{re.escape(model)}$", "provider": "vllm", "costs": costs})
        r = httpx.post(gql, json={"query": q[0], "variables": {"i": q[1]}}, timeout=10).json()
        if "errors" in r:
            print(f"[phoenix] price sync for {model} failed: {r['errors'][0]['message']}")
    print(f"[phoenix] synced GPU-measured prices for {len(prices_per_mtok)} models")


class PhoenixAnnotator:
    """Posts correctness feedback as span annotations so Phoenix shows quality next to cost."""

    def __init__(self, phoenix_url: str | None):
        self.url = phoenix_url.rstrip("/") if phoenix_url else None
        self._buf, self._lock = [], threading.Lock()
        if self.url:
            threading.Thread(target=self._loop, daemon=True).start()

    def add(self, span_id: str, correct: bool, strategy: str, model: str):
        if not self.url:
            return
        with self._lock:
            self._buf.append({"span_id": span_id, "name": "correctness", "annotator_kind": "CODE",
                              "result": {"label": "correct" if correct else "incorrect", "score": float(correct)},
                              "metadata": {"strategy": strategy, "model": model}})

    def _loop(self):
        import httpx

        while True:
            time.sleep(2)
            with self._lock:
                batch, self._buf = self._buf, []
            if batch:
                try:
                    httpx.post(f"{self.url}/v1/span_annotations", json={"data": batch}, timeout=10)
                except Exception as e:  # never break serving because of telemetry
                    print(f"[phoenix] annotation post failed: {e}")
