# app/utils/observability.py
from __future__ import annotations
import contextlib, time
from dataclasses import dataclass
from typing import Optional, AsyncIterator, Iterator, Dict, Any

from fastapi import FastAPI
from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.botocore import BotocoreInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from app.core.config import settings
import logging

logger = logging.getLogger("app.obs")


@dataclass
class ObsConfig:
    env: str = settings.ENV
    service_name: str = settings.OTEL_SERVICE_NAME
    service_version: str = settings.OTEL_SERVICE_VERSION
    sample_ratio: str | float = settings.OTEL_SAMPLE_RATIO
    enable_metrics: str | bool = settings.OTEL_ENABLE_METRICS

    betterstack_host: Optional[str] = (
        settings.BETTERSTACK_HOST
    )  # e.g. https://in-otel.betterstack.com
    betterstack_api_key: Optional[str] = settings.BETTERSTACK_API_KEY


_cfg = ObsConfig()
_tracer = trace.get_tracer(__name__)
_meter = None
_step_hist = None
_reuse_counter = None


def _to_float(x, default=1.0) -> float:
    try:
        v = float(x)
        return max(0.0, min(1.0, v))
    except Exception:
        return default


def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _join(base: str, path: str) -> str:
    base = base.rstrip("/")
    path = path if path.startswith("/") else "/" + path
    return base + path


def _build_resource() -> Resource:
    return Resource.create(
        {
            "service.name": _cfg.service_name,
            "service.version": _cfg.service_version,
            "deployment.environment": _cfg.env,
        }
    )


def _build_exporters(enable_metrics: bool):
    # Prod → Better Stack using BETTERSTACK_HOST
    if _cfg.env.lower() == "production" and _cfg.betterstack_host:
        traces_ep = _join(_cfg.betterstack_host, "/v1/traces")
        metrics_ep = _join(_cfg.betterstack_host, "/metrics")
        headers = {"Authorization": f"Bearer {_cfg.betterstack_api_key}"}
        span_exp = OTLPSpanExporter(endpoint=traces_ep, headers=headers)
        metric_exp = (
            OTLPMetricExporter(endpoint=metrics_ep, headers=headers)
            if enable_metrics
            else None
        )
        logger.info("Prod OTEL traces_ep=%s metrics_ep=%s", traces_ep, metrics_ep)
        return span_exp, metric_exp

    # Dev/staging → local/remote collector via OTEL_EXPORTER_OTLP_ENDPOINT
    base = (_cfg.betterstack_host or "http://localhost:4318").rstrip("/")
    traces_ep = _join(base, "/v1/traces")
    metrics_ep = _join(base, "/metrics")
    headers = {"Authorization": f"Bearer {_cfg.betterstack_api_key}"}
    span_exp = OTLPSpanExporter(endpoint=traces_ep)
    metric_exp = (
        OTLPMetricExporter(endpoint=metrics_ep, headers=headers)
        if enable_metrics
        else None
    )
    logger.info("Dev OTEL traces_ep=%s metrics_ep=%s", traces_ep, metrics_ep)
    return span_exp, metric_exp


def setup_observability(app: FastAPI, *, sqlalchemy_engine=None) -> None:
    sample_ratio = _to_float(_cfg.sample_ratio, 1.0)
    enable_metrics = _to_bool(_cfg.enable_metrics)

    resource = _build_resource()
    tp = TracerProvider(
        resource=resource, sampler=ParentBased(TraceIdRatioBased(sample_ratio))
    )
    span_exporter, metric_exporter = _build_exporters(enable_metrics)
    tp.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tp)

    if enable_metrics and metric_exporter:
        mp = MeterProvider(
            resource=resource,
            metric_readers=[PeriodicExportingMetricReader(metric_exporter)],
        )
        metrics.set_meter_provider(mp)

    global _meter, _step_hist, _reuse_counter
    _meter = metrics.get_meter("app.obs")
    if enable_metrics:
        _step_hist = _meter.create_histogram(
            "app.step.duration", unit="ms", description="Business step duration"
        )
        _reuse_counter = _meter.create_counter(
            "app.step.reused_total", description="Cache/reuse hits by step"
        )

    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="^/healthz$|^/metrics$|^/docs$",
    )
    RequestsInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    BotocoreInstrumentor().instrument()
    if sqlalchemy_engine is not None:
        SQLAlchemyInstrumentor().instrument(engine=sqlalchemy_engine)
    LoggingInstrumentor().instrument(set_logging_format=True)


# helpers
@contextlib.contextmanager
def step(name: str, **attrs: Any) -> Iterator[None]:
    start = time.perf_counter()
    with _tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(f"app.{k}", v)
        try:
            yield
            span.set_attribute("app.success", True)
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("app.success", False)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
        finally:
            if _step_hist:
                _step_hist.record((time.perf_counter() - start) * 1000, {"step": name})


@contextlib.asynccontextmanager
async def astep(name: str, **attrs: Any) -> AsyncIterator[None]:
    start = time.perf_counter()
    with _tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(f"app.{k}", v)
        try:
            yield
            span.set_attribute("app.success", True)
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("app.success", False)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
        finally:
            if _step_hist:
                _step_hist.record((time.perf_counter() - start) * 1000, {"step": name})


def mark_reuse(step_name: str) -> None:
    if _reuse_counter:
        _reuse_counter.add(1, {"step": step_name})
