# telemetry.py
import os
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
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from app.core.config import settings


def setup_otel(app, engine=None):
    if not settings.ENV == "production":
        return

    resource = Resource.create(
        {
            "service.name": settings.OTEL_SERVICE_NAME,
            "service.version": settings.OTEL_SERVICE_VERSION,
            "deployment.environment": settings.ENV,
        }
    )

    # Sampling: 100% by default; drop to 0.2 (20%) if you need to cut cost/noise
    sampler = ParentBased(TraceIdRatioBased(float(os.getenv("OTEL_SAMPLE", "1.0"))))

    # Traces
    provider = TracerProvider(resource=resource, sampler=sampler)
    trace.set_tracer_provider(provider)
    span_exporter = OTLPSpanExporter(
        endpoint=settings.BETTERSTACK_HOST,
        headers={
            "Authorization": f"Bearer {settings.BETTERSTACK_OPENTELEMETRY_API_KEY}"
        },
    )
    provider.add_span_processor(BatchSpanProcessor(span_exporter))

    # Metrics (optional but recommended)
    metric_exporter = OTLPMetricExporter(
        endpoint=settings.BETTERSTACK_HOST.rstrip(
            "/traces"
        )  # many OTLP backends accept same base
    )
    reader = PeriodicExportingMetricReader(metric_exporter)
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)

    # Auto-instrumentation
    FastAPIInstrumentor.instrument_app(app)
    RequestsInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    if engine is not None:
        SQLAlchemyInstrumentor().instrument(engine=engine)

    # Put trace/span ids into your Python logs
    LoggingInstrumentor().instrument(set_logging_format=True)
