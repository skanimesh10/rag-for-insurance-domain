"""
Phase 4, layer 1: generic distributed tracing via OpenTelemetry.

FastAPI is auto-instrumented (one span per request, showing path,
status code, and latency for free). On top of that, each pipeline
stage (vector search, BM25 search, RRF fusion, rerank, generation)
gets its own manual child span with attributes -- so a single trace
for one /ask request shows the full waterfall: how long retrieval
took vs. reranking vs. the LLM call.

Exports over OTLP/HTTP, which any modern backend speaks (Jaeger
locally via docker-compose, or Datadog/New Relic/Grafana Tempo in
production -- only otel_exporter_otlp_endpoint changes, no code
changes).

Fails safe: if the collector is unreachable, span *export* fails
silently in the background (the OTLP exporter's own retry/timeout
logic handles that) -- it never raises into request-handling code.
"""
import logging
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from app.config import settings

logger = logging.getLogger("app.telemetry")

_tracer = trace.get_tracer(settings.otel_service_name)  # no-op tracer until setup_telemetry() runs


def setup_telemetry(app) -> None:
    """Call once at app startup (see lifespan in app/main.py)."""
    global _tracer

    if not settings.otel_enabled:
        logger.info("OpenTelemetry disabled via OTEL_ENABLED=false")
        return

    try:
        resource = Resource.create({"service.name": settings.otel_service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)

        _tracer = trace.get_tracer(settings.otel_service_name)
        logger.info("OpenTelemetry configured, exporting to %s", settings.otel_exporter_otlp_endpoint)
    except Exception as e:
        # Never let observability setup take the app down.
        logger.warning("Failed to configure OpenTelemetry, continuing without tracing: %s", e)


def get_tracer():
    return _tracer
