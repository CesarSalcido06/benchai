"""
OpenTelemetry Monitoring for Multi-Agent System

Provides distributed tracing, metrics, and logging across agents:
- Request tracing with span context propagation
- Agent performance metrics
- Task routing analytics
- Error tracking and alerting
"""

import os
import time
import functools
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Try to import OTLP exporter (optional)
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False


class AgentTelemetry:
    """
    OpenTelemetry instrumentation for BenchAI multi-agent system.

    Features:
    - Distributed tracing across agent calls
    - Performance metrics (latency, throughput, errors)
    - Task routing analytics
    - Agent health monitoring
    """

    def __init__(
        self,
        service_name: str = "benchai",
        otlp_endpoint: Optional[str] = None,
        enable_console: bool = True
    ):
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        self.enable_console = enable_console
        self.propagator = TraceContextTextMapPropagator()

        # Initialize providers
        self._setup_tracing()
        self._setup_metrics()

        # Get tracer and meter
        self.tracer = trace.get_tracer(service_name, "1.0.0")
        self.meter = metrics.get_meter(service_name, "1.0.0")

        # Create metrics
        self._create_metrics()

    def _setup_tracing(self):
        """Configure trace provider with exporters."""
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: self.service_name,
            ResourceAttributes.SERVICE_VERSION: "3.5.0",
            "agent.type": "orchestrator",
            "deployment.environment": os.environ.get("ENVIRONMENT", "development")
        })

        provider = TracerProvider(resource=resource)

        # Add OTLP exporter if endpoint configured
        if self.otlp_endpoint and OTLP_AVAILABLE:
            otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        # Add console exporter for debugging
        if self.enable_console:
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(console_exporter))

        trace.set_tracer_provider(provider)

    def _setup_metrics(self):
        """Configure metrics provider."""
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: self.service_name,
        })

        readers = []

        # Add OTLP metric exporter if available
        if self.otlp_endpoint and OTLP_AVAILABLE:
            otlp_reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=self.otlp_endpoint),
                export_interval_millis=60000  # Export every minute
            )
            readers.append(otlp_reader)

        # Console exporter for debugging
        if self.enable_console:
            console_reader = PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=300000  # Every 5 minutes
            )
            readers.append(console_reader)

        if readers:
            provider = MeterProvider(resource=resource, metric_readers=readers)
            metrics.set_meter_provider(provider)

    def _create_metrics(self):
        """Create all metrics instruments."""
        # Request metrics
        self.request_counter = self.meter.create_counter(
            name="benchai.requests.total",
            description="Total number of requests",
            unit="1"
        )

        self.request_duration = self.meter.create_histogram(
            name="benchai.requests.duration",
            description="Request duration in milliseconds",
            unit="ms"
        )

        self.error_counter = self.meter.create_counter(
            name="benchai.errors.total",
            description="Total number of errors",
            unit="1"
        )

        # Agent metrics
        self.agent_task_counter = self.meter.create_counter(
            name="benchai.agent.tasks.total",
            description="Tasks routed to agents",
            unit="1"
        )

        self.agent_task_duration = self.meter.create_histogram(
            name="benchai.agent.tasks.duration",
            description="Agent task duration in milliseconds",
            unit="ms"
        )

        self.routing_confidence = self.meter.create_histogram(
            name="benchai.routing.confidence",
            description="Semantic routing confidence scores",
            unit="1"
        )

        # Memory/Knowledge metrics
        self.memory_operations = self.meter.create_counter(
            name="benchai.memory.operations",
            description="Memory store/search operations",
            unit="1"
        )

        self.zettelkasten_queries = self.meter.create_counter(
            name="benchai.zettelkasten.queries",
            description="Zettelkasten knowledge graph queries",
            unit="1"
        )

        # Experience metrics
        self.experience_counter = self.meter.create_counter(
            name="benchai.experience.recorded",
            description="Experiences recorded for learning",
            unit="1"
        )

        self.experience_success_rate = self.meter.create_histogram(
            name="benchai.experience.success_rate",
            description="Experience success scores",
            unit="1"
        )

    @contextmanager
    def trace_request(
        self,
        operation: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracing a request.

        Usage:
            with telemetry.trace_request("chat_completion", {"model": "general"}) as span:
                result = await process_request()
                span.set_attribute("tokens", result.tokens)
        """
        with self.tracer.start_as_current_span(operation) as span:
            start_time = time.time()

            # Set initial attributes
            if attributes:
                for key, value in attributes.items():
                    if value is not None:
                        span.set_attribute(key, str(value) if not isinstance(value, (int, float, bool)) else value)

            try:
                yield span
                span.set_status(Status(StatusCode.OK))

                # Record success metrics
                duration_ms = (time.time() - start_time) * 1000
                self.request_counter.add(1, {"operation": operation, "status": "success"})
                self.request_duration.record(duration_ms, {"operation": operation})

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)

                # Record error metrics
                self.error_counter.add(1, {"operation": operation, "error_type": type(e).__name__})
                self.request_counter.add(1, {"operation": operation, "status": "error"})

                raise

    def trace_agent_task(
        self,
        agent_id: str,
        task_type: str,
        confidence: float = 0.0
    ):
        """
        Decorator for tracing agent task execution.

        Usage:
            @telemetry.trace_agent_task("marunochiAI", "coding", 0.95)
            async def execute_task():
                ...
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(f"agent_task.{agent_id}") as span:
                    start_time = time.time()

                    span.set_attribute("agent.id", agent_id)
                    span.set_attribute("task.type", task_type)
                    span.set_attribute("routing.confidence", confidence)

                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))

                        # Record metrics
                        duration_ms = (time.time() - start_time) * 1000
                        self.agent_task_counter.add(1, {
                            "agent": agent_id,
                            "task_type": task_type,
                            "status": "success"
                        })
                        self.agent_task_duration.record(duration_ms, {"agent": agent_id})
                        self.routing_confidence.record(confidence, {"agent": agent_id})

                        return result

                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)

                        self.agent_task_counter.add(1, {
                            "agent": agent_id,
                            "task_type": task_type,
                            "status": "error"
                        })

                        raise

            return wrapper
        return decorator

    def record_memory_operation(self, operation: str, memory_type: str):
        """Record a memory operation."""
        self.memory_operations.add(1, {
            "operation": operation,
            "memory_type": memory_type
        })

    def record_zettelkasten_query(self, query_type: str, result_count: int):
        """Record a Zettelkasten query."""
        self.zettelkasten_queries.add(1, {
            "query_type": query_type,
            "result_count": str(result_count)
        })

    def record_experience(self, domain: str, success_score: float, agent: str):
        """Record an experience for learning."""
        self.experience_counter.add(1, {
            "domain": domain,
            "agent": agent
        })
        self.experience_success_rate.record(success_score, {
            "domain": domain,
            "agent": agent
        })

    def inject_context(self, carrier: Dict[str, str]):
        """Inject trace context into a carrier (e.g., HTTP headers)."""
        self.propagator.inject(carrier)

    def extract_context(self, carrier: Dict[str, str]):
        """Extract trace context from a carrier."""
        return self.propagator.extract(carrier)

    def get_current_trace_id(self) -> Optional[str]:
        """Get the current trace ID for correlation."""
        span = trace.get_current_span()
        if span:
            return format(span.get_span_context().trace_id, '032x')
        return None


# Global telemetry instance
_telemetry: Optional[AgentTelemetry] = None


def get_telemetry() -> AgentTelemetry:
    """Get or create the global telemetry instance."""
    global _telemetry
    if _telemetry is None:
        _telemetry = AgentTelemetry(
            service_name="benchai",
            enable_console=False  # Disable console output by default
        )
    return _telemetry


def init_telemetry(
    service_name: str = "benchai",
    otlp_endpoint: Optional[str] = None,
    enable_console: bool = False
) -> AgentTelemetry:
    """Initialize the global telemetry instance."""
    global _telemetry
    _telemetry = AgentTelemetry(
        service_name=service_name,
        otlp_endpoint=otlp_endpoint,
        enable_console=enable_console
    )
    return _telemetry


# Convenience decorators
def traced(operation: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Decorator to trace a function.

    Usage:
        @traced("process_query", {"endpoint": "/v1/chat"})
        async def process_query(request):
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            telemetry = get_telemetry()
            with telemetry.trace_request(operation, attributes):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            telemetry = get_telemetry()
            with telemetry.trace_request(operation, attributes):
                return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Import asyncio for the decorator
import asyncio
