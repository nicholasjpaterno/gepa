"""OpenTelemetry tracing for GEPA."""

import functools
import time
from typing import Any, Callable, Dict, Optional, TypeVar, cast
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
import structlog

from ..config import ObservabilityConfig

logger = structlog.get_logger(__name__)

T = TypeVar('T')

# Global tracer
_tracer: Optional[trace.Tracer] = None


class TracingManager:
    """Manages OpenTelemetry tracing setup."""
    
    def __init__(self, config: ObservabilityConfig, service_name: str = "gepa"):
        self.config = config
        self.service_name = service_name
        self.tracer_provider: Optional[TracerProvider] = None
    
    def setup_tracing(
        self,
        otlp_endpoint: Optional[str] = None,
        service_version: str = "unknown"
    ) -> None:
        """Setup OpenTelemetry tracing."""
        if not self.config.enable_tracing:
            logger.info("Tracing disabled")
            return
        
        # Create resource
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": service_version,
        })
        
        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(self.tracer_provider)
        
        # Add span processor
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            span_processor = BatchSpanProcessor(otlp_exporter)
            self.tracer_provider.add_span_processor(span_processor)
        
        # Instrument libraries
        HTTPXClientInstrumentor().instrument()
        SQLAlchemyInstrumentor().instrument()
        
        logger.info(
            "Tracing setup complete",
            service_name=self.service_name,
            otlp_endpoint=otlp_endpoint
        )
    
    def shutdown(self) -> None:
        """Shutdown tracing."""
        if self.tracer_provider:
            self.tracer_provider.shutdown()
            logger.info("Tracing shutdown complete")


def get_tracer(name: str = "gepa") -> trace.Tracer:
    """Get a tracer instance."""
    global _tracer
    
    if _tracer is None:
        _tracer = trace.get_tracer(name)
    
    return _tracer


@contextmanager
def trace_operation(
    operation_name: str,
    attributes: Optional[Dict[str, Any]] = None,
    tracer: Optional[trace.Tracer] = None
):
    """Context manager for tracing operations."""
    if tracer is None:
        tracer = get_tracer()
    
    with tracer.start_as_current_span(operation_name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def trace_async_method(
    operation_name: Optional[str] = None,
    include_args: bool = False,
    include_result: bool = False
):
    """Decorator for tracing async methods."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()
            op_name = operation_name or f"{func.__module__}.{func.__qualname__}"
            
            with tracer.start_as_current_span(op_name) as span:
                # Add method attributes
                span.set_attribute("method.name", func.__name__)
                span.set_attribute("method.module", func.__module__)
                
                # Add arguments if requested
                if include_args:
                    for i, arg in enumerate(args[1:]):  # Skip self
                        span.set_attribute(f"args.{i}", str(arg))
                    
                    for key, value in kwargs.items():
                        span.set_attribute(f"kwargs.{key}", str(value))
                
                try:
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    
                    # Record duration
                    duration = time.time() - start_time
                    span.set_attribute("method.duration_seconds", duration)
                    
                    # Add result if requested
                    if include_result and result is not None:
                        span.set_attribute("method.result", str(result)[:1000])  # Truncate
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    return decorator


def trace_method(
    operation_name: Optional[str] = None,
    include_args: bool = False,
    include_result: bool = False
):
    """Decorator for tracing sync methods."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            tracer = get_tracer()
            op_name = operation_name or f"{func.__module__}.{func.__qualname__}"
            
            with tracer.start_as_current_span(op_name) as span:
                # Add method attributes
                span.set_attribute("method.name", func.__name__)
                span.set_attribute("method.module", func.__module__)
                
                # Add arguments if requested
                if include_args:
                    for i, arg in enumerate(args[1:]):  # Skip self
                        span.set_attribute(f"args.{i}", str(arg))
                    
                    for key, value in kwargs.items():
                        span.set_attribute(f"kwargs.{key}", str(value))
                
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    
                    # Record duration
                    duration = time.time() - start_time
                    span.set_attribute("method.duration_seconds", duration)
                    
                    # Add result if requested
                    if include_result and result is not None:
                        span.set_attribute("method.result", str(result)[:1000])  # Truncate
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return cast(Callable[..., T], wrapper)
    return decorator


class GEPATracer:
    """GEPA-specific tracing utilities."""
    
    def __init__(self, tracer: Optional[trace.Tracer] = None):
        self.tracer = tracer or get_tracer()
    
    def trace_optimization_run(
        self,
        system_id: str,
        optimization_run_id: str,
        budget: int
    ):
        """Context manager for tracing entire optimization run."""
        return trace_operation(
            "gepa.optimization_run",
            attributes={
                "gepa.system_id": system_id,
                "gepa.optimization_run_id": optimization_run_id,
                "gepa.budget": budget,
            },
            tracer=self.tracer
        )
    
    def trace_generation(
        self,
        system_id: str,
        optimization_run_id: str,
        generation: int
    ):
        """Context manager for tracing a generation."""
        return trace_operation(
            "gepa.generation",
            attributes={
                "gepa.system_id": system_id,
                "gepa.optimization_run_id": optimization_run_id,
                "gepa.generation": generation,
            },
            tracer=self.tracer
        )
    
    def trace_mutation(
        self,
        system_id: str,
        mutation_type: str,
        module_id: str
    ):
        """Context manager for tracing mutation."""
        return trace_operation(
            "gepa.mutation",
            attributes={
                "gepa.system_id": system_id,
                "gepa.mutation_type": mutation_type,
                "gepa.module_id": module_id,
            },
            tracer=self.tracer
        )
    
    def trace_evaluation(
        self,
        system_id: str,
        candidate_id: str,
        dataset_size: int
    ):
        """Context manager for tracing evaluation."""
        return trace_operation(
            "gepa.evaluation",
            attributes={
                "gepa.system_id": system_id,
                "gepa.candidate_id": candidate_id,
                "gepa.dataset_size": dataset_size,
            },
            tracer=self.tracer
        )
    
    def trace_inference(
        self,
        provider: str,
        model: str,
        prompt_length: int
    ):
        """Context manager for tracing inference."""
        return trace_operation(
            "gepa.inference",
            attributes={
                "gepa.provider": provider,
                "gepa.model": model,
                "gepa.prompt_length": prompt_length,
            },
            tracer=self.tracer
        )
    
    def add_candidate_attributes(
        self,
        span: trace.Span,
        candidate_id: str,
        scores: Dict[str, float],
        cost: float,
        tokens_used: int
    ) -> None:
        """Add candidate attributes to current span."""
        span.set_attribute("gepa.candidate_id", candidate_id)
        span.set_attribute("gepa.cost", cost)
        span.set_attribute("gepa.tokens_used", tokens_used)
        
        for metric_name, score in scores.items():
            span.set_attribute(f"gepa.score.{metric_name}", score)
    
    def add_pareto_frontier_attributes(
        self,
        span: trace.Span,
        frontier_size: int,
        best_scores: Dict[str, float]
    ) -> None:
        """Add Pareto frontier attributes to current span."""
        span.set_attribute("gepa.frontier_size", frontier_size)
        
        for metric_name, score in best_scores.items():
            span.set_attribute(f"gepa.best_score.{metric_name}", score)


# Global tracer instance
_gepa_tracer: Optional[GEPATracer] = None


def get_gepa_tracer() -> GEPATracer:
    """Get the global GEPA tracer instance."""
    global _gepa_tracer
    
    if _gepa_tracer is None:
        _gepa_tracer = GEPATracer()
    
    return _gepa_tracer