"""Prometheus metrics for GEPA."""

import time
from typing import Dict, Optional
from enum import Enum

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    CollectorRegistry,
    start_http_server,
    REGISTRY
)
import structlog

from ..config import ObservabilityConfig

logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    INFO = "info"


class GEPAMetrics:
    """Prometheus metrics for GEPA operations."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
        
        # Optimization metrics
        self.optimization_runs_total = Counter(
            "gepa_optimization_runs_total",
            "Total number of optimization runs started",
            ["system_id", "status"],
            registry=self.registry
        )
        
        self.optimization_duration_seconds = Histogram(
            "gepa_optimization_duration_seconds",
            "Duration of optimization runs in seconds",
            ["system_id", "status"],
            registry=self.registry
        )
        
        self.rollouts_total = Counter(
            "gepa_rollouts_total",
            "Total number of rollouts executed",
            ["system_id", "optimization_run_id"],
            registry=self.registry
        )
        
        self.rollouts_cost_total = Counter(
            "gepa_rollouts_cost_total",
            "Total cost of rollouts in USD",
            ["system_id", "provider", "model"],
            registry=self.registry
        )
        
        # Pareto frontier metrics
        self.pareto_frontier_size = Gauge(
            "gepa_pareto_frontier_size",
            "Current size of Pareto frontier",
            ["system_id", "optimization_run_id"],
            registry=self.registry
        )
        
        self.best_score = Gauge(
            "gepa_best_score",
            "Best score achieved in optimization",
            ["system_id", "optimization_run_id", "metric_name"],
            registry=self.registry
        )
        
        self.generation_count = Gauge(
            "gepa_generation_count",
            "Current generation number",
            ["system_id", "optimization_run_id"],
            registry=self.registry
        )
        
        # Mutation metrics
        self.mutations_total = Counter(
            "gepa_mutations_total",
            "Total number of mutations performed",
            ["system_id", "mutation_type", "success"],
            registry=self.registry
        )
        
        self.mutation_duration_seconds = Histogram(
            "gepa_mutation_duration_seconds",
            "Duration of mutation operations in seconds",
            ["mutation_type"],
            registry=self.registry
        )
        
        # Inference metrics
        self.inference_requests_total = Counter(
            "gepa_inference_requests_total",
            "Total number of inference requests",
            ["provider", "model", "status"],
            registry=self.registry
        )
        
        self.inference_duration_seconds = Histogram(
            "gepa_inference_duration_seconds",
            "Duration of inference requests in seconds",
            ["provider", "model"],
            registry=self.registry
        )
        
        self.inference_tokens_total = Counter(
            "gepa_inference_tokens_total",
            "Total number of tokens processed",
            ["provider", "model", "token_type"],
            registry=self.registry
        )
        
        # Evaluation metrics
        self.evaluations_total = Counter(
            "gepa_evaluations_total",
            "Total number of evaluations performed",
            ["system_id", "metric_name"],
            registry=self.registry
        )
        
        self.evaluation_scores = Histogram(
            "gepa_evaluation_scores",
            "Distribution of evaluation scores",
            ["system_id", "metric_name"],
            registry=self.registry
        )
        
        # System info
        self.gepa_info = Info(
            "gepa_info",
            "Information about GEPA version and configuration",
            registry=self.registry
        )
    
    def record_optimization_start(self, system_id: str, optimization_run_id: str) -> None:
        """Record the start of an optimization run."""
        self.optimization_runs_total.labels(
            system_id=system_id,
            status="started"
        ).inc()
        
        logger.info(
            "Recorded optimization start",
            system_id=system_id,
            optimization_run_id=optimization_run_id
        )
    
    def record_optimization_end(
        self,
        system_id: str,
        optimization_run_id: str,
        duration: float,
        status: str
    ) -> None:
        """Record the end of an optimization run."""
        self.optimization_runs_total.labels(
            system_id=system_id,
            status=status
        ).inc()
        
        self.optimization_duration_seconds.labels(
            system_id=system_id,
            status=status
        ).observe(duration)
        
        logger.info(
            "Recorded optimization end",
            system_id=system_id,
            optimization_run_id=optimization_run_id,
            duration=duration,
            status=status
        )
    
    def record_rollout(
        self,
        system_id: str,
        optimization_run_id: str,
        cost: float = 0.0
    ) -> None:
        """Record a rollout execution."""
        self.rollouts_total.labels(
            system_id=system_id,
            optimization_run_id=optimization_run_id
        ).inc()
    
    def record_rollout_cost(
        self,
        system_id: str,
        provider: str,
        model: str,
        cost: float
    ) -> None:
        """Record rollout cost."""
        self.rollouts_cost_total.labels(
            system_id=system_id,
            provider=provider,
            model=model
        ).inc(cost)
    
    def update_pareto_frontier_size(
        self,
        system_id: str,
        optimization_run_id: str,
        size: int
    ) -> None:
        """Update Pareto frontier size."""
        self.pareto_frontier_size.labels(
            system_id=system_id,
            optimization_run_id=optimization_run_id
        ).set(size)
    
    def update_best_score(
        self,
        system_id: str,
        optimization_run_id: str,
        metric_name: str,
        score: float
    ) -> None:
        """Update best score achieved."""
        self.best_score.labels(
            system_id=system_id,
            optimization_run_id=optimization_run_id,
            metric_name=metric_name
        ).set(score)
    
    def update_generation(
        self,
        system_id: str,
        optimization_run_id: str,
        generation: int
    ) -> None:
        """Update current generation."""
        self.generation_count.labels(
            system_id=system_id,
            optimization_run_id=optimization_run_id
        ).set(generation)
    
    def record_mutation(
        self,
        system_id: str,
        mutation_type: str,
        duration: float,
        success: bool
    ) -> None:
        """Record a mutation operation."""
        self.mutations_total.labels(
            system_id=system_id,
            mutation_type=mutation_type,
            success=str(success).lower()
        ).inc()
        
        self.mutation_duration_seconds.labels(
            mutation_type=mutation_type
        ).observe(duration)
    
    def record_inference_request(
        self,
        provider: str,
        model: str,
        duration: float,
        status: str,
        input_tokens: int = 0,
        output_tokens: int = 0
    ) -> None:
        """Record an inference request."""
        self.inference_requests_total.labels(
            provider=provider,
            model=model,
            status=status
        ).inc()
        
        self.inference_duration_seconds.labels(
            provider=provider,
            model=model
        ).observe(duration)
        
        if input_tokens > 0:
            self.inference_tokens_total.labels(
                provider=provider,
                model=model,
                token_type="input"
            ).inc(input_tokens)
        
        if output_tokens > 0:
            self.inference_tokens_total.labels(
                provider=provider,
                model=model,
                token_type="output"
            ).inc(output_tokens)
    
    def record_evaluation(
        self,
        system_id: str,
        metric_name: str,
        score: float
    ) -> None:
        """Record an evaluation result."""
        self.evaluations_total.labels(
            system_id=system_id,
            metric_name=metric_name
        ).inc()
        
        self.evaluation_scores.labels(
            system_id=system_id,
            metric_name=metric_name
        ).observe(score)
    
    def set_gepa_info(self, version: str, config_info: Dict[str, str]) -> None:
        """Set GEPA system information."""
        info_dict = {"version": version}
        info_dict.update(config_info)
        self.gepa_info.info(info_dict)


# Global metrics instance
_metrics: Optional[GEPAMetrics] = None


def get_metrics(registry: Optional[CollectorRegistry] = None) -> GEPAMetrics:
    """Get the global metrics instance."""
    global _metrics
    
    if _metrics is None:
        _metrics = GEPAMetrics(registry)
    
    return _metrics


class MetricsServer:
    """Prometheus metrics HTTP server."""
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.server = None
        self.metrics = get_metrics()
    
    def start(self) -> None:
        """Start the metrics server."""
        if not self.config.enable_prometheus:
            logger.info("Prometheus metrics disabled")
            return
        
        try:
            start_http_server(self.config.prometheus_port)
            logger.info(
                "Started Prometheus metrics server",
                port=self.config.prometheus_port
            )
        except Exception as e:
            logger.error("Failed to start metrics server", error=str(e))
            raise
    
    def stop(self) -> None:
        """Stop the metrics server."""
        # Prometheus client doesn't provide a direct way to stop the server
        # In production, you might want to use a custom HTTP server
        logger.info("Metrics server stopped")


class MetricsCollector:
    """Collects and reports custom metrics."""
    
    def __init__(self, metrics: Optional[GEPAMetrics] = None):
        self.metrics = metrics or get_metrics()
        self._start_times: Dict[str, float] = {}
    
    def start_timer(self, operation_id: str) -> None:
        """Start timing an operation."""
        self._start_times[operation_id] = time.time()
    
    def end_timer(self, operation_id: str) -> float:
        """End timing an operation and return duration."""
        if operation_id not in self._start_times:
            logger.warning("Timer not started for operation", operation_id=operation_id)
            return 0.0
        
        start_time = self._start_times.pop(operation_id)
        duration = time.time() - start_time
        
        return duration
    
    def collect_optimization_metrics(
        self,
        system_id: str,
        optimization_run_id: str,
        generation: int,
        frontier_size: int,
        best_scores: Dict[str, float],
        rollouts_used: int,
        total_cost: float
    ) -> None:
        """Collect comprehensive optimization metrics."""
        self.metrics.update_generation(system_id, optimization_run_id, generation)
        self.metrics.update_pareto_frontier_size(system_id, optimization_run_id, frontier_size)
        
        for metric_name, score in best_scores.items():
            self.metrics.update_best_score(system_id, optimization_run_id, metric_name, score)
    
    def collect_system_info(self, version: str, config: Dict[str, str]) -> None:
        """Collect system information."""
        self.metrics.set_gepa_info(version, config)