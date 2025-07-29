"""Structured logging configuration for GEPA."""

import logging
import sys
from typing import Any, Dict, Optional

import structlog
from structlog.stdlib import LoggerFactory

from ..config import ObservabilityConfig


def configure_logging(config: ObservabilityConfig) -> None:
    """Configure structured logging for GEPA."""
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.add_logger_name,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
    ]
    
    if config.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True),
        ])
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.log_level.upper())
    )
    
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


class GEPALogger:
    """GEPA-specific logger with contextual information."""
    
    def __init__(self, name: str = "gepa"):
        self.logger = structlog.get_logger(name)
        self._context: Dict[str, Any] = {}
    
    def bind(self, **kwargs) -> "GEPALogger":
        """Create a new logger with additional context."""
        new_logger = GEPALogger()
        new_logger.logger = self.logger.bind(**kwargs)
        new_logger._context = {**self._context, **kwargs}
        return new_logger
    
    def with_optimization_context(
        self,
        system_id: str,
        optimization_run_id: str,
        generation: Optional[int] = None
    ) -> "GEPALogger":
        """Add optimization context to logger."""
        context = {
            "system_id": system_id,
            "optimization_run_id": optimization_run_id,
        }
        
        if generation is not None:
            context["generation"] = generation
        
        return self.bind(**context)
    
    def with_candidate_context(
        self,
        candidate_id: str,
        generation: Optional[int] = None,
        parent_id: Optional[str] = None
    ) -> "GEPALogger":
        """Add candidate context to logger."""
        context = {"candidate_id": candidate_id}
        
        if generation is not None:
            context["generation"] = generation
        if parent_id is not None:
            context["parent_id"] = parent_id
        
        return self.bind(**context)
    
    def with_inference_context(
        self,
        provider: str,
        model: str,
        request_id: Optional[str] = None
    ) -> "GEPALogger":
        """Add inference context to logger."""
        context = {
            "inference_provider": provider,
            "inference_model": model,
        }
        
        if request_id is not None:
            context["request_id"] = request_id
        
        return self.bind(**context)
    
    def log_optimization_start(
        self,
        system_id: str,
        optimization_run_id: str,
        budget: int,
        config_summary: Dict[str, Any]
    ) -> None:
        """Log optimization start with details."""
        self.logger.info(
            "Starting GEPA optimization",
            system_id=system_id,
            optimization_run_id=optimization_run_id,
            budget=budget,
            **config_summary
        )
    
    def log_optimization_end(
        self,
        system_id: str,
        optimization_run_id: str,
        status: str,
        duration: float,
        rollouts_used: int,
        total_cost: float,
        best_score: Optional[float] = None
    ) -> None:
        """Log optimization completion."""
        self.logger.info(
            "GEPA optimization completed",
            system_id=system_id,
            optimization_run_id=optimization_run_id,
            status=status,
            duration_seconds=duration,
            rollouts_used=rollouts_used,
            total_cost=total_cost,
            best_score=best_score
        )
    
    def log_generation_start(
        self,
        system_id: str,
        optimization_run_id: str,
        generation: int,
        frontier_size: int
    ) -> None:
        """Log generation start."""
        self.logger.info(
            "Starting generation",
            system_id=system_id,
            optimization_run_id=optimization_run_id,
            generation=generation,
            frontier_size=frontier_size
        )
    
    def log_generation_end(
        self,
        system_id: str,
        optimization_run_id: str,
        generation: int,
        candidates_created: int,
        candidates_added_to_frontier: int,
        best_scores: Dict[str, float]
    ) -> None:
        """Log generation completion."""
        self.logger.info(
            "Generation completed",
            system_id=system_id,
            optimization_run_id=optimization_run_id,
            generation=generation,
            candidates_created=candidates_created,
            candidates_added_to_frontier=candidates_added_to_frontier,
            best_scores=best_scores
        )
    
    def log_mutation_attempt(
        self,
        system_id: str,
        candidate_id: str,
        parent_id: str,
        mutation_type: str,
        module_id: str
    ) -> None:
        """Log mutation attempt."""
        self.logger.info(
            "Attempting mutation",
            system_id=system_id,
            candidate_id=candidate_id,
            parent_id=parent_id,
            mutation_type=mutation_type,
            module_id=module_id
        )
    
    def log_mutation_result(
        self,
        system_id: str,
        candidate_id: str,
        mutation_type: str,
        success: bool,
        duration: float,
        error: Optional[str] = None
    ) -> None:
        """Log mutation result."""
        level = "info" if success else "warning"
        
        getattr(self.logger, level)(
            "Mutation completed",
            system_id=system_id,
            candidate_id=candidate_id,
            mutation_type=mutation_type,
            success=success,
            duration_seconds=duration,
            error=error
        )
    
    def log_evaluation_start(
        self,
        system_id: str,
        candidate_id: str,
        dataset_size: int
    ) -> None:
        """Log evaluation start."""
        self.logger.info(
            "Starting evaluation",
            system_id=system_id,
            candidate_id=candidate_id,
            dataset_size=dataset_size
        )
    
    def log_evaluation_result(
        self,
        system_id: str,
        candidate_id: str,
        scores: Dict[str, float],
        cost: float,
        tokens_used: int,
        duration: float,
        added_to_frontier: bool
    ) -> None:
        """Log evaluation result."""
        self.logger.info(
            "Evaluation completed",
            system_id=system_id,
            candidate_id=candidate_id,
            scores=scores,
            cost=cost,
            tokens_used=tokens_used,
            duration_seconds=duration,
            added_to_frontier=added_to_frontier
        )
    
    def log_inference_request(
        self,
        provider: str,
        model: str,
        prompt_length: int,
        max_tokens: int,
        request_id: Optional[str] = None
    ) -> None:
        """Log inference request."""
        self.logger.debug(
            "Inference request",
            provider=provider,
            model=model,
            prompt_length=prompt_length,
            max_tokens=max_tokens,
            request_id=request_id
        )
    
    def log_inference_response(
        self,
        provider: str,
        model: str,
        duration: float,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        request_id: Optional[str] = None
    ) -> None:
        """Log inference response."""
        self.logger.debug(
            "Inference response",
            provider=provider,
            model=model,
            duration_seconds=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            request_id=request_id
        )
    
    def log_pareto_frontier_update(
        self,
        system_id: str,
        optimization_run_id: str,
        frontier_size: int,
        best_scores: Dict[str, float],
        diversity_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Log Pareto frontier update."""
        self.logger.info(
            "Pareto frontier updated",
            system_id=system_id,
            optimization_run_id=optimization_run_id,
            frontier_size=frontier_size,
            best_scores=best_scores,
            diversity_metrics=diversity_metrics
        )
    
    def log_error(
        self,
        message: str,
        error: Exception,
        **kwargs
    ) -> None:
        """Log error with context."""
        self.logger.error(
            message,
            error=str(error),
            error_type=type(error).__name__,
            **kwargs
        )
    
    def log_warning(
        self,
        message: str,
        **kwargs
    ) -> None:
        """Log warning with context."""
        self.logger.warning(message, **kwargs)
    
    def log_debug(
        self,
        message: str,
        **kwargs
    ) -> None:
        """Log debug message with context."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, **kwargs)


# Global logger instance
_gepa_logger: Optional[GEPALogger] = None


def get_logger(name: str = "gepa") -> GEPALogger:
    """Get the global GEPA logger instance."""
    global _gepa_logger
    
    if _gepa_logger is None:
        _gepa_logger = GEPALogger(name)
    
    return _gepa_logger