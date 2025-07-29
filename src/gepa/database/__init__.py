"""Database module for GEPA."""

from .models import (
    Base,
    OptimizationRun,
    Candidate,
    Trajectory,
    TrajectoryStep,
    OptimizationMetrics,
    CandidateResponse,
    TrajectoryResponse,
    OptimizationRunResponse,
)
from .connection import DatabaseManager, get_database_manager, get_db_session
from .repositories import (
    OptimizationRunRepository,
    CandidateRepository,
    TrajectoryRepository,
    MetricsRepository,
)
from .migrations import MigrationManager, init_database, reset_database

__all__ = [
    "Base",
    "OptimizationRun",
    "Candidate", 
    "Trajectory",
    "TrajectoryStep",
    "OptimizationMetrics",
    "CandidateResponse",
    "TrajectoryResponse",
    "OptimizationRunResponse",
    "DatabaseManager",
    "get_database_manager",
    "get_db_session",
    "OptimizationRunRepository",
    "CandidateRepository",
    "TrajectoryRepository", 
    "MetricsRepository",
    "MigrationManager",
    "init_database",
    "reset_database",
]