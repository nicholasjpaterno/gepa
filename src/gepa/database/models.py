"""Database models for GEPA prompt pool and trajectory storage."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, 
    String, 
    Integer, 
    Float, 
    DateTime, 
    Boolean, 
    Text,
    ForeignKey,
    Index,
    JSON
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from pydantic import BaseModel


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class OptimizationRun(Base):
    """Represents a single optimization run."""
    
    __tablename__ = "optimization_runs"
    
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    system_id: Mapped[str] = mapped_column(String(255), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # Hash of configuration
    budget: Mapped[int] = mapped_column(Integer, nullable=False)
    minibatch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    pareto_set_size: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Status and timing
    status: Mapped[str] = mapped_column(String(20), default="running")  # running, completed, failed
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Results
    rollouts_used: Mapped[int] = mapped_column(Integer, default=0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    best_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Metadata
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    candidates: Mapped[List["Candidate"]] = relationship(
        "Candidate", 
        back_populates="optimization_run",
        cascade="all, delete-orphan"
    )
    trajectories: Mapped[List["Trajectory"]] = relationship(
        "Trajectory",
        back_populates="optimization_run", 
        cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        Index("ix_optimization_runs_system_id", "system_id"),
        Index("ix_optimization_runs_started_at", "started_at"),
    )


class Candidate(Base):
    """Represents a candidate system in the Pareto frontier."""
    
    __tablename__ = "candidates"
    
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    optimization_run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("optimization_runs.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Candidate properties
    generation: Mapped[int] = mapped_column(Integer, default=0)
    parent_id: Mapped[Optional[UUID]] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    
    # System definition
    system_definition: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    module_prompts: Mapped[Dict[str, str]] = mapped_column(JSON, nullable=False)
    
    # Performance metrics
    scores: Mapped[Dict[str, float]] = mapped_column(JSON, nullable=False)
    cost: Mapped[float] = mapped_column(Float, nullable=False)
    tokens_used: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Mutation info
    mutation_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    mutated_module: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Status
    is_in_frontier: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    optimization_run: Mapped["OptimizationRun"] = relationship(
        "OptimizationRun",
        back_populates="candidates"
    )
    trajectories: Mapped[List["Trajectory"]] = relationship(
        "Trajectory",
        back_populates="candidate"
    )
    
    __table_args__ = (
        Index("ix_candidates_optimization_run_id", "optimization_run_id"),
        Index("ix_candidates_generation", "generation"),
        Index("ix_candidates_is_in_frontier", "is_in_frontier"),
        Index("ix_candidates_created_at", "created_at"),
    )


class Trajectory(Base):
    """Represents a system execution trajectory."""
    
    __tablename__ = "trajectories"
    
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    optimization_run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("optimization_runs.id", ondelete="CASCADE"),
        nullable=False
    )
    candidate_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("candidates.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Execution data
    input_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    output_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    
    # Performance
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    total_latency: Mapped[float] = mapped_column(Float, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Execution timestamp
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    optimization_run: Mapped["OptimizationRun"] = relationship(
        "OptimizationRun",
        back_populates="trajectories"
    )
    candidate: Mapped["Candidate"] = relationship(
        "Candidate",
        back_populates="trajectories"
    )
    steps: Mapped[List["TrajectoryStep"]] = relationship(
        "TrajectoryStep",
        back_populates="trajectory",
        cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        Index("ix_trajectories_candidate_id", "candidate_id"),
        Index("ix_trajectories_success", "success"),
        Index("ix_trajectories_executed_at", "executed_at"),
    )


class TrajectoryStep(Base):
    """Represents a single step in a trajectory."""
    
    __tablename__ = "trajectory_steps"
    
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    trajectory_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("trajectories.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Step information
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    module_id: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Data
    input_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    output_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    prompt_used: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Performance
    latency: Mapped[float] = mapped_column(Float, nullable=False)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationship
    trajectory: Mapped["Trajectory"] = relationship(
        "Trajectory",
        back_populates="steps"
    )
    
    __table_args__ = (
        Index("ix_trajectory_steps_trajectory_id", "trajectory_id"),
        Index("ix_trajectory_steps_step_index", "step_index"),
        Index("ix_trajectory_steps_module_id", "module_id"),
    )


class OptimizationMetrics(Base):
    """Stores aggregated metrics for optimization runs."""
    
    __tablename__ = "optimization_metrics"
    
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    optimization_run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("optimization_runs.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Metrics
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    frontier_size: Mapped[int] = mapped_column(Integer, nullable=False)
    best_score: Mapped[float] = mapped_column(Float, nullable=False)
    avg_score: Mapped[float] = mapped_column(Float, nullable=False)
    total_cost: Mapped[float] = mapped_column(Float, nullable=False)
    rollouts_used: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Diversity metrics
    score_std: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cost_range: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index("ix_optimization_metrics_run_id", "optimization_run_id"),
        Index("ix_optimization_metrics_generation", "generation"),
    )


# Pydantic models for API serialization
class CandidateResponse(BaseModel):
    """Response model for candidate data."""
    
    id: str
    generation: int
    parent_id: Optional[str]
    scores: Dict[str, float]
    cost: float
    tokens_used: int
    mutation_type: Optional[str]
    mutated_module: Optional[str]
    is_in_frontier: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class TrajectoryResponse(BaseModel):
    """Response model for trajectory data."""
    
    id: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success: bool
    total_latency: float
    error_message: Optional[str]
    executed_at: datetime
    
    class Config:
        from_attributes = True


class OptimizationRunResponse(BaseModel):
    """Response model for optimization run data."""
    
    id: str
    system_id: str
    budget: int
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    rollouts_used: int
    total_cost: float
    best_score: Optional[float]
    
    class Config:
        from_attributes = True