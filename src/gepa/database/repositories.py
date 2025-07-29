"""Repository classes for database operations."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from sqlalchemy import select, update, delete, desc, asc, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import structlog

from .models import (
    OptimizationRun,
    Candidate, 
    Trajectory,
    TrajectoryStep,
    OptimizationMetrics,
    CandidateResponse,
    TrajectoryResponse,
    OptimizationRunResponse
)
from ..core.pareto import Candidate as CoreCandidate
from ..core.mutation import Trajectory as CoreTrajectory, TrajectoryStep as CoreTrajectoryStep

logger = structlog.get_logger(__name__)


class OptimizationRunRepository:
    """Repository for optimization run operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(
        self,
        system_id: str,
        config_hash: str,
        budget: int,
        minibatch_size: int,
        pareto_set_size: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> OptimizationRun:
        """Create a new optimization run."""
        run = OptimizationRun(
            system_id=system_id,
            config_hash=config_hash,
            budget=budget,
            minibatch_size=minibatch_size,
            pareto_set_size=pareto_set_size,
            metadata_=metadata
        )
        
        self.session.add(run)
        await self.session.commit()
        await self.session.refresh(run)
        
        logger.info("Created optimization run", run_id=run.id, system_id=system_id)
        return run
    
    async def get_by_id(self, run_id: UUID) -> Optional[OptimizationRun]:
        """Get optimization run by ID."""
        result = await self.session.execute(
            select(OptimizationRun).where(OptimizationRun.id == run_id)
        )
        return result.scalar_one_or_none()
    
    async def update_status(
        self,
        run_id: UUID,
        status: str,
        rollouts_used: Optional[int] = None,
        total_cost: Optional[float] = None,
        best_score: Optional[float] = None
    ) -> None:
        """Update optimization run status and metrics."""
        update_data = {"status": status}
        
        if status in ["completed", "failed"]:
            update_data["completed_at"] = datetime.utcnow()
        
        if rollouts_used is not None:
            update_data["rollouts_used"] = rollouts_used
        if total_cost is not None:
            update_data["total_cost"] = total_cost
        if best_score is not None:
            update_data["best_score"] = best_score
        
        await self.session.execute(
            update(OptimizationRun)
            .where(OptimizationRun.id == run_id)
            .values(**update_data)
        )
        await self.session.commit()
        
        logger.info("Updated optimization run", run_id=run_id, status=status)
    
    async def list_runs(
        self,
        system_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[OptimizationRunResponse]:
        """List optimization runs with filtering."""
        query = select(OptimizationRun)
        
        if system_id:
            query = query.where(OptimizationRun.system_id == system_id)
        if status:
            query = query.where(OptimizationRun.status == status)
        
        query = query.order_by(desc(OptimizationRun.started_at)).offset(offset).limit(limit)
        
        result = await self.session.execute(query)
        runs = result.scalars().all()
        
        return [OptimizationRunResponse.from_orm(run) for run in runs]


class CandidateRepository:
    """Repository for candidate operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_from_core_candidate(
        self,
        optimization_run_id: UUID,
        core_candidate: CoreCandidate,
        generation: int = 0,
        parent_id: Optional[UUID] = None,
        mutation_type: Optional[str] = None,
        mutated_module: Optional[str] = None
    ) -> Candidate:
        """Create candidate from core candidate object."""
        # Serialize system definition
        system_definition = {
            "system_id": core_candidate.system.system_id,
            "modules": {
                mid: {
                    "id": module.id,
                    "model_weights": module.model_weights,
                }
                for mid, module in core_candidate.system.modules.items()
            }
        }
        
        # Extract module prompts
        module_prompts = core_candidate.system.get_module_prompts()
        
        candidate = Candidate(
            optimization_run_id=optimization_run_id,
            generation=generation,
            parent_id=parent_id,
            system_definition=system_definition,
            module_prompts=module_prompts,
            scores=core_candidate.scores,
            cost=core_candidate.cost,
            tokens_used=core_candidate.tokens_used,
            mutation_type=mutation_type,
            mutated_module=mutated_module
        )
        
        self.session.add(candidate)
        await self.session.commit()
        await self.session.refresh(candidate)
        
        logger.info("Created candidate", candidate_id=candidate.id, generation=generation)
        return candidate
    
    async def get_by_id(self, candidate_id: UUID) -> Optional[Candidate]:
        """Get candidate by ID."""
        result = await self.session.execute(
            select(Candidate).where(Candidate.id == candidate_id)
        )
        return result.scalar_one_or_none()
    
    async def get_pareto_frontier(
        self,
        optimization_run_id: UUID
    ) -> List[CandidateResponse]:
        """Get current Pareto frontier candidates."""
        result = await self.session.execute(
            select(Candidate)
            .where(
                Candidate.optimization_run_id == optimization_run_id,
                Candidate.is_in_frontier == True
            )
            .order_by(desc(Candidate.created_at))
        )
        
        candidates = result.scalars().all()
        return [CandidateResponse.from_orm(candidate) for candidate in candidates]
    
    async def update_frontier_status(
        self,
        optimization_run_id: UUID,
        frontier_candidate_ids: List[UUID]
    ) -> None:
        """Update which candidates are in the frontier."""
        # First, mark all candidates as not in frontier
        await self.session.execute(
            update(Candidate)
            .where(Candidate.optimization_run_id == optimization_run_id)
            .values(is_in_frontier=False)
        )
        
        # Then mark frontier candidates as in frontier
        if frontier_candidate_ids:
            await self.session.execute(
                update(Candidate)
                .where(Candidate.id.in_(frontier_candidate_ids))
                .values(is_in_frontier=True)
            )
        
        await self.session.commit()
        
        logger.info(
            "Updated frontier status",
            optimization_run_id=optimization_run_id,
            frontier_size=len(frontier_candidate_ids)
        )
    
    async def get_generation_candidates(
        self,
        optimization_run_id: UUID,
        generation: int
    ) -> List[CandidateResponse]:
        """Get all candidates from a specific generation."""
        result = await self.session.execute(
            select(Candidate)
            .where(
                Candidate.optimization_run_id == optimization_run_id,
                Candidate.generation == generation
            )
            .order_by(asc(Candidate.created_at))
        )
        
        candidates = result.scalars().all()
        return [CandidateResponse.from_orm(candidate) for candidate in candidates]


class TrajectoryRepository:
    """Repository for trajectory operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_from_core_trajectory(
        self,
        optimization_run_id: UUID,
        candidate_id: UUID,
        core_trajectory: CoreTrajectory
    ) -> Trajectory:
        """Create trajectory from core trajectory object."""
        trajectory = Trajectory(
            optimization_run_id=optimization_run_id,
            candidate_id=candidate_id,
            input_data=core_trajectory.input_data,
            output_data=core_trajectory.output_data,
            success=core_trajectory.success,
            total_latency=core_trajectory.total_latency,
            error_message=core_trajectory.error
        )
        
        self.session.add(trajectory)
        await self.session.flush()  # Get the ID without committing
        
        # Create trajectory steps
        for i, core_step in enumerate(core_trajectory.steps):
            step = TrajectoryStep(
                trajectory_id=trajectory.id,
                step_index=i,
                module_id=core_step.module_id,
                input_data=core_step.input_data,
                output_data=core_step.output_data,
                prompt_used=core_step.prompt_used,
                latency=core_step.latency,
                error_message=core_step.error
            )
            self.session.add(step)
        
        await self.session.commit()
        await self.session.refresh(trajectory)
        
        logger.info(
            "Created trajectory",
            trajectory_id=trajectory.id,
            candidate_id=candidate_id,
            success=trajectory.success
        )
        return trajectory
    
    async def get_by_candidate(
        self,
        candidate_id: UUID,
        limit: int = 100
    ) -> List[TrajectoryResponse]:
        """Get trajectories for a candidate."""
        result = await self.session.execute(
            select(Trajectory)
            .where(Trajectory.candidate_id == candidate_id)
            .order_by(desc(Trajectory.executed_at))
            .limit(limit)
        )
        
        trajectories = result.scalars().all()
        return [TrajectoryResponse.from_orm(trajectory) for trajectory in trajectories]
    
    async def get_recent_failures(
        self,
        optimization_run_id: UUID,
        module_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Trajectory]:
        """Get recent failed trajectories for analysis."""
        query = select(Trajectory).options(selectinload(Trajectory.steps))
        query = query.where(
            Trajectory.optimization_run_id == optimization_run_id,
            Trajectory.success == False
        )
        
        if module_id:
            # Filter by trajectories that have steps with the specified module
            query = query.join(TrajectoryStep).where(TrajectoryStep.module_id == module_id)
        
        query = query.order_by(desc(Trajectory.executed_at)).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().unique().all()
    
    async def get_success_rate(
        self,
        optimization_run_id: UUID,
        candidate_id: Optional[UUID] = None
    ) -> float:
        """Calculate success rate for trajectories."""
        query = select(func.avg(Trajectory.success.cast(float)))
        query = query.where(Trajectory.optimization_run_id == optimization_run_id)
        
        if candidate_id:
            query = query.where(Trajectory.candidate_id == candidate_id)
        
        result = await self.session.execute(query)
        success_rate = result.scalar()
        
        return success_rate or 0.0


class MetricsRepository:
    """Repository for optimization metrics."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def record_generation_metrics(
        self,
        optimization_run_id: UUID,
        generation: int,
        frontier_size: int,
        best_score: float,
        avg_score: float,
        total_cost: float,
        rollouts_used: int,
        score_std: Optional[float] = None,
        cost_range: Optional[float] = None
    ) -> OptimizationMetrics:
        """Record metrics for a generation."""
        metrics = OptimizationMetrics(
            optimization_run_id=optimization_run_id,
            generation=generation,
            frontier_size=frontier_size,
            best_score=best_score,
            avg_score=avg_score,
            total_cost=total_cost,
            rollouts_used=rollouts_used,
            score_std=score_std,
            cost_range=cost_range
        )
        
        self.session.add(metrics)
        await self.session.commit()
        await self.session.refresh(metrics)
        
        logger.info(
            "Recorded generation metrics",
            optimization_run_id=optimization_run_id,
            generation=generation,
            best_score=best_score
        )
        return metrics
    
    async def get_optimization_history(
        self,
        optimization_run_id: UUID
    ) -> List[OptimizationMetrics]:
        """Get optimization history for a run."""
        result = await self.session.execute(
            select(OptimizationMetrics)
            .where(OptimizationMetrics.optimization_run_id == optimization_run_id)
            .order_by(asc(OptimizationMetrics.generation))
        )
        
        return result.scalars().all()