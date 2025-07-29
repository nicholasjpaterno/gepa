"""Integration tests for database layer."""

import pytest
from uuid import uuid4
from typing import List

from gepa.database.connection import DatabaseManager
from gepa.database.models import OptimizationRun, Candidate, Trajectory, TrajectoryStep
from gepa.database.repositories import OptimizationRepository, CandidateRepository
from gepa.config import DatabaseConfig


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    @pytest.mark.asyncio
    async def test_database_initialization(self, test_db: DatabaseManager):
        """Test database initialization and table creation."""
        # Check that tables exist by trying to query them
        async with test_db.session() as session:
            # Should not raise any exceptions
            result = await session.execute("SELECT 1")
            assert result.scalar() == 1
    
    @pytest.mark.asyncio
    async def test_optimization_run_crud(self, test_db: DatabaseManager):
        """Test CRUD operations for optimization runs."""
        repo = OptimizationRepository(test_db)
        
        # Create optimization run
        run_data = {
            "task_id": "test-task",
            "config": {"budget": 50, "model": "gpt-3.5-turbo"},
            "status": "running",
            "best_score": None,
            "total_rollouts": 0,
            "total_cost": 0.0
        }
        
        run_id = await repo.create_optimization_run(**run_data)
        assert run_id is not None
        
        # Read optimization run
        run = await repo.get_optimization_run(run_id)
        assert run is not None
        assert run.task_id == "test-task"
        assert run.status == "running"
        assert run.config["budget"] == 50
        
        # Update optimization run
        await repo.update_optimization_run(
            run_id,
            status="completed",
            best_score=0.85,
            total_rollouts=25,
            total_cost=0.15
        )
        
        updated_run = await repo.get_optimization_run(run_id)
        assert updated_run.status == "completed"
        assert updated_run.best_score == 0.85
        assert updated_run.total_rollouts == 25
        assert updated_run.total_cost == 0.15
        
        # List optimization runs
        runs = await repo.list_optimization_runs(limit=10)
        assert len(runs) >= 1
        assert any(r.id == run_id for r in runs)
    
    @pytest.mark.asyncio
    async def test_candidate_crud(self, test_db: DatabaseManager, sample_system):
        """Test CRUD operations for candidates."""
        repo = CandidateRepository(test_db)
        opt_repo = OptimizationRepository(test_db)
        
        # Create optimization run first
        run_id = await opt_repo.create_optimization_run(
            task_id="test-task",
            config={"budget": 50},
            status="running"
        )
        
        # Create candidate
        candidate_data = {
            "optimization_run_id": run_id,
            "system_config": sample_system.to_dict(),
            "scores": {"exact_match": 0.8, "f1_score": 0.75},
            "cost": 0.05,
            "tokens_used": 150,
            "generation": 1,
            "parent_id": None
        }
        
        candidate_id = await repo.create_candidate(**candidate_data)
        assert candidate_id is not None
        
        # Read candidate
        candidate = await repo.get_candidate(candidate_id)
        assert candidate is not None
        assert candidate.optimization_run_id == run_id
        assert candidate.scores["exact_match"] == 0.8
        assert candidate.cost == 0.05
        
        # List candidates for run
        candidates = await repo.list_candidates_for_run(run_id)
        assert len(candidates) == 1
        assert candidates[0].id == candidate_id
        
        # Get best candidates by metric
        best_candidates = await repo.get_best_candidates_by_metric(
            run_id, "exact_match", limit=5
        )
        assert len(best_candidates) == 1
        assert best_candidates[0].id == candidate_id
    
    @pytest.mark.asyncio
    async def test_trajectory_storage(self, test_db: DatabaseManager, sample_system):
        """Test trajectory storage and retrieval."""
        opt_repo = OptimizationRepository(test_db)
        candidate_repo = CandidateRepository(test_db)
        
        # Create optimization run
        run_id = await opt_repo.create_optimization_run(
            task_id="test-task",
            config={"budget": 50},
            status="running"
        )
        
        # Create candidate
        candidate_id = await candidate_repo.create_candidate(
            optimization_run_id=run_id,
            system_config=sample_system.to_dict(),
            scores={"exact_match": 0.8},
            cost=0.05,
            tokens_used=150,
            generation=1
        )
        
        # Create trajectory
        async with test_db.session() as session:
            trajectory = Trajectory(
                candidate_id=candidate_id,
                input_data={"text": "Test input"},
                output_data={"summary": "Test output"},
                success=True,
                total_latency=1.5,
                error=None
            )
            session.add(trajectory)
            await session.commit()
            
            # Add trajectory steps
            steps = [
                TrajectoryStep(
                    trajectory_id=trajectory.id,
                    module_id="analyzer",
                    step_index=0,
                    input_data={"text": "Test input"},
                    output_data={"analysis": "Test analysis"},
                    latency=0.8,
                    cost=0.02,
                    tokens_used=75
                ),
                TrajectoryStep(
                    trajectory_id=trajectory.id,
                    module_id="summarizer", 
                    step_index=1,
                    input_data={"analysis": "Test analysis"},
                    output_data={"summary": "Test output"},
                    latency=0.7,
                    cost=0.03,
                    tokens_used=85
                )
            ]
            
            for step in steps:
                session.add(step)
            await session.commit()
            
            # Query trajectory with steps
            await session.refresh(trajectory)
            
            assert len(trajectory.steps) == 2
            assert trajectory.steps[0].module_id == "analyzer"
            assert trajectory.steps[1].module_id == "summarizer"
            assert trajectory.total_latency == 1.5
    
    @pytest.mark.asyncio
    async def test_complex_queries(self, test_db: DatabaseManager, sample_system):
        """Test complex database queries."""
        opt_repo = OptimizationRepository(test_db)
        candidate_repo = CandidateRepository(test_db)
        
        # Create multiple optimization runs
        run_ids = []
        for i in range(3):
            run_id = await opt_repo.create_optimization_run(
                task_id=f"test-task-{i}",
                config={"budget": 50 + i * 10},
                status="completed" if i < 2 else "running"
            )
            run_ids.append(run_id)
        
        # Create candidates for each run
        for run_id in run_ids:
            for j in range(3):
                await candidate_repo.create_candidate(
                    optimization_run_id=run_id,
                    system_config=sample_system.to_dict(),
                    scores={
                        "exact_match": 0.7 + j * 0.1,
                        "f1_score": 0.6 + j * 0.1
                    },
                    cost=0.05 + j * 0.01,
                    tokens_used=100 + j * 25,
                    generation=j + 1
                )
        
        # Test filtering by status
        completed_runs = await opt_repo.list_optimization_runs(
            status="completed", limit=10
        )
        assert len(completed_runs) == 2
        
        # Test getting top candidates across all runs
        async with test_db.session() as session:
            from sqlalchemy import select, desc
            from gepa.database.models import Candidate
            
            query = (
                select(Candidate)
                .order_by(desc(Candidate.scores["exact_match"].as_float()))
                .limit(5)
            )
            result = await session.execute(query)
            top_candidates = result.scalars().all()
            
            assert len(top_candidates) >= 5
            # Verify ordering
            for i in range(len(top_candidates) - 1):
                assert (top_candidates[i].scores["exact_match"] >= 
                       top_candidates[i + 1].scores["exact_match"])
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, test_db: DatabaseManager):
        """Test transaction rollback behavior."""
        repo = OptimizationRepository(test_db)
        
        try:
            async with test_db.session() as session:
                # Create an optimization run
                run = OptimizationRun(
                    task_id="test-rollback",
                    config={"budget": 50},
                    status="running"
                )
                session.add(run)
                await session.flush()  # Get the ID
                
                # Simulate an error
                raise ValueError("Simulated error")
                
        except ValueError:
            pass
        
        # Verify the run was not committed
        runs = await repo.list_optimization_runs(limit=10)
        rollback_runs = [r for r in runs if r.task_id == "test-rollback"]
        assert len(rollback_runs) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, test_db: DatabaseManager):
        """Test concurrent database access."""
        import asyncio
        
        repo = OptimizationRepository(test_db)
        
        async def create_run(task_suffix: str):
            return await repo.create_optimization_run(
                task_id=f"concurrent-task-{task_suffix}",
                config={"budget": 50},
                status="running"
            )
        
        # Create runs concurrently
        tasks = [create_run(str(i)) for i in range(5)]
        run_ids = await asyncio.gather(*tasks)
        
        assert len(run_ids) == 5
        assert len(set(run_ids)) == 5  # All unique
        
        # Verify all runs were created
        runs = await repo.list_optimization_runs(limit=10)
        concurrent_runs = [r for r in runs if r.task_id.startswith("concurrent-task")]
        assert len(concurrent_runs) == 5


@pytest.mark.integration 
class TestDatabaseMigrations:
    """Test database migration functionality."""
    
    @pytest.mark.asyncio
    async def test_migration_execution(self):
        """Test running database migrations."""
        # Create a fresh database
        config = DatabaseConfig(
            url="sqlite+aiosqlite:///:memory:",
            echo=False
        )
        
        db_manager = DatabaseManager(config)
        
        # Run migrations
        await db_manager.create_tables()
        
        # Verify tables exist
        async with db_manager.session() as session:
            # Try to create instances of each model
            from gepa.database.models import OptimizationRun, Candidate
            
            run = OptimizationRun(
                task_id="migration-test",
                config={"test": True},
                status="running"
            )
            session.add(run)
            await session.commit()
            
            # Should not raise any exceptions
            assert run.id is not None
        
        await db_manager.close()


@pytest.mark.integration
class TestDatabasePerformance:
    """Test database performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_bulk_insert_performance(self, test_db: DatabaseManager, sample_system):
        """Test bulk insert performance."""
        import time
        
        candidate_repo = CandidateRepository(test_db)
        opt_repo = OptimizationRepository(test_db)
        
        # Create optimization run
        run_id = await opt_repo.create_optimization_run(
            task_id="performance-test",
            config={"budget": 1000},
            status="running"
        )
        
        # Bulk insert candidates
        start_time = time.time()
        
        async with test_db.session() as session:
            candidates = []
            for i in range(100):
                candidate = Candidate(
                    optimization_run_id=run_id,
                    system_config=sample_system.to_dict(),
                    scores={"exact_match": 0.5 + (i % 50) / 100.0},
                    cost=0.01 + (i % 10) / 1000.0,
                    tokens_used=50 + i,
                    generation=i // 10 + 1
                )
                candidates.append(candidate)
            
            session.add_all(candidates)
            await session.commit()
        
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time (adjust as needed)
        assert elapsed < 5.0
        
        # Verify all candidates were inserted
        candidates = await candidate_repo.list_candidates_for_run(run_id)
        assert len(candidates) == 100
    
    @pytest.mark.asyncio 
    async def test_query_performance(self, test_db: DatabaseManager, sample_system):
        """Test query performance with larger datasets."""
        import time
        
        candidate_repo = CandidateRepository(test_db)
        opt_repo = OptimizationRepository(test_db)
        
        # Create optimization run with many candidates
        run_id = await opt_repo.create_optimization_run(
            task_id="query-performance-test",
            config={"budget": 500},
            status="completed"
        )
        
        # Insert test data
        async with test_db.session() as session:
            candidates = []
            for i in range(500):
                candidate = Candidate(
                    optimization_run_id=run_id,
                    system_config=sample_system.to_dict(),
                    scores={
                        "exact_match": (i % 100) / 100.0,
                        "f1_score": ((i + 50) % 100) / 100.0
                    },
                    cost=0.001 + (i % 100) / 10000.0,
                    tokens_used=10 + i % 200,
                    generation=i // 50 + 1
                )
                candidates.append(candidate)
            
            session.add_all(candidates)
            await session.commit()
        
        # Test query performance
        start_time = time.time()
        
        best_candidates = await candidate_repo.get_best_candidates_by_metric(
            run_id, "exact_match", limit=10
        )
        
        elapsed = time.time() - start_time
        
        # Should complete quickly
        assert elapsed < 1.0
        assert len(best_candidates) == 10
        
        # Verify ordering
        for i in range(len(best_candidates) - 1):
            assert (best_candidates[i].scores["exact_match"] >= 
                   best_candidates[i + 1].scores["exact_match"])